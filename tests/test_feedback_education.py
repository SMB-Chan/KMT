import copy
import unittest
from unittest.mock import Mock
from educate import curriculum
from feedback_education import experiment, candidates, choose
from ollama_pilot import Control, OllamaPilot, PilotError


class FeedbackTests(unittest.TestCase):
    def test_simulation_does_not_mutate_teacher_state(self):
        record=next(curriculum()); record['control']={'throttle':1.,'pitch_deg':4.}
        original=copy.deepcopy(record)
        a=experiment(record,Control(1,4));b=experiment(record,Control(1,4))
        self.assertEqual(record,original)
        self.assertEqual(a,b)

    def test_offers_previous_and_best_controls(self):
        record=next(curriculum()); record['control']={'throttle':1.,'pitch_deg':4.}
        results,offered=candidates(record)
        self.assertIn(0,[r['candidate_id'] for r in offered])
        self.assertEqual(offered[0]['score'],max(r['score'] for r in results))
        self.assertEqual(results[0]['control'],record['control'])

    def test_teacher_choice_is_checked(self):
        pilot=OllamaPilot(); record=next(curriculum())
        offered=[dict(candidate_id=2,control={'throttle':1,'pitch_deg':4},score=1,
                      terminal=None,final={'altitude_m':0,'forward_speed_m_s':2,'vertical_speed_m_s':0})]
        pilot._request=Mock(return_value={'done':True,'message':{'content':'{"candidate_id":2}'}})
        result,_=choose(pilot,record,offered)
        self.assertEqual(result['candidate_id'],2)
        for text in ('{"candidate_id":99}','{"candidate_id":true}','null'):
            pilot._request.return_value={'done':True,'message':{'content':text}}
            with self.assertRaises(PilotError):choose(pilot,record,offered)
