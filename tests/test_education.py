import unittest
import numpy as np
from educate import curriculum, flight
from teacher_student import StudentPilot, features


class EducationTests(unittest.TestCase):
    def test_curriculum_covers_phases_without_seed_leakage(self):
        rows=list(curriculum())
        self.assertEqual(len(rows),32)
        train={r['seed'] for r in rows if r['split']=='train'}
        val={r['seed'] for r in rows if r['split']=='validation'}
        self.assertFalse(train & val)
        self.assertFalse((train | val) & {101,102})
        self.assertEqual(len({r['id'] for r in rows}),32)
        for row in rows:
            self.assertTrue(np.isfinite(features(row['observation'],row['mission'])).all())
        for split in ('train','validation'):
            self.assertEqual(len({r['stage'] for r in rows if r['split']==split}),8)

    def test_evaluation_reports_timeout_without_success(self):
        result=flight(StudentPilot(), 'takeoff',0,101,duration=.1)
        self.assertEqual(result['status'],'time_limit')
        self.assertAlmostEqual(result['time'],.1)
        self.assertTrue(np.isfinite(result['final']['altitude_m']))
