import math
import unittest
from system_one import Logistic, choice_from_logits, noul_answer, softmax
from flight_jev import FlightJev, evaluate, features
from ollama_pilot import Control, SpatialControl


def obs(**kwargs):
    data = dict(altitude_m=0.3, forward_speed_m_s=2.0, vertical_speed_m_s=0.0,
                wave_elevation_m=0.0, keel_clearance_m=0.0, wave_preview_m=[0.0, 0.0, 0.0])
    data.update(kwargs)
    return data


class FlightJevTests(unittest.TestCase):
    def test_softmax_and_choice_are_typed(self):
        self.assertAlmostEqual(sum(softmax([0.0, 0.0, 0.0])), 1.0)
        answer = choice_from_logits(('a', 'b'), [2.0, 0.0])
        self.assertEqual(answer.choice, 'a')
        self.assertAlmostEqual(sum(answer.probabilities.values()), 1.0)
        self.assertGreater(answer.confidence, 0.5)
        noul = noul_answer(0.9)
        self.assertAlmostEqual(noul.noul, 0.9)
        self.assertGreater(noul.confidence, 0.8)

    def test_takeoff_phases(self):
        taxi = evaluate(obs(), dict(scenario='takeoff'))
        self.assertEqual(taxi['phase'].choice, 'taxi')
        climb = evaluate(obs(altitude_m=8.0, forward_speed_m_s=12.0, keel_clearance_m=7.7),
                         dict(scenario='takeoff'))
        self.assertEqual(climb['phase'].choice, 'climb')

    def test_landing_settle_and_compose(self):
        answers = evaluate(obs(altitude_m=0.5, forward_speed_m_s=10.0, vertical_speed_m_s=0.1,
                               keel_clearance_m=0.2), dict(scenario='landing'))
        self.assertEqual(answers['phase'].choice, 'settle')
        self.assertGreater(answers['ballooning'].noul, 0.6)
        control, meta = FlightJev().decide(
            obs(altitude_m=0.5, forward_speed_m_s=10.0, vertical_speed_m_s=0.2,
                keel_clearance_m=0.2), dict(scenario='landing', target_altitude_m=0, target_speed_m_s=8.5))
        self.assertIsInstance(control, Control)
        self.assertEqual(control.throttle, 0.0)
        self.assertLess(control.pitch_deg, 0.0)
        self.assertEqual(meta['source'], 'flight_jev')
        self.assertIn('phase', meta['answers'])

    def test_spatial_control_shape(self):
        control, _ = FlightJev(spatial=True).decide(
            obs(forward_speed_m_s=12.0, altitude_m=3.0),
            dict(scenario='takeoff', target_altitude_m=8, target_speed_m_s=8.5))
        self.assertIsInstance(control, SpatialControl)
        self.assertEqual(control.bank_deg, 0.0)

    def test_logistic_fits_and_roundtrips(self):
        import numpy as np
        import tempfile
        x = np.array([[1, 0], [1, 1], [1, 2], [1, 3]], dtype=float)
        y = np.array([0, 0, 1, 1], dtype=float)
        model = Logistic.fit(x, y, steps=300, lr=0.5)
        self.assertLess(model.predict([1, 0]), 0.5)
        self.assertGreater(model.predict([1, 3]), 0.5)
        with tempfile.TemporaryDirectory() as tmp:
            path = tmp + '/c.npz'
            np.savez(path, balloon_soon=np.zeros(len(features(obs()))))
            jev = FlightJev.load(path)
            self.assertIn('balloon_soon', jev.calibrator)
            p = jev.calibrator['balloon_soon'].predict(features(obs(altitude_m=0.5, vertical_speed_m_s=0.2)))
            self.assertGreaterEqual(p, 0.0)
            self.assertLessEqual(p, 1.0)


if __name__ == '__main__':
    unittest.main()
