import math
import unittest

from operator_training.envelope import (
    ControlEnvelope,
    ControlInput,
    EnvelopeViolation,
    PITCH_BEGINNER_HI,
    BANK_BEGINNER,
    BANK_EXTENDED,
    PITCH_EXTENDED_HI,
)


class EnvelopeTests(unittest.TestCase):
    def test_beginner_envelope_default(self):
        env = ControlEnvelope.beginner()
        self.assertEqual(env.pitch_lo, -8.0)
        self.assertEqual(env.pitch_hi, PITCH_BEGINNER_HI)
        self.assertEqual(env.bank_abs, BANK_BEGINNER)
        self.assertEqual(env.rudder_abs, 1.0)

    def test_extended_envelope_expands(self):
        env = ControlEnvelope.extended()
        self.assertEqual(env.pitch_hi, PITCH_EXTENDED_HI)
        self.assertEqual(env.bank_abs, BANK_EXTENDED)

    def test_valid_input_passes(self):
        env = ControlEnvelope.beginner()
        out = env.validate({
            "throttle": 0.5, "pitch_deg": 2.0,
            "bank_deg": -10.0, "rudder": 0.0,
        })
        self.assertIsInstance(out, ControlInput)
        self.assertEqual(out.throttle, 0.5)

    def test_pitch_above_envelope_rejected(self):
        env = ControlEnvelope.beginner()
        with self.assertRaises(EnvelopeViolation):
            env.validate({
                "throttle": 0.5, "pitch_deg": PITCH_BEGINNER_HI + 0.5,
                "bank_deg": 0.0, "rudder": 0.0,
            })

    def test_pitch_extended_above_beginner_passes_extended_only(self):
        env_b = ControlEnvelope.beginner()
        env_e = ControlEnvelope.extended()
        raw = {
            "throttle": 0.5, "pitch_deg": 13.0,
            "bank_deg": 0.0, "rudder": 0.0,
        }
        with self.assertRaises(EnvelopeViolation):
            env_b.validate(raw)
        out = env_e.validate(raw)
        self.assertEqual(out.pitch_deg, 13.0)

    def test_bank_above_envelope_rejected(self):
        env = ControlEnvelope.beginner()
        with self.assertRaises(EnvelopeViolation):
            env.validate({
                "throttle": 0.5, "pitch_deg": 0.0,
                "bank_deg": BANK_BEGINNER + 1.0, "rudder": 0.0,
            })

    def test_non_finite_rejected(self):
        env = ControlEnvelope.beginner()
        for bad in (float("inf"), float("-inf"), float("nan")):
            with self.subTest(bad=bad):
                with self.assertRaises(EnvelopeViolation):
                    env.validate({
                        "throttle": bad, "pitch_deg": 0.0,
                        "bank_deg": 0.0, "rudder": 0.0,
                    })

    def test_missing_axis_rejected(self):
        env = ControlEnvelope.beginner()
        with self.assertRaises(EnvelopeViolation):
            env.validate({"throttle": 0.5, "pitch_deg": 0.0, "bank_deg": 0.0})

    def test_throttle_out_of_range_rejected(self):
        env = ControlEnvelope.beginner()
        with self.assertRaises(EnvelopeViolation):
            env.validate({
                "throttle": 1.5, "pitch_deg": 0.0,
                "bank_deg": 0.0, "rudder": 0.0,
            })

    def test_clip_clamps_without_raising(self):
        env = ControlEnvelope.beginner()
        ci = ControlInput(throttle=1.5, pitch_deg=20.0, bank_deg=80.0, rudder=2.0)
        out = env.clip(ci)
        self.assertEqual(out.throttle, 1.0)
        self.assertEqual(out.pitch_deg, env.pitch_hi)
        self.assertEqual(out.bank_deg, env.bank_abs)
        self.assertEqual(out.rudder, 1.0)


if __name__ == "__main__":
    unittest.main()
