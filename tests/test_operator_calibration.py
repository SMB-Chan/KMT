import math
import unittest

from operator_training.calibration import (
    AxisCalibration,
    CalibrationError,
    GamepadProfile,
    ResponseCurve,
    default_profile,
    to_control_input,
)


class ResponseCurveTests(unittest.TestCase):
    def test_linear_identity(self):
        for x in (0.0, 0.25, 0.5, 0.75, 1.0):
            self.assertAlmostEqual(ResponseCurve.apply("linear", x), x)

    def test_gentle_zero_derivative_at_endpoints(self):
        # d/dx of 0.5*(1-cos(pi*x)) at 0 and 1 is 0; values match.
        self.assertAlmostEqual(ResponseCurve.apply("gentle", 0.0), 0.0)
        self.assertAlmostEqual(ResponseCurve.apply("gentle", 1.0), 1.0)
        self.assertAlmostEqual(ResponseCurve.apply("gentle", 0.5), 0.5)

    def test_aggressive_clamps(self):
        self.assertAlmostEqual(ResponseCurve.apply("aggressive", 0.0), 0.0)
        self.assertAlmostEqual(ResponseCurve.apply("aggressive", 1.0), 1.0)
        # Midpoint of smoothstep is exactly 0.5
        self.assertAlmostEqual(ResponseCurve.apply("aggressive", 0.5), 0.5)

    def test_clamps_out_of_range_input(self):
        self.assertAlmostEqual(ResponseCurve.apply("linear", -0.1), 0.0)
        self.assertAlmostEqual(ResponseCurve.apply("linear", 1.1), 1.0)

    def test_unknown_curve_raises(self):
        with self.assertRaises(CalibrationError):
            ResponseCurve.apply("hyperbolic", 0.5)


class AxisCalibrationTests(unittest.TestCase):
    def test_zero_in_deadzone(self):
        ac = AxisCalibration(deadzone=0.05)
        self.assertEqual(ac.apply(0.0), 0.0)
        self.assertEqual(ac.apply(0.04), 0.0)
        self.assertEqual(ac.apply(-0.04), 0.0)

    def test_invert_flips_sign(self):
        ac_pos = AxisCalibration(deadzone=0.05, invert=False)
        ac_neg = AxisCalibration(deadzone=0.05, invert=True)
        self.assertGreater(ac_pos.apply(0.5), 0.0)
        self.assertLess(ac_neg.apply(0.5), 0.0)

    def test_outer_reach_is_one(self):
        # If max_observed is 0.9 and deadzone is 0.05, a raw reading
        # of 0.9 should renormalise to 1.0 (the user's outer reach).
        ac = AxisCalibration(deadzone=0.05, max_observed=0.9)
        self.assertAlmostEqual(ac.apply(0.9), 1.0)

    def test_deadzone_too_large_rejected(self):
        with self.assertRaises(CalibrationError):
            AxisCalibration(deadzone=0.5)

    def test_max_observed_too_low_rejected(self):
        with self.assertRaises(CalibrationError):
            AxisCalibration(max_observed=0.4)

    def test_unknown_curve_rejected(self):
        with self.assertRaises(CalibrationError):
            AxisCalibration(response_curve="weird")

    def test_non_finite_rejected(self):
        ac = AxisCalibration()
        for bad in (float("inf"), float("-inf"), float("nan")):
            with self.subTest(bad=bad):
                with self.assertRaises(CalibrationError):
                    ac.apply(bad)


class GamepadProfileTests(unittest.TestCase):
    def test_default_profile_basic_mapping(self):
        prof = default_profile()
        axes = prof.apply({"pit": 0.0, "ban": 0.0, "rud": 0.0, "thr": 0.5})
        # Trigger centred at 0.5 -> centred symmetric -> 0.0
        self.assertAlmostEqual(axes["thr"], 0.0)

    def test_missing_axis_raises(self):
        prof = default_profile()
        with self.assertRaises(CalibrationError):
            prof.apply({"pit": 0.0, "ban": 0.0, "rud": 0.0})

    def test_full_throttle(self):
        prof = default_profile()
        axes = prof.apply({"pit": 0.0, "ban": 0.0, "rud": 0.0, "thr": 1.0})
        self.assertGreater(axes["thr"], 0.9)


class ToControlInputTests(unittest.TestCase):
    def test_in_range_input_passes_envelope(self):
        prof = default_profile()
        raw = {"pit": 0.2, "ban": 0.0, "rud": 0.0, "thr": 0.7}
        ci = to_control_input(prof, raw,
                              pitch_range_deg=(-8.0, 12.0),
                              bank_abs_deg=25.0)
        self.assertGreater(ci["throttle"], 0.0)
        self.assertGreater(ci["pitch_deg"], 0.0)

    def test_out_of_envelope_passes_through(self):
        # The function does NOT clamp; the envelope validator does.
        # Output pitch may exceed pitch_range_deg if axis saturates.
        prof = GamepadProfile(
            pit=AxisCalibration(deadzone=0.0, max_observed=1.0),
            ban=AxisCalibration(deadzone=0.0, max_observed=1.0),
            rud=AxisCalibration(deadzone=0.0, max_observed=1.0),
            thr=AxisCalibration(deadzone=0.0, max_observed=1.0),
        )
        raw = {"pit": 1.0, "ban": -1.0, "rud": -1.0, "thr": 1.0}
        ci = to_control_input(prof, raw,
                              pitch_range_deg=(-8.0, 12.0),
                              bank_abs_deg=25.0,
                              rudder_abs=1.0)
        self.assertEqual(ci["pitch_deg"], 12.0)
        self.assertEqual(ci["bank_deg"], -25.0)
        self.assertEqual(ci["rudder"], -1.0)
        self.assertEqual(ci["throttle"], 1.0)


class ProfileSerialisationTests(unittest.TestCase):
    def test_default_profile_serialisable(self):
        from operator_training.calibration import assert_profile_serialisable
        assert_profile_serialisable(default_profile())


if __name__ == "__main__":
    unittest.main()
