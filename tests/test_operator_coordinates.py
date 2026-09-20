import math
import unittest

from operator_training.coordinates import (
    Attitude,
    altitude_above_origin_neu,
    assert_sign_convention,
    neu_to_ruf,
    normalise_ruf_axis,
)


class CoordinateTransformTests(unittest.TestCase):
    def test_identity_attitude_zero_rotation(self):
        att = Attitude()
        rot = att.rotation_matrix()
        self.assertAlmostEqual(rot[0][0], 1.0)
        self.assertAlmostEqual(rot[1][1], 1.0)
        self.assertAlmostEqual(rot[2][2], 1.0)
        for i, row in enumerate(rot):
            for j, val in enumerate(row):
                if i == j:
                    continue
                self.assertAlmostEqual(val, 0.0, places=12,
                                       msg=f"({i},{j})={val}")

    def test_heading_90_east(self):
        att = Attitude(heading_deg=90.0)
        rot = att.rotation_matrix()
        # Forward (body X) in NEU should point east (+Y) at heading 90°.
        self.assertAlmostEqual(rot[0][0], 0.0, places=6)
        self.assertAlmostEqual(rot[1][0], 1.0, places=6)
        self.assertAlmostEqual(rot[2][0], 0.0, places=6)

    def test_pitch_15_lifts_forward(self):
        att = Attitude(pitch_deg=15.0)
        rot = att.rotation_matrix()
        # Forward (body X) in NEU should point up (+Z) at pitch +15°.
        self.assertAlmostEqual(rot[0][0], math.cos(math.radians(15)), places=6)
        self.assertAlmostEqual(rot[2][0], math.sin(math.radians(15)), places=6)

    def test_bank_15_tilts_right_wing_down(self):
        att = Attitude(bank_deg=15.0)
        rot = att.rotation_matrix()
        # Body right axis (Y) should rotate towards -Z (down).
        self.assertAlmostEqual(rot[2][1], -math.sin(math.radians(15)), places=6)
        # Body up axis (Z) should rotate towards +Y (east) when
        # heading north.
        self.assertAlmostEqual(rot[1][2], math.sin(math.radians(15)), places=6)

    def test_neu_to_ruf_axes(self):
        flat = neu_to_ruf({"x": 10, "y": -5, "z": 100.0},
                          Attitude())
        # scene.x = east = -5
        self.assertEqual(flat["position"]["x"], -5.0)
        # scene.y = up = 100
        self.assertEqual(flat["position"]["y"], 100.0)
        # scene.z = -north = -10
        self.assertEqual(flat["position"]["z"], -10.0)

    def test_sign_convention_invariants(self):
        result = assert_sign_convention(Attitude())
        # flat forward.z < 0
        self.assertLess(result["flat"]["forward_ruf"]["z"], 0)
        # flat right.x > 0
        self.assertGreater(result["flat"]["right_ruf"]["x"], 0)
        # flat up.y > 0
        self.assertGreater(result["flat"]["up_ruf"]["y"], 0)

    def test_bank_tilts_up_to_east(self):
        r = neu_to_ruf({"x": 0, "y": 0, "z": 0},
                       Attitude(bank_deg=15))
        # Body up should tilt in +X_RUF (east, +y_NE) when banked right.
        self.assertGreater(r["up_ruf"]["x"], 0)
        # Bank only rolls, it does not pitch forward. Forward stays
        # in the horizontal plane.
        self.assertAlmostEqual(r["forward_ruf"]["y"], 0.0)
        # Right wing (east) tilts down: scene.y < 0.
        self.assertLess(r["right_ruf"]["y"], 0)

    def test_forward_length_is_one(self):
        r = neu_to_ruf({"x": 0, "y": 0, "z": 0},
                       Attitude(heading_deg=30, pitch_deg=5, bank_deg=10))
        length, _ = normalise_ruf_axis(r["forward_ruf"])
        self.assertAlmostEqual(length, 1.0, places=6)
        length, _ = normalise_ruf_axis(r["right_ruf"])
        self.assertAlmostEqual(length, 1.0, places=6)
        length, _ = normalise_ruf_axis(r["up_ruf"])
        self.assertAlmostEqual(length, 1.0, places=6)

    def test_orthonormal_basis(self):
        """The forward, right, up axes should be mutually orthogonal."""
        r = neu_to_ruf({"x": 0, "y": 0, "z": 0},
                       Attitude(heading_deg=37, pitch_deg=-8, bank_deg=12))
        f = r["forward_ruf"]
        rg = r["right_ruf"]
        u = r["up_ruf"]
        # Dot products should be 0 (within FP tolerance).
        fr = f["x"] * rg["x"] + f["y"] * rg["y"] + f["z"] * rg["z"]
        fu = f["x"] * u["x"] + f["y"] * u["y"] + f["z"] * u["z"]
        ru = rg["x"] * u["x"] + rg["y"] * u["y"] + rg["z"] * u["z"]
        self.assertAlmostEqual(fr, 0.0, places=10)
        self.assertAlmostEqual(fu, 0.0, places=10)
        self.assertAlmostEqual(ru, 0.0, places=10)

    def test_altitude_above_origin(self):
        d = altitude_above_origin_neu(
            {"x": 3, "y": 4, "z": 0}, {"x": 0, "y": 0, "z": 0})
        self.assertAlmostEqual(d, 5.0)
        d = altitude_above_origin_neu(
            {"x": 1, "y": 2, "z": 2}, {"x": 1, "y": 2, "z": 2})
        self.assertEqual(d, 0.0)


class PositivePitchTests(unittest.TestCase):
    def test_positive_pitch_points_nose_up_at_every_heading(self):
        for heading in (0, 90, 180, -90):
            result = neu_to_ruf({}, Attitude(heading_deg=heading, pitch_deg=10, bank_deg=20))
            self.assertGreater(result["forward_ruf"]["y"], 0)
            self.assertLess(result["right_ruf"]["y"], 0)


class CoordinateWireFormatTests(unittest.TestCase):
    """Sanity-check that the JSON payload is well-formed."""

    def test_payload_json_serialisable(self):
        import json
        r = neu_to_ruf({"x": 100, "y": 200, "z": 50.0},
                       Attitude(heading_deg=10, pitch_deg=2, bank_deg=-5))
        text = json.dumps(r)
        self.assertIn("position", text)
        self.assertIn("forward_ruf", text)
        self.assertIn("right_ruf", text)
        self.assertIn("up_ruf", text)


if __name__ == "__main__":
    unittest.main()
