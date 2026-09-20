import math
import unittest
import numpy as np
from aircraft import Aircraft
from atmosphere import Atmosphere
from dynamics import HullContact, HullDrag, float_contacts
from spatial_dynamics import integrate


class FloatTests(unittest.TestCase):
    def setUp(self):
        self.ac = Aircraft()
        self.hull = HullContact()
        self.hd = HullDrag()
        self.air = Atmosphere()

    def integrate(self, state, dt=0.5, bank_command=0.0):
        return integrate(
            self.ac, self.hull, self.hd, self.air, lambda x, y, t: 0.0,
            state, dt=dt, t=0.0, pitch=0.0, throttle=0.2,
            bank_command=bank_command, rudder_command=0.0)

    def test_level_floats_clear_water(self):
        z = self.hull.h_keel - self.ac.W / (1025.0 * 9.80665 * self.hull.A_wp)
        left, right = float_contacts(self.ac, z, 0.0, 0.0, 0.0, 0.0, 0.0)
        self.assertEqual(left.N, 0)
        self.assertEqual(right.N, 0)

    def test_heel_loads_the_low_float_and_restores(self):
        z = 0.25
        left, right = float_contacts(self.ac, z, math.radians(8), 1.0, 0.0, 0.0, 0.0)
        self.assertGreater(right.N, left.N)
        state, _ = self.integrate([0, 0, z, 2, 0, 0, math.radians(8), 0], dt=1.0)
        self.assertLess(abs(state[6]), math.radians(8))

    def test_airborne_bank_still_follows_command(self):
        a, _ = self.integrate([0, 0, 25, 12, 0, 0, 0, 0], dt=2.0, bank_command=0.5)
        b, _ = self.integrate([0, 0, 25, 12, 0, 0, 0, 0], dt=2.0, bank_command=-0.5)
        self.assertGreater(a[6], 0.2)
        self.assertAlmostEqual(a[6], -b[6], places=5)

    def test_single_crest_graze_does_not_kick_bank(self):
        # One 10 ms substep with the right float on a crest: the old
        # angle-impulse model slammed bank by ~2.4 deg per substep.
        state, _ = integrate(
            self.ac, self.hull, self.hd, self.air, lambda x, y, t: 0.05 * y,
            [0, 0, 0.0, 2, 0, 0, 0, 0], dt=0.01, t=0.0, pitch=0.0,
            throttle=0.2, bank_command=0.0, rudder_command=0.0)
        self.assertLess(abs(state[6]), 0.01)

    def test_float_graze_keeps_servo_authority(self):
        cmd = 0.08
        state, _ = integrate(
            self.ac, self.hull, self.hd, self.air, lambda x, y, t: 0.0,
            [0, 0, 0.5, 11, 0, 0, cmd, 0], dt=0.1, t=0.0, pitch=0.0,
            throttle=0.5, bank_command=cmd, rudder_command=0.0)
        self.assertLess(abs(state[6] - cmd), 0.03)

    def test_mass_includes_both_floats(self):
        self.assertEqual(self.ac.mass.floats, 4.0)
        self.assertAlmostEqual(self.ac.mass.total, 84.0)


if __name__ == '__main__':
    unittest.main()
