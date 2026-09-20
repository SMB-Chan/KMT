import math
import unittest
import numpy as np
from aircraft import Aircraft, RHO, scaled_aircraft
from atmosphere import Atmosphere, AtmosphereConfig, ISA_TROPOPAUSE, isa_density


class ModelFidelityTests(unittest.TestCase):
    def test_isa_sea_level_and_tropopause(self):
        air = Atmosphere()
        self.assertAlmostEqual(air.density(0), RHO)
        self.assertAlmostEqual(air.temperature(0), 288.15)
        self.assertAlmostEqual(air.temperature(1000), 281.65)
        self.assertAlmostEqual(air.density(ISA_TROPOPAUSE), 0.3639, delta=0.002)
        self.assertEqual(air.density(-2), air.density(0))
        self.assertEqual(air.density(20000), air.density(ISA_TROPOPAUSE))

    def test_log_wind_matches_reference_height(self):
        air = Atmosphere(AtmosphereConfig(wind=(4, -2, 0.5)))
        np_wind = air.wind(0.0, 10.0)
        self.assertAlmostEqual(np_wind[0], 4)
        self.assertAlmostEqual(np_wind[1], -2)
        self.assertAlmostEqual(np_wind[2], 0.5)
        low = air.wind(0.0, 0.3)
        self.assertGreater(low[0], 0)
        self.assertLess(low[0], 4)
        self.assertAlmostEqual(low[2], 0.5)
        self.assertAlmostEqual(air.shear_factor(10), 1.0)

    def test_gusts_scale_with_height(self):
        air = Atmosphere(AtmosphereConfig(gust_rms=1.0), seed=3)
        self.assertAlmostEqual(isa_density(0), RHO)
        self.assertEqual(isa_density(1000), Atmosphere().density(1000))
        for t in np.linspace(0, 20, 40):
            ref = abs(air.wind(t, 10.0)[0])
            if ref > 0.1:
                self.assertLess(abs(air.wind(t, 1.0)[0]), ref)
                self.assertAlmostEqual(ref, abs(air.wind(t)[0]))
                return
        self.fail('no gust sample')

    def test_finite_wing_slope_and_ground_effect(self):
        ac = Aircraft()
        self.assertLess(ac.CL_alpha_3d, ac.aero.CL_alpha)
        self.assertAlmostEqual(ac.CL(0), ac.aero.CL0)
        self.assertAlmostEqual(ac.CL(0.1), ac.aero.CL0 + ac.CL_alpha_3d * 0.1)
        free = ac.CD(1.0)
        near = ac.CD(1.0, height_m=0.25)
        high = ac.CD(1.0, height_m=30)
        self.assertLess(near, free)
        self.assertAlmostEqual(high, free, places=3)
        self.assertAlmostEqual(ac.induced_drag_factor(None), 1.0)

    def test_scaled_aircraft_preserves_tw_and_ar(self):
        base = Aircraft()
        twin = scaled_aircraft(2.0)
        self.assertAlmostEqual(twin.geom.b, 30.0)
        self.assertAlmostEqual(twin.geom.AR, base.geom.AR)
        self.assertAlmostEqual(twin.mass.total, base.mass.total * 8)
        self.assertAlmostEqual(twin.prop.T_static / twin.W, base.prop.T_static / base.W, places=3)
        self.assertAlmostEqual(twin.V_stall / base.V_stall, math.sqrt(2.0), places=3)
        with self.assertRaises(ValueError):
            scaled_aircraft(0)

    def test_thrust_scales_with_density_and_keeps_sea_level_static(self):
        ac = Aircraft()
        self.assertAlmostEqual(ac.prop.T_static, ac.prop.thrust(0, 1, rho=RHO))
        self.assertAlmostEqual(ac.prop.thrust(0, 1, rho=2 * RHO), 2 * ac.prop.T_static)
        self.assertLess(ac.prop.thrust(ac.V_cruise, 1), ac.prop.T_static)

    def test_summary_reports_design_point(self):
        text = Aircraft().summary()
        self.assertIn("Weight", text)
        self.assertIn("Cruise speed (design point)", text)

    def test_power_required_includes_propeller_efficiency(self):
        ac = Aircraft()
        V = ac.V_cruise
        T = ac.prop.thrust(V, 0.7, rho=RHO)
        eta = ac.prop.eta_motor * ac.prop.eta_esc * ac.prop.eta_prop
        self.assertAlmostEqual(ac.prop.power_required(V, 0.7), T * V / eta)
        self.assertLess(ac.prop.eta_prop, 1.0)


if __name__ == '__main__':
    unittest.main()
