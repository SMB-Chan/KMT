import unittest
import numpy as np
from flight_diagnostics import takeoff_progress
from aircraft import Aircraft
from atmosphere import AtmosphereConfig
from mavlink_if import FlyingBoatVehicle
from ocean import Ocean


class FlightDiagnosticsTests(unittest.TestCase):
    def test_force_budget_closes_with_added_mass_and_crosswind(self):
        from unittest.mock import patch
        vehicle = FlyingBoatVehicle(Aircraft(), Ocean(Hs=0), spatial=True,
            atmosphere=AtmosphereConfig(wind=(1,3,0), gust_rms=.2))
        vehicle.arm()
        for dt in (.05,.023,.1):
            before = vehicle.Vx
            with patch('mavlink_if.effective_mass_increase', return_value=2):
                vehicle.step(dt=dt,action=[.9,.1,.2,-.1])
            b=vehicle.read_telemetry()[2]['force_budget']
            self.assertAlmostEqual(b['net_x_N'],b['thrust_x_N']+b['aerodynamic_x_N']+b['water_x_N'])
            self.assertAlmostEqual(b['net_x_N']/b['total_mass_kg'],(vehicle.Vx-before)/dt,places=10)
            self.assertAlmostEqual(b['mean_ax_m_s2'],(vehicle.Vx-before)/dt,places=10)

    def rows(self, speeds, throttle=1):
        return [dict(t=i*.05,Vx=float(v),throttle=throttle) for i,v in enumerate(speeds)]

    def test_plateau_requires_complete_high_throttle_window(self):
        self.assertTrue(takeoff_progress(self.rows(np.full(80,2.8)))['low_speed_plateau'])
        self.assertFalse(takeoff_progress(self.rows(np.linspace(1,5,80)))['low_speed_plateau'])
        self.assertFalse(takeoff_progress(self.rows(np.full(80,2.8),.5))['low_speed_plateau'])
        self.assertFalse(takeoff_progress(self.rows(np.full(80,5)))['low_speed_plateau'])
        self.assertFalse(takeoff_progress(self.rows(np.full(20,2.8)))['sufficient_history'])
        self.assertFalse(takeoff_progress([])['sufficient_history'])

    def test_force_mean_weights_step_duration(self):
        rows=[dict(t=t,Vx=2.8,throttle=1,force_budget={'net_x_N':f},
                   T_factor=.8,prop_clearance_m=c)
              for t,f,c in ((0,99,.5),(1,10,.9),(3,20,1.1))]
        result=takeoff_progress(rows)
        self.assertAlmostEqual(result['mean_force_budget']['net_x_N'],50/3)
        self.assertAlmostEqual(result['mean_T_factor'],.8)
        self.assertAlmostEqual(result['mean_prop_clearance_m'],(1*.9+2*1.1)/3)
        self.assertAlmostEqual(result['min_prop_clearance_m'],.5)

    def test_snapshot_records_prop_clearance(self):
        from damage import SprayModel
        from ocean import Ocean
        vehicle = FlyingBoatVehicle(Aircraft(), Ocean(Hs=0), spatial=True)
        vehicle.arm()
        vehicle.step(action=[1, 0, 0, 0])
        snap = vehicle.read_telemetry()[2]
        spray = SprayModel(prop_z_offset=vehicle.spray.prop_z_offset,
                           D_prop=vehicle.spray.D_prop)
        self.assertAlmostEqual(snap['prop_clearance_m'],
                               spray.prop_top_clearance(vehicle.z, snap['eta']))
        self.assertAlmostEqual(snap['prop_bottom_clearance_m'],
                               vehicle.z + spray.prop_z_offset - spray.D_prop/2 - snap['eta'])
        self.assertLess(snap['prop_bottom_clearance_m'], snap['prop_clearance_m'])
        self.assertGreater(snap['T_factor'], 0.99)
        self.assertGreater(snap['prop_clearance_m'], vehicle.spray.critical_clearance - 0.05)

    def test_lower_mount_reduces_resting_thrust_factor(self):
        from ocean import Ocean
        low = FlyingBoatVehicle(Aircraft(), Ocean(Hs=0), spatial=True)
        high = FlyingBoatVehicle(Aircraft(), Ocean(Hs=0), spatial=True)
        low.spray.prop_z_offset = 0.30
        low.arm(); high.arm()
        low.step(action=[1, 0, 0, 0]); high.step(action=[1, 0, 0, 0])
        self.assertGreater(high.read_telemetry()[2]['T_factor'],
                           low.read_telemetry()[2]['T_factor'])
        self.assertGreater(high.read_telemetry()[2]['prop_clearance_m'],
                           low.read_telemetry()[2]['prop_clearance_m'])
        self.assertLess(low.read_telemetry()[2]['T_factor'], 1)


if __name__=='__main__':
    unittest.main()
