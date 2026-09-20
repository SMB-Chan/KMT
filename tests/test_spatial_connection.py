import contextlib
import io
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
import numpy as np
from aircraft import Aircraft
from atmosphere import AtmosphereConfig
from env import EnvConfig, FlyingBoatEnv
from ocean_directional import DirectionalOcean
from mavlink_if import FlyingBoatVehicle, MAV_CMD_NAV_WAYPOINT, MAV_CMD_CONDITION_YAW
from ollama_pilot import SpatialControl, OllamaPilot, PilotError, observe, fallback_control
from policy_pilot import PolicyPilot
from fly_ollama import parser, run


class SpatialConnectionTests(unittest.TestCase):
    def vehicle(self):
        v = FlyingBoatVehicle(Aircraft(), DirectionalOcean(Hs=.3, seed=42),
                              spatial=True, atmosphere=AtmosphereConfig(wind=(0, 3, 0), gust_rms=.2))
        v.z = 25
        v.Vx = 13 * math.cos(math.radians(8))
        v.Vz = -13 * math.sin(math.radians(8))
        v.arm()
        return v

    def test_shared_physics_matches_environment(self):
        v = self.vehicle()
        env = FlyingBoatEnv(Aircraft(), EnvConfig(scenario='landing', spatial=True,
                           directional=True, Hs=.3, atmosphere=v.atmosphere_config))
        env.reset(42)
        action = [.4, .1, .2, -.1]
        for _ in range(10):
            env.step(action)
            with patch("mavlink_if.effective_thrust_factor", return_value=1.0), \
                 patch("mavlink_if.effective_mass_increase", return_value=0.0):
                v.step(action=action)
        np.testing.assert_allclose([v.x, v.y, v.z, v.Vx, v.Vy, v.Vz, v.bank, v.heading],
            [env._x, env._y, env._z, env._Vx, env._Vy, env._Vz, env._bank, env._heading], atol=1e-10)

    def test_four_actuators_and_disarm_failure(self):
        v = self.vehicle()
        SpatialControl(.4, 3, 15, -.2).apply(v)
        v.step()
        self.assertAlmostEqual(v.bank_command, math.radians(15))
        self.assertAlmostEqual(v.rudder_command, -.2)
        self.assertGreater(v.read_telemetry()[2]['T'], 0)
        v.disarm()
        SpatialControl(1, 3, 15, -.2).apply(v)
        v.step()
        self.assertEqual(v.read_telemetry()[2]['T'], 0)
        v.arm()
        v.damage.failed = True
        v.step(action=[1, 0, 0, 0])
        self.assertEqual(v.read_telemetry()[2]['T'], 0)
        self.assertFalse(v._armed)
        with self.assertRaises(ValueError):
            v.step(action=[1, 0])

    def test_spatial_schema_and_payload(self):
        text = '{"throttle":0.4,"pitch_deg":3,"bank_deg":15,"rudder":-0.2}'
        self.assertEqual(SpatialControl.parse(text), SpatialControl(.4, 3, 15, -.2))
        for mutation in ({'bank_deg':46}, {'rudder':True}, {'rudder':float('nan')}, {'extra':1}):
            data = dict(json.loads(text), **mutation)
            with self.assertRaises(PilotError):
                SpatialControl.parse(json.dumps(data))
        with self.assertRaises(PilotError):
            SpatialControl.parse('{"throttle":0,"pitch_deg":0}')
        pilot = OllamaPilot(spatial=True)
        pilot._request = Mock(return_value={'done':True, 'message':{'content':text}})
        self.assertIsInstance(pilot.decide({}, {})[0], SpatialControl)
        self.assertEqual(len(pilot._request.call_args.args[1]['format']['required']), 4)

    def test_telemetry_coordinates_attitude_and_wind(self):
        v = self.vehicle()
        v.x, v.y = 30, 12
        v.Vy = 2
        v.step(action=[.4, .1, .4, .2])
        hil, gpi, snap = v.read_telemetry()
        self.assertEqual(hil.fields['vy'], v.Vy)
        self.assertEqual(gpi.fields['vz'], int(-v.Vz * 100))
        self.assertEqual(gpi.fields['lon'], int((v.origin_lon+v.y/(111000*math.cos(math.radians(v.origin_lat))))*1e7))
        self.assertEqual(v.read_attitude().fields['roll'], v.bank)
        self.assertEqual(hil.fields['true_airspeed'], snap['airspeed'])
        self.assertEqual(observe(v)['wave_elevation_m'], v.wave_elevation())
        self.assertEqual(observe(v)['wind_m_s'], snap['wind'])

    def test_waypoint_yaw_and_fallback(self):
        v = self.vehicle()
        v.y, v.Vy = 5, 1
        control = fallback_control(v, 'landing', 0, 13)
        self.assertIsInstance(control, SpatialControl)
        self.assertLess(control.bank_deg, 0)
        far = FlyingBoatVehicle(Aircraft(), DirectionalOcean(Hs=.3, seed=42),
                                spatial=True, atmosphere=AtmosphereConfig(wind=(0, 3, 0), gust_rms=.2))
        far.y, far.Vy = 4, 0
        v.y, v.Vy = 2, 0
        self.assertLess(far.lateral_setpoint()[0], v.lateral_setpoint()[0])
        v.send_command(MAV_CMD_NAV_WAYPOINT, {'x':100, 'y':-10, 'alt':25})
        v.step()
        self.assertLess(v.bank_command, 0)
        self.assertLess(v.rudder_command, 0)
        v.send_command(MAV_CMD_CONDITION_YAW, {'heading':30})
        self.assertEqual(v._cmd_params['heading'], 30)
        v.step()
        self.assertGreater(v.rudder_command, 0)
        with self.assertRaises(ValueError):
            v.send_command(MAV_CMD_NAV_WAYPOINT, {'y':float('nan')})

    def test_policy_observation_matches_environment(self):
        v = self.vehicle()
        env = FlyingBoatEnv(Aircraft(), EnvConfig(scenario='landing', spatial=True,
                           directional=True, Hs=.3, atmosphere=v.atmosphere_config))
        expected = env.reset(42)
        v.throttle = .05
        pilot = PolicyPilot.__new__(PolicyPilot)
        pilot.spatial = True
        pilot.model = Mock()
        pilot.model.act.return_value = (np.array([.4, .1, .2, -.1]), 0, 0)
        control, _ = pilot.decide(observe(v), {'scenario':'landing'})
        np.testing.assert_allclose(pilot.model.act.call_args.args[0], expected, atol=1e-7)
        self.assertEqual(control, SpatialControl(.4, 3, 9, -.1))

    def test_seeded_reset_clears_spatial_state(self):
        v = self.vehicle()
        wind = v.atmosphere.wind(1).copy()
        v.step(action=[.4, 0, .5, .5])
        v.reset(42)
        self.assertEqual((v.y, v.Vy, v.bank, v.heading), (0,0,0,0))
        self.assertIsNone(v.read_attitude())
        np.testing.assert_array_equal(wind, v.atmosphere.wind(1))

    def test_runner_fallback_and_telemetry_log(self):
        pilot = Mock()
        pilot.check_model.return_value = {'name':'mock'}
        pilot.decide.side_effect = [(SpatialControl(.4, 3, 10, -.2), {}), PilotError('timeout')]
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp)/'flight'
            args = parser().parse_args(['--spatial','--directional','--scenario','landing',
                '--wind','0','3','0','--duration','.2','--interval','.1','--output',str(target)])
            with contextlib.redirect_stdout(io.StringIO()):
                summary = run(args, pilot)
            self.assertEqual(summary['physics_steps'],4)
            self.assertEqual(summary['fallback_decisions'],1)
            rows = [json.loads(line) for line in (target/'trajectory.jsonl').read_text().splitlines()]
            self.assertIn('bank_deg',rows[-1]['control'])
            self.assertIn('attitude',rows[-1]['telemetry'])
            self.assertIn('Vy',rows[-1]['snapshot'])

    def test_lateral_assist_preserves_request_and_recomputes(self):
        from ollama_pilot import stabilize_lateral
        v = self.vehicle()
        v.y, v.Vy = 3, 1
        requested = SpatialControl(.4, 3, 0, 0)
        applied = stabilize_lateral(v, requested)
        self.assertEqual((applied.throttle, applied.pitch_deg), (.4, 3))
        self.assertLess(applied.bank_deg, 0)
        self.assertEqual(requested.bank_deg, 0)
        pilot = Mock()
        pilot.check_model.return_value = {'name': 'mock'}
        pilot.decide.return_value = (requested, {})
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / 'assisted'
            args = parser().parse_args(['--spatial', '--lateral-assist',
                '--scenario', 'landing', '--wind', '0', '3', '0',
                '--duration', '.3', '--interval', '1', '--output', str(target)])
            with contextlib.redirect_stdout(io.StringIO()):
                summary = run(args, pilot)
            self.assertEqual(summary['decisions'], 1)
            rows = [json.loads(line) for line in (target/'trajectory.jsonl').read_text().splitlines()]
            self.assertTrue(all(r['control']['bank_deg'] == 0 for r in rows))
            self.assertNotEqual(rows[0]['applied_control']['bank_deg'], rows[-1]['applied_control']['bank_deg'])
            self.assertTrue(all(r['applied_control']['throttle'] == .4 for r in rows))
            self.assertTrue(summary['lateral_assist'])
        args = parser().parse_args(['--lateral-assist'])
        with self.assertRaises(ValueError):
            run(args, pilot)

    def test_added_water_mass_and_spray_affect_spatial_dynamics(self):
        a, b = self.vehicle(), self.vehicle()
        with patch('mavlink_if.effective_thrust_factor', return_value=1):
            a.step(action=[1, 0, 0, 0])
        with patch('mavlink_if.effective_thrust_factor', return_value=.5), \
             patch('mavlink_if.effective_mass_increase', return_value=10):
            b.step(action=[1, 0, 0, 0])
        self.assertLess(b.read_telemetry()[2]['T'], a.read_telemetry()[2]['T'])
        self.assertLess(b.Vx,a.Vx)
        self.assertEqual(b.read_telemetry()[2]['extra_mass'],10)


if __name__ == '__main__':
    unittest.main()
