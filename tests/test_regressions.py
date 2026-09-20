import math
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from aircraft import Aircraft
from ocean import Ocean
from ocean_real import RealOcean
from env import EnvConfig, FlyingBoatEnv
from policy import ActorCritic
from vectorized import VectorizedEnv
from mavlink_if import FlyingBoatVehicle, MAV_CMD_NAV_TAKEOFF, MAV_CMD_DO_SET_SERVO
from damage import DamageState, SprayModel, IngressModel, update_damage
from dynamics import HullDrag


def policy(seed=0):
    return ActorCritic(8, 2, np.array([0., -1.]), np.array([1., 1.]), hidden=8, seed=seed)


class RegressionTests(unittest.TestCase):
    def test_seed_and_single_action(self):
        a, b = policy(7), policy(7)
        np.testing.assert_array_equal(a.body.get_flat(), b.body.get_flat())
        state = FlyingBoatEnv(Aircraft()).reset()
        action, lp, value = a.act(state)
        self.assertEqual(action.shape, (2,))
        self.assertIsInstance(lp, float)
        self.assertIsInstance(value, float)
        np.testing.assert_array_equal(action, b.act(state)[0])
        self.assertFalse(np.array_equal(a.body.get_flat(), policy(8).body.get_flat()))

    def test_policy_gradient_matches_loss(self):
        p = policy()
        states = np.random.default_rng(3).normal(size=(6, 8)).astype(np.float32)
        actions, lp, values = p.act_batch(states)
        batch = dict(state=states, action=actions, log_p=lp,
                     advantage=np.linspace(-1, 1, 6), target_v=values + .3)
        checks = [(p.actor_head_b, 0), (p.critic_head_b, 0), (p.log_std, 0)]
        numeric = []
        for arr, idx in checks:
            orig = float(arr[idx]); eps = .001
            arr[idx] = orig + eps
            plus = p.update(batch, lr_body=0, lr_head=0, lr_std=0)['loss']
            arr[idx] = orig - eps
            minus = p.update(batch, lr_body=0, lr_head=0, lr_std=0)['loss']
            arr[idx] = orig
            numeric.append((plus - minus) / (2 * eps))
        before = [float(p.actor_head_b[0]), float(p.critic_head_b[0]), float(p.log_std[0])]
        p.update(batch, lr_body=0, lr_head=.001, lr_std=.001, clip_grad=1e6)
        after = [float(p.actor_head_b[0]), float(p.critic_head_b[0]), float(p.log_std[0])]
        np.testing.assert_allclose((np.array(before)-after)/.001, numeric, atol=.005, rtol=.03)

    def test_ppo_clipping_stops_actor_gradient(self):
        p = policy()
        states = np.zeros((4,8), np.float32)
        action, lp, value = p.act_batch(states)
        batch = dict(state=states, action=action, log_p=lp - math.log(2),
                     advantage=np.ones(4), target_v=value)
        before = p.actor_head_b.copy()
        p.update(batch, lr_body=0, lr_std=0)
        np.testing.assert_array_equal(before, p.actor_head_b)

    def test_env_validation_and_timeout(self):
        env = FlyingBoatEnv(Aircraft(), EnvConfig(max_steps=1))
        env.reset()
        for action in ([1], [1, np.nan], [[1, 0]]):
            with self.assertRaises(ValueError): env.step(action)
        env._z = 6; env._Vx = 10
        _, _, done, info = env.step([0, 0])
        self.assertTrue(done)
        self.assertFalse(info['success'])
        with self.assertRaises(RuntimeError): env.step([0, 0])

    def test_completed_vector_env_is_frozen(self):
        env = VectorizedEnv(2, EnvConfig(max_steps=3))
        env.envs[0].cfg = EnvConfig(max_steps=1)
        traj, infos = env.rollout(policy())
        self.assertEqual([len(t['rewards']) for t in traj], [1, 3])
        self.assertEqual(env.envs[0]._steps, 1)
        self.assertEqual(infos[0]['t'], .05)

    def test_wave_configuration(self):
        env = FlyingBoatEnv(Aircraft(), EnvConfig(Hs=0, Tp=4))
        env.reset()
        self.assertEqual(env._sea.Tp, 4)
        self.assertEqual(float(env._sea.eta([1], 0)[0]), 0)

    def test_sweep_passes_conditions_and_sample_count(self):
        import accelerate
        configs = []
        class FakeEnv:
            def __init__(self, n, cfg, base_seed):
                configs.append((n, cfg.Hs, cfg.Tp, base_seed)); self.n = n
            def rollout(self, *args, **kwargs):
                return ([{'rewards':[1]}] * self.n, [{'success':True}] * self.n)
        with patch.object(accelerate, 'VectorizedEnv', FakeEnv):
            accelerate.sea_state_sweep(policy(), Hs_list=[.2, 2], Tp_list=[4], n_seeds=3, n_envs=2)
        self.assertEqual(configs, [(2,.2,4,0),(1,.2,4,2),(2,2,4,0),(1,2,4,2)])

    def test_vehicle_seed_reset(self):
        for sea in (Ocean(), RealOcean()):
            v = FlyingBoatVehicle(Aircraft(), sea)
            v.reset(seed=9)
            first = v.z
            self.assertEqual(v.sea.seed, 9)
            v.step()
            v.reset(seed=9)
            self.assertEqual(v.z, first)

    def test_disarmed_and_failed_thrust(self):
        v = FlyingBoatVehicle(Aircraft(), Ocean(Hs=0))
        v.send_servo(1, 2000); v.step()
        self.assertEqual(v.read_telemetry()[2]['T'], 0)
        v.arm(); v.send_command(MAV_CMD_NAV_TAKEOFF)
        v.damage.water_mass = 1
        v.step()
        self.assertTrue(v.damage.failed)
        self.assertEqual(v.read_telemetry()[2]['T'], 0)
        with self.assertRaises(RuntimeError): v.arm()

    def test_cl_capped_at_stall(self):
        from aircraft import Aircraft
        ac = Aircraft()
        self.assertLessEqual(abs(ac.CL(math.radians(30))), ac.aero.CL_max + 1e-9)
        self.assertLessEqual(abs(ac.CL(math.radians(-30))), ac.aero.CL_max + 1e-9)
        self.assertTrue(math.isfinite(ac.CD(ac.CL(math.radians(30)))))

    def test_takeoff_setpoint_is_two_stage(self):
        from mavlink_if import LowLevelController
        ctl = LowLevelController()
        pitch, thr = ctl.takeoff_setpoint({"z": 0.3, "Vx": 2.0, "Vz": 0.0},
                                          10.0, 13.0)
        self.assertAlmostEqual(pitch, math.radians(4.0))
        self.assertEqual(thr, 1.0)
        pitch2, _ = ctl.takeoff_setpoint({"z": 0.3, "Vx": 10.0, "Vz": 0.0},
                                         10.0, 13.0)
        self.assertGreater(pitch2, math.radians(4.0))
        self.assertLessEqual(pitch2, math.radians(15.0))

    def test_hil_state_reports_accelerations(self):
        v = FlyingBoatVehicle(Aircraft(), Ocean(Hs=0))
        v.arm()
        v.send_command(MAV_CMD_NAV_TAKEOFF)
        v.step()
        hil, _, _ = v.read_telemetry()
        # Accelerating forward on step: positive longitudinal accel
        self.assertGreater(hil.fields["xacc"], 0.0)
        self.assertEqual(hil.fields["yacc"], 0.0)
        # Rest vertical accel: specific-force style ~ -9.81
        self.assertAlmostEqual(hil.fields["zacc"], -9.81, delta=2.0)
        # No rotational dynamics: angular rates are zero, not velocity
        self.assertEqual(hil.fields["pitchspeed"], 0.0)

    def test_direct_action_and_servo_are_consumed(self):
        v = FlyingBoatVehicle(Aircraft(), Ocean(Hs=0))
        v.arm(); v.send_command(MAV_CMD_NAV_TAKEOFF)
        v.step(action=[.2, -1])
        self.assertEqual(v.throttle, .2)
        # RL action envelope is [-8, 12] deg, shared with EnvConfig
        self.assertAlmostEqual(v.alpha, math.radians(-8))
        v.send_command(MAV_CMD_DO_SET_SERVO, {'servo':1, 'pwm':1300})
        v.step()
        self.assertAlmostEqual(v.throttle, .3)

    def test_sling_counts_entries_not_timesteps(self):
        s = DamageState(); spray = SprayModel(); ingress = IngressModel(submersion_rate=0)
        def step(z): update_damage(s, .01, z, 0, 0, 0, .3, spray, ingress)
        step(-1); step(-1)
        self.assertEqual(s.cumulative_sling_events, 1)
        self.assertFalse(s.failed)
        step(2); step(-1)
        self.assertTrue(s.failed)

    def test_no_airborne_ingress_and_drag_symmetry(self):
        self.assertEqual(IngressModel().step(0, 1, 10, 0, .3, 12, 2), 0)
        drag = HullDrag()
        self.assertEqual(drag.resistance(0), 0)
        self.assertEqual(drag.resistance(-3), drag.resistance(3))

    def test_buoy_fallback_and_zero_component(self):
        from ocean_real import load_buoy_default
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            sp, rt = Path(tmp)/'spectral', Path(tmp)/'realtime'
            sp.write_text('# empty\n')
            rt.write_text('2026 09 19 16 10 100 2 3 1.5 6 5\n')
            sea = load_buoy_default(rt, sp)
            self.assertAlmostEqual(math.hypot(sea.Hs_swell, sea.Hs_wind), 1.5)
            sp.write_text('2026 09 19 16 10 1.5 1.5 6 0 0 NW WNW STEEP\n')
            sea = load_buoy_default(rt, sp)
            self.assertTrue(np.isfinite(sea.eta([0], 0)).all())

    def test_vector_training_records_pre_action_state(self):
        import accelerate
        original = ActorCritic.update
        captured = []
        def check(p, batch, **kwargs):
            captured.append(batch['state'][0].copy())
            return original(p, batch, **kwargs)
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(accelerate, 'OUT', tmp), patch.object(ActorCritic, 'update', check):
                accelerate.train_vectorised(n_envs=1, episodes=1, max_updates=1)
        expected = FlyingBoatEnv(Aircraft()).reset(seed=0)
        np.testing.assert_array_equal(captured[0], expected)

    def test_short_training_runs(self):
        import train, accelerate
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(train, 'OUT', tmp):
                p, h = train.train(EnvConfig(max_steps=8), episodes=2, max_updates=2)
                self.assertTrue(np.isfinite(h['v_loss']).all())
            with patch.object(accelerate, 'OUT', tmp):
                p, h = accelerate.train_vectorised(n_envs=2, episodes=2, max_updates=2)
                self.assertTrue(np.isfinite(p.body.get_flat()).all())
                self.assertEqual(len(h['success']), 2)


if __name__ == '__main__':
    unittest.main()
