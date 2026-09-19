import math
import unittest
import numpy as np

from aircraft import Aircraft
from env import EnvConfig, FlyingBoatEnv
from ocean_directional import DirectionalOcean, cos2s_spreading
import advisor


class NewModulesTests(unittest.TestCase):
    def test_spreading_integrates_to_one(self):
        theta = np.linspace(-math.pi, math.pi, 64, endpoint=False)
        D = cos2s_spreading(theta, 0.0, 10)
        dtheta = theta[1] - theta[0]
        self.assertAlmostEqual(float(D.sum() * dtheta), 1.0, places=6)

    def test_eta_long_shape_finite_and_energy(self):
        sea = DirectionalOcean(Hs=1.5, Tp=6.0, seed=42)
        x = np.array([0.0, 5.0])
        t = np.linspace(0.0, 60.0, 500)
        eta = sea.eta_long(x, t)
        self.assertEqual(eta.shape, (len(t), len(x)))
        self.assertTrue(bool(np.isfinite(eta).all()))
        stats = sea.statistics()
        self.assertTrue(bool(np.isfinite(stats["Hs_observed"])))
        self.assertAlmostEqual(stats["Hs_observed"], 1.5, delta=0.35)

    def test_from_ndbc_reads_spectral_mwd_column(self):
        import math as _math
        from ocean_directional import from_ndbc
        sea = from_ndbc()
        # First data row: SwH=1.5 SwP=7.1 (dominant), MWD=305
        self.assertAlmostEqual(sea.Hs, 1.5)
        self.assertAlmostEqual(sea.Tp, 7.1)
        self.assertAlmostEqual(sea.theta_mean,
                               _math.radians((270.0 - 305.0) % 360.0))
        self.assertIn("305", sea.label)

    def test_from_ndbc_missing_mwd_reports_unknown(self):
        import tempfile, os
        from ocean_directional import from_ndbc
        with tempfile.TemporaryDirectory() as d:
            spec = os.path.join(d, "spec.txt")
            with open(spec, "w") as f:
                f.write("#YY MM DD hh mm WVHT SwH SwP WWH WWP SwD WWD STEEPNESS APD MWD\n")
                f.write("2026 09 19 16 10  1.7  1.5  7.1  0.8  5.0  NW WNW STEEP 5.9 MM\n")
            sea = from_ndbc(spectral_path=spec)
            self.assertAlmostEqual(sea.theta_mean, 0.0)
            self.assertIn("unknown", sea.label)
            rt = os.path.join(d, "rt.txt")
            with open(rt, "w") as f:
                f.write("#YY MM DD hh mm WDIR WSPD GST WVHT DPD APD MWD PRES\n")
                f.write("2026 09 19 16 00 310 6.0 8.0 2.0 9 8 270 1017.9\n")
            sea_rt = from_ndbc(realtime_path=rt,
                               spectral_path=os.path.join(d, "absent.txt"))
            self.assertAlmostEqual(sea_rt.Hs, 2.0)
            self.assertAlmostEqual(sea_rt.Tp, 9.0)
            self.assertAlmostEqual(sea_rt.theta_mean, 0.0)  # (270-270)%360
            self.assertIn("270", sea_rt.label)

    def test_eta_2d_shapes(self):
        import numpy as np
        sea = DirectionalOcean(Hs=1.5, Tp=6.0, seed=42)
        x = np.linspace(0, 50, 6); y = np.linspace(-10, 10, 4)
        e2 = sea.eta(x, y, 0.0)
        self.assertEqual(e2.shape, (4, 6))
        self.assertTrue(bool(np.isfinite(e2).all()))
        e3 = sea.eta(x, y, np.array([0.0, 0.5]))
        self.assertEqual(e3.shape, (2, 4, 6))
        self.assertTrue(bool(np.isfinite(e3).all()))

    def test_generate_curriculum_stub_runs_offline(self):
        from advisor import generate_curriculum
        cur = generate_curriculum(scenario="takeoff", n_examples=2,
                                  n_refinements=1, seeds=[300, 301],
                                  use_phi=False)
        self.assertEqual(len(cur), 2)
        for e in cur:
            self.assertTrue(0.0 <= e["action_throttle"] <= 1.0)
            self.assertTrue(math.isfinite(e["action_pitch_deg"]))
            self.assertIn(e["phase"], advisor.PHASE_LABELS_TAKEOFF)

    def test_sea_eta_1d_parity_with_ocean(self):
        import numpy as np
        from ocean import Ocean
        from ocean_directional import sea_eta_1d
        sea = Ocean(Hs=1.5, Tp=6.0, seed=3)
        x = np.array([0.0, 2.5, 10.0])
        np.testing.assert_allclose(sea_eta_1d(sea, x, 1.25), sea.eta(x, 1.25))

    def test_directional_env_rollout(self):
        import numpy as np
        from aircraft import Aircraft
        from env import EnvConfig, FlyingBoatEnv
        from ocean_directional import DirectionalOcean
        cfg = EnvConfig(scenario="takeoff", max_steps=50, directional=True,
                        theta_mean_deg=35.0, spread_s=8)
        env = FlyingBoatEnv(Aircraft(), cfg)
        s = env.reset(seed=7)
        self.assertIsInstance(env._sea, DirectionalOcean)
        self.assertTrue(bool(np.isfinite(s).all()))
        for _ in range(20):
            s, _, done, _ = env.step(np.array([0.8, 0.0], dtype=np.float32))
            self.assertTrue(bool(np.isfinite(s).all()))
            if done:
                break

    def test_directional_env_invalid_config(self):
        from aircraft import Aircraft
        from env import EnvConfig, FlyingBoatEnv
        with self.assertRaises(ValueError):
            FlyingBoatEnv(Aircraft(), EnvConfig(directional=True, spread_s=0))
        with self.assertRaises(ValueError):
            FlyingBoatEnv(Aircraft(), EnvConfig(directional=True,
                                                theta_mean_deg=float("nan")))

    def test_short_simulate_records_applied_pitch(self):
        import math as _math
        import numpy as np
        from aircraft import Aircraft
        from env import EnvConfig, FlyingBoatEnv
        import advisor as _ad
        env = FlyingBoatEnv(Aircraft(), EnvConfig(scenario="takeoff"))
        env.reset(seed=5)
        lo, hi = env.cfg.pitch_lo, env.cfg.pitch_hi
        o = _ad.short_simulate(env, (1.0, _math.radians(-8.0)), horizon_steps=4)
        self.assertAlmostEqual(o["pitch"], lo)
        self.assertTrue(o["clipped"])
        o2 = _ad.short_simulate(env, (1.0, _math.radians(4.0)), horizon_steps=4)
        self.assertAlmostEqual(o2["pitch"], _math.radians(4.0))
        self.assertFalse(o2["clipped"])

    def test_curriculum_records_applied_pitch(self):
        import math as _math
        from advisor import generate_curriculum
        from env import EnvConfig
        cur = generate_curriculum(scenario="landing", n_examples=2,
                                  n_refinements=0, seeds=[400, 401],
                                  use_phi=False)
        # Stub commands -5 deg on approach; env floor is -3 deg.
        for e in cur:
            if e["phase"] == "approach":
                self.assertAlmostEqual(e["action_pitch_deg"],
                                       _math.degrees(EnvConfig().pitch_lo))

    def test_directional_sweeps_run(self):
        import numpy as np
        from policy import ActorCritic
        import accelerate
        pol = ActorCritic(8, 2, np.array([0., -1.]),
                          np.array([1., 1.]), hidden=4, seed=0)
        res = accelerate.sea_state_sweep(
            pol, scenario="takeoff", Hs_list=[0.3], Tp_list=[4.0],
            n_seeds=1, n_envs=1, directional=True,
            theta_mean_deg=35.0, spread_s=8)
        self.assertIn((0.3, 4.0), res)
        rew, succ = res[(0.3, 4.0)]
        self.assertTrue(math.isfinite(rew))
        self.assertTrue(0.0 <= succ <= 1.0)
        res_m = accelerate.mavlink_sea_sweep(
            pol, scenario="takeoff", Hs_list=[0.3], Tp_list=[4.0],
            n_seeds=1, duration=2.0, directional=True,
            theta_mean_deg=35.0, spread_s=8)
        self.assertIn((0.3, 4.0), res_m)

    def test_directional_sweep_rejects_bad_spread(self):
        import numpy as np
        from policy import ActorCritic
        import accelerate
        pol = ActorCritic(8, 2, np.array([0., -1.]),
                          np.array([1., 1.]), hidden=4, seed=0)
        with self.assertRaises(ValueError):
            accelerate.mavlink_sea_sweep(
                pol, Hs_list=[0.3], Tp_list=[4.0], n_seeds=1,
                duration=1.0, directional=True, spread_s=0)

    def test_vehicle_accepts_directional_sea(self):
        import numpy as np
        from aircraft import Aircraft
        from mavlink_if import FlyingBoatVehicle, MAV_CMD_NAV_TAKEOFF
        from ocean_directional import DirectionalOcean
        v = FlyingBoatVehicle(Aircraft(), DirectionalOcean(seed=11))
        v.reset(seed=11)
        v.arm()
        v.send_command(MAV_CMD_NAV_TAKEOFF)
        for _ in range(5):
            v.step()
        self.assertGreater(v.t, 0.0)
        self.assertTrue(bool(np.isfinite([v.x, v.z, v.Vx, v.Vz]).all()))

    def test_short_simulate_uses_pitch_and_restores_state(self):
        env = FlyingBoatEnv(Aircraft(), EnvConfig(scenario="takeoff", max_steps=200))
        env.reset(seed=5)
        hi = (1.0, math.radians(12.0))
        lo = (1.0, math.radians(-3.0))
        o_hi = advisor.short_simulate(env, hi, horizon_steps=8)
        self.assertEqual(env._steps, 0)
        self.assertEqual(env._t, 0.0)
        self.assertFalse(env._done)
        self.assertEqual(len(env._traj), 0)
        o_lo = advisor.short_simulate(env, lo, horizon_steps=8)
        self.assertFalse(np.isclose(o_hi["dz"], o_lo["dz"])
                         and np.isclose(o_hi["dVx"], o_lo["dVx"]))


if __name__ == "__main__":
    unittest.main()
