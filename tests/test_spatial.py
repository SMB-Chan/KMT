import math
import unittest
import numpy as np
from aircraft import Aircraft
from atmosphere import Atmosphere, AtmosphereConfig
from env import EnvConfig, FlyingBoatEnv


class SpatialTests(unittest.TestCase):
    def env(self, **kwargs):
        return FlyingBoatEnv(Aircraft(), EnvConfig(
            scenario='landing', Hs=0, spatial=True, max_steps=200, **kwargs))

    def test_atmosphere_reproducible_and_order_independent(self):
        config = AtmosphereConfig(wind=(1, 2, 3), gust_rms=0.5)
        a, b = Atmosphere(config, 4), Atmosphere(config, 4)
        a.wind(8)
        np.testing.assert_array_equal(a.wind(1), b.wind(1))
        self.assertFalse(np.array_equal(a.wind(1), Atmosphere(config, 5).wind(1)))
        self.assertEqual(a.density(0), 1.225)
        self.assertLess(a.density(1000), a.density(0))
        np.testing.assert_array_equal(a.wind(1), a.wind(1, 10.0))
        for kwargs in ({'wind': (1, 2)}, {'gust_rms': -1}, {'gust_period': 0},
                        {'density_scale_height': float('nan')}, {'z0': 0}, {'z_ref': 0.0005}):
            with self.assertRaises(ValueError):
                AtmosphereConfig(**kwargs)

    def test_calm_spatial_matches_longitudinal(self):
        old = FlyingBoatEnv(Aircraft(), EnvConfig(scenario='landing', Hs=0, max_steps=200))
        new = self.env()
        old.reset(8)
        state = new.reset(8)
        self.assertEqual(state.shape, (19,))
        for _ in range(30):
            _, _, _, a = old.step([0.2, 0])
            _, _, _, b = new.step([0.2, 0, 0, 0])
        for key in ('x', 'z', 'Vx', 'Vz'):
            self.assertAlmostEqual(a[key], b[key], delta=0.01)
        self.assertEqual(b['y'], 0)
        self.assertEqual(b['Vy'], 0)

    def test_crosswind_mirror_and_reset(self):
        rows = []
        for wind in (-5, 5):
            env = self.env(atmosphere=AtmosphereConfig(wind=(0, wind, 0)))
            initial = env.reset(3)
            for _ in range(40):
                state, _, _, info = env.step([0.2, 0, 0, 0])
                self.assertTrue(np.isfinite(state).all())
            rows.append(info)
            np.testing.assert_array_equal(initial, env.reset(3))
            self.assertEqual(env._y, 0)
        self.assertGreater(rows[1]['Vy'], 0)
        self.assertAlmostEqual(rows[0]['y'], -rows[1]['y'])
        self.assertAlmostEqual(rows[0]['z'], rows[1]['z'])

    def test_bank_turns_and_lift_tilts(self):
        rows = []
        for bank in (-0.5, 0.5):
            env = self.env()
            env.reset(1)
            for _ in range(30):
                _, _, _, info = env.step([0.3, 0.2, bank, 0])
            rows.append(info)
        self.assertGreater(rows[1]['heading'], 0)
        self.assertGreater(rows[1]['Vy'], 0)
        self.assertAlmostEqual(rows[0]['y'], -rows[1]['y'])
        self.assertAlmostEqual(rows[0]['bank'], -rows[1]['bank'])

    def test_directional_sea_uses_y_and_preview(self):
        env = self.env(directional=True, theta_mean_deg=45)
        env.reset(2)
        env._y = 4
        self.assertEqual(env._eta(3, 1), float(env._sea.eta([3], [4], 1)[0, 0]))
        # Nonzero waves demonstrate off-axis position changes surface samples.
        from ocean_directional import DirectionalOcean
        env._sea = DirectionalOcean(Hs=1, Tp=6, theta_mean=math.pi/4, seed=2)
        self.assertNotAlmostEqual(env._eta(3, 1), env._eta(3, 1, 0))
        state = env._build_state()
        dx = env.cfg.preview_dx_m[0]
        speed = max(math.hypot(env._Vx, env._Vy), 1)
        self.assertAlmostEqual(float(state[8]), env._eta(dx, dx/speed, 4)/1.5, places=6)

    def test_action_validation_and_lateral_failure(self):
        env = self.env()
        env.reset(0)
        for action in ([0, 0], [0, 0, float('nan'), 0]):
            with self.assertRaises(ValueError):
                env.step(action)
        env._y = 51
        _, _, done, info = env.step([0, 0, 0, 0])
        self.assertTrue(done)
        self.assertFalse(info['success'])
        self.assertTrue(info['lateral_limit'])
        with self.assertRaises(RuntimeError):
            env.step([0, 0, 0, 0])


if __name__ == '__main__':
    unittest.main()
