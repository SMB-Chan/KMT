"""Regression tests for the environment-model enhancements.

Covers: JONSWAP peakedness and finite-depth dispersion (ocean.py,
ocean_directional.py), the Dryden-shaped gust model (atmosphere.py), and
surface-current support (spatial_dynamics.py, env.py, mavlink_if.py).
Every default must reproduce the previous behaviour exactly.
"""
import math
import unittest

import numpy as np

from aircraft import Aircraft
from atmosphere import Atmosphere, AtmosphereConfig
from dynamics import HullContact, HullDrag
from env import EnvConfig, FlyingBoatEnv
from mavlink_if import FlyingBoatVehicle
from ocean import Ocean, G, dispersion_k, jonswap_spectrum, pm_spectrum
from ocean_directional import DirectionalOcean
from spatial_dynamics import integrate


class JonswapSpectrumTests(unittest.TestCase):
    def test_gamma_one_is_pm_bitwise(self):
        omega = np.linspace(0.2, 4.0, 64)
        np.testing.assert_array_equal(
            jonswap_spectrum(omega, 1.5, 6.0, 1.0),
            pm_spectrum(omega, 1.5, 6.0))

    def test_peak_enhanced_and_energy_preserved(self):
        omega = np.linspace(0.05, 6.0, 4000)
        Hs, Tp, gamma = 2.0, 8.0, 3.3
        S_pm = pm_spectrum(omega, Hs, Tp)
        S_js = jonswap_spectrum(omega, Hs, Tp, gamma)
        d = omega[1] - omega[0]
        m0_pm = float(S_pm.sum() * d)
        m0_js = float(S_js.sum() * d)
        # A_gamma keeps m0 (hence Hs) close to the PM value.
        self.assertAlmostEqual(m0_js, m0_pm, delta=0.05 * m0_pm)
        self.assertAlmostEqual(m0_js, Hs ** 2 / 16.0, delta=0.02 * Hs ** 2)
        # Enhancement peaks at omega_p and decays away from it.
        omega_p = 2 * math.pi / Tp
        i_p = int(np.argmin(np.abs(omega - omega_p)))
        self.assertGreater(S_js[i_p], S_pm[i_p])
        i_lo = int(np.argmin(np.abs(omega - 0.3 * omega_p)))
        self.assertAlmostEqual(S_js[i_lo], S_pm[i_lo], delta=1e-3 * S_pm[i_p])

    def test_invalid_gamma_raises(self):
        omega = np.linspace(0.2, 4.0, 8)
        for bad in (0.5, 0.0, -1.0, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                jonswap_spectrum(omega, 1.5, 6.0, bad)


class DispersionTests(unittest.TestCase):
    def test_deep_water_limit(self):
        omega = np.linspace(0.2, 4.0, 32)
        for depth in (None, float("inf")):
            np.testing.assert_array_equal(dispersion_k(omega, depth),
                                          omega ** 2 / G)

    def test_finite_depth_solves_dispersion(self):
        omega = np.linspace(0.1, 4.0, 50)
        for depth in (2.0, 8.0, 50.0, 200.0):
            k = dispersion_k(omega, depth)
            residual = np.abs(G * k * np.tanh(k * depth) - omega ** 2)
            self.assertTrue(np.all(residual <= 1e-10 * omega ** 2))
            # Finite depth always increases k relative to deep water.
            self.assertTrue(np.all(k >= omega ** 2 / G - 1e-12))

    def test_shallow_water_limit(self):
        omega = np.array([0.05, 0.1])
        depth = 3.0
        k = dispersion_k(omega, depth)
        np.testing.assert_allclose(k, omega / math.sqrt(G * depth), rtol=1e-3)

    def test_invalid_depth_raises(self):
        for bad in (0.0, -5.0, float("nan")):
            with self.assertRaises(ValueError):
                dispersion_k(np.array([1.0]), bad)


class OceanGammaTests(unittest.TestCase):
    def test_defaults_unchanged(self):
        sea = Ocean(Hs=1.5, Tp=6.0, seed=42)
        self.assertEqual(sea.gamma, 1.0)
        self.assertFalse(sea.finite_depth)
        np.testing.assert_array_equal(sea.k, sea.omega ** 2 / G)

    def test_gamma_does_not_disturb_rng(self):
        base = Ocean(Hs=1.5, Tp=6.0, seed=7)
        peaked = Ocean(Hs=1.5, Tp=6.0, seed=7, gamma=3.3)
        np.testing.assert_array_equal(peaked.phases, base.phases)
        self.assertFalse(np.array_equal(peaked.amps, base.amps))

    def test_gamma_sea_keeps_significant_height(self):
        stats = Ocean(Hs=1.5, Tp=6.0, seed=42, gamma=3.3).statistics()
        self.assertAlmostEqual(stats["Hs_observed"], 1.5, delta=0.35)

    def test_finite_depth_changes_long_waves_most(self):
        deep = Ocean(Hs=1.5, Tp=6.0, seed=42)
        shallow = Ocean(Hs=1.5, Tp=6.0, seed=42, finite_depth=True, depth=6.0)
        ratio = shallow.k / deep.k
        self.assertGreater(ratio[0], ratio[-1])   # low omega most affected
        self.assertTrue(np.isfinite(shallow.eta(np.array([0.0, 5.0]), 1.0)).all())
        stats = shallow.statistics()
        self.assertAlmostEqual(stats["Hs_observed"], 1.5, delta=0.35)

    def test_invalid_gamma_raises(self):
        with self.assertRaises(ValueError):
            Ocean(gamma=0.9)


class DirectionalOceanGammaTests(unittest.TestCase):
    def test_gamma_and_finite_depth_plumbed(self):
        omega = np.linspace(0.2, 4.0, 32)
        sea = DirectionalOcean(Hs=1.5, Tp=6.0, seed=42, gamma=3.3)
        np.testing.assert_allclose(
            sea.S_1d, jonswap_spectrum(sea.omega, 1.5, 6.0, 3.3))
        np.testing.assert_array_equal(sea.k, sea.omega ** 2 / G)
        shallow = DirectionalOcean(Hs=1.5, Tp=6.0, seed=42,
                                   finite_depth=True, depth=7.0)
        k = shallow.k
        residual = np.abs(G * k * np.tanh(k * 7.0) - shallow.omega ** 2)
        self.assertTrue(np.all(residual <= 1e-10 * shallow.omega ** 2))
        e2 = shallow.eta(np.linspace(0, 30, 4), np.linspace(-5, 5, 3), 0.5)
        self.assertEqual(e2.shape, (3, 4))
        self.assertTrue(np.isfinite(e2).all())

    def test_defaults_match_previous_construction(self):
        a = DirectionalOcean(Hs=1.5, Tp=6.0, seed=42)
        b = DirectionalOcean(Hs=1.5, Tp=6.0, seed=42, gamma=1.0,
                             finite_depth=False, depth=50.0)
        np.testing.assert_array_equal(a.amps, b.amps)
        np.testing.assert_array_equal(a.k, b.k)


class _LegacyAtmosphere:
    """Copy of the pre-enhancement gust implementation (bitwise reference)."""

    def __init__(self, config, seed=0):
        self.config = config
        rng = np.random.default_rng(seed)
        self.phase = rng.uniform(0, 2 * math.pi, (3, 4))
        self.frequency = (rng.uniform(0.5, 1.5, (3, 4)) * 2 * math.pi
                          / self.config.gust_period)

    def shear_factor(self, altitude):
        z = max(float(altitude), self.config.z0)
        return (math.log(z / self.config.z0)
                / math.log(self.config.z_ref / self.config.z0))

    def wind(self, t, altitude=None):
        gust = np.sin(self.frequency * t + self.phase).sum(axis=1)
        mean = np.asarray(self.config.wind, dtype=float)
        factor = 1.0 if altitude is None else self.shear_factor(altitude)
        gust = self.config.gust_rms * math.sqrt(2 / 4) * gust * factor
        return (np.array([mean[0] * factor, mean[1] * factor, mean[2]],
                         dtype=float) + gust)


class AtmosphereGustModelTests(unittest.TestCase):
    def test_default_model_bitwise_identical_to_legacy(self):
        configs = [AtmosphereConfig(wind=(4, -2, 0.5), gust_rms=0.7),
                   AtmosphereConfig(gust_rms=1.0),
                   AtmosphereConfig(wind=(1, 2, 3), gust_rms=0.5,
                                    gust_period=9.0)]
        for cfg in configs:
            for seed in (0, 3, 7):
                new = Atmosphere(cfg, seed)
                old = _LegacyAtmosphere(cfg, seed)
                for t in (0.0, 0.137, 5.5, 31.9):
                    for alt in (None, 0.3, 10.0, 55.0):
                        np.testing.assert_array_equal(new.wind(t, alt),
                                                      old.wind(t, alt))

    def test_dryden_is_deterministic_and_repeated_reads_are_stable(self):
        cfg = AtmosphereConfig(gust_rms=1.0, gust_model="dryden")
        a = Atmosphere(cfg, seed=3)
        b = Atmosphere(cfg, seed=3)
        np.testing.assert_array_equal(a.wind(1.25, 10.0), b.wind(1.25, 10.0))
        np.testing.assert_array_equal(a.wind(1.25, 10.0), a.wind(1.25, 10.0))
        self.assertFalse(np.array_equal(a.wind(1.25, 10.0),
                                        Atmosphere(cfg, 5).wind(1.25, 10.0)))

    def test_dryden_rms_matches_gust_rms(self):
        rms = 0.8
        air = Atmosphere(AtmosphereConfig(gust_rms=rms, gust_model="dryden"),
                         seed=11)
        t = np.linspace(0.0, 1200.0, 120001)
        samples = np.array([air.wind(ti, 10.0) for ti in t[::37]])
        shear = air.shear_factor(10.0)
        std = np.std(samples, axis=0) / shear
        np.testing.assert_allclose(std, rms, rtol=0.15)

    def test_dryden_retains_more_low_frequency_energy(self):
        ts = np.linspace(0.0, 600.0, 6001)
        retention = {}
        for model in ("sum4", "dryden"):
            air = Atmosphere(AtmosphereConfig(gust_rms=1.0,
                                              gust_model=model), seed=3)
            x = np.array([air.wind(t, 10.0)[0] for t in ts])
            # Variance retained by a 2*gust_period moving average is a
            # low-frequency energy fraction; the Dryden-like shape must
            # keep clearly more of it than the flat sum4 band.
            window = np.ones(120) / 120.0
            smoothed = np.convolve(x, window, mode="valid")
            retention[model] = float(np.var(smoothed) / np.var(x))
        self.assertGreater(retention["dryden"], 4.0 * retention["sum4"])

    def test_dryden_scales_with_height(self):
        air = Atmosphere(AtmosphereConfig(gust_rms=1.0, gust_model="dryden"),
                         seed=3)
        found = False
        for t in np.linspace(0.1, 30.0, 50):
            ref = abs(air.wind(t, 10.0)[0])
            if ref > 0.2:
                found = True
                self.assertLess(abs(air.wind(t, 1.0)[0]), ref)
        self.assertTrue(found)

    def test_invalid_gust_model_raises(self):
        with self.assertRaises(ValueError):
            AtmosphereConfig(gust_model="kolmogorov")


def _float_state():
    ac = Aircraft()
    hull = HullContact()
    from ocean import Ocean
    sea = Ocean(Hs=0.0)
    eta0 = 0.0
    z = eta0 + hull.h_keel - ac.W / (1025.0 * 9.80665 * hull.A_wp)
    state = np.array([0.0, 0.0, z, 0.0, 0.0, 0.0, 0.0, 0.0])
    return ac, hull, state


class SeaCurrentTests(unittest.TestCase):
    def _integrate(self, current=None, dt=0.5, n=20):
        ac, hull, state = _float_state()
        hd = HullDrag(Bwl=ac.geom.Bwl, Lwl=ac.geom.Lwl)
        atm = Atmosphere(AtmosphereConfig(), seed=0)
        surface_at = lambda x, y, t: 0.0
        kwargs = {} if current is None else dict(current=current)
        x = state.copy()
        t = 0.0
        for _ in range(n):
            x, _forces = integrate(
                ac, hull, hd, atm, surface_at, x, dt=dt, t=t,
                pitch=0.0, throttle=0.0, bank_command=0.0,
                rudder_command=0.0, **kwargs)
            t += dt
        return x

    def test_zero_current_is_bitwise_default(self):
        base = self._integrate(None)
        zero = self._integrate((0.0, 0.0))
        np.testing.assert_array_equal(base, zero)

    def test_current_drifts_floating_boat_downcurrent(self):
        still = self._integrate((0.0, 0.0))
        drifted = self._integrate((0.5, 0.25))
        self.assertGreater(drifted[0], still[0] + 0.5)   # north drift
        self.assertGreater(drifted[1], still[1] + 0.2)   # east drift
        # Approaches the current speed, does not overshoot wildly.
        self.assertLess(math.hypot(drifted[3], drifted[4]), 1.2)

    def test_head_current_slows_moving_boat(self):
        ac, hull, state = _float_state()
        hd = HullDrag(Bwl=ac.geom.Bwl, Lwl=ac.geom.Lwl)
        atm = Atmosphere(AtmosphereConfig(), seed=0)
        state[3] = 5.0
        run = dict(dt=0.1, t=0.0, pitch=0.0, throttle=0.0,
                   bank_command=0.0, rudder_command=0.0)
        calm, _ = integrate(ac, hull, hd, atm, lambda x, y, t: 0.0,
                            state.copy(), **run)
        opposed, _ = integrate(ac, hull, hd, atm, lambda x, y, t: 0.0,
                               state.copy(), current=(-2.0, 0.0), **run)
        self.assertLess(opposed[3], calm[3])

    def test_invalid_current_raises(self):
        ac, hull, state = _float_state()
        hd = HullDrag(Bwl=ac.geom.Bwl, Lwl=ac.geom.Lwl)
        atm = Atmosphere(AtmosphereConfig(), seed=0)
        for bad in ((1.0,), (1.0, 2.0, 3.0), (float("nan"), 0.0)):
            with self.assertRaises(ValueError):
                integrate(ac, hull, hd, atm, lambda x, y, t: 0.0, state,
                          dt=0.1, t=0.0, pitch=0.0, throttle=0.0,
                          bank_command=0.0, rudder_command=0.0, current=bad)


class EnvConfigEnvironmentTests(unittest.TestCase):
    def test_gamma_and_depth_reach_the_sea(self):
        cfg = EnvConfig(gamma=2.5, finite_depth=True, depth=9.0)
        env = FlyingBoatEnv(Aircraft(), cfg)
        env.reset(seed=5)
        self.assertEqual(env._sea.gamma, 2.5)
        self.assertTrue(env._sea.finite_depth)
        residual = np.abs(G * env._sea.k * np.tanh(env._sea.k * 9.0)
                          - env._sea.omega ** 2)
        self.assertTrue(np.all(residual <= 1e-10 * env._sea.omega ** 2))
        conds = env.episode_conditions
        self.assertEqual(conds["gamma"], 2.5)
        self.assertTrue(conds["finite_depth"])
        self.assertEqual(conds["current"], [0.0, 0.0])

    def test_directional_gamma_reaches_the_sea(self):
        cfg = EnvConfig(directional=True, gamma=3.3)
        env = FlyingBoatEnv(Aircraft(), cfg)
        env.reset(seed=5)
        self.assertIsInstance(env._sea, DirectionalOcean)
        self.assertEqual(env._sea.gamma, 3.3)

    def test_current_advances_spatial_state(self):
        base = EnvConfig(spatial=True, max_steps=10,
                         atmosphere=AtmosphereConfig())
        still = FlyingBoatEnv(Aircraft(), base)
        still.reset(seed=1)
        flowed = FlyingBoatEnv(Aircraft(), EnvConfig(
            spatial=True, max_steps=10, atmosphere=AtmosphereConfig(),
            current=(1.0, 0.0)))
        flowed.reset(seed=1)
        zero_action = np.zeros(4, dtype=np.float32)
        for _ in range(10):
            still.step(zero_action)
            flowed.step(zero_action)
        self.assertGreater(flowed._x, still._x)

    def test_invalid_values_raise(self):
        for kwargs in (dict(gamma=0.5), dict(current=(1.0,)),
                       dict(current=(float("nan"), 0.0)),
                       dict(finite_depth=True, depth=0.0)):
            with self.assertRaises(ValueError):
                FlyingBoatEnv(Aircraft(), EnvConfig(**kwargs))

    def test_defaults_reproduce_previous_environment(self):
        env = FlyingBoatEnv(Aircraft(), EnvConfig())
        state = env.reset(seed=42)
        self.assertEqual(env._sea.gamma, 1.0)
        np.testing.assert_array_equal(env._sea.k, env._sea.omega ** 2 / G)
        self.assertTrue(np.isfinite(state).all())


class VehicleCurrentTests(unittest.TestCase):
    def test_default_current_is_still_water(self):
        v = FlyingBoatVehicle(Aircraft(), Ocean(Hs=0.3, seed=42))
        self.assertEqual(v.current, (0.0, 0.0))

    def test_invalid_current_raises(self):
        for bad in ((1.0,), (1.0, 2.0, 3.0), (float("inf"), 0.0)):
            with self.assertRaises(ValueError):
                FlyingBoatVehicle(Aircraft(), Ocean(Hs=0), current=bad)

    def test_spatial_vehicle_drifts_with_current(self):
        def run(current):
            v = FlyingBoatVehicle(Aircraft(), Ocean(Hs=0.2, seed=42),
                                  spatial=True,
                                  atmosphere=AtmosphereConfig(),
                                  current=current)
            v.arm()
            for _ in range(100):
                v.step(0.05, np.zeros(4))
            return v.x, v.y
        still = run((0.0, 0.0))
        drifted = run((0.8, 0.0))
        self.assertAlmostEqual(still[0], drifted[0], delta=5.0)
        self.assertGreater(drifted[0], still[0])

    def test_nonspatial_vehicle_feels_current(self):
        def run(current):
            v = FlyingBoatVehicle(Aircraft(), Ocean(Hs=0.0),
                                  current=current)
            v.arm()
            for _ in range(60):
                v.step(0.05, np.zeros(2))
            return v.Vx
        still = run((0.0, 0.0))
        pushed = run((1.5, 0.0))
        self.assertGreater(pushed, still + 0.05)


if __name__ == "__main__":
    unittest.main()
