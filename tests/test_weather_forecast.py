"""Regression tests for the VAR weather forecast model (weather_forecast.py).

Covers: lead-0 exactness (the analysis is the initial condition), fit and
forecast determinism, companion-matrix stability after shrinkage, the
reference forecasts (persistence / diurnal climatology), skill helpers,
the MetSeries back-transform (wind components, gust floor, RH clip,
stamps), forecast_weather as a drop-in HistoricalWeather whose sim time 0
is the issue time, the minimum-training guard that keeps every forecast
honest, and the env.py / mavlink_if.py wiring.
"""
import math
import unittest

import numpy as np

from aircraft import Aircraft
from env import EnvConfig, FlyingBoatEnv
from ocean import Ocean
from mavlink_if import FlyingBoatVehicle
from weather_real import (DEFAULT_MET_PATH, GUST_SIGMA_FACTOR,
                          HistoricalWeather, parse_ndbc_met)
from weather_forecast import (MIN_TRAIN_HOURS, N_STATE, VarForecastModel,
                              _state_matrix, climatology_forecast,
                              forecast_weather, issue_index,
                              persistence_forecast, rmse_vs_lead,
                              skill_score)

ISSUE = 800          # a verification-window issue with enough history
HORIZON = 24


class ForecastTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.series = parse_ndbc_met(DEFAULT_MET_PATH)
        cls.y = _state_matrix(cls.series)


class TestVarModel(ForecastTestCase):
    def test_lead0_is_the_analysis_exactly(self):
        m = VarForecastModel(self.series, order=3, train_stop=ISSUE)
        fc = m.predict(ISSUE, HORIZON)
        np.testing.assert_array_equal(fc[0], self.y[ISSUE])
        self.assertEqual(fc.shape, (HORIZON + 1, N_STATE))

    def test_fit_and_forecast_are_deterministic(self):
        a = VarForecastModel(self.series, order=4, train_stop=ISSUE)
        b = VarForecastModel(self.series, order=4, train_stop=ISSUE)
        np.testing.assert_array_equal(a.coef, b.coef)
        np.testing.assert_array_equal(a.predict(ISSUE, HORIZON),
                                      b.predict(ISSUE, HORIZON))
        sa = a.forecast_series(ISSUE, HORIZON)
        sb = b.forecast_series(ISSUE, HORIZON)
        np.testing.assert_array_equal(sa.wspd_m_s, sb.wspd_m_s)
        np.testing.assert_array_equal(sa.pres_hPa, sb.pres_hPa)

    def test_companion_radius_is_stable_for_every_order(self):
        for order in (1, 2, 3, 4, 6):
            m = VarForecastModel(self.series, order=order,
                                 train_stop=ISSUE)
            self.assertLessEqual(m._companion_radius(), 0.995 + 1e-12)

    def test_training_uses_only_the_slice_before_train_stop(self):
        m = VarForecastModel(self.series, order=2, train_stop=ISSUE)
        self.assertEqual(m.train_stop, ISSUE)
        full = VarForecastModel(self.series, order=2)
        self.assertEqual(full.train_stop, len(self.series))
        self.assertFalse(np.allclose(m.coef, full.coef))

    def test_invalid_configuration_raises(self):
        with self.assertRaises(ValueError):
            VarForecastModel(self.series, order=0, train_stop=ISSUE)
        with self.assertRaises(ValueError):
            VarForecastModel(self.series, order=3, train_stop=10)
        m = VarForecastModel(self.series, order=2, train_stop=ISSUE)
        with self.assertRaises(ValueError):
            m.predict(len(self.series), 4)

    def test_forecast_damps_towards_climatology(self):
        issues = [800, 850, 900, 950, 1000]
        norms = []
        for lead in (0, HORIZON):
            vals = []
            for i in issues:
                m = VarForecastModel(self.series, order=3, train_stop=i)
                fc = m.predict(i, HORIZON)
                clim = climatology_forecast(self.series, i, HORIZON,
                                            train_stop=i)
                vals.append(np.linalg.norm(fc[lead] - clim[lead]))
            norms.append(float(np.mean(vals)))
        self.assertLess(norms[1], norms[0])


class TestReferencesAndSkill(ForecastTestCase):
    def test_persistence_holds_the_analysis(self):
        p = persistence_forecast(self.series, ISSUE, HORIZON)
        self.assertEqual(p.shape, (HORIZON + 1, N_STATE))
        for row in p:
            np.testing.assert_array_equal(row, self.y[ISSUE])

    def test_climatology_ignores_the_analysis(self):
        c = climatology_forecast(self.series, ISSUE, HORIZON,
                                 train_stop=ISSUE)
        m = VarForecastModel(self.series, order=2, train_stop=ISSUE)
        np.testing.assert_allclose(c[0], m.clim.at_lead(ISSUE, 0))
        np.testing.assert_allclose(c[HORIZON],
                                   m.clim.at_lead(ISSUE, HORIZON))
        self.assertGreater(np.abs(c[6] - c[18]).sum(), 0.0)  # diurnal cycle

    def test_rmse_vs_lead_shapes_and_zero(self):
        truth = self.y[ISSUE:ISSUE + HORIZON + 1]
        single = rmse_vs_lead(truth, truth)
        self.assertEqual(single.shape, (HORIZON + 1, N_STATE))
        np.testing.assert_allclose(single, 0.0, atol=1e-12)
        stack = np.stack([truth + 1.0, truth + 1.0])
        rms = rmse_vs_lead(stack, truth)
        np.testing.assert_allclose(rms, 1.0)
        half = rmse_vs_lead(np.stack([truth, truth + 1.0]), truth)
        np.testing.assert_allclose(half, math.sqrt(0.5))

    def test_skill_score_endpoints(self):
        np.testing.assert_allclose(skill_score(2.0, 2.0), 0.0)
        np.testing.assert_allclose(skill_score(0.0, 2.0), 1.0)
        np.testing.assert_allclose(skill_score(3.0, 2.0), -0.5)

    def test_issue_index_is_nearest_hour(self):
        self.assertEqual(issue_index(self.series,
                                     float(self.series.hours[ISSUE])),
                         ISSUE)
        mid = 0.5 * (self.series.hours[ISSUE]
                     + self.series.hours[ISSUE + 1])
        self.assertIn(issue_index(self.series, mid), (ISSUE, ISSUE + 1))


class TestForecastSeries(ForecastTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.model = VarForecastModel(cls.series, order=3, train_stop=ISSUE)
        cls.fc = cls.model.forecast_series(ISSUE, HORIZON)

    def test_shape_stamps_and_hours(self):
        self.assertEqual(len(self.fc), HORIZON + 1)
        np.testing.assert_array_equal(self.fc.hours,
                                      np.arange(HORIZON + 1, dtype=float))
        self.assertEqual(self.fc.stamps[0], self.series.stamps[ISSUE])
        self.assertEqual(self.fc.stamps[HORIZON],
                         self.series.stamps[ISSUE + HORIZON])

    def test_fields_finite_and_physical(self):
        for arr in (self.fc.wdir_deg, self.fc.wspd_m_s, self.fc.gust_m_s,
                    self.fc.pres_hPa, self.fc.atmp_C, self.fc.dewp_C,
                    self.fc.rh, self.fc.u_north, self.fc.v_east):
            self.assertTrue(np.isfinite(arr).all())
        self.assertTrue((self.fc.wdir_deg >= 0.0).all())
        self.assertTrue((self.fc.wdir_deg < 360.0).all())
        self.assertTrue((self.fc.wspd_m_s >= 0.0).all())
        self.assertTrue((self.fc.gust_m_s >= self.fc.wspd_m_s - 1e-12).all())
        self.assertTrue(((self.fc.rh >= 0.0) & (self.fc.rh <= 1.0)).all())

    def test_lead0_reproduces_the_observation(self):
        obs = self.series
        self.assertAlmostEqual(float(self.fc.wspd_m_s[0]),
                               float(obs.wspd_m_s[ISSUE]), places=9)
        self.assertAlmostEqual(float(self.fc.wdir_deg[0]),
                               float(obs.wdir_deg[ISSUE]), places=6)
        self.assertAlmostEqual(float(self.fc.pres_hPa[0]),
                               float(obs.pres_hPa[ISSUE]), places=9)
        self.assertAlmostEqual(float(self.fc.atmp_C[0]),
                               float(obs.atmp_C[ISSUE]), places=9)
        self.assertAlmostEqual(
            float(self.fc.gust_m_s[0] - self.fc.wspd_m_s[0]),
            float(obs.gust_m_s[ISSUE] - obs.wspd_m_s[ISSUE]), places=9)

    def test_label_names_model_and_issue(self):
        self.assertIn("VAR(3)", self.fc.label)
        self.assertIn(self.series.stamps[ISSUE], self.fc.label)


class TestForecastWeather(ForecastTestCase):
    def test_drop_in_weather_starts_at_the_issue(self):
        hw, model = forecast_weather(self.series, ISSUE, HORIZON,
                                     time_scale=360.0, seed=3)
        self.assertIsInstance(hw, HistoricalWeather)
        self.assertIsInstance(model, VarForecastModel)
        sm = hw.summary(0.0, 0.0, 11.3)
        self.assertEqual(sm["record_utc"], self.series.stamps[ISSUE])
        self.assertAlmostEqual(sm["temperature_C"],
                               float(self.series.atmp_C[ISSUE]), places=9)
        self.assertAlmostEqual(sm["pressure_hPa"],
                               float(self.series.pres_hPa[ISSUE]), places=9)
        self.assertAlmostEqual(sm["humidity"],
                               float(self.series.rh[ISSUE]), places=9)
        # mean wind and gust level come from the analysis; the instantaneous
        # wind adds seeded turbulence on top, so check the synced config
        self.assertAlmostEqual(hw._atm.config.wind[0],
                               float(self.series.u_north[ISSUE]), places=9)
        self.assertAlmostEqual(hw._atm.config.wind[1],
                               float(self.series.v_east[ISSUE]), places=9)
        self.assertAlmostEqual(
            hw._atm.config.gust_rms,
            GUST_SIGMA_FACTOR * float(self.series.gust_excess()[ISSUE]),
            places=9)

    def test_time_scale_maps_sim_seconds_onto_lead_hours(self):
        hw, _ = forecast_weather(self.series, ISSUE, HORIZON,
                                 time_scale=360.0)
        self.assertAlmostEqual(hw.record_hour(10.0), 1.0, places=9)
        self.assertEqual(hw.series.stamp_at(1.0),
                         self.series.stamps[ISSUE + 1])
        far = hw.series.interp(10.0 * HORIZON)      # clamped at the horizon
        self.assertAlmostEqual(float(far["pres_hPa"]),
                               float(hw.series.pres_hPa[HORIZON]),
                               places=9)

    def test_minimum_training_guard(self):
        with self.assertRaises(ValueError):
            forecast_weather(self.series, 100, HORIZON)
        with self.assertRaises(ValueError):
            forecast_weather(self.series, ISSUE, HORIZON, train_stop=100)
        hw, model = forecast_weather(self.series, ISSUE, HORIZON)
        self.assertEqual(model.train_stop, ISSUE)   # default = issue
        self.assertGreaterEqual(model.train_stop, MIN_TRAIN_HOURS)

    def test_atmosphere_interface_is_usable(self):
        hw, _ = forecast_weather(self.series, ISSUE, HORIZON, seed=1)
        w = hw.wind(120.0)
        self.assertEqual(len(w), 3)
        self.assertTrue(all(math.isfinite(c) for c in w))
        self.assertGreater(hw.density(30.0), 1.0)


class TestEnvWiring(ForecastTestCase):
    def _cfg(self, **kw):
        base = dict(spatial=True, weather_forecast=str(DEFAULT_MET_PATH),
                    weather_forecast_issue=float(self.series.hours[ISSUE]),
                    weather_forecast_rate=360.0)
        base.update(kw)
        return EnvConfig(**base)

    def test_reset_builds_the_forecast_driver(self):
        env = FlyingBoatEnv(Aircraft(), self._cfg())
        env.reset(seed=5)
        cond = env.episode_conditions["weather_forecast"]
        self.assertEqual(cond["path"], str(DEFAULT_MET_PATH))
        self.assertEqual(cond["issue_hours"],
                         float(self.series.hours[ISSUE]))
        self.assertEqual(cond["rate"], 360.0)
        self.assertTrue(env._weather.series.label.startswith("VAR("))
        sm = env._weather.summary(0.0, 0.0, 11.3)
        self.assertEqual(sm["record_utc"], self.series.stamps[ISSUE])

    def test_env_forecast_matches_study_fit(self):
        env = FlyingBoatEnv(Aircraft(), self._cfg())
        env.reset(seed=5)
        hw, _ = forecast_weather(self.series, ISSUE,
                                 max(24, len(self.series) - 1 - ISSUE),
                                 time_scale=360.0, train_stop=ISSUE)
        np.testing.assert_array_equal(env._weather.series.wspd_m_s,
                                      hw.series.wspd_m_s)

    def test_step_telemetry_advances_along_the_lead(self):
        env = FlyingBoatEnv(Aircraft(),
                            self._cfg(weather_forecast_rate=3600.0))
        env.reset(seed=5)
        stamps = []
        for _ in range(40):
            a = np.zeros(env.action_dim)
            a[0] = 1.0
            _, _, done, info = env.step(a)
            stamps.append(info["weather"]["record_utc"])
            if done:
                break
        self.assertEqual(stamps[0], self.series.stamps[ISSUE])
        self.assertGreater(stamps[-1], stamps[0])

    def test_guards(self):
        with self.assertRaises(ValueError):
            FlyingBoatEnv(Aircraft(), self._cfg(spatial=False))
        with self.assertRaises(ValueError):
            FlyingBoatEnv(Aircraft(), self._cfg(
                weather_real=str(DEFAULT_MET_PATH)))
        with self.assertRaises(ValueError):
            FlyingBoatEnv(Aircraft(), self._cfg(weather_forecast_rate=-1.0))
        with self.assertRaises(ValueError):
            env = FlyingBoatEnv(Aircraft(), self._cfg(
                weather_forecast_issue=100.0))
            env.reset()


class TestVehicleWiring(ForecastTestCase):
    def test_vehicle_reset_builds_the_forecast_driver(self):
        veh = FlyingBoatVehicle(
            Aircraft(), Ocean(Hs=0.2, seed=42), spatial=True,
            weather_forecast=str(DEFAULT_MET_PATH),
            weather_forecast_issue=float(self.series.hours[ISSUE]),
            weather_forecast_rate=360.0)
        veh.reset()
        self.assertTrue(veh.weather.series.label.startswith("VAR("))
        sm = veh.weather.summary(0.0, 0.0, 11.3)
        self.assertEqual(sm["record_utc"], self.series.stamps[ISSUE])

    def test_vehicle_guards(self):
        with self.assertRaises(ValueError):
            FlyingBoatVehicle(Aircraft(), Ocean(Hs=0.0),
                              weather_forecast=str(DEFAULT_MET_PATH))
        with self.assertRaises(ValueError):
            FlyingBoatVehicle(Aircraft(), Ocean(Hs=0.0), spatial=True,
                              weather_forecast=str(DEFAULT_MET_PATH),
                              weather_real=str(DEFAULT_MET_PATH))
        with self.assertRaises(ValueError):
            FlyingBoatVehicle(Aircraft(), Ocean(Hs=0.0), spatial=True,
                              weather_forecast=str(DEFAULT_MET_PATH),
                              weather_forecast_rate=-1.0)


if __name__ == "__main__":
    unittest.main()
