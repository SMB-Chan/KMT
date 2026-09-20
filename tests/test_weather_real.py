"""Regression tests for historical NDBC weather replay (weather_real.py).

Covers: parsing of the bundled and synthetic NDBC realtime2 meteorological
columns (chronological ordering, 'MM' interpolation, required-column guards),
the MetSeries interpolation conventions (component-based wind so the
359->1 deg wrap cannot spin, edge clamping, nearest-stamp labelling), and
HistoricalWeather as a drop-in Weather/Atmosphere whose temperature,
pressure, humidity and wind track the interpolated record exactly and
reproducibly, including the env.py wiring.
"""
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from aircraft import Aircraft
from atmosphere import (AtmosphereConfig, isa_pressure, isa_temperature)
from env import EnvConfig, FlyingBoatEnv
from weather import Weather, WeatherConfig
from weather_real import (GUST_SIGMA_FACTOR, DEFAULT_MET_PATH,
                          HistoricalWeather, MetSeries, parse_ndbc_met)

HEADER = ("#YY MM DD hh mm WDIR WSPD GST WVHT DPD APD MWD PRES ATMP "
          "WTMP DEWP VIS PTDY TIDE")


def ndbc_line(yy, mo, dd, hh, mm, wdir, wspd, gst, pres, atmp, dewp,
              wvht="MM", wtmp="MM"):
    """One realtime2 row through DEWP (index 15)."""
    return (f"{yy:4d} {mo:2d} {dd:2d} {hh:2d} {mm:2d} {wdir} {wspd} {gst} "
            f"{wvht} MM MM MM {pres} {atmp} {wtmp} {dewp} MM MM MM")


# Chronological-on-disk synthetic record; written newest-first below to
# exercise the parser's sort.  Rows C->D carry a 350 deg -> 10 deg wrap.
ROWS_CHRONO = [
    ndbc_line(2026, 1, 1, 0, 0, 0, 4, 6, 1013.25, 15.0, 10.0),
    ndbc_line(2026, 1, 1, 1, 0, 90, 8, 10, 1010.0, 14.0, "MM"),
    ndbc_line(2026, 1, 1, 2, 0, 350, 6, 7, 1015.0, 16.0, 8.0),
    ndbc_line(2026, 1, 1, 3, 0, 10, 5, 6, 1012.0, 15.0, 9.0),
]


def write_ndbc(tmp, rows):
    """Write `rows` (newest-first) plus header; return the file path."""
    p = Path(tmp) / "ndbc.txt"
    p.write_text(HEADER + "\n" + "\n".join(reversed(rows)) + "\n",
                 encoding="utf-8")
    return p


class TestParseBundled(unittest.TestCase):
    """The shipped NDBC 46012 record parses into a clean chronology."""

    @classmethod
    def setUpClass(cls):
        cls.s = parse_ndbc_met()

    def test_shape_and_chronology(self):
        self.assertGreater(len(self.s), 1000)
        self.assertTrue(np.all(np.diff(self.s.hours) > 0))
        self.assertAlmostEqual(self.s.span_hours,
                               float(self.s.hours[-1] - self.s.hours[0]))
        self.assertLess(self.s.stamps[0], self.s.stamps[-1])

    def test_required_columns_finite(self):
        for col in (self.s.wdir_deg, self.s.wspd_m_s,
                    self.s.pres_hPa, self.s.atmp_C):
            self.assertTrue(np.all(np.isfinite(col)))

    def test_humidity_bounds_and_components(self):
        self.assertTrue(np.all(self.s.rh >= 0.0) and np.all(self.s.rh <= 1.0))
        th = np.radians(self.s.wdir_deg)
        np.testing.assert_allclose(self.s.u_north,
                                   -self.s.wspd_m_s * np.cos(th), atol=1e-9)
        np.testing.assert_allclose(self.s.v_east,
                                   -self.s.wspd_m_s * np.sin(th), atol=1e-9)

    def test_gust_excess_nonneg(self):
        self.assertTrue(np.all(self.s.gust_excess() >= 0.0))

    def test_interp_node_exact_and_vector_consistent(self):
        i = len(self.s) // 3
        o = self.s.interp(float(self.s.hours[i]))
        self.assertAlmostEqual(o["pres_hPa"], float(self.s.pres_hPa[i]))
        self.assertAlmostEqual(o["atmp_C"], float(self.s.atmp_C[i]))
        self.assertAlmostEqual(o["u_north"], float(self.s.u_north[i]))
        self.assertAlmostEqual(o["v_east"], float(self.s.v_east[i]))
        h = float(self.s.hours[i]) + 0.37
        o = self.s.interp(h)
        self.assertAlmostEqual(o["wspd_m_s"],
                               math.hypot(o["u_north"], o["v_east"]))
        self.assertAlmostEqual(
            o["wdir_deg"],
            math.degrees(math.atan2(-o["v_east"], -o["u_north"])) % 360.0)

    def test_interp_clamps_to_record_span(self):
        self.assertAlmostEqual(self.s.interp(-50.0)["pres_hPa"],
                               float(self.s.pres_hPa[0]))
        self.assertAlmostEqual(self.s.interp(1e6)["pres_hPa"],
                               float(self.s.pres_hPa[-1]))

    def test_stamp_at_nearest_record(self):
        self.assertEqual(self.s.stamp_at(float(self.s.hours[0]) + 0.4),
                         self.s.stamps[0])
        self.assertEqual(self.s.stamp_at(float(self.s.hours[0]) + 0.6),
                         self.s.stamps[1])


class TestParseSynthetic(unittest.TestCase):
    """Controlled files: ordering, 'MM' handling, guards, wrap behaviour."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.path = write_ndbc(self._tmp.name, ROWS_CHRONO)

    def test_newest_first_is_sorted_chronological(self):
        s = parse_ndbc_met(self.path)
        np.testing.assert_allclose(s.hours, [0.0, 1.0, 2.0, 3.0])
        self.assertEqual(s.stamps[0], "2026-01-01 00:00")
        self.assertEqual(s.stamps[-1], "2026-01-01 03:00")

    def test_missing_dewp_is_interpolated(self):
        s = parse_ndbc_met(self.path)
        # row B DEWP = MM -> linear midpoint of 10.0 and 8.0
        self.assertAlmostEqual(float(s.dewp_C[1]), 9.0)
        self.assertTrue(np.all(np.isfinite(s.rh)))

    def test_component_interpolation_does_not_spin_wrap(self):
        s = parse_ndbc_met(self.path)
        o = s.interp(2.5)          # between 350 deg and 10 deg
        self.assertFalse(90.0 < o["wdir_deg"] < 270.0,
                         f"wrap spun the long way: {o['wdir_deg']}")

    def test_required_column_all_missing_raises(self):
        bad = [r.replace(" 1013.25 ", " MM ").replace(" 1010.0 ", " MM ")
                .replace(" 1015.0 ", " MM ").replace(" 1012.0 ", " MM ")
               for r in ROWS_CHRONO]
        p = write_ndbc(self._tmp.name, bad)
        with self.assertRaises(ValueError):
            parse_ndbc_met(p)

    def test_too_few_records_raises(self):
        p = write_ndbc(self._tmp.name, ROWS_CHRONO[:1])
        with self.assertRaises(ValueError):
            parse_ndbc_met(p)

    def test_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            parse_ndbc_met(Path(self._tmp.name) / "nope.txt")


class TestMetSeriesEdge(unittest.TestCase):
    def test_gust_none_yields_zero_excess(self):
        s = MetSeries(hours=np.array([0.0, 1.0]), stamps=("a", "b"),
                      wdir_deg=np.array([0.0, 0.0]),
                      wspd_m_s=np.array([4.0, 4.0]), gust_m_s=None,
                      pres_hPa=np.array([1013.0, 1013.0]),
                      atmp_C=np.array([15.0, 15.0]), dewp_C=None,
                      wtmp_C=None, rh=None,
                      u_north=np.array([-4.0, -4.0]),
                      v_east=np.array([0.0, 0.0]))
        np.testing.assert_allclose(s.gust_excess(), [0.0, 0.0])
        o = s.interp(0.5)
        self.assertIsNone(o["gust_m_s"])
        self.assertEqual(o["gust_excess_m_s"], 0.0)
        self.assertIsNone(o["rh"])


class TestHistoricalWeather(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.s = parse_ndbc_met()

    def test_constructor_validation(self):
        with self.assertRaises(TypeError):
            HistoricalWeather(123)
        with self.assertRaises(ValueError):
            HistoricalWeather(self.s, t0_hours=float("nan"))
        with self.assertRaises(ValueError):
            HistoricalWeather(self.s, time_scale=-1.0)
        with self.assertRaises(ValueError):
            HistoricalWeather(self.s, time_scale=float("nan"))

    def test_accepts_path_string(self):
        hw = HistoricalWeather(str(DEFAULT_MET_PATH))
        self.assertIsInstance(hw.series, MetSeries)

    def test_record_hour_mapping_and_clamp(self):
        hw = HistoricalWeather(self.s, t0_hours=10.0, time_scale=360.0)
        self.assertAlmostEqual(hw.record_hour(10.0), 11.0)
        self.assertAlmostEqual(hw.record_hour(-100.0),
                               float(self.s.hours[0]))
        self.assertAlmostEqual(hw.record_hour(1e9), float(self.s.hours[-1]))

    def test_time_scale_zero_freezes_observed_record(self):
        hw = HistoricalWeather(self.s, t0_hours=100.0, time_scale=0.0)
        self.assertEqual(hw.observed(0.0), hw.observed(5000.0))
        hw.wind(0.0, 30.0)
        snap0 = (hw.config, hw._atm.config.wind, hw._atm.config.gust_rms)
        hw.wind(5000.0, 30.0)
        snap1 = (hw.config, hw._atm.config.wind, hw._atm.config.gust_rms)
        self.assertEqual(snap0, snap1)

    def test_summary_tracks_interpolated_record(self):
        hw = HistoricalWeather(self.s, t0_hours=500.0, time_scale=360.0)
        t, z = 1234.0, 30.0
        obs = hw.observed(t)
        s = hw.summary(t, z, 11.3)
        self.assertAlmostEqual(
            s["temperature_C"],
            obs["atmp_C"] + (isa_temperature(z) - 288.15), places=9)
        self.assertAlmostEqual(
            s["pressure_hPa"],
            (isa_pressure(z) + (obs["pres_hPa"] - 1013.25) * 100.0) / 100.0,
            places=9)
        self.assertAlmostEqual(s["humidity"], obs["rh"], places=9)
        self.assertEqual(s["record_utc"], obs["stamp"])
        self.assertAlmostEqual(s["record_hours_from_start"], obs["hour"])

    def test_wind_equals_equivalent_plain_weather(self):
        seed, t, z = 7, 900.0, 25.0
        hw = HistoricalWeather(self.s, seed=seed, t0_hours=200.0,
                               time_scale=360.0)
        obs = hw.observed(t)
        base = WeatherConfig(
            temp_offset_K=obs["atmp_C"] - 15.0,
            pressure_offset_Pa=(obs["pres_hPa"] - 1013.25) * 100.0,
            humidity=obs["rh"])
        atm = AtmosphereConfig(
            wind=(obs["u_north"], obs["v_east"], 0.0),
            gust_rms=GUST_SIGMA_FACTOR * obs["gust_excess_m_s"])
        ref = Weather(base, atm, seed=seed)
        np.testing.assert_allclose(hw.wind(t, z), ref.wind(t, z), atol=1e-12)

    def test_deterministic_per_seed(self):
        a = HistoricalWeather(self.s, seed=5, t0_hours=964.0)
        b = HistoricalWeather(self.s, seed=5, t0_hours=964.0)
        c = HistoricalWeather(self.s, seed=6, t0_hours=964.0)
        for t in (0.0, 60.0, 120.0):
            np.testing.assert_allclose(a.wind(t, 30.0), b.wind(t, 30.0),
                                       atol=0.0)
            self.assertFalse(np.allclose(a.wind(t, 30.0), c.wind(t, 30.0),
                                         atol=1e-9))


class TestEnvWiring(unittest.TestCase):
    def test_env_replays_record_and_reports_stamp(self):
        ecfg = EnvConfig(spatial=True, weather_real=str(DEFAULT_MET_PATH),
                         weather_real_t0=964.0, weather_real_rate=360.0,
                         max_steps=4)
        env = FlyingBoatEnv(Aircraft(), ecfg)
        env.reset(seed=3)
        a = np.zeros(env.action_dim)
        a[0] = 1.0
        info = None
        for _ in range(3):
            _, _, _, info = env.step(a)
        self.assertIn("record_utc", info["weather"])
        self.assertIsInstance(info["weather"]["record_utc"], str)
        self.assertEqual(
            env.episode_conditions["weather_real"]["path"],
            str(DEFAULT_MET_PATH))

    def test_guards(self):
        with self.assertRaises(ValueError):
            FlyingBoatEnv(Aircraft(),
                          EnvConfig(spatial=False,
                                    weather_real=str(DEFAULT_MET_PATH)))
        with self.assertRaises(ValueError):
            FlyingBoatEnv(Aircraft(),
                          EnvConfig(spatial=True,
                                    weather_real=str(DEFAULT_MET_PATH),
                                    weather_real_rate=-1.0))


if __name__ == "__main__":
    unittest.main()
