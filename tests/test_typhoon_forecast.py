"""Regression tests for the JMA typhoon forecast (typhoon_forecast.py and
typhoon_forecast_study.py).

Covers: the bulletin parsers (position sentences, warning areas, rain
outlooks, the time axis of the analysed track array), the snapshot rebuilt
from the committed files in data/jma_typhoon_2625, the BestTrack slicing that
keeps a hindcast honest, the ridge-VAR motion model (lead-0 exactness,
determinism, companion shrinkage, training guards, no-leakage rolling-origin
skill against the persistence references), the intensity model (pressure-wind
anchor, extratropical relaxation), the parametric rain shield and vortex wind
field (radial shape, consistency with the JMA warning radii, orographic gain,
amplitude cap), the simulator config builders (rain_terms / scale_rain
consistency, pressure offset, gust scaling), the official-anchored product
track of the study (splice, circle extrapolation, warning geometry, one-sided
speed at the end of the grid), the small study helpers, and a quick no-plots
run of typhoon_forecast_study.run_study.
"""
import json
import math
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from aircraft import Aircraft
from atmosphere import AtmosphereConfig
from weather import WeatherConfig
import typhoon_forecast as tf
import typhoon_forecast_study as ts
from typhoon_forecast import (AnalysisFix, BestTrack, OfficialForecastPoint,
                              TrackForecastModel, WarningSector,
                              bearing_deg, haversine_km, load_snapshot,
                              motion_persistence_track, orographic_factor,
                              persistence_track, rain_rate_mm_h, rain_terms,
                              scale_rain, sector_radius, state_at,
                              track_error_km, vortex_wind, wind_profile)

JST = timezone(timedelta(hours=9))
ISSUE_STAMP = datetime(2026, 9, 21, 0, 30, tzinfo=timezone.utc)
CENTRE = (32.3, 138.3)      # the last analysed position of the snapshot
ISSUE_HOUR = 107.0

# 実況 sentence, warning areas and one forecast circle, phrased as JMA phrases
# them (the shapes the regexes are written against).
BULLETIN = (
    "台風第25号は、21日09時には、北緯32度20分、東経138度20分にあって、"
    "中心の気圧は965ヘクトパスカル、中心付近の最大風速は35メートル、"
    "最大瞬間風速は50メートルで、1時間におよそ40キロの速さで"
    "北東へ進んでいます。"
    "中心から半径130キロ以内では風速25メートル以上、"
    "中心の北東側700キロ以内と南西側440キロ以内では"
    "風速15メートル以上の強い風が吹いています。"
    "21日21時には、北緯36度30分、東経144度20分を中心とする半径65キロの"
    "円内に達する見込みです。")

RAIN_SECTION = (
    "［雨の予想］<br>"
    "21日9時から22日9時までに予想される24時間降水量は多い所で<br>"
    "　伊豆諸島　300ミリ<br>　関東地方　250ミリ<br>"
    "その後、22日9時から23日9時までに予想される24時間降水量は多い所で<br>"
    "　東北地方　120ミリ<br>［風の予想］<br>")


class TyphoonTestCase(unittest.TestCase):
    """Loads the committed JMA snapshot once for every test class."""

    @classmethod
    def setUpClass(cls):
        cls.snap = load_snapshot()
        cls.track = cls.snap.track
        cls.storm, cls.gale = cls.snap.warnings_at(ISSUE_HOUR)


class TestGeometry(TyphoonTestCase):
    def test_haversine_known_distances(self):
        self.assertEqual(haversine_km(1.0, 2.0, 1.0, 2.0), 0.0)
        one_deg = haversine_km(32.3, 138.3, 33.3, 138.3)
        self.assertAlmostEqual(one_deg, 111.19, delta=0.5)
        quarter = haversine_km(0.0, 0.0, 90.0, 0.0)
        self.assertAlmostEqual(quarter, math.pi / 2 * tf.EARTH_RADIUS_KM,
                               delta=1.0)
        # a degree of longitude shrinks with the cosine of the latitude
        eq = haversine_km(0.0, 0.0, 0.0, 1.0)
        mid = haversine_km(60.0, 0.0, 60.0, 1.0)
        self.assertAlmostEqual(mid, eq * math.cos(math.radians(60.0)),
                               delta=0.5)

    def test_bearing_is_meteorological(self):
        self.assertAlmostEqual(bearing_deg(0, 0, 1, 0), 0.0, delta=1e-6)
        self.assertAlmostEqual(bearing_deg(0, 0, 0, 1), 90.0, delta=1e-6)
        self.assertAlmostEqual(bearing_deg(0, 0, -1, 0), 180.0, delta=1e-6)
        self.assertAlmostEqual(bearing_deg(0, 0, 0, -1), 270.0, delta=1e-6)
        self.assertAlmostEqual(bearing_deg(32.3, 138.3, 33.3, 139.3), 40.0,
                               delta=8.0)

    def test_sector_radius_reproduces_both_stated_radii(self):
        sym = WarningSector(0.0, 130.0, 130.0)
        for bearing in (0.0, 37.0, 180.0, 359.0):
            self.assertAlmostEqual(sector_radius(sym, bearing), 130.0)
        self.assertEqual(sym.mean_radius(), 130.0)
        two = WarningSector(45.0, 700.0, 440.0)
        self.assertAlmostEqual(sector_radius(two, 45.0), 700.0)
        self.assertAlmostEqual(sector_radius(two, 225.0), 440.0)
        self.assertAlmostEqual(sector_radius(two, 135.0), 570.0)
        self.assertEqual(two.mean_radius(), 570.0)

    def test_sector_radius_degenerates(self):
        self.assertTrue(math.isnan(sector_radius(None, 0.0)))
        self.assertTrue(math.isnan(sector_radius(WarningSector(0.0), 0.0)))
        self.assertTrue(math.isnan(WarningSector(0.0).mean_radius()))
        # a single stated radius stands for a circular area
        self.assertEqual(WarningSector(0.0, 95.0).mean_radius(), 95.0)
        self.assertAlmostEqual(sector_radius(WarningSector(0.0, 95.0), 200.0),
                               95.0)

    def test_orographic_factor_is_unity_over_the_open_ocean(self):
        self.assertAlmostEqual(orographic_factor(0.0, 160.0), 1.0, delta=1e-9)
        for lat, lon, _sigma, gain in tf.OROGRAPHIC_ANCHORS:
            self.assertGreater(orographic_factor(lat, lon), 1.0 + 0.4 * gain)
        # the gain is local: far from every anchor it is back to unity
        self.assertAlmostEqual(orographic_factor(35.0, 160.0), 1.0, delta=1e-6)


class TestBulletinParsing(TyphoonTestCase):
    def test_analysis_sentence(self):
        fixes, forecasts = tf.parse_bulletin_text(BULLETIN, ISSUE_STAMP,
                                                  source="synthetic")
        self.assertEqual(len(fixes), 1)
        f = fixes[0]
        self.assertEqual(f.stamp, datetime(2026, 9, 21, 0, 0,
                                           tzinfo=timezone.utc))
        self.assertAlmostEqual(f.lat, 32.0 + 20.0 / 60.0)
        self.assertAlmostEqual(f.lon, 138.0 + 20.0 / 60.0)
        self.assertEqual((f.pres_hPa, f.vmax_m_s, f.gust_m_s),
                         (965.0, 35.0, 50.0))
        self.assertEqual(f.course_deg, 45.0)
        self.assertEqual(f.speed_km_h, 40.0)
        self.assertEqual(f.issue, ISSUE_STAMP)
        self.assertEqual(f.source, "synthetic")
        self.assertAlmostEqual(f.lead_hours, -0.5)
        self.assertTrue(f.is_analysis)

    def test_warning_areas_of_both_shapes(self):
        f = tf.parse_bulletin_text(BULLETIN, ISSUE_STAMP)[0][0]
        self.assertEqual(f.storm, WarningSector(0.0, 130.0, 130.0))
        self.assertEqual(f.gale, WarningSector(45.0, 700.0, 440.0))

    def test_forecast_circle_is_not_an_analysis(self):
        fixes, forecasts = tf.parse_bulletin_text(BULLETIN, ISSUE_STAMP)
        self.assertEqual(len(forecasts), 1)
        p = forecasts[0]
        self.assertIsInstance(p, OfficialForecastPoint)
        self.assertEqual(p.stamp, datetime(2026, 9, 21, 12, 0,
                                           tzinfo=timezone.utc))
        self.assertAlmostEqual(p.lat, 36.5)
        self.assertAlmostEqual(p.lon, 144.0 + 20.0 / 60.0)
        self.assertEqual(p.radius_km, 65.0)

    def test_slow_and_stalled_wording(self):
        slow = ("台風第25号は、20日09時には、北緯30度00分、東経135度00分に"
                "あり、ゆっくりした速さで北へ進んでいます。")
        f = tf.parse_bulletin_text(slow, ISSUE_STAMP)[0][0]
        self.assertEqual(f.speed_km_h, 0.0)
        self.assertEqual(f.course_deg, 0.0)
        stall = ("台風第25号は、20日09時には、北緯30度00分、東経135度00分に"
                 "おいて、中心の気圧は985ヘクトパスカル、"
                 "ほとんど停滞しています。")
        g = tf.parse_bulletin_text(stall, ISSUE_STAMP)[0][0]
        self.assertTrue(math.isnan(g.course_deg))
        self.assertEqual(g.pres_hPa, 985.0)
        self.assertTrue(math.isnan(g.vmax_m_s))

    def test_full_width_digits_and_empty_input(self):
        zen = BULLETIN.translate(str.maketrans("0123456789", "０１２３４５６７８９"))
        fixes, forecasts = tf.parse_bulletin_text(zen, ISSUE_STAMP)
        self.assertEqual(len(fixes), 1)
        self.assertEqual(len(forecasts), 1)
        self.assertEqual(fixes[0].pres_hPa, 965.0)
        for bad in ("", None, "本日快晴なり"):
            self.assertEqual(tf.parse_bulletin_text(bad, ISSUE_STAMP), ([], []))

    def test_parse_jst_resolves_the_day_against_the_issue_time(self):
        self.assertEqual(tf._parse_jst(25, 9, ISSUE_STAMP),
                         datetime(2026, 9, 25, 0, 0, tzinfo=timezone.utc))
        self.assertEqual(tf._parse_jst(21, 9, ISSUE_STAMP),
                         datetime(2026, 9, 21, 0, 0, tzinfo=timezone.utc))
        # a day number far below the issue day belongs to the next month
        self.assertEqual(tf._parse_jst(1, 9, ISSUE_STAMP),
                         datetime(2026, 10, 1, 0, 0, tzinfo=timezone.utc))

    def test_rain_outlook_blocks(self):
        out = tf.parse_rain_outlook(RAIN_SECTION, ISSUE_STAMP)
        self.assertEqual(len(out), 2)
        first, second = out
        self.assertEqual(first.hours, 24)
        self.assertEqual(first.regions, {"伊豆諸島": 300.0, "関東地方": 250.0})
        self.assertEqual(first.max_mm(), 300.0)
        self.assertFalse(first.continued)
        self.assertEqual(first.start,
                         datetime(2026, 9, 21, 0, 0, tzinfo=timezone.utc))
        self.assertEqual(first.end,
                         datetime(2026, 9, 22, 0, 0, tzinfo=timezone.utc))
        self.assertTrue(second.continued)
        self.assertEqual(second.regions, {"東北地方": 120.0})
        self.assertEqual(tf.parse_rain_outlook("", ISSUE_STAMP), ())
        self.assertTrue(math.isnan(tf.RainOutlook(24, None, None,
                                                  {}).max_mm()))

    def test_track_times_places_the_array_on_the_genesis_axis(self):
        vt = tf._as_dt(self.track.stamps[-1])
        times, n3 = tf.track_times(len(self.track), vt, self.snap.genesis)
        self.assertEqual(n3, 35)
        self.assertEqual(times[0], self.snap.genesis)
        self.assertEqual(times[-1], vt)
        hours = np.array([(t - times[0]).total_seconds() / 3600.0
                          for t in times])
        np.testing.assert_allclose(hours, self.track.hours)
        self.assertEqual(tuple(self.track.kinds),
                         ("analysis",) * n3 + ("estimate",) * (len(times) - n3))

    def test_track_times_rejects_an_unplaceable_array(self):
        with self.assertRaises(ValueError):
            tf.track_times(3, datetime(2026, 9, 20, 7, 30,
                                       tzinfo=timezone.utc), None)
        # too few positions for the stated genesis: the array cannot end at
        # the bulletin valid time
        with self.assertRaises(ValueError):
            tf.track_times(5, datetime(2026, 9, 20, 7, 0,
                                       tzinfo=timezone.utc),
                           self.snap.genesis)


class TestSnapshot(TyphoonTestCase):
    def test_track_array_and_time_axis(self):
        self.assertEqual(len(self.track), 40)
        self.assertEqual(self.track.hours[0], 0.0)
        self.assertEqual(self.track.hours[-1], ISSUE_HOUR)
        self.assertEqual(set(np.round(np.diff(self.track.hours), 3).tolist()),
                         {1.0, 3.0})
        self.assertEqual(self.track.kinds.count("analysis"), 35)
        self.assertEqual(self.track.kinds.count("estimate"), 5)
        self.assertEqual(self.track.t0, self.snap.genesis)
        self.assertEqual(self.snap.genesis,
                         datetime(2026, 9, 16, 12, 0, tzinfo=timezone.utc))
        self.assertIn("実況", self.track.source)

    def test_intensity_is_stated_only_where_a_bulletin_states_it(self):
        self.assertEqual(int(np.isfinite(self.track.pres_hPa).sum()), 10)
        self.assertEqual(int(np.isfinite(self.track.vmax_m_s).sum()), 10)
        self.assertEqual(sum(1 for x in self.track.issues if x), 10)
        # state_at interpolates through the stated values alone
        self.assertEqual(state_at(self.track, 0.0)[2:], (998.0, 18.0))
        self.assertEqual(state_at(self.track, ISSUE_HOUR),
                         (32.3, 138.3, 965.0, 35.0))
        mid = state_at(self.track, 105.5)
        self.assertTrue(all(math.isfinite(v) for v in mid))

    def test_bulletin_text_fixes(self):
        self.assertEqual(len(self.snap.fixes), 10)
        self.assertTrue(all(f.is_analysis for f in self.snap.fixes))
        self.assertTrue(all(f.lead_hours < 0 for f in self.snap.fixes))
        text = self.snap.bulletin_track()
        self.assertEqual(len(text), len(self.snap.fixes))
        self.assertEqual(text.t0, self.track.t0)
        self.assertTrue(np.all(np.diff(text.hours) > 0))

    def test_official_forecast_issues(self):
        self.assertEqual(len(self.snap.official), 2)
        sources = [o.source for o in self.snap.official]
        self.assertIn("TC2630_forecastPreviousIssue.json", sources)
        specs = ts.specs_of(self.snap)
        self.assertIsNotNone(specs)
        self.assertEqual(len(specs.points), 9)
        leads = [(p.stamp - self.track.t0).total_seconds() / 3600.0
                 - ISSUE_HOUR for p in specs.points]
        self.assertEqual(leads[0], 1.0)
        self.assertEqual(leads[-1], 43.0)
        last = specs.points[-1]
        self.assertEqual(last.category, "LOW")   # 温帯低気圧
        self.assertEqual((last.pres_hPa, last.vmax_m_s, last.radius_km),
                         (984.0, 25.0, 155.0))
        self.assertIsNone(last.storm)            # no 暴風域 once it is a LOW
        # the earlier issue carries circles only, no intensity
        prev = [o for o in self.snap.official
                if "forecastPreviousIssue" in o.source][0]
        self.assertTrue(all(math.isnan(p.pres_hPa) for p in prev.points))

    def test_warning_areas_are_carried_forward_independently(self):
        self.assertEqual(self.snap.warnings_at(-1.0), (None, None))
        self.assertEqual(len(self.snap.warnings), 10)
        hours = [h for h, _, _ in self.snap.warnings]
        self.assertEqual(hours, sorted(hours))
        self.assertEqual(self.storm, WarningSector(270.0, 185.0, 150.0))
        self.assertEqual(self.gale, WarningSector(45.0, 700.0, 440.0))
        # a bulletin that restates only the 暴風域 does not cancel the 強風域
        storm, gale = self.snap.warnings_at(95.0)
        self.assertEqual(storm, WarningSector(0.0, 130.0, 130.0))
        self.assertEqual(gale, WarningSector(45.0, 700.0, 440.0))

    def test_amedas_and_manifest(self):
        self.assertEqual(len(self.snap.amedas), 72)
        self.assertEqual(sum(len(v) for v in self.snap.amedas.values()), 1008)
        self.assertEqual(len(self.snap.stations), 14)
        stamp = sorted(self.snap.amedas)[-1]
        rec = self.snap.amedas[stamp]["大島"]
        for field in ("precip1h_mm", "precip3h_mm", "precip24h_mm",
                      "wind_m_s", "wind_dir_deg", "temp_C", "humidity_pct"):
            self.assertIn(field, rec)
        meta = [m for m in self.snap.stations.values()
                if m["kjName"] == "大島"][0]
        self.assertEqual(meta["lat"], [34, 44.9])
        self.assertEqual(self.snap.manifest["bulletins"], 133)
        self.assertEqual(self.snap.manifest["amedas_rows"], 1008)
        self.assertEqual(self.snap.manifest["fetched_utc"],
                         "2026-09-21T00:32:36Z")
        self.assertTrue(self.snap.path.endswith("jma_typhoon_2625"))

    def test_gust_ratio_comes_from_the_bulletins(self):
        self.assertAlmostEqual(self.snap.gust_ratio, 50.0 / 35.0)
        self.assertEqual(tf.GUST_RATIO_FALLBACK, 1.43)

    def test_sim_configs_uses_the_published_warning_areas(self):
        got = self.snap.sim_configs(ISSUE_HOUR, 33.0, 139.0)
        want = tf.typhoon_configs(self.track, ISSUE_HOUR, 33.0, 139.0,
                                  storm=self.storm, gale=self.gale,
                                  gust_ratio=self.snap.gust_ratio)
        self.assertEqual(got, want)
        weather, atmo = got
        self.assertGreater(weather.rain_mm_h, 0.0)
        self.assertGreater(math.hypot(*atmo.wind[:2]), 0.0)

    def test_rain_outlooks_of_the_snapshot(self):
        self.assertEqual(len(self.snap.rain_outlooks), 122)
        issue, blocks, head = self.snap.rain_outlooks[0]
        self.assertEqual(len(blocks), 2)
        every = [b.hours for _, bl, _ in self.snap.rain_outlooks for b in bl]
        self.assertEqual(len(every), 358)
        self.assertEqual(set(every), {1, 24})   # 1時間 and 24時間降水量
        self.assertTrue(all(b.regions for _, bl, _ in self.snap.rain_outlooks
                            for b in bl))
        # JMA publishes the storm number in full-width digits (台風第２５号).
        self.assertIn("台風第２５号", head)
        self.assertTrue(all("台風第２５号" in h
                            for _, _, h in self.snap.rain_outlooks))


class TestBestTrack(TyphoonTestCase):
    def test_from_fixes_round_trip(self):
        bt = BestTrack.from_fixes(self.snap.fixes, epoch=self.track.t0)
        self.assertEqual(len(bt), len(self.snap.fixes))
        self.assertEqual(bt.t0, self.track.t0)
        self.assertEqual(bt.hour_of(bt.stamps[-1]), bt.hours[-1])
        self.assertEqual(bt.index_at(bt.hours[3]), 3)
        self.assertTrue(all(k == "analysis" for k in bt.kinds))
        self.assertTrue(all(x for x in bt.issues))
        self.assertEqual(bt.source, "JMA bulletins")
        self.assertEqual(bt.stamp_at(0), bt.stamps[0])

    def test_upto_keeps_only_the_past(self):
        cut = self.track.upto(60.0)
        self.assertEqual(cut.hours[-1], 60.0)
        self.assertTrue(np.all(cut.hours <= 60.0))
        self.assertLess(len(cut), len(self.track))
        self.assertEqual(len(self.track.upto(self.track.stamps[-1])),
                         len(self.track))
        with self.assertRaises(ValueError):
            self.track.upto(-5.0)
        # datetime, ISO string and plain hour are the same cut
        stamp = self.track.stamps[20]
        a = self.track.upto(stamp)
        b = self.track.upto(float(self.track.hours[20]))
        np.testing.assert_array_equal(a.hours, b.hours)

    def test_tail_segment_does_not_bridge_a_gap(self):
        self.assertEqual(len(self.track.tail_segment()), len(self.track))
        keep = np.ones(len(self.track), dtype=bool)
        keep[20:26] = False                   # an 18-hour hole, wider than the
        gapped = self.track._slice(keep)      # gap the segment may bridge
        tail = gapped.tail_segment()
        self.assertEqual(tail.hours[0], gapped.hours[20])
        self.assertEqual(tail.hours[-1], gapped.hours[-1])
        self.assertLess(len(tail), len(gapped))
        # a hole the segment may bridge is not a break
        small = self.track._slice(np.r_[np.ones(20, bool), False,
                                        np.ones(19, bool)])
        self.assertEqual(len(small.tail_segment()), len(small))
        self.assertTrue(np.all(np.diff(tail.hours) <= tf.SEGMENT_GAP_HOURS))

    def test_resample_guards(self):
        with self.assertRaises(ValueError):
            tf._resample(self.track, end_hour=200.0)
        with self.assertRaises(ValueError):
            tf._resample(self.track.upto(2.0), tf.GRID_HOURS)
        g, lat, lon, pres, vmax = tf._resample(self.track, end_hour=ISSUE_HOUR)
        self.assertEqual(g[-1], ISSUE_HOUR)
        np.testing.assert_allclose(np.diff(g), tf.GRID_HOURS)
        self.assertEqual(lat[-1], self.track.lat[-1])
        self.assertEqual(pres[-1], 965.0)
        self.assertEqual(vmax[-1], 35.0)

    def test_interp_known_skips_the_unstated_values(self):
        hours = np.array([0.0, 3.0, 6.0])
        vals = np.array([10.0, np.nan, 20.0])
        self.assertAlmostEqual(tf._interp_known(3.0, hours, vals), 15.0)
        self.assertTrue(math.isnan(
            tf._interp_known(3.0, hours, np.full(3, np.nan))))
        self.assertAlmostEqual(tf._interp_known(9.0, hours, vals), 20.0)

    def test_time_coercion(self):
        self.assertEqual(tf._as_hour(12.5, self.track.t0), 12.5)
        self.assertEqual(tf._as_hour(self.track.stamps[-1], self.track.t0),
                         ISSUE_HOUR)
        self.assertEqual(tf._as_dt("2026-09-20T23:00:00Z"),
                         datetime(2026, 9, 20, 23, 0, tzinfo=timezone.utc))
        self.assertEqual(tf._as_dt("2026-09-21T08:00:00+09:00"),
                         datetime(2026, 9, 20, 23, 0, tzinfo=timezone.utc))
        with self.assertRaises(TypeError):
            tf._as_dt(5)


class TestTrackModel(TyphoonTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.model = TrackForecastModel(cls.track)
        cls.fc = cls.model.predict(72.0)

    def test_lead0_is_the_analysis_exactly(self):
        self.assertEqual(self.fc["issue_hour"], ISSUE_HOUR)
        self.assertEqual(len(self.fc["hours"]), 72 // 3 + 1)
        np.testing.assert_allclose(self.fc["hours"][-1], 72.0)
        self.assertEqual(self.fc["lat"][0], self.track.lat[-1])
        self.assertEqual(self.fc["lon"][0], self.track.lon[-1])
        self.assertEqual(self.fc["pres_hPa"][0], 965.0)
        self.assertEqual(self.fc["vmax_m_s"][0], 35.0)

    def test_fit_and_forecast_are_deterministic(self):
        other = TrackForecastModel(self.track)
        np.testing.assert_array_equal(self.model.coef, other.coef)
        np.testing.assert_array_equal(self.fc["lat"], other.predict(72.0)["lat"])
        np.testing.assert_array_equal(self.fc["pres_hPa"],
                                      other.predict(72.0)["pres_hPa"])

    def test_companion_matrix_does_not_grow(self):
        for order in (1, 2, 3, 4, 6):
            m = TrackForecastModel(self.track, order=order)
            self.assertLessEqual(m.companion_radius(), tf.MAX_RADIUS + 1e-9)
            self.assertLessEqual(m.shrink_scale, 1.0)
            self.assertEqual(m.coef.shape, (2 * order, 2))
        # the ridge prior is motion persistence: each lag block is eye(2)/p
        p = self.model.order
        for b in range(p):
            np.testing.assert_allclose(
                self.model.prior[2 * b:2 * b + 2, :], np.eye(2) / p)

    def test_the_fit_uses_only_the_window_before_the_issue(self):
        self.assertEqual(self.model.window_hours, tf.TRACK_WINDOW_HOURS)
        self.assertTrue(np.all(self.model.used.hours
                               >= ISSUE_HOUR - tf.TRACK_WINDOW_HOURS - 1e-9))
        self.assertEqual(self.model.grid[-1], ISSUE_HOUR)
        self.assertGreaterEqual(len(self.model.motion), tf.MIN_TRAIN_STEPS)
        full = TrackForecastModel(self.track, window_hours=None)
        self.assertIsNone(full.window_hours)
        self.assertEqual(len(full.used), len(self.track))
        self.assertGreater(len(full.motion), len(self.model.motion))

    def test_train_stop_never_sees_the_future(self):
        m = TrackForecastModel(self.track, train_stop=60.0)
        self.assertEqual(m.issue_hour, 60.0)
        self.assertEqual(m.used.hours[-1], 60.0)
        self.assertEqual(m.grid[-1], 60.0)
        with self.assertRaises(ValueError):
            TrackForecastModel(self.track, train_stop=200.0)

    def test_guards(self):
        with self.assertRaises(ValueError):
            TrackForecastModel(self.track, order=0)
        with self.assertRaises(ValueError):
            TrackForecastModel(self.track, train_stop=12.0)   # 4 samples
        with self.assertRaises(ValueError):
            self.model.predict(1.0)                           # < one grid step

    def test_intensity_relaxes_towards_the_extratropical_low(self):
        pres = self.fc["pres_hPa"]
        self.assertTrue(np.all(pres >= tf.PRES_FLOOR))
        self.assertTrue(np.all(pres <= tf.PRES_CEIL))
        self.assertTrue(np.all(np.diff(pres) >= -1e-9))   # filling, not deepening
        self.assertLess(pres[-1], 1000.0)
        self.assertGreater(pres[-1], tf.PRES_RELAX_HPA - 20.0)
        # the maximum wind follows the pressure through the anchored law
        for p, v in zip(pres, self.fc["vmax_m_s"]):
            self.assertAlmostEqual(
                v, tf.vmax_from_pressure(p, pres[0], self.fc["vmax_m_s"][0]),
                delta=1e-9)
        self.assertLess(self.fc["vmax_m_s"][-1], self.fc["vmax_m_s"][0])

    def test_pressure_rate_is_a_recent_trend(self):
        rate = self.model.pressure_rate()
        self.assertTrue(math.isfinite(rate))
        self.assertLess(abs(rate), 2.0)
        empty = np.full(5, np.nan)
        m = TrackForecastModel(self.track)
        m.g_pres = empty
        self.assertEqual(m.pressure_rate(), 0.0)

    def test_vmax_from_pressure_is_anchored(self):
        self.assertAlmostEqual(tf.vmax_from_pressure(965.0, 965.0, 35.0), 35.0)
        self.assertAlmostEqual(tf.vmax_from_pressure(998.0, 998.0, 18.0), 18.0)
        self.assertGreater(tf.vmax_from_pressure(950.0, 965.0, 35.0), 35.0)
        self.assertLess(tf.vmax_from_pressure(990.0, 965.0, 35.0), 35.0)
        # a storm that has filled to ambient carries no wind at all
        self.assertEqual(tf.vmax_from_pressure(1013.0, 965.0, 35.0), 0.0)
        self.assertEqual(tf.vmax_from_pressure(1020.0, 965.0, 35.0), 0.0)

    def test_the_pressure_wind_exponent_is_what_the_bulletins_support(self):
        ok = (np.isfinite(self.track.pres_hPa)
              & np.isfinite(self.track.vmax_m_s))
        deficit = 1013.0 - self.track.pres_hPa[ok]
        vmax = self.track.vmax_m_s[ok]
        self.assertEqual(int(ok.sum()), 10)
        A = np.vstack([np.ones(deficit.size), np.log(deficit)]).T
        fit = np.linalg.lstsq(A, np.log(vmax), rcond=None)[0]
        self.assertAlmostEqual(float(fit[1]), tf.VMAX_PRES_EXP, delta=0.10)
        # anchored on the analysis, the shipped exponent reproduces every
        # (p, vmax) pair the bulletins state for this typhoon
        ref_p = float(self.track.pres_hPa[ok][-1])
        ref_v = float(vmax[-1])
        for p, w in zip(self.track.pres_hPa[ok], vmax):
            self.assertAlmostEqual(tf.vmax_from_pressure(p, ref_p, ref_v),
                                   float(w), delta=2.0)

    def test_extratropical_relaxation_switches_on_with_latitude(self):
        self.assertEqual(tf.et_relax_rate(20.0), 0.0)
        self.assertEqual(tf.et_relax_rate(tf.ET_LAT_START), 0.0)
        self.assertAlmostEqual(tf.et_relax_rate(35.0),
                               tf.PRES_RELAX_RATE / 2)
        self.assertAlmostEqual(tf.et_relax_rate(tf.ET_LAT_END),
                               tf.PRES_RELAX_RATE)
        self.assertAlmostEqual(tf.et_relax_rate(60.0), tf.PRES_RELAX_RATE)

    def test_reference_forecasts(self):
        per = persistence_track(self.track, ISSUE_HOUR, 24.0)
        mot = motion_persistence_track(self.track, ISSUE_HOUR, 24.0)
        for f in (per, mot):
            self.assertEqual(len(f["hours"]), 9)
            self.assertEqual(f["lat"][0], self.track.lat[-1])
            self.assertEqual(f["pres_hPa"][0], 965.0)
            self.assertEqual(f["vmax_m_s"][0], 35.0)
        np.testing.assert_allclose(per["lat"], per["lat"][0])
        np.testing.assert_allclose(per["lon"], per["lon"][0])
        self.assertGreater(abs(mot["lat"][-1] - mot["lat"][0]), 0.1)
        # a window of one step is the last analysed displacement itself
        one = motion_persistence_track(self.track, ISSUE_HOUR, 3.0, window=1)
        d = haversine_km(one["lat"][0], one["lon"][0], one["lat"][-1],
                         one["lon"][-1])
        g, lat, lon, _, _ = tf._resample(self.track, end_hour=ISSUE_HOUR)
        step = haversine_km(lat[-2], lon[-2], lat[-1], lon[-1])
        self.assertAlmostEqual(d, step, delta=1.0)
        self.assertGreater(step / tf.GRID_HOURS, 10.0)      # the storm moves

    def test_track_error_km_skips_leads_the_truth_does_not_cover(self):
        per = persistence_track(self.track, ISSUE_HOUR, 24.0)
        hours, err = track_error_km(per, self.track)
        self.assertEqual(hours.tolist(), [0.0])
        self.assertAlmostEqual(err[0], 0.0, places=9)
        earlier = persistence_track(self.track, 60.0, 24.0)
        hours, err = track_error_km(earlier, self.track)
        self.assertEqual(hours[-1], 24.0)
        self.assertAlmostEqual(err[0], 0.0, places=9)
        self.assertGreater(err[-1], err[0])

    def test_motion_direction_is_the_analysis_bearing(self):
        d = tf.motion_dir_deg(self.track, ISSUE_HOUR)
        self.assertTrue(0.0 <= d < 360.0)
        self.assertAlmostEqual(d, bearing_deg(*state_at(self.track, 104.0)[:2],
                                              *state_at(self.track, 107.0)[:2]),
                               delta=15.0)


class TestHindcastSkill(TyphoonTestCase):
    """Rolling-origin verification: every forecast is fitted only on
    positions at or before its own issue time."""

    ISSUES = tuple(range(21, 108, 3))
    LEADS = (3, 6, 9, 12, 15, 18, 21, 24)

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.err = {"var": {}, "motion": {}, "persistence": {}}
        for issue in cls.ISSUES:
            fc = {"var": TrackForecastModel(cls.track,
                                            train_stop=float(issue)).predict(24),
                  "motion": motion_persistence_track(cls.track, float(issue),
                                                     24.0),
                  "persistence": persistence_track(cls.track, float(issue),
                                                   24.0)}
            for key, f in fc.items():
                hours, err = track_error_km(f, cls.track)
                for lead in cls.LEADS:
                    k = int(np.argmin(np.abs(hours - lead)))
                    if abs(hours[k] - lead) < 1e-9:
                        cls.err[key].setdefault(lead, []).append(err[k])

    def rms(self, key, leads=None):
        vals = [v for lead, vs in self.err[key].items()
                if leads is None or lead in leads for v in vs]
        return float(np.sqrt(np.mean(np.square(vals))))

    def test_the_motion_model_beats_both_references(self):
        var = self.rms("var")
        motion = self.rms("motion")
        pers = self.rms("persistence")
        self.assertLess(var, 0.9 * motion)
        self.assertLess(motion, pers)
        self.assertLess(var, 200.0)
        self.assertGreater(pers, 250.0)

    def test_error_grows_with_lead_time(self):
        for key in self.err:
            short = self.rms(key, leads=(3, 6))
            long_ = self.rms(key, leads=(21, 24))
            self.assertLess(short, long_)
        self.assertLess(self.rms("var", leads=(3,)), 60.0)

    def test_every_lead0_error_is_zero(self):
        for issue in self.ISSUES:
            fc = TrackForecastModel(self.track,
                                    train_stop=float(issue)).predict(3)
            self.assertEqual(fc["lat"][0],
                             self.track.lat[self.track.index_at(float(issue))])
            hours, err = track_error_km(fc, self.track)
            self.assertAlmostEqual(err[0], 0.0, places=6)

    def test_no_issue_uses_positions_after_its_own_time(self):
        for issue in self.ISSUES:
            m = TrackForecastModel(self.track, train_stop=float(issue))
            self.assertEqual(m.issue_hour, float(issue))
            self.assertLessEqual(m.used.hours[-1], float(issue) + 1e-9)
            self.assertLessEqual(m.grid[-1], float(issue) + 1e-9)


class TestRainField(TyphoonTestCase):
    """The parametric rain shield: terms, scaling, orography, asymmetry, cap."""

    def test_rain_terms_are_monotonic_and_bounded(self):
        rates = [0.0, 5.0, 10.0, 40.0, 100.0, 1000.0]
        terms = [rain_terms(r) for r in rates]
        hum = [t["humidity"] for t in terms]
        vis = [t["visibility_m"] for t in terms]
        cld = [t["cloud_cover"] for t in terms]
        lig = [t["lightning_risk"] for t in terms]
        # humidity and cloud rise with rain, visibility falls, lightning rises
        self.assertTrue(all(a <= b for a, b in zip(hum, hum[1:])))
        self.assertTrue(all(a >= b for a, b in zip(vis, vis[1:])))
        self.assertTrue(all(a <= b for a, b in zip(cld, cld[1:])))
        self.assertTrue(all(a <= b for a, b in zip(lig, lig[1:])))
        # every term stays inside its physical bound
        self.assertTrue(all(0.0 <= h <= 1.0 for h in hum))
        self.assertTrue(all(v >= 1500.0 for v in vis))
        self.assertTrue(all(0.0 <= c <= 1.0 for c in cld))
        self.assertTrue(all(0.0 <= g <= 1.0 for g in lig))
        # dry air is the zero-rain limit; lightning saturates by 40 mm/h
        self.assertEqual(terms[0]["humidity"], 0.80)
        self.assertEqual(terms[0]["lightning_risk"], 0.0)
        self.assertAlmostEqual(terms[3]["lightning_risk"], 1.0)

    def test_scale_rain_carries_the_terms_and_is_identity_at_one(self):
        w = WeatherConfig(**rain_terms(10.0))
        self.assertIs(scale_rain(w, 1.0), w)
        w2 = scale_rain(w, 2.0)
        self.assertAlmostEqual(w2.rain_mm_h, 20.0)
        # the sky follows the rain: a 20 mm/h cell carries the 20 mm/h terms
        want = rain_terms(20.0)
        self.assertAlmostEqual(w2.visibility_m, want["visibility_m"])
        self.assertAlmostEqual(w2.humidity, want["humidity"])
        self.assertAlmostEqual(w2.cloud_cover, want["cloud_cover"])
        self.assertAlmostEqual(w2.lightning_risk, want["lightning_risk"])

    def test_orographic_factor_is_unity_over_ocean_and_peaks_at_an_anchor(self):
        self.assertAlmostEqual(orographic_factor(30.0, 150.0), 1.0, delta=1e-6)
        alat, alon, _sigma, gain = tf.OROGRAPHIC_ANCHORS[0]
        peak = orographic_factor(alat, alon)
        # every anchor gain is positive, so the peak is at least this anchor's
        # own gain above unity; the neighbouring anchors' tails add a little.
        self.assertGreaterEqual(peak, 1.0 + gain)
        self.assertAlmostEqual(peak, 1.0 + gain, delta=0.05)
        # the gain decays away from the anchor
        self.assertGreater(peak, orographic_factor(alat + 3.0, alon))

    def test_rain_falls_ahead_of_the_moving_centre(self):
        clat, clon = CENTRE
        mdir = tf.motion_dir_deg(self.track, ISSUE_HOUR)
        rg = sector_radius(self.gale, mdir)
        # a point RAIN_LEAD_FRAC*rg ahead of the centre along the motion sits
        # at the displaced-shield centre, so it is wetter than the mirror
        # point the same distance behind.
        d = tf.RAIN_LEAD_FRAC * rg
        klat = d * math.cos(math.radians(mdir)) / 111.19
        klon = d * math.sin(math.radians(mdir)) / (111.19 * math.cos(math.radians(clat)))
        ra = rain_rate_mm_h(clat, clon, 965.0, mdir, clat + klat, clon + klon,
                            gale=self.gale)
        rb = rain_rate_mm_h(clat, clon, 965.0, mdir, clat - klat, clon - klon,
                            gale=self.gale)
        self.assertGreater(ra, rb)

    def test_amplitude_is_capped_at_rain_max(self):
        clat, clon = CENTRE
        mdir = tf.motion_dir_deg(self.track, ISSUE_HOUR)
        # an open-ocean point so the orographic gain is exactly 1
        plat, plon = 30.0, 150.0
        self.assertAlmostEqual(orographic_factor(plat, plon), 1.0, delta=1e-6)
        # 1013-p = 219.5 hPa saturates the amplitude at RAIN_MAX_MM_H, so a
        # still deeper centre rains no harder, while a weak one rains less.
        r_sat = rain_rate_mm_h(clat, clon, 700.0, mdir, plat, plon, gale=self.gale)
        r_deep = rain_rate_mm_h(clat, clon, 600.0, mdir, plat, plon, gale=self.gale)
        r_weak = rain_rate_mm_h(clat, clon, 900.0, mdir, plat, plon, gale=self.gale)
        self.assertAlmostEqual(r_sat, r_deep, places=6)
        self.assertLess(r_weak, r_sat)
        self.assertLessEqual(r_sat, tf.RAIN_MAX_MM_H)


class TestVortexWind(TyphoonTestCase):
    """The Rankine-style vortex: cyclonic inflow, radial profile, gusts."""

    def test_wind_is_zero_at_the_centre(self):
        clat, clon = CENTRE
        self.assertEqual(
            vortex_wind(clat, clon, 35.0, 965.0, clat, clon,
                        storm=self.storm, gale=self.gale), (0.0, 0.0))

    def test_flow_is_cyclonic_with_inflow(self):
        clat, clon = CENTRE
        # Northern-Hemisphere cyclone: a point north of the centre gets wind
        # FROM the ENE (tangential from-east turned inward), a point east gets
        # wind FROM the SSE.
        sp_n, from_n = vortex_wind(clat, clon, 35.0, 965.0, clat + 0.5, clon,
                                   storm=self.storm, gale=self.gale)
        sp_e, from_e = vortex_wind(clat, clon, 35.0, 965.0, clat, clon + 0.5,
                                   storm=self.storm, gale=self.gale)
        self.assertGreater(sp_n, 0.0)
        self.assertGreater(sp_e, 0.0)
        self.assertTrue(0.0 < from_n < 90.0, from_n)     # ENE quadrant
        self.assertTrue(90.0 < from_e < 180.0, from_e)   # SSE quadrant
        # with no inflow the FROM-direction is the pure tangential (due east
        # north of the centre); inflow turns it towards the centre.
        _, from0 = vortex_wind(clat, clon, 35.0, 965.0, clat + 0.5, clon,
                               storm=self.storm, gale=self.gale, inflow_deg=0.0)
        self.assertAlmostEqual(from0, 90.0, delta=1e-6)
        self.assertLess(from_n, from0)

    def test_speed_decreases_outside_the_core(self):
        clat, clon = CENTRE
        # both points lie well outside the ~90 km core, on the falling branch
        near = vortex_wind(clat, clon, 35.0, 965.0, clat + 2.0, clon,
                           storm=self.storm, gale=self.gale)[0]
        far = vortex_wind(clat, clon, 35.0, 965.0, clat + 4.0, clon,
                          storm=self.storm, gale=self.gale)[0]
        self.assertGreater(near, far)

    def test_wind_profile_is_over_determined_by_both_radii(self):
        # 25 m/s at the storm edge and 15 m/s at the gale edge fix the
        # exponent; the storm radius then fixes the core radius.
        rmax, b = wind_profile(35.0, 185.0, 545.0, 965.0)
        self.assertAlmostEqual(
            b, math.log(25.0 / 15.0) / math.log(545.0 / 185.0), places=6)
        self.assertAlmostEqual(rmax, 185.0 * (25.0 / 35.0) ** (1.0 / b),
                               places=6)
        # the analysis mean-radius profile is the docstring's b~0.47, rmax~91
        self.assertAlmostEqual(b, 0.47, delta=0.01)
        self.assertAlmostEqual(rmax, 91.0, delta=1.0)

    def test_wind_profile_clamps_the_exponent(self):
        _, b_low = wind_profile(35.0, 100.0, 5000.0, 965.0)
        self.assertEqual(b_low, tf.WIND_POWER_RANGE[0])
        _, b_high = wind_profile(35.0, 185.0, 200.0, 965.0)
        self.assertEqual(b_high, tf.WIND_POWER_RANGE[1])

    def test_wind_profile_falls_back_without_radii(self):
        # gale only: the exponent is the fallback, the core follows the gale
        rmax_g, b_g = wind_profile(35.0, None, 650.0, 965.0)
        self.assertEqual(b_g, tf.WIND_POWER_FALLBACK)
        self.assertAlmostEqual(rmax_g, 650.0 * (15.0 / 35.0) ** (1.0 / b_g),
                               places=6)
        # neither radius: a pressure-based core and the fallback exponent
        rmax_p, b_p = wind_profile(35.0, None, None, 965.0)
        self.assertEqual(b_p, tf.WIND_POWER_FALLBACK)
        self.assertAlmostEqual(
            rmax_p, tf.RMAX_FALLBACK_KM + tf.RMAX_PER_HPA * (1010.0 - 965.0),
            places=6)

    def test_gust_ratio_scales_the_gust_rms(self):
        clat, clon = CENTRE
        mdir = tf.motion_dir_deg(self.track, ISSUE_HOUR)
        plat, plon = 30.0, 150.0
        ratio = 50.0 / 35.0
        _, a = tf.configs_at(clat, clon, 965.0, 35.0, mdir, plat, plon,
                             storm=self.storm, gale=self.gale, gust_ratio=ratio)
        speed = math.hypot(a.wind[0], a.wind[1])
        self.assertAlmostEqual(
            a.gust_rms, tf.GUST_SIGMA_FACTOR * (ratio - 1.0) * speed, places=9)
        # a ratio of 1 leaves no gust variance
        _, a1 = tf.configs_at(clat, clon, 965.0, 35.0, mdir, plat, plon,
                              storm=self.storm, gale=self.gale, gust_ratio=1.0)
        self.assertAlmostEqual(a1.gust_rms, 0.0, places=9)


class TestSimulatorConfigs(TyphoonTestCase):
    """configs_at / typhoon_configs build simulator-ready fields."""

    def test_pressure_and_temperature_offsets(self):
        clat, clon = CENTRE
        mdir = tf.motion_dir_deg(self.track, ISSUE_HOUR)
        w, _ = tf.configs_at(clat, clon, 965.0, 35.0, mdir, 30.0, 150.0,
                             storm=self.storm, gale=self.gale)
        self.assertAlmostEqual(w.pressure_offset_Pa, (965.0 - 1013.25) * 100.0)
        self.assertEqual(w.temp_offset_K, 1.5)
        self.assertEqual(w.cloud_base_m, 600.0)
        self.assertEqual(w.cloud_top_m, 8000.0)

    def test_rain_terms_match_the_rain_rate(self):
        clat, clon = CENTRE
        mdir = tf.motion_dir_deg(self.track, ISSUE_HOUR)
        plat, plon = 31.0, 139.0
        w, _ = tf.configs_at(clat, clon, 965.0, 35.0, mdir, plat, plon,
                             storm=self.storm, gale=self.gale)
        rain = rain_rate_mm_h(clat, clon, 965.0, mdir, plat, plon,
                              gale=self.gale)
        want = rain_terms(rain)
        self.assertAlmostEqual(w.rain_mm_h, want["rain_mm_h"], places=9)
        self.assertAlmostEqual(w.humidity, want["humidity"], places=9)
        self.assertAlmostEqual(w.visibility_m, want["visibility_m"], places=6)
        self.assertAlmostEqual(w.cloud_cover, want["cloud_cover"], places=9)
        self.assertAlmostEqual(w.lightning_risk, want["lightning_risk"], places=9)

    def test_typhoon_configs_matches_configs_at_the_analysis(self):
        plat, plon = 31.0, 139.0
        got = tf.typhoon_configs(self.track, ISSUE_HOUR, plat, plon,
                                 storm=self.storm, gale=self.gale,
                                 gust_ratio=self.snap.gust_ratio)
        clat, clo, pres, vmax = state_at(self.track, ISSUE_HOUR)
        want = tf.configs_at(clat, clo, pres, vmax,
                             tf.motion_dir_deg(self.track, ISSUE_HOUR),
                             plat, plon, storm=self.storm, gale=self.gale,
                             gust_ratio=self.snap.gust_ratio)
        self.assertEqual(got, want)

    def test_building_configs_leaves_aircraft_defaults_untouched(self):
        def defaults():
            a = Aircraft()
            return (a.V_cruise, a.V_stall, a.W, a.geom.b, a.geom.S,
                    a.aero.CL_max, a.prop.P_max)
        before = defaults()
        # build a whole field of typhoon configs across time and space
        for h in (0.0, 50.0, ISSUE_HOUR):
            for la, lo in ((31.0, 139.0), (33.0, 140.0), CENTRE):
                tf.typhoon_configs(self.track, h, la, lo,
                                   storm=self.storm, gale=self.gale)
        self.assertEqual(before, defaults())


class TestStudySmoke(TyphoonTestCase):
    """A quick, plot-free run of the study writes every product."""

    def test_quick_run_writes_report_summary_and_csvs(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            ts.run_study(ts.StudyConfig(out=out, data=Path(self.snap.path),
                                        quick=True, plots=False))
            self.assertTrue((out / "REPORT.md").exists())
            self.assertTrue((out / "summary.json").exists())
            self.assertTrue((out / "typhoon_forecast_study.py").exists())
            for name in ts.CSV_FILES:
                self.assertTrue((out / name).exists(), name)
            summary = json.loads((out / "summary.json").read_text("utf-8"))
            self.assertIn("files", summary)
            self.assertEqual(summary["config"]["quick"], True)
