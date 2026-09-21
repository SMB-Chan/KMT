#!/usr/bin/env python3
"""Real-operation forecast of Typhoon No. 25 (2026, Dujuan) from JMA bulletins.

Runs `typhoon_forecast` the way an operator would.  The snapshot in
``data/jma_typhoon_2625/`` holds every machine-readable Japan Meteorological
Agency product of this storm that ``fetch_typhoon_dujuan.py`` could reach:
133 bulletins (気象解説情報 / 台風発生報), the 実況 and 諸元 position arrays,
the official forecast circles, the hourly AMeDAS observations of 14 stations
along the track, and the latest 府県天気予報 (R1) XML of the nine affected
prefectures.  Nothing is fetched at run time, so the study is reproducible
from the committed files.

  A. snapshot       -- provenance, and the consistency of the positions the
                       bulletin text states with the analysed track array;
  B. track forecast -- the forecast issued at the last analysis
                       (2026-09-20T23:00Z, h = 107) out to +72 h, beside the
                       official JMA forecast of the same issue (諸元) and its
                       probability circles;
  C. hindcast       -- rolling-origin verification of the motion model against
                       the analysed track; every forecast is fitted only on
                       positions *before* its issue time.  References are
                       persistence and motion persistence, and the VAR order
                       is selected on the early half of the issues and checked
                       on the late half;
  D. intensity      -- central pressure and maximum wind: hindcast RMSE and
                       forecast against the official 諸元 values;
  E. precipitation  -- E1 the parametric rain field against the AMeDAS
                       station-hours of this typhoon; E2 the 「多い所で」 peaks
                       JMA publishes in the bulletins; E3 the official
                       6-hourly 降水確率 (府県天気予報 R1) over the same
                       windows as the model rain; E4 forecast totals at the
                       operating sites;
  F. operations     -- go/no-go timeline per site, with limits taken from the
                       repository's own weather presets and a thrust-margin
                       check from the airframe model;
  G. bulletins      -- what the parsed warning areas and rain outlooks say.

The forecast window (2026-09-21T00:00Z onwards) still lies in the future at
the snapshot time, so B, D (forecast part), E3 and F are guidance-versus-
guidance comparisons against JMA's own products; the verification against
observed truth is C (track) and E1 (rain).

Usage:
    python3 typhoon_forecast_study.py [--out results/typhoon_25_001]
                                      [--quick] [--no-plots] [--data DIR]
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import math
import os
import shutil
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

import typhoon_forecast as tf
from aircraft import Aircraft
from design_optimize import git_head
from typhoon_forecast import (TrackForecastModel, gale_sector_from_storm,
                              haversine_km, load_snapshot,
                              motion_persistence_track, persistence_track,
                              rain_rate_mm_h, sector_radius, state_at,
                              track_error_km)
from weather import Weather, get_preset
from weather_real import GUST_SIGMA_FACTOR
from wind_tunnel import trim_state

V_CRUISE = 11.3          # m/s, the reference cruise speed of the other studies
PATROL_ALT = 30.0        # m, patrol altitude
JST = timezone(timedelta(hours=9))
GUST_PEAK_FACTOR = 1.0 / GUST_SIGMA_FACTOR   # 1-h peak of the sim gust process
WET_MM_H = 0.1           # JMA counts precipitation from 0.1 mm/h
R1_DIRNAME = "feed_prefecture_R1"
BODY_NS = "{http://xml.kishou.go.jp/jmaxml1/body/meteorology1/}"
ELEM_NS = "{http://xml.kishou.go.jp/jmaxml1/elementBasis1/}"

# AMeDAS station -> 府県天気予報 sub-area code of the R1 XML.  甲府 (Yamanashi)
# has no R1 file in the snapshot, so it stays unmapped.
STATION_AREA = {
    "八丈島": "130030", "三宅島": "130020", "大島": "130020",
    "東京": "130010", "横浜": "140010", "千葉": "120010",
    "銚子": "120020", "甲府": None, "静岡": "220010",
    "水戸": "080020", "仙台": "040010", "宮古": "030020",
    "高知": "390010", "大阪": "270000",
}

# The R1 sub-areas that hold no AMeDAS station get the study's representative
# point (the principal town, ~0.1 deg).  The rain shield the model produces is
# smooth on the ~100 km scale of the 強風域, so where inside a district the
# point sits matters little; areas that do have a station take its own
# coordinates from the snapshot.
AREA_TOWN = {
    "120030": (34.99, 139.87),   # 千葉南部・館山
    "080010": (36.70, 140.72),   # 茨城北部・高萩
    "030010": (39.70, 141.15),   # 岩手内陸・盛岡
    "030030": (39.08, 141.72),   # 岩手沿岸南部・大船渡
    "140020": (35.25, 139.15),   # 神奈川西部・小田原
    "390020": (33.50, 134.10),   # 高知東部・安芸
    "390030": (32.98, 132.93),   # 高知西部・中村
    "040020": (38.57, 140.97),   # 宮城西部・古川
    "220020": (35.12, 138.92),   # 静岡伊豆・三島
    "220030": (35.16, 138.67),   # 静岡東部・富士
    "220040": (34.71, 137.73),   # 静岡西部・浜松
    "130040": (27.09, 142.19),   # 小笠原諸島・父島
}
DISTRICT_HALF_DEG = 0.25   # 「多い所で」 is a district maximum: search box
DISTRICT_GRID_N = 5
DEG_KM = 111.32            # km per degree of latitude, as elsewhere here

# The repository defines no operational weather limits, so this study takes
# them from its own weather presets: 'rain' is the most severe preset that is
# not 'storm', which makes it the go/no-go boundary; 'storm' is what the
# airframe is only flown for in the damage studies.
_RAIN = get_preset("rain")
OPS_WIND_MAX_M_S = math.hypot(*_RAIN.atmosphere.wind[:2])
OPS_GUST_MAX_M_S = (OPS_WIND_MAX_M_S
                    + GUST_PEAK_FACTOR * _RAIN.atmosphere.gust_rms)
OPS_RAIN_MAX_MM_H = _RAIN.weather.rain_mm_h
OPS_VIS_MIN_M = _RAIN.weather.visibility_m
OPS_LIGHTNING_MAX = 0.5    # Weather.condition_label calls >= 0.5 'storm'
OPS_MARGIN_MIN = 1.15      # thrust / drag at cruise, defined by this study
OPS_LIMITS = {
    "wind_m_s": OPS_WIND_MAX_M_S,
    "gust_m_s": OPS_GUST_MAX_M_S,
    "rain_mm_h": OPS_RAIN_MAX_MM_H,
    "visibility_m": OPS_VIS_MIN_M,
    "lightning_risk": OPS_LIGHTNING_MAX,
    "thrust_margin": OPS_MARGIN_MIN,
}
DIR_JA = {v: k for k, v in tf._COURSE.items()}


@dataclass
class StudyConfig:
    out: Path = Path("results/typhoon_25_001")
    data: Path = Path(tf.DATA_DIR)
    horizon: float = 72.0        # operational forecast length, h
    hind_first: float = 21.0     # first rolling-origin issue, h since genesis
    hind_stride: float = 3.0     # issue spacing of the hindcast, h
    hind_horizon: float = 24.0   # verified lead length of the hindcast, h
    leads: tuple = (3, 6, 9, 12, 15, 18, 21, 24)
    orders: tuple = (1, 2, 3, 4)
    ops_stride: float = 3.0      # go/no-go sampling, h
    plots: bool = True
    quick: bool = False

    def __post_init__(self):
        self.out = Path(self.out)
        self.data = Path(self.data)
        if self.quick:
            self.orders = (2, 3)
            self.hind_stride = 6.0
            self.horizon = 48.0


def write_csv(path: Path, rows: list, cols: list) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})


def iso(dt) -> str:
    return tf._as_dt(dt).strftime("%Y-%m-%dT%H:%M:%SZ")


def jst(dt) -> str:
    return tf._as_dt(dt).astimezone(JST).strftime("%m/%d %H:%M")


def rms(values) -> float:
    a = np.asarray([v for v in values if v is not None
                    and np.isfinite(v)], dtype=float)
    return float(np.sqrt(np.mean(a ** 2))) if a.size else float("nan")


def mean(values) -> float:
    a = np.asarray([v for v in values if v is not None
                    and np.isfinite(v)], dtype=float)
    return float(np.mean(a)) if a.size else float("nan")


def sector_ja(sector) -> str:
    """Japanese description of a warning area, as the bulletins phrase it."""
    if sector is None:
        return "―"
    ra, rb = sector.radius_km, sector.opposite_km
    if not math.isfinite(ra):
        return "―"
    if not math.isfinite(rb) or abs(rb - ra) < 1e-9:
        return f"半径{ra:.0f}km"
    return (f"{DIR_JA.get(sector.dir_deg, f'{sector.dir_deg:.0f}deg')}側"
            f"{ra:.0f}km/{DIR_JA.get((sector.dir_deg + 180.0) % 360.0, '?')}側"
            f"{rb:.0f}km")


def station_latlon(snap) -> dict:
    """{漢字名: (lat, lon)} of the AMeDAS stations of the snapshot."""
    out = {}
    for meta in snap.stations.values():
        la = meta["lat"][0] + meta["lat"][1] / 60.0
        lo = meta["lon"][0] + meta["lon"][1] / 60.0
        out[meta["kjName"]] = (float(la), float(lo))
    return out


def specs_of(snap):
    """The official forecast of the latest 諸元 issue (intensity included)."""
    cand = [o for o in snap.official if "specifications" in (o.source or "")]
    return cand[-1] if cand else None


def parse_pop(data_dir):
    """府県天気予報 (R1) 降水確率 of the snapshot, as flat records.

    Each record is {pref, area, code, start (UTC), hours, pop}; `pref` is the
    prefecture name the XML head carries, which is also the key the bulletin
    rain outlooks use, so the two official products can be joined on it.
    """
    d = Path(data_dir) / R1_DIRNAME
    head_ns = "{http://xml.kishou.go.jp/jmaxml1/informationBasis1/}"
    rows = []
    if not d.is_dir():
        return rows
    for path in sorted(d.glob("*.xml")):
        root = ET.parse(path).getroot()
        head = root.find(head_ns + "Head")
        title = head.findtext(head_ns + "Title") or path.stem
        pref = title.replace("府県天気予報", "").strip()
        for tsi in root.iter(BODY_NS + "TimeSeriesInfo"):
            types = [k.findtext(BODY_NS + "Property/" + BODY_NS + "Type")
                     for k in tsi.findall(BODY_NS + "Item/" + BODY_NS + "Kind")]
            if "降水確率" not in types:
                continue
            windows = {}
            for td in tsi.findall(BODY_NS + "TimeDefines/" + BODY_NS
                                  + "TimeDefine"):
                dur = (td.findtext(BODY_NS + "Duration") or "PT6H")
                hours = float(dur.replace("PT", "").replace("H", "") or 6)
                start = datetime.fromisoformat(
                    td.findtext(BODY_NS + "DateTime")).astimezone(timezone.utc)
                windows[td.get("timeId")] = (start, hours)
            for item in tsi.findall(BODY_NS + "Item"):
                if item.findtext(BODY_NS + "Kind/" + BODY_NS + "Property/"
                                 + BODY_NS + "Type") != "降水確率":
                    continue
                code = item.findtext(BODY_NS + "Area/" + BODY_NS + "Code")
                area = item.findtext(BODY_NS + "Area/" + BODY_NS + "Name")
                for pp in item.iter(ELEM_NS + "ProbabilityOfPrecipitation"):
                    win = windows.get(pp.get("refID"))
                    if win is None or pp.text is None:
                        continue
                    rows.append(dict(pref=pref, area=area, code=code,
                                     start=win[0], hours=win[1],
                                     pop=float(pp.text)))
            break
    return rows


# ----------------------------------------------------------------------
#  the operational forecast
# ----------------------------------------------------------------------
class OperationalForecast:
    """What an operator had at the last analysed position of the bulletins.

    The motion model is fitted on the analysed track up to (and including) the
    last 実況, so nothing after the issue time enters it.  JMA states a 暴風域
    for every official forecast position but stops stating a 強風域 after the
    first hours, so the warning geometry of a lead time is the official 暴風域
    of the nearest preceding forecast point (held beyond the last one that
    states one, and dropped where a later point states none, which is flagged)
    and the 強風域 the fitted wind profile implies.

    An instance built with `official=True` reports the operational product
    track instead: JMA's own forecast positions up to the last one JMA
    publishes, and past it the model's motion spliced onto that anchor.  The
    hindcast (C) puts the free-running model outside JMA's probability circle
    from about +10 h, so a product meant to be flown keeps JMA's skill where
    JMA has it and uses the model only for the leads nobody forecasts.
    """

    def __init__(self, snap, horizon_hours, order=None, official=False):
        self.snap = snap
        self.track = snap.track
        self.issue_hour = float(self.track.hours[-1])
        self.issue_stamp = tf._as_dt(self.track.stamps[-1])
        self.horizon = float(horizon_hours)
        self.model = TrackForecastModel(
            self.track, train_stop=self.track.stamps[-1],
            order=tf.TRACK_ORDER if order is None else order)
        self.fc = self.model.predict(self.horizon)
        self.specs = specs_of(snap)
        self.spec_leads = []
        if self.specs is not None:
            for p in self.specs.points:
                self.spec_leads.append(
                    ((p.stamp - self.track.t0).total_seconds() / 3600.0
                     - self.issue_hour, p))
        self.spec_leads.sort(key=lambda t: t[0])
        analysis_storm, analysis_gale = snap.warnings_at(self.issue_hour)
        self.analysis_storm = analysis_storm
        self.analysis_gale = analysis_gale
        self.stations = station_latlon(snap)
        self.official = bool(official)
        self.knots = self._knots()
        self.anchor_lead = (self.knots["lead"][-1] if self.knots["lead"]
                            else 0.0)
        self.product = self._product_track() if self.official else None

    # -- official knots and the product track ---------------------------
    def _knots(self) -> dict:
        """JMA's published forecast as interpolation knots, lead 0 = 実況."""
        tr = self.track
        k = {"lead": [0.0], "lat": [float(tr.lat[-1])],
             "lon": [float(tr.lon[-1])], "pres": [float(tr.pres_hPa[-1])],
             "vmax": [float(tr.vmax_m_s[-1])], "radius": [0.0]}
        for lead, p in self.spec_leads:
            k["lead"].append(float(lead))
            k["lat"].append(float(p.lat))
            k["lon"].append(float(p.lon))
            k["pres"].append(float(p.pres_hPa))
            k["vmax"].append(float(p.vmax_m_s))
            k["radius"].append(float(p.radius_km))
        return k

    def product_radius_km(self, lead) -> float:
        """Forecast-circle radius [km]: JMA's where published, then grown at
        the rate of JMA's last published segment (an extrapolation)."""
        k = self.knots
        fin = [(L, r) for L, r in zip(k["lead"], k["radius"])
               if math.isfinite(r)]
        if len(fin) < 2:
            return float("nan")
        (l1, r1), (l2, r2) = fin[-2], fin[-1]
        if lead <= l2:
            return float(np.interp(lead, [a for a, _ in fin],
                                   [b for _, b in fin]))
        return r2 + (r2 - r1) / (l2 - l1) * (lead - l2)

    def _product_track(self) -> dict:
        """The official-anchored product track on the model's own grid."""
        k = self.knots
        anchor = self.anchor_lead
        ala = float(np.interp(anchor, k["lead"], k["lat"]))
        alo = float(np.interp(anchor, k["lead"], k["lon"]))
        hours = [float(h) for h in
                 np.arange(0.0, self.horizon + 1e-9, tf.GRID_HOURS)]
        if hours[-1] < self.horizon - 1e-9:
            hours.append(self.horizon)
        out = {"hours": hours, "lat": [], "lon": [], "pres_hPa": [],
               "vmax_m_s": [], "radius_km": [], "src": []}
        mla0, mlo0, _, _, _ = self.model_state(anchor)
        for h in hours:
            if h <= anchor + 1e-9:
                la = float(np.interp(h, k["lead"], k["lat"]))
                lo = float(np.interp(h, k["lead"], k["lon"]))
                pr = float(np.interp(h, k["lead"], k["pres"]))
                vm = float(np.interp(h, k["lead"], k["vmax"]))
                src = "公式"
            else:
                mla, mlo, pr, vm, _ = self.model_state(h)
                mid = math.radians(0.5 * (mla0 + mla))
                dn = (mla - mla0) * DEG_KM
                de = (mlo - mlo0) * DEG_KM * math.cos(mid)
                la = ala + dn / DEG_KM
                lo = alo + de / (DEG_KM * math.cos(math.radians(la)))
                src = "モデル延長"
            out["lat"].append(la)
            out["lon"].append(lo)
            out["pres_hPa"].append(pr)
            out["vmax_m_s"].append(vm)
            out["radius_km"].append(self.product_radius_km(h))
            out["src"].append(src)
        return out

    # -- forecast state -------------------------------------------------
    def model_state(self, lead):
        """(lat, lon, pres, vmax, motion_dir) of the free-running model."""
        f = self.fc
        la = float(np.interp(lead, f["hours"], f["lat"]))
        lo = float(np.interp(lead, f["hours"], f["lon"]))
        pr = float(np.interp(lead, f["hours"], f["pres_hPa"]))
        vm = float(np.interp(lead, f["hours"], f["vmax_m_s"]))
        k = int(np.argmin(np.abs(np.asarray(f["hours"]) - lead)))
        k2 = min(k + 1, len(f["hours"]) - 1)
        k1 = max(k - 1, 0)
        md = tf.bearing_deg(f["lat"][k1], f["lon"][k1], f["lat"][k2], f["lon"][k2])
        return la, lo, pr, vm, md

    def state(self, lead):
        """(lat, lon, pres, vmax, motion_dir) of the forecast at a lead [h].

        The product track when this instance was built with `official=True`,
        the free-running model otherwise.
        """
        f = self.product if self.product is not None else self.fc
        la = float(np.interp(lead, f["hours"], f["lat"]))
        lo = float(np.interp(lead, f["hours"], f["lon"]))
        pr = float(np.interp(lead, f["hours"], f["pres_hPa"]))
        vm = float(np.interp(lead, f["hours"], f["vmax_m_s"]))
        k = int(np.argmin(np.abs(np.asarray(f["hours"], dtype=float) - lead)))
        k2 = min(k + 1, len(f["hours"]) - 1)
        k1 = max(k - 1, 0)
        md = tf.bearing_deg(f["lat"][k1], f["lon"][k1], f["lat"][k2], f["lon"][k2])
        return la, lo, pr, vm, md

    def product_src(self, lead) -> str:
        """'公式' or 'モデル延長': which side of the anchor a lead is on."""
        if self.product is None:
            return "モデル"
        p = self.product
        k = int(np.argmin(np.abs(np.asarray(p["hours"]) - float(lead))))
        return p["src"][k]

    def stamp(self, lead):
        return self.issue_stamp + timedelta(hours=float(lead))

    def speed_km_h(self, lead):
        """Ground speed [km/h] over one grid step at `lead`.

        One-sided at the end of the grid: `state` clamps past the last point,
        so a forward difference there would read as a standstill.
        """
        f = self.product if self.product is not None else self.fc
        step = tf.GRID_HOURS
        lead = float(lead)
        a, b = ((max(0.0, lead - step), lead)
                if lead + step > f["hours"][-1] + 1e-9
                else (lead, lead + step))
        if b - a < 1e-9:
            return 0.0
        la0, lo0, _, _, _ = self.state(a)
        la1, lo1, _, _, _ = self.state(b)
        return haversine_km(la0, lo0, la1, lo1) / (b - a)

    # -- warning geometry ----------------------------------------------
    def storm_sector(self, lead):
        """(sector, source) of the 暴風域 JMA published for that lead.

        A forecast point that states no 暴風域 ends the geometry: JMA drops it
        when it declares the storm 温帯低気圧, so the sector of an earlier
        point must not be held past it.
        """
        if lead <= (self.spec_leads[0][0] if self.spec_leads else 0.0) + 1e-9:
            return self.analysis_storm, "現況"
        best = None
        for L, p in self.spec_leads:
            if L <= lead + 1e-9:
                best = (L, p)
        if best is None:
            return self.analysis_storm, "現況(延長)"
        if best[1].storm is None:
            return None, f"公式+{best[0]:.0f}h(暴風域なし)"
        last = max(L for L, p in self.spec_leads if p.storm is not None)
        src = f"公式+{best[0]:.0f}h"
        if lead > last + 1e-9:
            src += "(延長)"
        return best[1].storm, src

    def gale_sector(self, lead):
        """強風域 implied by the 暴風域 and the fitted radial wind profile."""
        storm, _ = self.storm_sector(lead)
        _, _, pres, vmax, _ = self.state(lead)
        if storm is None:
            return None
        return gale_sector_from_storm(storm, vmax, pres)

    def configs(self, lead, lat, lon):
        """Simulator (WeatherConfig, AtmosphereConfig) at a lead and a point."""
        la, lo, pr, vm, md = self.state(lead)
        storm, _ = self.storm_sector(lead)
        return tf.configs_at(la, lo, pr, vm, md, lat, lon, storm=storm,
                             gale=self.gale_sector(lead),
                             gust_ratio=self.snap.gust_ratio)

    def rain(self, lead, lat, lon):
        la, lo, pr, vm, md = self.state(lead)
        return rain_rate_mm_h(la, lo, pr, md, lat, lon,
                              gale=self.gale_sector(lead))

    def wind(self, lead, lat, lon):
        from typhoon_forecast import vortex_wind
        la, lo, pr, vm, _ = self.state(lead)
        return vortex_wind(la, lo, vm, pr, lat, lon,
                           storm=self.storm_sector(lead)[0],
                           gale=self.gale_sector(lead))

    def official_at(self, lead, tol=0.51):
        """Official 諸元 point nearest to `lead`, or None."""
        best = None
        for L, p in self.spec_leads:
            if abs(L - lead) <= tol and (best is None or abs(L - lead)
                                         < abs(best[0] - lead)):
                best = (L, p)
        return best


# ----------------------------------------------------------------------
#  A. snapshot provenance
# ----------------------------------------------------------------------
def section_snapshot(snap, cfg: StudyConfig) -> dict:
    tr = snap.track
    man = dict(snap.manifest or {})
    issues = sorted([tf._as_dt(f.issue) for f in snap.fixes
                     if f.issue is not None]
                    + [tf._as_dt(i) for i, _, _ in snap.rain_outlooks])
    rows = []
    for f in snap.fixes:
        h = tr.hour_of(f.stamp)
        if h < tr.hours[0] - 1e-9 or h > tr.hours[-1] + 1e-9:
            continue
        la, lo, pr, vm = state_at(tr, h)
        rows.append(dict(stamp_utc=iso(f.stamp), stamp_jst=jst(f.stamp),
                         issue_utc=iso(f.issue) if f.issue else "",
                         lead_h=round(f.lead_hours, 2),
                         kind="analysis" if f.is_analysis else "forecast",
                         text_lat=f.lat, text_lon=f.lon,
                         track_lat=round(la, 4), track_lon=round(lo, 4),
                         dist_km=round(haversine_km(f.lat, f.lon, la, lo), 2),
                         pres_hPa=f.pres_hPa, vmax_m_s=f.vmax_m_s,
                         gust_m_s=f.gust_m_s, course_deg=f.course_deg,
                         speed_km_h=f.speed_km_h,
                         storm=sector_ja(f.storm), gale=sector_ja(f.gale),
                         source=f.source))
    kinds = {k: int(sum(1 for x in tr.kinds if x == k))
             for k in sorted(set(tr.kinds))}
    stamps = sorted(snap.amedas)
    n_obs = sum(len(v) for v in snap.amedas.values())
    out = {
        "manifest": man,
        "bulletins": int(man.get("bulletins", 0)),
        "denbun_files": int(man.get("denbun_files", 0)),
        "fetched_utc": man.get("fetched_utc", ""),
        "issue_first_utc": iso(issues[0]) if issues else "",
        "issue_last_utc": iso(issues[-1]) if issues else "",
        "genesis_utc": iso(snap.genesis) if snap.genesis else "",
        "genesis_jst": jst(snap.genesis) if snap.genesis else "",
        "track_n": len(tr),
        "track_first_utc": iso(tr.stamps[0]),
        "track_last_utc": iso(tr.stamps[-1]),
        "track_last_jst": jst(tr.stamps[-1]),
        "track_hours": float(tr.hours[-1]),
        "track_kinds": kinds,
        "track_speed_km_h": float(sum(
            haversine_km(tr.lat[i], tr.lon[i], tr.lat[i + 1], tr.lon[i + 1])
            for i in range(len(tr) - 1)) / (tr.hours[-1] - tr.hours[0])),
        "track_dist_km": float(sum(
            haversine_km(tr.lat[i], tr.lon[i], tr.lat[i + 1], tr.lon[i + 1])
            for i in range(len(tr) - 1))),
        "analysis_state": {
            "lat": float(tr.lat[-1]), "lon": float(tr.lon[-1]),
            "pres_hPa": float(tr.pres_hPa[-1]),
            "vmax_m_s": float(tr.vmax_m_s[-1]),
            "gust_m_s": float(tr.vmax_m_s[-1] * snap.gust_ratio),
            "storm": sector_ja(snap.warnings_at(tr.hours[-1])[0]),
            "gale": sector_ja(snap.warnings_at(tr.hours[-1])[1]),
            "min_pres_hPa": float(np.nanmin(tr.pres_hPa)),
            "max_vmax_m_s": float(np.nanmax(tr.vmax_m_s)),
        },
        "text_vs_track": {
            "n": len(rows),
            "max_dist_km": max((r["dist_km"] for r in rows), default=0.0),
            "mean_dist_km": mean([r["dist_km"] for r in rows]),
            "rows": rows,
        },
        "amedas": {"stamps": len(stamps), "records": n_obs,
                   "stations": len(snap.stations),
                   "first_utc": iso(stamps[0]) if stamps else "",
                   "last_utc": iso(stamps[-1]) if stamps else ""},
        "warnings": len(snap.warnings),
        "outlooks": len(snap.rain_outlooks),
        "gust_ratio": float(snap.gust_ratio),
        "official_issues": [{"issue_utc": iso(o.issue), "points": len(o.points),
                             "source": o.source} for o in snap.official],
    }
    return out


# ----------------------------------------------------------------------
#  B. the operational track forecast against the official JMA forecast
# ----------------------------------------------------------------------
def blend_section(op: OperationalForecast, opp: OperationalForecast,
                  cfg: StudyConfig) -> dict:
    """The operational product track: JMA's forecast, then the model's motion.

    `op` is the free-running model and `opp` the official-anchored product, so
    `cross_km` is what splicing onto JMA's positions changes.  JMA publishes
    positions to +43 h here and the model is used only past that, re-anchored
    to the last published point so the product has no jump at the splice.
    """
    fin = [L for L, r in zip(opp.knots["lead"], opp.knots["radius"])
           if math.isfinite(r)]
    r_last = max(fin) if fin else 0.0
    rate = 0.0
    if len(fin) >= 2:
        l1, l2 = fin[-2], fin[-1]
        rate = ((opp.product_radius_km(l2) - opp.product_radius_km(l1))
                / (l2 - l1))
    rows = []
    for lead in np.arange(0.0, cfg.horizon + 1e-9, tf.GRID_HOURS):
        lead = float(lead)
        la, lo, pr, vm, md = opp.state(lead)
        mla, mlo, _, _, _ = op.model_state(lead)
        r = opp.product_radius_km(lead)
        rows.append(dict(
            lead_h=lead, stamp_utc=iso(opp.stamp(lead)),
            stamp_jst=jst(opp.stamp(lead)),
            lat=round(la, 3), lon=round(lo, 3),
            pres_hPa=round(pr, 1), vmax_m_s=round(vm, 1),
            gust_m_s=round(vm * opp.snap.gust_ratio, 1),
            motion_deg=round(md, 1),
            speed_km_h=round(opp.speed_km_h(lead), 1),
            src=opp.product_src(lead),
            model_lat=round(mla, 3), model_lon=round(mlo, 3),
            cross_km=round(haversine_km(la, lo, mla, mlo), 1),
            circle_km=round(r, 1) if math.isfinite(r) else "",
            circle_src=("公式" if lead <= r_last + 1e-9 else "外挿")))
    worst = max(rows, key=lambda r: r["cross_km"]) if rows else None
    return {
        "rows": rows,
        "anchor_lead_h": round(opp.anchor_lead, 2),
        "anchor_stamp_jst": jst(opp.stamp(opp.anchor_lead)),
        "n_official_points": len(opp.spec_leads),
        "model_only_from_h": round(min(
            [r["lead_h"] for r in rows if r["src"] == "モデル延長"]
            or [float("nan")]), 1),
        "circle_rate_km_h": round(rate, 2),
        "cross_at_horizon_km": rows[-1]["cross_km"] if rows else float("nan"),
        "max_cross_km": worst["cross_km"] if worst else float("nan"),
        "max_cross_lead_h": worst["lead_h"] if worst else float("nan"),
        "end_state": dict(rows[-1]) if rows else {},
        "n_official_rows": sum(1 for r in rows if r["src"] == "公式"),
        "n_model_rows": sum(1 for r in rows if r["src"] == "モデル延長"),
    }


def section_forecast(op: OperationalForecast, opp: OperationalForecast,
                     cfg: StudyConfig) -> dict:
    tr = op.track
    rows = []
    for lead in np.arange(0.0, cfg.horizon + 1e-9, tf.GRID_HOURS):
        lead = float(lead)
        la, lo, pr, vm, md = op.state(lead)
        storm, storm_src = op.storm_sector(lead)
        gale = op.gale_sector(lead)
        rmax, b = tf.wind_profile(vm, sector_radius(storm, md),
                                  sector_radius(gale, md), pr)
        rows.append(dict(
            lead_h=lead, stamp_utc=iso(op.stamp(lead)),
            stamp_jst=jst(op.stamp(lead)),
            lat=round(la, 3), lon=round(lo, 3),
            pres_hPa=round(pr, 1), vmax_m_s=round(vm, 1),
            gust_m_s=round(vm * op.snap.gust_ratio, 1),
            motion_deg=round(md, 1), speed_km_h=round(op.speed_km_h(lead), 1),
            storm_km=("" if storm is None
                       else round(sector_radius(storm, md), 1)),
            gale_km=("" if gale is None else round(sector_radius(gale, md), 1)),
            storm_src=storm_src, rmax_km=round(rmax, 1),
            wind_power_b=round(b, 3)))
    official = []
    for L, p in op.spec_leads:
        la, lo, pr, vm, md = op.state(L)
        cross = haversine_km(la, lo, p.lat, p.lon)
        midlat = math.radians(0.5 * (la + p.lat))
        official.append(dict(
            lead_h=L, stamp_utc=iso(p.stamp), stamp_jst=jst(p.stamp),
            jma_lat=p.lat, jma_lon=p.lon,
            model_lat=round(la, 3), model_lon=round(lo, 3),
            cross_km=round(cross, 1),
            circle_km=("" if not math.isfinite(p.radius_km)
                       else round(p.radius_km, 1)),
            inside_circle=("" if not math.isfinite(p.radius_km)
                           else bool(cross <= p.radius_km)),
            d_north_km=round((la - p.lat) * 111.32, 1),
            d_east_km=round((lo - p.lon) * 111.32 * math.cos(midlat), 1),
            jma_pres_hPa=p.pres_hPa, model_pres_hPa=round(pr, 1),
            jma_vmax_m_s=p.vmax_m_s, model_vmax_m_s=round(vm, 1),
            jma_gust_m_s=p.gust_m_s, jma_speed_km_h=p.speed_km_h,
            jma_course_deg=p.course_deg, model_speed_km_h=round(
                op.speed_km_h(L), 1),
            category=p.category,
            storm=sector_ja(p.storm), gale=sector_ja(p.gale)))
    prev = [o for o in op.snap.official if "PreviousIssue" in (o.source or "")]
    previous = []
    if prev:
        p0 = prev[-1]
        for p in p0.points:
            L = ((p.stamp - tr.t0).total_seconds() / 3600.0) - op.issue_hour
            if L < -cfg.horizon or L > cfg.horizon:
                continue
            la, lo, _, _, _ = op.state(max(0.0, L))
            previous.append(dict(
                lead_h=round(L, 2), stamp_utc=iso(p.stamp),
                jma_lat=p.lat, jma_lon=p.lon,
                model_lat=round(la, 3), model_lon=round(lo, 3),
                cross_km=round(haversine_km(la, lo, p.lat, p.lon), 1),
                circle_km=("" if not math.isfinite(p.radius_km)
                           else round(p.radius_km, 1))))
    matched = [o for o in official if o["circle_km"] != ""]
    return {
        "issue_utc": iso(op.issue_stamp), "issue_jst": jst(op.issue_stamp),
        "issue_hour": op.issue_hour,
        "horizon_h": cfg.horizon,
        "order": op.model.order,
        "shrink_scale": float(op.model.shrink_scale),
        "companion_radius": float(op.model.companion_radius()),
        "train_points": len(op.model.used),
        "grid_points": len(op.model.grid),
        "window_hours": op.model.window_hours,
        "pressure_rate_hPa_h": float(op.model.pressure_rate()),
        "resid_rms_km": float(op.model.resid_rms),
        "rows": rows, "official": official, "previous_issue": previous,
        "previous_issue_time": iso(prev[-1].issue) if prev else "",
        "end_state": {
            "lead_h": cfg.horizon,
            "lat": rows[-1]["lat"], "lon": rows[-1]["lon"],
            "pres_hPa": rows[-1]["pres_hPa"],
            "vmax_m_s": rows[-1]["vmax_m_s"],
            "stamp_jst": rows[-1]["stamp_jst"]},
        "inside_circle": {
            "n": len(matched),
            "leads": [m["lead_h"] for m in matched],
            "cross_km": [m["cross_km"] for m in matched],
            "circle_km": [float(m["circle_km"]) for m in matched],
            "last_lead_inside_h": max(
                [m["lead_h"] for m in matched if m["inside_circle"]]
                or [float("nan")]),
            "max_cross_km": max((m["cross_km"] for m in matched),
                                default=float("nan"))},
        "mean_speed_km_h": mean([r["speed_km_h"] for r in rows]),
        "blend": blend_section(op, opp, cfg),
    }


# ----------------------------------------------------------------------
#  C. rolling-origin hindcast of the motion model
# ----------------------------------------------------------------------
def section_hindcast(snap, cfg: StudyConfig) -> dict:
    """Verify the motion model on the analysed track it was fitted to.

    Issues every `hind_stride` hours; each forecast is fitted with
    `train_stop` at the issue time, so the verified truth is never in the
    training slice.  The VAR order is chosen on the early half of the issues
    and the choice is then checked on the late half, which is the honest way
    to report a hyper-parameter that was tuned on this same record.
    """
    tr = snap.track
    leads = [float(L) for L in cfg.leads]
    issues = [float(h) for h in np.arange(cfg.hind_first,
                                          float(tr.hours[-1]) + 1e-9,
                                          cfg.hind_stride)]
    per_order = {o: [] for o in cfg.orders}     # (issue, {lead: err})
    ref = {"persistence": [], "motion_persistence": []}
    pres_err = {L: [] for L in leads}
    vmax_err = {L: [] for L in leads}
    for ih in issues:
        stamp = tr.t0 + timedelta(hours=ih)
        rec = {}
        for order in cfg.orders:
            try:
                m = TrackForecastModel(tr, order=order, train_stop=stamp)
            except ValueError:
                continue
            fc = m.predict(cfg.hind_horizon)
            hh, ee = track_error_km(fc, tr)
            for L in leads:
                k = int(np.argmin(np.abs(hh - L))) if len(hh) else -1
                if k >= 0 and abs(hh[k] - L) < 1e-6:
                    rec.setdefault(L, {})[order] = float(ee[k])
            if order == tf.TRACK_ORDER:
                for L in leads:
                    k = int(round(L / tf.GRID_HOURS))
                    th = ih + L
                    if th > float(tr.hours[-1]) + 1e-9 or k >= len(fc["hours"]):
                        continue
                    _, _, tpr, tvm = state_at(tr, th)
                    pres_err[L].append(float(fc["pres_hPa"][k]) - tpr)
                    vmax_err[L].append(float(fc["vmax_m_s"][k]) - tvm)
        if rec:
            per_order[tf.TRACK_ORDER].append((ih, rec))
        for name, fn in (("persistence", persistence_track),
                         ("motion_persistence", motion_persistence_track)):
            fc = fn(tr, stamp, cfg.hind_horizon)
            hh, ee = track_error_km(fc, tr)
            entry = {}
            for L in leads:
                k = int(np.argmin(np.abs(hh - L))) if len(hh) else -1
                if k >= 0 and abs(hh[k] - L) < 1e-6:
                    entry[L] = float(ee[k])
            if entry:
                ref[name].append((ih, entry))
    used_issues = [ih for ih, _ in per_order[tf.TRACK_ORDER]]
    mid = float(np.median(used_issues)) if used_issues else 0.0

    def collect(entries, order=None):
        out = {L: [] for L in leads}
        for ih, rec in entries:
            for L in leads:
                v = rec.get(L) if order is None else rec.get(L, {}).get(order)
                if v is not None:
                    out[L].append(v)
        return out

    var_all = collect(per_order[tf.TRACK_ORDER], tf.TRACK_ORDER)
    early = [(ih, r) for ih, r in per_order[tf.TRACK_ORDER] if ih <= mid]
    late = [(ih, r) for ih, r in per_order[tf.TRACK_ORDER] if ih > mid]
    order_rows = []
    sel_order, sel_score = tf.TRACK_ORDER, float("inf")
    for order in cfg.orders:
        e = collect(per_order[tf.TRACK_ORDER], order)
        l = collect(late, order)
        f = collect(early, order)
        score = mean([rms(f[L]) for L in leads if f[L]])
        if math.isfinite(score) and score < sel_score:
            sel_order, sel_score = order, score
        order_rows.append(dict(
            order=order,
            rmse_early_h=round(mean([rms(f[L]) for L in leads if f[L]]), 1),
            rmse_late_h=round(mean([rms(l[L]) for L in leads if l[L]]), 1),
            rmse_all_h=round(mean([rms(e[L]) for L in leads if e[L]]), 1),
            **{f"rmse_lead{int(L)}_km": round(rms(e[L]), 1) for L in leads}))
    pers = collect(ref["persistence"])
    mot = collect(ref["motion_persistence"])
    rows = []
    for L in leads:
        rows.append(dict(
            lead_h=int(L), n=len(var_all[L]),
            rmse_var_km=round(rms(var_all[L]), 1),
            rmse_persistence_km=round(rms(pers[L]), 1),
            rmse_motion_persistence_km=round(rms(mot[L]), 1),
            mae_var_km=round(mean([abs(v) for v in var_all[L]]), 1),
            max_var_km=round(max(var_all[L], default=float("nan")), 1),
            skill_vs_persistence=round(
                1.0 - rms(var_all[L]) / rms(pers[L]), 3)
            if pers[L] and rms(pers[L]) > 0 else "",
            skill_vs_motion=round(
                1.0 - rms(var_all[L]) / rms(mot[L]), 3)
            if mot[L] and rms(mot[L]) > 0 else ""))
    return {
        "issues": used_issues, "n_issues": len(used_issues),
        "first_issue_utc": iso(tr.t0 + timedelta(hours=used_issues[0]))
        if used_issues else "",
        "last_issue_utc": iso(tr.t0 + timedelta(hours=used_issues[-1]))
        if used_issues else "",
        "split_hour": mid,
        "leads": [int(L) for L in leads],
        "rows": rows,
        "order_rows": order_rows,
        "sel_order": sel_order,
        "shipped_order": tf.TRACK_ORDER,
        "rmse_mean_var_km": round(mean([rms(var_all[L]) for L in leads]), 1),
        "rmse_mean_persistence_km": round(
            mean([rms(pers[L]) for L in leads]), 1),
        "rmse_mean_motion_km": round(mean([rms(mot[L]) for L in leads]), 1),
        "rmse_pooled_var_km": round(rms([v for L in leads
                                         for v in var_all[L]]), 1),
        "pres_rmse_by_lead": {int(L): round(rms(pres_err[L]), 2)
                              for L in leads},
        "pres_bias_by_lead": {int(L): round(mean(pres_err[L]), 2)
                              for L in leads},
        "vmax_rmse_by_lead": {int(L): round(rms(vmax_err[L]), 2)
                              for L in leads},
        "vmax_bias_by_lead": {int(L): round(mean(vmax_err[L]), 2)
                              for L in leads},
        "pres_err": pres_err, "vmax_err": vmax_err,
    }


# ----------------------------------------------------------------------
#  D. intensity: forecast against the official 諸元
# ----------------------------------------------------------------------
def section_intensity(op: OperationalForecast, hind: dict,
                      cfg: StudyConfig) -> dict:
    official = []
    for L, p in op.spec_leads:
        la, lo, pr, vm, _ = op.state(L)
        official.append(dict(
            lead_h=L, stamp_jst=jst(p.stamp), category=p.category,
            jma_pres_hPa=p.pres_hPa, model_pres_hPa=round(pr, 1),
            d_pres_hPa=round(pr - p.pres_hPa, 1)
            if math.isfinite(p.pres_hPa) else "",
            jma_vmax_m_s=p.vmax_m_s, model_vmax_m_s=round(vm, 1),
            d_vmax_m_s=round(vm - p.vmax_m_s, 1)
            if math.isfinite(p.vmax_m_s) else "",
            jma_gust_m_s=p.gust_m_s,
            model_gust_m_s=round(vm * op.snap.gust_ratio, 1),
            storm=sector_ja(p.storm)))
    leads = [float(L) for L in cfg.leads]
    return {
        "hindcast_pres_rmse_hPa": {int(L): hind["pres_rmse_by_lead"][int(L)]
                                   for L in leads},
        "hindcast_pres_bias_hPa": {int(L): hind["pres_bias_by_lead"][int(L)]
                                   for L in leads},
        "hindcast_vmax_rmse_m_s": {int(L): hind["vmax_rmse_by_lead"][int(L)]
                                   for L in leads},
        "hindcast_vmax_bias_m_s": {int(L): hind["vmax_bias_by_lead"][int(L)]
                                   for L in leads},
        "hindcast_pres_rmse_mean_hPa": round(
            mean(list(hind["pres_rmse_by_lead"].values())), 2),
        "hindcast_vmax_rmse_mean_m_s": round(
            mean(list(hind["vmax_rmse_by_lead"].values())), 2),
        "official": official,
        "official_pres_rmse_hPa": round(rms(
            [o["d_pres_hPa"] for o in official if o["d_pres_hPa"] != ""]), 2),
        "official_vmax_rmse_m_s": round(rms(
            [o["d_vmax_m_s"] for o in official if o["d_vmax_m_s"] != ""]), 2),
        "analysis_pres_hPa": float(op.track.pres_hPa[-1]),
        "forecast_pres_end_hPa": float(op.fc["pres_hPa"][-1]),
        "et_lat_start": tf.ET_LAT_START, "et_lat_end": tf.ET_LAT_END,
        "pres_relax_hPa": tf.PRES_RELAX_HPA,
    }


# ----------------------------------------------------------------------
#  E. precipitation
# ----------------------------------------------------------------------
def blended_state(op: OperationalForecast, hour):
    """(lat, lon, pres, vmax, motion, storm, gale, source) at an hour.

    The analysed bulletins are used up to the issue time and the forecast
    beyond it, so that verification (E1) and forecast (E2-E4) share one
    function with no seam at the issue time.  `source` tells which side of the
    seam an hour fell on.
    """
    if hour <= op.issue_hour + 1e-9:
        la, lo, pr, vm = state_at(op.track, hour)
        md = tf.motion_dir_deg(op.track, hour)
        storm, gale = op.snap.warnings_at(hour)
        return la, lo, pr, vm, md, storm, gale, "実況"
    lead = hour - op.issue_hour
    la, lo, pr, vm, md = op.state(lead)
    storm, _ = op.storm_sector(lead)
    return la, lo, pr, vm, md, storm, op.gale_sector(lead), "予報"


def rain_at(op: OperationalForecast, hour, lat, lon):
    """(rain rate mm/h, source) of the model field at an hour and a point."""
    la, lo, pr, _vm, md, _storm, gale, src = blended_state(op, hour)
    return rain_rate_mm_h(la, lo, pr, md, lat, lon, gale=gale), src


def rain_total_mm(op: OperationalForecast, h0, h1, lat, lon, step=1.0):
    """(accumulation mm, peak rate mm/h, wet fraction, source) over [h0, h1].

    The rate is sampled hourly and integrated trapezoidally; the wet fraction
    is the share of the samples at or above `WET_MM_H`, the threshold from
    which JMA counts precipitation.
    """
    n = max(1, int(round((h1 - h0) / step)))
    hrs = np.linspace(float(h0), float(h1), n + 1)
    rates, srcs = [], set()
    for h in hrs:
        r, s = rain_at(op, float(h), lat, lon)
        rates.append(r)
        srcs.add(s)
    tot = float(sum(0.5 * (rates[i] + rates[i + 1]) * (hrs[i + 1] - hrs[i])
                    for i in range(n)))
    wet = float(np.mean([r >= WET_MM_H for r in rates]))
    return tot, max(rates), wet, "+".join(sorted(srcs))


def district_grid(lat, lon):
    """The small lat/lon box whose model maximum stands for 「多い所で」."""
    off = np.linspace(-DISTRICT_HALF_DEG, DISTRICT_HALF_DEG, DISTRICT_GRID_N)
    scale = 1.0 / max(0.2, math.cos(math.radians(lat)))
    return [(float(lat + a), float(lon + b * scale)) for a in off for b in off]


def area_points(snap):
    """{府県予報 area code: (lat, lon, label)} used by E2/E3/E4."""
    pts = station_latlon(snap)
    out = {}
    for name, code in sorted(STATION_AREA.items()):
        if code and code not in out and name in pts:
            out[code] = (pts[name][0], pts[name][1], f"{name}(AMeDAS)")
    for code, (la, lo) in AREA_TOWN.items():
        out.setdefault(code, (la, lo, "代表地点"))
    return out


def hits_metrics(obs, mod, threshold=WET_MM_H):
    """POD / FAR / CSI / frequency bias of a wet-or-dry decision."""
    o = np.asarray(obs, dtype=float) >= threshold
    m = np.asarray(mod, dtype=float) >= threshold
    hit = int(np.sum(o & m))
    miss = int(np.sum(o & ~m))
    false = int(np.sum(~o & m))
    return {
        "n": int(o.size), "hits": hit, "misses": miss, "false_alarms": false,
        "pod": round(hit / (hit + miss), 3) if hit + miss else "",
        "far": round(false / (hit + false), 3) if hit + false else "",
        "csi": round(hit / (hit + miss + false), 3)
        if hit + miss + false else "",
        "freq_bias": round((hit + false) / (hit + miss), 3)
        if hit + miss else "",
        "obs_wet_frac": round(float(np.mean(o)), 3),
        "model_wet_frac": round(float(np.mean(m)), 3),
    }


def corr(a, b):
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    if x.size < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman(a, b):
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    if x.size < 3:
        return float("nan")
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    return corr(rx, ry)


def section_rain_obs(op: OperationalForecast, cfg: StudyConfig) -> dict:
    """E1: the parametric rain field against every AMeDAS station-hour."""
    snap, tr = op.snap, op.track
    rows = []
    for utc in sorted(snap.amedas):
        stamp = tf._from_iso(utc)
        hour = tr.hour_of(stamp)
        if hour < float(tr.hours[0]) - 1e-9 or hour > op.issue_hour + cfg.horizon:
            continue
        la, lo, pr, vm, md, storm, gale, src = blended_state(op, hour)
        for name, vals in sorted(snap.amedas[utc].items()):
            obs = vals.get("precip1h_mm", float("nan"))
            if not math.isfinite(obs):
                continue
            slat, slon = op.stations[name]
            pred = rain_rate_mm_h(la, lo, pr, md, slat, slon, gale=gale)
            wspd, wfrom = tf.vortex_wind(la, lo, vm, pr, slat, slon,
                                         storm=storm, gale=gale)
            ow = vals.get("wind_m_s", float("nan"))
            od = vals.get("wind_dir_deg", float("nan"))
            rows.append(dict(
                stamp_utc=utc, stamp_jst=jst(stamp), hour=round(hour, 2),
                station=name, area=STATION_AREA.get(name) or "",
                obs_mm_h=round(obs, 1), model_mm_h=round(pred, 2),
                err_mm_h=round(pred - obs, 2), source=src,
                dist_km=round(haversine_km(la, lo, slat, slon), 1),
                center_pres_hPa=round(pr, 1),
                obs_wind_m_s=("" if not math.isfinite(ow) else round(ow, 1)),
                model_wind_m_s=round(wspd, 1),
                obs_wind_from_deg=("" if not math.isfinite(od)
                                   else round(od, 1)),
                model_wind_from_deg=round(wfrom, 1)))
    obs = [r["obs_mm_h"] for r in rows]
    mod = [r["model_mm_h"] for r in rows]
    err = [r["err_mm_h"] for r in rows]
    heavy = [r for r in rows if r["obs_mm_h"] >= 10.0]
    heavy_mod = [r["model_mm_h"] for r in heavy]
    wo = [r["obs_wind_m_s"] for r in rows if r["obs_wind_m_s"] != ""]
    wm = [r["model_wind_m_s"] for r in rows if r["obs_wind_m_s"] != ""]
    # The vortex field is a marine product: it reproduces JMA's warning radii
    # (25 m/s at the 暴風域, 15 m/s at the 強風域) but an AMeDAS anemometer sits
    # 10 m above rough, sheltered land.  One least-squares exposure factor
    # separates the two, and how well it does says whether the *shape* of the
    # field is right.  Only station-hours with a real vortex wind (model
    # > 5 m/s) enter the fit, so the quiet far field cannot drag it down.
    pairs = [(float(r["obs_wind_m_s"]), r["model_wind_m_s"]) for r in rows
             if r["obs_wind_m_s"] != "" and r["model_wind_m_s"] > 5.0]
    po = np.array([p[0] for p in pairs])
    pm = np.array([p[1] for p in pairs])
    land_k = float((po * pm).sum() / (pm ** 2).sum()) if len(pairs) else 1.0
    land_k = float(min(1.0, max(0.1, land_k)))
    dd = np.array([((float(r["obs_wind_from_deg"]) - r["model_wind_from_deg"]
                    + 180.0) % 360.0) - 180.0 for r in rows
                   if r["obs_wind_from_deg"] != ""
                   and r["model_wind_m_s"] > 5.0])
    # The calibration is a compromise, and its cost shows on the peaks: the
    # observed wind maxima of this typhoon (三宅島 19.7 m/s on 09/20 12:20 JST)
    # happen with environmental flow the vortex-only field does not contain, so
    # a factor fitted on the mean of the near field flattens them.
    peak_rows = sorted([r for r in rows if r["obs_wind_m_s"] != ""],
                       key=lambda r: -float(r["obs_wind_m_s"]))[:8]
    wind_by_dist = []
    for lo_km in range(0, 1400, 200):
        band = [r for r in rows if r["obs_wind_m_s"] != ""
                and lo_km <= r["dist_km"] < lo_km + 200]
        if not band:
            continue
        wind_by_dist.append(dict(
            dist_km=f"{lo_km}-{lo_km + 200}", n=len(band),
            obs_mean_m_s=round(mean([float(r["obs_wind_m_s"])
                                     for r in band]), 1),
            model_mean_m_s=round(mean([r["model_wind_m_s"] for r in band]), 1),
            obs_rain_mm_h=round(mean([r["obs_mm_h"] for r in band]), 2),
            model_rain_mm_h=round(mean([r["model_mm_h"] for r in band]), 2)))
    per_station = []
    for name in sorted({r["station"] for r in rows}):
        sr = [r for r in rows if r["station"] == name]
        per_station.append(dict(
            station=name, area=STATION_AREA.get(name) or "",
            n=len(sr),
            obs_max_mm_h=max(r["obs_mm_h"] for r in sr),
            model_max_mm_h=round(max(r["model_mm_h"] for r in sr), 1),
            obs_total_mm=round(sum(r["obs_mm_h"] for r in sr), 1),
            model_total_mm=round(sum(r["model_mm_h"] for r in sr), 1),
            rmse_mm_h=round(rms([r["err_mm_h"] for r in sr]), 2),
            bias_mm_h=round(mean([r["err_mm_h"] for r in sr]), 2),
            csi_wet=hits_metrics([r["obs_mm_h"] for r in sr],
                                 [r["model_mm_h"] for r in sr])["csi"]))
    per_hour = []
    for hour in sorted({r["hour"] for r in rows}):
        hr = [r for r in rows if r["hour"] == hour]
        per_hour.append(dict(
            hour=hour, stamp_jst=hr[0]["stamp_jst"], n=len(hr),
            obs_max_mm_h=max(r["obs_mm_h"] for r in hr),
            model_max_mm_h=round(max(r["model_mm_h"] for r in hr), 1),
            obs_mean_mm_h=round(mean([r["obs_mm_h"] for r in hr]), 2),
            model_mean_mm_h=round(mean([r["model_mm_h"] for r in hr]), 2),
            rmse_mm_h=round(rms([r["err_mm_h"] for r in hr]), 2)))
    return {
        "rows": rows, "per_station": per_station, "per_hour": per_hour,
        "n": len(rows),
        "n_analysis": sum(1 for r in rows if r["source"] == "実況"),
        "n_forecast": sum(1 for r in rows if r["source"] == "予報"),
        "stations": len(per_station), "hours": len(per_hour),
        "obs_mean_mm_h": round(mean(obs), 2),
        "model_mean_mm_h": round(mean(mod), 2),
        "bias_mm_h": round(mean(err), 2),
        "rmse_mm_h": round(rms(err), 2),
        "mae_mm_h": round(mean([abs(e) for e in err]), 2),
        "corr": round(corr(obs, mod), 3),
        "spearman": round(spearman(obs, mod), 3),
        "wet": hits_metrics(obs, mod),
        "wet_1mm": hits_metrics(obs, mod, 1.0),
        "wet_10mm": hits_metrics(obs, mod, 10.0),
        "heavy_n": len(heavy),
        "heavy_pod": hits_metrics([r["obs_mm_h"] for r in heavy],
                                  heavy_mod)["pod"],
        "heavy_underestimate_mm_h": round(
            mean([m - r["obs_mm_h"] for r, m in zip(heavy, heavy_mod)]), 2)
        if heavy else "",
        "wind": {
            "n": len(wo), "n_fit": len(pairs),
            "rmse_marine_m_s": round(
                rms([m - float(o) for o, m in zip(wo, wm)]), 2),
            "bias_marine_m_s": round(
                mean([m - float(o) for o, m in zip(wo, wm)]), 2),
            "land_factor": round(land_k, 3),
            "rmse_land_m_s": round(
                rms([land_k * m - float(o) for o, m in zip(wo, wm)]), 2),
            "bias_land_m_s": round(
                mean([land_k * m - float(o) for o, m in zip(wo, wm)]), 2),
            "corr": round(corr(wo, wm), 3),
            "peak_n": len(peak_rows),
            "peak_obs_m_s": round(
                mean([float(r["obs_wind_m_s"]) for r in peak_rows]), 1),
            "peak_model_m_s": round(
                mean([r["model_wind_m_s"] for r in peak_rows]), 1),
            "peak_calibrated_m_s": round(
                mean([land_k * r["model_wind_m_s"] for r in peak_rows]), 1),
            "dir_n": int(dd.size),
            "dir_mae_deg": round(float(np.mean(np.abs(dd))), 1) if dd.size
            else "",
            "dir_bias_deg": round(math.degrees(math.atan2(
                float(np.mean(np.sin(np.radians(dd)))),
                float(np.mean(np.cos(np.radians(dd)))))), 1) if dd.size else "",
            "by_distance": wind_by_dist,
        },
    }


def _pref_of(head):
    return head.split("気象解説情報")[0].strip()


def _outlook_area_rows(op: OperationalForecast, cfg: StudyConfig):
    """The latest prefecture outlook of every prefecture the R1 files cover."""
    pops = parse_pop(cfg.data)
    prefs = {r["pref"] for r in pops}
    code_of = {(r["pref"], r["area"]): r["code"] for r in pops}
    latest = {}
    for issue, blocks, head in op.snap.rain_outlooks:
        pref = _pref_of(head)
        if pref not in prefs:
            continue
        if pref not in latest or tf._as_dt(issue) > tf._as_dt(latest[pref][0]):
            latest[pref] = (issue, blocks, head)
    return latest, code_of, prefs


def section_rain_peaks(op: OperationalForecast, cfg: StudyConfig) -> dict:
    """E2: the 「多い所で」 peaks JMA publishes, over JMA's own windows.

    A 「多い所で」 value is the maximum accumulation over a district, so the
    model side is evaluated both at the district's observing point and as the
    maximum over a small box around it (`district_grid`), which is the closest
    thing the smooth model field has to an areal maximum.  The 1-hour peaks
    JMA states come from the blocks whose window it does not give, so the
    largest of them is taken.
    """
    latest, code_of, _ = _outlook_area_rows(op, cfg)
    pts = area_points(op.snap)
    rows = []
    for pref in sorted(latest):
        issue, blocks, head = latest[pref]
        acc = [b for b in blocks if b.start and b.end and b.hours >= 12]
        peaks = [b for b in blocks if b.hours == 1]
        for b in acc:
            h0 = op.track.hour_of(b.start)
            h1 = op.track.hour_of(b.end)
            for area, official in sorted(b.regions.items()):
                code = code_of.get((pref, area))
                pt = pts.get(code) if code else None
                if pt is None:
                    continue
                lat, lon, label = pt
                tot, pk, wet, src = rain_total_mm(op, h0, h1, lat, lon)
                g = [rain_total_mm(op, h0, h1, la, lo)
                     for la, lo in district_grid(lat, lon)]
                o1 = max([bl.regions[area] for bl in peaks
                          if area in bl.regions], default=float("nan"))
                rows.append(dict(
                    pref=pref, area=area, code=code, point=label,
                    lat=round(lat, 3), lon=round(lon, 3),
                    issue_utc=iso(issue),
                    window_jst=f"{jst(b.start)}〜{jst(b.end)}",
                    window_h0=round(h0, 2), window_h1=round(h1, 2),
                    hours=b.hours,
                    official_mm=official, model_mm=round(tot, 1),
                    district_max_mm=round(max(r[0] for r in g), 1),
                    ratio=round(tot / official, 2) if official else "",
                    ratio_max=round(max(r[0] for r in g) / official, 2)
                    if official else "",
                    official_peak_mm_h=o1 if math.isfinite(o1) else "",
                    model_peak_mm_h=round(pk, 1),
                    district_peak_mm_h=round(max(r[1] for r in g), 1),
                    wet_frac=round(wet, 2), source=src,
                    forecast_frac=round(max(0.0, min(1.0, (h1 - op.issue_hour)
                                                     / (h1 - h0))), 2)))
    ratios = [r["ratio"] for r in rows if r["ratio"] != ""]
    ratios_max = [r["ratio_max"] for r in rows if r["ratio_max"] != ""]
    worst = max(rows, key=lambda r: r["official_mm"]) if rows else None
    return {
        "rows": rows, "n": len(rows),
        "districts": len({r["code"] for r in rows}),
        "prefectures": sorted({r["pref"] for r in rows}),
        "windows": sorted({r["window_jst"] for r in rows}),
        "official_max_mm": worst["official_mm"] if worst else float("nan"),
        "official_max_where": (worst["pref"] + worst["area"]) if worst else "",
        "model_mm_at_official_max": worst["model_mm"] if worst else "",
        "district_max_at_official_max": worst["district_max_mm"]
        if worst else "",
        "model_max_mm": max((r["district_max_mm"] for r in rows),
                            default=float("nan")),
        "median_ratio": round(float(np.median(ratios)), 2) if ratios else "",
        "median_ratio_district_max": round(float(np.median(ratios_max)), 2)
        if ratios_max else "",
        "n_district_max_reaches_official": sum(1 for r in ratios_max
                                               if r >= 1.0),
        "peak_ratio_median": round(float(np.median(
            [r["district_peak_mm_h"] / r["official_peak_mm_h"] for r in rows
             if r["official_peak_mm_h"] != "" and r["official_peak_mm_h"]])), 2)
        if rows else "",
    }


def section_rain_pop(op: OperationalForecast, cfg: StudyConfig) -> dict:
    """E3: the official 6-hourly 降水確率 against the deterministic field.

    A probability cannot be verified against a deterministic forecast, so the
    comparison is made twice over: continuously (does the model accumulation
    rank the districts the way the official probability does?) and as a
    decision (official POP >= 50 % against model accumulation >= 1 mm, the
    threshold from which JMA counts a fall).
    """
    pops = parse_pop(cfg.data)
    pts = area_points(op.snap)
    rows = []
    for r in pops:
        pt = pts.get(r["code"])
        if pt is None:
            continue
        lat, lon, label = pt
        h0 = op.track.hour_of(r["start"])
        h1 = h0 + r["hours"]
        tot, pk, wet, src = rain_total_mm(op, h0, h1, lat, lon)
        g = [rain_total_mm(op, h0, h1, la, lo)
             for la, lo in district_grid(lat, lon)]
        gmax = max(x[0] for x in g)
        official_rain = bool(r["pop"] >= 50.0)
        model_rain = bool(gmax >= 1.0)
        rows.append(dict(
            window_utc=iso(r["start"]),
            window_jst=jst(r["start"]), hours=r["hours"],
            pref=r["pref"], area=r["area"], code=r["code"], point=label,
            pop_pct=r["pop"], model_mm=round(tot, 1),
            district_max_mm=round(gmax, 1),
            model_peak_mm_h=round(pk, 1),
            district_peak_mm_h=round(max(x[1] for x in g), 1),
            wet_frac=round(wet, 2), source=src,
            official_rain=official_rain, model_rain=model_rain,
            agree=bool(official_rain == model_rain)))
    pop = [r["pop_pct"] for r in rows]
    dm = [r["district_max_mm"] for r in rows]
    pm = [r["model_mm"] for r in rows]
    hit = sum(1 for r in rows if r["official_rain"] and r["model_rain"])
    miss = sum(1 for r in rows if r["official_rain"] and not r["model_rain"])
    false = sum(1 for r in rows if r["model_rain"] and not r["official_rain"])
    per_window = []
    for w in sorted({r["window_jst"] for r in rows}):
        wr = [r for r in rows if r["window_jst"] == w]
        per_window.append(dict(
            window_jst=w, n=len(wr),
            pop_mean=round(mean([r["pop_pct"] for r in wr]), 1),
            model_mean_mm=round(mean([r["district_max_mm"] for r in wr]), 1),
            model_max_mm=round(max(r["district_max_mm"] for r in wr), 1),
            agree=round(mean([1.0 if r["agree"] else 0.0 for r in wr]), 2)))
    bins = [(0, 20), (20, 50), (50, 80), (80, 101)]
    table = []
    for lo, hi in bins:
        band = [r for r in rows if lo <= r["pop_pct"] < hi]
        table.append(dict(
            pop_band=f"{lo}-{hi if hi < 101 else 100}", n=len(band),
            model_mean_mm=round(mean([r["district_max_mm"]
                                      for r in band]), 1) if band else "",
            model_median_mm=round(float(np.median(
                [r["district_max_mm"] for r in band])), 1) if band else "",
            wet_frac_mean=round(mean([r["wet_frac"] for r in band]), 2)
            if band else ""))
    return {
        "rows": rows, "per_window": per_window, "pop_bins": table,
        "n": len(rows),
        "areas": len({r["code"] for r in rows}),
        "windows": len(per_window),
        "window_first_jst": per_window[0]["window_jst"] if per_window else "",
        "window_last_jst": per_window[-1]["window_jst"] if per_window else "",
        "pop_mean": round(mean(pop), 1),
        "corr_pop_district_max": round(corr(pop, dm), 3),
        "spearman_pop_district_max": round(spearman(pop, dm), 3),
        "corr_pop_point": round(corr(pop, pm), 3),
        "hits": hit, "misses": miss, "false_alarms": false,
        "agree_frac": round((hit + sum(1 for r in rows
                                       if not r["official_rain"]
                                       and not r["model_rain"])) / len(rows), 3)
        if rows else "",
        "pod": round(hit / (hit + miss), 3) if hit + miss else "",
        "far": round(false / (hit + false), 3) if hit + false else "",
        "csi": round(hit / (hit + miss + false), 3)
        if hit + miss + false else "",
    }


def rain_anchors(peaks: dict, op: OperationalForecast) -> dict:
    """{area code: amplitude anchor from JMA's published 「多い所で」}.

    E2 shows the model field reaches only about half the areal maximum JMA
    publishes, with a fairly stable factor per district.  That factor is used
    here to set the amplitude of the product, so JMA's number carries the
    magnitude and the model supplies the spatial and temporal distribution it
    does not publish.  A district JMA did not state a value for falls back to
    the median factor of the districts it did state.

    The district maximum is recomputed on the track `op` forecasts, so that
    the anchor corrects the field the product actually integrates.  It is a
    single factor per district, taken over the window JMA states it for and
    held across the whole horizon: an assumption, not a calibration.
    """
    pts = area_points(op.snap)
    per = {}
    for r in peaks["rows"]:
        if not r.get("official_mm"):
            continue
        pt = pts.get(r["code"])
        if pt is None:
            continue
        g = [rain_total_mm(op, r["window_h0"], r["window_h1"], la, lo)
             for la, lo in district_grid(pt[0], pt[1])]
        dm = max(x[0] for x in g)
        if not dm or dm <= 0.0:
            continue
        per.setdefault(r["code"], []).append(dict(
            scale=r["official_mm"] / dm, official_mm=r["official_mm"],
            district_max_mm=round(dm, 1), window_jst=r["window_jst"],
            pref=r["pref"], area=r["area"]))
    scales = [x["scale"] for v in per.values() for x in v]
    med = float(np.median(scales)) if scales else float("nan")
    out = {}
    for code, items in per.items():
        out[code] = dict(
            scale=float(np.mean([i["scale"] for i in items])),
            n_windows=len(items), src="公式「多い所で」",
            windows=[i["window_jst"] for i in items],
            official_mm=max(i["official_mm"] for i in items),
            district_max_mm=max(i["district_max_mm"] for i in items))
    return {"per_code": out, "median_scale": med,
            "n_codes": len(out), "n_scales": len(scales),
            "min_scale": min(scales) if scales else float("nan"),
            "max_scale": max(scales) if scales else float("nan")}


def section_rain_sites(op: OperationalForecast, cfg: StudyConfig,
                       peaks: dict) -> dict:
    """E4: the forecast accumulations at every operating site.

    Hourly model rain over the whole forecast window at the AMeDAS sites, plus
    the 6-hour and 24-hour totals on the same windows JMA uses, so that the
    product an operator would read off is directly comparable with the
    official 「多い所で」 values of E2.  Each site also carries the product
    anchored to that official value (`rain_anchors`), run on the
    official-anchored track.
    """
    h0 = op.issue_hour
    hourly = []
    sites = []
    anchors = rain_anchors(peaks, op)
    per_code = anchors["per_code"]
    med = anchors["median_scale"]
    official = {}
    for r in peaks["rows"]:
        cur = official.get(r["code"])
        if cur is None or r["official_mm"] > cur["official_mm"]:
            official[r["code"]] = r
    for name in sorted(op.stations):
        lat, lon = op.stations[name]
        code = STATION_AREA.get(name) or ""
        anc = per_code.get(code)
        if anc is None and math.isfinite(med):
            anc = dict(scale=med, n_windows=0, src="他地域の中央値",
                       windows=[], official_mm="", district_max_mm="")
        scale = float(anc["scale"]) if anc else 1.0
        hrs = np.arange(0.0, cfg.horizon + 1e-9, 1.0)
        rates = []
        for h in hrs:
            r, _ = rain_at(op, h0 + float(h), lat, lon)
            rates.append(r)
            hourly.append(dict(site=name, lead_h=float(h),
                               stamp_utc=iso(op.stamp(float(h))),
                               stamp_jst=jst(op.stamp(float(h))),
                               rain_mm_h=round(r, 2),
                               anchored_mm_h=round(r * scale, 2)))
        tot = float(np.sum(rates[:-1]))
        i_pk = int(np.argmax(rates))
        first = next((float(h) for h, r in zip(hrs, rates) if r >= 1.0), None)
        last = next((float(h) for h, r in zip(hrs[::-1], rates[::-1])
                     if r >= 1.0), None)
        last = None if last is None else float(hrs[-1]) - last
        d24 = float(np.sum(rates[:24])) if len(rates) > 24 else float("nan")
        off = official.get(code)
        if off:
            o_tot, o_pk, _, _ = rain_total_mm(op, off["window_h0"],
                                              off["window_h1"], lat, lon)
        sites.append(dict(
            site=name, area=code,
            lat=round(lat, 3), lon=round(lon, 3),
            total_mm=round(tot, 1),
            first_24h_from_issue_mm=round(d24, 1),
            peak_mm_h=round(rates[i_pk], 1),
            peak_stamp_jst=jst(op.stamp(float(hrs[i_pk]))),
            peak_lead_h=float(hrs[i_pk]),
            hours_ge_1mm=round(float(np.sum(np.asarray(rates) >= 1.0)), 1),
            hours_ge_10mm=round(float(np.sum(np.asarray(rates) >= 10.0)), 1),
            rain_start_lead_h="" if first is None else first,
            rain_end_lead_h="" if last is None else last,
            official_window_jst=off["window_jst"] if off else "",
            official_24h_mm=off["official_mm"] if off else "",
            model_official_window_mm=round(o_tot, 1) if off else "",
            official_peak_mm_h=off["official_peak_mm_h"] if off else "",
            model_official_window_peak_mm_h=round(o_pk, 1) if off else "",
            anchor_scale=round(scale, 3), anchor_src=anc["src"] if anc else "",
            anchored_total_mm=round(tot * scale, 1),
            anchored_first_24h_mm=round(d24 * scale, 1),
            anchored_peak_mm_h=round(rates[i_pk] * scale, 1),
            anchored_official_window_mm=(round(o_tot * scale, 1)
                                         if off else "")))
    six = []
    lead_windows = [(float(k), float(k) + 6.0)
                    for k in np.arange(0.0, cfg.horizon - 5.9, 6.0)]
    scale_of = {r["site"]: r["anchor_scale"] for r in sites}
    for name in sorted(op.stations):
        lat, lon = op.stations[name]
        sc = scale_of.get(name, 1.0)
        for a, b in lead_windows:
            tot, pk, wet, _ = rain_total_mm(op, h0 + a, h0 + b, lat, lon,
                                            step=1.0)
            six.append(dict(site=name, start_lead_h=float(a),
                            start_jst=jst(op.stamp(a)),
                            end_jst=jst(op.stamp(b)),
                            total_mm=round(tot, 1), peak_mm_h=round(pk, 1),
                            anchored_total_mm=round(tot * sc, 1),
                            anchored_peak_mm_h=round(pk * sc, 1),
                            wet_frac=round(wet, 2)))
    return {
        "sites": sites, "six_hour": six, "hourly": hourly,
        "issue_jst": jst(op.issue_stamp),
        "horizon_h": cfg.horizon,
        "wettest_site": max(sites, key=lambda r: r["total_mm"])["site"]
        if sites else "",
        "max_total_mm": max((r["total_mm"] for r in sites), default=0.0),
        "max_peak_mm_h": max((r["peak_mm_h"] for r in sites), default=0.0),
        "anchored_wettest_site": max(
            sites, key=lambda r: r["anchored_total_mm"])["site"]
        if sites else "",
        "anchored_max_total_mm": max(
            (r["anchored_total_mm"] for r in sites), default=0.0),
        "anchored_max_peak_mm_h": max(
            (r["anchored_peak_mm_h"] for r in sites), default=0.0),
        "anchor": {k: v for k, v in anchors.items() if k != "per_code"},
        "anchor_scales": scale_of,
        "n_sites": len(sites),
    }


# ----------------------------------------------------------------------
#  F. operations: go / no-go at the sites
# ----------------------------------------------------------------------
def _scale_wind(a, k):
    if abs(k - 1.0) < 1e-9:
        return a
    return dataclasses.replace(
        a, wind=tuple(v * k for v in a.wind), gust_rms=a.gust_rms * k)


def site_state(ac, w, a):
    """The decision quantities of one site at one lead time.

    The simulator is asked for the weather terms (density, propeller loss,
    extra drag, visibility, lightning) and the airframe model for the trimmed
    cruise state, so the thrust margin is the same quantity the other studies
    of this repository report.  The wind and its gust are taken analytically
    from the configs -- mean plus the one-hour peak factor of the gust process
    -- so that no random draw enters the decision.
    """
    wx = Weather(w, a, seed=0)
    eff = wx.effects(0.0, PATROL_ALT, V_CRUISE, area=ac.geom.S)
    rho = float(eff.density_kg_m3)
    trim = trim_state(ac, rho, V_CRUISE)
    cd = ac.CD(trim["CL_trim"], height_m=PATROL_ALT) + float(eff.extra_cd)
    drag = float(trim["q_Pa"]) * ac.geom.S * cd
    thrust = ac.prop.thrust(V_CRUISE, 1.0, rho=rho) * float(eff.prop_factor)
    margin = thrust / drag if drag > 0.0 else float("nan")
    wspd = math.hypot(a.wind[0], a.wind[1])
    gust = wspd + GUST_PEAK_FACTOR * float(a.gust_rms)
    clmax = ac.aero.CL_max * max(0.1, float(eff.cl_max_factor))
    v_stall = math.sqrt(2.0 * ac.W / (rho * ac.geom.S * clmax))
    breach = []
    if wspd > OPS_WIND_MAX_M_S:
        breach.append("wind")
    if gust > OPS_GUST_MAX_M_S:
        breach.append("gust")
    if w.rain_mm_h > OPS_RAIN_MAX_MM_H:
        breach.append("rain")
    if w.visibility_m < OPS_VIS_MIN_M:
        breach.append("visibility")
    if w.lightning_risk >= OPS_LIGHTNING_MAX:
        breach.append("lightning")
    if not trim["trim_feasible"]:
        breach.append("trim")
    if not math.isfinite(margin) or margin < OPS_MARGIN_MIN:
        breach.append("thrust")
    return dict(
        go=not breach, breach=",".join(breach), n_breach=len(breach),
        wind_m_s=round(wspd, 1), gust_m_s=round(gust, 1),
        rain_mm_h=round(float(w.rain_mm_h), 1),
        visibility_m=round(float(w.visibility_m), 0),
        lightning_risk=round(float(w.lightning_risk), 2),
        condition=eff.condition,
        density_kg_m3=round(rho, 4), prop_factor=round(float(eff.prop_factor), 3),
        extra_cd=round(float(eff.extra_cd), 5),
        margin=round(margin, 3) if math.isfinite(margin) else "",
        v_stall_m_s=round(v_stall, 2),
        alpha_trim_deg=round(float(trim["alpha_trim_deg"]), 2))


def _go_windows(leads, flags):
    """Contiguous runs of go decisions as [start, end] lead pairs."""
    out, start = [], None
    for L, g in zip(leads, flags):
        if g and start is None:
            start = L
        elif not g and start is not None:
            out.append([start, L - (leads[1] - leads[0]) if len(leads) > 1
                        else start])
            start = None
    if start is not None:
        out.append([start, leads[-1]])
    return out


def section_ops(op: OperationalForecast, opm: OperationalForecast,
                cfg: StudyConfig, e1: dict, e4: dict) -> dict:
    """F: the go/no-go timeline of every site, from two tracks.

    The decision procedure is run twice on identical limits: once on the
    operational product track `op` (JMA's own forecast positions to the last
    one JMA publishes, the model's motion spliced past it) and once on the
    free-running VAR model `opm`, which is what the product would have been
    without JMA's positions.  Where the two disagree is what blending onto the
    official forecast changed.

    The primary decision uses the marine vortex wind, because the boat flies
    from open water and JMA states its warning radii in marine winds.  The
    Two sensitivities ride along with the decision instead of being folded
    into it.  `go_frac_land_exposure` applies the land-exposure factor E1
    fitted, which is not a pure exposure correction: E1 shows the observed
    station winds carry a large environmental component the vortex-only field
    does not contain.  `go_frac_anchored_rain` applies E4's anchor to the rain
    term, i.e. the rain the product actually publishes; the primary decision
    uses the un-anchored model field, so its rain breaches are a lower bound.
    """
    ac = Aircraft()
    land_k = float(e1["wind"]["land_factor"])
    anchor = float(op.anchor_lead)
    leads = [float(L) for L in np.arange(0.0, cfg.horizon + 1e-9,
                                         cfg.ops_stride)]
    official_peak = {r["area"]: r["official_peak_mm_h"] for r in e4["sites"]
                     if r["official_peak_mm_h"] != ""}
    rain_k = e4["anchor_scales"]
    rows, vrows, rain_rows, land_flags, rain_flags = [], [], [], {}, {}
    for name in sorted(op.stations):
        lat, lon = op.stations[name]
        code = STATION_AREA.get(name) or ""
        for lead in leads:
            w, a = op.configs(lead, lat, lon)
            st = site_state(ac, w, a)
            la, lo, pr, vm, md = op.state(lead)
            rows.append(dict(
                site=name, area=code, lead_h=lead,
                stamp_jst=jst(op.stamp(lead)),
                dist_km=round(haversine_km(la, lo, lat, lon), 1),
                track_src=op.product_src(lead),
                official_peak_mm_h=official_peak.get(code, ""), **st))
            land_flags.setdefault(name, []).append(
                site_state(ac, w, _scale_wind(a, land_k))["go"])
            rst = site_state(ac, tf.scale_rain(w, rain_k.get(name, 1.0)), a)
            rain_flags.setdefault(name, []).append(rst["go"])
            rain_rows.append(rst)
            vw, va = opm.configs(lead, lat, lon)
            vla, vlo, _, _, _ = opm.state(lead)
            vrows.append(dict(
                site=name, area=code, lead_h=lead,
                stamp_jst=jst(opm.stamp(lead)),
                dist_km=round(haversine_km(vla, vlo, lat, lon), 1),
                **site_state(ac, vw, va)))
    per_site = []
    for name in sorted(op.stations):
        sr = [r for r in rows if r["site"] == name]
        vr = [r for r in vrows if r["site"] == name]
        flags = [r["go"] for r in sr]
        breach = {}
        for r in sr:
            for b in (r["breach"].split(",") if r["breach"] else []):
                breach[b] = breach.get(b, 0) + 1
        first_go = next((r["lead_h"] for r in sr if r["go"]), None)
        per_site.append(dict(
            site=name, area=sr[0]["area"], n=len(sr),
            go_n=sum(flags), no_go_n=len(sr) - sum(flags),
            go_frac=round(sum(flags) / len(sr), 2) if sr else "",
            first_go_lead_h="" if first_go is None else first_go,
            go_windows=[f"{a:.0f}-{b:.0f}h"
                        for a, b in _go_windows(leads, flags)],
            min_margin=min([r["margin"] for r in sr if r["margin"] != ""],
                           default=float("nan")),
            max_wind_m_s=max(r["wind_m_s"] for r in sr),
            max_gust_m_s=max(r["gust_m_s"] for r in sr),
            max_rain_mm_h=max(r["rain_mm_h"] for r in sr),
            min_visibility_m=min(r["visibility_m"] for r in sr),
            max_lightning=max(r["lightning_risk"] for r in sr),
            binding=", ".join(f"{k}×{v}" for k, v in sorted(
                breach.items(), key=lambda kv: -kv[1])[:3]),
            go_frac_land=round(mean([1.0 if g else 0.0
                                     for g in land_flags[name]]), 2),
            go_frac_anchor_rain=round(mean(
                [1.0 if g else 0.0 for g in rain_flags[name]]), 2),
            anchor_rain_scale=round(float(rain_k.get(name, 1.0)), 2),
            var_agree_frac=round(mean(
                [1.0 if a["go"] == b["go"] else 0.0
                 for a, b in zip(sr, vr)]), 2) if vr else "",
            var_go_frac=round(sum(r["go"] for r in vr) / len(vr), 2)
            if vr else ""))
    per_lead = []
    for lead in leads:
        lr = [r for r in rows if r["lead_h"] == lead]
        vl = {r["site"]: r for r in vrows if r["lead_h"] == lead}
        per_lead.append(dict(
            lead_h=lead, stamp_jst=jst(op.stamp(lead)),
            track_src=lr[0]["track_src"] if lr else "",
            go_sites=sum(1 for r in lr if r["go"]), n_sites=len(lr),
            worst_site=min(lr, key=lambda r: (r["go"], r["margin"]
                                              if r["margin"] != "" else 9.9))[
                "site"],
            var_go_sites=(sum(1 for r in vl.values() if r["go"])
                          if vl else ""),
            var_n_sites=len(vl)))
    agree = [(a, b) for a, b in zip(rows, vrows)
             if a["site"] == b["site"]
             and abs(a["lead_h"] - b["lead_h"]) < 1e-9]
    disagree = [dict(site=a["site"], lead_h=a["lead_h"],
                     stamp_jst=a["stamp_jst"], track_src=a["track_src"],
                     product_go=a["go"], var_go=b["go"],
                     product_breach=a["breach"], var_breach=b["breach"],
                     product_wind=a["wind_m_s"], var_wind=b["wind_m_s"],
                     product_rain=a["rain_mm_h"], var_rain=b["rain_mm_h"],
                     product_dist_km=a["dist_km"], var_dist_km=b["dist_km"])
                for a, b in agree if a["go"] != b["go"]]
    def _breaches(rs):
        out = {}
        for r in rs:
            for b in (r["breach"].split(",") if r["breach"] else []):
                out[b] = out.get(b, 0) + 1
        return out

    breach_all = _breaches(rows)
    breach_rain_anchored = _breaches(rain_rows)
    return {
        "rows": rows, "var_rows": vrows, "per_site": per_site,
        "per_lead": per_lead, "disagree": disagree,
        "limits": OPS_LIMITS, "limits_source": "weather.PRESETS['rain']",
        "land_wind_factor": round(land_k, 3),
        "cruise_m_s": V_CRUISE, "patrol_alt_m": PATROL_ALT,
        "n_decisions": len(rows), "n_var_decisions": len(vrows),
        "anchor_lead_h": anchor,
        "go_frac": round(mean([1.0 if r["go"] else 0.0 for r in rows]), 3),
        "go_frac_land_exposure": round(mean(
            [1.0 if g else 0.0 for v in land_flags.values() for g in v]), 3),
        "go_frac_anchored_rain": round(mean(
            [1.0 if g else 0.0 for v in rain_flags.values() for g in v]), 3),
        "anchor_rain_median_scale": round(float(np.median(
            [v for v in rain_k.values()] or [1.0])), 2),
        "var_go_frac": round(mean([1.0 if r["go"] else 0.0
                                   for r in vrows]), 3),
        "agreement_frac": round(mean([1.0 if a["go"] == b["go"] else 0.0
                                      for a, b in agree]), 3) if agree else "",
        "agreement_frac_official_window": round(mean(
            [1.0 if a["go"] == b["go"] else 0.0
             for a, b in agree if a["lead_h"] <= anchor + 1e-9]), 3)
        if agree else "",
        "n_disagree": len(disagree),
        "n_disagree_product_go": sum(1 for d in disagree if d["product_go"]),
        "n_disagree_var_go": sum(1 for d in disagree if d["var_go"]),
        "n_disagree_official_window": sum(
            1 for d in disagree if d["lead_h"] <= anchor + 1e-9),
        "breach_counts": breach_all,
        "breach_counts_anchored_rain": breach_rain_anchored,
        "binding_constraint": max(breach_all, key=breach_all.get)
        if breach_all else "none",
        "all_go_from_lead_h": next(
            (r["lead_h"] for r in per_lead if r["go_sites"] == r["n_sites"]),
            float("nan")),
        "go_frac_official_window": round(mean(
            [1.0 if r["go"] else 0.0 for r in rows
             if r["lead_h"] <= anchor + 1e-9]), 3) if rows else "",
        "first_go_overall": min([r["lead_h"] for r in rows if r["go"]],
                                default=float("nan")),
    }


# ----------------------------------------------------------------------
#  G. what the bulletins themselves say
# ----------------------------------------------------------------------
def section_bulletins(snap, cfg: StudyConfig) -> dict:
    tr = snap.track
    warn_rows = []
    for h, storm, gale in snap.warnings:
        warn_rows.append(dict(
            hour=round(float(h), 2),
            stamp_utc=iso(tr.t0 + timedelta(hours=float(h))),
            stamp_jst=jst(tr.t0 + timedelta(hours=float(h))),
            storm=sector_ja(storm), gale=sector_ja(gale),
            storm_mean_km=("" if storm is None
                           else round(storm.mean_radius(), 1)),
            gale_mean_km=("" if gale is None else round(gale.mean_radius(), 1))))
    blocks = []
    for issue, outlooks, head in snap.rain_outlooks:
        for o in outlooks:
            blocks.append(dict(
                issue_utc=iso(issue), head=head, hours=o.hours,
                start_jst=jst(o.start) if o.start else "",
                end_jst=jst(o.end) if o.end else "",
                continued=o.continued, n_regions=len(o.regions),
                max_mm=o.max_mm(),
                regions="; ".join(f"{k} {v:.0f}ミリ"
                                  for k, v in sorted(o.regions.items()))))
    heads = {}
    for _issue, _blocks, head in snap.rain_outlooks:
        heads[head] = heads.get(head, 0) + 1
    src_heads = {}
    for f in snap.fixes:
        key = (f.source or "").split("|")[0]
        src_heads[key] = src_heads.get(key, 0) + 1
    circles = []
    for o in snap.official:
        for p in o.points:
            if math.isfinite(p.radius_km):
                circles.append(dict(issue_utc=iso(o.issue),
                                    stamp_jst=jst(p.stamp),
                                    lead_h=round((p.stamp - tr.t0
                                                  ).total_seconds() / 3600.0
                                                 - tr.hour_of(o.issue), 1),
                                    radius_km=round(p.radius_km, 1),
                                    source=o.source))
    man = dict(snap.manifest or {})
    return {
        "warnings": warn_rows, "outlook_blocks": blocks,
        "heads": heads, "fix_heads": src_heads,
        "probability_circles": circles,
        "sources": dict(man.get("sources", {})),
        "n_bulletins": int(man.get("bulletins", 0)),
        "n_fixes": len(snap.fixes),
        "n_analysis_fixes": sum(1 for f in snap.fixes if f.is_analysis),
        "n_forecast_fixes": sum(1 for f in snap.fixes if not f.is_analysis),
        "n_outlook_blocks": len(blocks),
        "n_outlooks": len(snap.rain_outlooks),
        "max_official_lead_h": max((c["lead_h"] for c in circles),
                                   default=float("nan")),
        "max_circle_km": max((c["radius_km"] for c in circles),
                             default=float("nan")),
        "storm_max_mean_km": max(
            [w["storm_mean_km"] for w in warn_rows if w["storm_mean_km"] != ""],
            default=float("nan")),
        "gale_max_mean_km": max(
            [w["gale_mean_km"] for w in warn_rows if w["gale_mean_km"] != ""],
            default=float("nan")),
        "peak_mm_h_published": max(
            [b["max_mm"] for b in blocks if b["hours"] == 1], default=float("nan")),
        "peak_24h_mm_published": max(
            [b["max_mm"] for b in blocks if b["hours"] == 24],
            default=float("nan")),
        "peak_1h_where": _peak_where(blocks, 1),
        "peak_24h_where": _peak_where(blocks, 24),
    }


def _peak_where(blocks: list, hours: int) -> str:
    """Region, issue and window of the largest published `hours`-hour total."""
    best = None
    for b in blocks:
        if b["hours"] != hours or not math.isfinite(b["max_mm"]):
            continue
        if best is None or b["max_mm"] > best["max_mm"]:
            best = b
    if best is None:
        return ""
    region = ""
    for pair in (best["regions"] or "").split("; "):
        name, _, val = pair.rpartition(" ")
        if val.rstrip("ミリ") == f"{best['max_mm']:.0f}":
            region = name
            break
    return (f"{region or '?'}({best['head'].split('（')[0]} "
            f"{jst(best['issue_utc'])} 発行、"
            f"{best['start_jst']} 〜 {best['end_jst']} JST)")


# ----------------------------------------------------------------------
#  plots
# ----------------------------------------------------------------------
def make_plots(cfg: StudyConfig, snap, op: OperationalForecast, A: dict,
               B: dict, C: dict, D: dict, E1: dict, E2: dict, E3: dict,
               E4: dict, F: dict) -> list:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.patches import Circle

    available = {f.name for f in font_manager.fontManager.ttflist}
    for cand in ("Noto Sans CJK JP", "Noto Sans CJK SC", "IPAexGothic"):
        if cand in available:
            plt.rcParams["font.family"] = cand
            break
    names = []
    tr = snap.track
    ih = op.issue_hour

    def save(fig, name, title, tight=True):
        """Title, layout and save.  `tight=False` for twin axes, which
        `tight_layout` refuses to handle."""
        fig.suptitle(title, fontsize=11)
        if tight:
            fig.tight_layout(rect=(0, 0, 1, 0.96))
        else:
            fig.subplots_adjust(left=0.08, right=0.90, top=0.90, bottom=0.06)
        p = cfg.out / name
        fig.savefig(p, dpi=140)
        names.append(p)
        plt.close(fig)
        return p

    # -- B. track map + cross-track error --------------------------------
    fig = plt.figure(figsize=(13.0, 6.4))
    ax = fig.add_subplot(1, 2, 1)
    kind_col = {"analysis": "tab:blue", "estimate": "tab:cyan"}
    for i in range(len(tr)):
        ax.plot(tr.lon[i], tr.lat[i], "o", ms=3.0,
                color=kind_col.get(tr.kinds[i], "tab:blue"))
    ax.plot(tr.lon, tr.lat, "-", color="tab:blue", lw=0.9, alpha=0.7)
    lat = [r["lat"] for r in B["rows"]]
    lon = [r["lon"] for r in B["rows"]]
    ax.plot(lon, lat, "-", color="tab:red", lw=1.8, label="本モデル予報")
    ax.plot(lon[0], lat[0], "k*", ms=13, label=f"発行 {B['issue_jst']} JST")
    for o in B["official"]:
        if o["circle_km"] != "":
            ax.add_patch(Circle((o["jma_lon"], o["jma_lat"]),
                                float(o["circle_km"]) / 111.32, fill=False,
                                ls="--", lw=0.9, color="tab:green"))
    jla = [o["jma_lat"] for o in B["official"]]
    jlo = [o["jma_lon"] for o in B["official"]]
    ax.plot(jlo, jla, "s--", color="tab:green", ms=4.5, lw=1.2,
            label="気象庁 諸元予報(円=確率半径)")
    bl = B["blend"]["rows"]
    ax.plot([r["lon"] for r in bl], [r["lat"] for r in bl], "-",
            color="tab:purple", lw=2.0, alpha=0.85,
            label="製品トラック(公式 + モデル延長)")
    last = bl[-1]
    if last["circle_km"] != "":
        ax.add_patch(Circle((last["lon"], last["lat"]),
                            float(last["circle_km"]) / 111.32, fill=False,
                            ls=":", lw=1.0, color="tab:purple"))
    for nm, (sla, slo) in sorted(op.stations.items()):
        ax.plot(slo, sla, "^", color="gray", ms=4.0)
        ax.annotate(nm, (slo, sla), fontsize=6.5, color="gray",
                    xytext=(2, 2), textcoords="offset points")
    ax.set_xlabel("経度 [deg]")
    ax.set_ylabel("緯度 [deg]")
    ax.grid(alpha=0.3)
    ax.set_aspect(1.0 / max(0.2, math.cos(math.radians(36.0))))
    ax.legend(fontsize=7, loc="upper left")
    ax.set_title("進路: 実況・本モデル・気象庁公式", fontsize=10)

    ax = fig.add_subplot(1, 2, 2)
    ol = [o["lead_h"] for o in B["official"]]
    oc = [o["cross_km"] for o in B["official"]]
    orad = [float(o["circle_km"]) if o["circle_km"] != "" else float("nan")
            for o in B["official"]]
    ax.fill_between(ol, 0.0, orad, color="tab:green", alpha=0.18,
                    label="気象庁 確率半径")
    ax.plot(ol, orad, "--", color="tab:green", lw=1.0)
    ax.plot(ol, oc, "o-", color="tab:red", ms=4.0, lw=1.5,
            label="本モデルとの中心距離")
    hl = [r["lead_h"] for r in C["rows"]]
    ax.plot(hl, [r["rmse_var_km"] for r in C["rows"]], ":", color="tab:blue",
            lw=1.4, label="検証済み hindcast RMSE")
    bl = B["blend"]
    ax.plot([r["lead_h"] for r in bl["rows"]],
            [max(r["cross_km"], 1e-3) for r in bl["rows"]], "-",
            color="tab:purple", lw=1.5,
            label="製品とモデルの差(=公式に固定した効果)")
    ax.plot([r["lead_h"] for r in bl["rows"] if r["circle_km"] != ""],
            [r["circle_km"] for r in bl["rows"] if r["circle_km"] != ""],
            "-", color="tab:purple", lw=1.0, alpha=0.5,
            label="製品の予報円(終端以降は外挿)")
    ax.axvline(bl["anchor_lead_h"], color="k", ls="--", lw=0.9, alpha=0.7)
    ax.set_xlabel("リードタイム [h]")
    ax.set_ylabel("距離 [km]")
    ax.set_yscale("log")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7, loc="upper left")
    ax.set_title("公式予報円と本モデルのずれ", fontsize=10)
    save(fig, "track_forecast.png",
         f"台風第{tf.TY_NUMBER[-2:]}号({tf.TC_ID}) 進路予報 "
         f"発行 {B['issue_jst']} JST — 気象庁通報のみを用いた再現")

    # -- C. hindcast skill ------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.6))
    ax[0].plot(hl, [r["rmse_persistence_km"] for r in C["rows"]], "--",
               color="gray", lw=1.2, label="永続予報")
    ax[0].plot(hl, [r["rmse_motion_persistence_km"] for r in C["rows"]], "-.",
               color="tab:green", lw=1.2, label="移動永続予報")
    ax[0].plot(hl, [r["rmse_var_km"] for r in C["rows"]], "-",
               color="tab:red", lw=1.8,
               label=f"VAR({C['shipped_order']}) 本モデル")
    ax[0].set_xlabel("リードタイム [h]")
    ax[0].set_ylabel("位置 RMSE [km]")
    ax[0].grid(alpha=0.3)
    ax[0].legend(fontsize=8)
    ax[0].set_title(f"rolling-origin 検証 {C['n_issues']} 発行", fontsize=10)
    orders = [r["order"] for r in C["order_rows"]]
    wd = 0.35
    ax[1].bar([o - wd / 2 for o in orders],
              [r["rmse_early_h"] for r in C["order_rows"]], wd,
              label=f"前半(次数選択に使用, h<={C['split_hour']:.0f})")
    ax[1].bar([o + wd / 2 for o in orders],
              [r["rmse_late_h"] for r in C["order_rows"]], wd,
              label="後半(選択の確認)")
    ax[1].axvline(C["sel_order"], color="tab:red", ls=":",
                  label=f"前半で選ばれた次数 {C['sel_order']}")
    ax[1].axvline(C["shipped_order"], color="k", ls="--",
                  label=f"出荷次数 {C['shipped_order']}")
    ax[1].set_xlabel("VAR 次数")
    ax[1].set_ylabel("平均 RMSE [km]")
    ax[1].set_xticks(orders)
    ax[1].grid(alpha=0.3, axis="y")
    ax[1].legend(fontsize=7)
    ax[1].set_title("次数の選択", fontsize=10)
    save(fig, "hindcast.png", "進路モデルの検証: 発行時刻より後の実況のみを答えとする")

    # -- D. intensity -----------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.4))
    fh = [ih + r["lead_h"] for r in B["rows"]]
    oh = [ih + o["lead_h"] for o in B["official"]]
    ax[0].plot(tr.hours, tr.pres_hPa, "-", color="tab:blue", lw=1.4,
               label="実況(気象庁)")
    ax[0].plot(fh, [r["pres_hPa"] for r in B["rows"]], "-", color="tab:red",
               lw=1.8, label="本モデル")
    ax[0].plot(oh, [o["jma_pres_hPa"] for o in B["official"]], "s--",
               color="tab:green", ms=4.0, lw=1.2, label="気象庁 諸元予報")
    ax[0].axvline(ih, color="k", ls=":", lw=0.9)
    ax[0].set_xlabel("発生からの時間 [h]")
    ax[0].set_ylabel("中心気圧 [hPa]")
    ax[0].grid(alpha=0.3)
    ax[0].legend(fontsize=8)
    ax[0].set_title(f"中心気圧 (RMSE 本モデル {D['official_pres_rmse_hPa']}"
                    f" hPa / 検証 {D['hindcast_pres_rmse_mean_hPa']} hPa)",
                    fontsize=9)
    ax[1].plot(tr.hours, tr.vmax_m_s, "-", color="tab:blue", lw=1.4,
               label="実況(気象庁)")
    ax[1].plot(fh, [r["vmax_m_s"] for r in B["rows"]], "-", color="tab:red",
               lw=1.8, label="本モデル")
    ax[1].plot(oh, [o["jma_vmax_m_s"] for o in B["official"]], "s--",
               color="tab:green", ms=4.0, lw=1.2, label="気象庁 諸元予報")
    ax[1].axvline(ih, color="k", ls=":", lw=0.9)
    ax[1].set_xlabel("発生からの時間 [h]")
    ax[1].set_ylabel("最大風速 [m/s]")
    ax[1].grid(alpha=0.3)
    ax[1].legend(fontsize=8)
    ax[1].set_title(f"最大風速 (RMSE 本モデル {D['official_vmax_rmse_m_s']}"
                    f" m/s / 検証 {D['hindcast_vmax_rmse_mean_m_s']} m/s)",
                    fontsize=9)
    save(fig, "intensity.png",
         "強度: 温帯低気圧化(北上に伴う中心気圧の緩和)までの予報")

    # -- E. precipitation --------------------------------------------------
    fig, ax = plt.subplots(2, 2, figsize=(12.0, 8.6))
    o = [r["obs_mm_h"] for r in E1["rows"]]
    m = [r["model_mm_h"] for r in E1["rows"]]
    ax[0, 0].plot([0, max(max(o), max(m))], [0, max(max(o), max(m))], "k:",
                  lw=0.9)
    ax[0, 0].plot(o, m, ".", ms=2.6, alpha=0.45, color="tab:blue")
    ax[0, 0].set_xlabel("AMeDAS 観測 1時間降水量 [mm/h]")
    ax[0, 0].set_ylabel("モデル [mm/h]")
    ax[0, 0].grid(alpha=0.3)
    ax[0, 0].set_title(
        f"E1 観測 {E1['n']} 地点時: RMSE {E1['rmse_mm_h']} mm/h, "
        f"相関 {E1['corr']}, bias {E1['bias_mm_h']}", fontsize=9)
    ph = E1["per_hour"]
    x = np.arange(len(ph))
    ax[0, 1].plot(x, [r["obs_max_mm_h"] for r in ph], "-", color="tab:blue",
                  lw=1.4, label="観測 最大")
    ax[0, 1].plot(x, [r["model_max_mm_h"] for r in ph], "-", color="tab:red",
                  lw=1.4, label="モデル 最大")
    ax[0, 1].plot(x, [r["obs_mean_mm_h"] for r in ph], "--", color="tab:blue",
                  lw=0.9, label="観測 平均")
    ax[0, 1].plot(x, [r["model_mean_mm_h"] for r in ph], "--",
                  color="tab:red", lw=0.9, label="モデル 平均")
    ticks = list(range(0, len(ph), max(1, len(ph) // 8)))
    ax[0, 1].set_xticks(ticks)
    ax[0, 1].set_xticklabels([ph[i]["stamp_jst"][3:] for i in ticks],
                             fontsize=7, rotation=30)
    ax[0, 1].set_xlabel("時刻 [JST]")
    ax[0, 1].set_ylabel("降水強度 [mm/h]")
    ax[0, 1].grid(alpha=0.3)
    ax[0, 1].legend(fontsize=7)
    ax[0, 1].set_title("E1 14地点の時間変化(実況期間)", fontsize=9)
    pop = [r["pop_pct"] for r in E3["rows"]]
    dm = [r["district_max_mm"] for r in E3["rows"]]
    ax[1, 0].plot(pop, dm, ".", ms=9, alpha=0.4, color="tab:purple")
    bx = [b["pop_band"] for b in E3["pop_bins"]]
    by = [b["model_mean_mm"] for b in E3["pop_bins"]]
    xc = [float(b.split("-")[0]) + 10.0 for b in bx]
    ax[1, 0].plot(xc, by, "k-o", ms=5, lw=1.4, label="帯別平均")
    for b, cx in zip(bx, xc):
        ax[1, 0].annotate(b, (cx, 0.0), fontsize=7, ha="center",
                          xytext=(0, 3), textcoords="offset points")
    ax[1, 0].set_xlabel("気象庁 降水確率 [%]")
    ax[1, 0].set_ylabel("モデル 6時間降水量 地区最大 [mm]")
    ax[1, 0].grid(alpha=0.3)
    ax[1, 0].legend(fontsize=7, loc="upper left")
    ax[1, 0].set_title(
        f"E3 公式降水確率 {E3['n']} 件: 相関 {E3['corr_pop_district_max']}, "
        f"順位相関 {E3['spearman_pop_district_max']}", fontsize=9)
    rr = sorted(E2["rows"], key=lambda r: -r["official_mm"])
    xi = np.arange(len(rr))
    ax[1, 1].bar(xi - 0.2, [r["official_mm"] for r in rr], 0.4,
                 color="tab:green", label="気象庁「多い所で」24時間")
    ax[1, 1].bar(xi + 0.2, [r["district_max_mm"] for r in rr], 0.4,
                 color="tab:red", label="モデル 地区最大")
    ax[1, 1].set_xticks(xi)
    ax[1, 1].set_xticklabels([r["pref"].replace("県", "").replace("都", "")
                              + r["area"] for r in rr], fontsize=6.5,
                             rotation=60, ha="right")
    ax[1, 1].set_ylabel("24時間降水量 [mm]")
    ax[1, 1].grid(alpha=0.3, axis="y")
    ax[1, 1].legend(fontsize=7)
    ax[1, 1].set_title("E2 公式の24時間ピークとの比較", fontsize=9)
    save(fig, "precipitation.png",
         "降水: AMeDAS観測・公式降水確率・公式「多い所で」との比較")

    # -- E4. the anchored rain product ------------------------------------
    ss = sorted(E4["sites"], key=lambda r: -r["anchored_total_mm"])
    fig, ax = plt.subplots(2, 1, figsize=(12.5, 7.2))
    for r, c in zip(ss[:4], ("tab:blue", "tab:orange", "tab:green",
                             "tab:purple")):
        hh = [h for h in E4["hourly"] if h["site"] == r["site"]]
        ax[0].plot([h["lead_h"] for h in hh], [h["rain_mm_h"] for h in hh],
                   "-", color=c, lw=0.8, alpha=0.55,
                   label=f"{r['site']} モデル場")
        ax[0].plot([h["lead_h"] for h in hh],
                   [h["anchored_mm_h"] for h in hh], "-", color=c, lw=1.6,
                   label=f"{r['site']} アンカー後")
    ax[0].axhline(OPS_RAIN_MAX_MM_H, color="k", ls="--", lw=0.9, alpha=0.7)
    ax[0].text(0.5, OPS_RAIN_MAX_MM_H,
               f" 運航限界 {OPS_RAIN_MAX_MM_H:.0f} mm/h", fontsize=7,
               va="bottom")
    ax[0].set_xlabel("発行からのリードタイム [h]", fontsize=8)
    ax[0].set_ylabel("1時間降水量 [mm/h]", fontsize=8)
    ax[0].grid(alpha=0.3)
    ax[0].legend(fontsize=7, ncol=2)
    ax[0].set_title("E4 合計の多い上位 4 地点(細線=モデル場、"
                    "太線=公式「多い所で」にアンカーした製品)", fontsize=9)
    xi = np.arange(len(ss))
    ax[1].bar(xi - 0.2, [r["total_mm"] for r in ss], 0.4, color="tab:blue",
              label="モデル場")
    ax[1].bar(xi + 0.2, [r["anchored_total_mm"] for r in ss], 0.4,
              color="tab:orange", label="アンカー後(製品)")
    off = [(i, r["official_24h_mm"]) for i, r in enumerate(ss)
           if r["official_24h_mm"] != ""]
    if off:
        ax[1].plot([i for i, _ in off], [v for _, v in off], "k_", ms=11,
                   mew=2.2, label="公式 24h「多い所で」(窓がある地点)")
    ax[1].set_xticks(xi)
    ax[1].set_xticklabels([r["site"] for r in ss], fontsize=7, rotation=45,
                          ha="right")
    ax[1].set_ylabel("予報期間の合計降水量 [mm]", fontsize=8)
    ax[1].grid(alpha=0.3, axis="y")
    ax[1].legend(fontsize=7)
    ax[1].set_title(f"E4 {E4['n_sites']} 地点の予報合計: 最大 "
                    f"{E4['anchored_max_total_mm']:.0f} mm"
                    f"({E4['anchored_wettest_site']}、アンカー後)", fontsize=9)
    save(fig, "rain_forecast.png",
         "降水予報: 気象庁「多い所で」にアンカーした製品")

    # -- F. operations ------------------------------------------------------
    sites = sorted(op.stations)
    leads = sorted({r["lead_h"] for r in F["rows"]})
    grid = np.full((len(sites), len(leads)), np.nan)
    for r in F["rows"]:
        grid[sites.index(r["site"]), leads.index(r["lead_h"])] = \
            1.0 if r["go"] else 0.0
    vleads = sorted({r["lead_h"] for r in F["var_rows"]})
    vgrid = np.full((len(sites), len(vleads)), np.nan)
    for r in F["var_rows"]:
        vgrid[sites.index(r["site"]), vleads.index(r["lead_h"])] = \
            1.0 if r["go"] else 0.0
    fig = plt.figure(figsize=(12.5, 8.0))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.5, 1.5, 1.2], hspace=0.55)
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(["tab:red", "tab:green"])
    a0 = fig.add_subplot(gs[0])
    a0.imshow(grid, aspect="auto", cmap=cmap, vmin=0, vmax=1,
              origin="lower", extent=(leads[0], leads[-1], -0.5,
                                      len(sites) - 0.5))
    a0.set_yticks(range(len(sites)))
    a0.set_yticklabels(sites, fontsize=8)
    a0.set_title(f"製品トラック(気象庁予報 + モデル延長)による運航可否 "
                 f"(緑=可, 赤=不可, 可 {F['go_frac'] * 100:.0f}%)", fontsize=10)
    a0.set_xlabel("リードタイム [h]", fontsize=8)
    a0.axvline(F["anchor_lead_h"], color="k", ls="--", lw=0.9, alpha=0.7)
    a0.text(F["anchor_lead_h"] + 0.6, len(sites) - 1.2,
            f"気象庁予報の終端 +{F['anchor_lead_h']:.0f}h",
            fontsize=7, va="top")
    a1 = fig.add_subplot(gs[1])
    a1.imshow(vgrid, aspect="auto", cmap=cmap, vmin=0, vmax=1,
              origin="lower", extent=(vleads[0], vleads[-1], -0.5,
                                      len(sites) - 0.5))
    a1.set_yticks(range(len(sites)))
    a1.set_yticklabels(sites, fontsize=8)
    a1.axvline(F["anchor_lead_h"], color="k", ls="--", lw=0.9, alpha=0.7)
    a1.set_title(f"自由走行の VAR モデル(気象庁予報に固定しない)を同じ"
                 f"判定基準にかけた場合 (可 {F['var_go_frac'] * 100:.0f}%, "
                 f"製品との一致 {F['agreement_frac'] * 100:.0f}%)", fontsize=10)
    a1.set_xlabel("リードタイム [h]", fontsize=8)
    a2 = fig.add_subplot(gs[2])
    a2.plot([r["lead_h"] for r in F["per_lead"]],
            [r["go_sites"] for r in F["per_lead"]], "o-", color="tab:red",
            ms=3.5, lw=1.4, label="製品: 運航可の地点数")
    a2.plot([r["lead_h"] for r in F["per_lead"] if r["var_n_sites"]],
            [r["var_go_sites"] for r in F["per_lead"] if r["var_n_sites"]],
            "s--", color="tab:green", ms=3.5, lw=1.2,
            label="自由走行 VAR: 運航可の地点数")
    a2.axvline(F["anchor_lead_h"], color="k", ls="--", lw=0.9, alpha=0.7)
    a2b = a2.twinx()
    wx = {}
    for r in F["rows"]:
        wx[r["lead_h"]] = max(wx.get(r["lead_h"], 0.0), r["wind_m_s"])
    a2b.plot(sorted(wx), [wx[k] for k in sorted(wx)], ":", color="tab:blue",
             lw=1.4, label="全地点中の最大平均風速")
    a2b.axhline(OPS_WIND_MAX_M_S, color="tab:blue", ls="-", lw=0.8, alpha=0.6)
    a2b.set_ylabel("最大平均風速 [m/s]", fontsize=8, color="tab:blue")
    a2.set_xlabel("リードタイム [h]", fontsize=8)
    a2.set_ylabel("運航可の地点数 (14地点中)", fontsize=8)
    a2.set_ylim(-0.5, len(sites) + 0.5)
    a2.grid(alpha=0.3)
    h1, l1 = a2.get_legend_handles_labels()
    h2, l2 = a2b.get_legend_handles_labels()
    a2.legend(h1 + h2, l1 + l2, fontsize=7, loc="lower left")
    a2.set_title(f"判定の内訳 (限界値の出典: {F['limits_source']}, "
                 f"風 {OPS_WIND_MAX_M_S:.1f} m/s, 突風 "
                 f"{OPS_GUST_MAX_M_S:.1f} m/s, 降水 {OPS_RAIN_MAX_MM_H:.0f}"
                 f" mm/h)", fontsize=9)
    save(fig, "operations.png",
         "運航可否: 製品トラック(気象庁予報 + モデル延長)と自由走行 VAR の"
         "並列判定", tight=False)
    return [Path(p).name for p in names]


# ----------------------------------------------------------------------
#  report
# ----------------------------------------------------------------------
def num(v, nd=1, dash="―"):
    """Format a possibly missing / non-finite table cell."""
    if v is None or v == "":
        return dash
    try:
        x = float(v)
    except (TypeError, ValueError):
        return str(v)
    return f"{x:.{nd}f}" if math.isfinite(x) else dash


CSV_FILES = [
    "snapshot_fixes.csv", "track_forecast.csv", "track_product.csv",
    "track_vs_official.csv",
    "hindcast.csv", "order_selection.csv", "intensity_official.csv",
    "rain_obs.csv", "rain_obs_station.csv", "rain_peaks_official.csv",
    "rain_pop_official.csv", "rain_sites.csv", "rain_sites_6h.csv",
    "rain_sites_hourly.csv", "ops_decisions.csv", "ops_decisions_var.csv",
    "ops_site.csv", "ops_lead.csv", "ops_disagree.csv",
    "bulletin_warnings.csv", "bulletin_outlooks.csv",
]


def write_report(cfg: StudyConfig, snap, op: OperationalForecast, A: dict,
                 B: dict, C: dict, D: dict, E1: dict, E2: dict, E3: dict,
                 E4: dict, F: dict, G: dict, plot_names: list,
                 runtime_s: float) -> None:
    """REPORT.md: what was forecast, what verified, and what it means."""
    man = A["manifest"]
    an = A["analysis_state"]
    bl = B["blend"]
    L = []
    A_ = L.append
    A_("# 台風25号(2026年・ドゥージェン)実戦予報レポート")
    A_("")
    A_(f"`{cfg.out}` / 発行 {B['issue_jst']} JST"
        f"({B['issue_utc']} UTC)/ 予報ホライズン {cfg.horizon:.0f} h")
    A_("")
    A_("気象庁の気象通報(電文本文・諸元 XML・府県天気予報 R1 XML)だけを入力に、"
        "台風25号(`TC2630`)の進路・強度・降水を実運用の形で予報し、"
        "気象庁自身の公式予報と AMeDAS 観測に突き合わせた。実行時に"
        "ネットワークへは行かない: 入力は `fetch_typhoon_dujuan.py` が取得して"
        f"`{cfg.data}` に固定したスナップショットのみで、結果は再現する。")
    A_("")
    A_("## A. 入力データと来歴")
    A_("")
    A_(f"- 取得時刻: **{A['fetched_utc']} UTC**。台風電文 {A['bulletins']} 報"
        f"(本文から読めた位置 {G['n_fixes']} 点、うち実況 "
        f"{G['n_analysis_fixes']} 点;解析トラック {A['track_n']} 点)、"
        f"府県天気予報 R1 XML "
        f"{man.get('prefecture_R1_xml', 0)} 府県、AMeDAS "
        f"{A['amedas']['records']} レコード。")
    A_(f"- 台風発生: {A['genesis_jst']} JST({A['genesis_utc']} UTC)。"
        f"解析トラックは {A['track_first_utc']} 〜 {A['track_last_utc']} UTC"
        f"(発生後 {A['track_hours']:.0f} h)、内訳 {A['track_kinds']}。"
        f"移動距離 {A['track_dist_km']:.0f} km・平均 "
        f"{A['track_speed_km_h']:.0f} km/h。")
    A_(f"- 最終実況({A['track_last_jst']} JST): 北緯 {an['lat']:.1f} 度・"
        f"東経 {an['lon']:.1f} 度、中心気圧 **{an['pres_hPa']:.0f} hPa**、"
        f"最大風速 **{an['vmax_m_s']:.0f} m/s**(瞬間 {an['gust_m_s']:.0f} m/s)、"
        f"暴風域 {an['storm']}、強風域 {an['gale']}。"
        f"生涯最強は {an['min_pres_hPa']:.0f} hPa / {an['max_vmax_m_s']:.0f} m/s。")
    tv = A["text_vs_track"]
    A_(f"- 通報本文が述べた位置と解析トラック配列の一致: {tv['n']} 点、"
        f"最大 {tv['max_dist_km']:.1f} km・平均 {tv['mean_dist_km']:.2f} km"
        "(電文テキストと諸元 XML が同じ台風を指していることの確認)。")
    am = A["amedas"]
    A_(f"- AMeDAS(1 時間値): {am['stations']} 地点 × {am['stamps']} 時刻 = "
        f"{am['records']} レコード({am['first_utc']} 〜 {am['last_utc']} UTC)。"
        "進路沿いの 14 地点(高知・大阪・静岡・東京・伊豆諸島・千葉・茨城・"
        "宮城・岩手)。")
    A_(f"- 府県天気予報(R1)の 6 時間降水確率: {E3['n']} レコード / "
        f"{E3['areas']} 地域 / {E3['windows']} 時間窓"
        f"({E3['window_first_jst']} 〜 {E3['window_last_jst']} JST)。"
        "通報の「雨の見通し」(「多い所で」)は "
        f"{G['n_outlook_blocks']} ブロック / {G['n_outlooks']} 報。")
    A_(f"- ガスト比(最大瞬間風速 ÷ 最大風速)= {A['gust_ratio']:.2f}"
        "(電文の実況値から求めた台風固有の値。既定 "
        f"{tf.GUST_RATIO_FALLBACK:.2f} の代わりに使用)。")
    A_("- 出典(すべて気象庁):")
    for name, url in sorted((man.get("sources") or {}).items()):
        A_(f"  - `{name}` — {url}")
    A_("")
    A_("## B. 進路予報(実運用)")
    A_("")
    A_(f"最後の解析実況 {B['issue_jst']} JST(発生後 {B['issue_hour']:.0f} h)を"
        f"発行時刻とし、それ以前の位置だけで VAR({B['order']}) 運動モデルを"
        f"当てはめて +{cfg.horizon:.0f} h まで予報した"
        f"(学習 {B['train_points']} 点、直近窓 {B['window_hours']:.0f} h、"
        f"残差 RMS {B['resid_rms_km']:.1f} km、収縮率 "
        f"{B['shrink_scale']:.2f})。予報終端: "
        f"{B['end_state']['stamp_jst']} JST に北緯 "
        f"{B['end_state']['lat']:.1f} 度・東経 {B['end_state']['lon']:.1f} 度、"
        f"中心気圧 {B['end_state']['pres_hPa']:.0f} hPa、"
        f"最大風速 {B['end_state']['vmax_m_s']:.0f} m/s。"
        f"平均移動速度 {B['mean_speed_km_h']:.0f} km/h。")
    A_("")
    A_("同一発行時刻の気象庁公式予報(諸元)との差。`圏内` は予報円"
        "(確率圏)の中に本モデルの位置が入るか。")
    A_("")
    A_("| lead | 時刻 JST | 気象庁 緯度/経度 | 本モデル 緯度/経度 | 距離 km "
       "| 予報円 km | 圏内 |")
    A_("|---|---|---|---|---|---|---|")
    for r in B["official"]:
        A_(f"| +{r['lead_h']:.0f} h | {r['stamp_jst']} "
          f"| {r['jma_lat']:.1f} / {r['jma_lon']:.1f} "
          f"| {r['model_lat']:.1f} / {r['model_lon']:.1f} "
          f"| **{r['cross_km']:.0f}** | {num(r['circle_km'], 0)} "
          f"| {'' if r['circle_km'] == '' else ('○' if r['inside_circle'] else '×')} |")
    ic = B["inside_circle"]
    A_("")
    n_in = sum(1 for x in B["official"]
               if x["circle_km"] != "" and x["inside_circle"])
    A_(f"- 予報円と比べられる {ic['n']} 時刻のうち円内だったのは {n_in} 時刻"
        f"(+{num(ic['last_lead_inside_h'], 0)} h まで)。"
        f"円中心からの最大偏差 {ic['max_cross_km']:.0f} km"
        f"(最大円半径 {G['max_circle_km']:.0f} km)。")
    A_(f"- 公式予報の最長リードは +{G['max_official_lead_h']:.1f} h"
        f"(以後は温帯低気圧として扱うため台風諸元が出ない)。本モデルは "
        f"+{cfg.horizon:.0f} h まで機械的に外挿する。")
    A_("")
    A_("### B2. 運用に出す製品トラック(公式予報 + モデル延長)")
    A_("")
    A_("実運用では上の自由走行モデルをそのまま使わない。気象庁が諸元で出す"
        f"予報点 {bl['n_official_points']} 個を +{bl['anchor_lead_h']:.0f} h"
        f"({bl['anchor_stamp_jst']} JST)まで**そのまま**採用し、それ以降に"
        "モデルの 3 時間変位を接続したものを製品トラックとする(接続点で"
        "飛ぶことがないよう、モデルは最終の公式点に再固定する)。強度も公式点の"
        "内側は公式値(温帯低気圧化を含む)、外側はモデル値。予報円は公式が"
        f"実値を出す最後の点までを使い、以降はその区間の成長率 "
        f"{bl['circle_rate_km_h']:.1f} km/h で外挿して延長する。"
        f"3 時間刻み {len(bl['rows'])} 行のうち公式 {bl['n_official_rows']} 行・"
        f"モデル延長 {bl['n_model_rows']} 行。")
    A_("")
    A_(f"- 製品終端: {bl['end_state']['stamp_jst']} JST に北緯 "
        f"{bl['end_state']['lat']:.1f} 度・東経 "
        f"{bl['end_state']['lon']:.1f} 度、中心気圧 "
        f"{bl['end_state']['pres_hPa']:.0f} hPa、最大風速 "
        f"{bl['end_state']['vmax_m_s']:.0f} m/s。")
    A_(f"- 公式に固定した効果: 製品と自由走行モデルの位置差は "
        f"+{bl['anchor_lead_h']:.0f} h で {bl['cross_at_horizon_km']:.0f} km"
        f"(最大 {bl['max_cross_km']:.0f} km、+{bl['max_cross_lead_h']:.0f} h)。"
        "以降の E4・F はすべてこの製品トラックから作る。")
    A_("")
    A_("| lead | 時刻 JST | 出典 | 緯度/経度 | 気圧 hPa | 最大風速 m/s "
       "| 突風 m/s | 移動 km/h | モデルとの差 km | 予報円 km(出典) |")
    A_("|---|---|---|---|---|---|---|---|---|---|")
    for r in bl["rows"]:
        A_(f"| +{r['lead_h']:.0f} h | {r['stamp_jst']} | {r['src']} "
          f"| {r['lat']:.2f} / {r['lon']:.2f} | {r['pres_hPa']:.0f} "
          f"| {r['vmax_m_s']:.0f} | {r['gust_m_s']:.0f} "
          f"| {r['speed_km_h']:.0f} | {r['cross_km']:.0f} "
          f"| {num(r['circle_km'], 0)}({r['circle_src']}) |")
    A_("")
    A_("## C. 進路モデルのヒンドキャスト検証(観測真値に対して)")
    A_("")
    A_(f"発行時刻を {cfg.hind_first:.0f} h から {cfg.hind_stride:.0f} h 刻みで"
        f"動かし、各予報を**発行時刻より前の実況だけ**で学習させて"
        f"解析トラックに当てた({C['n_issues']} 件、"
        f"{C['first_issue_utc']} 〜 {C['last_issue_utc']} UTC)。"
        "基準は永続予報と移動永続予報。")
    A_("")
    A_("| lead | n | RMSE VAR km | RMSE 永続 km | RMSE 移動永続 km "
       "| MAE km | 最大 km | 対永続スキル | 対移動永続スキル |")
    A_("|---|---|---|---|---|---|---|---|---|")
    for r in C["rows"]:
        A_(f"| +{r['lead_h']} h | {r['n']} | **{num(r['rmse_var_km'])}** "
          f"| {num(r['rmse_persistence_km'])} "
          f"| {num(r['rmse_motion_persistence_km'])} "
          f"| {num(r['mae_var_km'])} | {num(r['max_var_km'])} "
          f"| {num(r['skill_vs_persistence'], 2)} "
          f"| {num(r['skill_vs_motion'], 2)} |")
    A_("")
    A_(f"全リード平均 RMSE: VAR **{C['rmse_mean_var_km']:.1f} km** / 永続 "
        f"{C['rmse_mean_persistence_km']:.1f} km / 移動永続 "
        f"{C['rmse_mean_motion_km']:.1f} km"
        f"(プールした RMSE {C['rmse_pooled_var_km']:.1f} km)。")
    A_("")
    A_("次数は前期窓(発行時刻の中央値 "
        f"{C['split_hour']:.0f} h 以前)だけで選び、後期窓で確認した:")
    A_("")
    A_("| order | RMSE 前期 km | RMSE 後期 km | RMSE 全件 km |")
    A_("|---|---|---|---|")
    for r in C["order_rows"]:
        mark = " ←前期最良" if r["order"] == C["sel_order"] else ""
        ship = " (出荷値)" if r["order"] == C["shipped_order"] else ""
        A_(f"| VAR({r['order']}){ship}{mark} | {num(r['rmse_early_h'])} "
          f"| {num(r['rmse_late_h'])} | {num(r['rmse_all_h'])} |")
    A_("")
    if C["sel_order"] == C["shipped_order"]:
        A_(f"→ 前期窓が選ぶ次数は出荷値 VAR({C['shipped_order']}) と一致した。")
    else:
        A_(f"→ 前期窓は VAR({C['sel_order']}) を選ぶが、出荷値は "
            f"VAR({C['shipped_order']})。後期窓の RMSE も併せて上表で確認すること"
            "(1 台風の {0} 件での選択なので過学習の余地がある)。".format(
                C["n_issues"]))
    A_("")
    A_("## D. 強度予報(中心気圧・最大風速)")
    A_("")
    A_("強度は運動モデルと同じ枠組みで予報する: 中心気圧は直近の"
        f"気圧傾向({D['hindcast_pres_rmse_mean_hPa']:.1f} hPa RMSE)を"
        "持続させ、温帯低気圧化する緯度帯に入れば標準気圧へ緩和する"
        f"(北緯 {D['et_lat_start']:.0f}〜{D['et_lat_end']:.0f} 度、緩和先 "
        f"{D['pres_relax_hPa']:.0f} hPa)。最大風速は気圧-風速関係から求める。")
    A_("")
    A_("ヒンドキャスト(観測真値に対する RMSE / バイアス、C と同じ発行時刻):")
    A_("")
    A_("| lead | 気圧 RMSE hPa | 気圧バイアス hPa | 風速 RMSE m/s "
       "| 風速バイアス m/s |")
    A_("|---|---|---|---|---|")
    for r in C["rows"]:
        k = r["lead_h"]
        A_(f"| +{k} h | {num(D['hindcast_pres_rmse_hPa'][k], 2)} "
          f"| {num(D['hindcast_pres_bias_hPa'][k], 2)} "
          f"| {num(D['hindcast_vmax_rmse_m_s'][k], 2)} "
          f"| {num(D['hindcast_vmax_bias_m_s'][k], 2)} |")
    A_("")
    A_(f"全リード平均: 気圧 RMSE {D['hindcast_pres_rmse_mean_hPa']:.2f} hPa、"
        f"最大風速 RMSE {D['hindcast_vmax_rmse_mean_m_s']:.2f} m/s。")
    A_("")
    A_("公式予報(諸元)との差。バイアス = 本モデル − 気象庁。")
    A_("")
    A_("| lead | 時刻 JST | 階級 | 気圧 気象庁/本モデル/差 hPa "
       "| 最大風速 気象庁/本モデル/差 m/s | 暴風域 |")
    A_("|---|---|---|---|---|---|")
    for r in D["official"]:
        A_(f"| +{r['lead_h']:.0f} h | {r['stamp_jst']} | {r['category'] or '―'} "
          f"| {num(r['jma_pres_hPa'], 0)} / {num(r['model_pres_hPa'])} "
          f"/ {num(r['d_pres_hPa'])} "
          f"| {num(r['jma_vmax_m_s'], 0)} / {num(r['model_vmax_m_s'])} "
          f"/ {num(r['d_vmax_m_s'])} | {r['storm']} |")
    A_("")
    A_(f"- 公式値との RMSE: 気圧 {D['official_pres_rmse_hPa']:.1f} hPa、"
        f"最大風速 {D['official_vmax_rmse_m_s']:.1f} m/s。"
        f"実況 {D['analysis_pres_hPa']:.0f} hPa から予報終端 "
        f"{D['forecast_pres_end_hPa']:.0f} hPa へ。")
    A_("- 注意: 予報終端は気象庁が温帯低気圧化を見込む時刻より先であり、"
        "そこでの値は緩和則の外挿にすぎない。強度の検証真値は C の"
        "ヒンドキャスト(実況)であって、公式予報との差は「別の予報との差」。")
    A_("")
    A_("## E. 降水予報")
    A_("")
    A_("### E1. パラメトリック降水場の検証(AMeDAS 観測に対して)")
    A_("")
    A_(f"降水は中心気圧から振幅を、強風域半径から空間スケールを決める解析的な"
        f"渦フィールド(`rain_rate_mm_h`、地形係数を含む)。台風25号の "
        f"AMeDAS {E1['n']} 観測地点時刻(実況側 {E1['n_analysis']}、"
        f"予報側 {E1['n_forecast']};{E1['stations']} 地点 × {E1['hours']} 時刻)"
        "にそのまま当てた。")
    A_("")
    A_("**注意(インサンプル)**: `RAIN_SCALE` / `RAIN_DECAY_FRAC` / "
        "`RAIN_LEAD_FRAC` の 3 定数は、まさにこの AMeDAS 地点時刻に対して"
        "当てた値である。したがって E1 の指標は**当てはまりの良さ**であって"
        "未知データに対する予報スキルではない(楽観側にバイアスする)。"
        "定数を決めるのに使っていない独立の検証は E2(気象庁の「多い所で」)"
        "と E3(府県天気予報の降水確率)であり、降水の絶対量を語るときは"
        "そちらを根拠にする。地点別の過不足(内陸の過大予想など)は"
        "インサンプルでも残っているので、定数の再当てはめでは直らない"
        "構造的な限界を示している。")
    A_("")
    w = E1["wet"]
    A_(f"- 1 時間降水: 平均 観測 {E1['obs_mean_mm_h']:.2f} mm/h・"
        f"モデル {E1['model_mean_mm_h']:.2f} mm/h、バイアス "
        f"{E1['bias_mm_h']:+.2f} mm/h、RMSE **{E1['rmse_mm_h']:.2f} mm/h**、"
        f"MAE {E1['mae_mm_h']:.2f} mm/h、相関 {E1['corr']:.2f}"
        f"(順位相関 {E1['spearman']:.2f})。")
    A_(f"- 降水あり/なし(≥{WET_MM_H} mm/h、気象庁の計数閾値): POD "
        f"{num(w['pod'], 2)}・FAR {num(w['far'], 2)}・CSI {num(w['csi'], 2)}・"
        f"頻度バイアス {num(w['freq_bias'], 2)}(観測の降水率 "
        f"{w['obs_wet_frac'] * 100:.0f}%、モデル {w['model_wet_frac'] * 100:.0f}%)。"
        f"≥1 mm/h では CSI {num(E1['wet_1mm']['csi'], 2)}、"
        f"≥10 mm/h では CSI {num(E1['wet_10mm']['csi'], 2)}。")
    A_(f"- 強い降水(観測 ≥10 mm/h、{E1['heavy_n']} 件)の捕捉率 POD "
        f"{num(E1['heavy_pod'], 2)}、その平均過小評価 "
        f"{num(E1['heavy_underestimate_mm_h'], 1)} mm/h。"
        "分布の裾を意図的に抑えた設計(振幅上限 "
        f"{tf.RAIN_MAX_MM_H:.0f} mm/h)なので、豪雨の絶対量は過小になる。")
    A_("")
    A_("地点別(観測の最大 1 時間値とモデルの対応):")
    A_("")
    A_("| 地点 | 地域コード | n | 観測 max mm/h | モデル max mm/h "
       "| 観測合計 mm | モデル合計 mm | RMSE | バイアス | CSI |")
    A_("|---|---|---|---|---|---|---|---|---|---|")
    for r in E1["per_station"]:
        A_(f"| {r['station']} | {r['area'] or '―'} | {r['n']} "
          f"| {r['obs_max_mm_h']:.1f} | {r['model_max_mm_h']:.1f} "
          f"| {r['obs_total_mm']:.0f} | {r['model_total_mm']:.0f} "
          f"| {r['rmse_mm_h']:.2f} | {r['bias_mm_h']:+.2f} "
          f"| {num(r['csi_wet'], 2)} |")
    A_("")
    wv = E1["wind"]
    A_("同じ渦場から出る風(運航可否の入力)も観測と突き合わせた:")
    A_("")
    A_(f"- 海上風のまま(係数なし): RMSE {wv['rmse_marine_m_s']:.2f} m/s"
        f"(バイアス {wv['bias_marine_m_s']:+.2f})、相関 {wv['corr']:.2f}、"
        f"風向の平均絶対差 {num(wv['dir_mae_deg'], 0)} 度"
        f"(バイアス {num(wv['dir_bias_deg'], 0)} 度、n={wv['dir_n']})。")
    A_(f"- 陸上曝露係数を最小二乗(原点通過)で当てると k = "
        f"{wv['land_factor']:.2f} で RMSE {wv['rmse_land_m_s']:.2f} m/s まで"
        f"落ちる(affine 2 係数でも RMSE {wv['rmse_land_m_s']:.2f} m/s から"
        "改善せず、負の風速が出るので採らない)。")
    A_(f"- ただしこの係数は**ピークを潰す**: 観測風速の大きい上位 "
        f"{wv['peak_n']} 件(観測平均 {wv['peak_obs_m_s']:.1f} m/s)で、"
        f"海上風のモデル値は平均 {wv['peak_model_m_s']:.1f} m/s とほぼ妥当"
        f"なのに対し、k を掛けると {wv['peak_calibrated_m_s']:.1f} m/s まで"
        "落ちる。強い観測風は渦そのものではなく環境場(前線・気圧傾度)が"
        "支配する時刻に起きており、渦のみの場にはそれが含まれないため。"
        "よって k は運航判定には使わず感度としてだけ示す(F 参照)。")
    A_("")
    A_("### E2. 気象庁の「多い所で」との比較")
    A_("")
    A_(f"通報が府県別に公表する雨量(「多い所で」)を、同じ時間窓で本モデルが"
        f"積分した値と比べた({E2['districts']} 地域 × "
        f"{len(E2['windows'])} 窓 = {E2['n']} 行、"
        f"{len(E2['prefectures'])} 府県にまたがる)。"
        "「多い所で」は地域内の最大値なので、モデル側は観測点値と"
        f"その周辺 ±{DISTRICT_HALF_DEG} 度グリッドの最大値の両方を出す。")
    A_("")
    A_(f"- 比(モデル ÷ 公式)の中央値: 地点値 {num(E2['median_ratio'], 2)}、"
        f"地域最大値 {num(E2['median_ratio_district_max'], 2)}。"
        f"公式値に届いた地域窓は {E2['n_district_max_reaches_official']}"
        f" / {len([r for r in E2['rows'] if r['ratio_max'] != ''])}。")
    A_(f"- 1 時間ピークの比の中央値 {num(E2['peak_ratio_median'], 2)}。")
    A_(f"- 参考: 上の表は府県天気予報(R1 XML)の値だが、気象解説情報の"
        f"「雨の見通し」({G['n_outlook_blocks']} ブロック)にはそれより大きい"
        f"値も出る。公表された最大は 1 時間 "
        f"{G['peak_mm_h_published']:.0f} mm = {G['peak_1h_where']}、"
        f"24 時間 {G['peak_24h_mm_published']:.0f} mm = "
        f"{G['peak_24h_where']}。発行時刻が上表より前なので、"
        "予報が後から下方修正された分を含む。")
    A_(f"- 公式の最大雨量は **{E2['official_max_where']} の "
        f"{E2['official_max_mm']:.0f} mm**。同じ窓のモデル値は地点 "
        f"{num(E2['model_mm_at_official_max'], 0)} mm・地域最大 "
        f"{num(E2['district_max_at_official_max'], 0)} mm。"
        f"モデル全体の最大は {E2['model_max_mm']:.0f} mm。")
    A_("")
    top = sorted(E2["rows"], key=lambda r: -r["official_mm"])[:10]
    A_("公式雨量の大きい上位 10 行:")
    A_("")
    A_("| 府県・地域 | 窓 JST | 公式 mm | モデル地点 mm | 地域最大 mm | 比 "
       "| 1h 公式/モデル mm |")
    A_("|---|---|---|---|---|---|---|")
    for r in top:
        A_(f"| {r['pref']} {r['area']} | {r['window_jst']} "
          f"| {r['official_mm']:.0f} | {r['model_mm']:.0f} "
          f"| {r['district_max_mm']:.0f} | {num(r['ratio_max'], 2)} "
          f"| {num(r['official_peak_mm_h'], 0)} / "
          f"{r['district_peak_mm_h']:.0f} |")
    A_("")
    A_("### E3. 公式 6 時間降水確率(R1)との比較")
    A_("")
    A_(f"府県天気予報の降水確率 {E3['n']} 件(9 府県)を、同じ窓で積分した"
        "モデル雨量と突き合わせた。確率と決定値は直接比べられないので、"
        "連続量(順位が合うか)と決定(公式 POP ≥ 50% 対 モデル ≥ 1 mm)の"
        "2 通りで見る。")
    A_("")
    A_(f"- 相関: POP と地域最大雨量 r = {E3['corr_pop_district_max']:.2f}"
        f"(順位 {E3['spearman_pop_district_max']:.2f})、"
        f"POP と地点雨量 r = {E3['corr_pop_point']:.2f}。")
    A_(f"- 決定: 一致率 {num(E3['agree_frac'], 2)}、POD {num(E3['pod'], 2)}、"
        f"FAR {num(E3['far'], 2)}、CSI {num(E3['csi'], 2)}"
        f"(命中 {E3['hits']}・見逃し {E3['misses']}・空振り "
        f"{E3['false_alarms']})。")
    A_("")
    A_("| POP 帯 % | n | モデル地域最大 平均 mm | 中央値 mm | 降水時間率 |")
    A_("|---|---|---|---|---|")
    for r in E3["pop_bins"]:
        A_(f"| {r['pop_band']} | {r['n']} | {num(r['model_mean_mm'])} "
          f"| {num(r['model_median_mm'])} | {num(r['wet_frac_mean'], 2)} |")
    A_("")
    A_("時間窓ごと:")
    A_("")
    A_("| 窓 開始 JST | n | POP 平均 % | モデル地域最大 平均/最大 mm | 一致率 |")
    A_("|---|---|---|---|---|")
    for r in E3["per_window"]:
        A_(f"| {r['window_jst']} | {r['n']} | {r['pop_mean']:.0f} "
          f"| {r['model_mean_mm']:.1f} / {r['model_max_mm']:.1f} "
          f"| {r['agree']:.2f} |")
    A_("")
    A_("### E4. 運航地点の予報雨量")
    A_("")
    A_(f"B2 の製品トラックを発行 {E4['issue_jst']} JST から "
        f"+{E4['horizon_h']:.0f} h、1 時間刻みで積分した {E4['n_sites']} 地点"
        f"予報。モデル場そのままでは最も濡れるのは **{E4['wettest_site']}**"
        f"(合計 {E4['max_total_mm']:.0f} mm)、最大 1 時間値 "
        f"{E4['max_peak_mm_h']:.1f} mm/h。")
    A_("")
    an4 = E4["anchor"]
    A_(f"**公式にアンカーした製品**: E2 が示すとおり解析的な渦場は気象庁の"
        f"「多い所で」の半分ほどしか降らせない(地域最大値比の中央値 "
        f"{num(E2['median_ratio_district_max'], 2)} 倍)。そこで地域ごとに、"
        "気象庁が値を出した窓での地域最大値を公式値に一致させる倍率を 1 本"
        f"決めて全体に掛ける({an4['n_codes']} 地域・{an4['n_scales']} 窓から"
        f"決定、倍率 {an4['min_scale']:.2f}〜{an4['max_scale']:.2f}、中央値 "
        f"{an4['median_scale']:.2f}。値のない地域は中央値を流用)。役割分担は"
        "**量が気象庁、時間と空間の分布がモデル**(気象庁は分布を公表しない)。"
        "倍率は 1 地域 1 本を全ホライズン一定で当てる**仮定であって校正では"
        "ない**ことに注意。")
    A_("")
    A_(f"アンカー後: 最も濡れるのは **{E4['anchored_wettest_site']}**"
        f"(合計 {E4['anchored_max_total_mm']:.0f} mm)、最大 1 時間値 "
        f"{E4['anchored_max_peak_mm_h']:.1f} mm/h。6 時間値は "
        "`rain_sites_6h.csv`、1 時間値は `rain_sites_hourly.csv` に"
        "(どちらもアンカー前/後の両列を持つ)。")
    A_("")
    A_("| 地点 | 合計 mm | 発行後 24h mm | ピーク mm/h(時刻 JST, lead) "
       "| ≥1mm 時間 | ≥10mm 時間 | 降り始め/終わり lead h "
       "| 公式窓の公式 24h mm / モデル mm | アンカー倍率(出典) "
       "| アンカー後 合計 mm | アンカー後 24h mm | アンカー後ピーク mm/h |")
    A_("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in E4["sites"]:
        A_(f"| {r['site']} | {r['total_mm']:.0f} "
          f"| {r['first_24h_from_issue_mm']:.0f} "
          f"| {r['peak_mm_h']:.1f}({r['peak_stamp_jst']}, "
          f"+{r['peak_lead_h']:.0f}h) | {r['hours_ge_1mm']:.0f} "
          f"| {r['hours_ge_10mm']:.0f} "
          f"| {num(r['rain_start_lead_h'], 0)} / {num(r['rain_end_lead_h'], 0)} "
          f"| {num(r['official_24h_mm'], 0)} / "
          f"{num(r['model_official_window_mm'], 0)} "
          f"| {r['anchor_scale']:.2f}({r['anchor_src'] or '―'}) "
          f"| {r['anchored_total_mm']:.0f} "
          f"| {r['anchored_first_24h_mm']:.0f} "
          f"| {r['anchored_peak_mm_h']:.1f} |")
    A_("")
    lim = F["limits"]
    A_("## F. 運航可否(製品トラックと自由走行モデルの並列判定)")
    A_("")
    A_("このリポジトリには運航限界の規定がないので、限界値はリポジトリ自身の"
        f"天気プリセットから取った(**出典: `{F['limits_source']}`**、"
        "『storm』ではない最も厳しいプリセットが境界):")
    A_("")
    A_(f"- 平均風速 ≤ {lim['wind_m_s']:.2f} m/s、1 時間ピーク突風 ≤ "
        f"{lim['gust_m_s']:.2f} m/s(= 平均 + {GUST_PEAK_FACTOR:.2f} × "
        "gust_rms、シミュレータのガスト過程の解析的ピーク。乱数は引かない)、"
        f"降水 ≤ {lim['rain_mm_h']:.1f} mm/h、視程 ≥ "
        f"{lim['visibility_m']:.0f} m、雷リスク < "
        f"{lim['lightning_risk']:.1f}(`Weather.condition_label` が storm と"
        f"呼ぶ閾値)、推力余裕 ≥ {lim['thrust_margin']:.2f}。")
    A_(f"- 機体は既定の `Aircraft()`、巡航 {F['cruise_m_s']} m/s・高度 "
        f"{F['patrol_alt_m']:.0f} m でトリムし、推力/抗力比を出す"
        "(他の study と同じ量)。")
    A_("")
    A_(f"判定は {F['n_decisions']} 件({len(op.stations)} 地点 × "
        f"{cfg.ops_stride:.0f} h 刻み × +{cfg.horizon:.0f} h)。同じ基準を "
        f"{F['n_var_decisions']} 件の比較系列にも掛ける。一次判定に使うのは "
        f"B2 の**製品トラック**(気象庁予報 +{F['anchor_lead_h']:.0f} h まで"
        "をそのまま使い、以降はモデルの移動を接続)、比較に使うのは気象庁"
        "予報に固定しない**自由走行の VAR モデル**。両者の判定が食い違う"
        "ところが、公式予報に載せた効果そのもの。")
    A_("")
    A_(f"- 運航可の割合: 製品トラック **{F['go_frac'] * 100:.0f}%**、"
        f"自由走行 VAR {F['var_go_frac'] * 100:.0f}%、一致率 "
        f"**{num(F['agreement_frac'], 2)}**。不一致 {F['n_disagree']} 件の"
        f"内訳は製品のみ可 {F['n_disagree_product_go']} 件・VAR のみ可 "
        f"{F['n_disagree_var_go']} 件。")
    first_sites = [r["site"] for r in F["per_site"]
                   if r["first_go_lead_h"] == F["first_go_overall"]]
    A_(f"- 最初に運航可になるのは +{num(F['first_go_overall'], 0)} h"
        f"({'・'.join(first_sites) if first_sites else 'なし'})、"
        f"**全地点が可になるのは +{num(F['all_go_from_lead_h'], 0)} h 以降**。"
        f"限界超過の内訳 {F['breach_counts']} → 束縛条件は "
        f"**{F['binding_constraint']}**(推力余裕は最小 "
        f"{min(r['min_margin'] for r in F['per_site']):.1f} で限界 "
        f"{F['limits']['thrust_margin']:.2f} を一度も割らない。"
        "この台風では推力ではなく風で決まる)。")
    A_(f"- 気象庁が予報を出している +{F['anchor_lead_h']:.0f} h までの窓"
        "(製品トラックが公式位置そのものを使う区間)では、可は "
        f"{F['go_frac_official_window'] * 100:.0f}%、自由走行 VAR との一致率 "
        f"{num(F['agreement_frac_official_window'], 2)}(この窓の不一致 "
        f"{F['n_disagree_official_window']} 件)。この窓での差は風の物理では"
        "なく、両トラックの位置差(B2 の cross-track)が運航限界の風速に"
        f"効いたもの。+{F['anchor_lead_h']:.0f} h 以降は製品もモデル延長に"
        "なるので差は再び縮む(接続点で再固定しているため)。運航判断は"
        "安全側(飛べないほう)に倒す。")
    A_(f"- 感度: E1 の陸上曝露係数 k = {F['land_wind_factor']:.2f} を風に"
        f"掛けると運航可は {F['go_frac_land_exposure'] * 100:.0f}% まで上がる"
        "(= ほぼ飛べてしまう)。E1 が示すとおり k は観測風のピークを "
        f"{E1['wind']['peak_obs_m_s']:.0f} → "
        f"{E1['wind']['peak_calibrated_m_s']:.0f} m/s に"
        "潰すので、これ単独での採用は危険。本判定は海上風の渦場をそのまま"
        "使う(艇は水面から飛び、気象庁の警報域半径も海上風で述べられる)。")
    n_rain = F["breach_counts"].get("rain", 0)
    n_rain_a = F["breach_counts_anchored_rain"].get("rain", 0)
    d_rain = F["go_frac_anchored_rain"] - F["go_frac"]
    A_(f"- 感度: 降水に E4 のアンカー倍率(中央値 ×"
        f"{F['anchor_rain_median_scale']:.2f}、= 公表する製品雨量)を掛けると、"
        f"運航可は {F['go_frac_anchored_rain'] * 100:.0f}%"
        f"({d_rain * 100:+.0f} ポイント)、降水超過は {n_rain} 件 → "
        f"{n_rain_a} 件。"
        + ("この台風では風が先に限界を越えるので可否そのものは変わらず、"
           "超過の内訳としての降水だけが過少に数えられる。"
           if abs(d_rain) < 1e-9 else
           "風だけでなく降水でも判定が動いており、アンカー前の場を使った"
           "一次判定は降水を過少に見ている。")
        + "倍率は「地域内で最も多い所」の値を地点に当てる仮定なので、"
          "地点の真値はこの両者の間にあるはず。")
    A_("")
    A_("地点別:")
    A_("")
    A_("| 地点 | n | 可/否 | 可の割合 | 最初の可 lead h | 可の窓 "
       "| 最大風 m/s | 最大突風 m/s | 最大降水 mm/h | 最小推力余裕 "
       "| 束縛 | 自由走行 VAR との一致 | (感度) 陸上係数で可 "
       "| (感度) アンカー雨量で可 |")
    A_("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in F["per_site"]:
        A_(f"| {r['site']} | {r['n']} | {r['go_n']}/{r['no_go_n']} "
          f"| {num(r['go_frac'], 2)} | {num(r['first_go_lead_h'], 0)} "
          f"| {', '.join(r['go_windows']) or '―'} "
          f"| {r['max_wind_m_s']:.1f} | {r['max_gust_m_s']:.1f} "
          f"| {r['max_rain_mm_h']:.1f} | {num(r['min_margin'], 2)} "
          f"| {r['binding'] or '―'} | {num(r['var_agree_frac'], 2)} "
          f"| {num(r['go_frac_land'], 2)} "
          f"| {num(r['go_frac_anchor_rain'], 2)}(×"
          f"{r['anchor_rain_scale']:.2f}) |")
    A_("")
    A_("リードタイム別(14 地点中いくつ飛べるか):")
    A_("")
    A_("| lead | 時刻 JST | トラックの出典 | 可の地点数(製品) "
       "| 可の地点数(自由走行 VAR) | 最悪地点 |")
    A_("|---|---|---|---|---|---|")
    for r in F["per_lead"]:
        A_(f"| +{r['lead_h']:.0f} h | {r['stamp_jst']} | {r['track_src']} "
          f"| {r['go_sites']}/{r['n_sites']} "
          f"| {'' if r['var_n_sites'] == 0 else f'{r['var_go_sites']}/{r['var_n_sites']}'} "
          f"| {r['worst_site']} |")
    A_("")
    if F["disagree"]:
        A_(f"不一致 {F['n_disagree']} 件のうち先頭 12 件"
            "(製品 = 気象庁予報 + モデル延長、VAR = 自由走行):")
        A_("")
        A_("| 地点 | lead | 時刻 JST | トラックの出典 | 製品 | VAR "
           "| 製品超過 | VAR 超過 | 風 m/s 製品/VAR | 中心距離 km 製品/VAR |")
        A_("|---|---|---|---|---|---|---|---|---|---|")
        for r in F["disagree"][:12]:
            A_(f"| {r['site']} | +{r['lead_h']:.0f} h | {r['stamp_jst']} "
              f"| {r['track_src']} "
              f"| {'可' if r['product_go'] else '否'} "
              f"| {'可' if r['var_go'] else '否'} "
              f"| {r['product_breach'] or '―'} | {r['var_breach'] or '―'} "
              f"| {r['product_wind']:.1f} / {r['var_wind']:.1f} "
              f"| {r['product_dist_km']:.0f} / {r['var_dist_km']:.0f} |")
        A_("")
    A_("## G. 通報そのものが言っていること")
    A_("")
    A_(f"- 解析した台風電文 {G['n_bulletins']} 報。位置を本文から読めたのは "
        f"{G['n_fixes']} 点(実況 {G['n_analysis_fixes']}・予報 "
        f"{G['n_forecast_fixes']};公式予報位置は諸元 XML から取る)。")
    A_(f"- 警報域の推移 {G['warnings'][0]['stamp_jst']} 〜 "
        f"{G['warnings'][-1]['stamp_jst']} JST: 暴風域の最大平均半径 "
        f"{G['storm_max_mean_km']:.0f} km、強風域 {G['gale_max_mean_km']:.0f} km。")
    A_(f"- 公式予報円(確率圏)の最大半径 {G['max_circle_km']:.0f} km、"
        f"最長リード +{G['max_official_lead_h']:.1f} h。")
    A_(f"- 雨の見通し {G['n_outlooks']} 報 / {G['n_outlook_blocks']} ブロック。")
    A_("")
    A_("警報域の推移(電文から解析した全時刻):")
    A_("")
    A_("| 時刻 JST | 暴風域 | 強風域 | 暴風域平均半径 km | 強風域平均半径 km |")
    A_("|---|---|---|---|---|")
    for r in G["warnings"]:
        A_(f"| {r['stamp_jst']} | {r['storm']} | {r['gale']} "
          f"| {num(r['storm_mean_km'], 0)} | {num(r['gale_mean_km'], 0)} |")
    A_("")
    A_("## 結論と限界")
    A_("")
    last = C["rows"][-1]
    A_(f"1. **進路**: 実況に対するヒンドキャスト RMSE は +3 h で "
        f"{C['rows'][0]['rmse_var_km']:.0f} km、+24 h で "
        f"{last['rmse_var_km']:.0f} km(全リード平均 "
        f"{C['rmse_mean_var_km']:.1f} km)。永続予報"
        f"({C['rmse_mean_persistence_km']:.1f} km)と移動永続予報"
        f"({C['rmse_mean_motion_km']:.1f} km)の双方を上回り、"
        f"+{num(B['inside_circle']['last_lead_inside_h'], 0)} h まで気象庁の"
        "予報円内に収まった。1 台風の 107 時間で当てた運動モデルとしては"
        "運用に足るが、円は公式のほうが広い(=公式のほうが安全側)。")
    A_(f"2. **強度**: 気圧 RMSE {D['hindcast_pres_rmse_mean_hPa']:.1f} hPa・"
        f"風速 RMSE {D['hindcast_vmax_rmse_mean_m_s']:.1f} m/s。公式予報との"
        f"差は気圧 {D['official_pres_rmse_hPa']:.1f} hPa・風速 "
        f"{D['official_vmax_rmse_m_s']:.1f} m/s で、温帯低気圧化の扱い"
        "(緩和則)が差の主因。")
    A_(f"3. **降水**: 観測に対する RMSE {E1['rmse_mm_h']:.2f} mm/h・"
        f"バイアス {E1['bias_mm_h']:+.2f} mm/h・CSI {num(E1['wet']['csi'], 2)}。"
        "平均的な雨量はほぼ偏りなく再現するが、≥10 mm/h の豪雨は "
        f"POD {num(E1['heavy_pod'], 2)}・平均 {num(E1['heavy_underestimate_mm_h'], 0)}"
        " mm/h の過小評価で、**雨量の絶対値を根拠にしてはいけない**。"
        f"公式「多い所で」に対しては地域最大値でも中央値 "
        f"{num(E2['median_ratio_district_max'], 2)} 倍、降水確率との順位相関は "
        f"{E3['spearman_pop_district_max']:.2f}。したがって公表する量は"
        f"気象庁の値にアンカーし(E4)、モデルは時間と空間の分布だけを出す: "
        f"アンカー後は最大 {E4['anchored_max_total_mm']:.0f} mm・最大 1 時間 "
        f"{E4['anchored_max_peak_mm_h']:.0f} mm/h(**{E4['anchored_wettest_site']}**)。")
    A_(f"4. **運航**: 製品トラックでは判定の {F['go_frac'] * 100:.0f}% が"
        f"運航可、気象庁予報に固定しない自由走行 VAR では "
        f"{F['var_go_frac'] * 100:.0f}%、一致率 {num(F['agreement_frac'], 2)}。"
        f"束縛条件は **{F['binding_constraint']}**。公式予報に載せるとは、"
        "進路を自分の推定から気象庁のそれに差し替えることであり、その位置差"
        "が限界風速に効いて判定を動かす。運用では製品を使い、さらに予報円の"
        "外側(円半径ぶん中心がずれる場合)を想定して安全側に倒す。")
    A_(f"5. **限界**: (a) 1 台風・{A['track_hours']:.0f} 時間の学習であり、"
        "他台風への外挿は未検証。(b) 降水場は中心気圧と強風域半径だけで決まる解析的な渦で、"
        "降雨帯の構造・前線の影響・地形性豪雨の詳細は持たない。"
        "(c) 予報風は海上の渦風であり、地上の AMeDAS 風とは曝露も環境場も"
        "異なる(E1)。(d) 予報ホライズンの後半は気象庁が温帯低気圧と"
        f"見込む期間を含み、台風としての予報は +{F['anchor_lead_h']:.0f} h "
        "までしか公式に検証できない(製品トラックはそれ以降モデル延長)。"
        "(e) 運航限界値はプリセット由来の便宜的なもので、実機の規定ではない。")
    A_("")
    A_("## 生成ファイル")
    A_("")
    for nm in ["REPORT.md", "summary.json"] + CSV_FILES + [
            "typhoon_forecast_study.py"]:
        A_(f"- `{nm}`")
    for p in plot_names:
        A_(f"- `{Path(p).name}`")
    A_("")
    A_(f"実行時間: {runtime_s:.1f} s / git HEAD: `{git_head()[:12]}` / "
        f"データ: `{cfg.data}`(取得 {A['fetched_utc']} UTC)")
    (cfg.out / "REPORT.md").write_text("\n".join(L) + "\n", encoding="utf-8")


# ----------------------------------------------------------------------
#  driver
# ----------------------------------------------------------------------
def dump_csv(path: Path, rows: list) -> bool:
    """Write a list of dicts; False (and no file) when there is nothing."""
    if not rows:
        return False
    write_csv(path, rows, list(rows[0].keys()))
    return True


def run_study(cfg: StudyConfig) -> dict:
    t0 = time.time()
    cfg.out.mkdir(parents=True, exist_ok=True)
    snap = load_snapshot(cfg.data)
    print(f"snapshot: {snap.manifest.get('bulletins')} bulletins, "
          f"track {len(snap.track)} points to {jst(snap.track.stamps[-1])} JST, "
          f"AMeDAS {sum(len(v) for v in snap.amedas.values())} records")

    A = section_snapshot(snap, cfg)
    G = section_bulletins(snap, cfg)
    C = section_hindcast(snap, cfg)
    print(f"hindcast: {C['n_issues']} issues, VAR RMSE mean "
          f"{C['rmse_mean_var_km']:.1f} km (persistence "
          f"{C['rmse_mean_persistence_km']:.1f}, motion "
          f"{C['rmse_mean_motion_km']:.1f}), order selected VAR"
          f"({C['sel_order']}) of VAR({C['shipped_order']})")

    # The order is picked on the early half of the hindcast issues, all of
    # which precede the operational issue time, so the forecast below still
    # uses nothing from after it.
    op = OperationalForecast(snap, cfg.horizon, order=C["sel_order"])
    # the product an operator would fly: JMA's own positions to the last one
    # JMA publishes, the model's motion spliced on past it
    opp = OperationalForecast(snap, cfg.horizon, order=C["sel_order"],
                              official=True)
    B = section_forecast(op, opp, cfg)
    bl = B["blend"]
    print(f"forecast issued {B['issue_jst']} JST -> +{cfg.horizon:.0f} h, "
          f"end {B['end_state']['stamp_jst']} JST "
          f"{B['end_state']['lat']:.1f}N {B['end_state']['lon']:.1f}E "
          f"{B['end_state']['pres_hPa']:.0f} hPa")
    print(f"product track: JMA to +{bl['anchor_lead_h']:.0f} h "
          f"({bl['n_official_points']} official points), model past it; "
          f"end {bl['end_state']['stamp_jst']} JST "
          f"{bl['end_state']['lat']:.1f}N {bl['end_state']['lon']:.1f}E, "
          f"{bl['cross_at_horizon_km']:.0f} km from the free-running model")
    D = section_intensity(op, C, cfg)
    print(f"intensity: hindcast RMSE {D['hindcast_pres_rmse_mean_hPa']:.2f} hPa "
          f"/ {D['hindcast_vmax_rmse_mean_m_s']:.2f} m/s, vs official "
          f"{D['official_pres_rmse_hPa']:.1f} hPa "
          f"/ {D['official_vmax_rmse_m_s']:.1f} m/s")

    E1 = section_rain_obs(op, cfg)
    print(f"rain vs AMeDAS: n={E1['n']} RMSE {E1['rmse_mm_h']:.2f} mm/h "
          f"bias {E1['bias_mm_h']:+.2f} corr {E1['corr']:.2f} "
          f"CSI {E1['wet']['csi']}")
    E2 = section_rain_peaks(op, cfg)
    E3 = section_rain_pop(op, cfg)
    E4 = section_rain_sites(opp, cfg, E2)
    print(f"rain vs bulletins: {E2['n']} district-windows, median ratio "
          f"{E2['median_ratio_district_max']}; POP {E3['n']} records, "
          f"corr {E3['corr_pop_district_max']:.2f}, CSI {E3['csi']}")
    print(f"rain forecast: wettest {E4['wettest_site']} "
          f"{E4['max_total_mm']:.0f} mm, peak {E4['max_peak_mm_h']:.1f} mm/h; "
          f"anchored to JMA (median x{E4['anchor']['median_scale']:.2f}): "
          f"wettest {E4['anchored_wettest_site']} "
          f"{E4['anchored_max_total_mm']:.0f} mm, peak "
          f"{E4['anchored_max_peak_mm_h']:.1f} mm/h")

    F = section_ops(opp, op, cfg, E1, E4)
    print(f"operations: go {F['go_frac'] * 100:.0f}% on the product track "
          f"(free-running VAR {F['var_go_frac'] * 100:.0f}%), agreement "
          f"{F['agreement_frac'] * 100:.0f}%, binding {F['binding_constraint']}")

    for name, rows in (
            ("snapshot_fixes.csv", A["text_vs_track"]["rows"]),
            ("track_forecast.csv", B["rows"]),
            ("track_product.csv", B["blend"]["rows"]),
            ("track_vs_official.csv", B["official"]),
            ("hindcast.csv", C["rows"]),
            ("order_selection.csv", C["order_rows"]),
            ("intensity_official.csv", D["official"]),
            ("rain_obs.csv", E1["rows"]),
            ("rain_obs_station.csv", E1["per_station"]),
            ("rain_peaks_official.csv", E2["rows"]),
            ("rain_pop_official.csv", E3["rows"]),
            ("rain_sites.csv", E4["sites"]),
            ("rain_sites_6h.csv", E4["six_hour"]),
            ("rain_sites_hourly.csv", E4["hourly"]),
            ("ops_decisions.csv", F["rows"]),
            ("ops_decisions_var.csv", F["var_rows"]),
            ("ops_site.csv", F["per_site"]),
            ("ops_lead.csv", F["per_lead"]),
            ("ops_disagree.csv", F["disagree"]),
            ("bulletin_warnings.csv", G["warnings"]),
            ("bulletin_outlooks.csv", G["outlook_blocks"])):
        dump_csv(cfg.out / name, rows)

    plot_names = []
    if cfg.plots:
        plot_names = make_plots(cfg, snap, op, A, B, C, D, E1, E2, E3, E4, F)
        print("plots:", plot_names)

    runtime_s = time.time() - t0
    write_report(cfg, snap, op, A, B, C, D, E1, E2, E3, E4, F, G, plot_names,
                 runtime_s)
    man = A["manifest"]
    summary = {
        "study": "typhoon_25_dujuan",
        "title_ja": "台風25号(2026年・ドゥージェン)の実戦予報と検証",
        "typhoon": {"tc_id": tf.TC_ID, "jma_number": tf.TY_NUMBER,
                    "name_en": "DUJUAN", "name_ja": "ドゥージェン"},
        "data": str(cfg.data),
        "source": "気象庁 (台風電文・諸元 XML・府県天気予報 R1 XML・AMeDAS)",
        "config": {
            "horizon_h": cfg.horizon, "hind_first_h": cfg.hind_first,
            "hind_stride_h": cfg.hind_stride,
            "hind_horizon_h": cfg.hind_horizon, "leads": list(cfg.leads),
            "orders": list(cfg.orders), "ops_stride_h": cfg.ops_stride,
            "plots": cfg.plots, "quick": cfg.quick,
            "ops_limits": F["limits"], "ops_limits_source": F["limits_source"],
        },
        "snapshot": {
            **{k: v for k, v in A.items()
               if k not in ("manifest", "text_vs_track", "official_issues")},
            "manifest": {k: v for k, v in man.items() if k != "sources"},
            "sources": man.get("sources", {}),
            "text_vs_track": {k: v for k, v in A["text_vs_track"].items()
                              if k != "rows"},
            "official_issues": A["official_issues"],
        },
        "forecast": {
            **{k: v for k, v in B.items()
               if k not in ("rows", "previous_issue", "blend")},
            "blend": {k: v for k, v in B["blend"].items() if k != "rows"}},
        "hindcast": {k: v for k, v in C.items()
                     if k not in ("issues", "pres_err", "vmax_err")},
        "intensity": D,
        "rain_obs": {k: v for k, v in E1.items()
                     if k not in ("rows", "per_hour")},
        "rain_peaks_official": {k: v for k, v in E2.items() if k != "rows"},
        "rain_pop_official": {k: v for k, v in E3.items() if k != "rows"},
        "rain_sites": {k: v for k, v in E4.items()
                       if k not in ("hourly", "six_hour")},
        "operations": {k: v for k, v in F.items()
                       if k not in ("rows", "var_rows", "disagree")},
        "bulletins": {k: v for k, v in G.items()
                      if k not in ("outlook_blocks", "probability_circles",
                                   "fix_heads")},
        "files": ["REPORT.md", "summary.json"] + CSV_FILES
                 + [Path(p).name for p in plot_names]
                 + ["typhoon_forecast_study.py"],
        "runtime_s": round(runtime_s, 1),
        "git_head": git_head(),
    }
    (cfg.out / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=float),
        encoding="utf-8")
    shutil.copy2(Path(__file__).resolve(),
                 cfg.out / "typhoon_forecast_study.py")
    print(f"done in {runtime_s:.1f} s -> {cfg.out}")
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Typhoon No. 25 (2026 Dujuan) forecast study from JMA "
                    "bulletins")
    ap.add_argument("--out", default=str(StudyConfig.out),
                    help="output directory")
    ap.add_argument("--data", default=str(tf.DATA_DIR),
                    help="JMA bulletin snapshot directory")
    ap.add_argument("--quick", action="store_true",
                    help="fewer issues / shorter horizon / fewer orders")
    ap.add_argument("--no-plots", action="store_true", help="skip PNG plots")
    a = ap.parse_args(argv)
    cfg = StudyConfig(out=Path(a.out), data=Path(a.data), quick=a.quick,
                      plots=not a.no_plots)
    run_study(cfg)


if __name__ == "__main__":
    main()
