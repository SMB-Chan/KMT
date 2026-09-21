"""Typhoon track and rainfall forecast built on JMA bulletins.

The module consumes the Japan Meteorological Agency bulletins of Typhoon
No. 25 (Dujuan, TC2630) that `fetch_typhoon_dujuan.py` snapshots into
``data/jma_typhoon_2625/``:

* the typhoon bulletins (``TC2630_forecast.json`` / ``_specifications.json``)
  with the analysed position, intensity, warning areas and the official
  forecast positions with their probability circles;
* the typhoon-related meteorological bulletins (general / regional /
  prefectural weather explanation information, ``denbun/*.json``) whose text
  carries timestamped analysed positions, intensity and the official
  rainfall outlooks ("24-hour rainfall up to ... mm");
* hourly AMeDAS observations of the stations along the track, used as the
  verifying "observed" rain.

From those bulletins the module rebuilds

* `analysis_track` -- the timestamped best-track-like analysis series
  (position / central pressure / maximum wind) by parsing the bulletin text;
* `official_forecasts` -- every official forecast point (issue time, valid
  time, position, probability-circle radius) as the operational reference;
* `TrackForecastModel` -- our own forecast: a ridge-regularised VAR on the
  3-hourly motion anomalies of the analysis series (same linear-inverse /
  companion-stability machinery as `weather_forecast.VarForecastModel`),
  integrated forward from the analysed position at the issue time, with a
  damped-trend central-pressure forecast;
* the warning areas (暴風域 / 強風域) every bulletin states, which drive
  both the bulletin-anchored wind profile `vortex_wind` and the rain shield
  `rain_rate_mm_h` calibrated on the AMeDAS observations of this typhoon;
  together they produce simulator-ready (`WeatherConfig`,
  `AtmosphereConfig`) snapshots through `TyphoonSnapshot.sim_configs`.

Everything is deterministic: no RNG, no network access at import time.
"""

import csv
import json
import math
import os
import re
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone

import numpy as np

from atmosphere import AtmosphereConfig
from weather import WeatherConfig
from weather_real import GUST_SIGMA_FACTOR

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "data", "jma_typhoon_2625")
TC_ID = "TC2630"
TY_NUMBER = "2625"

EARTH_RADIUS_KM = 6371.0088
GRID_HOURS = 3.0            # analysis/forecast grid of the JMA bulletins
MIN_TRAIN_STEPS = 6         # >= 6 motion samples (18 h) before an issue time
SEGMENT_GAP_HOURS = 15.0    # longer gap between two positions => new segment;
                            # a segment is never bridged by interpolation
TRACK_ORDER = 3             # VAR order of the motion model
PRIOR_WEIGHT = 1.0          # weight of the motion-persistence prior
TRACK_WINDOW_HOURS = 36.0   # leg the motion model is fitted on
MAX_RADIUS = 1.0            # companion-matrix modes may not grow
PRES_FLOOR, PRES_CEIL = 900.0, 1015.0
PRES_RELAX_HPA = 996.0      # central pressure of the extratropical low the
                            # storm relaxes towards once it leaves the tropics
PRES_DECAY = 0.80           # per-step damping of the recent pressure trend
PRES_RELAX_RATE = 0.10      # per-step relaxation, reached at ET_LAT_END
ET_LAT_START = 30.0         # latitude where extratropical transition begins
ET_LAT_END = 40.0           # latitude where it is complete
VMAX_PRES_EXP = 0.5         # vmax ~ (1013 - p)**0.5, anchored on the analysis
FALLBACK_VMAX_M_S = 25.0      # used when no bulletin states a maximum wind

# Rain shield, calibrated on 980 AMeDAS station-hours of this typhoon: a
# disc of radius RAIN_DECAY_FRAC * gale_radius whose centre is displaced
# RAIN_LEAD_FRAC * gale_radius forward along the motion, so the rain falls
# ahead of the centre and dies out near the edge of the gale area.
RAIN_SCALE = 0.205          # mm/h per hPa of central-pressure deficit
RAIN_DECAY_FRAC = 0.30      # e-folding radius / gale radius
RAIN_LEAD_FRAC = 0.60       # forward displacement of the shield / gale radius
RAIN_ORO_POWER = 0.5        # exponent of the orographic gain
RAIN_MAX_MM_H = 45.0        # cap of the radial amplitude
FALLBACK_GALE_KM = 400.0    # gale radius when no bulletin states one

# Wind profile V(r) = vmax inside rmax and vmax * (rmax/r)**b outside.  Both
# numbers come from the warning areas the bulletins state: 25 m/s at the edge
# of the 暴風域 and 15 m/s at the edge of the 強風域 fix b, and the 暴風域
# radius then fixes rmax.
STORM_WIND_M_S = 25.0       # JMA 暴風域 threshold
GALE_WIND_M_S = 15.0        # JMA 強風域 threshold
WIND_POWER_FALLBACK = 0.47  # radial exponent when one of the radii is missing
WIND_POWER_RANGE = (0.20, 1.20)   # physical bounds of the fitted exponent
GUST_RATIO_FALLBACK = 1.43  # 最大瞬間風速 / 最大風速 of these bulletins
RMAX_FALLBACK_KM = 50.0     # rmax when no warning area is stated at all
RMAX_PER_HPA = 0.5

# Fixed orographic gain (lat, lon, sigma km, gain): the mountain districts
# the track passes; a schematic, deterministic stand-in for a real
# orographic rainfall model.
OROGRAPHIC_ANCHORS = (
    (33.10, 139.80, 60.0, 0.8),   # Hachijo / south Izu islands
    (34.40, 139.30, 70.0, 0.7),   # Izu islands / Izu peninsula
    (35.30, 139.10, 70.0, 0.8),   # Tanzawa / Kanto mountains south
    (36.60, 138.60, 90.0, 0.7),   # Kanto mountains
    (35.40, 138.20, 90.0, 0.8),   # Akaishi / Tokai mountains
    (33.90, 135.80, 90.0, 0.6),   # Kii mountains
    (37.30, 140.60, 90.0, 0.6),   # Abukuma highlands
    (39.30, 141.20, 110.0, 0.5),  # Ou mountains north
)

_ZEN = str.maketrans("０１２３４５６７８９", "0123456789")


def _zen2han(text):
    return text.translate(_ZEN)


def _parse_jst(day, hour, reference):
    """Day/hour-of-month in JMA bulletin text (JST) to a UTC datetime."""
    base = reference.astimezone(timezone(timedelta(hours=9)))
    month, year = base.month, base.year
    if day < base.day - 10:          # rolled into the next month
        month = 12 if month == 12 else month + 1
        if month == 1:
            year += 1
    dt = datetime(year, month, day, hour, 0,
                  tzinfo=timezone(timedelta(hours=9)))
    return dt.astimezone(timezone.utc)


def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(min(1.0, math.sqrt(a)))


def bearing_deg(lat1, lon1, lat2, lon2):
    """Initial great-circle bearing from point 1 to point 2, deg from N."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    x = math.sin(dl) * math.cos(p2)
    y = (math.cos(p1) * math.sin(p2)
         - math.sin(p1) * math.cos(p2) * math.cos(dl))
    return math.degrees(math.atan2(x, y)) % 360.0


# --------------------------------------------------------------------------
# bulletin parsing
# --------------------------------------------------------------------------

_POSITION_RE = re.compile(
    r"(\d{1,2})日(\d{1,2})時(?:には)?、?[^。\n]{0,60}?北緯(\d{1,2})度(\d{1,2})分、"
    r"東経(\d{1,3})度(\d{1,2})分")
_PRESSURE_RE = re.compile(r"中心の気圧は(\d{3,4})ヘクトパスカル")
_VMAX_RE = re.compile(r"中心付近の最大風速は(\d{1,2})メートル")
_SPEED_RE = re.compile(r"1時間におよそ(\d{1,2})キロの速さで")
_SLOW_RE = re.compile(r"ゆっくりした速さで")
_STALL_RE = re.compile(r"ほとんど停滞し")
_COURSE_RE = re.compile(r"([\u30a1-\u30ff\u4e00-\u9fff]{1,6})へ進んでいます")
_FORECAST_TAIL_RE = re.compile(r"^を中心とする半径(\d{1,4})キロ")
_ANALYSIS_TAIL_RE = re.compile(r"^(にあって|において|にあり)")
_COURSE = {"北": 0.0, "北北東": 22.5, "北東": 45.0, "東北東": 67.5,
           "東": 90.0, "東南東": 112.5, "南東": 135.0, "南南東": 157.5,
           "南": 180.0, "南南西": 202.5, "南西": 225.0, "西南西": 247.5,
           "西": 270.0, "西北西": 292.5, "北西": 315.0, "北北西": 337.5,
           "ほとんど停滞": None}


@dataclass(frozen=True)
class WarningSector:
    """One JMA warning area (暴風域 / 強風域) as a pair of half-disc radii.

    `dir_deg` is the compass direction the first stated radius faces and
    `radius_km` that radius; `opposite_km` is the radius of the opposite
    half-disc.  A symmetric 「中心から半径130キロ」 area is stored with both
    radii equal, 「中心の北東側700キロ以内と南西側440キロ以内」 as
    (45, 700, 440).
    """
    dir_deg: float = 0.0
    radius_km: float = float("nan")
    opposite_km: float = float("nan")

    def mean_radius(self):
        """Radius of the equivalent circular area."""
        if not math.isfinite(self.radius_km):
            return float("nan")
        if not math.isfinite(self.opposite_km):
            return self.radius_km
        return 0.5 * (self.radius_km + self.opposite_km)


def sector_radius(sector, bearing):
    """Radius [km] of a warning area in the direction `bearing` from centre.

    The two half-disc radii JMA states are blended with a cosine, which
    reproduces both stated values exactly and interpolates smoothly in
    between: R(phi) = (ra+rb)/2 + (ra-rb)/2 * cos(phi - dir_a).
    """
    if sector is None:
        return float("nan")
    ra = float(sector.radius_km)
    if not math.isfinite(ra):
        return float("nan")
    rb = float(sector.opposite_km)
    if not math.isfinite(rb):
        rb = ra
    return (0.5 * (ra + rb) + 0.5 * (ra - rb)
            * math.cos(math.radians(bearing - sector.dir_deg)))


# 「中心から半径130キロ以内では風速25メートル以上」 (symmetric) and
# 「中心の北東側700キロ以内と南西側440キロ以内では風速15メートル以上」
# (two half-discs) are the only two shapes JMA uses.
_SECTOR_FMT = (r"中心(?:から半径(\d+)キロ|の([\u4e00-\u9fff]{1,3})側(\d+)"
               r"キロ以内と([\u4e00-\u9fff]{1,3})側(\d+)キロ)以内では"
               r"風速%dメートル以上")
_STORM_RE = re.compile(_SECTOR_FMT % 25)
_GALE_RE = re.compile(_SECTOR_FMT % 15)
_GUST_RE = re.compile(r"最大瞬間風速は(\d{1,3})メートル")


def _sector(m):
    """`WarningSector` from a `_STORM_RE` / `_GALE_RE` match, or None."""
    if m is None:
        return None
    if m.group(1):
        r = float(m.group(1))
        return WarningSector(0.0, r, r)
    da, db = _COURSE.get(m.group(2)), _COURSE.get(m.group(4))
    if da is None or db is None:
        return None
    return WarningSector(da, float(m.group(3)), float(m.group(5)))


@dataclass(frozen=True)
class AnalysisFix:
    """One timestamped state taken from a JMA bulletin.

    `stamp` is the valid time the bulletin refers to and `issue` the time the
    bulletin was published.  The 気象解説情報 series describes the state of the
    most recent 3-hourly analysis, so `lead_hours` is normally negative there;
    `lead_hours` > 1 marks a position JMA had not analysed yet when it
    published the bulletin.
    """
    stamp: datetime
    lat: float
    lon: float
    pres_hPa: float
    vmax_m_s: float
    gust_m_s: float = float("nan")   # 最大瞬間風速, when the bulletin states it
    course_deg: object = None
    speed_km_h: float = float("nan")
    issue: object = None
    source: str = ""
    storm: object = None             # WarningSector of the 暴風域
    gale: object = None              # WarningSector of the 強風域

    @property
    def lead_hours(self):
        if self.issue is None:
            return 0.0
        return (self.stamp - self.issue).total_seconds() / 3600.0

    @property
    def is_analysis(self):
        return self.lead_hours <= 1.0


@dataclass(frozen=True)
class OfficialForecastPoint:
    stamp: datetime          # valid time (UTC)
    lat: float
    lon: float
    radius_km: float = float("nan")
    pres_hPa: float = float("nan")
    vmax_m_s: float = float("nan")
    gust_m_s: float = float("nan")
    speed_km_h: float = float("nan")
    course_deg: float = float("nan")
    category: str = ""       # 台風 (TY) / 温帯低気圧 (LOW) ...
    gale_km: float = float("nan")    # largest radius stated for each area
    storm_km: float = float("nan")
    gale: object = None              # WarningSector, direction resolved
    storm: object = None


@dataclass(frozen=True)
class OfficialForecast:
    issue: datetime
    points: tuple
    source: str = ""


def _deg_min(deg, minute):
    return deg + minute / 60.0


def parse_bulletin_text(text, issue, source=""):
    """Parse the stated state, movement and any forecast circles out of the
    free text of one JMA meteorological bulletin.

    Position sentences come in two shapes: the analysed/reported state
    ("...is at 24N 40, 146E 35, moving...") and the forecast ("...within a
    circle of radius 55 km centred on 32N 20, 138E 20"); the tail right
    after the coordinates tells them apart.  Central pressure, maximum wind
    and movement are read from the same paragraph as the position, so that a
    bulletin describing several states cannot mix them up.
    """
    text = _zen2han(text or "")
    fixes, forecasts = [], []
    for m in _POSITION_RE.finditer(text):
        day, hour, la, lm, lo, lm2 = (int(g) for g in m.groups())
        stamp = _parse_jst(day, hour, issue)
        lat, lon = _deg_min(la, lm), _deg_min(lo, lm2)
        tail = text[m.end():m.end() + 24]
        fm = _FORECAST_TAIL_RE.match(tail)
        if fm:
            forecasts.append(OfficialForecastPoint(
                stamp, lat, lon, float(fm.group(1))))
            continue
        if not _ANALYSIS_TAIL_RE.match(tail):
            continue
        after = text.find("\n", m.end())
        before = text.rfind("\n", 0, m.start())
        scope = text[before + 1:after if after >= 0 else len(text)]
        pm = _PRESSURE_RE.search(scope)
        vm = _VMAX_RE.search(scope)
        sm = _SPEED_RE.search(scope)
        cm = _COURSE_RE.search(scope)
        gm = _GUST_RE.search(scope)
        speed = float(sm.group(1)) if sm else float("nan")
        if not sm and _SLOW_RE.search(scope):
            speed = 0.0            # JMA wording for "less than about 5 km/h"
        course = _COURSE.get(cm.group(1)) if cm else None
        if course is None and _STALL_RE.search(scope):
            course = float("nan")
        fixes.append(AnalysisFix(
            stamp, lat, lon,
            float(pm.group(1)) if pm else float("nan"),
            float(vm.group(1)) if vm else float("nan"),
            float(gm.group(1)) if gm else float("nan"),
            course, speed, issue, source,
            _sector(_STORM_RE.search(scope)),
            _sector(_GALE_RE.search(scope))))
    return fixes, forecasts


# The ［雨の予想］ section is a run of blocks: a head line naming the
# accumulation window and its length, then one ideographic-space-padded
# 「地域　Nミリ」 line per district.  A block ends at the next head line, at a
# blank line or at the next ［見出し］ -- never at a 「。」, which does not
# occur inside a block.
_RAIN_HEAD_RE = re.compile(
    r"(?P<cont>その後、)?(?P<d1>\d{1,2})日(?:(?P<h1>\d{1,2})時)?"
    r"(?P<sep>から|に)"
    r"(?:(?P<d2>\d{1,2})日(?:(?P<h2>\d{1,2})時)?(?:までに|にかけて))?"
    r"予想される(?P<hours>\d{1,2})時間降水量は多い所で")
_RAIN_LINE_RE = re.compile(
    r"^[\s　]*([^\s　\d][^　\d]*?)[\s　]*(\d{2,4})ミリ\s*$")


@dataclass(frozen=True)
class RainOutlook:
    """One 「N時間降水量は多い所で」 block of a bulletin.

    `hours` is the accumulation length, `start` / `end` the window it covers
    in UTC (both None for a 「N日に」 whole-day window) and `regions` maps a
    district name to the peak accumulation JMA expects there.
    """
    hours: int
    start: object
    end: object
    regions: dict
    continued: bool = False

    def max_mm(self):
        """Largest 「多い所で」 value of the block."""
        return max(self.regions.values()) if self.regions else float("nan")


def _rain_window(m, issue):
    """(hours, start, end, continued) of one rain-outlook head line."""
    d1, h1 = int(m.group("d1")), m.group("h1")
    d2, h2 = m.group("d2"), m.group("h2")
    start = _parse_jst(d1, int(h1), issue) if issue and h1 else None
    end = None
    if issue and d2:
        end = _parse_jst(int(d2), int(h2) if h2 else 0, issue)
        if h2 is None:                 # 「N日にかけて」 runs to the next day
            end += timedelta(hours=24)
    return int(m.group("hours")), start, end, bool(m.group("cont"))


def parse_rain_outlook(text, issue=None):
    """(RainOutlook, ...) of the ［雨の予想］ section of a bulletin."""
    out, head, regions = [], None, {}

    def flush():
        if head is not None and regions:
            out.append(RainOutlook(head[0], head[1], head[2], dict(regions),
                                   head[3]))

    for line in re.split(r"<br>|\n", _zen2han(text or "")):
        hm = _RAIN_HEAD_RE.search(line)
        if hm:
            flush()
            head, regions = _rain_window(hm, issue), {}
            continue
        lm = _RAIN_LINE_RE.match(line)
        if head is not None and lm:
            regions[lm.group(1).strip()] = float(lm.group(2))
        elif head is not None and (not line.strip()
                                   or line.strip().startswith("［")):
            flush()
            head, regions = None, {}
    flush()
    return tuple(out)


def _iso(dt):
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _from_iso(text):
    return datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ").replace(
        tzinfo=timezone.utc)


def _as_dt(value):
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, str):
        text = value.strip()
        if text.endswith("Z"):
            return _from_iso(text)
        return datetime.fromisoformat(text).astimezone(timezone.utc)
    raise TypeError(f"not a time: {value!r}")


def _as_hour(value, epoch):
    """datetime / ISO string / plain hour number -> hours since `epoch`."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return (_as_dt(value) - epoch).total_seconds() / 3600.0


def _interp_known(at, hours, values):
    """Interpolate `values` at `at`, using the finite entries only."""
    ok = np.isfinite(values)
    if not ok.any():
        return float("nan")
    if ok.all():
        return float(np.interp(at, hours, values))
    return float(np.interp(at, np.asarray(hours)[ok], np.asarray(values)[ok]))


def track_times(n, validtime, genesis, max_hourly=8):
    """Valid times of the analysed positions inside one JMA track array.

    JMA analyses a typhoon every 3 hours; a bulletin issued between two of
    those times carries hourly estimates at the end of the array.  Given the
    genesis time (stated in the 「台風第N号になりました」 bulletin) the split
    between the two cadences is exact, and the result is checked against the
    bulletin valid time.
    """
    m = None
    if genesis is not None:
        dh = (genesis - validtime).total_seconds() / 3600.0
        cand = (3.0 * (n - 1) + dh) / 2.0
        if abs(cand - round(cand)) < 1e-6 and 0 <= round(cand) <= max_hourly:
            m = int(round(cand))
    if m is None:                      # fall back to the 3-hourly grid of JMA
        for k in range(max_hourly + 1):
            shifted = validtime - timedelta(hours=k)
            if shifted.minute == 0 and shifted.hour % 3 == 0:
                m = k
                break
        else:
            raise ValueError("cannot place the track array on a time axis")
    n3 = n - m
    start = genesis if genesis is not None else \
        validtime - timedelta(hours=m + 3 * (n3 - 1))
    out = [start + timedelta(hours=3 * k) for k in range(n3)]
    for k in range(1, m + 1):
        out.append(out[-1] + timedelta(hours=1))
    if abs((out[-1] - validtime).total_seconds()) > 1:
        raise ValueError("track times do not end at the bulletin valid time")
    return out, n3


# --------------------------------------------------------------------------
# snapshot container
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class BestTrack:
    """Timestamped position series rebuilt from the JMA bulletins.

    `hours` count from `epoch`, a reference shared by every sub-track cut out
    of the same series, so forecasts issued from different sub-tracks stay
    directly comparable.  `kinds[i]` is "analysis" for a true 現況 and
    "reported" for a position a bulletin stated ahead of its valid time.
    `issues[i]` is the time of the bulletin that states position `i`, empty
    when no bulletin in the snapshot does.
    """
    stamps: tuple
    hours: object            # np.ndarray, hours since `epoch`
    lat: object
    lon: object
    pres_hPa: object
    vmax_m_s: object
    source: str = ""
    epoch: object = None
    kinds: tuple = ()
    issues: tuple = ()

    def __len__(self):
        return len(self.stamps)

    def stamp_at(self, idx):
        return self.stamps[idx]

    @property
    def t0(self):
        return self.epoch if self.epoch is not None else _from_iso(self.stamps[0])

    def hour_of(self, stamp):
        return (_as_dt(stamp) - self.t0).total_seconds() / 3600.0

    def index_at(self, hour):
        return int(np.argmin(np.abs(self.hours - hour)))

    def _slice(self, sl):
        return BestTrack(tuple(np.asarray(self.stamps, dtype=object)[sl]),
                         self.hours[sl], self.lat[sl], self.lon[sl],
                         self.pres_hPa[sl], self.vmax_m_s[sl], self.source,
                         self.epoch, tuple(np.asarray(self.kinds)[sl]),
                         tuple(np.asarray(self.issues, dtype=object)[sl]))

    def upto(self, when, tol=1e-6):
        """Sub-track holding every position at or before `when` (the state of
        the record as an operational forecaster saw it at that time)."""
        h = _as_hour(when, self.t0)
        keep = self.hours <= h + tol
        if not keep.any():
            raise ValueError("no analysed position at or before that time")
        return self._slice(keep)

    def tail_segment(self, max_gap_hours=SEGMENT_GAP_HOURS):
        """Longest unbroken run of positions ending at the last one.

        A gap longer than `max_gap_hours` is not interpolated over: the
        bulletin record simply says nothing about the stretch, and inventing
        a straight, constant-speed path there would train the motion model on
        a fiction.
        """
        start = 0
        for k in range(1, len(self.hours)):
            if self.hours[k] - self.hours[k - 1] > max_gap_hours + 1e-9:
                start = k
        return self if start == 0 else self._slice(slice(start, len(self.hours)))

    @classmethod
    def from_fixes(cls, fixes, epoch=None, source="JMA bulletins"):
        """Build the series from parsed `AnalysisFix` items."""
        fixes = sorted(fixes, key=lambda f: f.stamp)
        epoch = epoch or fixes[0].stamp
        hours = np.array([(f.stamp - epoch).total_seconds() / 3600.0
                          for f in fixes], dtype=float)
        return cls(tuple(_iso(f.stamp) for f in fixes), hours,
                   np.array([f.lat for f in fixes], dtype=float),
                   np.array([f.lon for f in fixes], dtype=float),
                   np.array([f.pres_hPa for f in fixes], dtype=float),
                   np.array([f.vmax_m_s for f in fixes], dtype=float),
                   source, epoch,
                   tuple("analysis" if f.is_analysis else "reported"
                         for f in fixes),
                   tuple(_iso(f.issue) if f.issue is not None else ""
                         for f in fixes))


@dataclass(frozen=True)
class TyphoonSnapshot:
    """Everything `fetch_typhoon_dujuan.py` stored, parsed once."""
    fixes: tuple                       # AnalysisFix from the bulletin text
    track: BestTrack                   # analysed positions (3-hourly + hourly)
    official: tuple                    # OfficialForecast, sorted by issue
    rain_outlooks: tuple               # (issue, (RainOutlook, ...), title)
    specifications: tuple              # raw specification parts
    amedas: object                     # {stamp: {station: {field: value}}}
    stations: object                   # {station_id: meta}
    manifest: object
    genesis: object = None             # datetime the TD became a typhoon
    path: str = ""
    warnings: tuple = ()               # (issue hour, storm sector, gale sector)
    gust_ratio: float = GUST_RATIO_FALLBACK   # 最大瞬間風速 / 最大風速

    def warnings_at(self, hour):
        """(storm, gale) `WarningSector` published by `hour` since the epoch.

        Each area is carried forward on its own: a bulletin that restates the
        暴風域 without the 強風域 does not cancel the 強風域 an earlier
        bulletin stated.  The result is what an operator could have known at
        that moment; both are None before the first bulletin states an area.
        """
        storm = gale = None
        for h, s, g in self.warnings:
            if h > hour + 1e-9:
                break
            if s is not None:
                storm = s
            if g is not None:
                gale = g
        return storm, gale

    def sim_configs(self, hour, lat, lon):
        """Simulator configs at `hour` since the epoch for a point.

        Combines the analysed storm state with the warning areas published by
        that time, which is what `configs_at` needs to scale the rain shield
        and the wind profile.
        """
        storm, gale = self.warnings_at(hour)
        cla, clo, pres, vmax = state_at(self.track, hour)
        return configs_at(cla, clo, pres, vmax,
                          motion_dir_deg(self.track, hour), lat, lon,
                          storm=storm, gale=gale, gust_ratio=self.gust_ratio)

    def bulletin_track(self):
        """Positions exactly as the bulletin text states them.

        Independent of the analysed track array, so the two can be compared
        (and the time axis of the array checked) in the study.
        """
        return BestTrack.from_fixes(self.fixes, epoch=self.track.t0)


def _load_amedas(path):
    table, rows = {}, []
    tpath = os.path.join(path, "amedas_table_subset.json")
    if os.path.exists(tpath):
        with open(tpath, encoding="utf-8") as fh:
            table = json.load(fh)
    cpath = os.path.join(path, "amedas_stations.csv")
    if os.path.exists(cpath):
        with open(cpath, encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                rows.append(row)
    amedas = {}
    for row in rows:
        vals = {}
        for key in ("precip1h_mm", "precip3h_mm", "precip24h_mm",
                    "wind_m_s", "wind_dir_deg", "temp_C", "humidity_pct"):
            raw = row.get(key, "")
            vals[key] = float(raw) if raw not in ("", None) else float("nan")
        amedas.setdefault(row["utc"], {})[row["station_name"]] = vals
    return amedas, table


def load_snapshot(path=None):
    """Parse the committed JMA bulletin snapshot into a `TyphoonSnapshot`.

    Two independent records of the typhoon are read: the positions stated in
    the bulletin text (気象解説情報 / 台風発生報) and the analysed track array
    that ships with the 実況 part of the typhoon JSON.  The bulletin text also
    supplies the genesis time, which is what puts the track array on a time
    axis.
    """
    path = path or DATA_DIR
    fixes, official, outlooks, specs = [], [], [], []
    ddir = os.path.join(path, "denbun")
    if os.path.isdir(ddir):
        for name in sorted(os.listdir(ddir)):
            with open(os.path.join(ddir, name), encoding="utf-8") as fh:
                doc = json.load(fh)
            issue = datetime.fromisoformat(doc["reportDatetime"])
            issue = issue.astimezone(timezone.utc)
            text = "\n".join(
                item.get("textHonbun", "")
                for info in doc.get("meteorologicalInfos", [])
                for group in info.get("info", [])
                for prop in group.get("item", [])
                for item in [p for p in prop.get("property", [])])
            head = doc.get("headTitle", "")
            f, fc = parse_bulletin_text(text, issue, f"{head}|{name}")
            fixes.extend(f)
            if fc:
                official.append(OfficialForecast(issue, tuple(fc), head))
            outlook = parse_rain_outlook(doc.get("commentText", "") + text,
                                         issue)
            if outlook:
                outlooks.append((issue, outlook, head))

    analysed = []
    spec_pts, specs_issue = [], None
    spath = os.path.join(path, f"{TC_ID}_specifications.json")
    if os.path.exists(spath):
        with open(spath, encoding="utf-8") as fh:
            specs = json.load(fh)
        specs_issue = _issue_of(specs)
        for part in specs:
            if part.get("advancedHours", 0) == 0:
                fix = _spec_fix(part, specs_issue)   # 現況, not a forecast
                if fix is not None:
                    fixes.append(fix)
                continue
            pt = _spec_point(part)
            if pt is not None:
                spec_pts.append(pt)
    for fname in (f"{TC_ID}_forecastPreviousIssue.json", f"{TC_ID}_forecast.json"):
        fpath = os.path.join(path, fname)
        if not os.path.exists(fpath):
            continue
        with open(fpath, encoding="utf-8") as fh:
            parts = json.load(fh)
        issue = _issue_of(parts)
        pts = []
        for part in parts:
            if part.get("part") == "title" or part.get("advancedHours", 0) == 0:
                continue
            vt = part.get("validtime", {}).get("UTC")
            center = part.get("center")
            if vt is None or center is None:
                continue
            rad = part.get("probabilityCircle", {}).get("radius")
            pts.append(OfficialForecastPoint(
                _from_iso(vt), center[0], center[1],
                (rad / 1000.0) if rad else float("nan")))
        if issue is not None and pts and issue != specs_issue:
            official.append(OfficialForecast(issue, tuple(pts), fname))
        an = _analysis_part(parts)
        if an is not None:
            analysed.append((an, fname))
    if specs_issue is not None and spec_pts:
        # the 諸元 table restates the same issue with the intensity added
        official.append(OfficialForecast(
            specs_issue, tuple(sorted(spec_pts, key=lambda p: p.stamp)),
            f"{TC_ID}_specifications.json"))

    best = {}
    for f in sorted(fixes, key=lambda f: (f.is_analysis,
                                          f.issue or f.stamp)):
        best[(f.stamp, round(f.lat, 4), round(f.lon, 4))] = f
    fixes = sorted(best.values(), key=lambda f: f.stamp)
    official = sorted(official, key=lambda o: o.issue)
    outlooks = sorted(outlooks, key=lambda o: o[0])

    track, genesis = _analysed_track(analysed, fixes)
    amedas, stations = _load_amedas(path)
    mpath = os.path.join(path, "manifest.json")
    manifest = {}
    if os.path.exists(mpath):
        with open(mpath, encoding="utf-8") as fh:
            manifest = json.load(fh)
    # Warning areas keyed by the hour they were published, so that a forecast
    # can be replayed against what JMA had stated at the time.  Fixes of the
    # analysis itself win over the 解説情報 restating an older one.
    stated = {}
    for f in sorted(fixes, key=lambda f: (f.is_analysis, f.issue or f.stamp)):
        if f.issue is None or (f.storm is None and f.gale is None):
            continue
        stated[round(track.hour_of(f.issue), 3)] = (f.storm, f.gale)
    warnings = tuple((h, s, g) for h, (s, g) in sorted(stated.items()))
    gusts = [f for f in fixes if f.is_analysis and math.isfinite(f.gust_m_s)
             and math.isfinite(f.vmax_m_s) and f.vmax_m_s > 0.0]
    gust_ratio = gusts[-1].gust_m_s / gusts[-1].vmax_m_s if gusts \
        else GUST_RATIO_FALLBACK

    return TyphoonSnapshot(tuple(fixes), track, tuple(official),
                           tuple(outlooks), tuple(specs), amedas, stations,
                           manifest, genesis=genesis, path=path,
                           warnings=warnings, gust_ratio=gust_ratio)


def _issue_of(parts):
    for part in parts:
        if part.get("part") == "title":
            return _from_iso(part["issue"]["UTC"])
    return None


def _analysis_part(parts):
    for part in parts:
        name = part.get("part")
        if (isinstance(name, dict) and name.get("en") == "Analysis"
                and part.get("advancedHours", 0) == 0):
            return part
    return None


def _spec_fix(part, issue):
    """`AnalysisFix` from the 現況 (advancedHours 0) part of the 諸元 table."""
    vt = part.get("validtime", {}).get("UTC")
    pos = part.get("position", {}).get("deg")
    pres = part.get("pressure")
    if not vt or not pos or not pres:
        return None
    wind = part.get("maximumWind", {})
    course = _COURSE.get(part.get("course", ""))
    return AnalysisFix(
        _from_iso(vt), pos[0], pos[1], float(pres),
        float(wind.get("sustained", {}).get("m/s", "nan")),
        float(wind.get("gust", {}).get("m/s", "nan")), course,
        float(part.get("speed", {}).get("km/h", "nan")),
        issue, "specifications.json",
        _spec_sector(part, "stormWarning"),
        _spec_sector(part, "galeWarning"))


def _spec_sector(part, key):
    """`WarningSector` from the galeWarning / stormWarning array of a 諸元 part.

    The array holds one entry for a symmetric area (「全域」) or two for the
    half-discs JMA analysed; the compass name of the first entry is the
    direction the sector faces.
    """
    entries = []
    for e in part.get(key) or []:
        rng = e.get("range") or {}
        if "km" not in rng:
            continue
        area = e.get("area")
        name = area.get("jp") if isinstance(area, dict) else area
        entries.append((_COURSE.get(str(name)) if name else None,
                        float(rng["km"])))
    if not entries:
        return None
    if len(entries) == 1 or entries[0][0] is None:
        r = max(r for _, r in entries)
        return WarningSector(0.0, r, r)
    return WarningSector(entries[0][0], entries[0][1], entries[-1][1])


def _spec_point(part):
    """`OfficialForecastPoint` from one 諸元 (specifications) part.

    The 諸元 table is the only JMA product that publishes the intensity of a
    forecast position, so it is what turns the official forecast circles into
    a full pressure/wind reference.
    """
    vt = part.get("validtime", {}).get("UTC")
    pos = part.get("position", {}).get("deg")
    if not vt or not pos:
        return None

    def _warn(key):
        vals = [float(r["range"]["km"]) for r in part.get(key, [])
                if isinstance(r.get("range"), dict) and "km" in r["range"]]
        return max(vals) if vals else float("nan")

    wind = part.get("maximumWind", {})
    rad = part.get("probabilityCircleRadius", {}).get("km")
    cat = part.get("category", {})
    course = _COURSE.get(part.get("course", ""))
    return OfficialForecastPoint(
        _from_iso(vt), pos[0], pos[1],
        float(rad) if rad else float("nan"),
        float(part.get("pressure", "nan")),
        float(wind.get("sustained", {}).get("m/s", "nan")),
        float(wind.get("gust", {}).get("m/s", "nan")),
        float(part.get("speed", {}).get("km/h", "nan")),
        float(course) if course is not None else float("nan"),
        cat.get("en", "") if isinstance(cat, dict) else str(cat or ""),
        _warn("galeWarning"), _warn("stormWarning"),
        _spec_sector(part, "galeWarning"), _spec_sector(part, "stormWarning"))


def _analysed_track(analysed, fixes):
    """`BestTrack` of the analysed positions, plus the genesis time.

    The longest track array wins (it is the most recent bulletin); the genesis
    bulletin fixes the start of the time axis, and the central pressure and
    maximum wind of every position come from the bulletin that states them.
    """
    if not analysed:
        raise ValueError("snapshot holds no analysed typhoon track")
    part, source = max(analysed, key=lambda a: len(a[0]["track"]["typhoon"]))
    pts = part["track"]["typhoon"]
    validtime = _from_iso(part["validtime"]["UTC"])
    genesis = None
    for fix in fixes:                       # earliest bulletin position that
        if (abs(fix.lat - pts[0][0]) <= 0.11      # is the genesis point
                and abs(fix.lon - pts[0][1]) <= 0.11):
            genesis = fix.stamp
            break
    times, n3 = track_times(len(pts), validtime, genesis)
    epoch = times[0]
    hours = np.array([(t - epoch).total_seconds() / 3600.0 for t in times],
                     dtype=float)
    pres = np.full(len(pts), np.nan)
    vmax = np.full(len(pts), np.nan)
    issues = []
    for i, t in enumerate(times):
        near = [fix for fix in fixes
                if abs((fix.stamp - t).total_seconds()) <= 1800.0
                and math.isfinite(fix.pres_hPa)]
        issue = None
        if near:
            fix = min(near, key=lambda f: abs((f.stamp - t).total_seconds()))
            pres[i], vmax[i] = fix.pres_hPa, fix.vmax_m_s
            issue = fix.issue
        issues.append(_iso(issue) if issue is not None else "")
    kinds = tuple(["analysis"] * n3 + ["estimate"] * (len(pts) - n3))
    track = BestTrack(tuple(_iso(t) for t in times), hours,
                      np.array([p[0] for p in pts], dtype=float),
                      np.array([p[1] for p in pts], dtype=float),
                      pres, vmax, f"JMA 実況 ({source})", epoch, kinds,
                      tuple(issues))
    return track, genesis


# --------------------------------------------------------------------------
# track forecast model
# --------------------------------------------------------------------------

def _resample(track, grid_hours=GRID_HOURS, end_hour=None):
    """Linear resample of the analysed positions onto a forecast grid.

    The grid ends exactly at `end_hour` (the last analysed position by
    default), so the issue time of a forecast is always a grid point and every
    motion sample that feeds the model spans exactly `grid_hours`.
    """
    t0, t1 = float(track.hours[0]), float(track.hours[-1])
    end = t1 if end_hour is None else float(end_hour)
    if end > t1 + 1e-9:
        raise ValueError("grid end beyond the analysed track")
    if end < t0 + grid_hours - 1e-9:
        raise ValueError("track shorter than one grid step")
    k = int(math.floor((end - t0) / grid_hours + 1e-9))
    g = end - grid_hours * np.arange(k, -1, -1, dtype=float)
    lat = np.interp(g, track.hours, track.lat)
    lon = np.interp(g, track.hours, track.lon)
    pres = np.array([_interp_known(x, track.hours, track.pres_hPa) for x in g])
    vmax = np.array([_interp_known(x, track.hours, track.vmax_m_s) for x in g])
    return g, lat, lon, pres, vmax


def _motion_km(g, lat, lon):
    """(east, north) displacement per grid step, km."""
    out = np.zeros((len(g) - 1, 2))
    for k in range(len(g) - 1):
        mid = math.radians(0.5 * (lat[k] + lat[k + 1]))
        dy = (lat[k + 1] - lat[k]) * 111.32
        dx = (lon[k + 1] - lon[k]) * 111.32 * math.cos(mid)
        out[k] = (dx, dy)
    return out


def vmax_from_pressure(pres_hPa, ref_pres_hPa, ref_vmax_m_s):
    """Maximum wind from a central pressure, anchored on a known pair.

    The (p, vmax) pairs the bulletins state for this typhoon follow a power
    law in the pressure deficit: a least-squares fit of ln vmax on
    ln(1013 - p) over all ten of them gives an exponent of 0.58, for which
    `VMAX_PRES_EXP` = 0.5 is the round stand-in.  Anchoring the scaling on
    the analysed pair then reproduces every stated pair within 2 m/s (998
    hPa / 18 m/s reads as 19.6, 970 hPa / 35 m/s as 33.1), lets lead 0 be
    exact, and lets the intensity follow the forecast pressure.
    """
    dref = max(1.0, 1013.0 - float(ref_pres_hPa))
    d = max(0.0, 1013.0 - float(pres_hPa))
    return float(ref_vmax_m_s) * (d / dref) ** VMAX_PRES_EXP


def et_relax_rate(lat):
    """Per-step pull of the central pressure towards `PRES_RELAX_HPA`.

    A storm inside the tropics keeps its intensity; the relaxation switches on
    with latitude, the only proxy for the extratropical transition the
    bulletins themselves offer (they never state a transition time for this
    typhoon, but the 諸元 table turns it into a 温帯低気圧 north of 40N).
    """
    f = (float(lat) - ET_LAT_START) / max(1e-9, ET_LAT_END - ET_LAT_START)
    return PRES_RELAX_RATE * min(1.0, max(0.0, f))


class TrackForecastModel:
    """Ridge VAR on the motion of the JMA analysed track.

    The state is the `grid_hours`-step (east, north) displacement in km.  A
    VAR(p) is fitted by ridge least squares whose penalty is centred on the
    motion-persistence matrix -- the next displacement being the mean of the
    last `p` -- so the fit only moves away from persistence where the record
    supports it; `prior_weight` is that trust, in units of the data term.  The
    companion matrix is then shrunk until no mode grows (spectral radius <=
    `max_radius`), the same guard `weather_forecast.VarForecastModel` uses.
    A forecast integrates the predicted displacements from the analysed
    position at the issue time, so lead 0 reproduces the analysis exactly.

    The central pressure follows the recent trend of the bulletin pressures,
    damped by `PRES_DECAY` per step, plus the latitude-dependent relaxation of
    `et_relax_rate` towards `PRES_RELAX_HPA`; the maximum wind follows the
    pressure through `vmax_from_pressure`.

    `train_stop` is the issue time (datetime, ISO string or hour since the
    track epoch).  Only positions at or before it are used, which makes the
    model usable for rolling-origin hindcasts that never see the future.
    `window_hours` limits the fit to the leg just before the issue time: a
    recurving typhoon is then fitted on the motion it is still carrying out
    instead of on the whole history, whose single big turn the recursion would
    otherwise replay as a spurious oscillation.  The window is ignored when it
    would leave too few motion samples.
    """

    def __init__(self, track, order=TRACK_ORDER, train_stop=None,
                 prior_weight=PRIOR_WEIGHT, window_hours=TRACK_WINDOW_HOURS,
                 max_radius=MAX_RADIUS, grid_hours=GRID_HOURS):
        self.track = track
        self.order = int(order)
        if self.order < 1:
            raise ValueError("order must be >= 1")
        self.grid_hours = float(grid_hours)
        self.prior_weight = float(prior_weight)
        self.max_radius = float(max_radius)
        self.epoch = track.t0
        self.issue_hour = float(track.hours[-1]) if train_stop is None \
            else _as_hour(train_stop, self.epoch)
        if self.issue_hour > float(track.hours[-1]) + 1e-9:
            raise ValueError("train_stop beyond the analysed track")
        used = track if train_stop is None else track.upto(self.issue_hour)
        used = used.tail_segment()
        self.window_hours = None
        if window_hours is not None:
            keep = used.hours >= self.issue_hour - float(window_hours) - 1e-9
            if int(keep.sum()) - 1 >= max(MIN_TRAIN_STEPS, self.order + 4):
                used = used._slice(keep)
                self.window_hours = float(window_hours)
        self.used = used
        (self.grid, self.g_lat, self.g_lon, self.g_pres,
         self.g_vmax) = _resample(self.used, self.grid_hours,
                                  end_hour=self.issue_hour)
        self.motion = _motion_km(self.grid, self.g_lat, self.g_lon)
        if len(self.motion) < MIN_TRAIN_STEPS \
                or len(self.motion) - self.order < 4:
            raise ValueError(
                f"issue time leaves {len(self.motion)} motion samples "
                f"(need >= {MIN_TRAIN_STEPS})")
        self.train_mean = self.motion.mean(axis=0)
        p = self.order
        rows, targets = [], []
        for k in range(p, len(self.motion)):
            rows.append(self.motion[k - p:k].ravel())
            targets.append(self.motion[k])
        X = np.asarray(rows)
        Y = np.asarray(targets)
        self.prior = np.zeros((2 * p, 2))
        for b in range(p):
            self.prior[2 * b:2 * b + 2, :] = np.eye(2) / p
        w = self.prior_weight * float(np.trace(X.T @ X)) / (2 * p)
        self.coef = np.linalg.solve(X.T @ X + w * np.eye(2 * p),
                                    X.T @ Y + w * self.prior)
        self.resid_rms = float(np.sqrt(((X @ self.coef - Y) ** 2).mean()))
        self.shrink_scale = 1.0
        for _ in range(200):
            if self._companion_radius(self.coef * self.shrink_scale) \
                    <= self.max_radius + 1e-9:
                break
            self.shrink_scale *= 0.97
        self.coef = self.coef * self.shrink_scale

    def _companion_radius(self, coef):
        p = self.order
        n = 2 * p
        C = np.zeros((n, n))
        C[:2, :] = coef.T
        if n > 2:
            C[2:, :n - 2] = np.eye(n - 2)
        return float(np.max(np.abs(np.linalg.eigvals(C))))

    def companion_radius(self):
        return self._companion_radius(self.coef)

    def pressure_rate(self):
        """Central-pressure trend of the bulletins, hPa per hour."""
        if not np.isfinite(self.g_pres).any():
            return 0.0
        i = len(self.g_pres) - 1
        for j in range(i - 1, -1, -1):
            span = self.grid[i] - self.grid[j]
            if span >= 6.0 - 1e-9 and np.isfinite(self.g_pres[j]):
                return float((self.g_pres[i] - self.g_pres[j]) / span)
        return 0.0

    def predict(self, horizon_hours):
        """Forecast positions, central pressure and maximum wind.

        Every array holds `horizon/grid_hours + 1` entries, the first being
        the analysed state at the issue time.
        """
        n = int(round(float(horizon_hours) / self.grid_hours))
        if n < 1:
            raise ValueError("horizon shorter than one grid step")
        p = self.order
        hist = [self.motion[k]
                for k in range(len(self.motion) - p, len(self.motion))]
        lat0, lon0 = float(self.g_lat[-1]), float(self.g_lon[-1])
        pres0 = float(self.g_pres[-1])
        vmax0 = float(self.g_vmax[-1])
        if not math.isfinite(pres0):
            # no bulletin states a central pressure yet
            pres0 = PRES_RELAX_HPA
        cur_lat, cur_lon, cur_pres = lat0, lon0, pres0
        dp_h = self.pressure_rate()
        lats, lons, pres = [lat0], [lon0], [cur_pres]
        for step in range(n):
            nxt = self.coef.T @ np.concatenate(hist[-p:])
            hist.append(nxt)
            mid = math.radians(cur_lat)
            cur_lon += nxt[0] / (111.32 * math.cos(mid))
            cur_lat += nxt[1] / 111.32
            cur_pres += (PRES_DECAY ** step) * dp_h * self.grid_hours
            if cur_pres < PRES_RELAX_HPA:
                cur_pres += et_relax_rate(cur_lat) * (PRES_RELAX_HPA
                                                      - cur_pres)
            cur_pres = float(min(PRES_CEIL, max(PRES_FLOOR, cur_pres)))
            lats.append(cur_lat)
            lons.append(cur_lon)
            pres.append(cur_pres)
        pres_arr = np.array(pres)
        if math.isfinite(vmax0):
            vmax_arr = np.array([vmax_from_pressure(x, pres0, vmax0)
                                 for x in pres_arr])
        else:
            vmax_arr = np.full(n + 1, FALLBACK_VMAX_M_S)
        hours = np.arange(n + 1) * self.grid_hours
        return {"hours": hours, "lat": np.array(lats), "lon": np.array(lons),
                "pres_hPa": pres_arr, "vmax_m_s": vmax_arr,
                "issue_hour": self.issue_hour}


def _fallback_vmax(vmax):
    v = float(vmax)
    return v if math.isfinite(v) else FALLBACK_VMAX_M_S


def persistence_track(track, issue, horizon_hours, grid_hours=GRID_HOURS):
    """Hold the analysed position (the classical persistence reference)."""
    ih = _as_hour(issue, track.t0)
    _, lat, lon, pres, vmax = _resample(track, grid_hours, end_hour=ih)
    n = int(round(float(horizon_hours) / grid_hours))
    hours = np.arange(n + 1) * grid_hours
    p0 = float(pres[-1]) if math.isfinite(pres[-1]) else PRES_RELAX_HPA
    return {"hours": hours, "lat": np.full(n + 1, lat[-1]),
            "lon": np.full(n + 1, lon[-1]), "pres_hPa": np.full(n + 1, p0),
            "vmax_m_s": np.full(n + 1, _fallback_vmax(vmax[-1])),
            "issue_hour": ih}


def motion_persistence_track(track, issue, horizon_hours, window=2,
                             grid_hours=GRID_HOURS):
    """Continue with the mean motion of the last `window` grid steps."""
    ih = _as_hour(issue, track.t0)
    g, lat, lon, pres, vmax = _resample(track, grid_hours, end_hour=ih)
    motion = _motion_km(g, lat, lon)
    m = motion[max(0, len(motion) - int(window)):].mean(axis=0)
    n = int(round(float(horizon_hours) / grid_hours))
    cur_lat, cur_lon = float(lat[-1]), float(lon[-1])
    lats, lons = [cur_lat], [cur_lon]
    for _ in range(n):
        mid = math.radians(cur_lat)
        cur_lon += m[0] / (111.32 * math.cos(mid))
        cur_lat += m[1] / 111.32
        lats.append(cur_lat)
        lons.append(cur_lon)
    hours = np.arange(n + 1) * grid_hours
    p0 = float(pres[-1]) if math.isfinite(pres[-1]) else PRES_RELAX_HPA
    return {"hours": hours, "lat": np.array(lats), "lon": np.array(lons),
            "pres_hPa": np.full(n + 1, p0),
            "vmax_m_s": np.full(n + 1, _fallback_vmax(vmax[-1])),
            "issue_hour": ih}


def track_error_km(fc, truth_track):
    """Position error (km) of a forecast dict against the analysis track,
    evaluated at every forecast valid time that the analysis covers."""
    errs = []
    hours = []
    for h, la, lo in zip(fc["hours"], fc["lat"], fc["lon"]):
        th = fc["issue_hour"] + h
        if th > truth_track.hours[-1] + 1e-9:
            break
        k = truth_track.index_at(th)
        if abs(truth_track.hours[k] - th) > 1e-6:
            tla = np.interp(th, truth_track.hours, truth_track.lat)
            tlo = np.interp(th, truth_track.hours, truth_track.lon)
        else:
            tla, tlo = truth_track.lat[k], truth_track.lon[k]
        errs.append(haversine_km(la, lo, tla, tlo))
        hours.append(h)
    return np.array(hours), np.array(errs)


# --------------------------------------------------------------------------
# rainfall field, wind vortex and simulator snapshots
# --------------------------------------------------------------------------

def orographic_factor(lat, lon):
    gain = 1.0
    for alat, alon, sigma, g in OROGRAPHIC_ANCHORS:
        d = haversine_km(lat, lon, alat, alon)
        gain += g * math.exp(-(d / sigma) ** 2)
    return gain


def rain_rate_mm_h(center_lat, center_lon, pres_hPa, motion_dir_deg,
                   lat, lon, gale=None):
    """Parametric typhoon rain rate [mm/h] at a point.

    The shield is a disc whose e-folding radius is `RAIN_DECAY_FRAC` of the
    強風域 radius in the direction of the point and whose centre is displaced
    `RAIN_LEAD_FRAC` of that radius forward along the motion, so the rain
    falls ahead of the centre and dies out near the edge of the gale area.
    The amplitude scales with the central-pressure deficit and carries the
    orographic gain of the terrain.  The three constants were fitted to the
    AMeDAS observations of this typhoon (980 station-hours: RMSE 3.85 mm/h,
    correlation 0.51, bias +0.01 mm/h).

    `gale` is a `WarningSector`; a circular fallback stands in when no
    bulletin states a 強風域.
    """
    d = haversine_km(center_lat, center_lon, lat, lon)
    phi = bearing_deg(center_lat, center_lon, lat, lon)
    rg = sector_radius(gale, phi)
    if not math.isfinite(rg) or rg <= 0.0:
        rg = FALLBACK_GALE_KM
    md = 0.0 if motion_dir_deg is None or not math.isfinite(motion_dir_deg) \
        else float(motion_dir_deg)
    psi = math.radians(phi - md)
    rr = math.hypot(d * math.sin(psi),
                    d * math.cos(psi) - RAIN_LEAD_FRAC * rg)
    amp = min(RAIN_MAX_MM_H, RAIN_SCALE * max(0.0, 1013.0 - pres_hPa))
    return max(0.0, amp * math.exp(-rr / (RAIN_DECAY_FRAC * rg))
               * orographic_factor(lat, lon) ** RAIN_ORO_POWER)


def wind_profile(vmax_m_s, storm_km, gale_km, pres_hPa=float("nan")):
    """(rmax km, radial exponent b) of the profile the warning areas imply.

    For V(r) = vmax * (rmax/r)**b outside the core, JMA's two warning radii
    over-determine the profile: 25 m/s at the 暴風域 edge and 15 m/s at the
    強風域 edge fix b, and the 暴風域 radius then fixes rmax.  For the
    analysis of this typhoon (35 m/s, 暴風域 185 km all round, 強風域
    東650 / 西440 km) that gives b = 0.47 and rmax = 91 km.  With only one
    radius stated, `WIND_POWER_FALLBACK` is used; with none, a pressure-based
    guess.
    """
    vmax = float(vmax_m_s) if vmax_m_s is not None else float("nan")
    rs = float(storm_km) if storm_km is not None else float("nan")
    rg = float(gale_km) if gale_km is not None else float("nan")
    b = WIND_POWER_FALLBACK
    if math.isfinite(vmax) and vmax > STORM_WIND_M_S \
            and math.isfinite(rs) and math.isfinite(rg) and rg > rs > 0.0:
        b = math.log(STORM_WIND_M_S / GALE_WIND_M_S) / math.log(rg / rs)
        b = min(max(b, WIND_POWER_RANGE[0]), WIND_POWER_RANGE[1])
    if math.isfinite(vmax) and math.isfinite(rs) and rs > 0.0 \
            and vmax > STORM_WIND_M_S:
        return rs * (STORM_WIND_M_S / vmax) ** (1.0 / b), b
    if math.isfinite(vmax) and math.isfinite(rg) and rg > 0.0 \
            and vmax > GALE_WIND_M_S:
        return rg * (GALE_WIND_M_S / vmax) ** (1.0 / b), b
    p = float(pres_hPa) if pres_hPa is not None else float("nan")
    if not math.isfinite(p):
        p = 1010.0
    return (RMAX_FALLBACK_KM + RMAX_PER_HPA * max(0.0, 1010.0 - p),
            WIND_POWER_FALLBACK)


def gale_sector_from_storm(storm, vmax_m_s, pres_hPa=float("nan")):
    """強風域 sector implied by a 暴風域 sector and the fitted profile.

    JMA states the 暴風域 of every forecast position but stops stating the
    強風域 after the first hours, so the outer radius is recovered from the
    same power law instead of being left undefined.
    """
    if storm is None or vmax_m_s is None or not math.isfinite(vmax_m_s) \
            or vmax_m_s <= GALE_WIND_M_S:
        return None
    rmax, b = wind_profile(vmax_m_s, storm.radius_km, float("nan"), pres_hPa)
    rg = rmax * (vmax_m_s / GALE_WIND_M_S) ** (1.0 / b)
    if not math.isfinite(rg) or rg <= 0.0:
        return None
    scale = rg / storm.mean_radius()
    return WarningSector(storm.dir_deg, storm.radius_km * scale,
                         (storm.opposite_km if math.isfinite(storm.opposite_km)
                          else storm.radius_km) * scale)


def vortex_wind(center_lat, center_lon, vmax_m_s, pres_hPa, lat, lon,
                storm=None, gale=None, inflow_deg=20.0):
    """(speed m/s, meteorological FROM-direction deg) of a Rankine-style
    cyclonic vortex with inflow, at a point.

    The core radius and the radial exponent follow the 暴風域 / 強風域 radii
    the bulletins state in the direction of the point, so the vortex inherits
    the asymmetry JMA analysed instead of a fixed circular guess.

    `inflow_deg` turns the cyclonic tangential flow towards the centre, the
    way Northern-Hemisphere surface inflow does, so a point north of the
    centre gets wind from the ENE rather than from due east.
    """
    d = haversine_km(center_lat, center_lon, lat, lon)
    if d <= 1e-9:
        return 0.0, 0.0
    vmax = float(vmax_m_s)
    if not math.isfinite(vmax) or vmax <= 0.0:
        vmax = FALLBACK_VMAX_M_S
    phi = bearing_deg(center_lat, center_lon, lat, lon)
    rmax, b = wind_profile(vmax, sector_radius(storm, phi),
                           sector_radius(gale, phi), pres_hPa)
    speed = vmax * (d / rmax) if d < rmax else vmax * (rmax / d) ** b
    heading = phi - 90.0 - inflow_deg      # cyclonic, spiralling inwards
    from_dir = (heading + 180.0) % 360.0
    return speed, from_dir


def motion_dir_deg(track, hour):
    """Analysis motion direction (deg from N) at a time on the grid."""
    g, lat, lon, _, _ = _resample(track)
    k = int(np.argmin(np.abs(g - hour)))
    k = min(k, len(g) - 2)
    return bearing_deg(lat[k], lon[k], lat[k + 1], lon[k + 1])


def state_at(track, hour):
    """Interpolated analysed (lat, lon, pres, vmax) at a time [h since epoch].

    Pressure and maximum wind are only stated in some bulletins, so they are
    interpolated through the stated values alone.
    """
    return (float(np.interp(hour, track.hours, track.lat)),
            float(np.interp(hour, track.hours, track.lon)),
            _interp_known(hour, track.hours, track.pres_hPa),
            _interp_known(hour, track.hours, track.vmax_m_s))


def rain_terms(rain_mm_h):
    """The rain-dependent `WeatherConfig` fields for a rain rate in mm/h.

    One place states how a rain rate becomes humidity, visibility, cloud cover
    and lightning risk, so that `configs_at` and `scale_rain` cannot drift.
    """
    rain = max(0.0, float(rain_mm_h))
    return dict(
        rain_mm_h=rain,
        humidity=min(1.0, 0.80 + 0.004 * rain),
        visibility_m=max(1500.0, 20000.0 - 350.0 * rain),
        cloud_cover=min(1.0, 0.4 + 0.02 * rain),
        lightning_risk=min(1.0, rain / 40.0))


def scale_rain(weather, k):
    """`weather` with its rain rate multiplied by `k`, terms carried along.

    Used to turn the model rain field into the anchored product: JMA states a
    magnitude ("up to N mm") and the model states the distribution, so the
    product scales the model field to JMA's number.  Scaling only `rain_mm_h`
    would leave a sky that contradicts its own rain, hence `rain_terms`.
    """
    if abs(float(k) - 1.0) < 1e-9:
        return weather
    return replace(weather, **rain_terms(float(weather.rain_mm_h) * float(k)))


def configs_at(center_lat, center_lon, pres_hPa, vmax_m_s, motion_deg,
               lat, lon, storm=None, gale=None,
               gust_ratio=GUST_RATIO_FALLBACK):
    """Simulator-ready (WeatherConfig, AtmosphereConfig) for a point, given
    the storm centre, its intensity and its direction of motion.

    `gust_ratio` is the 最大瞬間風速 / 最大風速 ratio of the bulletin.  It is
    turned into `gust_rms` with the `weather_real.GUST_SIGMA_FACTOR`
    calibration, so that the one-hour peak of the simulator gust process
    reproduces the stated 瞬間風速 where the vortex wind equals `vmax_m_s`.
    """
    rain = rain_rate_mm_h(center_lat, center_lon, pres_hPa, motion_deg,
                          lat, lon, gale=gale)
    speed, from_dir = vortex_wind(center_lat, center_lon, vmax_m_s,
                                  pres_hPa, lat, lon, storm=storm, gale=gale)
    theta = math.radians(from_dir)
    u_north = -speed * math.cos(theta)
    u_east = -speed * math.sin(theta)
    weather = WeatherConfig(
        temp_offset_K=1.5,
        pressure_offset_Pa=(pres_hPa - 1013.25) * 100.0,
        cloud_base_m=600.0, cloud_top_m=8000.0,
        **rain_terms(rain))
    ratio = float(gust_ratio) if math.isfinite(gust_ratio) else 1.0
    atmo = AtmosphereConfig(wind=(u_north, u_east, 0.0),
                            gust_rms=GUST_SIGMA_FACTOR
                            * max(0.0, ratio - 1.0) * speed)
    return weather, atmo


def typhoon_configs(track, hour, lat, lon, storm=None, gale=None,
                    gust_ratio=GUST_RATIO_FALLBACK):
    """`configs_at` driven by the analysed storm state at `hour`.

    `TyphoonSnapshot.sim_configs` is the form that also looks the warning
    areas up; this one takes them from the caller.
    """
    cla, clo, pres, vmax = state_at(track, hour)
    return configs_at(cla, clo, pres, vmax, motion_dir_deg(track, hour),
                      lat, lon, storm=storm, gale=gale, gust_ratio=gust_ratio)


def forecast_configs(fc, lead_hours, lat, lon, vmax_m_s=None, storm=None,
                     gale=None, gust_ratio=GUST_RATIO_FALLBACK):
    """`configs_at` driven by a forecast dict at a lead time [h]."""
    k = int(np.argmin(np.abs(np.asarray(fc["hours"]) - lead_hours)))
    vmax = fc.get("vmax_m_s")
    if vmax is None:
        vmax = vmax_m_s
    else:
        vmax = np.asarray(vmax, dtype=float).ravel()[
            min(k, len(np.asarray(vmax).ravel()) - 1)]
    if vmax is None or not math.isfinite(float(vmax)):
        vmax = FALLBACK_VMAX_M_S
    mdir = bearing_deg(fc["lat"][k], fc["lon"][k],
                       fc["lat"][min(k + 1, len(fc["lat"]) - 1)],
                       fc["lon"][min(k + 1, len(fc["lon"]) - 1)])
    if gale is None and storm is not None:
        gale = gale_sector_from_storm(storm, float(vmax), fc["pres_hPa"][k])
    return configs_at(fc["lat"][k], fc["lon"][k], fc["pres_hPa"][k],
                      float(vmax), mdir, lat, lon, storm=storm, gale=gale,
                      gust_ratio=gust_ratio)
