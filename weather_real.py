"""Historical weather replay driven by real NOAA NDBC buoy observations.

`parse_ndbc_met` reads the meteorological columns of an NDBC realtime2
.txt file (the same hourly files ocean_real.py already uses for waves):

    WDIR  degT   direction the wind blows FROM        -> wind vector
    WSPD  m/s    hourly mean wind speed at ~10 m      -> mean wind at z_ref
    GST   m/s    peak gust speed within the hour      -> turbulence level
    PRES  hPa    sea-level pressure                   -> pressure_offset_Pa
    ATMP  degC   air temperature                      -> temp_offset_K
    DEWP  degC   dew point                            -> relative humidity

`HistoricalWeather` is a drop-in `Weather` subclass whose temperature,
pressure, humidity and wind evolve with simulation time by linear
interpolation of the record, so spatial_dynamics / env / mavlink_if
replay an observed 45-day weather history instead of a steady preset.

Conventions and assumptions (documented, deterministic):

* Frame mapping: simulation +x = true north, +y = true east, +z = up.
  A meteorological direction theta (degT, wind FROM) becomes the flow
  vector -WSPD * (cos theta, sin theta) in (north, east).
* Wind direction is interpolated as (u, v) components, never as degrees,
  so the 359 deg -> 1 deg wrap cannot spin the wind the long way round.
* GST - WSPD is the hourly peak speed excess.  The sum4 gust process of
  atmosphere.py reaches a 1-hour peak of ~2.755 * gust_rms (calibrated
  in weather_study.py over the bundled record), hence
  gust_rms = GUST_SIGMA_FACTOR * (GST - WSPD).
* PRES/ATMP anchor the ISA profile at sea level: temperature(h) =
  isa_temperature(h) + (ATMP - 15 C), pressure(h) = isa_pressure(h) +
  (PRES - 1013.25 hPa).  Humidity is held with height (Weather model).
* Observations are at the buoy anemometer height (~10 m = z_ref), so the
  log-shear profile scales them to flight altitude unchanged.
* Columns the record does not carry (rain, cloud, visibility) keep the
  values of the base WeatherConfig passed at construction; a dry replay
  is the honest default for an observational file with VIS = MM.
* Before the first / after the last record the edge values are held.

Everything is analytic and deterministic: no RNG beyond the fixed gust
phases of the underlying Atmosphere (same seed => same series).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from atmosphere import AtmosphereConfig
from weather import Weather, WeatherConfig

# ----- calibration and reference constants -----
GUST_SIGMA_FACTOR = 0.363   # gust_rms = factor * (GST - WSPD); 1/2.755,
                            # the measured 1-hour peak excess of the sum4
                            # gust process per unit RMS (weather_study.py)
P_REF_HPA = 1013.25         # ISA sea-level pressure, hPa
T_REF_C = 15.0              # ISA sea-level temperature, degC

DEFAULT_MET_PATH = (Path(__file__).resolve().parent / "data"
                    / "ndbc_46012_realtime.txt")

# realtime2 .txt column indices (0-based)
_IX = dict(year=0, month=1, day=2, hour=3, minute=4, wdir=5, wspd=6,
           gust=7, pres=12, atmp=13, wtmp=14, dewp=15)
_REQUIRED = ("wdir", "wspd", "pres", "atmp")
_OPTIONAL = ("gust", "dewp", "wtmp")


def _saturation_vapor_pressure_C(T_C):
    """Vectorised Magnus formula (same constants as weather.py), Pa."""
    T = np.asarray(T_C, dtype=float)
    return np.where(
        T >= 0.0,
        611.2 * np.exp(17.62 * T / (243.12 + T)),
        611.2 * np.exp(22.46 * T / (272.62 + T)))


def _fill_missing(hours, col):
    """Linear interpolation of NaN ('MM') entries; None if all missing."""
    good = np.isfinite(col)
    if not good.any():
        return None
    if good.all():
        return col
    return np.interp(hours, hours[good], col[good])


@dataclass(frozen=True, eq=False)
class MetSeries:
    """Chronological hourly meteorological record (SI-ish units)."""

    hours: np.ndarray        # elapsed hours since the first record
    stamps: tuple            # 'YYYY-MM-DD HH:MM' UTC per record
    wdir_deg: np.ndarray     # meteorological direction, degT (FROM)
    wspd_m_s: np.ndarray     # hourly mean wind speed
    gust_m_s: np.ndarray | None    # hourly peak gust (None if unreported)
    pres_hPa: np.ndarray
    atmp_C: np.ndarray
    dewp_C: np.ndarray | None
    wtmp_C: np.ndarray | None
    rh: np.ndarray | None    # relative humidity 0..1 from ATMP/DEWP
    u_north: np.ndarray      # wind vector components, m/s
    v_east: np.ndarray
    label: str = ""
    path: str = ""

    def __len__(self):
        return len(self.hours)

    @property
    def span_hours(self):
        return float(self.hours[-1] - self.hours[0])

    def gust_excess(self):
        """GST - WSPD per record (zeros when gusts are unreported)."""
        if self.gust_m_s is None:
            return np.zeros_like(self.wspd_m_s)
        return np.maximum(0.0, self.gust_m_s - self.wspd_m_s)

    def stamp_at(self, hour):
        """UTC stamp of the record nearest to `hour` (elapsed hours)."""
        h = min(max(float(hour), self.hours[0]), self.hours[-1])
        return self.stamps[int(np.argmin(np.abs(self.hours - h)))]

    def interp(self, hour):
        """All observed quantities linearly interpolated at `hour`.

        Wind is interpolated through its components; the reported
        wspd/wdir are reconstructed from them so they always agree with
        the vector the simulator consumes.
        """
        h = min(max(float(hour), self.hours[0]), self.hours[-1])
        xp = self.hours
        u = float(np.interp(h, xp, self.u_north))
        v = float(np.interp(h, xp, self.v_east))
        wspd = math.hypot(u, v)
        wdir = (math.degrees(math.atan2(-v, -u))) % 360.0
        out = dict(hour=h, stamp=self.stamp_at(h),
                   u_north=u, v_east=v,
                   wspd_m_s=wspd, wdir_deg=wdir,
                   pres_hPa=float(np.interp(h, xp, self.pres_hPa)),
                   atmp_C=float(np.interp(h, xp, self.atmp_C)))
        if self.gust_m_s is not None:
            g = float(np.interp(h, xp, self.gust_m_s))
            w = float(np.interp(h, xp, self.wspd_m_s))
            out["gust_m_s"] = g
            out["gust_excess_m_s"] = max(0.0, g - w)
        else:
            out["gust_m_s"] = None
            out["gust_excess_m_s"] = 0.0
        if self.rh is not None:
            out["rh"] = float(np.interp(h, xp, self.rh))
            out["dewp_C"] = float(np.interp(h, xp, self.dewp_C))
        else:
            out["rh"] = None
            out["dewp_C"] = None
        return out


def parse_ndbc_met(path=None, label=None):
    """Parse the meteorological columns of an NDBC realtime2 .txt file.

    The file is newest-first on disk; the returned series is
    chronological.  'MM' entries are NaN and get linearly interpolated
    between valid neighbours (edge-held at the ends); a required column
    without any valid value raises ValueError.
    """
    path = Path(path) if path is not None else DEFAULT_MET_PATH
    if not path.exists():
        raise FileNotFoundError(f"no NDBC realtime file at {path}")
    rows = []
    with open(path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            if len(parts) < 14:          # need through ATMP (index 13)
                continue
            try:
                stamp = datetime(int(parts[0]), int(parts[1]), int(parts[2]),
                                 int(parts[3]), int(parts[4]),
                                 tzinfo=timezone.utc)
            except ValueError:
                continue
            rows.append((stamp.timestamp(),
                         stamp.strftime("%Y-%m-%d %H:%M"), parts))
    if len(rows) < 2:
        raise ValueError(f"{path}: need at least 2 valid records")
    rows.sort(key=lambda r: r[0])        # chronological, stable
    epoch = np.array([r[0] for r in rows])
    parts_all = [r[2] for r in rows]
    hours = (epoch - epoch[0]) / 3600.0
    stamps = tuple(r[1] for r in rows)

    def col(name):
        idx = _IX[name]
        vals = []
        for p in parts_all:
            tok = p[idx] if idx < len(p) else "MM"
            try:
                vals.append(float(tok))
            except ValueError:           # 'MM' or garbage
                vals.append(math.nan)
        return np.array(vals)

    fields = {}
    for name in _REQUIRED:
        c = _fill_missing(hours, col(name))
        if c is None:
            raise ValueError(f"{path}: column {name} has no valid observations")
        fields[name] = c
    for name in _OPTIONAL:
        fields[name] = _fill_missing(hours, col(name))

    wdir, wspd = fields["wdir"], fields["wspd"]
    th = np.radians(wdir)
    u = -wspd * np.cos(th)               # +x = north, +y = east
    v = -wspd * np.sin(th)
    if fields["dewp"] is not None:
        rh = np.clip(_saturation_vapor_pressure_C(fields["dewp"])
                     / _saturation_vapor_pressure_C(fields["atmp"]), 0.0, 1.0)
    else:
        rh = None
    return MetSeries(
        hours=hours, stamps=stamps,
        wdir_deg=wdir, wspd_m_s=wspd, gust_m_s=fields["gust"],
        pres_hPa=fields["pres"], atmp_C=fields["atmp"],
        dewp_C=fields["dewp"], wtmp_C=fields["wtmp"], rh=rh,
        u_north=u, v_east=v,
        label=label or f"NDBC met replay ({path.name}, {len(rows)} records)",
        path=str(path))


def load_met_default(path=None):
    """Parse the bundled NDBC 46012 record (or `path`)."""
    return parse_ndbc_met(path)


class HistoricalWeather(Weather):
    """Weather that replays an observed record as simulation time runs.

    Drop-in replacement for Weather (and therefore Atmosphere) in
    spatial_dynamics / env / mavlink_if: each query at time t maps to the
    record hour `t0_hours + t * time_scale / 3600` (clamped to the
    record span) and interpolates PRES/ATMP/DEWP/WIND/GST into the
    underlying WeatherConfig + AtmosphereConfig.  Hydrometeors the file
    does not report (rain, cloud, visibility, downdraft, lightning) come
    from the base `weather` config, defaulting to a dry replay.

    time_scale is real seconds per simulation second (1 = wall-clock
    replay, 360 = one hour of history per 10 s episode, 0 = frozen at
    t0_hours, i.e. steady observed conditions).
    """

    def __init__(self, series, weather=None, atmosphere=None, seed=0,
                 t0_hours=0.0, time_scale=1.0):
        if isinstance(series, (str, Path)):
            series = parse_ndbc_met(series)
        if not isinstance(series, MetSeries):
            raise TypeError("series must be a MetSeries or an NDBC file path")
        if not math.isfinite(t0_hours):
            raise ValueError("t0_hours must be finite")
        if not math.isfinite(time_scale) or time_scale < 0.0:
            raise ValueError("time_scale must be finite and >= 0")
        self.series = series
        self.t0_hours = float(t0_hours)
        self.time_scale = float(time_scale)
        super().__init__(weather, atmosphere, seed=seed)
        self._base_weather = self.config          # validated WeatherConfig
        self._base_atm = self._atm.config         # validated AtmosphereConfig
        self._sync(0.0)

    # ----- record-time mapping -----
    def record_hour(self, t):
        """Record hour (elapsed since first observation) at sim time t."""
        h = self.t0_hours + float(t) * self.time_scale / 3600.0
        return min(max(h, self.series.hours[0]), self.series.hours[-1])

    def observed(self, t):
        """Interpolated raw observations at sim time t (dict)."""
        return self.series.interp(self.record_hour(t))

    def _sync(self, t):
        obs = self.series.interp(self.record_hour(t))
        w = dict(temp_offset_K=obs["atmp_C"] - T_REF_C,
                 pressure_offset_Pa=(obs["pres_hPa"] - P_REF_HPA) * 100.0)
        if obs["rh"] is not None:
            w["humidity"] = obs["rh"]
        self.config = replace(self._base_weather, **w)
        self._atm.config = replace(
            self._base_atm,
            wind=(obs["u_north"], obs["v_east"], self._base_atm.wind[2]),
            gust_rms=GUST_SIGMA_FACTOR * obs["gust_excess_m_s"])
        return obs

    # ----- time-varying overrides -----
    def wind(self, t, altitude=None):
        self._sync(t)
        return super().wind(t, altitude)

    def effects(self, t, altitude, airspeed, area=None):
        self._sync(t)
        return super().effects(t, altitude, airspeed, area=area)

    def summary(self, t=0.0, altitude=30.0, airspeed=11.3, area=None):
        obs = self._sync(t)
        s = super().summary(t, altitude, airspeed, area=area)
        s["record_utc"] = obs["stamp"]
        s["record_hours_from_start"] = float(obs["hour"])
        return s


if __name__ == "__main__":
    series = load_met_default()
    print(f"{series.label}: {len(series)} records, "
          f"span {series.span_hours:.0f} h "
          f"({series.stamps[0]} .. {series.stamps[-1]} UTC)")
    print(f"  PRES {series.pres_hPa.min():.1f} .. {series.pres_hPa.max():.1f} hPa"
          f"  ATMP {series.atmp_C.min():.1f} .. {series.atmp_C.max():.1f} C")
    print(f"  WSPD {series.wspd_m_s.min():.1f} .. {series.wspd_m_s.max():.1f} m/s"
          f"  GST  {series.gust_m_s.min():.1f} .. {series.gust_m_s.max():.1f} m/s")
    print(f"  RH   {series.rh.min():.2f} .. {series.rh.max():.2f}")
    hw = HistoricalWeather(series, t0_hours=0.0, time_scale=360.0)
    for t in (0.0, 300.0, 600.0):
        s = hw.summary(t, 30.0, 11.3)
        print(f"  t={t:5.0f}s -> {s['record_utc']}  "
              f"T={s['temperature_C']:5.1f}C p={s['pressure_hPa']:7.1f}hPa "
              f"rho={s['density_kg_m3']:.4f} rh={s['humidity']:.2f} "
              f"|w|={math.dist((0,0), s['wind_m_s'][:2]):.1f}m/s "
              f"[{s['condition']}]")
