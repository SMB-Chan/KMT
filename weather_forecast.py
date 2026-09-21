"""Weather *forecast* model grown out of the historical replay layer.

`weather_real.HistoricalWeather` replays what a buoy observed.  This module
adds the next level: a deterministic statistical forecast model that is
*initialised* on the observed state at an issue time and *predicts* the
evolution forward, so the simulator can fly under a forecast instead of a
replay, and the forecast can be verified against the held-out observations.

Model (`VarForecastModel`, a linear inverse model / VAR on anomalies):

* The hourly state is y = (PRES, ATMP, DEWP, u, v, GST-WSPD).  Wind enters
  as (u, v) components so direction stays coherent and the 359->1 deg wrap
  cannot spin the forecast.
* A diurnal climatology (mean per UTC hour-of-day over the training slice)
  is removed; the VAR(p) recursion is fitted on the anomalies by ridge
  least squares, so a forecast relaxes towards climatology as lead grows,
  exactly like an operational NWP guidance curve.
* The companion matrix is checked for stability and the autoregressive
  coefficients are shrunk until every mode decays (spectral radius < 1),
  which guarantees bounded, damping forecasts.

Because a forecast trajectory is just an hourly series of the same
quantities the replay consumes, `forecast_series` returns a `MetSeries` and
`forecast_weather` hands it to `HistoricalWeather`.  Every existing consumer
(env / mavlink_if / fly_ollama) therefore flies under a forecast unchanged.

References for verification: `persistence_forecast` (hold the analysis) and
`climatology_forecast` (diurnal cycle only).  Skill is reported as RMSE and
as a skill score 1 - RMSE_model / RMSE_reference versus lead time.

Everything is analytic and deterministic: least squares and the recursion
carry no RNG, so a given issue time always yields the same forecast.
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import numpy as np

from weather_real import (GUST_SIGMA_FACTOR, HistoricalWeather, MetSeries,
                          _saturation_vapor_pressure_C, parse_ndbc_met)

# forecast state layout
_IX_PRES, _IX_ATMP, _IX_DEWP, _IX_U, _IX_V, _IX_GEX = range(6)
N_STATE = 6
MIN_TRAIN_HOURS = 336   # >= 2 weeks of history before an issue time


def _hour_of_day(series):
    """UTC hour-of-day of every record (integer elapsed hours)."""
    hod0 = int(series.stamps[0][11:13])
    return (hod0 + np.rint(series.hours).astype(int)) % 24


def _state_matrix(series):
    """(N, 6) observed state: pres, atmp, dewp, u, v, gust excess."""
    dewp = series.dewp_C
    if dewp is None:                      # no dew point: assume RH 0.5-ish
        dewp = series.atmp_C - 5.0
    return np.column_stack([
        series.pres_hPa, series.atmp_C, dewp,
        series.u_north, series.v_east, series.gust_excess()])


class DiurnalClimatology:
    """Mean state per UTC hour-of-day, fitted on a training slice."""

    def __init__(self, series, train_stop):
        y = _state_matrix(series)
        hod = _hour_of_day(series)
        self.mean = np.mean(y[:train_stop], axis=0)
        self.table = np.tile(self.mean, (24, 1))
        for h in range(24):
            m = hod[:train_stop] == h
            if m.any():
                self.table[h] = np.mean(y[:train_stop][m], axis=0)
        self.hod = hod

    def at_record(self, idx):
        """Climatology row at record index (its hour-of-day)."""
        return self.table[self.hod[idx]]

    def at_lead(self, issue_idx, lead):
        """Climatology row `lead` hours after record `issue_idx`."""
        hod = (int(self.hod[issue_idx]) + int(round(lead))) % 24
        return self.table[hod]


class VarForecastModel:
    """Deterministic VAR(p) forecast of the buoy state anomalies.

    Fit on records [0, train_stop); predict from an issue index using the
    observed state as the initial condition.  Lead 0 is the analysis itself.
    """

    def __init__(self, series, order=3, train_stop=None, ridge=1e-6,
                 max_radius=0.995):
        self.series = series
        self.order = int(order)
        n = len(series)
        self.train_stop = n if train_stop is None else int(train_stop)
        if not self.order >= 1:
            raise ValueError("order must be >= 1")
        if self.train_stop <= self.order + N_STATE + 1:
            raise ValueError("training slice too short for the chosen order")
        self.clim = DiurnalClimatology(series, self.train_stop)
        y = _state_matrix(series)[:self.train_stop]
        a = y - self.clim.table[self.clim.hod[:self.train_stop]]
        p = self.order
        rows_y = a[p:]
        X = np.hstack([np.ones((len(rows_y), 1))]
                      + [a[p - k:-k if k else None] for k in range(1, p + 1)])
        XtX = X.T @ X
        pen = np.eye(XtX.shape[0])
        pen[0, 0] = 0.0                       # do not shrink the intercept
        lam = ridge * np.trace(XtX) / XtX.shape[0]
        self.coef = np.linalg.solve(XtX + lam * pen, X.T @ rows_y)
        self._stabilize(max_radius)
        self.resid_rms = float(np.sqrt(np.mean(
            np.sum((rows_y - X @ self.coef) ** 2, axis=1))))

    # ----- stability -----
    def _companion_radius(self):
        p, c = self.order, self.coef
        T = c[1:].T.reshape(N_STATE, p, N_STATE)   # T[i, k, j] = A_{k+1}[i, j]
        comp = np.zeros((N_STATE * p, N_STATE * p))
        comp[:N_STATE, :] = np.hstack([T[:, k, :] for k in range(p)])
        if p > 1:
            comp[N_STATE:N_STATE * p, :N_STATE * (p - 1)] = np.eye(
                N_STATE * (p - 1))
        return float(np.max(np.abs(np.linalg.eigvals(comp))))

    def _stabilize(self, max_radius):
        """Shrink AR blocks uniformly until all modes decay."""
        p = self.order
        for _ in range(60):
            if self._companion_radius() <= max_radius:
                return
            self.coef[1:] *= 0.95
        raise RuntimeError("could not stabilise the forecast model")

    # ----- prediction -----
    def analysis(self, issue_idx):
        """Observed state (not anomaly) at the issue record."""
        return _state_matrix(self.series)[issue_idx]

    def predict(self, issue_idx, horizon):
        """(horizon+1, 6) forecast state; lead 0 = the analysis."""
        s = self.series
        if not 0 <= issue_idx < len(s):
            raise ValueError("issue_idx outside the record")
        y = _state_matrix(s)
        p = self.order
        hist = [y[i] - self.clim.at_record(i)
                for i in range(max(0, issue_idx - p + 1), issue_idx + 1)]
        while len(hist) < p:                  # pad with the oldest anomaly
            hist.insert(0, hist[0])
        hist = hist[::-1]                     # hist[0] = most recent
        out = [y[issue_idx]]
        stack = list(hist)
        for lead in range(1, int(horizon) + 1):
            x = np.concatenate([[1.0], *stack])
            nxt = self.coef.T @ x
            out.append(nxt + self.clim.at_lead(issue_idx, lead))
            stack = [nxt] + stack[:p - 1]
        return np.array(out)

    def forecast_series(self, issue_idx, horizon, label=None):
        """Forecast trajectory as a replay-consumable MetSeries."""
        f = self.predict(issue_idx, horizon)
        pres, atmp, dewp = f[:, 0], f[:, 1], f[:, 2]
        u, v, gex = f[:, 3], f[:, 4], np.maximum(f[:, 5], 0.0)
        wspd = np.hypot(u, v)
        wdir = np.degrees(np.arctan2(-v, -u)) % 360.0
        rh = np.clip(_saturation_vapor_pressure_C(dewp)
                     / _saturation_vapor_pressure_C(atmp), 0.0, 1.0)
        t0 = datetime.strptime(self.series.stamps[issue_idx],
                               "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
        stamps = tuple((t0 + timedelta(hours=int(l)))
                       .strftime("%Y-%m-%d %H:%M")
                       for l in range(len(f)))
        return MetSeries(
            hours=np.arange(len(f), dtype=float), stamps=stamps,
            wdir_deg=wdir, wspd_m_s=wspd, gust_m_s=wspd + gex,
            pres_hPa=pres, atmp_C=atmp, dewp_C=dewp, wtmp_C=None, rh=rh,
            u_north=u, v_east=v,
            label=label or f"VAR({self.order}) forecast from "
                           f"{self.series.stamps[issue_idx]} UTC",
            path=self.series.path)


# ----- reference forecasts and skill -----
def persistence_forecast(series, issue_idx, horizon):
    """Hold the analysis state for every lead (lead 0 = analysis)."""
    y = _state_matrix(series)[issue_idx]
    return np.tile(y, (int(horizon) + 1, 1))


def climatology_forecast(series, issue_idx, horizon, train_stop=None):
    """Diurnal climatology only (no knowledge of the analysis)."""
    clim = DiurnalClimatology(
        series, len(series) if train_stop is None else train_stop)
    return np.array([clim.at_lead(issue_idx, l)
                     for l in range(int(horizon) + 1)])


def rmse_vs_lead(forecast, truth):
    """Per-lead, per-state RMSE.

    Accepts a single forecast (L+1, 6) or a stack of issues (M, L+1, 6);
    returns (L+1, 6) root-mean-square error across the issue axis.
    """
    d = np.asarray(forecast, float) - np.asarray(truth, float)
    if d.ndim == 2:
        d = d[None, ...]
    return np.sqrt(np.mean(d ** 2, axis=0))


def skill_score(model_err, ref_err):
    """1 - RMSE_model / RMSE_reference (>0 means the model wins)."""
    return 1.0 - np.asarray(model_err) / np.maximum(np.asarray(ref_err),
                                                     1e-12)


def issue_index(series, hours):
    """Record index nearest to `hours` elapsed since the first observation."""
    return int(np.argmin(np.abs(np.asarray(series.hours, float)
                                - float(hours))))


def forecast_weather(series, issue_idx, horizon, weather=None,
                     atmosphere=None, seed=0, time_scale=1.0, order=3,
                     train_stop=None):
    """A drop-in HistoricalWeather that flies the *forecast*, not the replay.

    Simulation time 0 corresponds to the issue time; `time_scale` maps
    simulation seconds onto forecast lead seconds exactly as for a replay.

    Training defaults to the records preceding the issue so the forecast
    never sees the future it is verified against.
    """
    if train_stop is None:
        train_stop = int(issue_idx)
    if train_stop < MIN_TRAIN_HOURS:
        raise ValueError(f"forecast needs >= {MIN_TRAIN_HOURS} h of record "
                         f"before the issue time (got {train_stop})")
    model = VarForecastModel(series, order=order, train_stop=train_stop)
    fc = model.forecast_series(issue_idx, horizon)
    return HistoricalWeather(fc, weather=weather, atmosphere=atmosphere,
                             seed=seed, t0_hours=0.0,
                             time_scale=time_scale), model


if __name__ == "__main__":
    s = parse_ndbc_met()
    issue = int(len(s) * 0.7)
    hw, model = forecast_weather(s, issue, 48, time_scale=360.0)
    truth = _state_matrix(s)[issue:issue + 49]
    fc = model.predict(issue, 48)
    print(f"{model.__class__.__name__} order={model.order} "
          f"resid_rms={model.resid_rms:.3f} radius={model._companion_radius():.3f}")
    for lead in (0, 6, 12, 24, 48):
        e_m = math.dist(fc[lead], truth[lead])
        e_p = math.dist(persistence_forecast(s, issue, 48)[lead], truth[lead])
        print(f"  lead {lead:2d} h  |err| model {e_m:6.3f}  persistence {e_p:6.3f}")
    sm = hw.summary(0.0, 30.0, 11.3)
    print(f"  t=0 -> {sm['record_utc']}  T={sm['temperature_C']:.1f} C "
          f"p={sm['pressure_hPa']:.1f} hPa |w|="
          f"{math.dist((0, 0), sm['wind_m_s'][:2]):.1f} m/s")
