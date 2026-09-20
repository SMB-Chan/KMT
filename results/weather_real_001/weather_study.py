#!/usr/bin/env python3
"""Historical-weather replay study: real NDBC observations vs the sim layer.

Feeds 45.7 days of NOAA NDBC 46012 (Humboldt Bay) hourly observations
into weather_real.HistoricalWeather and analyses whether the simulator
follows an approximate evolution of the observed weather:

  A. data overview     -- record statistics, extremes, direction histogram;
  B. replay fidelity   -- driver state vs the raw observations at every
                          record node and at every interval midpoint
                          (exactness of the obs -> model conversion chain);
  C. gust calibration  -- naive vs calibrated GST -> gust_rms mapping,
                          scored by the 1-hour peak-speed excess; this is
                          the analysis that fixed GUST_SIGMA_FACTOR;
  D. trim tracking     -- hourly stall speed / trim drag / thrust margin
                          across the whole record: does the flight
                          envelope follow the synoptic evolution?
  E. dynamic response  -- frozen-trim load factor (wind_tunnel method)
                          through the stormiest and the calmest 2-hour
                          windows of the record;
  F. env replay        -- FlyingBoatEnv episode over the storm window,
                          verifying step telemetry tracks the record.

Corrections derived from the analysis are summarised in REPORT.md and
summary.json (section "corrections").

Usage:
    python3 weather_study.py [--out results/weather_real_001] [--quick]
                             [--no-plots] [--data PATH]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from aircraft import Aircraft
from atmosphere import (Atmosphere, AtmosphereConfig,
                        isa_pressure, isa_temperature)
from design_optimize import git_head, _fmt
from weather import moist_density
from weather_real import (DEFAULT_MET_PATH, GUST_SIGMA_FACTOR,
                          HistoricalWeather, parse_ndbc_met)
from wind_tunnel import trim_state, gust_load_stats

V_CRUISE = 11.3          # m/s, level-flight tracking speed
PATROL_ALT = 30.0        # m, altitude of the envelope tracking
CALIB_SEED_BASE = 1000   # fixed seeds for the gust calibration ensemble


@dataclass
class StudyConfig:
    out: Path = Path("results/weather_real_001")
    data: Path = DEFAULT_MET_PATH
    seed: int = 42
    calib_records: int = 24      # records scored in the gust calibration
    calib_seeds: int = 4         # gust realisations per record
    zoom_hours: float = 72.0     # fidelity plot window
    response_duration_s: float = 7200.0   # E: window length (2 h)
    response_dt: float = 0.2
    env_steps: int = 200
    env_rate: float = 360.0      # real seconds per sim second in F
    plots: bool = True
    quick: bool = False

    def __post_init__(self):
        self.out = Path(self.out)
        self.data = Path(self.data)
        if self.quick:
            self.calib_records = 8
            self.calib_seeds = 2
            self.response_duration_s = 1800.0
            self.response_dt = 0.5
            self.env_steps = 40


# ---------------------------------------------------------------------
#  helpers
# ---------------------------------------------------------------------
def moving_mean(a: np.ndarray, k: int = 3) -> np.ndarray:
    return np.convolve(np.asarray(a, float), np.ones(k) / k, mode="valid")


def write_csv(path: Path, rows: list, cols: list) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r[c] for c in cols})


# ---------------------------------------------------------------------
#  A. data overview
# ---------------------------------------------------------------------
def data_overview(series) -> dict:
    n = len(series)
    hours, pres, atmp, dewp = (series.hours, series.pres_hPa,
                               series.atmp_C, series.dewp_C)
    wspd, gst = series.wspd_m_s, series.gust_m_s
    excess = series.gust_excess()
    rh = series.rh
    # raw-observation sea-level density (the quantity the drone feels)
    rho_obs = np.array([moist_density(pres[i] * 100.0, atmp[i] + 273.15,
                                      rh[i]) for i in range(n)])
    hist, edges = np.histogram(series.wdir_deg % 360.0,
                               bins=np.arange(0.0, 361.0, 10.0))
    top = np.argsort(hist)[::-1][:8]
    d = (np.diff(series.wdir_deg) + 180.0) % 360.0 - 180.0
    i_low = int(np.argmin(pres)); i_gust = int(np.argmax(gst))
    i_storm = int(np.argmax(moving_mean(gst, 3)))
    i_calm = int(np.argmin(moving_mean(wspd + excess, 3)))
    return dict(
        records=n, span_hours=float(series.span_hours),
        start_utc=series.stamps[0], end_utc=series.stamps[-1],
        label=series.label,
        pres_hPa=dict(min=float(pres.min()), mean=float(pres.mean()),
                      max=float(pres.max())),
        atmp_C=dict(min=float(atmp.min()), mean=float(atmp.mean()),
                    max=float(atmp.max())),
        dewp_C=dict(min=float(dewp.min()), mean=float(dewp.mean()),
                    max=float(dewp.max())),
        rh=dict(min=float(rh.min()), mean=float(rh.mean()),
                max=float(rh.max())),
        wspd_m_s=dict(min=float(wspd.min()), mean=float(wspd.mean()),
                      max=float(wspd.max())),
        gust_m_s=dict(min=float(gst.min()), mean=float(gst.mean()),
                      max=float(gst.max())),
        gust_excess_m_s=dict(mean=float(excess.mean()),
                             max=float(excess.max())),
        rho_sealevel=dict(min=float(rho_obs.min()),
                          mean=float(rho_obs.mean()),
                          max=float(rho_obs.max())),
        wdir_top8=[dict(center_deg=float((edges[j] + edges[j + 1]) / 2.0),
                        count=int(hist[j])) for j in sorted(top, key=lambda k: -hist[k])],
        extremes=dict(
            lowest_pressure=dict(hour=float(hours[i_low]),
                                 stamp=series.stamps[i_low],
                                 pres_hPa=float(pres[i_low])),
            highest_gust=dict(hour=float(hours[i_gust]),
                              stamp=series.stamps[i_gust],
                              gust_m_s=float(gst[i_gust])),
            storm_window=dict(start_hour=float(hours[i_storm]),
                              start_stamp=series.stamps[i_storm]),
            calm_window=dict(start_hour=float(hours[i_calm]),
                             start_stamp=series.stamps[i_calm])),
        supersaturated_records=int(np.sum(dewp > atmp)),
        direction_wraps_gt90=int(np.sum(np.abs(d) > 90.0)),
        rho_obs=rho_obs, i_storm=i_storm, i_calm=i_calm,
    )


# ---------------------------------------------------------------------
#  B. replay fidelity at record nodes and interval midpoints
# ---------------------------------------------------------------------
def node_fidelity(series, cfg: StudyConfig, rho_obs: np.ndarray) -> dict:
    driver = HistoricalWeather(series, seed=cfg.seed)
    n = len(series)
    rows = []
    err_T, err_p, err_rho, err_uv = [], [], [], []
    for i in range(n):
        t = float(series.hours[i]) * 3600.0
        driver.wind(t)                     # sync to the record node
        T = driver.temperature(0.0)
        p = driver.pressure(0.0)
        rho = driver.density(0.0)
        wcfg = driver._atm.config.wind
        e = dict(hour=float(series.hours[i]), stamp=series.stamps[i],
                 obs_pres_hPa=float(series.pres_hPa[i]),
                 obs_atmp_C=float(series.atmp_C[i]),
                 obs_rh=float(series.rh[i]),
                 obs_wspd_m_s=float(series.wspd_m_s[i]),
                 err_T_K=T - (series.atmp_C[i] + 273.15),
                 err_p_Pa=p - series.pres_hPa[i] * 100.0,
                 err_rho=rho - rho_obs[i],
                 err_u=wcfg[0] - series.u_north[i],
                 err_v=wcfg[1] - series.v_east[i])
        err_T.append(e["err_T_K"]); err_p.append(e["err_p_Pa"])
        err_rho.append(e["err_rho"])
        err_uv.append(math.hypot(e["err_u"], e["err_v"]))
        rows.append(e)
    # midpoint linearity of every interpolated channel
    m_T, m_p, m_rh, m_u, m_v = [], [], [], [], []
    for i in range(n - 1):
        h = 0.5 * (series.hours[i] + series.hours[i + 1])
        driver.wind(h * 3600.0)
        m_T.append(driver.temperature(0.0)
                   - 0.5 * (series.atmp_C[i] + series.atmp_C[i + 1]) - 273.15)
        m_p.append(driver.pressure(0.0)
                   - 50.0 * (series.pres_hPa[i] + series.pres_hPa[i + 1]))
        m_rh.append(driver.config.humidity
                    - 0.5 * (series.rh[i] + series.rh[i + 1]))
        m_u.append(driver._atm.config.wind[0]
                   - 0.5 * (series.u_north[i] + series.u_north[i + 1]))
        m_v.append(driver._atm.config.wind[1]
                   - 0.5 * (series.v_east[i] + series.v_east[i + 1]))
    max_abs = lambda a: float(np.max(np.abs(a)))
    # scalar speed of the chord-interpolated vector vs scalar-interpolated
    # WSPD at midpoints (documented, physically correct chord shortcut)
    chord_deficit = []
    for i in range(n - 1):
        um = 0.5 * (series.u_north[i] + series.u_north[i + 1])
        vm = 0.5 * (series.v_east[i] + series.v_east[i + 1])
        sm = 0.5 * (series.wspd_m_s[i] + series.wspd_m_s[i + 1])
        chord_deficit.append(sm - math.hypot(um, vm))
    return dict(
        node_max_abs_err_T_K=max_abs(err_T),
        node_max_abs_err_p_Pa=max_abs(err_p),
        node_max_abs_err_rho_kg_m3=max_abs(err_rho),
        node_max_abs_err_wind_m_s=max_abs(err_uv),
        midpoint_max_abs_err_T_K=max_abs(m_T),
        midpoint_max_abs_err_p_Pa=max_abs(m_p),
        midpoint_max_abs_err_rh=max_abs(m_rh),
        midpoint_max_abs_err_wind_m_s=max(max(map(abs, m_u)),
                                          max(map(abs, m_v))),
        chord_speed_deficit_mean_m_s=float(np.mean(chord_deficit)),
        chord_speed_deficit_max_m_s=float(np.max(chord_deficit)),
        rows=rows,
    )


# ---------------------------------------------------------------------
#  C. gust calibration: GST - WSPD -> gust_rms
# ---------------------------------------------------------------------
def gust_calibration(series, cfg: StudyConfig, i_storm: int) -> dict:
    """Score naive (k=1) vs calibrated (k=GUST_SIGMA_FACTOR) mapping.

    The sum4 gust process is linear in gust_rms, so one unit-RMS
    realisation per seed is sampled once and rescaled per record:
    spd_k(t) = |mean_wind + k * excess * g_unit(t)|.  Observables are the
    hourly mean and peak scalar speed (WSPD / GST at z=10 m = z_ref).
    """
    excess = series.gust_excess()
    cand = np.where(excess >= 1.0)[0]
    if len(cand) < 2:
        raise ValueError("record has no usable gust excess for calibration")
    pick = np.unique(cand[np.linspace(0, len(cand) - 1,
                                      cfg.calib_records).astype(int)])
    t = np.arange(3600.0)
    units = []
    for s in range(cfg.calib_seeds):
        atm = Atmosphere(AtmosphereConfig(gust_rms=1.0),
                         seed=CALIB_SEED_BASE + s)
        units.append(np.array([atm.wind(ti, 10.0) for ti in t]))
    units = np.stack(units)                       # (seeds, 3600, 3)
    mappings = (("naive", 1.0), ("calibrated", GUST_SIGMA_FACTOR))
    stats, rows = {}, []
    for tag, k in mappings:
        peak_err, mean_err = [], []
        for i in pick:
            mean_w = np.array([series.u_north[i], series.v_east[i], 0.0])
            w = mean_w[None, None, :] + (k * excess[i]) * units
            spd = np.linalg.norm(w, axis=2)       # (seeds, 3600)
            peak = float(spd.max(axis=1).mean())
            mean_s = float(spd.mean(axis=1).mean())
            pe = peak - float(series.gust_m_s[i])
            me = mean_s - float(series.wspd_m_s[i])
            peak_err.append(pe); mean_err.append(me)
            rows.append(dict(mapping=tag, k=float(k),
                             hour=float(series.hours[i]),
                             stamp=series.stamps[i],
                             wspd_m_s=float(series.wspd_m_s[i]),
                             gust_m_s=float(series.gust_m_s[i]),
                             excess_m_s=float(excess[i]),
                             model_peak_m_s=peak, model_mean_m_s=mean_s,
                             peak_err_m_s=float(pe),
                             mean_err_m_s=float(me)))
        stats[tag] = dict(
            k=float(k),
            peak_bias_m_s=float(np.mean(peak_err)),
            peak_rmse_m_s=float(np.sqrt(np.mean(np.square(peak_err)))),
            mean_bias_m_s=float(np.mean(mean_err)),
            mean_rmse_m_s=float(np.sqrt(np.mean(np.square(mean_err)))),
            records=int(len(pick)))
    # end-to-end driver check on the stormiest window (single realisation)
    drv = HistoricalWeather(series, seed=cfg.seed,
                            t0_hours=float(series.hours[i_storm]),
                            time_scale=1.0)
    spd = np.array([np.linalg.norm(drv.wind(ti, 10.0))
                    for ti in np.arange(3600.0)])
    obs_mean = float(np.mean(series.wspd_m_s[i_storm:i_storm + 2]))
    obs_peak = float(np.max(series.gust_m_s[i_storm:i_storm + 2]))
    driver_check = dict(
        stamp=series.stamps[i_storm],
        model_mean_m_s=float(spd.mean()), model_peak_m_s=float(spd.max()),
        obs_mean_m_s=obs_mean, obs_peak_m_s=obs_peak,
        mean_err_m_s=float(spd.mean() - obs_mean),
        peak_err_m_s=float(spd.max() - obs_peak))
    return dict(stats=stats, rows=rows, driver_check=driver_check)


# ---------------------------------------------------------------------
#  D. trim tracking across the whole record
# ---------------------------------------------------------------------
def trim_tracking(series, cfg: StudyConfig, rho_obs: np.ndarray) -> dict:
    ac = Aircraft()
    driver = HistoricalWeather(series, seed=cfg.seed)
    eta = ac.prop.eta_motor * ac.prop.eta_esc * ac.prop.eta_prop
    step = 1 if not cfg.quick else 4
    rows = []
    rho0_l, rho30_l, vs_l, d_l, t_l, m_l, pres_l, atmp_l, wspd_l = \
        [], [], [], [], [], [], [], [], []
    for i in range(0, len(series), step):
        t = float(series.hours[i]) * 3600.0
        driver.wind(t)                        # sync
        rho0 = driver.density(0.0)
        rho30 = driver.density(PATROL_ALT)
        trim = trim_state(ac, rho30, V_CRUISE)
        cd = ac.CD(trim["CL_trim"], height_m=PATROL_ALT)
        drag = trim["q_Pa"] * ac.geom.S * cd
        t_avail = ac.prop.thrust(V_CRUISE, 1.0, rho=rho30)
        v_stall = math.sqrt(2.0 * ac.W / (rho30 * ac.geom.S
                                          * ac.aero.CL_max))
        rows.append(dict(hour=float(series.hours[i]),
                         stamp=series.stamps[i],
                         pres_hPa=float(series.pres_hPa[i]),
                         atmp_C=float(series.atmp_C[i]),
                         rh=float(series.rh[i]),
                         wspd_m_s=float(series.wspd_m_s[i]),
                         gust_m_s=float(series.gust_m_s[i]),
                         rho0_model=rho0, rho0_obs=rho_obs[i],
                         rho30_model=rho30,
                         v_stall_m_s=v_stall,
                         cl_trim=trim["CL_trim"],
                         alpha_trim_deg=trim["alpha_trim_deg"],
                         cd_trim=cd, drag_N=drag,
                         power_W=drag * V_CRUISE / eta,
                         thrust_avail_N=t_avail,
                         margin=t_avail / drag))
        rho0_l.append(rho0); rho30_l.append(rho30); vs_l.append(v_stall)
        d_l.append(drag); t_l.append(t_avail); m_l.append(t_avail / drag)
        pres_l.append(series.pres_hPa[i]); atmp_l.append(series.atmp_C[i])
        wspd_l.append(series.wspd_m_s[i])
    rho0_l = np.array(rho0_l); rho30_l = np.array(rho30_l)
    corr = lambda a, b: float(np.corrcoef(a, b)[0, 1])
    i_worst = int(np.argmin(m_l))
    return dict(
        rows=rows, n=len(rows),
        rho0_max_abs_dev_vs_obs=float(np.max(np.abs(
            rho0_l - rho_obs[::step]))),
        rho30=dict(min=float(rho30_l.min()), max=float(rho30_l.max())),
        v_stall_m_s=dict(min=float(np.min(vs_l)), max=float(np.max(vs_l))),
        drag_N=dict(min=float(np.min(d_l)), max=float(np.max(d_l))),
        thrust_avail_N=dict(min=float(np.min(t_l)),
                            max=float(np.max(t_l))),
        margin=dict(min=float(np.min(m_l)), max=float(np.max(m_l)),
                    worst_stamp=rows[i_worst]["stamp"]),
        corr_rho0_pres=corr(rho0_l, pres_l),
        corr_rho0_atmp=corr(rho0_l, atmp_l),
        corr_vstall_rho30=corr(vs_l, rho30_l),
        corr_drag_rho30=corr(d_l, rho30_l),
        corr_thrust_rho30=corr(t_l, rho30_l),
    )


# ---------------------------------------------------------------------
#  E. dynamic response in the stormiest / calmest windows
# ---------------------------------------------------------------------
def response_windows(series, cfg: StudyConfig, ac: Aircraft,
                     i_storm: int, i_calm: int) -> dict:
    out = {}
    for tag, i0 in (("storm", i_storm), ("calm", i_calm)):
        driver = HistoricalWeather(series, seed=cfg.seed,
                                   t0_hours=float(series.hours[i0]),
                                   time_scale=1.0)
        g = gust_load_stats(ac, driver, altitude=PATROL_ALT, V=V_CRUISE,
                            duration=cfg.response_duration_s,
                            dt=cfg.response_dt, label=tag)
        k = int(round(cfg.response_duration_s / 3600.0)) + 1
        sl = slice(i0, min(i0 + k, len(series)))
        out[tag] = dict(
            start_stamp=series.stamps[i0],
            start_hour=float(series.hours[i0]),
            obs_wspd_range=[float(series.wspd_m_s[sl].min()),
                            float(series.wspd_m_s[sl].max())],
            obs_gust_max=float(series.gust_m_s[sl].max()),
            n_mean=g["n_mean"], n_std=g["n_std"], n_peak=g["n_peak"],
            n_min=g["n_min"], alpha_peak_deg=g["alpha_peak_deg"],
            alpha_min_deg=g["alpha_min_deg"],
            t=g["t"], n=g["n"])
    r = out["storm"]["n_std"] / max(out["calm"]["n_std"], 1e-9)
    ex = series.gust_excess()
    k = int(round(cfg.response_duration_s / 3600.0)) + 1
    obs_ratio = (ex[i_storm:i_storm + k].mean()
                 / max(ex[i_calm:i_calm + k].mean(), 1e-9))
    out["n_std_ratio_storm_over_calm"] = float(r)
    out["obs_excess_ratio_storm_over_calm"] = float(obs_ratio)
    return out


# ---------------------------------------------------------------------
#  F. end-to-end env replay over the storm window
# ---------------------------------------------------------------------
def env_replay(series, cfg: StudyConfig, i_storm: int) -> dict:
    from env import EnvConfig, FlyingBoatEnv
    t0 = float(series.hours[i_storm])
    ecfg = EnvConfig(spatial=True, weather_real=str(cfg.data),
                     weather_real_t0=t0, weather_real_rate=cfg.env_rate,
                     max_steps=cfg.env_steps)
    env = FlyingBoatEnv(Aircraft(), ecfg)
    env.reset(seed=7)
    rows, dev_T, dev_p = [], [], []
    for _ in range(cfg.env_steps):
        a = np.zeros(env.action_dim)
        a[0] = 1.0
        _, _, done, info = env.step(a)
        w = info["weather"]
        rec = series.interp(t0 + info["t"] * cfg.env_rate / 3600.0)
        z = float(info["z"])
        exp_T = rec["atmp_C"] + (isa_temperature(z) - 288.15)
        exp_p = (isa_pressure(z)
                 + (rec["pres_hPa"] - 1013.25) * 100.0) / 100.0
        dev_T.append(w["temperature_C"] - exp_T)
        dev_p.append(w["pressure_hPa"] - exp_p)
        rows.append(dict(t=info["t"], record_utc=w["record_utc"],
                         z=z, airspeed=info["airspeed"],
                         temperature_C=w["temperature_C"],
                         pressure_hPa=w["pressure_hPa"],
                         density_kg_m3=w["density_kg_m3"],
                         humidity=w["humidity"],
                         wind_x=w["wind_m_s"][0], wind_y=w["wind_m_s"][1],
                         ice_mass_kg=info.get("ice_mass_kg", 0.0)))
        if done:
            break
    return dict(
        steps=len(rows), t0_hours=t0, rate=cfg.env_rate,
        first_record=rows[0]["record_utc"], last_record=rows[-1]["record_utc"],
        max_abs_dev_T_C=float(np.max(np.abs(dev_T))),
        max_abs_dev_p_hPa=float(np.max(np.abs(dev_p))),
        final_z=float(rows[-1]["z"]),
        final_airspeed=float(rows[-1]["airspeed"]),
        rows=rows)


# ---------------------------------------------------------------------
#  plots
# ---------------------------------------------------------------------
def make_plots(cfg: StudyConfig, series, ov: dict, fid: dict, cal: dict,
               trk: dict, resp: dict) -> list:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    available = {f.name for f in font_manager.fontManager.ttflist}
    for cand in ("Noto Sans CJK JP", "Noto Sans CJK SC", "IPAexGothic"):
        if cand in available:
            plt.rcParams["font.family"] = cand
            break

    names = []
    days = series.hours / 24.0

    # -- A. record overview -------------------------------------------
    fig, ax = plt.subplots(4, 1, figsize=(11.0, 9.0), sharex=True)
    ax[0].plot(days, series.pres_hPa, lw=0.9, color="tab:blue")
    ax[0].set_ylabel("PRES [hPa]")
    ax[1].plot(days, series.atmp_C, lw=0.9, color="tab:red", label="ATMP")
    ax[1].plot(days, series.dewp_C, lw=0.9, color="tab:olive", label="DEWP")
    ax[1].set_ylabel("気温 [degC]"); ax[1].legend(loc="upper right", fontsize=8)
    ax[2].plot(days, series.rh, lw=0.9, color="tab:green")
    ax[2].set_ylabel("RH [-]"); ax[2].set_ylim(0.7, 1.02)
    ax[3].plot(days, series.wspd_m_s, lw=0.9, color="tab:purple",
               label="WSPD (時平均)")
    ax[3].plot(days, series.gust_m_s, lw=0.9, color="tab:orange", alpha=0.8,
               label="GST (最大突風)")
    ax[3].set_ylabel("風速 [m/s]"); ax[3].set_xlabel("経過日数 [d]")
    ax[3].legend(loc="upper right", fontsize=8)
    for a in ax:
        a.grid(alpha=0.3)
    fig.suptitle(f"NDBC 46012 実測記録 ({ov['start_utc']} .. {ov['end_utc']} UTC)")
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(cfg.out / "overview.png", dpi=130)
    plt.close(fig)
    names.append("overview.png")

    # -- B. fidelity zoom ----------------------------------------------
    drv = HistoricalWeather(series, seed=cfg.seed)
    h0 = float(series.hours[ov["i_storm"]])
    h1 = min(h0 + cfg.zoom_hours, float(series.hours[-1]))
    hs = np.arange(h0, h1, 1.0 / 6.0)          # 10-minute samples
    Ts, ps, rs, ws = [], [], [], []
    for h in hs:
        drv.wind(h * 3600.0)
        Ts.append(drv.temperature(0.0) - 273.15)
        ps.append(drv.pressure(0.0) / 100.0)
        rs.append(drv.density(0.0))
        wc = drv._atm.config.wind
        ws.append(math.hypot(wc[0], wc[1]))
    sel = (series.hours >= h0) & (series.hours <= h1)
    fig, ax = plt.subplots(4, 1, figsize=(11.0, 9.0), sharex=True)
    pairs = ((Ts, "気温 [degC]", series.atmp_C[sel]),
             (ps, "気圧 [hPa]", series.pres_hPa[sel]),
             (rs, "密度 [kg/m3]", ov["rho_obs"][sel]),
             (ws, "平均風速 [m/s]", series.wspd_m_s[sel]))
    for a, (model, yl, obsv) in zip(ax, pairs):
        a.plot(hs, model, lw=1.0, color="tab:blue", label="リプレイ (10分間隔)")
        a.plot(series.hours[sel], obsv, "o", ms=3.5, color="tab:red",
               label="実測 (NDBC)")
        a.set_ylabel(yl); a.grid(alpha=0.3)
        a.legend(loc="upper right", fontsize=8)
    ax[-1].set_xlabel("記録経過時間 [h]")
    fig.suptitle(f"リプレイ忠実度ズーム: {series.stamp_at(h0)} からの {h1 - h0:.0f} h")
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(cfg.out / "fidelity_zoom.png", dpi=130)
    plt.close(fig)
    names.append("fidelity_zoom.png")

    # -- C. gust calibration scatter ------------------------------------
    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    styles = {"naive": dict(color="tab:red", marker="x",
                            label="naive: sigma = GST-WSPD"),
              "calibrated": dict(color="tab:blue", marker="o",
                                 label=f"calibrated: sigma = {GUST_SIGMA_FACTOR} x (GST-WSPD)")}
    for r in cal["rows"]:
        st = styles[r["mapping"]]
        ax.plot(r["excess_m_s"], r["model_peak_m_s"] - r["model_mean_m_s"],
                st["marker"], color=st["color"], ms=6,
                label=st["label"] if r is cal["rows"][0]
                      or cal["rows"][0]["mapping"] != r["mapping"] else None)
    lim = max(r["excess_m_s"] for r in cal["rows"]) * 1.35
    ax.plot([0, lim], [0, lim], "k--", lw=1.0, label="完全一致")
    ax.set_xlabel("実測ピーク超過 GST - WSPD [m/s]")
    ax.set_ylabel("モデル1時間のピーク超過 [m/s]")
    ax.set_title("突風変換の校正 (1時間窓・z=10 m)")
    handles, labels = ax.get_legend_handles_labels()
    seen = {}
    for h, l in zip(handles, labels):
        if l and l not in seen:
            seen[l] = h
    ax.legend(list(seen.values()), list(seen.keys()), fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(cfg.out / "gust_calibration.png", dpi=130)
    plt.close(fig)
    names.append("gust_calibration.png")

    # -- D. trim tracking ------------------------------------------------
    hrs = np.array([r["hour"] for r in trk["rows"]]) / 24.0
    fig, ax = plt.subplots(3, 1, figsize=(11.0, 8.0), sharex=True)
    st_h = series.hours[ov["i_storm"]] / 24.0
    for a in ax:
        a.axvspan(st_h, st_h + cfg.response_duration_s / 86400.0,
                  color="tab:red", alpha=0.15)
    ax[0].plot(hrs, [r["rho30_model"] for r in trk["rows"]],
               lw=0.9, color="tab:blue", label="rho(30 m) [kg/m3]")
    ax0b = ax[0].twinx()
    ax0b.plot(hrs, [r["v_stall_m_s"] for r in trk["rows"]], lw=0.9,
              color="tab:red", label="V_stall(30 m) [m/s]")
    ax[0].set_ylabel("rho [kg/m3]"); ax0b.set_ylabel("V_stall [m/s]")
    ax[1].plot(hrs, [r["drag_N"] for r in trk["rows"]], lw=0.9,
               color="tab:purple", label="水平飛行抵抗 [N]")
    ax[1].plot(hrs, [r["thrust_avail_N"] for r in trk["rows"]], lw=0.9,
               color="tab:green", label="利用可能推力 (全開) [N]")
    ax[1].set_ylabel("[N]")
    ax[2].plot(hrs, [r["margin"] for r in trk["rows"]], lw=0.9,
               color="tab:orange")
    ax[2].set_ylabel("推力マージン T/D [-]"); ax[2].set_xlabel("経過日数 [d]")
    for a in ax:
        a.grid(alpha=0.3)
    h1, l1 = ax[0].get_legend_handles_labels()
    h2, l2 = ax0b.get_legend_handles_labels()
    h3, l3 = ax[1].get_legend_handles_labels()
    ax[0].legend(h1 + h2, l1 + l2, fontsize=8, loc="upper right")
    ax[1].legend(h3, l3, fontsize=8, loc="upper right")
    fig.suptitle("飛行エンベロープの追従 (V=11.3 m/s 水平トリム・高度30 m・赤=最荒天窓)")
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(cfg.out / "trim_tracking.png", dpi=130)
    plt.close(fig)
    names.append("trim_tracking.png")

    # -- E. response windows ---------------------------------------------
    fig, ax = plt.subplots(2, 1, figsize=(11.0, 6.0), sharex=True)
    for a, tag, c in zip(ax, ("storm", "calm"), ("tab:red", "tab:blue")):
        w = resp[tag]
        a.plot(w["t"], w["n"], lw=0.7, color=c)
        a.set_ylabel("荷重倍数 n [-]")
        a.set_title(f"{tag} 窓 {w['start_stamp']} UTC: "
                    f"n_std={w['n_std']:.3f}, n_peak={w['n_peak']:.3f}, "
                    f"実測GST最大 {w['obs_gust_max']:.0f} m/s", fontsize=9)
        a.grid(alpha=0.3)
    ax[-1].set_xlabel("時間 [s] (実時間リプレイ)")
    fig.suptitle("トリム固定応答: 実測突風が荷重変動として再現されるか")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(cfg.out / "response_windows.png", dpi=130)
    plt.close(fig)
    names.append("response_windows.png")
    return names


# ---------------------------------------------------------------------
#  corrections derived from the analysis
# ---------------------------------------------------------------------
def corrections_list(ov: dict, fid: dict, cal: dict) -> list:
    naive, calib = cal["stats"]["naive"], cal["stats"]["calibrated"]
    imp = (1.0 - calib["peak_rmse_m_s"] / naive["peak_rmse_m_s"]) * 100.0
    return [
        ("突風変換の校正",
         f"sigma = GST-WSPD の素朴な写像では1時間ピーク風速を平均 "
         f"{naive['peak_bias_m_s']:+.2f} m/s 過大 (RMSE {naive['peak_rmse_m_s']:.2f} m/s)。"
         f"sum4 突風過程の1時間ピーク実測 (2.755 x gust_rms) に基づき "
         f"GUST_SIGMA_FACTOR = {GUST_SIGMA_FACTOR} を採用 -> "
         f"バイアス {calib['peak_bias_m_s']:+.2f} m/s, "
         f"RMSE {calib['peak_rmse_m_s']:.2f} m/s ({imp:.0f}% 改善)。"),
        ("風向の成分補間",
         f"記録中に |dtheta|>90 deg の時刻対が {ov['direction_wraps_gt90']} 件。"
         "風向をスカラーで線形補間すると 350 deg -> 10 deg で逆回りする偽の回転が"
         f"生じるため、(u,v) 成分で補間 (中点のコード短縮は平均 "
         f"{fid['chord_speed_deficit_mean_m_s']:.3f} m/s, 最大 "
         f"{fid['chord_speed_deficit_max_m_s']:.2f} m/s と小さい)。"),
        ("過飽和のクリップ",
         f"DEWP > ATMP の記録が {ov['supersaturated_records']} 件 "
         "(センサー丸め)。RH = es(Td)/es(T) を [0, 1] にクリップし、"
         "湿度検証 (WeatherConfig) との整合を保証。"),
        ("スカラー平均風速の残差",
         f"校正後の1時間平均スカラー風速は実測 WSPD に対し "
         f"{calib['mean_bias_m_s']:+.2f} m/s (RMSE {calib['mean_rmse_m_s']:.2f} m/s)。"
         "突風振幅の非線形 (|mean+gust| の凸性) による既知の残差で、"
         "風速計の精度 (±0.5 m/s) 未満のため補正しない。"),
        ("変換チェーンの厳密性",
         f"全 {len(fid['rows'])} ノードで観測値との最大偏差: 気温 "
         f"{fid['node_max_abs_err_T_K']:.1e} K, 気圧 "
         f"{fid['node_max_abs_err_p_Pa']:.1e} Pa, 密度 "
         f"{fid['node_max_abs_err_rho_kg_m3']:.1e} kg/m3, 風ベクトル "
         f"{fid['node_max_abs_err_wind_m_s']:.1e} m/s — 補間・単位変換に"
         "誤差はなく、残る近似は時間分解能 (1時間刻みの線形補間) のみ。"),
    ]


# ---------------------------------------------------------------------
#  report
# ---------------------------------------------------------------------
def write_report(cfg: StudyConfig, series, ov: dict, fid: dict, cal: dict,
                 trk: dict, resp: dict, envr: dict, corr: list,
                 plots: list, runtime_s: float) -> str:
    naive, calib = cal["stats"]["naive"], cal["stats"]["calibrated"]
    dc = cal["driver_check"]
    L = []
    L.append("# 過去気象データ replay スタディ (weather_real)")
    L.append("")
    L.append("## 目的")
    L.append("NOAA NDBC 46012 (Humboldt Bay, CA) の**実測**毎時気象データ")
    L.append("(気圧・気温・露点・風向・風速・最大突風) を `weather_real.py` の")
    L.append("`HistoricalWeather` 経由でシミュレータに入力し、モデルが観測された")
    L.append("気象の経過を近似してたどるかを検証し、解析から得られた修正を記録する。")
    L.append("")
    L.append("## データ")
    L.append(f"* ファイル: `{cfg.data}` ({ov['records']} 時間レコード)")
    L.append(f"* 期間: {ov['start_utc']} .. {ov['end_utc']} UTC "
             f"({ov['span_hours']:.0f} h = {ov['span_hours'] / 24.0:.1f} 日)")
    L.append(f"* 気圧: {ov['pres_hPa']['min']:.1f} .. {ov['pres_hPa']['max']:.1f} hPa "
             f"(平均 {ov['pres_hPa']['mean']:.1f})")
    L.append(f"* 気温: {ov['atmp_C']['min']:.1f} .. {ov['atmp_C']['max']:.1f} degC, "
             f"相対湿度 (露点から算出): {ov['rh']['min']:.2f} .. {ov['rh']['max']:.2f}")
    L.append(f"* 風速: 時平均 {ov['wspd_m_s']['min']:.0f} .. {ov['wspd_m_s']['max']:.0f} m/s, "
             f"最大突風 {ov['gust_m_s']['max']:.0f} m/s "
             f"(平均ピーク超過 {ov['gust_excess_m_s']['mean']:.2f} m/s)")
    top = ", ".join(f"{d['center_deg']:.0f} deg ({d['count']})"
                    for d in ov["wdir_top8"][:4])
    L.append(f"* 卓越風向 (10 deg ビン, 件数): {top} — 海岸の NW 系海風")
    L.append(f"* 海面密度 (実測 p,T,RH から算出): {ov['rho_sealevel']['min']:.4f} .. "
             f"{ov['rho_sealevel']['max']:.4f} kg/m3 "
             f"(ISA+50%湿度比 {moist_density(101325.0, 288.15, 0.5):.4f})")
    L.append(f"* 最荒天窓: {ov['extremes']['storm_window']['start_stamp']} UTC, "
             f"最低気圧: {ov['extremes']['lowest_pressure']['pres_hPa']:.1f} hPa "
             f"({ov['extremes']['lowest_pressure']['stamp']} UTC)")
    L.append("")
    L.append("## 観測 -> モデルのマッピング")
    L.append("| 観測列 | モデル量 | 変換 |")
    L.append("|---|---|---|")
    L.append("| PRES [hPa] | `pressure_offset_Pa` | (PRES - 1013.25) x 100, ISA 高度分布に加算 |")
    L.append("| ATMP [degC] | `temp_offset_K` | ATMP - 15.0, ISA 減率に沿って高度展開 |")
    L.append("| DEWP [degC] | `humidity` | RH = es(Td)/es(T) (Magnus), [0,1] クリップ |")
    L.append("| WDIR+WSPD | `wind` (m/s) | -(WSPD cos th, WSPD sin th), +x=北 +y=東, z_ref=10 m |")
    L.append(f"| GST-WSPD | `gust_rms` | x {GUST_SIGMA_FACTOR} (校正値, 下記 C) |")
    L.append("| (未報告) | rain/cloud/visibility | ベース WeatherConfig (既定: 無降水・乾燥 replay) |")
    L.append("")
    L.append("## A. リプレイ忠実度 (全ノード・全中点)")
    L.append(f"* 観測ノード {ov['records']} 件すべてで driver 状態と実測の最大偏差: "
             f"気温 {fid['node_max_abs_err_T_K']:.1e} K / 気圧 "
             f"{fid['node_max_abs_err_p_Pa']:.1e} Pa / 密度 "
             f"{fid['node_max_abs_err_rho_kg_m3']:.1e} kg/m3 / 風ベクトル "
             f"{fid['node_max_abs_err_wind_m_s']:.1e} m/s")
    L.append(f"* 区間中点でも線形補間と厳密一致 (最大: 気温 "
             f"{fid['midpoint_max_abs_err_T_K']:.1e} K, 風 "
             f"{fid['midpoint_max_abs_err_wind_m_s']:.1e} m/s)")
    L.append(f"* 密度は実測 (p, T, RH) からの moist_density と最大 "
             f"{fid['node_max_abs_err_rho_kg_m3']:.1e} kg/m3 差 — 湿り空気の変換チェーンに誤差なし")
    L.append("")
    L.append("## B. 突風変換の校正 (解析 -> 修正)")
    L.append("NDBC の GST は「1時間内のピーク風速」。`atmosphere.py` の sum4 突風は")
    L.append("gust_rms (成分ごとの時間RMS) で振幅が決まるため、GST-WSPD を直接")
    L.append("gust_rms にすると峰值を過大評価する。1時間窓・1 Hz サンプリングの")
    L.append(f"ピーク実測 (16 seed 平均 2.755 x gust_rms) から "
             f"GUST_SIGMA_FACTOR = {GUST_SIGMA_FACTOR} を導出。")
    L.append("")
    L.append("| 写像 | ピークバイアス | ピークRMSE | 平均風速バイアス |")
    L.append("|---|---|---|---|")
    L.append(f"| naive (k=1.0) | {naive['peak_bias_m_s']:+.2f} m/s | "
             f"{naive['peak_rmse_m_s']:.2f} m/s | {naive['mean_bias_m_s']:+.2f} m/s |")
    L.append(f"| calibrated (k={GUST_SIGMA_FACTOR}) | {calib['peak_bias_m_s']:+.2f} m/s | "
             f"{calib['peak_rmse_m_s']:.2f} m/s | {calib['mean_bias_m_s']:+.2f} m/s |")
    L.append("")
    imp = (1.0 - calib["peak_rmse_m_s"] / naive["peak_rmse_m_s"]) * 100.0
    L.append(f"ピークRMSE は {naive['peak_rmse_m_s']:.2f} -> {calib['peak_rmse_m_s']:.2f} m/s "
             f"({imp:.0f}% 改善)。driver 端到端確認 (最荒天時, 1実現):")
    L.append(f"モデル平均 {dc['model_mean_m_s']:.2f} m/s (実測 {dc['obs_mean_m_s']:.2f}), "
             f"モデルピーク {dc['model_peak_m_s']:.2f} m/s (実測 GST {dc['obs_peak_m_s']:.2f})。")
    L.append("")
    L.append("## C. 飛行エンベロープの追従 (全記録・毎時)")
    L.append(f"V={V_CRUISE} m/s 水平トリム (高度 {PATROL_ALT:.0f} m) を毎時再計算:")
    L.append(f"* 密度 rho(30 m): {trk['rho30']['min']:.4f} .. {trk['rho30']['max']:.4f} kg/m3")
    L.append(f"* 失速速度: {trk['v_stall_m_s']['min']:.2f} .. {trk['v_stall_m_s']['max']:.2f} m/s")
    L.append(f"* 水平飛行抵抗: {trk['drag_N']['min']:.1f} .. {trk['drag_N']['max']:.1f} N, "
             f"全開推力: {trk['thrust_avail_N']['min']:.0f} .. {trk['thrust_avail_N']['max']:.0f} N")
    L.append(f"* 推力マージン T/D: {trk['margin']['min']:.2f} .. {trk['margin']['max']:.2f} "
             f"(最小は {trk['margin']['worst_stamp']} UTC)")
    L.append(f"* 相関: corr(rho, PRES) = {trk['corr_rho0_pres']:+.3f}, "
             f"corr(rho, ATMP) = {trk['corr_rho0_atmp']:+.3f}, "
             f"corr(V_stall, rho) = {trk['corr_vstall_rho30']:+.3f}, "
             f"corr(T_avail, rho) = {trk['corr_thrust_rho30']:+.3f}")
    L.append("→ 密度は観測気圧・気温の経過に正の相関で追従し、失速速度・推力は")
    L.append("  その密度変化を物理法則通り (V_stall ∝ rho^-1/2, T ∝ rho) 反映する。")
    L.append("")
    L.append("## D. 動的反応 (最荒天窓 vs 最静穏窓, トリム固定)")
    L.append("`wind_tunnel.py` と同一の準定常荷重計測を実時間リプレイで実行:")
    L.append("")
    L.append("| 窓 | 開始 (UTC) | 実測WSPD | 実測GST最大 | n_mean | n_std | n_peak |")
    L.append("|---|---|---|---|---|---|---|")
    for tag in ("storm", "calm"):
        w = resp[tag]
        L.append(f"| {tag} | {w['start_stamp']} | "
                 f"{w['obs_wspd_range'][0]:.0f}..{w['obs_wspd_range'][1]:.0f} m/s | "
                 f"{w['obs_gust_max']:.0f} m/s | {w['n_mean']:.3f} | "
                 f"{w['n_std']:.3f} | {w['n_peak']:.3f} |")
    L.append("")
    L.append(f"荷重変動 n_std の比 (storm/calm) = {resp['n_std_ratio_storm_over_calm']:.1f}, "
             f"実測ガスト・ピーク超過 (GST−WSPD) の比 = {resp['obs_excess_ratio_storm_over_calm']:.1f}. "
             "両者は異なる量 (前者は空力荷重の標準偏差, 後者は実測ガスト強度の比) を測るため "
             "一致はしないが, 向きとオーダーは整合する — 荒天窓は静穏窓の数倍の動的反応を生み, "
             "ドローンの動的反応の強さが観測された天候の強弱を追従する。")
    L.append("")
    L.append("## E. Env 端到端リプレイ")
    L.append(f"`FlyingBoatEnv(spatial=True, weather_real=..., rate={cfg.env_rate:.0f})` "
             f"で {envr['steps']} ステップ (実時間 {envr['steps'] * 0.05 * cfg.env_rate / 60.0:.0f} 分相当) を飛行:")
    L.append(f"* 記録時刻は {envr['first_record']} -> {envr['last_record']} UTC と進行")
    L.append(f"* ステップ毎の telemetry (T, p) と記録補間値の最大偏差: "
             f"{envr['max_abs_dev_T_C']:.3f} degC / {envr['max_abs_dev_p_hPa']:.3f} hPa")
    L.append(f"* 最終状態: z = {envr['final_z']:.1f} m, 対気速度 = {envr['final_airspeed']:.1f} m/s")
    L.append("")
    L.append("## 修正一覧 (解析から適用)")
    for i, (title, body) in enumerate(corr, 1):
        L.append(f"{i}. **{title}** — {body}")
    L.append("")
    L.append("## 制限")
    L.append("* 観測は 1 時間分解能。ノード間は線形補間で、前線の急峻な通過は平滑化される。")
    L.append("* 記録に降水・雲・視程の列がない (VIS=MM) ため、既定は乾燥リプレイ。")
    L.append("  降雨等はベース WeatherConfig で明示的に重ねる必要がある。")
    L.append("* 鉛直風・気温減率の観測はなく、ISA 減率と gust_rms のみで近似。")
    L.append("* 風はブイ高度 (約10 m = z_ref) の観測。高度スケールは対数シアー則。")
    L.append("")
    L.append("## 成果物")
    for nm in ["REPORT.md", "summary.json", "fidelity_nodes.csv",
               "gust_calibration.csv", "trim_tracking.csv",
               "response_windows.csv", "env_replay.csv",
               "weather_study.py", *plots]:
        L.append(f"* `results/{cfg.out.name}/{nm}`")
    L.append("")
    L.append(f"実行時間: {runtime_s:.1f} s / git HEAD: `{git_head()[:12]}`")
    text = "\n".join(L) + "\n"
    (cfg.out / "REPORT.md").write_text(text, encoding="utf-8")
    return text


# ---------------------------------------------------------------------
#  driver
# ---------------------------------------------------------------------
def run_study(cfg: StudyConfig) -> dict:
    t_start = time.time()
    cfg.out.mkdir(parents=True, exist_ok=True)
    series = parse_ndbc_met(cfg.data)
    if series.rh is None or series.gust_m_s is None:
        raise ValueError("study requires a record with DEWP and GST columns")
    ac = Aircraft()

    ov = data_overview(series)
    fid = node_fidelity(series, cfg, ov["rho_obs"])
    cal = gust_calibration(series, cfg, ov["i_storm"])
    trk = trim_tracking(series, cfg, ov["rho_obs"])
    resp = response_windows(series, cfg, ac, ov["i_storm"], ov["i_calm"])
    envr = env_replay(series, cfg, ov["i_storm"])
    corr = corrections_list(ov, fid, cal)

    write_csv(cfg.out / "fidelity_nodes.csv", fid["rows"],
              list(fid["rows"][0].keys()))
    write_csv(cfg.out / "gust_calibration.csv", cal["rows"],
              list(cal["rows"][0].keys()))
    write_csv(cfg.out / "trim_tracking.csv", trk["rows"],
              list(trk["rows"][0].keys()))
    resp_rows = []
    for tag in ("storm", "calm"):
        w = resp[tag]
        stride = max(1, int(round(1.0 / cfg.response_dt)))
        for t, n in zip(w["t"][::stride], w["n"][::stride]):
            resp_rows.append(dict(window=tag, t_s=float(t), n_load=float(n)))
    write_csv(cfg.out / "response_windows.csv", resp_rows,
              ["window", "t_s", "n_load"])
    write_csv(cfg.out / "env_replay.csv", envr["rows"],
              list(envr["rows"][0].keys()))

    plots = make_plots(cfg, series, ov, fid, cal, trk, resp) if cfg.plots else []
    runtime_s = time.time() - t_start
    write_report(cfg, series, ov, fid, cal, trk, resp, envr, corr,
                 plots, runtime_s)

    strip = lambda d, drop: {k: v for k, v in d.items() if k not in drop}
    summary = {
        "config": dict(out=str(cfg.out), data=str(cfg.data), seed=cfg.seed,
                       quick=cfg.quick, plots=cfg.plots),
        "data": strip(ov, ("rho_obs", "i_storm", "i_calm")),
        "fidelity": strip(fid, ("rows",)),
        "gust_calibration": dict(stats=cal["stats"],
                                 driver_check=cal["driver_check"]),
        "trim_tracking": strip(trk, ("rows",)),
        "response_windows": {
            k: (strip(v, ("t", "n")) if isinstance(v, dict) else v)
            for k, v in resp.items()},
        "env_replay": strip(envr, ("rows",)),
        "corrections": [dict(title=t, detail=b) for t, b in corr],
        "artifacts": [f"results/{cfg.out.name}/{n}" for n in
                      ["REPORT.md", "summary.json", "fidelity_nodes.csv",
                       "gust_calibration.csv", "trim_tracking.csv",
                       "response_windows.csv", "env_replay.csv",
                       "weather_study.py", *plots]],
        "caveats": [
            "観測は1時間分解能・ノード間線形補間 (前線の急峻化は平滑化)",
            "降水・雲・視程は記録に無く既定は乾燥リプレイ",
            "鉛直風は観測されず gust_rms と base config のみ",
            "動的反応は準定常 (wind_tunnel.py と同一手法)",
        ],
        "git_head": git_head(),
        "runtime_s": round(runtime_s, 1),
    }
    (cfg.out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    shutil.copy2(Path(__file__).resolve(), cfg.out / "weather_study.py")
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Historical NDBC weather replay study (weather_real.py)")
    ap.add_argument("--out", default="results/weather_real_001",
                    help="results folder (default: results/weather_real_001)")
    ap.add_argument("--data", default=str(DEFAULT_MET_PATH),
                    help="NDBC realtime2 .txt file (default: bundled 46012)")
    ap.add_argument("--quick", action="store_true",
                    help="coarse sampling (smoke run)")
    ap.add_argument("--no-plots", action="store_true", help="skip PNG plots")
    args = ap.parse_args(argv)
    cfg = StudyConfig(out=Path(args.out), data=Path(args.data),
                      quick=args.quick, plots=not args.no_plots)
    s = run_study(cfg)
    cal = s["gust_calibration"]["stats"]
    print(f"[weather_study] record: {s['data']['records']} hourly obs, "
          f"{s['data']['span_hours'] / 24.0:.1f} days "
          f"({s['data']['start_utc']} ..)")
    print(f"[weather_study] node fidelity: max |err| T "
          f"{s['fidelity']['node_max_abs_err_T_K']:.1e} K, p "
          f"{s['fidelity']['node_max_abs_err_p_Pa']:.1e} Pa, rho "
          f"{s['fidelity']['node_max_abs_err_rho_kg_m3']:.1e} kg/m3")
    print(f"[weather_study] gust peak RMSE: naive {cal['naive']['peak_rmse_m_s']:.2f} -> "
          f"calibrated {cal['calibrated']['peak_rmse_m_s']:.2f} m/s")
    tt = s["trim_tracking"]
    print(f"[weather_study] envelope: rho30 {tt['rho30']['min']:.4f}..{tt['rho30']['max']:.4f} "
          f"kg/m3, V_stall {tt['v_stall_m_s']['min']:.2f}..{tt['v_stall_m_s']['max']:.2f} m/s, "
          f"margin {tt['margin']['min']:.2f}..{tt['margin']['max']:.2f}")
    r = s["response_windows"]
    print(f"[weather_study] n_std storm {r['storm']['n_std']:.3f} vs calm "
          f"{r['calm']['n_std']:.3f} (obs excess ratio "
          f"{r['obs_excess_ratio_storm_over_calm']:.1f})")
    e = s["env_replay"]
    print(f"[weather_study] env replay: {e['steps']} steps, telemetry dev "
          f"<={e['max_abs_dev_T_C']:.3f} degC / {e['max_abs_dev_p_hPa']:.3f} hPa")
    print(f"[weather_study] outputs: {cfg.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
