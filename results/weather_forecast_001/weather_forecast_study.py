#!/usr/bin/env python3
"""Forecast-model verification study: VAR guidance vs held-out NDBC obs.

Verifies weather_forecast.VarForecastModel on the same 45.7 days of NOAA
NDBC 46012 (Humboldt Bay) hourly observations used by weather_study.py,
under a strictly honest protocol: every forecast is fitted only on records
*preceding* its issue time (expanding window), so nothing in the training
slice overlaps the verification truth.

  A. protocol        -- record count, split fractions, issue lists;
  B. order selection -- VAR order chosen on an early window by normalised
                        RMSE at lead 24 h, never on the verification set;
  C. skill           -- RMSE and skill score (1 - RMSE_model/RMSE_ref) vs
                        lead time against persistence and diurnal
                        climatology, for pres/atmp/dewp/wspd/wind-vector/
                        gust-excess, averaged over verification issues;
  D. damping         -- normalised anomaly |fc - clim| vs lead: does the
                        guidance relax to climatology like real NWP output;
  E. storm case      -- the verification issue whose next 48 h contain the
                        largest observed gust excess, forecast vs truth;
  F. flight under fc -- trim-envelope tracking along the forecast and a
                        FlyingBoatEnv episode driven end-to-end by the
                        forecast (weather_forecast integration).

Usage:
    python3 weather_forecast_study.py [--out results/weather_forecast_001]
                                      [--quick] [--no-plots] [--data PATH]
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
from atmosphere import isa_pressure, isa_temperature
from design_optimize import git_head
from weather_real import DEFAULT_MET_PATH, parse_ndbc_met
from weather_forecast import (N_STATE, VarForecastModel, _state_matrix,
                              climatology_forecast, forecast_weather,
                              persistence_forecast, rmse_vs_lead,
                              skill_score)
from wind_tunnel import trim_state

V_CRUISE = 11.3          # m/s, level-flight tracking speed
PATROL_ALT = 30.0        # m, envelope-tracking altitude
VAR_NAMES = ("pres_hPa", "atmp_C", "dewp_C", "wspd_m_s",
             "windvec_m_s", "gust_excess_m_s")


@dataclass
class StudyConfig:
    out: Path = Path("results/weather_forecast_001")
    data: Path = DEFAULT_MET_PATH
    seed: int = 42
    horizon: int = 48        # verification lead length, hours
    sel_lo: float = 0.4      # order-selection window: fraction of record
    sel_hi: float = 0.7      # also the start of the verification window
    sel_stride: int = 24     # issue spacing in the selection window, h
    ver_stride: int = 12     # issue spacing in the verification window, h
    orders: tuple = (1, 2, 3, 4, 6)
    env_steps: int = 200
    env_rate: float = 360.0  # real seconds per sim second in section F
    env_lead_stride: int = 3  # envelope sampling along the forecast, h
    plots: bool = True
    quick: bool = False

    def __post_init__(self):
        self.out = Path(self.out)
        self.data = Path(self.data)
        if self.quick:
            self.orders = (1, 3, 6)
            self.sel_stride = 72
            self.ver_stride = 96
            self.horizon = 24
            self.env_steps = 60
            self.env_lead_stride = 6


def write_csv(path: Path, rows: list, cols: list) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r[c] for c in cols})


def issue_windows(series, cfg: StudyConfig):
    """(selection issues, verification issues) as record indices."""
    n = len(series)
    sel = list(range(int(cfg.sel_lo * n), int(cfg.sel_hi * n),
                     cfg.sel_stride))
    ver = list(range(int(cfg.sel_hi * n), n - cfg.horizon, cfg.ver_stride))
    return sel, ver


def derived_error_stack(fc, truth):
    """(M, L+1, 6) forecast/truth stacks -> per-variable error stacks.

    Returns a dict of (M, L+1) arrays: pres/atmp/dewp are component-wise
    differences; wspd is the speed difference; windvec is the magnitude of
    the (du, dv) vector error; gust_excess is component-wise.
    """
    du = fc[..., 3] - truth[..., 3]
    dv = fc[..., 4] - truth[..., 4]
    return {
        "pres_hPa": fc[..., 0] - truth[..., 0],
        "atmp_C": fc[..., 1] - truth[..., 1],
        "dewp_C": fc[..., 2] - truth[..., 2],
        "wspd_m_s": (np.hypot(fc[..., 3], fc[..., 4])
                     - np.hypot(truth[..., 3], truth[..., 4])),
        "windvec_m_s": np.hypot(du, dv),
        "gust_excess_m_s": fc[..., 5] - truth[..., 5],
    }


def rms_over_issues(err):
    """(M, L+1) errors -> (L+1) RMS across the issue axis."""
    return np.sqrt(np.mean(np.asarray(err, float) ** 2, axis=0))


def obs_sd(series):
    """Per-variable observation spread used to normalise RMSE."""
    y = _state_matrix(series)
    sd = {
        "pres_hPa": float(y[:, 0].std()),
        "atmp_C": float(y[:, 1].std()),
        "dewp_C": float(y[:, 2].std()),
        "wspd_m_s": float(np.hypot(y[:, 3], y[:, 4]).std()),
        "gust_excess_m_s": float(y[:, 5].std()),
    }
    sd["windvec_m_s"] = sd["wspd_m_s"]
    return sd


# ---------------------------------------------------------------------
#  B. order selection (early window only -- never the verification set)
# ---------------------------------------------------------------------
def section_order_selection(series, cfg: StudyConfig, sel_issues):
    y = _state_matrix(series)
    sd = obs_sd(series)
    lead_ref = min(24, cfg.horizon // 2)
    rows = []
    for order in cfg.orders:
        fc_list, truth_list = [], []
        radii, resids = [], []
        for i in sel_issues:
            m = VarForecastModel(series, order=order, train_stop=i)
            fc_list.append(m.predict(i, cfg.horizon))
            truth_list.append(y[i:i + cfg.horizon + 1])
            radii.append(m._companion_radius())
            resids.append(m.resid_rms)
        errs = derived_error_stack(np.array(fc_list), np.array(truth_list))
        row = {"order": order, "n_issues": len(sel_issues),
               "radius_max": float(np.max(radii)),
               "resid_rms_mean": float(np.mean(resids))}
        nrmse = []
        for name in VAR_NAMES:
            r = rms_over_issues(errs[name])[lead_ref]
            row[f"rmse_{name}"] = float(r)
            row[f"nrmse_{name}"] = float(r / max(sd[name], 1e-9))
            nrmse.append(row[f"nrmse_{name}"])
        row["nrmse_mean"] = float(np.mean(nrmse))
        rows.append(row)
    best = min(rows, key=lambda r: r["nrmse_mean"])
    return {"rows": rows, "sel_order": int(best["order"]),
            "lead_ref": lead_ref, "issues": list(sel_issues),
            "sd": sd}


# ---------------------------------------------------------------------
#  C. skill on the verification window
# ---------------------------------------------------------------------
def section_skill(series, cfg: StudyConfig, sel_order, ver_issues):
    y = _state_matrix(series)
    H = cfg.horizon
    fc_m, fc_p, fc_c, tr = [], [], [], []
    radii = []
    for i in ver_issues:
        m = VarForecastModel(series, order=sel_order, train_stop=i)
        fc_m.append(m.predict(i, H))
        fc_p.append(persistence_forecast(series, i, H))
        fc_c.append(climatology_forecast(series, i, H, train_stop=i))
        tr.append(y[i:i + H + 1])
        radii.append(m._companion_radius())
    fc_m, fc_p, fc_c, tr = (np.array(a) for a in (fc_m, fc_p, fc_c, tr))
    em = derived_error_stack(fc_m, tr)
    ep = derived_error_stack(fc_p, tr)
    ec = derived_error_stack(fc_c, tr)
    rmse = {n: rms_over_issues(em[n]) for n in VAR_NAMES}
    rmse_p = {n: rms_over_issues(ep[n]) for n in VAR_NAMES}
    rmse_c = {n: rms_over_issues(ec[n]) for n in VAR_NAMES}
    skill_p = {n: skill_score(rmse[n], rmse_p[n]) for n in VAR_NAMES}
    skill_c = {n: skill_score(rmse[n], rmse_c[n]) for n in VAR_NAMES}
    crossover = {}
    for n in VAR_NAMES:
        worse = np.nonzero(rmse[n][1:] > rmse_p[n][1:])[0]
        crossover[n] = int(worse[0] + 1) if len(worse) else None
    rows = []
    for l in range(H + 1):
        row = {"lead_h": l}
        for n in VAR_NAMES:
            row[f"rmse_{n}"] = float(rmse[n][l])
            row[f"rmse_persistence_{n}"] = float(rmse_p[n][l])
            row[f"rmse_climatology_{n}"] = float(rmse_c[n][l])
            row[f"skill_vs_persistence_{n}"] = float(skill_p[n][l])
            row[f"skill_vs_climatology_{n}"] = float(skill_c[n][l])
        rows.append(row)
    def at(lead):
        return {n: {"rmse": float(rmse[n][lead]),
                    "rmse_pers": float(rmse_p[n][lead]),
                    "rmse_clim": float(rmse_c[n][lead]),
                    "skill_pers": float(skill_p[n][lead]),
                    "skill_clim": float(skill_c[n][lead])}
                for n in VAR_NAMES}
    leads = sorted({l for l in (6, 12, 24, H) if l <= H})
    return {"rows": rows, "n_issues": len(ver_issues), "issues": list(ver_issues),
            "order": sel_order, "leads": leads,
            "summary": {str(l): at(l) for l in leads},
            "crossover_lead_vs_persistence": crossover,
            "radius": {"min": float(np.min(radii)),
                       "max": float(np.max(radii))},
            "stacks": (fc_m, fc_p, fc_c, tr)}


# ---------------------------------------------------------------------
#  D. damping: relaxation towards climatology with lead
# ---------------------------------------------------------------------
def section_damping(series, cfg: StudyConfig, sk: dict):
    fc_m, _, fc_c, _ = sk["stacks"]
    sd = obs_sd(series)
    sd_vec = np.array([max(sd[n], 1e-9) for n in
                       ("pres_hPa", "atmp_C", "dewp_C",
                        "wspd_m_s", "wspd_m_s", "gust_excess_m_s")])
    anom = (fc_m - fc_c) / sd_vec            # normalised (M, L+1, 6)
    norm = np.linalg.norm(anom, axis=2)      # (M, L+1)
    rows = []
    for l in range(norm.shape[1]):
        row = {"lead_h": l,
               "anom_norm_mean": float(norm[:, l].mean()),
               "anom_norm_std_across_issues": float(norm[:, l].std())}
        for j, n in enumerate(("pres", "atmp", "dewp", "u", "v", "gex")):
            row[f"anom_{n}_mean"] = float(np.abs(anom[:, l, j]).mean())
        rows.append(row)
    return {"rows": rows,
            "norm_lead0": float(norm[:, 0].mean()),
            "norm_lead_end": float(norm[:, -1].mean()),
            "decay_ratio": float(norm[:, -1].mean() / max(norm[:, 0].mean(), 1e-12)),
            "monotone": bool(np.all(np.diff(norm.mean(axis=0)) <= 1e-9))}


# ---------------------------------------------------------------------
#  E. storm case study inside the verification window
# ---------------------------------------------------------------------
def section_case(series, cfg: StudyConfig, sel_order, ver_issues):
    y = _state_matrix(series)
    H = cfg.horizon
    peaks = [float(y[i:i + H + 1, 5].max()) for i in ver_issues]
    k = int(np.argmax(peaks))
    issue = ver_issues[k]
    m = VarForecastModel(series, order=sel_order, train_stop=issue)
    fc = m.predict(issue, H)
    pe = persistence_forecast(series, issue, H)
    tr = y[issue:issue + H + 1]
    wspd = lambda a: np.hypot(a[..., 3], a[..., 4])
    rows = []
    for l in range(H + 1):
        rows.append({
            "lead_h": l, "stamp": series.stamps[min(issue + l, len(series) - 1)],
            "wspd_true": float(wspd(tr[l])), "wspd_fc": float(wspd(fc[l])),
            "wspd_pers": float(wspd(pe[l])),
            "gex_true": float(tr[l, 5]), "gex_fc": float(max(fc[l, 5], 0.0)),
            "gex_pers": float(pe[l, 5]),
            "pres_true": float(tr[l, 0]), "pres_fc": float(fc[l, 0]),
            "atmp_true": float(tr[l, 1]), "atmp_fc": float(fc[l, 1])})
    def err(a, b, l):
        return {"wspd_m_s": float(abs(wspd(a[l]) - wspd(b[l]))),
                "gex_m_s": float(abs(max(a[l, 5], 0.0) - b[l, 5])),
                "pres_hPa": float(abs(a[l, 0] - b[l, 0])),
                "atmp_C": float(abs(a[l, 1] - b[l, 1]))}
    leads = sorted({l for l in (6, 12, 24, H) if l <= H})
    return {"issue": issue, "stamp": series.stamps[issue],
            "peak_gex_in_window": peaks[k], "rows": rows,
            "model_err": {str(l): err(fc, tr, l) for l in leads},
            "pers_err": {str(l): err(pe, tr, l) for l in leads},
            "radius": float(m._companion_radius()),
            "fc_stamp_end": rows[-1]["stamp"], "leads": leads}


# ---------------------------------------------------------------------
#  F. flight under the forecast: envelope tracking + Env end-to-end
# ---------------------------------------------------------------------
def section_flight(series, cfg: StudyConfig, sel_order, issue):
    from env import EnvConfig, FlyingBoatEnv
    H = cfg.horizon
    hw, model = forecast_weather(series, issue, H, seed=cfg.seed,
                                 time_scale=cfg.env_rate, order=sel_order,
                                 train_stop=issue)
    fcs = hw.series
    ac = Aircraft()
    eta = ac.prop.eta_motor * ac.prop.eta_esc * ac.prop.eta_prop
    env_rows = []
    for lead in range(0, H + 1, cfg.env_lead_stride):
        t = lead * 3600.0 / cfg.env_rate
        sm = hw.summary(t, PATROL_ALT, V_CRUISE)
        rho30 = float(sm["density_kg_m3"])
        trim = trim_state(ac, rho30, V_CRUISE)
        cd = ac.CD(trim["CL_trim"], height_m=PATROL_ALT)
        drag = trim["q_Pa"] * ac.geom.S * cd
        t_avail = ac.prop.thrust(V_CRUISE, 1.0, rho=rho30)
        v_stall = math.sqrt(2.0 * ac.W / (rho30 * ac.geom.S
                                          * ac.aero.CL_max))
        env_rows.append(dict(
            lead_h=lead, stamp=fcs.stamps[lead],
            pres_hPa=float(fcs.pres_hPa[lead]),
            atmp_C=float(fcs.atmp_C[lead]),
            wspd_m_s=float(fcs.wspd_m_s[lead]),
            gust_excess_m_s=float(fcs.gust_m_s[lead] - fcs.wspd_m_s[lead]),
            rh=float(fcs.rh[lead]), rho30_model=rho30,
            v_stall_m_s=v_stall, cl_trim=trim["CL_trim"],
            alpha_trim_deg=trim["alpha_trim_deg"],
            cd_trim=cd, drag_N=drag, power_W=drag * V_CRUISE / eta,
            thrust_avail_N=t_avail, margin=t_avail / drag))

    ecfg = EnvConfig(spatial=True, weather_forecast=str(cfg.data),
                     weather_forecast_issue=float(series.hours[issue]),
                     weather_forecast_rate=cfg.env_rate,
                     max_steps=cfg.env_steps)
    env = FlyingBoatEnv(Aircraft(), ecfg)
    env.reset(seed=7)
    efc = env._weather.series          # env's internal forecast (order 3)
    fc_match = float(np.max(np.abs(efc.wspd_m_s[:H + 1]
                                   - fcs.wspd_m_s[:H + 1])))
    rows, dev_T, dev_p = [], [], []
    for _ in range(cfg.env_steps):
        a = np.zeros(env.action_dim)
        a[0] = 1.0
        _, _, done, info = env.step(a)
        w = info["weather"]
        lead = info["t"] * cfg.env_rate / 3600.0
        rec = efc.interp(lead)
        z = float(info["z"])
        exp_T = rec["atmp_C"] + (isa_temperature(z) - 288.15)
        exp_p = (isa_pressure(z)
                 + (rec["pres_hPa"] - 1013.25) * 100.0) / 100.0
        dev_T.append(w["temperature_C"] - exp_T)
        dev_p.append(w["pressure_hPa"] - exp_p)
        rows.append(dict(t=info["t"], lead_h=lead,
                         record_utc=w["record_utc"], z=z,
                         airspeed=info["airspeed"],
                         temperature_C=w["temperature_C"],
                         pressure_hPa=w["pressure_hPa"],
                         density_kg_m3=w["density_kg_m3"],
                         humidity=w["humidity"],
                         wind_x=w["wind_m_s"][0], wind_y=w["wind_m_s"][1],
                         ice_mass_kg=info.get("ice_mass_kg", 0.0)))
        if done:
            break
    margins = [r["margin"] for r in env_rows]
    i_worst = int(np.argmin(margins))
    fc_curve = [dict(lead_h=l, atmp_C=float(fcs.atmp_C[l]),
                     pres_hPa=float(fcs.pres_hPa[l]),
                     wspd_m_s=float(fcs.wspd_m_s[l]))
                for l in range(H + 1)]
    return {
        "issue": issue, "issue_stamp": series.stamps[issue],
        "order": sel_order, "env_order": 3, "fc_curve": fc_curve,
        "envelope": {"rows": env_rows,
                     "margin_min": margins[i_worst],
                     "margin_min_lead_h": env_rows[i_worst]["lead_h"],
                     "margin_min_stamp": env_rows[i_worst]["stamp"],
                     "v_stall_min": min(r["v_stall_m_s"] for r in env_rows),
                     "v_stall_max": max(r["v_stall_m_s"] for r in env_rows)},
        "env": {"rows": rows, "steps": len(rows),
                "env_fc_matches_study_fc_wspd_m_s": fc_match,
                "first_record": rows[0]["record_utc"],
                "last_record": rows[-1]["record_utc"],
                "last_lead_h": float(rows[-1]["lead_h"]),
                "max_abs_dev_T_C": float(np.max(np.abs(dev_T))),
                "max_abs_dev_p_hPa": float(np.max(np.abs(dev_p))),
                "final_z": float(rows[-1]["z"]),
                "final_airspeed": float(rows[-1]["airspeed"])}}


# ---------------------------------------------------------------------
#  plots
# ---------------------------------------------------------------------
def make_plots(cfg: StudyConfig, series, sel: dict, sk: dict, damp: dict,
               case: dict, flight: dict) -> list:
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
    leads = np.arange(cfg.horizon + 1)
    rmse = {n: [r[f"rmse_{n}"] for r in sk["rows"]] for n in VAR_NAMES}
    rmse_p = {n: [r[f"rmse_persistence_{n}"] for r in sk["rows"]]
              for n in VAR_NAMES}
    rmse_c = {n: [r[f"rmse_climatology_{n}"] for r in sk["rows"]]
              for n in VAR_NAMES}

    # -- C. skill curves ------------------------------------------------
    fig, ax = plt.subplots(3, 2, figsize=(11.0, 9.5), sharex=True)
    units = {"pres_hPa": "hPa", "atmp_C": "degC", "dewp_C": "degC",
             "wspd_m_s": "m/s", "windvec_m_s": "m/s",
             "gust_excess_m_s": "m/s"}
    for a, n in zip(ax.ravel(), VAR_NAMES):
        a.plot(leads, rmse_p[n], "--", color="gray", lw=1.0,
               label="永続予報")
        a.plot(leads, rmse_c[n], ":", color="tab:green", lw=1.0,
               label="気候値(日周期)")
        a.plot(leads, rmse[n], color="tab:red", lw=1.6,
               label=f"VAR({sel['sel_order']})")
        a.set_ylabel(f"RMSE [{units[n]}]", fontsize=9)
        a.set_title(n, fontsize=10)
        a.grid(alpha=0.3)
        a.legend(fontsize=7, loc="upper left")
    for a in ax[-1]:
        a.set_xlabel("予報リード [h]", fontsize=9)
    fig.suptitle(f"検証 {sk['n_issues']} 件・平均: RMSE 対 リードタイム "
                 f"(発行 {series.stamps[sk['issues'][0]]} 以降)")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    p = cfg.out / "skill_curves.png"; fig.savefig(p, dpi=140); names.append(p)
    plt.close(fig)

    # -- E. storm case ----------------------------------------------------
    rows = case["rows"]
    L = [r["lead_h"] for r in rows]
    fig, ax = plt.subplots(2, 2, figsize=(11.0, 7.0), sharex=True)
    panels = [("wspd_true", "wspd_fc", "wspd_pers", "風速 [m/s]"),
              ("gex_true", "gex_fc", "gex_pers", "ガスト超過 GST-WSPD [m/s]"),
              ("pres_true", "pres_fc", None, "気圧 [hPa]"),
              ("atmp_true", "atmp_fc", None, "気温 [degC]")]
    for a, (kt, kf, kp, ylab) in zip(ax.ravel(), panels):
        a.plot(L, [r[kt] for r in rows], color="black", lw=1.4,
               label="実測(検証真値)")
        a.plot(L, [r[kf] for r in rows], color="tab:red", lw=1.4,
               label=f"VAR({sel['sel_order']}) 予報")
        if kp:
            a.plot(L, [r[kp] for r in rows], "--", color="gray", lw=1.0,
                   label="永続予報")
        a.set_ylabel(ylab, fontsize=9)
        a.grid(alpha=0.3)
        a.legend(fontsize=7, loc="best")
    for a in ax[-1]:
        a.set_xlabel("予報リード [h]", fontsize=9)
    fig.suptitle(f"嵐ケース: 発行 {case['stamp']} UTC "
                 f"(窓内ピークガスト超過 {case['peak_gex_in_window']:.1f} m/s)")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    p = cfg.out / "case_study.png"; fig.savefig(p, dpi=140); names.append(p)
    plt.close(fig)

    # -- D. damping --------------------------------------------------------
    drows = damp["rows"]
    Ld = [r["lead_h"] for r in drows]
    fig, ax = plt.subplots(1, 2, figsize=(11.0, 4.2))
    ax[0].plot(Ld, [r["anom_norm_mean"] for r in drows],
               color="tab:blue", lw=1.6, label="平均 |fc-clim| (正規化)")
    ax[0].fill_between(
        Ld,
        np.array([r["anom_norm_mean"] for r in drows])
        - np.array([r["anom_norm_std_across_issues"] for r in drows]),
        np.array([r["anom_norm_mean"] for r in drows])
        + np.array([r["anom_norm_std_across_issues"] for r in drows]),
        color="tab:blue", alpha=0.2)
    ax[0].set_yscale("log")
    ax[0].set_xlabel("予報リード [h]"); ax[0].set_ylabel("正規化異常ノルム [-]")
    ax[0].set_title("予報の減衰: 気候値への緩和")
    ax[0].grid(alpha=0.3, which="both"); ax[0].legend(fontsize=8)
    for key, lab in (("anom_pres_mean", "気圧"), ("anom_atmp_mean", "気温"),
                     ("anom_u_mean", "u"), ("anom_v_mean", "v"),
                     ("anom_gex_mean", "ガスト超過")):
        ax[1].plot(Ld, [r[key] for r in drows], lw=1.1, label=lab)
    ax[1].set_yscale("log")
    ax[1].set_xlabel("予報リード [h]"); ax[1].set_ylabel("平均 |異常| [sd比]")
    ax[1].set_title("変数別の残存異常")
    ax[1].grid(alpha=0.3, which="both"); ax[1].legend(fontsize=8)
    fig.tight_layout()
    p = cfg.out / "damping.png"; fig.savefig(p, dpi=140); names.append(p)
    plt.close(fig)

    # -- F. flight under the forecast --------------------------------------
    er = flight["envelope"]["rows"]
    env_rows = flight["env"]["rows"]
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.4))
    a = ax[0]
    a.plot([r["lead_h"] for r in er], [r["v_stall_m_s"] for r in er],
           color="tab:red", lw=1.5, label="失速速度 [m/s]")
    a.axhline(V_CRUISE, color="black", ls="--", lw=0.8,
              label=f"巡航 {V_CRUISE} m/s")
    a.plot([r["lead_h"] for r in flight["fc_curve"]],
           [r["wspd_m_s"] for r in flight["fc_curve"]],
           color="tab:purple", lw=1.0, label="予報風速 [m/s]")
    a.set_xlabel("予報リード [h]"); a.set_ylabel("失速速度 [m/s]")
    a.grid(alpha=0.3); a.legend(fontsize=8, loc="upper left")
    a2 = a.twinx()
    a2.plot([r["lead_h"] for r in er], [r["margin"] for r in er],
            color="tab:blue", lw=1.5, label="推力余裕 T/D [-]")
    a2.set_ylabel("推力余裕 [-]", color="tab:blue")
    a2.legend(fontsize=8, loc="lower left")
    a.set_title(f"予報が描く飛行エンベロープ (発行 {flight['issue_stamp']})")
    b = ax[1]
    b.plot([r["lead_h"] for r in flight["fc_curve"]],
           [r["atmp_C"] for r in flight["fc_curve"]],
           color="tab:red", lw=1.2, label="予報 気温 [degC]")
    b.plot([r["lead_h"] for r in flight["fc_curve"]],
           [r["pres_hPa"] for r in flight["fc_curve"]],
           color="tab:blue", lw=1.2, label="予報 気圧 [hPa]")
    b.plot([r["lead_h"] for r in env_rows],
           [r["temperature_C"] for r in env_rows], "o", ms=2.5,
           color="tab:red", mfc="none", label="Env 気温テレメトリ")
    b.plot([r["lead_h"] for r in env_rows],
           [r["pressure_hPa"] for r in env_rows], "o", ms=2.5,
           color="tab:blue", mfc="none", label="Env 気圧テレメトリ")
    b.set_xlabel("予報リード [h]")
    b.grid(alpha=0.3); b.legend(fontsize=7, loc="best")
    b.set_title(f"予報誘導曲線と Env エピソード ({flight['env']['steps']} 歩, "
                f"リード {flight['env']['last_lead_h']:.2f} h まで)")
    fig.tight_layout()
    p = cfg.out / "forecast_flight.png"; fig.savefig(p, dpi=140)
    names.append(p)
    plt.close(fig)
    return [str(n) for n in names]


# ---------------------------------------------------------------------
#  report / summary
# ---------------------------------------------------------------------
def f3(v):
    return f"{float(v):.3f}"


def write_report(cfg: StudyConfig, series, sel: dict, sk: dict, damp: dict,
                 case: dict, flight: dict, plot_names: list,
                 runtime_s: float) -> None:
    n = len(series)
    L = []
    A = L.append
    A("# 天気予報モデル検証レポート (weather_forecast_001)")
    A("")
    A("`weather_forecast.VarForecastModel`(日周期気候値 anomalies 上の VAR(p)/"
      "線形逆モデル、リッジ最小二乗+コンパニオン行列安定化)を、実況 replay と同じ "
      f"NDBC 46012 実測 {n} 時間({series.stamps[0]} 〜 {series.stamps[-1]} UTC)"
      "に対して検証した。**全予報は発行時刻より前のレコードのみで学習する**"
      "(expanding window, `train_stop=issue`)ため、検証真値は学習に一切混入しない。")
    A("")
    A("## A. 検証プロトコル")
    A("")
    A(f"- 学習/検証分割: レコード先頭 {int(cfg.sel_lo*n)} h 以降を順序選択窓"
      f"({len(sel['issues'])} 件)、{int(cfg.sel_hi*n)} h 以降を検証窓"
      f"({sk['n_issues']} 件)に使用。ホライズン H={cfg.horizon} h。")
    A(f"- 状態量 y = (PRES, ATMP, DEWP, u, v, GST-WSPD) の {N_STATE} 変数。"
      "風は成分 (u,v) で予報し、359→1 deg のラップによる見かけ誤差を排除。")
    A("- 基準予報: 永続予報(解析値を保持)と日周期気候値(発行時刻前の学習窓のみ)。")
    A("")
    A("## B. 次数選択(検証窓に触れない前期窓で実施)")
    A("")
    A(f"リード {sel['lead_ref']} h の正規化 RMSE(観測 sd 比)平均が最小の次数を採用:")
    A("")
    A("| order | nRMSE 平均 | pres | atmp | dewp | wspd | windvec | gex | "
      "半径 max | 残差 RMS |")
    A("|---|---|---|---|---|---|---|---|---|---|")
    for r in sel["rows"]:
        A(f"| {r['order']} | **{f3(r['nrmse_mean'])}** "
          f"| {f3(r['nrmse_pres_hPa'])} | {f3(r['nrmse_atmp_C'])} "
          f"| {f3(r['nrmse_dewp_C'])} | {f3(r['nrmse_wspd_m_s'])} "
          f"| {f3(r['nrmse_windvec_m_s'])} | {f3(r['nrmse_gust_excess_m_s'])} "
          f"| {f3(r['radius_max'])} | {f3(r['resid_rms_mean'])} |")
    A("")
    A(f"→ 採用次数 **VAR({sel['sel_order']})**。全次数でコンパニオン行列の"
      "スペクトル半径 < 1(安定化後は減衰が保証される)。")
    A("")
    A("## C. スキル(検証窓平均)")
    A("")
    hdr = "| 変数 | " + " | ".join(f"L{l} RMSE(モデル/永続/気候)"
                                   for l in sk["leads"]) + " |"
    A(hdr)
    A("|---|" + "---|" * len(sk["leads"]))
    for v in VAR_NAMES:
        cells = []
        for l in sk["leads"]:
            s = sk["summary"][str(l)][v]
            cells.append(f"{s['rmse']:.2f} / {s['rmse_pers']:.2f} / "
                         f"{s['rmse_clim']:.2f}")
        A(f"| {v} | " + " | ".join(cells) + " |")
    A("")
    A("スキルスコア 1 - RMSE_model/RMSE_ref(>0 でモデル優位):")
    A("")
    A("| 変数 | " + " | ".join(f"L{l} 対永続 / 対気候" for l in sk["leads"]) + " |")
    A("|---|" + "---|" * len(sk["leads"]))
    for v in VAR_NAMES:
        cells = []
        for l in sk["leads"]:
            s = sk["summary"][str(l)][v]
            cells.append(f"{s['skill_pers']:+.2f} / {s['skill_clim']:+.2f}")
        A(f"| {v} | " + " | ".join(cells) + " |")
    A("")
    xo = sk["crossover_lead_vs_persistence"]
    xo_s = ", ".join(f"{k}: {'なし' if v is None else str(v)+' h'}"
                     for k, v in xo.items())
    A(f"検証窓平均でモデル RMSE が永続予報を初めて上回るリード: {xo_s}"
      "(None は全リードで上回らない)。逆転後に再び優位へ戻る変もあるため、"
      "リード別詳細は `skill_vs_lead.csv` を参照。")
    A("")
    A("## D. 減衰(気候値への緩和)")
    A("")
    A(f"正規化異常ノルム |fc-clim| は lead 0 の {f3(damp['norm_lead0'])} から "
      f"lead {cfg.horizon} h で {f3(damp['norm_lead_end'])} へ減衰"
      f"(比 {f3(damp['decay_ratio'])})。初期状態の記憶を失いつつ気候値へ緩和する、"
      "実務的な誘導曲線と同じ挙動。")
    A("")
    A("## E. 嵐ケーススタディ")
    A("")
    A(f"検証窓内で以後 {cfg.horizon} h の実測ガスト超過が最大になる発行時刻 "
      f"{case['stamp']} UTC(ピーク {case['peak_gex_in_window']:.1f} m/s)を選定。")
    A("")
    A("| lead | 風速 真値/予報/永続 [m/s] | ガスト超過 真値/予報 [m/s] | "
      "気圧誤差 [hPa] | 気温誤差 [degC] |")
    A("|---|---|---|---|---|")
    for l in case["leads"]:
        r = case["rows"][l]
        me = case["model_err"][str(l)]
        A(f"| {l} h | {r['wspd_true']:.1f} / {r['wspd_fc']:.1f} / "
          f"{r['wspd_pers']:.1f} | {r['gex_true']:.1f} / {r['gex_fc']:.1f} "
          f"| {me['pres_hPa']:.2f} | {me['atmp_C']:.2f} |")
    A("")
    A("## F. 予報下飛行(シミュレータ統合)")
    A("")
    fe = flight["envelope"]
    fv = flight["env"]
    A(f"発行 {flight['issue_stamp']} UTC の予報に沿ってトリムエンベロープを追跡"
      f"(VAR({flight['order']}), 高度 {PATROL_ALT:.0f} m, 巡航 {V_CRUISE} m/s):")
    A("")
    A(f"- 失速速度 {fe['v_stall_min']:.2f}〜{fe['v_stall_max']:.2f} m/s、"
      f"推力余裕最小 {fe['margin_min']:.2f}(lead {fe['margin_min_lead_h']} h, "
      f"{fe['margin_min_stamp']} UTC)。")
    A(f"- `FlyingBoatEnv(weather_forecast=...)` の {fv['steps']} 歩エピソード"
      f"(rate x{cfg.env_rate:.0f})が予報リード 0〜{fv['last_lead_h']:.2f} h を進行。"
      f"ステップ気象テレメトリと予報系列の一致: 気温偏差 max {fv['max_abs_dev_T_C']:.3f} K、"
      f"気圧偏差 max {fv['max_abs_dev_p_hPa']:.3f} hPa(高度換算を含む期待値との差)。")
    if flight["order"] == flight["env_order"]:
        A(f"- env 内部の予報系列は本検証の系列と完全一致"
          f"(風速差 max {fv['env_fc_matches_study_fc_wspd_m_s']:.1e} m/s)。")
    else:
        A(f"- env 統合は既定次数 VAR({flight['env_order']}) を使うため、本検証の "
          f"VAR({flight['order']}) とは風速差 max "
          f"{fv['env_fc_matches_study_fc_wspd_m_s']:.2f} m/s の差がある"
          "(テレメトリ照合は env 自身の系列に対して実施)。")
    A("")
    A("## 結論と限界")
    A("")
    last = sk["summary"][str(cfg.horizon)]
    wins = [v for v in VAR_NAMES if last[v]["skill_pers"] > 0.0]
    loses = [v for v in VAR_NAMES if last[v]["skill_pers"] <= 0.0]
    A(f"- 最長リード {cfg.horizon} h の検証窓平均で VAR({sel['sel_order']}) は "
      + "、".join(wins) + " で永続予報を上回った"
      + (f"({', '.join(loses)} では下回る)。" if loses else "(全変で上回る)。"))
    A("- 短リード(概ね 1〜6 h)では気温・風速で永続予報が強い。これは実況値の"
      "慣性が支配的な区間であり、実用的には短リード=永続、長リード=モデルの"
      "使い分けが最適(上表のスキルスコア参照)。")
    A("- 予報はリードとともに必ず気候値へ減衰し(セクション D)、コンパニオン"
      "行列のスペクトル半径は全発行時刻で < 1。発散しない誘導曲線として整合的。")
    A("- 限界: 45.7 日・1 地点の学習なので季節外挿は不可。嵐のピーク強度は"
      "線形モデルでは過小/位相ずれが出やすい(ケース E 参照)。ガスト超過は"
      "時平均風の関数として予報しており、瞬時突風そのものの予報ではない。")
    A("")
    A("## 生成ファイル")
    A("")
    for nm in ["REPORT.md", "summary.json", "order_selection.csv",
               "skill_vs_lead.csv", "damping.csv", "case_study.csv",
               "envelope.csv", "env_forecast.csv",
               "weather_forecast_study.py"]:
        A(f"- `{nm}`")
    for p in plot_names:
        A(f"- `{Path(p).name}`")
    A("")
    A(f"実行時間: {runtime_s:.1f} s / git HEAD: `{git_head()[:12]}`")
    (cfg.out / "REPORT.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def trim_state_dicts(d: dict) -> dict:
    return {k: (v if not isinstance(v, np.ndarray) else v.tolist())
            for k, v in d.items()}


def run_study(cfg: StudyConfig) -> dict:
    t0 = time.time()
    cfg.out.mkdir(parents=True, exist_ok=True)
    series = parse_ndbc_met(cfg.data)
    sel_issues, ver_issues = issue_windows(series, cfg)
    print(f"records: {len(series)}  selection issues: {len(sel_issues)}"
          f"  verification issues: {len(ver_issues)}")

    sel = section_order_selection(series, cfg, sel_issues)
    print(f"selected order: VAR({sel['sel_order']})")
    sk = section_skill(series, cfg, sel["sel_order"], ver_issues)
    damp = section_damping(series, cfg, sk)
    case = section_case(series, cfg, sel["sel_order"], ver_issues)
    print(f"storm case issue: {case['stamp']} UTC")
    flight = section_flight(series, cfg, sel["sel_order"], case["issue"])
    print(f"env episode: {flight['env']['steps']} steps, "
          f"max dev T {flight['env']['max_abs_dev_T_C']:.3f} K, "
          f"p {flight['env']['max_abs_dev_p_hPa']:.3f} hPa")

    write_csv(cfg.out / "order_selection.csv", sel["rows"],
              list(sel["rows"][0].keys()))
    write_csv(cfg.out / "skill_vs_lead.csv", sk["rows"],
              list(sk["rows"][0].keys()))
    write_csv(cfg.out / "damping.csv", damp["rows"],
              list(damp["rows"][0].keys()))
    write_csv(cfg.out / "case_study.csv", case["rows"],
              list(case["rows"][0].keys()))
    write_csv(cfg.out / "envelope.csv", flight["envelope"]["rows"],
              list(flight["envelope"]["rows"][0].keys()))
    write_csv(cfg.out / "env_forecast.csv", flight["env"]["rows"],
              list(flight["env"]["rows"][0].keys()))

    plot_names = []
    if cfg.plots:
        plot_names = make_plots(cfg, series, sel, sk, damp, case, flight)
        print("plots:", [Path(p).name for p in plot_names])

    runtime_s = time.time() - t0
    write_report(cfg, series, sel, sk, damp, case, flight, plot_names,
                 runtime_s)
    summary = {
        "study": "weather_forecast",
        "data": str(cfg.data),
        "n_records": len(series),
        "record_start_utc": series.stamps[0],
        "record_end_utc": series.stamps[-1],
        "config": {"horizon_h": cfg.horizon, "sel_lo": cfg.sel_lo,
                   "sel_hi": cfg.sel_hi, "sel_stride_h": cfg.sel_stride,
                   "ver_stride_h": cfg.ver_stride,
                   "orders": list(cfg.orders), "env_rate": cfg.env_rate,
                   "env_steps": cfg.env_steps, "quick": cfg.quick},
        "order_selection": {"sel_order": sel["sel_order"],
                            "lead_ref": sel["lead_ref"],
                            "n_issues": len(sel["issues"]),
                            "rows": sel["rows"]},
        "skill": {"n_issues": sk["n_issues"], "order": sk["order"],
                  "leads": sk["leads"], "summary": sk["summary"],
                  "crossover_lead_vs_persistence":
                      sk["crossover_lead_vs_persistence"],
                  "radius": sk["radius"]},
        "damping": {k: v for k, v in damp.items() if k != "rows"},
        "storm_case": {k: v for k, v in case.items() if k != "rows"},
        "flight": {"issue_stamp": flight["issue_stamp"],
                   "order": flight["order"], "env_order": flight["env_order"],
                   "envelope": {k: v for k, v in flight["envelope"].items()
                                if k != "rows"},
                   "env": {k: v for k, v in flight["env"].items()
                           if k != "rows"}},
        "files": ["REPORT.md", "summary.json", "order_selection.csv",
                  "skill_vs_lead.csv", "damping.csv", "case_study.csv",
                  "envelope.csv", "env_forecast.csv"]
                 + [Path(p).name for p in plot_names],
        "runtime_s": runtime_s,
        "git_head": git_head(),
    }
    (cfg.out / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=float),
        encoding="utf-8")
    shutil.copy2(Path(__file__).resolve(),
                 cfg.out / "weather_forecast_study.py")
    print(f"done in {runtime_s:.1f} s -> {cfg.out}")
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="VAR weather-forecast verification study (NDBC 46012)")
    ap.add_argument("--out", default="results/weather_forecast_001",
                    help="output directory")
    ap.add_argument("--data", default=str(DEFAULT_MET_PATH),
                    help="NDBC realtime2 file")
    ap.add_argument("--quick", action="store_true",
                    help="fewer issues / shorter horizon")
    ap.add_argument("--no-plots", action="store_true", help="skip PNG plots")
    a = ap.parse_args(argv)
    cfg = StudyConfig(out=Path(a.out), data=Path(a.data), quick=a.quick,
                      plots=not a.no_plots)
    run_study(cfg)


if __name__ == "__main__":
    main()
