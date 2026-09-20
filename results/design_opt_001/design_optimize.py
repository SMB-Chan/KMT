#!/usr/bin/env python3
"""Parametric airframe design optimization for the KMT flying boat.

A deterministic compass (pattern) search over 9 design variables
minimizes a weighted, constraint-penalized cost built from:

  * a water-run takeoff  -- own semi-implicit Euler loop with a hull
    model consistent with the airframe geometry
    (HullContact(A_wp = Bwl*Lwl) + HullDrag(Bwl, Lwl)).
    dynamics.simulate_takeoff hard-codes the default waterplane area;
    at the baseline design the two agree exactly (0.55*2.6 = Bwl*Lwl).
  * an approach/landing run -- peak water normal load and runout time
  * analytic cruise power, static thrust-to-weight, stall speed

The baseline design variables reproduce aircraft.Aircraft() exactly, so
the study cannot silently shift the default airframe; results are written
to a results/ study folder (Japanese REPORT.md, summary.json,
best_design.json, convergence.csv, sensitivity.csv, plots) following the
repository convention.

Usage:
    python3 design_optimize.py [--out results/design_opt_001] [--quick]
                               [--no-plots]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from aircraft import Aircraft, G, RHO, RHO_W
from atmosphere import isa_density
from dynamics import (HULL_HYSTERESIS_M, HullContact, HullDrag, hull_force)
from ocean import Ocean


# ---------------------------------------------------------------------
#  Design variables
# ---------------------------------------------------------------------
# name: (baseline, lo, hi, description-ja)
DESIGN_VARS: dict[str, tuple[float, float, float, str]] = {
    "S_scale":    (1.000, 0.850, 1.200, "翼弦長スケール(翼面積 b*c を変更)"),
    "b_scale":    (1.000, 0.900, 1.150, "翼幅スケール"),
    "margin_kg":  (0.000, -3.500, 6.000, "設計マージン質量の増減 [kg]"),
    "CL_max":     (1.400, 1.200, 1.700, "最大揚力係数(フラップ設計)"),
    "CD0":        (0.0250, 0.0200, 0.0300, "零揚力抗力係数(表面仕上げ)"),
    "D_prop":     (0.710, 0.610, 0.810, "プロペラ直径 [m]"),
    "prop_pitch": (0.300, 0.250, 0.400, "プロペラピッチ [m]"),
    "Bwl_scale":  (1.000, 0.850, 1.200, "船体水線幅スケール"),
    "Lwl_scale":  (1.000, 0.900, 1.300, "船体水線長スケール"),
}
VAR_ORDER = tuple(DESIGN_VARS)          # fixed search order -> deterministic
BASELINE_DV = {name: spec[0] for name, spec in DESIGN_VARS.items()}

# Mass-coupling exponents (documented heuristics; exact 1.0 at baseline)
WING_MASS_EXP = 1.2      # wing_structure ~ (S/S0)^1.2  (foam + spar + skin)
HULL_MASS_EXP_B = 0.5    # hull_structure ~ Lwl_scale * Bwl_scale^0.5
PROP_MASS_EXP = 2.0      # propeller ~ (D_prop/D0)^2    (carbon blade area)

# Propulsion electrical power is held at the baseline 12 kW install; the
# search re-matches the propeller (D, pitch) to that fixed power plant.


def apply_design(dv: dict) -> Aircraft:
    """Build an Aircraft from design variables.

    BASELINE_DV reproduces ``Aircraft()`` field-for-field (the scaling
    factors are exactly 1.0, so all products are bitwise unchanged).
    """
    base = Aircraft()
    g0, m0, p0 = base.geom, base.mass, base.prop
    b = g0.b * dv["b_scale"]
    c = g0.c * dv["S_scale"]
    S = b * c
    geom = replace(g0,
                   b=b, c=c, S=S, AR=b * b / S,
                   Bwl=g0.Bwl * dv["Bwl_scale"],
                   Lwl=g0.Lwl * dv["Lwl_scale"],
                   S_t=g0.S_t * dv["S_scale"],
                   float_y=g0.float_y * dv["b_scale"])
    mass = replace(m0,
                   wing_structure=m0.wing_structure * (S / g0.S) ** WING_MASS_EXP,
                   hull_structure=(m0.hull_structure * dv["Lwl_scale"]
                                   * dv["Bwl_scale"] ** HULL_MASS_EXP_B),
                   propeller=(m0.propeller
                              * (dv["D_prop"] / p0.D_prop) ** PROP_MASS_EXP),
                   margin=m0.margin + dv["margin_kg"])
    aero = replace(base.aero, CL_max=dv["CL_max"], CD0=dv["CD0"])
    prop = replace(p0, D_prop=dv["D_prop"], pitch=dv["prop_pitch"])
    return Aircraft(geom=geom, mass=mass, aero=aero, prop=prop)


def clip_dv(dv: dict) -> dict:
    """Clip design variables into their bounds."""
    return {name: min(max(float(dv[name]), spec[1]), spec[2])
            for name, spec in DESIGN_VARS.items()}


# ---------------------------------------------------------------------
#  Study configuration
# ---------------------------------------------------------------------
@dataclass
class StudyConfig:
    out: Path = Path("results/design_opt_001")
    # simulation resolution used inside the search
    takeoff_duration: float = 25.0
    takeoff_dt: float = 0.01
    landing_duration: float = 32.0   # baseline touchdown ~24.2 s + runout
    landing_dt: float = 0.002
    # high-resolution re-runs for the final report
    final_landing_duration: float = 35.0
    final_landing_dt: float = 0.001
    # pattern-search budget
    initial_step_frac: float = 0.25     # fraction of each variable span
    min_step_frac: float = 1.0 / 64.0   # stop when steps shrink below this
    shrink: float = 0.5
    max_rounds: int = 40
    # design sea state
    Hs: float = 1.5
    Tp: float = 6.0
    seed: int = 42
    plots: bool = True
    quick: bool = False

    def __post_init__(self):
        self.out = Path(self.out)
        if self.quick:
            self.takeoff_duration = 20.0
            self.takeoff_dt = 0.02
            self.landing_dt = 0.004
            self.final_landing_duration = 30.0
            self.final_landing_dt = 0.002
            self.initial_step_frac = 0.5
            self.min_step_frac = 1.0 / 8.0
            self.max_rounds = 12


# ---------------------------------------------------------------------
#  Objective and constraints
# ---------------------------------------------------------------------
WEIGHTS = {"t_liftoff": 0.30, "x_liftoff": 0.15, "R_hump": 0.20,
           "P_cruise": 0.25, "N_land_peak": 0.10}
LIMITS = {"TW_static_min": 0.55, "V_stall_max": 13.0,
          "N_land_peak_max": 3.0, "mass_max": 95.0}
PENALTY = 10.0             # per unit squared normalized violation
NO_LIFTOFF_PENALTY = 10.0  # flat, when the run never leaves the water
LIFTOFF_HOLD_S = 0.5       # airborne this long => sustained liftoff

def make_eta(sea: Ocean):
    """Scalar eta(x, t) closure, bitwise equal to sea.eta([x], t)[0].

    Same value/order of floating-point operations as Ocean.eta for a
    single query point, without the per-call outer-product overhead.
    """
    amps, k, om, ph = sea.amps, sea.k, sea.omega, sea.phases

    def eta(x: float, t: float) -> float:
        return float((amps * np.cos(k * x - om * t + ph)).sum())

    return eta


def _integrate_run(ac: Aircraft, eta, *, x0: float, z0: float,
                   Vx0: float, Vz0: float, alpha: float, throttle: float,
                   duration: float, dt: float):
    """Longitudinal water-run integrator mirroring dynamics.simulate_*.

    Force assembly, hull contact and the semi-implicit Euler update are
    copied term-for-term from dynamics.py so that the baseline design
    reproduces simulate_takeoff / simulate_landing bitwise; only the hull
    geometry is made design-consistent (A_wp = Bwl * Lwl).
    """
    hull = HullContact(A_wp=ac.geom.Bwl * ac.geom.Lwl)
    hd = HullDrag(Bwl=ac.geom.Bwl, Lwl=ac.geom.Lwl)
    n = int(duration / dt) + 1
    t = np.linspace(0.0, duration, n)
    x = np.zeros(n); z = np.zeros(n)
    Vx = np.zeros(n); Vz = np.zeros(n)
    Rh = np.zeros(n); Nw = np.zeros(n); ph = np.zeros(n, dtype=int)
    x[0], z[0], Vx[0], Vz[0] = x0, z0, Vx0, Vz0
    m = ac.mass.total
    cos_a = math.cos(alpha + ac.aero.alpha_T)
    sin_a = math.sin(alpha + ac.aero.alpha_T)
    for i in range(n - 1):
        e = eta(x[i], t[i])
        V = math.hypot(Vx[i], Vz[i])
        rho = isa_density(z[i])
        T_i = ac.prop.thrust(V, throttle, rho=rho)
        q = 0.5 * rho * V ** 2
        gamma = math.atan2(Vz[i], Vx[i])
        alpha_eff = alpha - gamma
        CL = ac.CL(alpha_eff)
        CD = ac.CD(CL, height_m=z[i] - e)
        L_i = q * ac.geom.S * CL
        D_i = q * ac.geom.S * CD
        wf = hull_force(z[i], Vx[i], Vz[i], e, hull)
        R_hull = hd.resistance(Vx[i]) if wf.N > 0 else 0.0
        cos_g, sin_g = math.cos(gamma), math.sin(gamma)
        T_x = T_i * cos_a
        T_z = T_i * sin_a
        L_x = -L_i * sin_g
        L_z = +L_i * cos_g
        D_x = -D_i * cos_g
        D_z = -D_i * sin_g
        Fx = T_x + L_x + D_x + wf.Rt - math.copysign(R_hull, Vx[i])
        Fz = T_z + L_z + D_z + wf.N - ac.W
        a_x = Fx / m
        a_z = Fz / m
        Vx[i + 1] = Vx[i] + a_x * dt
        Vz[i + 1] = Vz[i] + a_z * dt
        x[i + 1] = x[i] + Vx[i + 1] * dt
        z[i + 1] = z[i] + Vz[i + 1] * dt
        Rh[i + 1] = R_hull
        Nw[i + 1] = wf.N
        ph[i + 1] = 1 if z[i + 1] - hull.h_keel > e + HULL_HYSTERESIS_M else 0
    finite = bool(np.isfinite(x).all() and np.isfinite(z).all()
                  and np.isfinite(Vx).all() and np.isfinite(Vz).all())
    return {"t": t, "x": x, "z": z, "Vx": Vx, "Vz": Vz,
            "R_hull": Rh, "N_water": Nw, "phase": ph,
            "h_keel": hull.h_keel, "finite": finite}


def run_takeoff(ac: Aircraft, sea: Ocean, cfg: StudyConfig) -> dict:
    """Full-throttle water run at alpha = 4 deg; extract liftoff metrics."""
    eta = make_eta(sea)
    hull_A = ac.geom.Bwl * ac.geom.Lwl
    delta_static = ac.W / (RHO_W * G * hull_A)
    z0 = eta(0.0, 0.0) + HullContact.h_keel - delta_static
    run = _integrate_run(ac, eta, x0=0.0, z0=z0, Vx0=0.5, Vz0=0.0,
                         alpha=math.radians(4.0), throttle=1.0,
                         duration=cfg.takeoff_duration, dt=cfg.takeoff_dt)
    dt = cfg.takeoff_dt
    hold = max(1, int(round(LIFTOFF_HOLD_S / dt)))
    ph = run["phase"]
    lift_i = -1
    for j in range(0, len(ph) - hold + 1):
        if ph[j] == 1 and ph[j:j + hold].min() == 1:
            lift_i = j
            break
    liftoff = lift_i >= 0 and run["finite"]
    water = ph == 0
    return {**run,
            "liftoff": liftoff,
            "t_liftoff": float(run["t"][lift_i]) if liftoff else cfg.takeoff_duration,
            "x_liftoff": float(run["x"][lift_i]) if liftoff else float(run["x"][-1]),
            "Vx_liftoff": float(run["Vx"][lift_i]) if liftoff else float(run["Vx"][-1]),
            "R_hump": float(run["R_hull"][water].max()) if water.any() else 0.0,
            "Vx_max": float(run["Vx"].max()) if run["finite"] else float("inf")}


def run_landing(ac: Aircraft, sea: Ocean, cfg: StudyConfig,
                duration: float | None = None, dt: float | None = None) -> dict:
    """8-deg glide at 13 m/s, idle-ish throttle; touchdown and runout."""
    duration = cfg.landing_duration if duration is None else duration
    dt = cfg.landing_dt if dt is None else dt
    eta = make_eta(sea)
    glide = math.radians(8.0)
    run = _integrate_run(ac, eta, x0=0.0, z0=30.0,
                         Vx0=13.0 * math.cos(glide),
                         Vz0=-13.0 * math.sin(glide),
                         alpha=math.radians(-5.0), throttle=0.05,
                         duration=duration, dt=dt)
    Nw, Vx, Vz, t = run["N_water"], run["Vx"], run["Vz"], run["t"]
    contact = np.where(Nw > 0.0)[0]
    touchdown = len(contact) > 0 and run["finite"]
    if touchdown:
        td = int(contact[0])
        slow = np.where(Vx[td:] < 1.0)[0]
        runout = float(t[td + int(slow[0])] - t[td]) if len(slow) else float(t[-1] - t[td])
        Vz_td = float(abs(Vz[td]))
        t_td = float(t[td])
    else:
        runout, Vz_td, t_td = duration, float("nan"), float("nan")
    return {**run,
            "touchdown": touchdown,
            "t_touchdown": t_td, "Vz_touchdown": Vz_td, "runout": runout,
            "N_peak_W": float(Nw.max() / ac.W) if run["finite"] else float("inf")}


def cruise_power(ac: Aircraft) -> float:
    """Electrical cruise power at the design point (V_stall * sqrt(3))."""
    V = ac.V_cruise
    q = 0.5 * RHO * V ** 2
    CL = ac.W / (q * ac.geom.S)
    CD = ac.CD(CL)
    eta = ac.prop.eta_motor * ac.prop.eta_esc * ac.prop.eta_prop
    return q * ac.geom.S * CD * V / eta

# ---------------------------------------------------------------------
#  Evaluation
# ---------------------------------------------------------------------
@dataclass
class EvalResult:
    dv: dict
    metrics: dict
    violations: dict
    cost: float


class Evaluator:
    """Deterministic design evaluator.

    The first evaluation (the baseline) fixes the cost normalisation, so
    J(baseline) = sum(WEIGHTS.values()) = 1.0 exactly when feasible.
    """

    def __init__(self, cfg: StudyConfig):
        self.cfg = cfg
        self.sea = Ocean(Hs=cfg.Hs, Tp=cfg.Tp, seed=cfg.seed)
        self.n_evals = 0
        self.ref: dict | None = None

    def metrics(self, dv: dict, sea: Ocean | None = None) -> dict:
        sea = self.sea if sea is None else sea
        ac = apply_design(dv)
        tk = run_takeoff(ac, sea, self.cfg)
        ld = run_landing(ac, sea, self.cfg)
        return {
            "dv": dict(dv),
            "ac": ac,
            "takeoff": tk,
            "landing": ld,
            "t_liftoff": tk["t_liftoff"],
            "x_liftoff": tk["x_liftoff"],
            "Vx_liftoff": tk["Vx_liftoff"],
            "R_hump": tk["R_hump"],
            "liftoff": tk["liftoff"],
            "diverged": not (tk["finite"] and ld["finite"]),
            "touchdown": ld["touchdown"],
            "N_land_peak": ld["N_peak_W"],
            "runout_s": ld["runout"],
            "Vz_touchdown": ld["Vz_touchdown"],
            "P_cruise": cruise_power(ac),
            "TW_static": ac.prop.T_static / ac.W,
            "V_stall": ac.V_stall,
            "V_cruise": ac.V_cruise,
            "mass": ac.mass.total,
            "S": ac.geom.S,
            "AR": ac.geom.AR,
        }

    def violations(self, m: dict) -> dict:
        ac = m["ac"]
        v = {
            "TW_static": max(0.0, (LIMITS["TW_static_min"] - m["TW_static"])
                             / LIMITS["TW_static_min"]),
            "V_stall": max(0.0, (m["V_stall"] - LIMITS["V_stall_max"])
                           / LIMITS["V_stall_max"]),
            "N_land_peak": max(0.0, (m["N_land_peak"] - LIMITS["N_land_peak_max"])
                               / LIMITS["N_land_peak_max"]),
            "P_cruise": max(0.0, (m["P_cruise"] - ac.prop.P_max) / ac.prop.P_max),
            "mass": max(0.0, (m["mass"] - LIMITS["mass_max"]) / LIMITS["mass_max"]),
        }
        if not m["touchdown"]:
            v["no_touchdown"] = 1.0
        if m["diverged"]:
            v["diverged"] = 1.0
        return {k: val for k, val in v.items() if val > 0.0}

    def cost(self, m: dict) -> float:
        if self.ref is None:
            self.ref = {k: max(abs(float(m[k])), 1e-9) for k in WEIGHTS}
        c = sum(w * float(m[k]) / self.ref[k] for k, w in WEIGHTS.items())
        if not m["liftoff"] or m["diverged"]:
            c += NO_LIFTOFF_PENALTY
        c += PENALTY * sum(val * val for val in self.violations(m).values())
        return c

    def evaluate(self, dv: dict) -> EvalResult:
        self.n_evals += 1
        m = self.metrics(dv)
        return EvalResult(dv=dict(dv), metrics=m,
                          violations=self.violations(m), cost=self.cost(m))


# ---------------------------------------------------------------------
#  Deterministic compass (pattern) search
# ---------------------------------------------------------------------
def pattern_search(ev: Evaluator, cfg: StudyConfig):
    """Coordinate compass search: +-step per variable in fixed order,
    first improvement accepted, all steps shrink when a full sweep fails.
    No RNG anywhere -> fully reproducible."""
    dv = dict(BASELINE_DV)
    spans = {n: DESIGN_VARS[n][2] - DESIGN_VARS[n][1] for n in VAR_ORDER}
    steps = {n: spans[n] * cfg.initial_step_frac for n in VAR_ORDER}
    min_steps = {n: spans[n] * cfg.min_step_frac for n in VAR_ORDER}
    best = ev.evaluate(dv)
    history = [{"eval": ev.n_evals, "round": 0, "move": "baseline",
                "cost": best.cost}]
    rounds = 0
    while (rounds < cfg.max_rounds
           and any(steps[n] > min_steps[n] for n in VAR_ORDER)):
        improved = False
        rounds += 1
        for name in VAR_ORDER:
            lo, hi = DESIGN_VARS[name][1], DESIGN_VARS[name][2]
            for sgn in (+1.0, -1.0):
                v = min(max(dv[name] + sgn * steps[name], lo), hi)
                if v == dv[name]:
                    continue
                trial = dict(dv)
                trial[name] = v
                r = ev.evaluate(trial)
                history.append({"eval": ev.n_evals, "round": rounds,
                                "move": f"{name}{'+' if sgn > 0 else '-'}",
                                "cost": r.cost})
                if r.cost < best.cost - 1e-12:
                    dv, best, improved = trial, r, True
                    break
        if not improved:
            steps = {n: s * cfg.shrink for n, s in steps.items()}
    return dv, best, history, rounds


def sensitivity_scan(ev: Evaluator, center: dict) -> list[dict]:
    """Cost at each variable's lo/hi bound, all others at ``center``."""
    j_center = ev.evaluate(center).cost
    rows = []
    for name in VAR_ORDER:
        row = {"var": name, "J_center": j_center}
        for tag, idx in (("lo", 1), ("hi", 2)):
            trial = dict(center)
            trial[name] = DESIGN_VARS[name][idx]
            r = ev.evaluate(trial)
            row[f"J_{tag}"] = r.cost
            row[f"t_liftoff_{tag}"] = r.metrics["t_liftoff"]
            row[f"R_hump_{tag}"] = r.metrics["R_hump"]
            row[f"P_cruise_{tag}"] = r.metrics["P_cruise"]
        row["J_range"] = abs(row["J_hi"] - row["J_lo"])
        rows.append(row)
    return rows


def robustness_scan(ev: Evaluator, cfg: StudyConfig, best_dv: dict) -> dict:
    """Re-run baseline and best on calm / rough seas (metrics only)."""
    out = {}
    for label, hs in (("calm_Hs0.3", 0.3), ("rough_Hs2.5", 2.5)):
        sea = Ocean(Hs=hs, Tp=cfg.Tp, seed=cfg.seed)
        out[label] = {}
        for tag, dv in (("baseline", BASELINE_DV), ("best", best_dv)):
            m = ev.metrics(dv, sea=sea)
            out[label][tag] = {
                "t_liftoff": m["t_liftoff"], "R_hump": m["R_hump"],
                "N_land_peak": m["N_land_peak"], "runout_s": m["runout_s"],
                "liftoff": m["liftoff"], "touchdown": m["touchdown"],
            }
    return out

def final_runs(cfg: StudyConfig, sea: Ocean, best_dv: dict) -> dict:
    """High-resolution landing re-runs for baseline and best design."""
    out = {}
    for tag, dv in (("baseline", BASELINE_DV), ("best", best_dv)):
        ac = apply_design(dv)
        ld = run_landing(ac, sea, cfg,
                         duration=cfg.final_landing_duration,
                         dt=cfg.final_landing_dt)
        out[tag] = {"N_land_peak": ld["N_peak_W"], "runout_s": ld["runout"],
                    "Vz_touchdown": ld["Vz_touchdown"],
                    "t_touchdown": ld["t_touchdown"],
                    "touchdown": ld["touchdown"]}
    return out


METRIC_KEYS = ("t_liftoff", "x_liftoff", "Vx_liftoff", "R_hump", "liftoff",
               "touchdown", "N_land_peak", "runout_s", "Vz_touchdown",
               "P_cruise", "TW_static", "V_stall", "V_cruise", "mass",
               "S", "AR")


def metrics_json(m: dict) -> dict:
    """JSON-safe subset of an evaluation's metrics."""
    out = {}
    for k in METRIC_KEYS:
        v = m[k]
        out[k] = bool(v) if isinstance(v, (bool, np.bool_)) else float(v)
    return out


def git_head() -> str:
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"],
                           capture_output=True, text=True, timeout=10)
        return r.stdout.strip() if r.returncode == 0 else ""
    except Exception:
        return ""


def make_plots(cfg: StudyConfig, tk_base: dict, tk_best: dict,
               sens_rows: list[dict]) -> list[str]:
    """Takeoff profile comparison + sensitivity bars. Returns file names."""
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
    fig, axes = plt.subplots(2, 1, figsize=(9.0, 7.0), sharex=True)
    for ax, key, ylab in ((axes[0], "Vx", "Vx [m/s]"),
                          (axes[1], "R_hull", "船体抵抗 [N]")):
        for run, label, style in ((tk_base, "ベースライン", "--"),
                                  (tk_best, "最適化", "-")):
            ax.plot(run["t"], run[key], style, label=label, linewidth=1.4)
            if run["liftoff"]:
                ax.axvline(run["t_liftoff"], color="gray", linewidth=0.7,
                           alpha=0.6)
        ax.set_ylabel(ylab)
        ax.grid(alpha=0.3)
    axes[0].legend()
    axes[0].set_title(f"離水滑走比較 (Hs={cfg.Hs} m, Tp={cfg.Tp} s, 全開 α=4°)")
    axes[1].set_xlabel("t [s]")
    fig.tight_layout()
    p = cfg.out / "takeoff_profiles.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    names.append(p.name)

    fig, ax = plt.subplots(figsize=(9.0, 4.5))
    labels = [r["var"] for r in sens_rows]
    xs = np.arange(len(labels))
    j0 = sens_rows[0]["J_center"]
    ax.bar(xs - 0.2, [r["J_lo"] - j0 for r in sens_rows], 0.4, label="下限")
    ax.bar(xs + 0.2, [r["J_hi"] - j0 for r in sens_rows], 0.4, label="上限")
    ax.set_xticks(xs, labels, rotation=30, ha="right")
    ax.set_ylabel("ΔJ (ベースライン比)")
    ax.set_title("設計感度: 各変数を単独で限界値まで動かしたときの目的関数変化")
    ax.grid(alpha=0.3, axis="y")
    ax.legend()
    fig.tight_layout()
    p = cfg.out / "sensitivity.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    names.append(p.name)
    return names

def _fmt(v, nd=2):
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        if math.isnan(v):
            return "nan"
        if not math.isfinite(v):
            return "inf"
    return f"{v:.{nd}f}"


def write_outputs(cfg: StudyConfig, ev: Evaluator, base_r: EvalResult,
                  best_r: EvalResult, best_dv: dict, history: list,
                  rounds: int, sens_rows: list, robust: dict, final: dict,
                  plots: list[str], runtime_s: float) -> dict:
    cfg.out.mkdir(parents=True, exist_ok=True)
    bm, bsm = base_r.metrics, best_r.metrics
    imp = (base_r.cost - best_r.cost) / base_r.cost * 100.0

    # -- convergence.csv ---------------------------------------------
    with open(cfg.out / "convergence.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["eval", "round", "move", "cost"])
        for h in history:
            w.writerow([h["eval"], h["round"], h["move"], f"{h['cost']:.6f}"])

    # -- sensitivity.csv ----------------------------------------------
    with open(cfg.out / "sensitivity.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["var", "lo", "hi", "J_center", "J_lo", "J_hi", "J_range",
                    "t_liftoff_lo", "t_liftoff_hi", "R_hump_lo", "R_hump_hi",
                    "P_cruise_lo", "P_cruise_hi"])
        for r in sens_rows:
            lo, hi = DESIGN_VARS[r["var"]][1], DESIGN_VARS[r["var"]][2]
            w.writerow([r["var"], lo, hi,
                        f"{r['J_center']:.6f}", f"{r['J_lo']:.6f}",
                        f"{r['J_hi']:.6f}", f"{r['J_range']:.6f}",
                        f"{r['t_liftoff_lo']:.3f}", f"{r['t_liftoff_hi']:.3f}",
                        f"{r['R_hump_lo']:.1f}", f"{r['R_hump_hi']:.1f}",
                        f"{r['P_cruise_lo']:.1f}", f"{r['P_cruise_hi']:.1f}"])

    # -- best_design.json ----------------------------------------------
    best_design = {
        "design_variables": best_dv,
        "bounds": {n: [DESIGN_VARS[n][1], DESIGN_VARS[n][2]] for n in VAR_ORDER},
        "descriptions": {n: DESIGN_VARS[n][3] for n in VAR_ORDER},
        "weights": WEIGHTS,
        "limits": LIMITS,
        "J_baseline": base_r.cost,
        "J_best": best_r.cost,
        "violations_best": best_r.violations,
        "metrics": {"baseline": metrics_json(bm), "best": metrics_json(bsm)},
    }
    (cfg.out / "best_design.json").write_text(
        json.dumps(best_design, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")

    # -- summary.json (repo convention) ---------------------------------
    top_sens = sorted(sens_rows, key=lambda r: -r["J_range"])[:3]
    summary = {
        "study": "航空機設計のパラメトリック最適化 (決定論的コンパス探索, 9設計変数)",
        "train": {
            "search": "compass/pattern (first-improvement, deterministic)",
            "evals": ev.n_evals,
            "rounds": rounds,
            "sea": {"Hs": cfg.Hs, "Tp": cfg.Tp, "seed": cfg.seed,
                    "spectrum": "PM (JONSWAP gamma=1)"},
            "weights": WEIGHTS,
            "limits": LIMITS,
            "quick": cfg.quick,
        },
        "eval": {
            "J_baseline": base_r.cost,
            "J_best": best_r.cost,
            "improvement_pct": imp,
            "metrics_baseline": metrics_json(bm),
            "metrics_best": metrics_json(bsm),
            "final_landing_dt": cfg.final_landing_dt,
            "final_landing": final,
            "robustness": robust,
            "sensitivity_top": [{"var": r["var"], "J_range": r["J_range"]}
                                for r in top_sens],
        },
        "finding": (f"J を {imp:.1f}% 改善 ({base_r.cost:.3f} -> {best_r.cost:.3f}, "
                    f"評価 {ev.n_evals} 回 / {rounds} ラウンド)。"
                    f"感度上位は {', '.join(r['var'] for r in top_sens)}。"),
        "artifacts": [f"results/{cfg.out.name}/{n}" for n in
                      ["REPORT.md", "summary.json", "best_design.json",
                       "convergence.csv", "sensitivity.csv",
                       "design_optimize.py", *plots]],
        "caveats": [
            "縦動のみ (横方向・飛行制御・構造強度は目的関数外)",
            "尾翼/フロートは幾何スケールのみで縦力に寄与しない",
            "質量結合はヒューリスティック (翼~S^1.2, 船体~Lwl*Bwl^0.5, プロペラ~D^2)",
            "動力は 12 kW 固定でプロペラ整合のみを探索",
            "最適化海象は単一シード (seed=42)。Hs=0.3/2.5 で頑健性を別途確認",
        ],
        "git_head": git_head(),
        "runtime_s": round(runtime_s, 1),
    }
    (cfg.out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    return summary


def write_report(cfg: StudyConfig, ev: Evaluator, base_r: EvalResult,
                 best_r: EvalResult, best_dv: dict, sens_rows: list,
                 robust: dict, final: dict, rounds: int, runtime_s: float):
    bm, bsm = base_r.metrics, best_r.metrics
    imp = (base_r.cost - best_r.cost) / base_r.cost * 100.0
    L = []
    L.append("# 航空機設計最適化スタディ (design_opt)")
    L.append("")
    L.append("## 目的")
    L.append("離水性能・巡航電力・着水荷重を同時に改善する機体パラメータを、")
    L.append("デフォルト設計 (`aircraft.Aircraft()`) を変更せずに探索する。")
    L.append("")
    L.append("## 手法")
    L.append(f"- 探索: 決定論的コンパス探索 (±{cfg.initial_step_frac:g} スパン開始、"
             f"改善なしで ×{cfg.shrink:g}、下限 {cfg.min_step_frac:.4g} スパン)。RNG 不使用。")
    L.append(f"- 目的関数 J = Σ wᵢ·(指標ᵢ/ベースライン値) + ペナルティ。重み: "
             + ", ".join(f"{k}={v:g}" for k, v in WEIGHTS.items()))
    L.append("- 制約 (正規化違反量の二乗ペナルティ): "
             + ", ".join(f"{k}={v:g}" for k, v in LIMITS.items())
             + ", P_cruise ≤ 12 kW, 離水/着水の成立")
    L.append(f"- 評価海象: Hs={cfg.Hs} m, Tp={cfg.Tp} s, PM スペクトル, seed={cfg.seed}")
    L.append(f"- 離水滑走: 全開 α=4°, dt={cfg.takeoff_dt} s, {cfg.takeoff_duration:g} s。"
             "水線面面積を Bwl×Lwl とした機体幾何整合の船体モデル"
             " (ベースラインでは dynamics.simulate_takeoff と完全一致)。")
    L.append(f"- 着水: 進入高度 30 m / 進入速度 13 m/s / 滑空角 8° / α=-5° / "
             f"スロットル 0.05, dt={cfg.landing_dt} s。"
             f"最終数値のみ dt={cfg.final_landing_dt} s で再計算。")
    L.append("- 質量結合: 翼 (S/S0)^1.2, 船体 Lwl×Bwl^0.5, プロペラ (D/D0)², "
             "マージン ±kg。")
    L.append("")
    L.append("## 結果")
    L.append("")
    L.append("| 項目 | ベースライン | 最適化 |")
    L.append("|---|---|---|")
    rows = [
        ("目的関数 J", base_r.cost, best_r.cost),
        ("質量 [kg]", bm["mass"], bsm["mass"]),
        ("翼面積 S [m²]", bm["S"], bsm["S"]),
        ("アスペクト比 AR", bm["AR"], bsm["AR"]),
        ("失速速度 [m/s]", bm["V_stall"], bsm["V_stall"]),
        ("巡航速度 [m/s]", bm["V_cruise"], bsm["V_cruise"]),
        ("静止推力比 T/W", bm["TW_static"], bsm["TW_static"]),
        ("離水時間 [s]", bm["t_liftoff"], bsm["t_liftoff"]),
        ("離水距離 [m]", bm["x_liftoff"], bsm["x_liftoff"]),
        ("離水速度 [m/s]", bm["Vx_liftoff"], bsm["Vx_liftoff"]),
        ("ハンプ抵抗ピーク [N]", bm["R_hump"], bsm["R_hump"]),
        ("巡航電力 [W]", bm["P_cruise"], bsm["P_cruise"]),
        ("着水ピーク荷重 [g]", bm["N_land_peak"], bsm["N_land_peak"]),
        ("着水後停止滑走時間 [s]", bm["runout_s"], bsm["runout_s"]),
    ]
    for name, a, b in rows:
        L.append(f"| {name} | {_fmt(a, 3 if name == '目的関数 J' else 2)} "
                 f"| {_fmt(b, 3 if name == '目的関数 J' else 2)} |")
    L.append("")
    L.append(f"目的関数は **{imp:.1f}% 改善** "
             f"(評価 {ev.n_evals} 回 / {rounds} ラウンド, {runtime_s:.0f} s)。")
    L.append("")
    L.append("### 最適設計変数")
    L.append("")
    L.append("| 変数 | 説明 | 下限 | ベースライン | 上限 | 最適値 |")
    L.append("|---|---|---|---|---|---|")
    for name in VAR_ORDER:
        base_v, lo, hi, desc = DESIGN_VARS[name]
        L.append(f"| `{name}` | {desc} | {lo:g} | {base_v:g} | {hi:g} "
                 f"| {best_dv[name]:.4g} |")
    L.append("")
    L.append("### 感度 (単変数を限界まで動かした ΔJ)")
    L.append("")
    L.append("| 変数 | J(下限) | J(上限) | |ΔJ| |")
    L.append("|---|---|---|---|")
    for r in sorted(sens_rows, key=lambda r: -r["J_range"]):
        L.append(f"| `{r['var']}` | {r['J_lo']:.3f} | {r['J_hi']:.3f} "
                 f"| {r['J_range']:.3f} |")
    L.append("")
    L.append("### 頑健性 (別海象での再評価)")
    L.append("")
    L.append("| 海象 | 設計 | 離水 [s] | ハンプ抵抗 [N] | 着水ピーク [g] |")
    L.append("|---|---|---|---|---|")
    for label, pairs in robust.items():
        for tag in ("baseline", "best"):
            m = pairs[tag]
            L.append(f"| {label} | {tag} | {_fmt(m['t_liftoff'])} "
                     f"| {_fmt(m['R_hump'], 0)} | {_fmt(m['N_land_peak'])} |")
    L.append("")
    L.append(f"### 高分解能着水再評価 (dt={cfg.final_landing_dt} s)")
    L.append("")
    L.append("| 設計 | 着水 Vz [m/s] | ピーク荷重 [g] | 停止滑走時間 [s] |")
    L.append("|---|---|---|---|")
    for tag in ("baseline", "best"):
        m = final[tag]
        L.append(f"| {tag} | {_fmt(m['Vz_touchdown'])} "
                 f"| {_fmt(m['N_land_peak'])} | {_fmt(m['runout_s'])} |")
    L.append("")
    L.append("## 考察")
    top = sorted(sens_rows, key=lambda r: -r["J_range"])[:3]
    L.append(f"- 感度が最も大きい設計変数は "
             + ", ".join(f"`{r['var']}` (|ΔJ|={r['J_range']:.2f})" for r in top)
             + "。目的関数は推力整合 (プロペラ) と重量・抗力に支配される。")
    moved = [n for n in VAR_ORDER
             if abs(best_dv[n] - DESIGN_VARS[n][0])
             > 0.2 * (DESIGN_VARS[n][2] - DESIGN_VARS[n][1])]
    L.append("- ベースラインから大きく動いた変数: "
             + (", ".join(f"`{n}`" for n in moved) if moved else "なし (ベースラインが既に良好)"))
    active = best_r.violations
    L.append("- 最適設計で活動中の制約違反: "
             + (", ".join(f"{k}={v:.3f}" for k, v in active.items())
                if active else "なし"))
    L.append("")
    L.append("## 留意点")
    L.append("- 縦動のみを評価。横動揺・飛行制御・構造強度は目的関数に含まれない。")
    L.append("- 尾翼とフロートの幾何はスケールのみで縦力に寄与しない (cosmetic)。")
    L.append("- 質量結合はヒューリスティック。実設計では構造見積りで置換すること。")
    L.append("- 推進系は 12 kW 固定。モータ/バッテリの再選定は本スタディの範囲外。")
    L.append("- `dynamics.simulate_takeoff/landing` は既定水線面面積をハードコードしており、")
    L.append("  本スクリプトは Bwl×Lwl を使う (ベースラインでは両者が完全一致することをテストで担保)。")
    L.append("")
    L.append("## 成果物")
    L.append("- `summary.json` / `best_design.json` / `convergence.csv` / `sensitivity.csv`")
    if cfg.plots:
        L.append("- `takeoff_profiles.png` (離水滑走比較) / `sensitivity.png` (感度)")
    L.append("- `design_optimize.py` (このスクリプトの自己完結コピー)")
    L.append("")
    (cfg.out / "REPORT.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def run_study(cfg: StudyConfig) -> dict:
    t0 = time.time()
    cfg.out.mkdir(parents=True, exist_ok=True)
    ev = Evaluator(cfg)
    base_r = ev.evaluate(BASELINE_DV)          # fixes cost normalisation
    best_dv, best_r, history, rounds = pattern_search(ev, cfg)
    sens_rows = sensitivity_scan(ev, BASELINE_DV)
    robust = robustness_scan(ev, cfg, best_dv)
    final = final_runs(cfg, ev.sea, best_dv)
    plots = (make_plots(cfg, base_r.metrics["takeoff"],
                        best_r.metrics["takeoff"], sens_rows)
             if cfg.plots else [])
    runtime_s = time.time() - t0
    summary = write_outputs(cfg, ev, base_r, best_r, best_dv, history,
                            rounds, sens_rows, robust, final, plots,
                            runtime_s)
    write_report(cfg, ev, base_r, best_r, best_dv, sens_rows, robust,
                 final, rounds, runtime_s)
    shutil.copy2(Path(__file__).resolve(), cfg.out / "design_optimize.py")
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Deterministic parametric design optimization study.")
    ap.add_argument("--out", default="results/design_opt_001",
                    help="study output folder (default: %(default)s)")
    ap.add_argument("--quick", action="store_true",
                    help="coarse dt / short runs / small search budget")
    ap.add_argument("--no-plots", action="store_true", help="skip PNG output")
    args = ap.parse_args(argv)
    cfg = StudyConfig(out=Path(args.out), quick=args.quick,
                      plots=not args.no_plots)
    summary = run_study(cfg)
    ev = summary["eval"]
    print(f"J: {ev['J_baseline']:.4f} -> {ev['J_best']:.4f} "
          f"({ev['improvement_pct']:+.1f}%)")
    print(f"t_liftoff {ev['metrics_baseline']['t_liftoff']:.2f} s -> "
          f"{ev['metrics_best']['t_liftoff']:.2f} s | "
          f"P_cruise {ev['metrics_baseline']['P_cruise']:.0f} W -> "
          f"{ev['metrics_best']['P_cruise']:.0f} W")
    print(f"report: {cfg.out}/REPORT.md")


if __name__ == "__main__":
    main()

