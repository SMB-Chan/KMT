#!/usr/bin/env python3
"""Virtual wind-tunnel testing of the KMT flying boat, driven by atmosphere.py.

The tunnel reproduces the project's atmosphere model in the test section:

  * ISA density / temperature / pressure at the requested altitude
    (atmosphere.isa_*), converted to Reynolds and Mach numbers with a
    Sutherland viscosity;
  * the log-shear mean-wind profile (Atmosphere.shear_factor);
  * seeded, reproducible turbulence fields (AtmosphereConfig gust_model
    "sum4" / "dryden") fed through Atmosphere.wind(t, altitude).

Experiments (all measured from sweeps of the airframe model, never from
the constants that generated them):

  1. alpha polar  -- CL/CD/L-D sweeps, recovery of the 3-D lift slope,
     stall angle, CL_max, CD0, induced-drag factor and (L/D)_max;
  2. altitude series -- test-section conditions and stall/cruise speeds
     at 0/50/150/500 m;
  3. level-flight velocity sweep -- trim CL, drag, shaft power, L/D and
     installed thrust vs speed, with min-power / min-drag points;
  4. ground-effect sweep -- induced-drag relief vs height/span;
  5. unsteady gust tunnel -- effective angle of attack and load factor
     time series through Atmosphere.wind, for both gust models, several
     turbulence levels and a steady shear wind case.

A deterministic compass search over the five aerodynamically active
design variables (S_scale, b_scale, margin_kg, CL_max, CD0 -- bounds and
mass coupling identical to design_optimize.py) then minimizes a weighted
tunnel objective: cruise shaft power, 1/(L/D)_max and gust-load RMS,
with stall-speed / mass limits as quadratic penalties.  The default
airframe constants are never modified; the baseline design variables
reproduce aircraft.Aircraft() exactly.

Usage:
    python3 wind_tunnel.py [--out results/wind_tunnel_001] [--quick]
                           [--no-plots]
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
from atmosphere import ISA_R, Atmosphere, AtmosphereConfig
from design_optimize import (BASELINE_DV, DESIGN_VARS, VAR_ORDER,
                             apply_design, clip_dv, git_head, _fmt)

# ---------------------------------------------------------------------
#  Tunnel constants
# ---------------------------------------------------------------------
GAMMA_AIR = 1.4           # ratio of specific heats (Mach number)
MU_SUTH_A = 1.458e-6      # Sutherland viscosity, kg/(m s K^0.5)
MU_SUTH_S = 110.4         # Sutherland temperature, K

# Design variables the tunnel objective can see (aerodynamics + mass).
# Propeller / hull variables are frozen at baseline: they do not enter
# any tunnel measurement (they were optimized in design_optimize.py).
TUNNEL_VARS = ("S_scale", "b_scale", "margin_kg", "CL_max", "CD0")

WEIGHTS = {"P_cruise": 0.45, "inv_LDmax": 0.35, "n_gust_rms": 0.20}
LIMITS = {"V_stall_max": 13.0, "mass_max": 95.0}
PENALTY = 10.0

METRIC_KEYS = ("P_cruise", "inv_LDmax", "n_gust_rms", "LD_max", "CL_max_meas",
               "alpha_stall_deg", "CD0_meas", "K_meas", "CL_alpha_rad",
               "CL_trim_cruise", "V_stall", "V_cruise", "mass", "S", "AR")


def metrics_json(m: dict) -> dict:
    """JSON-safe subset of a tunnel evaluation's metrics."""
    return {k: (bool(m[k]) if isinstance(m[k], (bool, np.bool_))
                else float(m[k])) for k in METRIC_KEYS}


@dataclass
class TunnelConfig:
    out: Path = Path("results/wind_tunnel_001")
    # -- steady test section ------------------------------------------
    altitudes: tuple = (0.0, 50.0, 150.0, 500.0)   # m, ISA troposphere
    alpha_min_deg: float = -8.0
    alpha_max_deg: float = 24.0
    alpha_step_deg: float = 0.5
    vel_points: int = 80
    ge_fracs: tuple = (0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.2, 2.0)  # h/b
    # -- unsteady (gust) test section ----------------------------------
    cruise_altitude: float = 30.0   # m, patrol altitude over water
    gust_period: float = 6.0        # s
    gust_seed: int = 42
    gust_duration: float = 60.0     # s, report runs
    gust_dt: float = 0.02           # s, report runs
    gust_rms_levels: tuple = (1.0, 2.0, 3.0)   # m/s per component
    shear_heights: tuple = (1.0, 2.0, 5.0, 10.0, 30.0, 100.0)
    # -- objective / search (mirror design_optimize.py) ----------------
    opt_gust_rms: float = 1.0       # m/s, turbulence in the objective
    opt_gust_duration: float = 20.0  # s window inside the search
    opt_gust_dt: float = 0.05       # s
    initial_step_frac: float = 0.25
    min_step_frac: float = 1.0 / 64.0
    shrink: float = 0.5
    max_rounds: int = 40
    cross_check: Path = Path("results/design_opt_001/best_design.json")
    plots: bool = True
    quick: bool = False

    def __post_init__(self):
        self.out = Path(self.out)
        if self.quick:
            self.alpha_step_deg = 2.0
            self.vel_points = 30
            self.gust_duration = 20.0
            self.gust_dt = 0.05
            self.opt_gust_duration = 10.0
            self.initial_step_frac = 0.5
            self.min_step_frac = 1.0 / 8.0
            self.max_rounds = 12


# ---------------------------------------------------------------------
#  Test-section conditions from atmosphere.py
# ---------------------------------------------------------------------
def sutherland_viscosity(T: float) -> float:
    """Dynamic viscosity of air, kg/(m s) (Sutherland's law)."""
    return MU_SUTH_A * T ** 1.5 / (T + MU_SUTH_S)


def speed_of_sound(T: float) -> float:
    return math.sqrt(GAMMA_AIR * ISA_R * T)


def tunnel_conditions(atm: Atmosphere, altitude: float) -> dict:
    """Thermodynamic state of the test section at `altitude`."""
    T = atm.temperature(altitude)
    return {
        "altitude_m": altitude,
        "temperature_K": T,
        "pressure_Pa": atm.pressure(altitude),
        "density_kg_m3": atm.density(altitude),
        "sound_speed_m_s": speed_of_sound(T),
        "viscosity_kg_ms": sutherland_viscosity(T),
        "shear_factor": atm.shear_factor(altitude),
    }


def reynolds(rho: float, V: float, chord: float, mu: float) -> float:
    return rho * V * chord / mu


def mach_number(V: float, T: float) -> float:
    return V / speed_of_sound(T)


# ---------------------------------------------------------------------
#  Experiment 1/2: alpha polar and altitude series
# ---------------------------------------------------------------------
def alpha_grid(cfg: TunnelConfig) -> np.ndarray:
    n = int(round((cfg.alpha_max_deg - cfg.alpha_min_deg)
                  / cfg.alpha_step_deg)) + 1
    return np.linspace(cfg.alpha_min_deg, cfg.alpha_max_deg, n)


def polar_sweep(ac: Aircraft, alphas_deg: np.ndarray,
                height_m: float | None = None) -> list[dict]:
    """Balance readings: CL/CD/L-D at each alpha (no side force)."""
    rows = []
    for a_deg in alphas_deg:
        a = math.radians(float(a_deg))
        CL = ac.CL(a)
        CD = ac.CD(CL, height_m)
        rows.append({"alpha_deg": float(a_deg), "alpha_rad": a,
                     "CL": CL, "CD": CD,
                     "L_D": (CL / CD) if CD > 0 else float("nan")})
    return rows


def measure_polar(rows: list[dict]) -> dict:
    """Reduce a polar sweep to tunnel measurements.

    The lift slope is least-squares fitted over the longest run of
    strictly increasing CL (the linear, unstalled region); CD is fitted
    as CD0 + K*CL^2; stall is where CL first saturates at its maximum.
    """
    al = np.array([r["alpha_deg"] for r in rows])
    cl = np.array([r["CL"] for r in rows])
    cd = np.array([r["CD"] for r in rows])

    # Linear (unstalled) region: leading run of segments whose slope
    # equals the first non-saturated segment's slope to machine
    # precision.  The segment crossing into the CL cap has a reduced
    # slope and terminates the run.
    d = np.diff(cl) / np.diff(al)
    i0 = int(np.argmax(np.abs(d) > 1e-12))
    s0 = d[i0]
    n_lin = 0
    for v in np.abs(d[i0:] - s0) < 1e-9 * abs(s0):
        if v:
            n_lin += 1
        else:
            break
    lin = slice(i0, i0 + n_lin + 1)
    slope, intercept = np.polyfit(np.radians(al[lin]), cl[lin], 1)

    cl_max = float(cl.max())
    alpha_stall = float(al[int(np.argmax(cl >= cl_max - 1e-9))])

    A = np.column_stack([np.ones_like(cl), cl ** 2])
    (cd0, k_ind), *_ = np.linalg.lstsq(A, cd, rcond=None)

    ld = cl / cd
    i_ld = int(np.argmax(ld))
    cl_star = math.sqrt(max(float(cd0), 1e-12) / max(float(k_ind), 1e-12))
    return {
        "CL_alpha_rad": float(slope),
        "CL_alpha_deg": float(slope * math.pi / 180.0),
        "CL0": float(intercept),
        "CL_max_meas": cl_max,
        "alpha_stall_deg": alpha_stall,
        "CD0_meas": float(cd0),
        "K_meas": float(k_ind),
        "LD_max": float(ld[i_ld]),
        "CL_at_LDmax": float(cl[i_ld]),
        "alpha_LDmax_deg": float(al[i_ld]),
        "CL_at_LDmax_analytic": cl_star,          # sqrt(CD0/K)
        "LD_max_analytic": cl_star / (2.0 * float(cd0)),
    }


def altitude_series(ac: Aircraft, atm: Atmosphere,
                    cfg: TunnelConfig) -> list[dict]:
    """Test-section state + stall/cruise performance vs altitude."""
    rows = []
    for h in cfg.altitudes:
        c = tunnel_conditions(atm, h)
        rho = c["density_kg_m3"]
        V_stall = math.sqrt(2.0 * ac.W / (rho * ac.geom.S * ac.aero.CL_max))
        V_cruise = V_stall * math.sqrt(3.0)
        q = 0.5 * rho * V_cruise ** 2
        rows.append({
            **c,
            "V_stall_m_s": V_stall,
            "V_cruise_m_s": V_cruise,
            "q_cruise_Pa": q,
            "Re_cruise": reynolds(rho, V_cruise, ac.geom.c,
                                  c["viscosity_kg_ms"]),
            "Mach_cruise": mach_number(V_cruise, c["temperature_K"]),
        })
    return rows


def shear_series(atm: Atmosphere, cfg: TunnelConfig) -> list[dict]:
    """Log-law mean-wind profile of the tunnel (factor at z_ref = 1)."""
    return [{"height_m": h, "shear_factor": atm.shear_factor(h)}
            for h in cfg.shear_heights]


# ---------------------------------------------------------------------
#  Experiment 3: level-flight velocity sweep
# ---------------------------------------------------------------------
def velocity_sweep(ac: Aircraft, rho: float, cfg: TunnelConfig,
                   V_lo: float | None = None,
                   V_hi: float | None = None) -> list[dict]:
    """Trimmed level flight: for each speed the balance must carry
    CL = W/(qS); record drag, shaft power, L/D and installed thrust."""
    V_stall = math.sqrt(2.0 * ac.W / (rho * ac.geom.S * ac.aero.CL_max))
    lo = V_stall * 1.05 if V_lo is None else V_lo
    hi = (ac.prop.V_max * 0.999) if V_hi is None else V_hi
    eta = ac.prop.eta_motor * ac.prop.eta_esc * ac.prop.eta_prop
    rows = []
    for V in np.linspace(lo, hi, cfg.vel_points):
        V = float(V)
        q = 0.5 * rho * V * V
        CL = ac.W / (q * ac.geom.S)
        feasible = CL <= ac.aero.CL_max + 1e-12
        CL = min(CL, ac.aero.CL_max)
        CD = ac.CD(CL)
        D = q * ac.geom.S * CD
        alpha_trim = (CL - ac.aero.CL0) / ac.CL_alpha_3d
        T_avail = ac.prop.thrust(V, 1.0, rho)
        rows.append({
            "V_m_s": V, "q_Pa": q, "CL_trim": CL,
            "alpha_trim_deg": math.degrees(alpha_trim),
            "CD": CD, "D_N": D, "P_req_W": D * V / eta,
            "L_D": CL / CD if CD > 0 else float("nan"),
            "T_avail_N": T_avail,
            "thrust_margin_N": T_avail - D,
            "feasible": bool(feasible),
        })
    return rows


def sweep_landmarks(rows: list[dict]) -> dict:
    """Min-power / min-drag / max-L-D points of a velocity sweep."""
    P = np.array([r["P_req_W"] for r in rows])
    D = np.array([r["D_N"] for r in rows])
    V = np.array([r["V_m_s"] for r in rows])
    LD = np.array([r["L_D"] for r in rows])
    i_p, i_d, i_ld = int(np.argmin(P)), int(np.argmin(D)), int(np.argmax(LD))
    margin = np.array([r["thrust_margin_N"] for r in rows])
    ok = margin >= 0.0
    i_lvl = int(V.size - 1 - np.argmax(ok[::-1])) if ok.any() else -1
    return {
        "V_minpower_m_s": float(V[i_p]), "P_min_W": float(P[i_p]),
        "V_mindrag_m_s": float(V[i_d]), "D_min_N": float(D[i_d]),
        "LD_max_sweep": float(LD[i_ld]),
        "V_maxlevel_m_s": float(V[i_lvl]) if i_lvl >= 0 else float("nan"),
    }


# ---------------------------------------------------------------------
#  Experiment 4: ground effect
# ---------------------------------------------------------------------
def ground_effect_sweep(ac: Aircraft, rho: float,
                        cfg: TunnelConfig) -> list[dict]:
    """Induced-drag relief vs height above the water (h/b)."""
    V = ac.V_cruise
    q = 0.5 * rho * V * V
    CL = min(ac.W / (q * ac.geom.S), ac.aero.CL_max)
    cd_oge = ac.CD(CL)                      # out of ground effect
    rows = []
    for frac in cfg.ge_fracs:
        h = frac * ac.geom.b
        cd = ac.CD(CL, h)
        rows.append({
            "h_over_b": frac, "h_m": h,
            "ge_factor": ac.induced_drag_factor(h),
            "CD": cd, "CD_OGE": cd_oge,
            "D_reduction_pct": (cd_oge - cd) / cd_oge * 100.0,
        })
    return rows


# ---------------------------------------------------------------------
#  Experiment 5: unsteady gust tunnel
# ---------------------------------------------------------------------
def make_gust_atmosphere(cfg: TunnelConfig, model: str, rms: float,
                         mean_wind: tuple = (0.0, 0.0, 0.0),
                         seed: int | None = None) -> Atmosphere:
    return Atmosphere(AtmosphereConfig(wind=mean_wind, gust_rms=rms,
                                       gust_period=cfg.gust_period,
                                       gust_model=model),
                      seed=cfg.gust_seed if seed is None else seed)


def trim_state(ac: Aircraft, rho: float, V: float) -> dict:
    """Angle of attack and CL that hold level flight at speed V."""
    q = 0.5 * rho * V * V
    CL = ac.W / (q * ac.geom.S)
    feasible = CL <= ac.aero.CL_max + 1e-12
    CL = min(CL, ac.aero.CL_max)
    alpha = (CL - ac.aero.CL0) / ac.CL_alpha_3d
    return {"q_Pa": q, "CL_trim": CL, "alpha_trim_rad": alpha,
            "alpha_trim_deg": math.degrees(alpha),
            "trim_feasible": bool(feasible)}


def gust_load_series(ac: Aircraft, atm: Atmosphere, altitude: float,
                     V: float, rho: float, alpha_trim: float,
                     t: np.ndarray):
    """Load-factor history of a frozen-trim airframe convected through
    the turbulent test section (airspeed = ground speed - wind)."""
    S, W = ac.geom.S, ac.W
    n = np.empty(t.size)
    alpha_eff = np.empty(t.size)
    q = np.empty(t.size)
    for i, ti in enumerate(t):
        w = atm.wind(float(ti), altitude)
        vx = V - w[0]
        q[i] = 0.5 * rho * (vx * vx + w[1] * w[1] + w[2] * w[2])
        alpha_eff[i] = alpha_trim + math.atan2(w[2], max(vx, 1e-6))
        n[i] = q[i] * S * ac.CL(alpha_eff[i]) / W
    return n, alpha_eff, q


def gust_load_stats(ac: Aircraft, atm: Atmosphere, *, altitude: float,
                    V: float, duration: float, dt: float,
                    label: str = "") -> dict:
    rho = atm.density(altitude)
    trim = trim_state(ac, rho, V)
    t = np.arange(0.0, duration + 0.5 * dt, dt)
    n, alpha_eff, q = gust_load_series(ac, atm, altitude, V, rho,
                                       trim["alpha_trim_rad"], t)
    return {
        "label": label, "altitude_m": altitude, "V_m_s": V,
        "duration_s": duration, "dt_s": dt, "n_samples": int(t.size),
        **{k: trim[k] for k in ("CL_trim", "alpha_trim_deg",
                                "trim_feasible")},
        "n_mean": float(n.mean()), "n_std": float(n.std()),
        "n_peak": float(n.max()), "n_min": float(n.min()),
        "dn_peak": float(n.max() - 1.0), "dn_min": float(n.min() - 1.0),
        "alpha_peak_deg": float(np.degrees(alpha_eff.max())),
        "alpha_min_deg": float(np.degrees(alpha_eff.min())),
        "t": t, "n": n, "alpha_eff": alpha_eff, "q": q,
    }


def gust_run(ac: Aircraft, cfg: TunnelConfig, model: str, rms: float,
             mean_wind: tuple = (0.0, 0.0, 0.0), label: str = "",
             duration: float | None = None, dt: float | None = None,
             V: float | None = None, altitude: float | None = None) -> dict:
    """One unsteady tunnel run: build the atmosphere, convect, reduce."""
    atm = make_gust_atmosphere(cfg, model, rms, mean_wind)
    alt = cfg.cruise_altitude if altitude is None else altitude
    out = gust_load_stats(ac, atm, altitude=alt,
                          V=ac.V_cruise if V is None else V,
                          duration=cfg.gust_duration if duration is None
                          else duration,
                          dt=cfg.gust_dt if dt is None else dt,
                          label=label)
    out.update({"model": model, "gust_rms": rms,
                "mean_wind_x": float(mean_wind[0]),
                "rho_kg_m3": atm.density(alt)})
    return out


# ---------------------------------------------------------------------
#  Tunnel objective + deterministic search
# ---------------------------------------------------------------------
def tunnel_session_metrics(ac: Aircraft, atm_gust: Atmosphere,
                           cfg: TunnelConfig) -> dict:
    """Compact tunnel session used as the objective (one design point)."""
    rho = atm_gust.density(cfg.cruise_altitude)
    V = ac.V_cruise
    trim = trim_state(ac, rho, V)
    eta = ac.prop.eta_motor * ac.prop.eta_esc * ac.prop.eta_prop
    P_cruise = trim["q_Pa"] * ac.geom.S * ac.CD(trim["CL_trim"]) * V / eta
    polar = measure_polar(polar_sweep(ac, alpha_grid(cfg)))
    g = gust_load_stats(ac, atm_gust, altitude=cfg.cruise_altitude, V=V,
                        duration=cfg.opt_gust_duration, dt=cfg.opt_gust_dt)
    return {
        "P_cruise": P_cruise,
        "inv_LDmax": 1.0 / polar["LD_max"],
        "n_gust_rms": g["n_std"],
        "LD_max": polar["LD_max"],
        "CL_max_meas": polar["CL_max_meas"],
        "alpha_stall_deg": polar["alpha_stall_deg"],
        "CD0_meas": polar["CD0_meas"],
        "K_meas": polar["K_meas"],
        "CL_alpha_rad": polar["CL_alpha_rad"],
        "CL_trim_cruise": trim["CL_trim"],
        "V_stall": ac.V_stall,
        "V_cruise": V,
        "mass": ac.mass.total,
        "S": ac.geom.S,
        "AR": ac.geom.AR,
    }


@dataclass
class TunnelEval:
    dv: dict
    metrics: dict
    violations: dict
    cost: float


class TunnelEvaluator:
    """J = sum_i w_i * m_i/ref_i + PENALTY * sum v_i^2, ref = baseline.

    `ref` is fixed by the first evaluation (the baseline design), so a
    feasible baseline scores exactly sum(WEIGHTS) = 1.0.  No RNG: the
    gust field is the seeded analytic Atmosphere, and every sweep is a
    deterministic grid.
    """

    def __init__(self, cfg: TunnelConfig, atm_gust: Atmosphere):
        self.cfg = cfg
        self.atm = atm_gust
        self.ref: dict | None = None
        self.n_evals = 0

    def evaluate(self, dv: dict, count: bool = True) -> TunnelEval:
        dv = clip_dv(dict(dv))
        m = tunnel_session_metrics(apply_design(dv), self.atm, self.cfg)
        if self.ref is None:
            self.ref = {k: max(abs(float(m[k])), 1e-9) for k in WEIGHTS}
        violations = {}
        for key, lim in (("V_stall", LIMITS["V_stall_max"]),
                         ("mass", LIMITS["mass_max"])):
            excess = (float(m[key]) - lim) / lim
            if excess > 0.0:
                violations[key + "_max"] = excess
        cost = sum(WEIGHTS[k] * float(m[k]) / self.ref[k] for k in WEIGHTS)
        cost += PENALTY * sum(v * v for v in violations.values())
        if count:
            self.n_evals += 1
        return TunnelEval(dv=dv, metrics=m, violations=violations,
                          cost=cost)


def compass_search(ev: TunnelEvaluator, cfg: TunnelConfig):
    """Deterministic +-step first-improvement search over TUNNEL_VARS
    (same scheme as design_optimize.pattern_search, restricted set)."""
    dv = dict(BASELINE_DV)
    best = ev.evaluate(dv)
    history = [{"eval": ev.n_evals, "round": 0, "move": "baseline",
                "cost": best.cost}]
    spans = {n: DESIGN_VARS[n][2] - DESIGN_VARS[n][1] for n in TUNNEL_VARS}
    steps = {n: spans[n] * cfg.initial_step_frac for n in TUNNEL_VARS}
    rounds = 0
    while (rounds < cfg.max_rounds
           and any(steps[n] > spans[n] * cfg.min_step_frac
                   for n in TUNNEL_VARS)):
        rounds += 1
        improved = False
        for name in TUNNEL_VARS:
            for sign in (+1.0, -1.0):
                trial = dict(dv)
                trial[name] = dv[name] + sign * steps[name]
                trial = clip_dv(trial)
                if abs(trial[name] - dv[name]) < 1e-12:
                    continue
                r = ev.evaluate(trial)
                history.append({"eval": ev.n_evals, "round": rounds,
                                "move": f"{name}{'+' if sign > 0 else '-'}",
                                "cost": r.cost})
                if r.cost < best.cost - 1e-12:
                    dv, best, improved = trial, r, True
                    break
            if improved:
                break
        if not improved:
            steps = {n: s * cfg.shrink for n, s in steps.items()}
    return dv, best, history, rounds


def tunnel_sensitivity(ev: TunnelEvaluator, center: dict) -> list[dict]:
    """Single-variable limit sweeps around `center` (usually the best)."""
    c0 = ev.evaluate(center)
    rows = []
    for name in TUNNEL_VARS:
        lo, hi = DESIGN_VARS[name][1], DESIGN_VARS[name][2]
        r_lo = ev.evaluate({**center, name: lo})
        r_hi = ev.evaluate({**center, name: hi})
        rows.append({
            "var": name, "J_center": c0.cost,
            "J_lo": r_lo.cost, "J_hi": r_hi.cost,
            "J_range": abs(r_hi.cost - r_lo.cost),
            "P_cruise_lo": r_lo.metrics["P_cruise"],
            "P_cruise_hi": r_hi.metrics["P_cruise"],
            "LD_max_lo": r_lo.metrics["LD_max"],
            "LD_max_hi": r_hi.metrics["LD_max"],
            "n_rms_lo": r_lo.metrics["n_gust_rms"],
            "n_rms_hi": r_hi.metrics["n_gust_rms"],
        })
    return rows


def cross_check_design(ev: TunnelEvaluator, cfg: TunnelConfig):
    """Score the design_opt_001 optimum in this tunnel (if present)."""
    if not cfg.cross_check.exists():
        return None
    dopt = json.loads(cfg.cross_check.read_text(encoding="utf-8"))
    dv = {**BASELINE_DV, **dopt["design_variables"]}
    r = ev.evaluate(dv, count=False)
    return {"source": str(cfg.cross_check), "dv": r.dv, "cost": r.cost,
            "metrics": r.metrics, "violations": r.violations}


# ---------------------------------------------------------------------
#  Plots
# ---------------------------------------------------------------------
def make_plots(cfg: TunnelConfig, pol_base: list, meas_base: dict,
               pol_best: list, meas_best: dict, vel_base: list,
               vel_best: list, gust_sum4: dict, gust_dry_base: dict,
               gust_dry_best: dict, sens_rows: list) -> list[str]:
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

    # -- polars --------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8))
    for pol, meas, label in ((pol_base, meas_base, "ベースライン"),
                             (pol_best, meas_best, "風洞最適化")):
        al = [r["alpha_deg"] for r in pol]
        axes[0].plot(al, [r["CL"] for r in pol],
                     "--" if label == "ベースライン" else "-",
                     linewidth=1.4, label=label)
        axes[0].plot(meas["alpha_stall_deg"], meas["CL_max_meas"], "o",
                     markersize=4)
        axes[1].plot([r["CD"] for r in pol], [r["CL"] for r in pol],
                     "--" if label == "ベースライン" else "-",
                     linewidth=1.4, label=label)
        i = max(range(len(pol)), key=lambda k: pol[k]["L_D"])
        axes[1].plot(pol[i]["CD"], pol[i]["CL"], "s", markersize=4)
    axes[0].set_xlabel("迎え角 α [deg]")
    axes[0].set_ylabel("CL [-]")
    axes[0].set_title(f"揚力線図 (● 失速: α={meas_base['alpha_stall_deg']:.1f}"
                      f"→{meas_best['alpha_stall_deg']:.1f} deg)")
    axes[1].set_xlabel("CD [-]")
    axes[1].set_ylabel("CL [-]")
    axes[1].set_title(f"抗力ポーラ (■ (L/D)max: {meas_base['LD_max']:.1f}"
                      f"→{meas_best['LD_max']:.1f})")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    p = cfg.out / "polars.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    names.append(p.name)

    # -- velocity sweep --------------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(9.0, 7.0), sharex=True)
    for vel, label in ((vel_base, "ベースライン"), (vel_best, "風洞最適化")):
        style = "--" if label == "ベースライン" else "-"
        V = [r["V_m_s"] for r in vel]
        axes[0].plot(V, [r["P_req_W"] for r in vel], style, linewidth=1.4,
                     label=label)
        axes[1].plot(V, [r["L_D"] for r in vel], style, linewidth=1.4,
                     label=label)
    axes[0].set_ylabel("水平飛行所要電力 [W]")
    axes[1].set_ylabel("L/D [-]")
    axes[1].set_xlabel("対気速度 [m/s] (水平飛行トリム)")
    axes[0].set_title("速度スイープ (海面気圧, トリム CL=W/qS)")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    p = cfg.out / "power.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    names.append(p.name)

    # -- gust loads ------------------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(9.0, 7.0), sharex=True)
    axes[0].plot(gust_sum4["t"], gust_sum4["n"], linewidth=0.9,
                 label=f"sum4 (RMS {gust_sum4['gust_rms']:.0f} m/s)")
    axes[0].plot(gust_dry_base["t"], gust_dry_base["n"], linewidth=0.9,
                 label=f"dryden (RMS {gust_dry_base['gust_rms']:.0f} m/s)")
    axes[0].set_title("突風中の荷重倍率 n(t) — ベースライン")
    axes[1].plot(gust_dry_base["t"], gust_dry_base["n"], "--",
                 linewidth=0.9, label="ベースライン")
    axes[1].plot(gust_dry_best["t"], gust_dry_best["n"], "-",
                 linewidth=0.9, label="風洞最適化")
    axes[1].set_title("突風中の荷重倍率 n(t) — dryden 比較")
    axes[1].set_xlabel("t [s]")
    for ax in axes:
        ax.set_ylabel("n [-]")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    p = cfg.out / "gust.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    names.append(p.name)

    # -- sensitivity ------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    labels = [r["var"] for r in sens_rows]
    xs = np.arange(len(labels))
    j0 = sens_rows[0]["J_center"]
    ax.bar(xs - 0.2, [r["J_lo"] - j0 for r in sens_rows], 0.4, label="下限")
    ax.bar(xs + 0.2, [r["J_hi"] - j0 for r in sens_rows], 0.4, label="上限")
    ax.set_xticks(xs, labels, rotation=20, ha="right")
    ax.set_ylabel("ΔJ (最適設計比)")
    ax.set_title("風洞感度: 各変数を単独で限界値まで動かしたときの目的関数変化")
    ax.grid(alpha=0.3, axis="y")
    ax.legend()
    fig.tight_layout()
    p = cfg.out / "sensitivity.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    names.append(p.name)
    return names


# ---------------------------------------------------------------------
#  Outputs
# ---------------------------------------------------------------------
def _write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


GUST_CSV_KEYS = ("label", "model", "gust_rms", "mean_wind_x", "altitude_m",
                 "V_m_s", "rho_kg_m3", "CL_trim", "alpha_trim_deg",
                 "duration_s", "dt_s", "n_samples", "n_mean", "n_std",
                 "n_peak", "n_min", "dn_peak", "dn_min",
                 "alpha_peak_deg", "alpha_min_deg")


def write_outputs(cfg: TunnelConfig, ev: TunnelEvaluator, base_r: TunnelEval,
                  best_r: TunnelEval, best_dv: dict, history: list,
                  rounds: int, sens_rows: list, alts: list, shear: list,
                  pol_base: list, pol_best: list, vel_base: list,
                  vel_best: list, ge_base: list, gust_rows: list,
                  gust_best: dict, cross: dict | None, plots: list[str],
                  runtime_s: float) -> dict:
    cfg.out.mkdir(parents=True, exist_ok=True)
    bm, bsm = base_r.metrics, best_r.metrics
    imp = (base_r.cost - best_r.cost) / base_r.cost * 100.0

    _write_csv(cfg.out / "altitude_conditions.csv",
               ["altitude_m", "temperature_K", "pressure_Pa",
                "density_kg_m3", "sound_speed_m_s", "viscosity_kg_ms",
                "shear_factor", "V_stall_m_s", "V_cruise_m_s",
                "q_cruise_Pa", "Re_cruise", "Mach_cruise"],
               [[f"{r[k]:.6g}" for k in
                 ("altitude_m", "temperature_K", "pressure_Pa",
                  "density_kg_m3", "sound_speed_m_s", "viscosity_kg_ms",
                  "shear_factor", "V_stall_m_s", "V_cruise_m_s",
                  "q_cruise_Pa", "Re_cruise", "Mach_cruise")]
                for r in alts])

    _write_csv(cfg.out / "shear_profile.csv", ["height_m", "shear_factor"],
               [[r["height_m"], f"{r['shear_factor']:.6f}"] for r in shear])

    pol_rows = []
    for tag, pol in (("baseline", pol_base), ("best", pol_best)):
        for r in pol:
            pol_rows.append([tag, f"{r['alpha_deg']:.4g}",
                             f"{r['CL']:.8f}", f"{r['CD']:.8f}",
                             f"{r['L_D']:.6f}"])
    _write_csv(cfg.out / "polars.csv",
               ["design", "alpha_deg", "CL", "CD", "L_D"], pol_rows)

    vel_rows = []
    for tag, vel in (("baseline", vel_base), ("best", vel_best)):
        for r in vel:
            vel_rows.append([tag, f"{r['V_m_s']:.4f}", f"{r['q_Pa']:.3f}",
                             f"{r['CL_trim']:.6f}",
                             f"{r['alpha_trim_deg']:.4f}",
                             f"{r['CD']:.6f}", f"{r['D_N']:.3f}",
                             f"{r['P_req_W']:.2f}", f"{r['L_D']:.4f}",
                             f"{r['T_avail_N']:.2f}",
                             f"{r['thrust_margin_N']:.2f}",
                             int(r["feasible"])])
    _write_csv(cfg.out / "velocity_sweep.csv",
               ["design", "V_m_s", "q_Pa", "CL_trim", "alpha_trim_deg",
                "CD", "D_N", "P_req_W", "L_D", "T_avail_N",
                "thrust_margin_N", "feasible"], vel_rows)

    _write_csv(cfg.out / "ground_effect.csv",
               ["h_over_b", "h_m", "ge_factor", "CD", "CD_OGE",
                "D_reduction_pct"],
               [[f"{r['h_over_b']:.4g}", f"{r['h_m']:.4f}",
                 f"{r['ge_factor']:.6f}", f"{r['CD']:.8f}",
                 f"{r['CD_OGE']:.8f}", f"{r['D_reduction_pct']:.4f}"]
                for r in ge_base])

    gust_csv = []
    for g in [*gust_rows, gust_best]:
        gust_csv.append([f"{g[k]:.6g}" if isinstance(g[k], (int, float))
                         else g[k] for k in GUST_CSV_KEYS])
    _write_csv(cfg.out / "gust_runs.csv", list(GUST_CSV_KEYS), gust_csv)

    _write_csv(cfg.out / "convergence.csv", ["eval", "round", "move", "cost"],
               [[h["eval"], h["round"], h["move"], f"{h['cost']:.6f}"]
                for h in history])

    _write_csv(cfg.out / "sensitivity.csv",
               ["var", "lo", "hi", "J_center", "J_lo", "J_hi", "J_range",
                "P_cruise_lo", "P_cruise_hi", "LD_max_lo", "LD_max_hi",
                "n_rms_lo", "n_rms_hi"],
               [[r["var"], DESIGN_VARS[r["var"]][1], DESIGN_VARS[r["var"]][2],
                 f"{r['J_center']:.6f}", f"{r['J_lo']:.6f}",
                 f"{r['J_hi']:.6f}", f"{r['J_range']:.6f}",
                 f"{r['P_cruise_lo']:.2f}", f"{r['P_cruise_hi']:.2f}",
                 f"{r['LD_max_lo']:.4f}", f"{r['LD_max_hi']:.4f}",
                 f"{r['n_rms_lo']:.6f}", f"{r['n_rms_hi']:.6f}"]
                for r in sens_rows])

    best_design = {
        "design_variables": best_dv,
        "tunnel_vars": list(TUNNEL_VARS),
        "bounds": {n: [DESIGN_VARS[n][1], DESIGN_VARS[n][2]]
                   for n in TUNNEL_VARS},
        "descriptions": {n: DESIGN_VARS[n][3] for n in TUNNEL_VARS},
        "weights": WEIGHTS,
        "limits": LIMITS,
        "J_baseline": base_r.cost,
        "J_best": best_r.cost,
        "violations_best": best_r.violations,
        "metrics": {"baseline": metrics_json(bm),
                    "best": metrics_json(bsm)},
        "cross_check_design_opt_001": (
            None if cross is None else
            {"cost": cross["cost"], "metrics": metrics_json(cross["metrics"])}),
    }
    (cfg.out / "best_design.json").write_text(
        json.dumps(best_design, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")

    top_sens = sorted(sens_rows, key=lambda r: -r["J_range"])[:3]
    summary = {
        "study": "大気モデル (atmosphere.py) 駆動の仮想風洞実験と空力設計最適化",
        "train": {
            "tunnel": "ISA 高度別条件 + 対数シアー平均風 + sum4/dryden 突風",
            "search": "compass/pattern (first-improvement, deterministic)",
            "vars": list(TUNNEL_VARS),
            "evals": ev.n_evals,
            "rounds": rounds,
            "objective": {"weights": WEIGHTS, "limits": LIMITS,
                          "penalty": PENALTY,
                          "cruise_altitude_m": cfg.cruise_altitude,
                          "gust_model": "dryden",
                          "gust_rms_m_s": cfg.opt_gust_rms,
                          "gust_seed": cfg.gust_seed},
            "quick": cfg.quick,
        },
        "eval": {
            "J_baseline": base_r.cost,
            "J_best": best_r.cost,
            "improvement_pct": imp,
            "metrics_baseline": metrics_json(bm),
            "metrics_best": metrics_json(bsm),
            "sensitivity_top": [{"var": r["var"], "J_range": r["J_range"]}
                                for r in top_sens],
            "cross_check_design_opt_001": (
                None if cross is None else
                {"J": cross["cost"],
                 "metrics": metrics_json(cross["metrics"])}),
        },
        "finding": (
            f"風洞目的関数 J を {imp:.1f}% 改善 "
            f"({base_r.cost:.3f} -> {best_r.cost:.3f}, 評価 {ev.n_evals} 回 / "
            f"{rounds} ラウンド)。巡航電力 {bm['P_cruise']:.0f} -> "
            f"{bsm['P_cruise']:.0f} W, (L/D)max {bm['LD_max']:.1f} -> "
            f"{bsm['LD_max']:.1f}, 突風荷重 RMS {bm['n_gust_rms']:.4f} -> "
            f"{bsm['n_gust_rms']:.4f}。感度上位は "
            + ", ".join(r["var"] for r in top_sens) + "。"),
        "artifacts": [f"results/{cfg.out.name}/{n}" for n in
                      ["REPORT.md", "summary.json", "best_design.json",
                       "altitude_conditions.csv", "shear_profile.csv",
                       "polars.csv", "velocity_sweep.csv",
                       "ground_effect.csv", "gust_runs.csv",
                       "convergence.csv", "sensitivity.csv",
                       "wind_tunnel.py", *plots]],
        "caveats": [
            "空力モデルはレイノルズ数非依存 (CD0/CLα は Re・マッハで不変と仮定)",
            "突風荷重は準定常 (有効迎え角のみ変動、空力遅れ・構造弾性は不含)",
            "トリム固定の荷重計測 (操縦応答・安定性は対象外)",
            "最適化は空力系 5 変数のみ (プロペラ/船体変数は design_optimize.py 管轄)",
            "質量結合は design_optimize.py と同一のヒューリスティック",
        ],
        "git_head": git_head(),
        "runtime_s": round(runtime_s, 1),
    }
    (cfg.out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    return summary


# ---------------------------------------------------------------------
#  Report
# ---------------------------------------------------------------------
def write_report(cfg: TunnelConfig, ev: TunnelEvaluator, base_r: TunnelEval,
                 best_r: TunnelEval, best_dv: dict, sens_rows: list,
                 alts: list, shear: list, ac_base: Aircraft,
                 ac_best: Aircraft, meas_base: dict, meas_best: dict,
                 lm_base: dict, lm_best: dict, ge_base: list,
                 gust_rows: list, gust_best: dict, cross: dict | None,
                 rounds: int, runtime_s: float, plots: list[str]):
    bm, bsm = base_r.metrics, best_r.metrics
    imp = (base_r.cost - best_r.cost) / base_r.cost * 100.0
    L = []
    L.append("# 風洞実験スタディ (wind_tunnel)")
    L.append("")
    L.append("## 目的")
    L.append("本プロジェクトの大気モデル (`atmosphere.py`: ISA 高度特性・")
    L.append("対数シアー平均風・sum4/dryden 突風) を試験部の条件として使い、")
    L.append("現行機体 (`aircraft.Aircraft()` = 設計変数ベースライン) の")
    L.append("仮想風洞実験を行い、空力計測に基づく最適設計を探索する。")
    L.append("デフォルト機体定数は一切変更しない。")
    L.append("")
    L.append("## 風洞の構成")
    L.append(f"- 試験部気体: ISA (atmosphere.isa_*)、高度 {', '.join(f'{h:g}' for h in cfg.altitudes)} m")
    L.append(f"- 平均風: 対数シアー (z_ref=10 m, z0=0.001 m)、高度 "
             f"{cfg.cruise_altitude:g} m の係数 "
             f"{Atmosphere(seed=0).shear_factor(cfg.cruise_altitude):.3f}")
    L.append(f"- 突風: AtmosphereConfig(gust_rms, gust_period={cfg.gust_period:g} s, "
             f"gust_model=sum4|dryden), seed={cfg.gust_seed} (完全再現)")
    L.append(f"- 計測: α ポーラ ({cfg.alpha_min_deg:g}〜{cfg.alpha_max_deg:g} deg, "
             f"{cfg.alpha_step_deg:g} deg 刻み)、速度スイープ (トリム水平飛行)、"
             "地面効果、突風荷重時系列")
    L.append("")
    L.append("### 高度別試験部条件 (ベースライン機体)")
    L.append("")
    L.append("| 高度 [m] | 気温 [K] | 気圧 [kPa] | 密度 [kg/m³] | 音速 [m/s] "
             "| 粘性 [1e-5 Pa·s] | V_stall [m/s] | Re (巡航) | Mach |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for r in alts:
        L.append(f"| {r['altitude_m']:g} | {r['temperature_K']:.2f} "
                 f"| {r['pressure_Pa']/1000:.2f} | {r['density_kg_m3']:.4f} "
                 f"| {r['sound_speed_m_s']:.1f} "
                 f"| {r['viscosity_kg_ms']*1e5:.3f} | {r['V_stall_m_s']:.2f} "
                 f"| {r['Re_cruise']:.3e} | {r['Mach_cruise']:.4f} |")
    L.append("")
    L.append("### 平均風シアー係数 (z_ref=10 m 基準)")
    L.append("")
    L.append("| 高度 [m] | " + " | ".join(f"{r['height_m']:g}" for r in shear) + " |")
    L.append("|---|" + "---|" * len(shear))
    L.append("| 係数 | " + " | ".join(f"{r['shear_factor']:.3f}" for r in shear) + " |")
    L.append("")
    L.append("## ベースライン計測")
    L.append("")
    L.append("### ポーラ還元 (実測 → 空力パラメータ同定)")
    L.append("")
    k_theory = 1.0 / (math.pi * ac_base.aero.e * ac_base.geom.AR)
    a_stall_theory = math.degrees((ac_base.aero.CL_max - ac_base.aero.CL0)
                                  / ac_base.CL_alpha_3d)
    L.append("| 項目 | 風洞実測 | 理論値 | 差 |")
    L.append("|---|---|---|---|")
    for name, meas, theo, nd in (
            ("CLα [/rad]", meas_base["CL_alpha_rad"], ac_base.CL_alpha_3d, 4),
            ("CL0 [-]", meas_base["CL0"], ac_base.aero.CL0, 4),
            ("CL_max [-]", meas_base["CL_max_meas"], ac_base.aero.CL_max, 4),
            ("CD0 [-]", meas_base["CD0_meas"], ac_base.aero.CD0, 5),
            ("誘導抗力係数 K", meas_base["K_meas"], k_theory, 5),
            ("(L/D)max", meas_base["LD_max"], meas_base["LD_max_analytic"], 3),
            ("失速角 [deg]", meas_base["alpha_stall_deg"], a_stall_theory, 2)):
        L.append(f"| {name} | {meas:.{nd}f} | {theo:.{nd}f} "
                 f"| {abs(meas-theo):.1e} |")
    L.append("")
    L.append("計測値は最小二乗還元 (揚力線形域フィッティング、CD0+K·CL² 二次フィッティング)")
    L.append("で理論値を機械精度レベルで再現しており、風洞パイプラインの整合を示す。")
    L.append("(L/D)max と失速角は α グリッド分解能の影響を受ける (±1 ステップ以内)。")
    L.append("")
    L.append("### 速度スイープ (海面、トリム水平飛行)")
    L.append("")
    L.append("| 項目 | ベースライン |")
    L.append("|---|---|")
    L.append(f"| 最小電力速度 / 電力 | {lm_base['V_minpower_m_s']:.2f} m/s / {lm_base['P_min_W']:.0f} W |")
    L.append(f"| 最小抗力速度 | {lm_base['V_mindrag_m_s']:.2f} m/s |")
    L.append(f"| (L/D)max (スイープ上) | {lm_base['LD_max_sweep']:.2f} |")
    L.append(f"| 全開水平飛行可能最大速度 | {_fmt(lm_base['V_maxlevel_m_s'])} m/s |")
    L.append("")
    L.append("### 地面効果 (巡航トリム CL、海面)")
    L.append("")
    L.append("| h/b | " + " | ".join(f"{r['h_over_b']:g}" for r in ge_base) + " |")
    L.append("|---|" + "---|" * len(ge_base))
    L.append("| 抗力低減 [%] | "
             + " | ".join(f"{r['D_reduction_pct']:.2f}" for r in ge_base) + " |")
    L.append("")
    L.append("離着水滑走 (design_optimize.py 参照) は h/b≲0.05 の領域で起こり、")
    L.append("誘導抗力が 1 割以上緩和される (McCormick 地面効果モデル)。")
    L.append("")
    L.append("### 突風荷重 (高度 "
             f"{cfg.cruise_altitude:g} m、V=V_cruise、トリム固定)")
    L.append("")
    L.append("| ケース | 乱流 RMS [m/s] | n_mean | n_std | Δn_peak | Δn_min "
             "| α_peak [deg] |")
    L.append("|---|---|---|---|---|---|---|")
    for g in gust_rows:
        L.append(f"| {g['label']} | {g['gust_rms']:g} | {g['n_mean']:.3f} "
                 f"| {g['n_std']:.4f} | {g['dn_peak']:+.3f} "
                 f"| {g['dn_min']:+.3f} | {g['alpha_peak_deg']:.2f} |")
    L.append("")
    L.append("shear_z1m は同じ dryden 乱流 (RMS 1 m/s) を海面直下 z=1 m で")
    L.append("再計測した例: シアー係数が "
             f"{Atmosphere(seed=0).shear_factor(1.0):.3f} まで下がり、"
             "突風荷重が軽減される (対数シアーの実演)。")
    L.append("同一 seed のとき時系列はビット単位で再現する (解析的ガウス過程)。")
    L.append("")
    L.append("## 空力設計の最適化")
    L.append("")
    L.append(f"- 目的関数 J = Σ wᵢ·(指標ᵢ/ベースライン値) + {PENALTY:g}·Σ(違反量²)、"
             "重み: " + ", ".join(f"{k}={v:g}" for k, v in WEIGHTS.items()))
    L.append("  - P_cruise: 巡航高度 "
             f"{cfg.cruise_altitude:g} m の ISA 密度での水平飛行電力 "
             "(電気入力換算 D·V/η、各設計固有の V_cruise)")
    L.append("  - inv_LDmax: ポーラ実測 (L/D)max の逆数")
    L.append(f"  - n_gust_rms: dryden 乱流 (RMS {cfg.opt_gust_rms:g} m/s, "
             f"seed={cfg.gust_seed}) 中の荷重倍率変動 RMS")
    L.append("- 制約: " + ", ".join(f"{k}={v:g}" for k, v in LIMITS.items()))
    L.append(f"- 探索: 決定論的コンパス探索 ({len(TUNNEL_VARS)} 変数: "
             + ", ".join(f"`{n}`" for n in TUNNEL_VARS)
             + f", ±{cfg.initial_step_frac:g} スパン開始、改善なしで ×{cfg.shrink:g})。"
             "RNG 不使用。")
    L.append("- プロペラ/船体系変数 (D_prop, prop_pitch, Bwl_scale, Lwl_scale) は")
    L.append("  風洞計測に寄与しないためベースラインに固定 "
             "(design_optimize.py の管轄)。")
    L.append("")
    L.append("### 結果")
    L.append("")
    L.append("| 項目 | ベースライン | 風洞最適化 |")
    L.append("|---|---|---|")
    rows = [
        ("目的関数 J", base_r.cost, best_r.cost),
        ("巡航電力 [W]", bm["P_cruise"], bsm["P_cruise"]),
        ("(L/D)max", bm["LD_max"], bsm["LD_max"]),
        ("突風荷重 n_std (目的関数窓)", bm["n_gust_rms"], bsm["n_gust_rms"]),
        ("失速速度 [m/s]", bm["V_stall"], bsm["V_stall"]),
        ("巡航速度 [m/s]", bm["V_cruise"], bsm["V_cruise"]),
        ("質量 [kg]", bm["mass"], bsm["mass"]),
        ("翼面積 S [m²]", bm["S"], bsm["S"]),
        ("アスペクト比 AR", bm["AR"], bsm["AR"]),
        ("CL_max (実測)", bm["CL_max_meas"], bsm["CL_max_meas"]),
        ("失速角 [deg]", bm["alpha_stall_deg"], bsm["alpha_stall_deg"]),
        ("CD0 (実測)", bm["CD0_meas"], bsm["CD0_meas"]),
        ("誘導抗力係数 K", bm["K_meas"], bsm["K_meas"]),
    ]
    nd_map = {"目的関数 J": 3, "突風荷重 n_std (目的関数窓)": 4,
              "誘導抗力係数 K": 5, "CD0 (実測)": 5, "CL_max (実測)": 3}
    for name, a, b in rows:
        nd = nd_map.get(name, 2)
        L.append(f"| {name} | {_fmt(a, nd)} | {_fmt(b, nd)} |")
    L.append("")
    L.append(f"目的関数は **{imp:.1f}% 改善** "
             f"(評価 {ev.n_evals} 回 / {rounds} ラウンド, {runtime_s:.0f} s)。")
    L.append("")
    L.append("| スピープ項目 | ベースライン | 風洞最適化 |")
    L.append("|---|---|---|")
    L.append(f"| 最小電力 [W] | {lm_base['P_min_W']:.0f} | {lm_best['P_min_W']:.0f} |")
    L.append(f"| 最小電力速度 [m/s] | {lm_base['V_minpower_m_s']:.2f} "
             f"| {lm_best['V_minpower_m_s']:.2f} |")
    L.append(f"| (L/D)max (スイープ) | {lm_base['LD_max_sweep']:.2f} "
             f"| {lm_best['LD_max_sweep']:.2f} |")
    L.append("")
    L.append(f"| 突風荷重 (dryden RMS {cfg.opt_gust_rms:g} m/s, レポート窓 "
             f"{cfg.gust_duration:g} s) | ベースライン | 風洞最適化 |")
    L.append("|---|---|---|")
    gd = next(g for g in gust_rows if g["label"] == f"dryden_rms{cfg.opt_gust_rms:g}")
    L.append(f"| n_mean | {gd['n_mean']:.3f} | {gust_best['n_mean']:.3f} |")
    L.append(f"| n_std | {gd['n_std']:.4f} | {gust_best['n_std']:.4f} |")
    L.append(f"| Δn_peak | {gd['dn_peak']:+.3f} | {gust_best['dn_peak']:+.3f} |")
    L.append(f"| Δn_min | {gd['dn_min']:+.3f} | {gust_best['dn_min']:+.3f} |")
    L.append("")
    L.append(f"RMS {cfg.opt_gust_rms:g} m/s でも鉛直突風成分は巡航速度比で大きく")
    L.append("(σ_w/V ≈ 10%)、荷重応答は強く非線形 (失速クリップに接近) なため、")
    L.append("設計間の n_std 差は数%に留まる。風洞最適化の利得は主に")
    L.append("巡航電力と (L/D)max 側にある。")
    L.append("")
    L.append("### 最適設計変数")
    L.append("")
    L.append("| 変数 | 説明 | 下限 | ベースライン | 上限 | 風洞最適値 |")
    L.append("|---|---|---|---|---|---|")
    for name in TUNNEL_VARS:
        base_v, lo, hi, desc = DESIGN_VARS[name]
        L.append(f"| `{name}` | {desc} | {lo:g} | {base_v:g} | {hi:g} "
                 f"| {best_dv[name]:.4g} |")
    L.append("")
    L.append("### 感度 (最適設計周辺、単変数を限界まで動かした ΔJ)")
    L.append("")
    L.append("| 変数 | J(下限) | J(上限) | |ΔJ| |")
    L.append("|---|---|---|---|")
    for r in sorted(sens_rows, key=lambda r: -r["J_range"]):
        L.append(f"| `{r['var']}` | {r['J_lo']:.3f} | {r['J_hi']:.3f} "
                 f"| {r['J_range']:.3f} |")
    L.append("")
    L.append("### design_opt_001 (離水・着水込み最適設計) との相互評価")
    L.append("")
    if cross is None:
        L.append("(results/design_opt_001/best_design.json が見つからないため省略)")
    else:
        cm = cross["metrics"]
        L.append("| 項目 | ベースライン | design_opt_001 | 風洞最適化 |")
        L.append("|---|---|---|---|")
        L.append(f"| 風洞目的関数 J | {base_r.cost:.3f} | {cross['cost']:.3f} "
                 f"| {best_r.cost:.3f} |")
        L.append(f"| 巡航電力 [W] | {bm['P_cruise']:.0f} "
                 f"| {cm['P_cruise']:.0f} | {bsm['P_cruise']:.0f} |")
        L.append(f"| (L/D)max | {bm['LD_max']:.2f} | {cm['LD_max']:.2f} "
                 f"| {bsm['LD_max']:.2f} |")
        L.append(f"| 突風荷重 n_std | {bm['n_gust_rms']:.4f} "
                 f"| {cm['n_gust_rms']:.4f} | {bsm['n_gust_rms']:.4f} |")
        L.append(f"| 質量 [kg] | {bm['mass']:.1f} | {cm['mass']:.1f} "
                 f"| {bsm['mass']:.1f} |")
        L.append("")
        L.append("離水・着水性能を主に改善した design_opt_001 の設計も、")
        L.append("風洞目的関数でベースラインより良好な位置にある (巡航電力・")
        L.append("(L/D)max の改善が寄与)。純空力目的の風洞最適化はさらに")
        L.append("軽量化と高アスペクト比方向へ踏み込む。")
    L.append("")
    L.append("## 成果物")
    L.append("")
    for n in ["REPORT.md", "summary.json", "best_design.json",
              "altitude_conditions.csv", "shear_profile.csv", "polars.csv",
              "velocity_sweep.csv", "ground_effect.csv", "gust_runs.csv",
              "convergence.csv", "sensitivity.csv", "wind_tunnel.py", *plots]:
        L.append(f"- `results/{cfg.out.name}/{n}`")
    L.append("")
    L.append("## 限界・注意")
    L.append("")
    for c in ("空力モデルはレイノルズ数・マッハ数に依存しない (計測条件は記録のみ)",
              "突風荷重は準定常仮定 (空力的遅れ・構造弾性・制御応答は不含)",
              "トリム固定計測のため、突風中の姿勢保持 (操縦) は評価していない",
              "最適化は空力系 5 変数のみ。プロペラ・船体は design_optimize.py 参照",
              "margin_kg の削減は構造・積載要件で下駄を履かせる必要あり "
              "(下限 -3.5 kg は設計マージン 0.4 kg を確保)"):
        L.append(f"- {c}")
    L.append("")
    (cfg.out / "REPORT.md").write_text("\n".join(L), encoding="utf-8")


# ---------------------------------------------------------------------
#  Study driver
# ---------------------------------------------------------------------
def run_study(cfg: TunnelConfig) -> dict:
    t0 = time.time()
    cfg.out.mkdir(parents=True, exist_ok=True)
    atm0 = Atmosphere(seed=0)                  # steady test section
    ac_base = apply_design(BASELINE_DV)
    al_grid = alpha_grid(cfg)
    rho0 = atm0.density(0.0)                   # == aircraft.RHO (ISA pinned)

    # -- steady measurements (baseline) --------------------------------
    alts = altitude_series(ac_base, atm0, cfg)
    shear = shear_series(atm0, cfg)
    pol_base = polar_sweep(ac_base, al_grid)
    meas_base = measure_polar(pol_base)
    vel_base = velocity_sweep(ac_base, rho0, cfg)
    lm_base = sweep_landmarks(vel_base)
    ge_base = ground_effect_sweep(ac_base, rho0, cfg)

    # -- unsteady measurements (baseline) -------------------------------
    gust_rows = []
    for model in ("sum4", "dryden"):
        for rms in cfg.gust_rms_levels:
            gust_rows.append(gust_run(ac_base, cfg, model, rms,
                                      label=f"{model}_rms{rms:g}"))
    gust_rows.append(gust_run(ac_base, cfg, "dryden", 1.0,
                              label="shear_z1m", altitude=1.0))

    # -- optimization ----------------------------------------------------
    atm_g = make_gust_atmosphere(cfg, "dryden", cfg.opt_gust_rms)
    ev = TunnelEvaluator(cfg, atm_g)
    best_dv, best_r, history, rounds = compass_search(ev, cfg)
    base_r = ev.evaluate(BASELINE_DV, count=False)
    sens_rows = tunnel_sensitivity(ev, best_dv)
    cross = cross_check_design(ev, cfg)

    # -- final session on the best design --------------------------------
    ac_best = apply_design(best_dv)
    pol_best = polar_sweep(ac_best, al_grid)
    meas_best = measure_polar(pol_best)
    vel_best = velocity_sweep(ac_best, rho0, cfg)
    lm_best = sweep_landmarks(vel_best)
    gust_best = gust_run(ac_best, cfg, "dryden", cfg.opt_gust_rms,
                         label="best_objective")
    gust_sum4 = next(g for g in gust_rows
                     if g["label"] == f"sum4_rms{cfg.opt_gust_rms:g}")
    gust_dry_base = next(g for g in gust_rows
                         if g["label"] == f"dryden_rms{cfg.opt_gust_rms:g}")

    plots = []
    if cfg.plots:
        plots = make_plots(cfg, pol_base, meas_base, pol_best, meas_best,
                           vel_base, vel_best, gust_sum4, gust_dry_base,
                           gust_best, sens_rows)
    shutil.copy2(Path(__file__).resolve(), cfg.out / "wind_tunnel.py")

    runtime_s = time.time() - t0
    summary = write_outputs(
        cfg, ev, base_r, best_r, best_dv, history, rounds, sens_rows,
        alts, shear, pol_base, pol_best, vel_base, vel_best, ge_base,
        gust_rows, gust_best, cross, plots, runtime_s)
    write_report(cfg, ev, base_r, best_r, best_dv, sens_rows, alts, shear,
                 ac_base, ac_best, meas_base, meas_best, lm_base, lm_best,
                 ge_base, gust_rows, gust_best, cross, rounds, runtime_s,
                 plots)
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Virtual wind-tunnel study driven by atmosphere.py")
    ap.add_argument("--out", default="results/wind_tunnel_001",
                    help="results folder (default: results/wind_tunnel_001)")
    ap.add_argument("--quick", action="store_true",
                    help="coarse grids and short search (smoke run)")
    ap.add_argument("--no-plots", action="store_true", help="skip PNG plots")
    args = ap.parse_args(argv)
    cfg = TunnelConfig(out=Path(args.out), quick=args.quick,
                       plots=not args.no_plots)
    summary = run_study(cfg)
    ev = summary["eval"]
    print(f"[wind_tunnel] J: {ev['J_baseline']:.4f} -> {ev['J_best']:.4f} "
          f"({ev['improvement_pct']:.1f}% improvement, "
          f"{summary['train']['evals']} evals / {summary['train']['rounds']} rounds)")
    print(f"[wind_tunnel] P_cruise {ev['metrics_baseline']['P_cruise']:.0f} -> "
          f"{ev['metrics_best']['P_cruise']:.0f} W, "
          f"(L/D)max {ev['metrics_baseline']['LD_max']:.2f} -> "
          f"{ev['metrics_best']['LD_max']:.2f}, "
          f"n_std {ev['metrics_baseline']['n_gust_rms']:.4f} -> "
          f"{ev['metrics_best']['n_gust_rms']:.4f}")
    print(f"[wind_tunnel] outputs: {cfg.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
