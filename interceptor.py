"""Interceptor drone models: airframe, guidance, and warhead effects.

Two warhead types are compared.  Both are scored at the geometric
closest approach (impact parameter), not at the range where a sphere
is first entered — that entry range is always ≈ the sphere radius and
would mark every hit as a graze.

    1. Fragmentation -- detonates at closest approach.  Lethal radius
       is small (R_lethal ≈ 6 m).  P_kill is the Poisson expected hits
       from fragment density × target presenting area.
    2. Kinetic impact -- contact if the impact parameter is inside the
       3 m collision radius (airframe + propeller disc).  Kill depends
       on relative speed at closest approach and how central the hit is.

Guidance laws return a desired heading.  The engagement loop tracks it
with a bank servo.  Proportional navigation integrates N × LOS-rate
into that command.  Re-basing the command on the current heading each
step leaves only one sample of N·Δλ, which this servo cannot track.

Disturbances (optional, default off):

* ``WindField`` — steady wind plus an AR(1) gust.  The interceptor's
  forces and cruise cap act on air-relative velocity; ground track is
  air velocity plus wind.  Guidance has no wind estimate, so a crosswind
  is an unmodelled bias.  The target holds its commanded ground track
  (``target_drift`` = 0, a position-controlled multirotor).
* ``comm_delay`` — the guidance law is fed the observation from
  ``t - comm_delay``, not the live one.  Sensor noise is drawn at the
  sample time, so the delayed fix is stale *and* noisy.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

from aircraft import RHO, G


# ---------------------------------------------------------------------
#  Interceptor airframe
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class InterceptorDrone:
    """Small, high-speed interceptor drone (class-2 CUAS)."""
    mass:        float = 5.0      # total mass, kg
    S:           float = 0.20     # reference area, m^2
    CD0:         float = 0.035    # zero-lift drag coefficient
    T_max:       float = 150.0    # maximum thrust, N  (T/W ≈ 3.0)
    bank_tau:    float = 0.15     # bank servo time constant, s
    max_bank:    float = math.radians(80)
    max_accel:   float = 80.0     # max lateral accel, m/s^2 (~8 g)
    v_cruise:    float = 50.0     # speed cap during engagement, m/s

    @property
    def W(self) -> float:
        return self.mass * G

    def thrust(self, speed: float, throttle: float = 1.0) -> float:
        # Throttle only matters below cruise speed; above that, drag dominates.
        return self.T_max * float(np.clip(throttle, 0.0, 1.0))

    def CD(self) -> float:
        """Parasite drag at small α (wing-level)."""
        return self.CD0


# ---------------------------------------------------------------------
#  Warhead types — deliberately *pessimistic* for both
# ---------------------------------------------------------------------
class WarheadType(Enum):
    FRAGMENTATION = "fragmentation"
    KINETIC_IMPACT = "kinetic_impact"


@dataclass(frozen=True)
class FragmentationWarhead:
    """Small-UAV fragmentation warhead.

    A 5 kg class interceptor can carry ≈0.5 kg of warhead (charge +
    casing).  With ~300 fragments of ~1 g each, lethal radius against
    a small multirotor airframe (propeller, arm, battery) is limited.

    P_kill(r) = 1 − exp(−N_frag × A_tgt × frag_lethality(r) / (4π r²))
    frag_lethality accounts for velocity decay AND minimum energy per
    fragment to penetrate a foam/carbon spar (≈10 J).
    """
    R_lethal:   float = 6.0      # max lethal radius, m (pessimistic)
    N_frag:     int   = 300      # fragments (small warhead)
    V_frag:     float = 1200.0   # fragment initial velocity, m/s
    A_target:   float = 0.8      # target presenting area, m^2
    frag_mass:  float = 0.001    # fragment mass, kg (1 g)
    E_pen:      float = 10.0     # min energy to penetrate, J

    def kill_probability(self, miss_distance: float) -> float:
        r = max(miss_distance, 0.1)
        if r > self.R_lethal:
            return 0.0
        # Fragment velocity decay (exponential in air)
        R_char = 30.0
        V_r = self.V_frag * math.exp(-r / R_char)
        KE_r = 0.5 * self.frag_mass * V_r * V_r
        if KE_r < self.E_pen:
            return 0.0
        # Fraction of fragments that still penetrate
        lethality = min(1.0, (KE_r - self.E_pen) / KE_r)
        mu = self.N_frag * self.A_target * lethality / (4.0 * math.pi * r * r)
        return 1.0 - math.exp(-mu)


@dataclass(frozen=True)
class KineticImpactWarhead:
    """Direct collision (kinetic kill vehicle).

    KE_threshold = 1500 J corresponds to a 3 kg object at 30 m/s.
    Effective collision radius = 3 m (includes propeller disc).
    Kill probability has a *smooth ramp*: below 500 J → negligible,
    above 1500 J → near-certain, with a sigmoid in between.
    """
    KE_threshold: float = 1500.0   # J, 50 % kill point
    KE_floor:     float = 500.0    # J, below this → ~0
    effective_radius: float = 3.0  # m (airframe + prop disc)
    mass: float = 5.0

    def kill_probability(self, miss_distance: float,
                         closing_speed: float) -> float:
        if miss_distance > self.effective_radius:
            return 0.0
        speed = max(abs(closing_speed), 0.0)
        KE = 0.5 * self.mass * speed * speed
        # Glancing hit reduces effective KE
        hit_quality = 1.0 - (miss_distance / self.effective_radius) ** 2
        eff_KE = KE * hit_quality
        # Sigmoid ramp between floor and threshold
        if eff_KE <= self.KE_floor:
            return 0.0
        if eff_KE >= self.KE_threshold:
            return 1.0
        return (eff_KE - self.KE_floor) / (self.KE_threshold - self.KE_floor)

    def kinetic_energy(self, closing_speed: float) -> float:
        return 0.5 * self.mass * abs(closing_speed) ** 2


# ---------------------------------------------------------------------
#  Guidance laws
#
#  Each law exposes a `desired_heading(ri, vi, rt, vt, dt) -> float` method
#  returning the heading angle (rad, +x = 0, +y = +π/2) that the interceptor
#  should fly in.  The bank servo in `run_engagement` then steers toward
#  that heading with time constant `bank_tau`.
#
#  Stateful laws (PN, APN) carry per-trial state in `self.state` and
#  should call `reset()` between trials so the LOS history does not bleed
#  across simulations.
# ---------------------------------------------------------------------
class GuidanceLaw:
    """Base class.  Stateless unless `reset()` is overridden."""

    def reset(self) -> None:
        self.state = {}

    def set_wind_estimate(self, wind) -> None:
        """Optional: air-data + INS wind estimate from the airframe."""

    def desired_heading(self, ri, vi, rt, vt, dt=0.01) -> float:
        raise NotImplementedError


class PurePursuit(GuidanceLaw):
    """Point at the target's *current* position.

    No prediction.  For constant-velocity targets and a fast interceptor
    this converges to a collision course (because the interceptor just
    chases the target).  Miss is dominated by how much the target can
    translate during closure; against maneuvering targets this law lags.
    """

    def desired_heading(self, ri, vi, rt, vt, dt=0.01) -> float:
        return math.atan2(float(rt[1] - ri[1]), float(rt[0] - ri[0]))


class PNGuidance(GuidanceLaw):
    """Proportional navigation.

    Heading-rate command = N × ω, where ω is LOS rate in the inertial
    plane.  The command is integrated on its own state:

        ψ_cmd ← ψ_cmd + N · ω · dt

    Adding N·ω·dt to the *current* velocity heading each step does not
    accumulate, and the bank servo (τ ≈ 0.15 s) then never turns.

    ``los_tau`` > 0 filters the raw one-step LOS rate before integration.
    τ = 0 keeps the raw difference.  A 100 Hz difference of a 0.5 m
    position fix is not a usable LOS rate near the target.
    """
    N: float = 4.0

    def __init__(self, N: float = 4.0, los_tau: float = 0.15):
        self.N = N
        self.los_tau = float(los_tau)
        self.reset()

    def reset(self) -> None:
        self.state = {"los_prev": None, "psi_cmd": None, "omega_f": 0.0}

    @staticmethod
    def _unwrap(prev, curr):
        d = curr - prev
        while d > math.pi:
            d -= 2 * math.pi
        while d < -math.pi:
            d += 2 * math.pi
        return d

    def _heading_rate(self, omega_f, rel, R, vi, vt, dt) -> float:
        return self.N * omega_f

    def desired_heading(self, ri, vi, rt, vt, dt=0.01) -> float:
        rel = np.asarray(rt, dtype=float) - np.asarray(ri, dtype=float)
        R = float(np.linalg.norm(rel))
        hdg = math.atan2(float(vi[1]), float(vi[0]))
        if R < 1.0:
            return math.atan2(float(rt[1] - ri[1]), float(rt[0] - ri[0]))
        los = math.atan2(float(rel[1]), float(rel[0]))
        prev = self.state.get("los_prev")
        if prev is None or self.state.get("psi_cmd") is None:
            self.state["los_prev"] = los
            self.state["psi_cmd"] = hdg
            self.state["omega_f"] = 0.0
            return hdg
        omega = self._unwrap(prev, los) / max(dt, 1e-6)
        if self.los_tau > 1e-6:
            alpha = 1.0 - math.exp(-dt / self.los_tau)
            omega_f = self.state["omega_f"] + alpha * (omega - self.state["omega_f"])
        else:
            omega_f = omega
        self.state["omega_f"] = omega_f
        self.state["los_prev"] = los
        rate = self._heading_rate(omega_f, rel, R, vi, vt, dt)
        psi = self.state["psi_cmd"] + rate * dt
        psi = math.atan2(math.sin(psi), math.cos(psi))
        self.state["psi_cmd"] = psi
        return psi


class APNGuidance(PNGuidance):
    """Augmented proportional navigation.

    Same integrated heading command as PN, with the target's LOS-normal
    acceleration added the way the classical law does:

        ω_eff = ω + ½ a_t_perp / Vc
        ψ_cmd ← ψ_cmd + N · ω_eff · dt

    Target velocity is filtered with the same ``los_tau`` before the
    difference.  A raw (v_k − v_{k−1}) / dt at 100 Hz and σ_v = 0.2 m/s
    is an acceleration noise of tens of m/s².
    """

    def __init__(self, N: float = 4.0, los_tau: float = 0.15):
        super().__init__(N=N, los_tau=los_tau)

    def reset(self) -> None:
        super().reset()
        self.state["vt_f"] = None

    def _heading_rate(self, omega_f, rel, R, vi, vt, dt) -> float:
        vt_arr = np.asarray(vt, dtype=float)
        prev_f = self.state.get("vt_f")
        if prev_f is None:
            vt_f = vt_arr.copy()
            a_t_perp = 0.0
        elif self.los_tau > 1e-6:
            alpha = 1.0 - math.exp(-dt / self.los_tau)
            vt_f = prev_f + alpha * (vt_arr - prev_f)
            a_t = (vt_f - prev_f) / max(dt, 1e-6)
            los_unit = rel / max(R, 1e-6)
            perp = np.array([-los_unit[1], los_unit[0], 0.0])
            a_t_perp = float(np.dot(a_t, perp))
        else:
            vt_f = vt_arr
            a_t = (vt_arr - prev_f) / max(dt, 1e-6)
            los_unit = rel / max(R, 1e-6)
            perp = np.array([-los_unit[1], los_unit[0], 0.0])
            a_t_perp = float(np.dot(a_t, perp))
        self.state["vt_f"] = np.array(vt_f, dtype=float, copy=True)
        los_unit = rel / max(R, 1e-6)
        rel_vel = vt_arr - np.asarray(vi, dtype=float)
        Vc = max(-float(np.dot(rel_vel, los_unit)), 1.0)
        return self.N * (omega_f + 0.5 * a_t_perp / Vc)


class PredictivePursuit(GuidanceLaw):
    """Aim at the constant-velocity intercept point.

    Solves |r + u t| = v_i t in the horizontal plane (collision
    triangle) and aims at the target position at that t.  If no
    positive root exists, falls back to pointing at the current
    target position.  Stateless.  A maneuver breaks the constant-
    velocity assumption and the aim lags until the next solve.

    The returned angle is a *ground* track.  The bank servo flies it
    as an air heading, so an unmodelled crosswind slides the airframe
    off the collision course.  ``CrabPredictive`` offsets the heading
    into the wind so the ground track holds the aim point.
    """

    def desired_heading(self, ri, vi, rt, vt, dt=0.01) -> float:
        rx = float(rt[0] - ri[0])
        ry = float(rt[1] - ri[1])
        R = math.hypot(rx, ry)
        if R < 1.0:
            return math.atan2(ry, rx)
        v_i = math.hypot(float(vi[0]), float(vi[1]))
        if v_i < 1.0:
            return math.atan2(ry, rx)
        ux = float(vt[0])
        uy = float(vt[1])
        a = ux * ux + uy * uy - v_i * v_i
        b = 2.0 * (rx * ux + ry * uy)
        c = rx * rx + ry * ry
        t_go = None
        if abs(a) < 1e-6:
            if b < -1e-6:
                t_go = -c / b
        else:
            disc = b * b - 4.0 * a * c
            if disc >= 0.0:
                root = math.sqrt(disc)
                cand = [t for t in ((-b - root) / (2.0 * a),
                                    (-b + root) / (2.0 * a)) if t > 0.05]
                if cand:
                    t_go = min(cand)
        if t_go is None:
            return math.atan2(ry, rx)
        aim_x = float(rt[0]) + ux * t_go
        aim_y = float(rt[1]) + uy * t_go
        return math.atan2(aim_y - float(ri[1]), aim_x - float(ri[0]))


def _crab_heading(track: float, wind, airspeed: float) -> float:
    """Air heading whose ground track is ``track`` in wind ``wind``.

    Ground velocity is air velocity + wind.  Holding the track θ
    requires pointing into the wind:

        ψ = θ − asin(W⊥ / V_a)

    where W⊥ is the wind component across the track.  Falls back to
    the track itself if the crosswind exceeds the airspeed.
    """
    if airspeed < 1.0:
        return track
    wx, wy = float(wind[0]), float(wind[1])
    # Across-track unit normal of the desired ground track.
    nx, ny = -math.sin(track), math.cos(track)
    w_perp = wx * nx + wy * ny
    ratio = w_perp / airspeed
    if ratio >= 0.999:
        return track - math.pi / 2
    if ratio <= -0.999:
        return track + math.pi / 2
    return track - math.asin(ratio)


class CrabPredictive(GuidanceLaw):
    """Wind-aware collision triangle, then fly the required air heading.

    In wind the interceptor ground velocity is ``V_a·u(ψ) + W``, so the
    triangle is solved in the airmass frame:

        |r + (v_t − W) · t| = V_a · t

    The aim point is the target's *ground* position at that t.  The
    command is the direction of the air velocity that reaches it:

        ψ = atan2( (aim − r_i)/t_go − W )

    Wind comes from ``set_wind_estimate`` (air-data + INS, fed by the
    engagement loop) or, if ``estimate_wind=False``, from the
    constructor ``wind`` as a perfect measurement.
    """

    def __init__(self, wind=(0.0, 0.0, 0.0), airspeed: float = 50.0,
                 estimate_wind: bool = True):
        self.wind = np.asarray(wind, dtype=float)
        self.airspeed = float(airspeed)
        self.estimate_wind = bool(estimate_wind)
        self.reset()

    def reset(self) -> None:
        self.state = {"wind_est": np.array(self.wind, dtype=float, copy=True)}

    def set_wind_estimate(self, wind) -> None:
        self.state["wind_est"] = np.asarray(wind, dtype=float)

    def _wind_now(self) -> np.ndarray:
        return self.state["wind_est"]

    def desired_heading(self, ri, vi, rt, vt, dt=0.01) -> float:
        wind = self._wind_now()
        wx, wy = float(wind[0]), float(wind[1])
        rx = float(rt[0] - ri[0])
        ry = float(rt[1] - ri[1])
        R = math.hypot(rx, ry)
        if R < 1.0:
            return math.atan2(ry, rx)
        # Target motion relative to the airmass.
        ux = float(vt[0]) - wx
        uy = float(vt[1]) - wy
        Va = self.airspeed if self.airspeed > 1.0 else 50.0
        a = ux * ux + uy * uy - Va * Va
        b = 2.0 * (rx * ux + ry * uy)
        c = rx * rx + ry * ry
        t_go = None
        if abs(a) < 1e-6:
            if b < -1e-6:
                t_go = -c / b
        else:
            disc = b * b - 4.0 * a * c
            if disc >= 0.0:
                root = math.sqrt(disc)
                cand = [t for t in ((-b - root) / (2.0 * a),
                                    (-b + root) / (2.0 * a)) if t > 0.05]
                if cand:
                    t_go = min(cand)
        if t_go is None:
            track = math.atan2(ry, rx)
            return _crab_heading(track, (wx, wy, 0.0), Va)
        # Ground aim = target position at intercept.
        aim_x = float(rt[0]) + float(vt[0]) * t_go
        aim_y = float(rt[1]) + float(vt[1]) * t_go
        # Air velocity that closes aim − r_i in t_go against the wind.
        ax = (aim_x - float(ri[0])) / t_go - wx
        ay = (aim_y - float(ri[1])) / t_go - wy
        if math.hypot(ax, ay) < 0.5:
            track = math.atan2(aim_y - float(ri[1]),
                               aim_x - float(ri[0]))
            return _crab_heading(track, (wx, wy, 0.0), Va)
        psi = math.atan2(ay, ax)
        return math.atan2(math.sin(psi), math.cos(psi))


class CrabPursuit(GuidanceLaw):
    """Pure pursuit on the ground LOS, crabbed into the wind.

    Same wind source as ``CrabPredictive``.
    """

    def __init__(self, wind=(0.0, 0.0, 0.0), airspeed: float = 50.0,
                 estimate_wind: bool = True):
        self.wind = np.asarray(wind, dtype=float)
        self.airspeed = float(airspeed)
        self.estimate_wind = bool(estimate_wind)
        self.reset()

    def reset(self) -> None:
        self.state = {"wind_est": np.array(self.wind, dtype=float, copy=True)}

    def set_wind_estimate(self, wind) -> None:
        self.state["wind_est"] = np.asarray(wind, dtype=float)

    def _wind_now(self) -> np.ndarray:
        return self.state["wind_est"]

    def desired_heading(self, ri, vi, rt, vt, dt=0.01) -> float:
        track = math.atan2(float(rt[1] - ri[1]), float(rt[0] - ri[0]))
        wind = self._wind_now()
        Va = self.airspeed if self.airspeed > 1.0 else 50.0
        return _crab_heading(track, wind, Va)


# ---------------------------------------------------------------------
#  Wind and link delay
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class WindField:
    """Steady wind plus a horizontal AR(1) gust.

    ``mean`` is the world-frame wind the air is moving toward (m/s).
    Interceptor ground velocity = air velocity + wind.  The target
    drifts with ``target_drift`` × wind (0 = holds ground track).
    Gusts are seeded from a separate RNG stream so the sensor-noise
    draw is unchanged when only the wind is switched on.
    """
    mean: tuple = (0.0, 0.0, 0.0)
    gust_rms: float = 0.0    # m/s, per horizontal component
    gust_tau: float = 2.0    # s, gust correlation time
    target_drift: float = 0.0

    @property
    def speed(self) -> float:
        return float(math.hypot(self.mean[0], self.mean[1]))

    def is_calm(self) -> bool:
        return self.speed < 1e-9 and self.gust_rms < 1e-9


# ---------------------------------------------------------------------
#  Sensor noise model
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class SensorNoise:
    """Gaussian sensor noise added to the interceptor's observations."""
    pos_sigma: float = 0.5   # m, position measurement σ (GPS-grade)
    vel_sigma: float = 0.2   # m/s, velocity measurement σ (radar/INS)


@dataclass(frozen=True)
class SensorCraft:
    """Non-expending ranger.  Holds a ground track.  Does not guide or kill.

    The fix is a bearing and a range from the craft's own GPS position.
    Cross-track error grows with range (``angle_sigma`` radians).
    Along-track error is ``range_sigma`` metres.  The interceptor fuses
    that fix with its own isotropic target fix.  Velocity still comes
    from the interceptor's own sensor.
    """
    pos0: tuple
    vel: tuple
    angle_sigma: float = 1.0e-3   # rad, ~1 mrad
    range_sigma: float = 1.0      # m
    link_sigma: float = 0.2       # m, cooperative baseline, not a second GPS pin

    def position(self, t: float) -> np.ndarray:
        return np.asarray(self.pos0, dtype=float) + np.asarray(self.vel, dtype=float) * t


def fuse_sensor_fix(own_rt, own_ri, own_sigma: float, craft: SensorCraft,
                    target_true, ri_true, t: float, rng) -> np.ndarray:
    """Fuse the own-ship relative fix with a ranging link, in xy.

    The craft sends target-relative-to-craft.  The interceptor adds the
    craft's position relative to itself, with ``link_sigma``.  Adding the
    craft's GPS pin instead cannot beat the interceptor's own GPS, so
    that handoff is not the measurement.

    Altitude stays on the own-ship fix.  The craft holds a ground track.
    """
    own_rt = np.asarray(own_rt, dtype=float)
    own_ri = np.asarray(own_ri, dtype=float)
    out = own_rt.copy()
    rs = craft.position(t)
    rel_ct = np.asarray(target_true, dtype=float) - rs
    rx, ry = float(rel_ct[0]), float(rel_ct[1])
    R = math.hypot(rx, ry)
    if R < 1.0:
        return out
    ux, uy = rx / R, ry / R
    along = R + float(rng.normal(0.0, craft.range_sigma))
    cross = float(rng.normal(0.0, R * craft.angle_sigma))
    meas_ct = np.array([along * ux - cross * uy, along * uy + cross * ux])
    baseline = (rs - np.asarray(ri_true, dtype=float))[:2]
    baseline = baseline + rng.normal(0.0, craft.link_sigma, 2)
    meas_rel = baseline + meas_ct
    own_rel = (own_rt - own_ri)[:2]
    sig_a = math.hypot(craft.range_sigma, craft.link_sigma)
    sig_c = math.hypot(R * craft.angle_sigma, craft.link_sigma)
    rot = np.array([[ux, -uy], [uy, ux]])
    p_s = rot @ np.diag([max(sig_a, 1e-3) ** 2, max(sig_c, 1e-3) ** 2]) @ rot.T
    sig_o = max(float(own_sigma) * math.sqrt(2.0), 1e-3)
    p_o = np.eye(2) * sig_o * sig_o
    prec = np.linalg.inv(p_o) + np.linalg.inv(p_s)
    p_f = np.linalg.inv(prec)
    fused = p_f @ (np.linalg.inv(p_o) @ own_rel + np.linalg.inv(p_s) @ meas_rel)
    out[0] = float(own_ri[0] + fused[0])
    out[1] = float(own_ri[1] + fused[1])
    return out


# ---------------------------------------------------------------------
#  Engagement configuration & result
# ---------------------------------------------------------------------
@dataclass
class EngagementConfig:
    interceptor: InterceptorDrone = field(default_factory=InterceptorDrone)
    warhead_type: WarheadType = WarheadType.FRAGMENTATION
    frag_warhead: FragmentationWarhead = field(default_factory=FragmentationWarhead)
    kinetic_warhead: KineticImpactWarhead = field(default_factory=KineticImpactWarhead)
    guidance_law: GuidanceLaw = field(default_factory=PurePursuit)
    sensor: SensorNoise = field(default_factory=SensorNoise)
    wind: WindField = field(default_factory=WindField)
    comm_delay: float = 0.0
    wind_est_sigma: float = 0.0
    sensor_craft: SensorCraft | None = None
    dt: float = 0.01
    max_time: float = 60.0
    store_trajectory: bool = True

    @property
    def fuse_range(self) -> float:
        """Proximity-fuse trigger distance (type-dependent)."""
        if self.warhead_type == WarheadType.KINETIC_IMPACT:
            return self.kinetic_warhead.effective_radius
        return self.frag_warhead.R_lethal


@dataclass
class EngagementResult:
    hit: bool = False
    kill: bool = False
    miss_distance: float = np.inf
    kill_probability: float = 0.0
    engagement_time: float = 0.0
    closing_speed: float = 0.0   # m/s, positive = closing
    trajectory_interceptor: list = field(default_factory=list)
    trajectory_target: list = field(default_factory=list)
    warhead_type: str = ""
    seed: int = 0
    approach_speed: float = 0.0  # |v_target − v_interceptor| at closest approach
    kinematic_check: dict = field(default_factory=dict)


# ---------------------------------------------------------------------
#  Kinematic sanity check
# ---------------------------------------------------------------------
def _kinematic_check(ti, tt, t_effect, dt, cfg,
                     min_idx=0, initial_closing=0.0):
    """Verify engagement time and speed against simple kinematics.

    Uses the *approach phase* (samples before the closest approach) so that
    the computed closing speed reflects the moment of closure, not the
    trajectory wandering after the engagement ended.
    """
    ti = np.array(ti)
    tt = np.array(tt)
    if len(ti) < 2:
        return {}
    R0 = float(np.linalg.norm(tt[0] - ti[0]))
    # Sample closing speed in a 1 s window BEFORE closest approach
    window_steps = max(10, int(1.0 / dt))
    end_idx = max(2, min(min_idx, len(ti) - 1))
    start_idx = max(0, end_idx - window_steps)
    sample_step = max(1, (end_idx - start_idx) // 20)
    cs_samples = []
    for i in range(start_idx, end_idx, sample_step):
        j = min(i + 10, len(tt) - 1)
        if i >= j:
            continue
        R_i = float(np.linalg.norm(tt[i] - ti[i]))
        R_j = float(np.linalg.norm(tt[j] - ti[j]))
        dt_sample = (j - i) * dt
        cs_samples.append(-(R_j - R_i) / dt_sample)
    if cs_samples:
        mean_cs = float(np.mean(cs_samples))
    else:
        mean_cs = initial_closing
    if mean_cs > 0:
        t_expected = R0 / mean_cs
        rel_err = (t_effect - t_expected) / max(t_expected, 1e-3)
        consistency = ("ok" if abs(rel_err) < 0.4 else "drift")
    else:
        t_expected = float('inf')
        consistency = "non-closing"
    return {
        "R0_m": round(R0, 1),
        "initial_closing_ms": round(initial_closing, 1),
        "approach_closing_ms": round(mean_cs, 1),
        "t_expected_s": (round(t_expected, 1)
                         if t_expected != float('inf') else "inf"),
        "t_actual_s": round(t_effect, 2),
        "consistency": consistency,
    }


def score_at_cpa(miss_distance: float, approach_speed: float,
                 warhead_type: WarheadType,
                 frag: FragmentationWarhead | None = None,
                 kinetic: KineticImpactWarhead | None = None) -> tuple[bool, float]:
    """Score a warhead at the geometric closest approach.

    ``miss_distance`` is the impact parameter.  Kinetic energy uses
    ``approach_speed`` = |v_rel| at that instant.  The line-of-sight
    component of relative velocity is ~0 at closest approach by
    definition, so it is not the energy of the hit.
    """
    frag = frag or FragmentationWarhead()
    kinetic = kinetic or KineticImpactWarhead()
    if not math.isfinite(miss_distance):
        return False, 0.0
    if warhead_type == WarheadType.KINETIC_IMPACT:
        if miss_distance > kinetic.effective_radius:
            return False, 0.0
        return True, kinetic.kill_probability(miss_distance, approach_speed)
    if miss_distance > frag.R_lethal:
        return False, 0.0
    return True, frag.kill_probability(miss_distance)


class TargetSwitcher:
    """Pick which target the interceptor is guiding on.

    ``select`` returns an index into the target list.  ``Fixed`` never
    moves.  ``Nearest`` follows range.  ``BestPk`` follows the predicted
    kinetic kill of a straight-line continuation (impact parameter and
    closing speed).  ``FirstLock`` picks once at t = 0.
    """

    def reset(self) -> None:
        self.state = {"locked": 0}

    def select(self, targets, ri, vi, t, dt) -> int:
        raise NotImplementedError


class FixedSwitch(TargetSwitcher):
    def __init__(self, index: int = 0):
        self.index = int(index)

    def select(self, targets, ri, vi, t, dt) -> int:
        return self.index


class FirstLockSwitch(TargetSwitcher):
    def select(self, targets, ri, vi, t, dt) -> int:
        if "locked" not in self.state:
            best = min(
                range(len(targets)),
                key=lambda i: float(np.linalg.norm(targets[i][0] - ri)))
            self.state["locked"] = best
        return self.state["locked"]


class NearestSwitch(TargetSwitcher):
    def __init__(self, rescore_s: float = 0.5):
        self.rescore_s = float(rescore_s)

    def select(self, targets, ri, vi, t, dt) -> int:
        last = self.state.get("last_t", -1e9)
        if t - last < self.rescore_s and "locked" in self.state:
            return self.state["locked"]
        self.state["last_t"] = t
        self.state["locked"] = min(
            range(len(targets)),
            key=lambda i: float(np.linalg.norm(targets[i][0] - ri)))
        return self.state["locked"]


class BestPkSwitch(TargetSwitcher):
    """Switch to the target a straight-line continuation kills best.

    Impact parameter of the closing triangle is |r × v| / |v| and the
    energy scales with |v|.  Pk proxy = exp(−(b/3)²) · min(|v|/150, 1).
    """

    def __init__(self, rescore_s: float = 0.5, margin: float = 0.05):
        self.rescore_s = float(rescore_s)
        self.margin = float(margin)

    def _proxy(self, rt, vt, ri, vi) -> float:
        r = np.asarray(rt, dtype=float)[:2] - np.asarray(ri, dtype=float)[:2]
        v = np.asarray(vt, dtype=float)[:2] - np.asarray(vi, dtype=float)[:2]
        speed = float(np.linalg.norm(v))
        if speed < 1e-3:
            return 0.0
        b = abs(r[0] * v[1] - r[1] * v[0]) / speed
        return math.exp(-((b / 3.0) ** 2)) * min(speed / 150.0, 1.0)

    def select(self, targets, ri, vi, t, dt) -> int:
        last = self.state.get("last_t", -1e9)
        if t - last < self.rescore_s and "locked" in self.state:
            return self.state["locked"]
        self.state["last_t"] = t
        scores = [self._proxy(rt, vt, ri, vi) for rt, vt, _ in targets]
        best = int(np.argmax(scores))
        cur = self.state.get("locked", best)
        if scores[best] > scores[cur] + self.margin:
            self.state["locked"] = best
        else:
            self.state["locked"] = cur
        return self.state["locked"]


def run_multi_target(targets, interceptor_pos0, interceptor_vel0,
                     cfg: EngagementConfig, switcher: TargetSwitcher,
                     seed: int = 0):
    """Fly against several targets with a mid-course switcher.

    ``targets`` is a list of ``(pos0, vel0, heading_fn_or_None)``.
    Guidance sees only the selected target's observation.  Returns a
    dict with per-target closest-approach miss, the final selection,
    and the number of switches.  Only the selected target is scored
    for kill, matching a one-shot interceptor.
    """
    rng = np.random.default_rng(seed)
    rng_gust = np.random.default_rng(seed + 991)
    dt = cfg.dt
    intr = cfg.interceptor
    sn = cfg.sensor
    wind_cfg = cfg.wind
    law = cfg.guidance_law
    if hasattr(law, "reset"):
        law.reset()
    if hasattr(switcher, "reset"):
        switcher.reset()

    ri = np.array(interceptor_pos0, dtype=float) + rng.normal(0, sn.pos_sigma, 3)
    vi = np.array(interceptor_vel0, dtype=float) + rng.normal(0, sn.vel_sigma, 3)
    ri[2] = max(ri[2], 5.0)
    launch_speed = float(np.linalg.norm(vi))
    v_max = intr.v_cruise if intr.v_cruise > 1.0 else launch_speed
    if launch_speed > 1e-3:
        vi = vi * (v_max / launch_speed)

    tstates = []
    for pos0, vel0, hdg_fn in targets:
        rt = np.array(pos0, dtype=float) + rng.normal(0, sn.pos_sigma, 3)
        vt = np.array(vel0, dtype=float) + rng.normal(0, sn.vel_sigma, 3)
        rt[2] = max(rt[2], 5.0)
        tstates.append({"r": rt, "v": vt, "hdg": hdg_fn,
                        "min": float(np.linalg.norm(rt - ri)),
                        "approach": float(np.linalg.norm(vt - vi))})

    wind_mean = np.asarray(wind_cfg.mean, dtype=float)
    gust = np.zeros(3)

    def _wind_vec():
        nonlocal gust
        if wind_cfg.gust_rms > 1e-9:
            phi = math.exp(-dt / max(wind_cfg.gust_tau, 1e-3))
            sig = wind_cfg.gust_rms * math.sqrt(max(0.0, 1.0 - phi * phi))
            gust = phi * gust + sig * rng_gust.normal(0.0, 1.0, 3)
            gust[2] = 0.0
        return wind_mean + gust

    bank = 0.0
    heading_i = math.atan2(float(vi[1]), float(vi[0]))
    pitch_i = math.atan2(float(vi[2]),
                          math.hypot(float(vi[0]), float(vi[1])))
    delay_steps = int(round(max(cfg.comm_delay, 0.0) / dt))
    obs_hist = deque(maxlen=delay_steps + 1)
    sel = 0
    switches = 0
    t = 0.0
    # Hold the launch altitude.  A pitch command aimed at the target's
    # elevation is ~0 at long range, so gravity drops the airframe
    # before the crossing and every policy records a ~40 m miss.
    intr_alt0 = float(ri[2])
    while t < cfg.max_time:
        wind_now = _wind_vec()
        sel = int(switcher.select(
            [(s["r"], s["v"], None) for s in tstates], ri, vi, t, dt))
        if sel != getattr(switcher.state.get("shown", None), "x", sel):
            pass
        st = tstates[sel]
        rt, vt = st["r"], st["v"]
        ri_obs = ri + rng.normal(0, sn.pos_sigma, 3)
        rt_obs = rt + rng.normal(0, sn.pos_sigma, 3)
        vi_obs = vi + wind_now + rng.normal(0, sn.vel_sigma, 3)
        vt_obs = vt + wind_now * wind_cfg.target_drift \
            + rng.normal(0, sn.vel_sigma, 3)
        for s in tstates:
            R = float(np.linalg.norm(s["r"] - ri))
            if R < s["min"]:
                s["min"] = R
                s["approach"] = float(np.linalg.norm(
                    s["v"] + wind_now * wind_cfg.target_drift
                    - (vi + wind_now)))
        obs_hist.append((ri_obs, vi_obs, rt_obs, vt_obs))
        ri_g, vi_g, rt_g, vt_g = obs_hist[0]
        if hasattr(law, "set_wind_estimate"):
            if getattr(law, "estimate_wind", True):
                speed_air = max(float(np.linalg.norm(vi)), 1.0)
                u_psi = np.array(
                    [math.cos(heading_i), math.sin(heading_i), 0.0])
                w_hat = (vi + wind_now) - speed_air * u_psi
                if cfg.wind_est_sigma > 1e-9:
                    w_hat = w_hat + rng.normal(0.0, cfg.wind_est_sigma, 3)
            else:
                w_hat = wind_now
            law.set_wind_estimate(w_hat)
        desired_heading = law.desired_heading(ri_g, vi_g, rt_g, vt_g, dt=dt)
        desired_heading = math.atan2(math.sin(desired_heading),
                                      math.cos(desired_heading))
        err = desired_heading - heading_i
        err = math.atan2(math.sin(err), math.cos(err))
        bank_cmd = float(np.clip(err * 3.0, -intr.max_bank, intr.max_bank))
        bank += (bank_cmd - bank) * (1 - math.exp(-dt / intr.bank_tau))
        turn_rate = 9.81 * math.tan(bank) / max(
            max(float(np.linalg.norm(vi)), 5.0), 1.0)
        heading_i = heading_i + turn_rate * dt
        heading_i = math.atan2(math.sin(heading_i), math.cos(heading_i))
        alt_err = intr_alt0 - ri[2]
        pitch_cmd = float(np.clip(alt_err * 0.5 - vi[2] * 0.3,
                                   -math.radians(20), math.radians(20)))
        pitch_i += (pitch_cmd - pitch_i) * (1 - math.exp(-dt / 0.1))
        thrust_dir = np.array([
            math.cos(pitch_i) * math.cos(heading_i),
            math.cos(pitch_i) * math.sin(heading_i),
            math.sin(pitch_i)])
        speed = float(np.linalg.norm(vi))
        vel_dir = vi / speed if speed > 1e-3 else thrust_dir
        drag = 0.5 * RHO * speed * speed * intr.S * intr.CD()
        force = (intr.thrust(speed, 1.0) * thrust_dir - drag * vel_dir
                 - intr.mass * G * np.array([0.0, 0.0, 1.0]))
        vi = vi + (force / intr.mass) * dt
        speed_new = float(np.linalg.norm(vi))
        if speed_new > v_max and speed_new > 1e-3:
            vi *= v_max / speed_new
        ri = ri + (vi + wind_now) * dt
        if ri[2] < 0.0:
            ri[2] = 0.0
            vi[2] = max(vi[2], 0.0)
        for s in tstates:
            hdg_fn = s["hdg"]
            vt = s["v"]
            if hdg_fn is not None:
                new_hdg = hdg_fn(t, s["r"], vt)
                spd = float(np.linalg.norm(vt))
                s["v"] = np.array([spd * math.cos(new_hdg),
                                   spd * math.sin(new_hdg), 0.0])
            s["r"] = s["r"] + (s["v"] + wind_now * wind_cfg.target_drift) * dt
            s["r"][2] = max(s["r"][2], 0.0)
        # count switches
        if getattr(switcher, "_last_sel", None) is not None \
                and sel != switcher._last_sel:
            switches += 1
        switcher._last_sel = sel
        t += dt

    finals = [{
        "miss": s["min"],
        "approach_speed": s["approach"],
    } for s in tstates]
    return {
        "miss_by_target": [f["miss"] for f in finals],
        "approach_by_target": [f["approach_speed"] for f in finals],
        "final_target": sel,
        "switches": switches,
        "engagement_time": t,
    }


# ---------------------------------------------------------------------
#  Engagement simulation (rewrite)
# ---------------------------------------------------------------------
def run_engagement(target_pos0, target_vel0,
                   interceptor_pos0, interceptor_vel0,
                   cfg: EngagementConfig, seed: int = 0,
                   target_heading_fn=None) -> EngagementResult:
    """Simulate a single engagement with sensor noise and jitter."""
    rng = np.random.default_rng(seed)
    # Gusts draw from a separate stream so switching wind on does not
    # redraw the sensor noise and break paired law comparisons.
    rng_gust = np.random.default_rng(seed + 991)
    dt = cfg.dt
    intr = cfg.interceptor
    sn = cfg.sensor
    wind_cfg = cfg.wind
    # Stateful guidance laws (PN, APN) keep LOS history; reset between trials
    if hasattr(cfg.guidance_law, "reset"):
        cfg.guidance_law.reset()

    # --- Apply initial-condition jitter (±σ each) ---
    # interceptor_vel0 is the air-relative launch vector.
    ri = np.array(interceptor_pos0, dtype=float) + rng.normal(0, sn.pos_sigma, 3)
    vi = np.array(interceptor_vel0, dtype=float) + rng.normal(0, sn.vel_sigma, 3)
    rt = np.array(target_pos0, dtype=float) + rng.normal(0, sn.pos_sigma, 3)
    vt = np.array(target_vel0, dtype=float) + rng.normal(0, sn.vel_sigma, 3)
    # Keep altitudes positive
    ri[2] = max(ri[2], 5.0)
    rt[2] = max(rt[2], 5.0)

    # --- Speed cap (cruise): the airframe has T/W ≈ 3 with drag far below
    # thrust, so without this cap the missile accelerates past its launch
    # speed and the engagement time collapses to half the kinematic value
    # R0 / |initial_closing|.  Holding speed at the launch command makes the
    # geometry close on the intended collision course.
    launch_speed = float(np.linalg.norm(vi))
    v_max = intr.v_cruise if intr.v_cruise > 1.0 else launch_speed
    if launch_speed > 1e-3:
        vi = vi * (v_max / launch_speed)

    wind_mean = np.asarray(wind_cfg.mean, dtype=float)
    gust = np.zeros(3)

    def _wind_vec():
        nonlocal gust
        if wind_cfg.gust_rms > 1e-9:
            phi = math.exp(-dt / max(wind_cfg.gust_tau, 1e-3))
            sig = wind_cfg.gust_rms * math.sqrt(max(0.0, 1.0 - phi * phi))
            gust = phi * gust + sig * rng_gust.normal(0.0, 1.0, 3)
            gust[2] = 0.0
        return wind_mean + gust

    wind_now = _wind_vec()
    vi_ground = vi + wind_now
    vt_ground = vt + wind_now * wind_cfg.target_drift

    # Initial closing speed along the line of sight (positive = closing)
    rel0 = rt - ri
    R0 = float(np.linalg.norm(rel0))
    initial_closing = -float(np.dot(vt_ground - vi_ground, rel0 / max(R0, 1e-6)))

    bank = 0.0
    heading_i = math.atan2(float(vi[1]), float(vi[0]))
    pitch_i = math.atan2(float(vi[2]),
                          math.hypot(float(vi[0]), float(vi[1])))
    intr_alt0 = float(ri[2])

    min_dist = R0
    min_dist_time = 0.0
    cs_at_min = initial_closing
    rel_speed_at_min = float(np.linalg.norm(vt_ground - vi_ground))
    warhead_triggered = False
    p_kill = 0.0
    t_effect = 0.0
    cpa_found = False

    traj_i = [ri.copy()]
    traj_t = [rt.copy()]
    min_idx = 0

    delay_steps = int(round(max(cfg.comm_delay, 0.0) / dt))
    obs_hist = deque(maxlen=delay_steps + 1)

    t = 0.0
    while t < cfg.max_time:
        wind_now = _wind_vec()
        vi_ground = vi + wind_now
        vt_ground = vt + wind_now * wind_cfg.target_drift

        # --- Sensor noise on observations (drawn at sample time) ---
        ri_obs = ri + rng.normal(0, sn.pos_sigma, 3)
        rt_obs = rt + rng.normal(0, sn.pos_sigma, 3)
        vi_obs = vi_ground + rng.normal(0, sn.vel_sigma, 3)
        vt_obs = vt_ground + rng.normal(0, sn.vel_sigma, 3)
        if cfg.sensor_craft is not None:
            rt_obs = fuse_sensor_fix(
                rt_obs, ri_obs, sn.pos_sigma, cfg.sensor_craft, rt, ri, t, rng)

        rel_pos = rt_obs - ri_obs
        R = float(np.linalg.norm(rt - ri))  # true distance for fuse

        # Track true minimum distance
        if R < min_dist:
            min_dist = R
            min_dist_time = t
            min_idx = len(traj_i)
            rel_vel_true = vt_ground - vi_ground
            los_true = (rt - ri) / max(R, 1e-6)
            cs_at_min = -float(np.dot(rel_vel_true, los_true))
            rel_speed_at_min = float(np.linalg.norm(rel_vel_true))

        # --- Guidance sees a delayed observation (stale + noisy) ---
        obs_hist.append((ri_obs, vi_obs, rt_obs, vt_obs))
        ri_g, vi_g, rt_g, vt_g = obs_hist[0]
        # Air-data + INS wind estimate: ground velocity minus airspeed
        # along the actual nose.  Optional error stands in for heading /
        # airspeed calibration.
        if hasattr(cfg.guidance_law, "set_wind_estimate"):
            if getattr(cfg.guidance_law, "estimate_wind", True):
                speed_air = max(float(np.linalg.norm(vi)), 1.0)
                u_psi = np.array(
                    [math.cos(heading_i), math.sin(heading_i), 0.0])
                w_hat = (vi + wind_now) - speed_air * u_psi
                if cfg.wind_est_sigma > 1e-9:
                    w_hat = w_hat + rng.normal(0.0, cfg.wind_est_sigma, 3)
            else:
                w_hat = wind_now
            cfg.guidance_law.set_wind_estimate(w_hat)
        desired_heading = cfg.guidance_law.desired_heading(
            ri_g, vi_g, rt_g, vt_g, dt=dt)
        # Wrap to [-pi, pi]
        desired_heading = math.atan2(math.sin(desired_heading),
                                      math.cos(desired_heading))

        speed_i = float(np.linalg.norm(vi))
        if speed_i < 1.0:
            speed_i = 1.0

        # Heading error and bank servo
        heading_err = desired_heading - heading_i
        heading_err = math.atan2(math.sin(heading_err), math.cos(heading_err))

        # Bank servo
        bank_cmd = float(np.clip(heading_err * 3.0, -intr.max_bank, intr.max_bank))
        bank += (bank_cmd - bank) * (1 - math.exp(-dt / intr.bank_tau))

        heading_rate = G * math.tan(bank) / speed_i
        # max_accel is the lateral-g cap.  At the default 80° bank the
        # geometric accel (~5.7 g) is below the 80 m/s² field, so the
        # cap does not change the 50 m/s drone.
        rate_cap = intr.max_accel / speed_i
        if abs(heading_rate) > rate_cap:
            heading_rate = math.copysign(rate_cap, heading_rate)
        heading_i += heading_rate * dt
        heading_i = math.atan2(math.sin(heading_i), math.cos(heading_i))

        # Altitude hold
        alt_err = intr_alt0 - ri[2]
        desired_pitch = float(np.clip(alt_err * 0.5 - vi[2] * 0.3,
                                       -math.radians(20), math.radians(20)))
        pitch_i += (desired_pitch - pitch_i) * (1 - math.exp(-dt / 0.1))

        # Forces
        T = intr.thrust(speed_i, 1.0)
        CD = intr.CD()
        rho = RHO
        q = 0.5 * rho * speed_i ** 2
        D = q * intr.S * CD

        thrust_dir = np.array([
            math.cos(pitch_i) * math.cos(heading_i),
            math.cos(pitch_i) * math.sin(heading_i),
            math.sin(pitch_i)])
        vel_dir = vi / speed_i
        force = T * thrust_dir - D * vel_dir - intr.mass * G * np.array([0, 0, 1])

        vi = vi + (force / intr.mass) * dt
        # Cruise cap on airspeed.  Without it the acceleration from T_max
        # (150 N) dwarfs drag (≈10 N at 50 m/s) and the missile reaches
        # ~160 m/s in 5 s, doubling the closure rate.
        speed_new = float(np.linalg.norm(vi))
        if speed_new > v_max and speed_new > 1e-3:
            vi *= v_max / speed_new
        # Ground track = air velocity + wind (guidance has no wind estimate)
        ri = ri + (vi + wind_now) * dt
        if ri[2] < 0.0:
            ri[2] = 0.0; vi[2] = max(vi[2], 0.0)

        # Target holds its commanded ground track (multirotor).
        if target_heading_fn is not None:
            new_hdg = target_heading_fn(t, rt, vt)
            spd = float(np.linalg.norm(vt))
            vt = np.array([spd * math.cos(new_hdg),
                           spd * math.sin(new_hdg), 0.0])
        rt = rt + (vt + wind_now * wind_cfg.target_drift) * dt
        rt[2] = max(rt[2], 0.0)

        if cfg.store_trajectory:
            traj_i.append(ri.copy())
            traj_t.append(rt.copy())

        # Score at closest approach.  Ending on first entry of the
        # collision sphere records a miss of ≈ the sphere radius for
        # every intercept, so a central hit and a graze look the same.
        passed_cpa = (t > 0.05
                      and t > min_dist_time + 0.02
                      and R > min_dist + 0.3)
        if passed_cpa:
            cpa_found = True
            t_effect = min_dist_time
            warhead_triggered, p_kill = score_at_cpa(
                min_dist, rel_speed_at_min, cfg.warhead_type,
                cfg.frag_warhead, cfg.kinetic_warhead)
            break

        t += dt

    kill = bool(rng.random() < p_kill) if warhead_triggered else False

    kc = _kinematic_check(traj_i, traj_t, t_effect, dt, cfg,
                          min_idx=min_idx, initial_closing=initial_closing)

    return EngagementResult(
        hit=warhead_triggered,
        kill=kill,
        miss_distance=min_dist,
        kill_probability=p_kill,
        engagement_time=t_effect if cpa_found else t,
        closing_speed=cs_at_min,
        trajectory_interceptor=traj_i,
        trajectory_target=traj_t,
        warhead_type=cfg.warhead_type.value,
        seed=seed,
        approach_speed=rel_speed_at_min,
        kinematic_check=kc,
    )


# ---------------------------------------------------------------------
#  Scenario generators
# ---------------------------------------------------------------------
def scenario_head_on(distance=1000.0, target_speed=15.0,
                     interceptor_speed=50.0, altitude=50.0):
    return (
        np.array([distance, 0.0, altitude]),
        np.array([-target_speed, 0.0, 0.0]),
        np.array([0.0, 0.0, altitude]),
        np.array([interceptor_speed, 0.0, 0.0]),
    )


def scenario_tail_chase(distance=1000.0, target_speed=15.0,
                        interceptor_speed=50.0, altitude=50.0):
    return (
        np.array([distance, 0.0, altitude]),
        np.array([target_speed, 0.0, 0.0]),
        np.array([0.0, 0.0, altitude]),
        np.array([interceptor_speed, 0.0, 0.0]),
    )


def scenario_beam(distance=1000.0, target_speed=15.0,
                  interceptor_speed=50.0, altitude=50.0):
    """Beam: interceptor starts ahead + to the side, heading toward
    the target's projected path."""
    tp = np.array([distance, 0.0, altitude])
    tv = np.array([target_speed, 0.0, 0.0])
    ip = np.array([distance * 0.5, 100.0, altitude])
    angle = math.atan2(-100.0, distance * 0.5)
    iv = np.array([interceptor_speed * math.cos(angle),
                    interceptor_speed * math.sin(angle), 0.0])
    return tp, tv, ip, iv


def scenario_evasive(distance=1000.0, target_speed=15.0,
                     interceptor_speed=50.0, altitude=50.0):
    return scenario_head_on(distance, target_speed, interceptor_speed, altitude)


def evasive_heading_fn(jink_freq=0.5, jink_amp=30.0, seed=0):
    rng = np.random.default_rng(seed)
    jink_times = sorted(rng.uniform(2, 30, size=5))
    jink_dirs = rng.choice([-1, 1], size=5) * rng.uniform(15, jink_amp, size=5)

    def heading_fn(t, pos, vel):
        base_hdg = math.atan2(float(vel[1]), float(vel[0]))
        for jt, jd in zip(jink_times, jink_dirs):
            if jt < t < jt + 2.0:
                return base_hdg + math.radians(jd) * math.sin(
                    2 * math.pi * jink_freq * (t - jt))
        return base_hdg
    return heading_fn


def sustained_heading_fn(jink_freq=0.35, jink_amp=25.0, seed=0):
    """Continuous weave for the whole engagement (not 2 s bursts).

    Amplitude and frequency are drawn once per trial so the paired
    comparison shares the same weave across laws.
    """
    rng = np.random.default_rng(seed)
    phase = float(rng.uniform(0.0, 2.0 * math.pi))
    amp = float(rng.uniform(0.6, 1.0) * jink_amp)
    freq = float(jink_freq * rng.uniform(0.8, 1.2))

    def heading_fn(t, pos, vel):
        base_hdg = math.atan2(float(vel[1]), float(vel[0]))
        return base_hdg + math.radians(amp) * math.sin(
            2.0 * math.pi * freq * t + phase)
    return heading_fn


# ---------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Interceptor Drone ===")
    intr = InterceptorDrone()
    print(f"  Mass: {intr.mass} kg, T_max: {intr.T_max} N, T/W: {intr.T_max/intr.W:.1f}")

    print("\n=== Fragmentation Warhead ===")
    fw = FragmentationWarhead()
    for r in [1, 2, 3, 4, 5, 6, 8, 10]:
        print(f"  P_kill(r={r:2d}m) = {fw.kill_probability(r):.4f}")

    print("\n=== Kinetic Impact Warhead ===")
    kw = KineticImpactWarhead()
    for v in [10, 20, 30, 40, 50, 60]:
        print(f"  KE(V={v}m/s) = {kw.kinetic_energy(v):.0f} J, "
              f"P_kill(r=1m) = {kw.kill_probability(1.0, v):.3f}, "
              f"P_kill(r=3m) = {kw.kill_probability(3.0, v):.3f}")

    print("\n=== Head-on (fragmentation) ===")
    cfg = EngagementConfig(warhead_type=WarheadType.FRAGMENTATION)
    tp, tv, ip, iv = scenario_head_on()
    res = run_engagement(tp, tv, ip, iv, cfg, seed=42)
    print(f"  Hit: {res.hit}, Kill: {res.kill}, Miss: {res.miss_distance:.2f} m, "
          f"Pk: {res.kill_probability:.3f}, Time: {res.engagement_time:.2f} s, "
          f"CS: {res.closing_speed:.1f} m/s")
    print(f"  Kinematics: {res.kinematic_check}")

    print("\n=== Head-on (kinetic) ===")
    cfg2 = EngagementConfig(warhead_type=WarheadType.KINETIC_IMPACT)
    res2 = run_engagement(tp, tv, ip, iv, cfg2, seed=42)
    print(f"  Hit: {res2.hit}, Kill: {res2.kill}, Miss: {res2.miss_distance:.2f} m, "
          f"Pk: {res2.kill_probability:.3f}, Time: {res2.engagement_time:.2f} s, "
          f"CS: {res2.closing_speed:.1f} m/s")
    print(f"  Kinematics: {res2.kinematic_check}")
