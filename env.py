"""RL environment wrapping the dynamics simulator.

Two scenarios:
    * 'takeoff'  : start on water; goal = reach z > z_goal with Vx > V_lo
    * 'landing'  : start in air at altitude; goal = touch down with low |Vz|

The agent issues two continuous actions:
    throttle in [0, 1]
    pitch   in [pitch_lo, pitch_hi]  (radians)

State vector (8 + n_preview features, all normalised):
    0  z      / z_scale
    1  Vx     / V_scale
    2  Vz     / V_scale
    3  eta(x,t) / eta_scale
    4  Veta(x,t) / eta_scale * T_scale
    5  hull_in_water (0 or 1)
    6  Vx / V_stall  (takeoff) OR (1 - z/z_goal) (landing)
    7  throttle (memory of last action)
    8+ encounter-time eta at preview_dx_m ahead / eta_scale
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
import numpy as np

from atmosphere import Atmosphere, AtmosphereConfig
from aircraft import Aircraft
from ocean import Ocean
from ocean_directional import DirectionalOcean, PREVIEW_DX_M, sea_eta_1d, wave_preview
from dynamics import HullContact, HullDrag, hull_force


# ---------------------------------------------------------------------
@dataclass
class EnvConfig:
    spatial: bool = False
    atmosphere: AtmosphereConfig = field(default_factory=AtmosphereConfig)
    scenario: str = "takeoff"          # "takeoff" or "landing"
    dt: float = 0.05                   # control step
    max_steps: int = 200
    z_goal: float = 8.0                # takeoff target altitude
    V_lo: float = 8.5                  # takeoff liftoff speed (m/s)
    approach_alt: float = 25.0         # landing initial altitude
    V_approach: float = 13.0           # landing approach speed
    pitch_lo: float = math.radians(-8)
    pitch_hi: float = math.radians(12)
    Hs: float = 1.5
    Tp: float = 6.0
    wave_seed: int = 42
    directional: bool = False       # use DirectionalOcean longitudinal slice
    theta_mean_deg: float = 0.0     # mean wave direction (deg, CCW from +x)
    spread_s: int = 10              # cos^{2s} spreading; larger = narrower
    preview_dx_m: tuple = PREVIEW_DX_M  # upstream encounter-time look-aheads (m)


class FlyingBoatEnv:
    """Gymnasium-like env:  reset() -> state,  step(action) -> (state, r, done, info)."""

    metadata = {"render.modes": []}

    def __init__(self, ac: Aircraft, cfg: EnvConfig | None = None):
        self.ac = ac
        self.cfg = cfg or EnvConfig()
        dx = np.asarray(self.cfg.preview_dx_m, dtype=float)
        if (self.cfg.scenario not in ("takeoff", "landing")
                or not np.isfinite(self.cfg.dt) or self.cfg.dt <= 0
                or self.cfg.max_steps <= 0 or self.cfg.z_goal <= 0
                or not np.isfinite(self.cfg.theta_mean_deg)
                or not isinstance(self.cfg.spread_s, (int, np.integer))
                or self.cfg.spread_s < 1
                or dx.ndim != 1 or not np.isfinite(dx).all()
                or np.any(dx <= 0)):
            raise ValueError("invalid environment configuration")
        if not self.cfg.spatial and self.cfg.atmosphere != AtmosphereConfig():
            raise ValueError("atmosphere configuration requires spatial=True")
        self.action_dim = 4 if self.cfg.spatial else 2
        self.state_dim = 8 + int(dx.size) + (8 if self.cfg.spatial else 0)
        self._build_state_scales()
        self._sea = None
        self._state = None
        self._traj = []

    # ----- scaling helpers -----
    def _build_state_scales(self):
        self.z_scale   = 10.0                  # metres -> [-]
        self.V_scale   = 15.0                  # m/s  -> [-]
        self.eta_scale = 1.5                   # m    -> [-]
        self.T_scale   = 6.0                   # s    -> [-]
        self.W         = self.ac.W
        self.V_stall   = self.ac.V_stall
        self.hull = HullContact()
        self.hd = HullDrag(Bwl=self.ac.geom.Bwl, Lwl=self.ac.geom.Lwl)

    # ----- internal physics step -----
    # Explicit Euler substeps: at high dynamic pressure the aero forces
    # stiffen (a single 0.05 s step once ran away to 388 m/s in a dive),
    # so the control step is subdivided. Returned forces/surface are the
    # last-substep values; Veta stays full-step scale for the state vector.
    n_sub: int = 5

    def _physics_step(self, x, z, Vx, Vz, alpha, throttle, sea, t):
        if self.cfg.spatial:
            from spatial_dynamics import advance
            return advance(self, x, z, Vx, Vz, alpha, throttle, sea, t)
        eta = float(sea_eta_1d(sea, np.array([x]), t)[0])
        # Rate of change of eta (use central difference)
        eta_next = float(sea_eta_1d(sea, np.array([x]), t + self.cfg.dt)[0])
        Veta = (eta_next - eta) / self.cfg.dt

        h = self.cfg.dt / self.n_sub
        t_sub = t
        for _ in range(self.n_sub):
            eta_s = float(sea_eta_1d(sea, np.array([x]), t_sub)[0])
            V = math.hypot(Vx, Vz)
            T_i = self.ac.prop.thrust(V, throttle)
            q = 0.5 * 1.225 * V ** 2
            gamma = math.atan2(Vz, Vx)
            alpha_eff = alpha - gamma
            CL = self.ac.CL(alpha_eff)
            CD = self.ac.CD(CL)
            L_i = q * self.ac.geom.S * CL
            D_i = q * self.ac.geom.S * CD
            wf = hull_force(z, Vx, Vz, eta_s, self.hull)
            R_hull = self.hd.resistance(Vx) if wf.N > 0 else 0.0
            cos_a, sin_a = math.cos(alpha), math.sin(alpha)
            cos_g, sin_g = math.cos(gamma), math.sin(gamma)
            T_x = T_i * cos_a;  T_z = T_i * sin_a
            L_x = -L_i * sin_g; L_z = +L_i * cos_g
            D_x = -D_i * cos_g; D_z = -D_i * sin_g
            Fx = T_x + L_x + D_x + wf.Rt - math.copysign(R_hull, Vx)
            Fz = T_z + L_z + D_z + wf.N - self.W
            a_x = Fx / self.ac.mass.total
            a_z = Fz / self.ac.mass.total
            Vx = Vx + a_x * h
            Vz = Vz + a_z * h
            x = x + Vx * h
            z = z + Vz * h
            t_sub += h
        return x, z, Vx, Vz, eta, Veta, wf.N, T_i, L_i, D_i

    # ----- API -----
    def reset(self, seed: int | None = None):
        if seed is None:
            seed = self.cfg.wave_seed
        if self.cfg.directional:
            self._sea = DirectionalOcean(Hs=self.cfg.Hs, Tp=self.cfg.Tp,
                                         theta_mean=math.radians(self.cfg.theta_mean_deg),
                                         s=self.cfg.spread_s, seed=seed)
        else:
            self._sea = Ocean(Hs=self.cfg.Hs, Tp=self.cfg.Tp, seed=seed)
        self._y = self._Vy = self._bank = self._heading = 0.0
        self._bank_command = self._rudder_command = 0.0
        self._atmosphere = Atmosphere(self.cfg.atmosphere, seed=seed)
        self._t = 0.0
        if self.cfg.scenario == "takeoff":
            # Place hull at hydrostatic equilibrium on the wave surface
            delta = self.W / (1025.0 * 9.80665 * HullContact.A_wp)
            eta0 = float(sea_eta_1d(self._sea, np.array([0.0]), 0.0)[0])
            x = 0.0
            z = eta0 + self.hull.h_keel - delta
            Vx = 0.5; Vz = 0.0
            self._prev_throttle = 0.0
        elif self.cfg.scenario == "landing":
            gs = math.radians(8.0)
            x = 0.0
            z = self.cfg.approach_alt
            Vx = self.cfg.V_approach * math.cos(gs)
            Vz = -self.cfg.V_approach * math.sin(gs)
            self._prev_throttle = 0.05
        else:
            raise ValueError(self.cfg.scenario)
        self._x, self._z, self._Vx, self._Vz = x, z, Vx, Vz
        self._traj = []
        self._steps = 0
        self._done = False
        self._state = self._build_state()
        return self._state.copy()

    def _eta(self, x, t, y=None):
        if self.cfg.spatial and isinstance(self._sea, DirectionalOcean):
            return float(self._sea.eta([x], [self._y if y is None else y], t)[0, 0])
        return float(sea_eta_1d(self._sea, np.array([x]), t)[0])

    def _build_state(self):
        eta = self._eta(self._x, self._t)
        eta_next = self._eta(self._x, self._t + self.cfg.dt)
        Veta = (eta_next - eta) / self.cfg.dt
        hull_in_water = 1.0 if (eta + self.hull.h_keel > self._z) else 0.0
        if self.cfg.scenario == "takeoff":
            speed_metric = self._Vx / self.V_stall
        else:
            speed_metric = max(0.0, 1.0 - self._z / self.cfg.z_goal)
        preview = wave_preview(self._sea, self._x, self._t, self._Vx,
                               dxs=self.cfg.preview_dx_m) / self.eta_scale
        if self.cfg.spatial:
            speed = max(math.hypot(self._Vx, self._Vy), 1.0)
            direction = math.atan2(self._Vy, self._Vx)
            preview = np.array([self._eta(
                self._x + dx * math.cos(direction), self._t + dx / speed,
                self._y + dx * math.sin(direction))
                for dx in self.cfg.preview_dx_m]) / self.eta_scale
        s = np.concatenate((
            np.array([
                self._z / self.z_scale,
                self._Vx / self.V_scale,
                self._Vz / self.V_scale,
                eta / self.eta_scale,
                Veta / self.eta_scale * self.T_scale,
                hull_in_water,
                speed_metric,
                self._prev_throttle,
            ], dtype=np.float32),
            preview.astype(np.float32),
        ))
        if self.cfg.spatial:
            s = np.concatenate((s, np.array([
                self._y / 10.0, self._Vy / self.V_scale,
                self._bank / math.radians(45), math.sin(self._heading),
                math.cos(self._heading),
                *(self._atmosphere.wind(self._t) / self.V_scale),
            ], dtype=np.float32)))
        return s

    def step(self, action):
        if self._state is None or self._done:
            raise RuntimeError("reset() is required before stepping")
        action = np.asarray(action)
        if action.shape != (self.action_dim,) or not np.isfinite(action).all():
            raise ValueError(f"action must be a finite vector of shape ({self.action_dim},)")
        if self.cfg.spatial:
            self._bank_command = float(np.clip(action[2], -1, 1)) * math.radians(45)
            self._rudder_command = float(np.clip(action[3], -1, 1))
        throttle = float(np.clip(action[0], 0.0, 1.0))
        pitch    = float(action[1]) * (self.cfg.pitch_hi - self.cfg.pitch_lo)/2.0 \
                 + (self.cfg.pitch_hi + self.cfg.pitch_lo)/2.0
        pitch = float(np.clip(pitch, self.cfg.pitch_lo, self.cfg.pitch_hi))
        self._prev_throttle = throttle

        xn, zn, Vxn, Vzn, eta, Veta, N_water, T_i, L_i, D_i = \
            self._physics_step(self._x, self._z, self._Vx, self._Vz,
                               pitch, throttle, self._sea, self._t)
        self._t += self.cfg.dt
        self._x, self._z = xn, zn
        self._Vx, self._Vz = Vxn, Vzn
        self._steps += 1

        eta = self._eta(xn, self._t)
        Veta = float((self._eta(xn, self._t + self.cfg.dt) - eta) / self.cfg.dt)
        r, done, info = self._reward_and_done(eta, Veta, N_water,
                                              T_i, L_i, D_i)
        if self.cfg.spatial:
            r -= 0.01 * self._y ** 2 + 0.05 * self._Vy ** 2
            lateral_ok = abs(self._y) < 10 and abs(self._Vy) < 1.5 and abs(self._bank) < math.radians(10)
            if info.get("success") and not lateral_ok:
                info["success"] = False
                r -= 100.0
            if abs(self._y) > 50:
                done = True
                info.update(success=False, lateral_limit=True)
                r -= 100.0
        self._done = done
        info.update({"t": self._t, "x": self._x, "z": self._z,
                     "Vx": self._Vx, "Vz": self._Vz,
                     "throttle": throttle, "alpha": pitch,
                     "eta": eta, "Veta": Veta, "N_water": N_water,
                     "T": T_i, "L": L_i, "D": D_i})
        if self.cfg.spatial:
            wind = self._atmosphere.wind(self._t)
            info.update(force_budget=dict(self._last_force_budget),
                        y=self._y, Vy=self._Vy, bank=self._bank,
                        heading=self._heading, wind=wind.tolist(),
                        airspeed=float(np.linalg.norm(
                            np.array([self._Vx, self._Vy, self._Vz]) - wind)),
                        density=self._atmosphere.density(self._z))
        self._traj.append(info)
        self._state = self._build_state()
        return self._state.copy(), float(r), bool(done), info

    # ----- reward shaping -----
    def _reward_and_done(self, eta, Veta, N_water, T_i, L_i, D_i):
        cfg = self.cfg
        W = self.W
        info = {}

        if cfg.scenario == "takeoff":
            # Per-step dense reward: altitude + speed progress
            r_alt = 0.20 * self._z                    # m -> reward up to ~1.6
            r_spd = 0.05 * self._Vx                   # m/s -> reward up to ~1
            # comfort: penalise high impact & wave-induced motion
            r_comfort = (-0.002 * (N_water / W) ** 2
                         - 0.05 * (self._Vz ** 2))
            # tiny time penalty
            r_step = -0.05
            r = r_alt + r_spd + r_comfort + r_step
            done = False
            success = (self._z > cfg.z_goal) and (self._Vx > cfg.V_lo)
            if success:
                r += 50.0
                done = True
                info["success"] = True
            # crash: hull impact > 8 W or altitude collapses below sea level
            if N_water > 8 * W or self._z < -2.0:
                r -= 50.0
                done = True
                info["success"] = False
            if self._steps >= cfg.max_steps and not done:
                done = True
                info["success"] = False
                info["truncated"] = True
            return r, done, info

        # ----- landing -----
        # Per-step dense reward: gentle descent, getting closer to water.
        # The touchdown bonus below is scaled to dominate the background
        # accumulated over a ~30 s episode (previously +-30 vs ~-1300,
        # which made success invisible to the gradient).
        r_descent = -0.30 * (self._Vz ** 2) / 25.0     # punish sink rate
        r_prox    = -0.10 * self._z / cfg.z_goal        # encourage descent
        r_comfort = (-0.002 * (N_water / W) ** 2
                     - 0.05 * (self._Vz ** 2))
        r_step = -0.05
        # Flare guidance near the surface: reward killing sink rate low down
        r_flare = (0.5 * max(0.0, 1.0 - abs(self._Vz) / 1.5)
                   * max(0.0, 1.0 - self._z / 5.0)) if self._z < 5.0 else 0.0
        r = r_descent + r_prox + r_comfort + r_step + r_flare
        done = False
        # touchdown: first sample with hull in water
        hull_in_water = (eta + self.hull.h_keel) > self._z
        if hull_in_water:
            # reward soft touchdown (linear in |Vz|)
            r += 100.0 * max(0.0, 1.0 - abs(self._Vz) / 1.5)
            # bonus for low hull impact
            if N_water < 3 * W and abs(self._Vz) < 1.5:
                r += 100.0
                info["success"] = True
            else:
                r -= 100.0
                info["success"] = False
            done = True
        # catastrophic
        if N_water > 8 * W:
            r -= 100.0
            done = True
            info["success"] = False
        if self._steps >= cfg.max_steps and not done:
            done = True
            info["success"] = False
        return r, done, info

    def trajectory(self):
        return list(self._traj)


# ---------------------------------------------------------------------
if __name__ == "__main__":
    # Quick smoke test
    ac = Aircraft()
    env = FlyingBoatEnv(ac, EnvConfig(scenario="takeoff", max_steps=100))
    s = env.reset(seed=42)
    print("Initial state:", s)
    total_r = 0.0
    for t in range(50):
        a = np.array([0.9, 0.05], dtype=np.float32)   # full throttle, slight pitch up
        s, r, done, info = env.step(a)
        total_r += r
        if t % 10 == 0:
            print(f"  t={info['t']:.2f} z={info['z']:.2f} Vx={info['Vx']:.2f} "
                  f"N={info['N_water']:.0f} R={r:+.2f}")
        if done:
            break
    print(f"Total reward: {total_r:.1f}, success={info.get('success')}")