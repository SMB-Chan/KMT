"""Longitudinal dynamics of a flying-boat drone: water-taxi -> takeoff
-> climb, and  glide -> touchdown -> water-deceleration.

State vector (length 4):
    s = [x, z, Vx, Vz]
where x is horizontal ground-track position, z is altitude above
still water level (ASL), Vx horizontal speed, Vz vertical speed.

Forces:
    * Propulsion thrust along body axis (constant pitch alpha).
    * Aerodynamic lift & drag (free stream = sqrt(Vx^2 + Vz^2)).
    * Hydrodynamic contact force when hull touches water.
        - Normal force: linear spring + damper (only at contact).
        - Tangential drag: friction coefficient times normal force.
    * Gravity.

The hull/water contact is treated at a single point on the keel at
depth h_keel below the CG.  Wave elevation eta(x, t) is provided by
the Ocean model.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
import numpy as np

from aircraft import Aircraft, RHO, RHO_W, G
from ocean import Ocean


# ---------------------------------------------------------------------
#  Hull contact (spring-damper)
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class HullContact:
    h_keel:    float = 0.30    # keel depth below CG, m
    A_wp:      float = 0.55 * 2.6  # waterplane area, m^2
    Cv_drag:   float = 0.6     # vertical drag coefficient (added-mass-like)
    mu_s:      float = 0.05    # hull tangential friction coefficient


@dataclass
class WaterForce:
    N:  float = 0.0   # normal force (up), N
    Rt: float = 0.0   # tangential drag (back), N


def hull_force(z: float, Vx: float, Vz: float,
               eta: float, h: HullContact) -> WaterForce:
    """Vertical buoyancy + tangential friction.

    CG is at altitude z above mean sea level.  Keel at z_keel = z - h_keel.
    Wave surface at eta.  Penetration delta = eta + h_keel - z.

    Buoyancy uses Archimedes' principle:
        N = rho_w * g * A_wp * delta
    with a damping term that only acts when the hull is moving down
    (entering the water) -- avoids artificial suction.
    """
    delta = eta + h.h_keel - z
    if delta <= 0.0:
        return WaterForce()
    # Hydrostatic buoyancy (Archimedes)
    N = RHO_W * G * h.A_wp * delta
    # Quadratic vertical drag when entering water (Vz < 0)
    if Vz < 0.0:
        N_d = 0.5 * RHO_W * h.Cv_drag * h.A_wp * Vz ** 2
        N += N_d
    sign = -1.0 if Vx > 0 else (1.0 if Vx < 0 else 0.0)
    Rt = sign * h.mu_s * N
    return WaterForce(N=N, Rt=Rt)


# ---------------------------------------------------------------------
#  Hull drag in displacement regime (Froude-style hump)
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class HullDrag:
    Bwl:    float = 0.55   # waterline beam (from aircraft)
    Lwl:    float = 2.6    # waterline length
    Cf:     float = 0.005  # ITTC skin friction
    S_wet:  float = 2.5    # wetted surface area, m^2 (hull only)
    hump_Fr:    float = 0.45   # Froude number at resistance peak
    hump_Cv:    float = 0.18   # peak resistance coeff (R / 0.5 rho V^2 B^2)
    hump_width: float = 0.20   # Gaussian width in Fr
    Cv_planing: float = 0.012  # planing drag coeff (Savitsky beta ~0)

    def resistance(self, V: float) -> float:
        """Total hydrodynamic resistance.

        Skin friction + hump drag (Gaussian) + Savitsky planing drag.
        The hump and planing curves meet smoothly without singularities.
        """
        V = abs(V)
        Fr = V / math.sqrt(G * self.Lwl)
        # Skin friction (always present)
        R_f = self.Cf * 0.5 * RHO_W * V ** 2 * self.S_wet
        # Hump drag (Gaussian in Froude number)
        R_hump = (self.hump_Cv * 0.5 * RHO_W * V ** 2 * self.Bwl ** 2
                  * math.exp(- ((Fr - self.hump_Fr) / self.hump_width) ** 2))
        # Planing drag (Savitsky, low-deadrise)
        lam = V / math.sqrt(G * self.Bwl)
        if lam < 1.4:
            Cv = self.hump_Cv                  # pre-planing
        else:
            Cv = self.Cv_planing / math.sqrt(lam)
        R_planing = Cv * 0.5 * RHO_W * V ** 2 * self.Bwl ** 2
        return max(R_f, R_f + R_hump + R_planing)


# ---------------------------------------------------------------------
#  Takeoff / Landing simulator
# ---------------------------------------------------------------------
@dataclass
class SimResult:
    t:  np.ndarray
    x:  np.ndarray
    z:  np.ndarray
    Vx: np.ndarray
    Vz: np.ndarray
    T:  np.ndarray
    L:  np.ndarray
    D:  np.ndarray
    R:  np.ndarray        # total resistance (hull drag + aerodynamic)
    N_water: np.ndarray
    alpha: float
    throttle: float
    phase: np.ndarray      # 0 = water, 1 = airborne


def simulate_takeoff(ac: Aircraft, sea: Ocean,
                     duration: float = 12.0,
                     dt: float = 0.01,
                     throttle: float = 1.0,
                     alpha: float = math.radians(4.0),
                     start_x: float = 0.0,
                     x0: float = 0.0) -> SimResult:
    """Take-off run: water-taxi to climb-out at constant pitch and throttle."""
    hull = HullContact()
    hd = HullDrag(Bwl=ac.geom.Bwl, Lwl=ac.geom.Lwl)
    n = int(duration / dt) + 1
    t  = np.linspace(0.0, duration, n)
    x  = np.zeros(n); z  = np.zeros(n)
    Vx = np.zeros(n); Vz = np.zeros(n)
    T  = np.zeros(n); L = np.zeros(n)
    D  = np.zeros(n); R_ = np.zeros(n)
    Nw = np.zeros(n); ph = np.zeros(n)

    # initial state -- CG positioned at hydrostatic equilibrium draft ON
    # the local wave surface (so the hull starts in water, not in air).
    delta_static = ac.W / (RHO_W * G * HullContact.A_wp)
    eta0 = float(sea.eta(np.array([start_x]), 0.0)[0])
    z_init = eta0 + hull.h_keel - delta_static
    x[0] = start_x
    z[0] = z_init
    Vx[0] = 0.5
    Vz[0] = 0.0

    for i in range(n - 1):
        # Wave elevation under CG
        eta = float(sea.eta(np.array([x[i]]), t[i])[0])
        V = math.hypot(Vx[i], Vz[i])
        # Forces
        T_i = ac.prop.thrust(V, throttle)
        q = 0.5 * RHO * V ** 2
        # Flight-path angle and effective angle of attack
        gamma = math.atan2(Vz[i], Vx[i])
        alpha_eff = alpha - gamma
        CL = ac.CL(alpha_eff)
        CD = ac.CD(CL)
        L_i = q * ac.geom.S * CL
        D_i = q * ac.geom.S * CD
        # Hydrodynamic (hull) reaction
        wf = hull_force(z[i], Vx[i], Vz[i], eta, hull)
        R_hull = hd.resistance(Vx[i]) if wf.N > 0 else 0.0
        # Thrust body-axis aligned; L, D velocity-vector aligned
        cos_a, sin_a = math.cos(alpha), math.sin(alpha)
        cos_g, sin_g = math.cos(gamma), math.sin(gamma)
        # Body-axis thrust
        T_x = T_i * cos_a
        T_z = T_i * sin_a
        # Lift perpendicular to velocity; drag along -velocity
        L_x = -L_i * sin_g
        L_z = +L_i * cos_g
        D_x = -D_i * cos_g
        D_z = -D_i * sin_g
        # Sum forces (inertial x, z)
        Fx = T_x + L_x + D_x + wf.Rt - math.copysign(R_hull, Vx[i])
        Fz = T_z + L_z + D_z + wf.N - ac.W
        a_x = Fx / ac.mass.total
        a_z = Fz / ac.mass.total
        # Integration (forward Euler)
        Vx[i + 1] = Vx[i] + a_x * dt
        Vz[i + 1] = Vz[i] + a_z * dt
        x[i + 1]  = x[i]  + Vx[i + 1] * dt
        z[i + 1]  = z[i]  + Vz[i + 1] * dt

        # Logging
        T[i + 1]  = T_i
        L[i + 1]  = L_i
        D[i + 1]  = D_i
        R_[i + 1] = R_hull + D_i
        Nw[i + 1] = wf.N
        ph[i + 1] = 1 if z[i + 1] - hull.h_keel > eta + 0.05 else 0

    # Set first sample
    T[0]  = ac.prop.thrust(Vx[0], throttle)
    q0 = 0.5 * RHO * Vx[0] ** 2
    gamma0 = math.atan2(Vz[0], Vx[0])
    L[0] = q0 * ac.geom.S * ac.CL(alpha - gamma0)
    D[0] = q0 * ac.geom.S * ac.CD(L[0] / (q0 * ac.geom.S))
    eta0 = float(sea.eta(np.array([x[0]]), t[0])[0])
    wf0 = hull_force(z[0], Vx[0], Vz[0], eta0, hull)
    Nw[0] = wf0.N
    R_[0] = hd.resistance(Vx[0]) if wf0.N > 0 else 0.0
    ph[0] = 0

    return SimResult(t=t, x=x, z=z, Vx=Vx, Vz=Vz,
                     T=T, L=L, D=D, R=R_, N_water=Nw,
                     alpha=alpha, throttle=throttle, phase=ph)


def simulate_landing(ac: Aircraft, sea: Ocean,
                     approach_alt: float = 30.0,
                     approach_speed: float = 13.0,
                     glide_slope: float = math.radians(8.0),
                     alpha_body: float = math.radians(-5.0),
                     throttle: float = 0.05,
                     duration: float = 25.0,
                     dt: float = 0.001) -> SimResult:
    """Approach + touchdown + water-run deceleration.

    Initial conditions are set so that the flight path makes the
    requested glide slope below horizontal.  The body pitch alpha_body
    is fixed (no control loop) -- pick it so that L ~ W at the chosen
    approach speed.
    """
    hull = HullContact()
    hd = HullDrag(Bwl=ac.geom.Bwl, Lwl=ac.geom.Lwl)
    n = int(duration / dt) + 1
    t  = np.linspace(0.0, duration, n)
    x  = np.zeros(n); z  = np.zeros(n)
    Vx = np.zeros(n); Vz = np.zeros(n)
    T  = np.zeros(n); L = np.zeros(n)
    D  = np.zeros(n); R_ = np.zeros(n)
    Nw = np.zeros(n); ph = np.zeros(n)

    x[0]  = 0.0
    z[0]  = approach_alt
    Vx[0] = approach_speed * math.cos(glide_slope)
    Vz[0] = -approach_speed * math.sin(glide_slope)
    alpha = alpha_body

    for i in range(n - 1):
        eta = float(sea.eta(np.array([x[i]]), t[i])[0])
        V = math.hypot(Vx[i], Vz[i])
        T_i = ac.prop.thrust(V, throttle)
        q = 0.5 * RHO * V ** 2
        gamma = math.atan2(Vz[i], Vx[i])
        alpha_eff = alpha - gamma
        CL = ac.CL(alpha_eff)
        CD = ac.CD(CL)
        L_i = q * ac.geom.S * CL
        D_i = q * ac.geom.S * CD
        wf = hull_force(z[i], Vx[i], Vz[i], eta, hull)
        R_hull = hd.resistance(Vx[i]) if wf.N > 0 else 0.0
        # Thrust body-axis; L, D velocity-vector aligned
        cos_a, sin_a = math.cos(alpha), math.sin(alpha)
        cos_g, sin_g = math.cos(gamma), math.sin(gamma)
        T_x = T_i * cos_a
        T_z = T_i * sin_a
        L_x = -L_i * sin_g
        L_z = +L_i * cos_g
        D_x = -D_i * cos_g
        D_z = -D_i * sin_g
        Fx = T_x + L_x + D_x + wf.Rt - math.copysign(R_hull, Vx[i])
        Fz = T_z + L_z + D_z + wf.N - ac.W
        a_x = Fx / ac.mass.total
        a_z = Fz / ac.mass.total
        Vx[i + 1] = Vx[i] + a_x * dt
        Vz[i + 1] = Vz[i] + a_z * dt
        x[i + 1]  = x[i]  + Vx[i + 1] * dt
        z[i + 1]  = z[i]  + Vz[i + 1] * dt

        T[i + 1]  = T_i
        L[i + 1]  = L_i
        D[i + 1]  = D_i
        R_[i + 1] = R_hull + D_i
        Nw[i + 1] = wf.N
        ph[i + 1] = 1 if z[i + 1] - hull.h_keel > eta + 0.05 else 0

    # First-sample logging
    T[0] = ac.prop.thrust(Vx[0], throttle)
    q0 = 0.5 * RHO * math.hypot(Vx[0], Vz[0]) ** 2
    gamma0 = math.atan2(Vz[0], Vx[0])
    L[0] = q0 * ac.geom.S * ac.CL(alpha - gamma0)
    D[0] = q0 * ac.geom.S * ac.CD(L[0] / (q0 * ac.geom.S))
    eta0 = float(sea.eta(np.array([x[0]]), t[0])[0])
    wf0 = hull_force(z[0], Vx[0], Vz[0], eta0, hull)
    Nw[0] = wf0.N
    R_[0] = hd.resistance(Vx[0]) if wf0.N > 0 else 0.0
    ph[0] = 1

    return SimResult(t=t, x=x, z=z, Vx=Vx, Vz=Vz,
                     T=T, L=L, D=D, R=R_, N_water=Nw,
                     alpha=alpha, throttle=throttle, phase=ph)


# ---------------------------------------------------------------------
if __name__ == "__main__":
    from aircraft import Aircraft
    from ocean import Ocean

    ac = Aircraft()
    sea = Ocean(Hs=1.5, Tp=6.0, seed=42)

    print("Take-off run, full throttle, alpha = 4 deg")
    res = simulate_takeoff(ac, sea, duration=12.0, dt=0.001,
                           throttle=1.0, alpha=math.radians(4.0))
    lift_off = int(np.argmax(res.phase))
    print(f"  Lift-off at t = {res.t[lift_off]:.2f} s, "
          f"x = {res.x[lift_off]:.1f} m, "
          f"Vx = {res.Vx[lift_off]:.2f} m/s")
    print(f"  Top speed reached: {res.Vx.max():.2f} m/s")
    print(f"  Peak hull drag   : {res.R.max():.0f} N "
          f"({res.R.max()/ac.W*100:.0f} % of W)")

    print("\nLanding, glide slope 8 deg, idle-ish throttle")
    resL = simulate_landing(ac, sea,
                            glide_slope=math.radians(8.0),
                            approach_speed=13.0,
                            alpha_body=math.radians(-5.0),
                            throttle=0.05,
                            duration=40.0, dt=0.001)
    # find first water contact
    contact_idx = np.where(resL.N_water > 0)[0]
    if len(contact_idx) > 0:
        td = int(contact_idx[0])
        print(f"  Touchdown at t = {resL.t[td]:.2f} s, "
              f"x = {resL.x[td]:.1f} m, "
              f"Vx = {resL.Vx[td]:.2f} m/s, "
              f"|Vz| = {abs(resL.Vz[td]):.2f} m/s")
        print(f"  Peak N_water    : {resL.N_water.max():.0f} N "
              f"({resL.N_water.max()/ac.W:.1f} W)")
        print(f"  Final speed     : Vx = {resL.Vx[-1]:.2f} m/s")
    else:
        print("  No water contact during run -- check glide parameters")