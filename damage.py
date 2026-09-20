"""Sea-spray and water-ingress damage models for a water-borne drone.

Two failure modes are modelled:

1.  Sea spray on the propeller
    --------------------------------
    When the propeller disk is close to the water surface, especially in
    the presence of waves, sea spray impacts the blades.  The effect is
        * reduced aerodynamic efficiency  (eta_prop *= spray_factor)
        * additional torque load on the motor  (extra power draw)
        * at extreme proximity:  "sling" -- a sudden impulse that can
          shatter a blade or strip the gearbox.

    The spray severity is a function of
        * prop_clearance  = z_prop - eta(x, t)
        * wave steepness  (crest rate-of-change)

2.  Water ingress into the airframe
    ---------------------------------
    The hull has a finite volumetric water capacity.  While the keel is
    submerged (or partially submerged with water over the bow) water
    seeps in at a rate that depends on
        * submersion depth
        * wave splash over the bow (function of wave elevation rate)
    Beyond a critical mass, avionics fail and the vehicle is lost.

The models are intentionally simple and parameterised so they can be
tuned from test data.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
import numpy as np

from aircraft import Aircraft, RHO_W, G


# ---------------------------------------------------------------------
#  Sea-spray propeller model
# ---------------------------------------------------------------------
@dataclass
class SprayModel:
    """Sea-spray load on the propeller.

    Parameters
    ----------
    prop_z_offset : altitude of the propeller above the CG, m
    D_prop        : propeller diameter, m
    critical_clearance : clearance below which spray severity rises
                         sharply (typically 0.5 - 1 m for small props)
    """
    prop_z_offset:      float = 0.90    # propeller above CG (positive up)
    D_prop:             float = 0.71    # propeller diameter, m
    critical_clearance: float = 1.5     # m
    # efficiency factor when prop is fully wetted
    min_efficiency:     float = 0.55
    # Extra torque coefficient per unit spray load (dimensionless)
    K_torque_spray:     float = 0.18
    # Sling impact threshold: prop TOP touches water
    sling_threshold:    float = 0.0     # additional clearance below prop top

    def prop_top_clearance(self, z_cg: float, eta: float) -> float:
        """Clearance between prop TOP and water surface (positive = above)."""
        prop_top = z_cg + self.prop_z_offset + self.D_prop / 2.0
        return prop_top - eta

    def spray_factor(self, z_cg: float, eta: float, Veta: float) -> float:
        """Efficiency multiplier in [min_efficiency, 1] for the propeller.

        eta = 1.0 means no spray; eta = min_efficiency means fully wetted.
        """
        clearance = self.prop_top_clearance(z_cg, eta)
        # Normalised severity (0 = far above water, 1 = prop fully under)
        s = max(0.0, 1.0 - clearance / self.critical_clearance)
        # Wave-rate contribution (dynamic spray on the bow)
        s += 0.20 * abs(Veta) / 3.0
        s = min(1.0, s)
        return 1.0 - s * (1.0 - self.min_efficiency)

    def is_sling(self, z_cg: float, eta: float) -> bool:
        """True if the prop TOP is at or below the water surface."""
        return self.prop_top_clearance(z_cg, eta) <= self.sling_threshold

    def extra_power_fraction(self, z_cg: float, eta: float, Veta: float) -> float:
        """Extra motor load (fraction of nominal) due to spray."""
        if self.is_sling(z_cg, eta):
            return 10.0                     # catastrophic
        s = max(0.0, 1.0 - self.prop_top_clearance(z_cg, eta)
                       / self.critical_clearance)
        return self.K_torque_spray * s


# ---------------------------------------------------------------------
#  Water-ingress model
# ---------------------------------------------------------------------
@dataclass
class IngressModel:
    """Cumulative water mass in the hull.

    water_mass = 0 (dry) at start of flight.  Increases when the hull is
    in contact with the water, with contributions from
        * static submersion (keel under water for a long time)
        * bow spray (waves washing over the bow at high speed)
    A critical_mass threshold causes total avionics failure.
    """
    # Critical mass (kg) above which the vehicle is considered lost
    critical_mass:    float = 0.50
    # Warning mass (kg) above which the operator should be alerted
    warning_mass:     float = 0.20
    # Submersion ingress rate (kg / s) at full submersion
    submersion_rate:  float = 0.15
    # Bow-spray coefficient (kg / s per m^2 / s of bow-area × V)
    bow_area:         float = 0.30    # bow cross-section above waterline
    bow_spray_rate:   float = 0.04
    # Mass offset (does NOT subtract from buoyancy, just adds to weight)
    add_to_mass:      bool = True

    def step(self, water_mass: float, dt: float,
             z_cg: float, eta: float, h_keel: float,
             Vx: float, Veta: float) -> float:
        """Integrate water mass forward by one time step.

        Returns the updated water mass in kg.
        """
        # 1. Submersion
        submersion_depth = max(0.0, (eta + h_keel) - z_cg)
        submersion_norm = min(1.0, submersion_depth / 0.30)
        dm_sub = self.submersion_rate * submersion_norm * dt

        # 2. Bow spray from wave crests washing over the bow at speed.
        #    Empirically proportional to Vx and to the rate of wave
        #    elevation rising under the bow (Veta > 0).
        bow_spray = 0.0
        if submersion_depth > 0 and Veta > 0.5 and Vx > 4.0:
            bow_spray = (self.bow_spray_rate
                         * self.bow_area
                         * (Vx / 12.0)
                         * (Veta / 2.0)
                         * dt)
        return water_mass + dm_sub + bow_spray

    def status(self, water_mass: float) -> str:
        if water_mass >= self.critical_mass:
            return "FAILURE"
        if water_mass >= self.warning_mass:
            return "WARNING"
        return "OK"


# ---------------------------------------------------------------------
#  Combined damage model
# ---------------------------------------------------------------------
@dataclass
class DamageState:
    water_mass: float = 0.0
    cumulative_spray: float = 0.0     # time-integrated spray severity
    cumulative_sling_events: int = 0  # count of sling impacts
    sling_active: bool = False
    failed: bool = False
    failure_reason: str = ""


def update_damage(state: DamageState, dt: float,
                  z_cg: float, Vx: float,
                  eta: float, Veta: float,
                  h_keel: float,
                  spray: SprayModel, ingress: IngressModel) -> DamageState:
    """Advance the damage state by dt seconds."""
    if state.failed:
        return state
    # 1. Spray
    sf = spray.spray_factor(z_cg, eta, Veta)
    state.cumulative_spray += (1.0 - sf) * dt
    sling = spray.is_sling(z_cg, eta)
    entered = sling and not state.sling_active
    state.sling_active = sling
    if entered:
        state.cumulative_sling_events += 1
        # First sling is a warning, second is catastrophic
        if state.cumulative_sling_events >= 2:
            state.failed = True
            state.failure_reason = "propeller sling (prop hit solid water)"
            return state
    # 2. Ingress
    state.water_mass = ingress.step(state.water_mass, dt,
                                    z_cg, eta, h_keel, Vx, Veta)
    if state.water_mass >= ingress.critical_mass:
        state.failed = True
        state.failure_reason = (f"hull flooding: {state.water_mass:.2f} kg "
                                f">= critical {ingress.critical_mass:.2f} kg")
        return state
    return state


# ---------------------------------------------------------------------
#  Effective thrust and mass adjustments
# ---------------------------------------------------------------------
def effective_thrust_factor(state: DamageState,
                            spray: SprayModel,
                            z_cg: float, eta: float, Veta: float) -> float:
    """Multiplier on the nominal thrust for the current spray load."""
    return 0.0 if state.failed else spray.spray_factor(z_cg, eta, Veta)


def effective_mass_increase(state: DamageState,
                            ingress: IngressModel) -> float:
    """Extra mass (kg) of water inside the hull, if applicable."""
    if ingress.add_to_mass:
        return state.water_mass
    return 0.0


# ---------------------------------------------------------------------
if __name__ == "__main__":
    # Demo: track water ingress during a slow taxi through 1.5 m waves
    spray = SprayModel()
    ingress = IngressModel()
    state = DamageState()
    h_keel = 0.30
    z = 0.30 - 0.055      # hydrostatic equilibrium draft
    eta = 0.0
    Veta = 0.0
    Vx = 3.0
    print(f"{'t':>4} {'z':>6} {'eta':>6} {'water':>8} {'spray':>6} "
          f"{'T_fac':>6} {'status':>10}")
    for t in np.arange(0.0, 10.0, 0.05):
        eta = 0.6 * math.sin(0.7 * t)        # 0.6 m waves, T~9 s
        Veta = 0.6 * 0.7 * math.cos(0.7 * t)
        state = update_damage(state, 0.05, z, Vx, eta, Veta,
                             h_keel, spray, ingress)
        sf = effective_thrust_factor(state, spray, z, eta, Veta)
        if t % 1.0 < 0.05:
            print(f"{t:4.1f} {z:6.2f} {eta:6.2f} {state.water_mass:8.4f} "
                  f"{(1-sf):6.3f} {sf:6.3f} "
                  f"{ingress.status(state.water_mass):>10}")
        if state.failed:
            print(f"  ** FAILED: {state.failure_reason}")
            break