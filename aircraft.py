"""Drone flying-boat geometric & mass design (15 m wingspan, EPS foam).

All values are design-point estimates consistent with public data for
large foam-skinned UAVs (Predator-class wingspan) and Li-Po powered
amateur-built flying boats.
"""
from dataclasses import dataclass
import math

# --- Constants --------------------------------------------------------
G      = 9.80665        # gravity, m/s^2
RHO    = 1.225          # air density at sea level, kg/m^3
RHO_W  = 1025.0         # seawater density, kg/m^3
NU_W   = 1.0e-6         # seawater kinematic viscosity, m^2/s
G_SEA  = 9.80665        # gravity (for waves)

# --- Geometry ---------------------------------------------------------
@dataclass(frozen=True)
class Geometry:
    b:    float = 15.0   # wing span, m
    c:    float = 1.5    # mean aerodynamic chord, m
    S:    float = 22.5   # wing area (b * c), m^2
    AR:   float = 10.0   # aspect ratio
    t_c:  float = 0.12   # thickness-to-chord ratio
    # Hull (slenderness typical of flying-boat hulls, length / beam ≈ 4-6)
    Lwl:  float = 2.6    # waterline length, m
    Bwl:  float = 0.55   # waterline beam, m
    # Tail
    S_t:  float = 2.4    # tail area, m^2

    @property
    def wing_volume(self) -> float:
        # foam wing volume assuming ~70% solid fraction
        return self.b * self.c * self.t_c * 0.70

# --- Mass breakdown (kg) ---------------------------------------------
@dataclass(frozen=True)
class MassBreakdown:
    wing_structure:  float = 32.0   # EPS foam core + carbon spar + glass skin
    hull_structure:  float = 18.0   # foam hull + glass laminate
    tail_structure:  float =  4.0
    motor:           float =  3.5   # large outrunner (e.g. 8108-class)
    esc:             float =  0.6
    propeller:       float =  2.0   # 22x10 carbon prop
    battery:         float =  6.5   # 2x 6S 22000 mAh Li-Po
    avionics:        float =  3.0   # flight controller, GPS, RC rx, servos
    wiring_misc:     float =  1.5
    payload:         float =  5.0   # camera/sensor
    margin:          float =  3.9   # design margin

    @property
    def total(self) -> float:
        return sum(self.__dict__.values())

# --- Aerodynamics ----------------------------------------------------
@dataclass(frozen=True)
class Aerodynamics:
    CL_max:  float = 1.4        # max lift coefficient (flap-equipped foam wing)
    CL0:     float = 0.30       # lift at alpha=0
    CL_alpha: float = 5.5       # per radian, 2-D airfoil section
    CD0:     float = 0.025      # zero-lift drag (foam/glass clean)
    e:       float = 0.85       # Oswald efficiency
    alpha_T: float = math.radians(2.0)  # thrust incidence offset

# --- Propulsion -------------------------------------------------------
@dataclass(frozen=True)
class Propulsion:
    n_motors:   int   = 2         # twin-motor pusher/tractor
    P_max:      float = 12000.0   # total peak electrical power, W
    P_cruise:   float =  3000.0   # total cruise electrical power, W
    eta_motor:  float = 0.90      # motor efficiency
    eta_esc:    float = 0.97      # ESC efficiency
    D_prop:     float = 0.71      # propeller diameter, m (28 inch)
    pitch:      float = 0.30      # propeller pitch, m    (12 inch)
    n_rpm:      float = 5500.0    # nominal RPM
    C_T_static: float = 0.105     # static thrust coeff (n in rev/s)

    @property
    def n_hz(self) -> float:
        """Rotational frequency in revolutions per second."""
        return self.n_rpm / 60.0

    @property
    def T_static(self) -> float:
        """Static thrust of the full propulsion system."""
        T_one = (self.C_T_static * RHO
                 * self.n_hz**2 * self.D_prop**4)
        return T_one * self.n_motors

    @property
    def V_max(self) -> float:
        """Ideal no-load advance velocity, m/s."""
        return self.n_hz * self.pitch * 0.85

    def thrust(self, V: float, throttle: float = 1.0) -> float:
        """Quadratic thrust-vs-speed curve, bounded by static thrust."""
        V_eff = max(0.0, min(V, self.V_max))
        T = self.T_static * throttle * (1.0 - (V_eff / self.V_max) ** 2)
        return max(T, 0.0)

    def power_required(self, V: float, throttle: float = 1.0) -> float:
        """Electrical power to deliver thrust(T) at speed V."""
        if V < 0.5:
            V = 0.5
        T = self.thrust(V, throttle)
        eta = self.eta_motor * self.eta_esc
        return T * V / eta

# --- Aircraft aggregate ----------------------------------------------
@dataclass(frozen=True)
class Aircraft:
    geom: Geometry = Geometry()
    mass: MassBreakdown = MassBreakdown()
    aero: Aerodynamics = Aerodynamics()
    prop: Propulsion = Propulsion()

    @property
    def W(self) -> float:
        return self.mass.total * G                  # weight, N

    @property
    def V_stall(self) -> float:
        return math.sqrt(2 * self.W / (RHO * self.geom.S * self.aero.CL_max))

    @property
    def V_cruise(self) -> float:
        """Speed for minimum power (L/D max × √3)."""
        return self.V_stall * math.sqrt(3.0)

    def CL(self, alpha: float) -> float:
        return self.aero.CL0 + self.aero.CL_alpha * alpha

    def CD(self, CL: float) -> float:
        return (self.aero.CD0
                + CL**2 / (math.pi * self.aero.e * self.geom.AR))

    def L_D(self, CL: float) -> float:
        return CL / self.CD(CL)

    def summary(self) -> str:
        m = self.mass
        a = self
        lines = [
            "=" * 60,
            "Drone flying-boat  (15 m wingspan, EPS foam)",
            "=" * 60,
            f"Wing span / chord / area        : {a.geom.b:.1f} m / "
            f"{a.geom.c:.1f} m / {a.geom.S:.1f} m^2",
            f"Aspect ratio                    : {a.geom.AR:.1f}",
            f"Waterline length / beam         : {a.geom.Lwl:.2f} / "
            f"{a.geom.Bwl:.2f} m",
            f"Total mass                      : {m.total:.1f} kg",
            f"  wing  {m.wing_structure:5.1f}  hull {m.hull_structure:5.1f}  "
            f"tail {m.tail_structure:5.1f}",
            f"  motor {m.motor:5.1f}  esc {m.esc:4.1f}  prop {m.propeller:4.1f}  "
            f"batt {m.battery:4.1f}",
            f"  avionics {m.avionics:4.1f}  wiring {m.wiring_misc:4.1f}  "
            f"payload {m.payload:4.1f}  margin {m.margin:4.1f}",
            f"Weight                          : {a.W:.0f} N",
            f"Stall speed (CL_max={a.aero.CL_max:.2f})        : "
            f"{a.V_stall:.2f} m/s  ({a.V_stall*3.6:.1f} km/h)",
            f"Cruise speed (L/D max)          : {a.V_cruise:.2f} m/s  "
            f"({a.V_cruise*3.6:.1f} km/h)",
            f"Thrust @ cruise, full throttle  : "
            f"{a.prop.thrust(a.V_cruise, 1.0):.0f} N  "
            f"(W={a.W:.0f} N, ratio "
            f"{a.prop.thrust(a.V_cruise, 1.0)/a.W:.2f})",
            f"Static thrust                   : {a.prop.T_static:.0f} N  "
            f"(T/W_static = {a.prop.T_static/a.W:.2f})",
            f"Battery energy (2x 6S 22 Ah)    : "
            f"{2 * 22.0 * 22.2 * 3600 / 1e6:.2f} MJ",
        ]
        return "\n".join(lines)


if __name__ == "__main__":
    ac = Aircraft()
    print(ac.summary())
    # Quick sanity check: at cruise speed, lift = weight?
    V = ac.V_cruise
    CL = ac.W / (0.5 * RHO * V**2 * ac.geom.S)
    print(f"\nDesign CL @ cruise: {CL:.3f}")
    print(f"Design CD @ cruise: {ac.CD(CL):.4f}")
    print(f"L/D @ cruise      : {ac.L_D(CL):.1f}")
    # Energy budget
    P_cruise = ac.prop.power_required(V, throttle=0.7)
    t_endurance = (2 * 22.0 * 22.2 * 3600 * 0.8) / P_cruise / 60.0
    print(f"Cruise power @ 70 % throttle     : {P_cruise:.0f} W")
    print(f"Endurance @ 70 % throttle        : {t_endurance:.1f} min")