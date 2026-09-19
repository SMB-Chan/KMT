"""PX4 / ArduPilot-compatible command interface for the flying-boat drone.

We define the MAVLink command constants and message classes used by PX4 /
ArduPilot for vertical-takeoff-and-landing (VTOL) and fixed-wing control
of a flying-boat drone, then implement a thin wrapper around the
dynamics simulator. This is an in-process adapter, not a wire-protocol
implementation; it does not connect to real autopilot hardware.

Supported MAV_CMD commands (subset)
-----------------------------------
    MAV_CMD_NAV_TAKEOFF            (22) - takeoff to target altitude
    MAV_CMD_NAV_LAND               (21) - land at current/guided position
    MAV_CMD_DO_CHANGE_SPEED        (178) - set target airspeed
    MAV_CMD_NAV_WAYPOINT           (16)  - fly to NED waypoint
    MAV_CMD_DO_SET_SERVO           (183) - direct actuator command

Supported MAVLink telemetry messages (subset)
---------------------------------------------
    HIL_STATE                      (90)  - simulated vehicle state
    ATTITUDE                       (30)  - roll/pitch/yaw + rates
    GLOBAL_POSITION_INT            (33)  - lat/lon/alt

The wrapper class `FlyingBoatVehicle` accepts MAV_CMD commands and emits
MAVLink-style telemetry.  An inner `LowLevelController` translates
high-level commands into body-axis throttle / pitch setpoints that the
dynamics simulator consumes.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, replace
from typing import Any
import numpy as np

from aircraft import Aircraft, RHO, RHO_W, G
from ocean    import Ocean
from ocean_real import RealOcean
from ocean_directional import DirectionalOcean, sea_eta_1d
from dynamics import HullContact, HullDrag, hull_force
from damage import (SprayModel, IngressModel, DamageState,
                    update_damage, effective_thrust_factor,
                    effective_mass_increase)


# ---------------------------------------------------------------------
# MAVLink command constants (subset, from common.xml MAVLink 2.0)
# ---------------------------------------------------------------------
MAV_CMD_NAV_WAYPOINT         = 16
MAV_CMD_NAV_LAND             = 21
MAV_CMD_NAV_TAKEOFF          = 22
MAV_CMD_DO_CHANGE_SPEED      = 178
MAV_CMD_DO_SET_SERVO         = 183
MAV_CMD_CONDITION_YAW        = 115

MAV_FRAME_GLOBAL_RELATIVE_ALT = 6
MAV_FRAME_LOCAL_NED           = 1

MAV_MODE_FLAG_SAFETY_ARMED   = 0x80
MAV_STATE_ACTIVE             = 4
MAV_STATE_STANDBY            = 3


# ---------------------------------------------------------------------
@dataclass
class MAVLinkMessage:
    """Tiny stand-in for a MAVLink message."""
    msgid: int
    name:  str
    fields: dict

    def __repr__(self):
        body = ", ".join(f"{k}={v!r}" for k, v in self.fields.items())
        return f"<MAVLink {self.name}({self.msgid}) {body}>"


# ---------------------------------------------------------------------
class LowLevelController:
    """Translates MAV_CMD commands into low-level setpoints.

    This mimics the role of PX4's position controller + attitude
    controller cascade.  For the simplified flying-boat dynamics we
    only need a 1-D vertical + 1-D forward controller:

        Takeoff controller:
            e_alt = target_alt - z ;  desired_climb_rate = Kp*e_alt + Kd*Vz
            target_pitch = clamp(desired_climb_rate / Vx + pitch_trim)
            target_throttle = throttle_hover + Kp_e*e_alt

        Landing controller:
            glide_slope_rad drives descent; as the aircraft nears the
            surface, throttle is reduced and pitch is held at -5 deg.
    """

    def __init__(self,
                 Kp_alt: float = 1.0, Kd_alt: float = 0.4,
                 Kp_speed: float = 0.10,
                 pitch_trim: float = math.radians(4.0),
                 pitch_lo: float = math.radians(-8.0),
                 pitch_hi: float = math.radians(15.0),
                 throttle_min: float = 0.0,
                 throttle_max: float = 1.0):
        self.Kp_alt = Kp_alt
        self.Kd_alt = Kd_alt
        self.Kp_speed = Kp_speed
        self.pitch_trim = pitch_trim
        self.pitch_lo = pitch_lo
        self.pitch_hi = pitch_hi
        self.throttle_min = throttle_min
        self.throttle_max = throttle_max

    def takeoff_setpoint(self, state, target_alt: float, target_speed: float):
        z, Vx, Vz = state["z"], state["Vx"], state["Vz"]
        e_alt = target_alt - z
        # Cap desired climb rate
        climb_rate = np.clip(self.Kp_alt * e_alt - self.Kd_alt * Vz,
                             -5.0, 8.0)
        # Pitch needed to achieve climb rate at current speed (body frame)
        if Vx > 1.0:
            target_pitch = math.atan2(climb_rate, Vx) + self.pitch_trim
        else:
            target_pitch = self.pitch_trim
        target_pitch = float(np.clip(target_pitch, self.pitch_lo, self.pitch_hi))
        # Throttle: high while far from target, low as we approach
        throttle = 0.85 + 0.4 * math.tanh(0.3 * e_alt)
        throttle = float(np.clip(throttle, self.throttle_min, self.throttle_max))
        # Add small speed correction
        throttle += self.Kp_speed * (target_speed - Vx)
        throttle = float(np.clip(throttle, self.throttle_min, self.throttle_max))
        return target_pitch, throttle

    def landing_setpoint(self, state, target_alt: float, glide_slope: float):
        z, Vx, Vz = state["z"], state["Vx"], state["Vz"]
        # We aim to track the glide slope: gamma = -glide_slope (descending)
        # If we're above the line, descend faster; below, level off.
        # Use target_alt (touchdown altitude) as the reference.
        e_alt = z - max(target_alt, 0.0)
        # If we are high, lower pitch (more nose-down) to descend
        if e_alt > 1.0:
            # Aim for 4 degrees nose-down when high
            target_pitch = math.radians(-5.0)
            throttle = 0.10
        else:
            # Flare: pitch up to slow descent
            flare_factor = math.exp(-e_alt / 1.5)
            target_pitch = (math.radians(-5.0) * (1 - flare_factor)
                            + math.radians(2.0) * flare_factor)
            throttle = 0.05
        target_pitch = float(np.clip(target_pitch, self.pitch_lo, self.pitch_hi))
        throttle = float(np.clip(throttle, self.throttle_min, self.throttle_max))
        return target_pitch, throttle

    def hold_setpoint(self, state, target_alt: float, target_speed: float):
        return self.takeoff_setpoint(state, target_alt, target_speed)


# ---------------------------------------------------------------------
@dataclass
class VehicleState:
    """MAVLink-style HIL_STATE payload."""
    timestamp_us: int = 0
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    rollspeed: float = 0.0
    pitchspeed: float = 0.0
    yawspeed: float = 0.0
    lat: int = 0          # degE7
    lon: int = 0          # degE7
    alt: int = 0          # mm AMSL
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    ind_airspeed: float = 0.0
    true_airspeed: float = 0.0
    xacc: float = 0.0
    yacc: float = 0.0
    zacc: float = 0.0


class FlyingBoatVehicle:
    """PX4/ArduPilot-compatible vehicle wrapper around the dynamics simulator.

    The autopilot interacts with this object exclusively via
    send_command(MAV_CMD, params) and reads HIL_STATE / GLOBAL_POSITION_INT
    telemetry via read_telemetry().
    """

    MSG_HIL_STATE            = 90
    MSG_GLOBAL_POSITION_INT   = 33
    MSG_ATTITUDE              = 30

    def __init__(self, aircraft: Aircraft, sea,
                 origin_lat: float = 36.0, origin_lon: float = -122.0):
        self.ac = aircraft
        self.sea = sea
        self.origin_lat = origin_lat
        self.origin_lon = origin_lon
        self.ctl = LowLevelController()
        self.dt = 0.05
        self.hull = HullContact()
        self.hd = HullDrag(Bwl=aircraft.geom.Bwl, Lwl=aircraft.geom.Lwl)
        # Damage models
        self.spray = SprayModel(prop_z_offset=0.30,
                                D_prop=aircraft.prop.D_prop)
        self.ingress = IngressModel()
        self.damage = DamageState()
        # MAVLink command queue (last command wins)
        self._cmd_queue = []
        self._active_cmd = None
        self._cmd_params = None
        self._msg_log = []
        self._armed = False
        self._mode = "STANDBY"
        # State
        self.reset()

    # ----- lifecycle -----
    def reset(self, seed: int | None = None):
        if seed is not None:
            if not isinstance(self.sea, (Ocean, RealOcean, DirectionalOcean)):
                raise TypeError("seeded reset requires Ocean, RealOcean, or DirectionalOcean")
            self.sea = replace(self.sea, seed=seed)
        # Reset dynamics state
        self.x = 0.0
        eta0 = float(sea_eta_1d(self.sea, np.array([0.0]), 0.0)[0])
        self.z = eta0 + self.hull.h_keel - self.ac.W / (RHO_W * G * self.hull.A_wp)
        self.Vx = 0.5
        self.Vz = 0.0
        self.alpha = math.radians(0.0)
        self.throttle = 0.0
        self.t = 0.0
        self._msg_log = []
        self._cmd_queue.clear()
        self._active_cmd = None
        self._cmd_params = None
        self.ctl = LowLevelController()
        self._mode = "STANDBY"
        self._armed = False
        self.damage = DamageState()

    def arm(self):
        if self.damage.failed:
            raise RuntimeError("cannot arm a failed vehicle; reset first")
        self._armed = True
        self._mode = "GUIDED"

    def disarm(self):
        self._armed = False
        self._mode = "STANDBY"
        self.throttle = 0.0

    # ----- MAVLink command interface -----
    def send_command(self, cmd: int, params: dict | None = None):
        """Enqueue a MAV_CMD.  Last command is the active one."""
        params = dict(params or {})
        if cmd == MAV_CMD_DO_SET_SERVO:
            self.send_servo(int(params["servo"]), float(params["pwm"]))
            return
        if cmd == MAV_CMD_DO_CHANGE_SPEED:
            speed = float(params["speed"])
            if not math.isfinite(speed) or speed <= 0:
                raise ValueError("speed must be finite and positive")
            self._cmd_params = dict(self._cmd_params or {}, speed=speed)
            return
        if cmd not in (MAV_CMD_NAV_TAKEOFF, MAV_CMD_NAV_LAND, MAV_CMD_NAV_WAYPOINT):
            raise ValueError(f"unsupported command: {cmd}")
        self._cmd_queue.append((cmd, params))
        self._active_cmd = cmd
        self._cmd_params = params
        # Reflect mode changes
        if cmd == MAV_CMD_NAV_TAKEOFF:
            self._mode = "TAKEOFF"
        elif cmd == MAV_CMD_NAV_LAND:
            self._mode = "LAND"

    def send_servo(self, servo: int, pwm: int):
        """Direct actuator command (MAV_CMD_DO_SET_SERVO)."""
        if servo not in (1, 2) or not math.isfinite(pwm):
            raise ValueError("invalid servo or PWM")
        self._active_cmd = MAV_CMD_DO_SET_SERVO
        self._cmd_params = None
        # We use servo 1 = throttle (PWM 1000-2000), servo 2 = elevator
        if servo == 1:
            self.throttle = np.clip((pwm - 1000) / 1000.0, 0.0, 1.0)
        elif servo == 2:
            # 1500 us center, +/- 500 us -> pitch +/- 15 deg
            self.alpha = np.clip((pwm - 1500) / 500.0 * math.radians(15.0),
                                 math.radians(-15.0), math.radians(15.0))

    # ----- one inner loop step -----
    def step(self, dt: float | None = None, action=None):
        dt = self.dt if dt is None else dt
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive")
        if action is not None:
            action = np.asarray(action)
            if action.shape != (2,) or not np.isfinite(action).all():
                raise ValueError("action must have shape (2,) and be finite")
        # 1) Translate active command into low-level setpoints
        if self._armed:
            if self._active_cmd == MAV_CMD_NAV_TAKEOFF:
                p = self._cmd_params or {}
                tgt_alt = float(p.get("alt", 10.0))
                tgt_spd = float(p.get("speed", 13.0))
                self.alpha, self.throttle = self.ctl.takeoff_setpoint(
                    {"z": self.z, "Vx": self.Vx, "Vz": self.Vz},
                    tgt_alt, tgt_spd)
            elif self._active_cmd == MAV_CMD_NAV_LAND:
                p = self._cmd_params or {}
                tgt_alt = float(p.get("alt", 0.0))
                glide = float(p.get("glide", 8.0))
                self.alpha, self.throttle = self.ctl.landing_setpoint(
                    {"z": self.z, "Vx": self.Vx, "Vz": self.Vz},
                    tgt_alt, math.radians(glide))
            elif self._active_cmd == MAV_CMD_NAV_WAYPOINT:
                # Simple loiter: hold current altitude with light throttle
                p = self._cmd_params or {}
                tgt_alt = float(p.get("alt", self.z))
                tgt_spd = float(p.get("speed", 12.0))
                self.alpha, self.throttle = self.ctl.hold_setpoint(
                    {"z": self.z, "Vx": self.Vx, "Vz": self.Vz},
                    tgt_alt, tgt_spd)
            else:
                # Hold current setpoints
                pass

        if action is not None and self._armed:
            self.throttle = float(np.clip(action[0], 0.0, 1.0))
            self.alpha = math.radians(4.5 + 7.5 * float(np.clip(action[1], -1, 1)))
        if not self._armed:
            self.throttle = 0.0

        # 2) Integrate dynamics
        eta = float(sea_eta_1d(self.sea, np.array([self.x]), self.t)[0])
        eta_next = float(sea_eta_1d(self.sea, np.array([self.x]), self.t + dt)[0])
        Veta = (eta_next - eta) / dt
        V = math.hypot(self.Vx, self.Vz)

        # ---- Damage update ----
        self.damage = update_damage(self.damage, dt,
                                    self.z, self.Vx, eta, Veta,
                                    self.hull.h_keel,
                                    self.spray, self.ingress)
        if self.damage.failed:
            self._armed = False
            self._mode = "EMERGENCY"
            self.throttle = 0.0

        # ---- Spray-adjusted thrust and water-mass-adjusted weight ----
        T_factor = effective_thrust_factor(self.damage,
                                           self.spray,
                                           self.z, eta, Veta)
        T_i = self.ac.prop.thrust(V, self.throttle) * T_factor
        extra_mass = effective_mass_increase(self.damage, self.ingress)
        W_eff = (self.ac.mass.total + extra_mass) * G
        q = 0.5 * RHO * V ** 2
        gamma = math.atan2(self.Vz, self.Vx)
        alpha_eff = self.alpha - gamma
        CL = self.ac.CL(alpha_eff)
        CD = self.ac.CD(CL)
        L_i = q * self.ac.geom.S * CL
        D_i = q * self.ac.geom.S * CD
        wf = hull_force(self.z, self.Vx, self.Vz, eta, self.hull)
        R_hull = self.hd.resistance(self.Vx) if wf.N > 0 else 0.0
        cos_a, sin_a = math.cos(self.alpha), math.sin(self.alpha)
        cos_g, sin_g = math.cos(gamma), math.sin(gamma)
        T_x = T_i * cos_a;  T_z = T_i * sin_a
        L_x = -L_i * sin_g; L_z = +L_i * cos_g
        D_x = -D_i * cos_g; D_z = -D_i * sin_g
        Fx = T_x + L_x + D_x + wf.Rt - math.copysign(R_hull, self.Vx)
        Fz = T_z + L_z + D_z + wf.N - W_eff
        m_total = self.ac.mass.total + extra_mass
        a_x = Fx / m_total
        a_z = Fz / m_total
        self.Vx += a_x * dt
        self.Vz += a_z * dt
        self.x  += self.Vx * dt
        self.z  += self.Vz * dt
        self.t  += dt

        # 3) Emit telemetry
        self._emit_telemetry(eta, T_i, L_i, D_i, wf.N,
                             T_factor, extra_mass)

    # ----- telemetry -----
    def _emit_telemetry(self, eta, T, L, D, N_water,
                        T_factor=1.0, extra_mass=0.0):
        # Snapshot state at the moment of telemetry emission.
        snap = dict(x=self.x, z=self.z, Vx=self.Vx, Vz=self.Vz,
                    alpha=self.alpha, throttle=self.throttle,
                    mode=self._mode,
                    eta=eta, T=T, L=L, D=D, N_water=N_water,
                    T_factor=T_factor, extra_mass=extra_mass,
                    water_mass=self.damage.water_mass,
                    spray_severity=self.damage.cumulative_spray,
                    sling_events=self.damage.cumulative_sling_events,
                    damage_status="OK" if not self.damage.failed else
                                  self.damage.failure_reason)
        # Convert local NED position to lat/lon (simple linear mapping)
        lat = self.origin_lat + self.x / 111000.0
        lon = self.origin_lon + self.x / (111000.0 * math.cos(math.radians(self.origin_lat)))
        alt_mm = int(self.z * 1000.0)
        vx = self.Vx
        vz = self.Vz
        # HIL_STATE
        hil = MAVLinkMessage(
            self.MSG_HIL_STATE, "HIL_STATE",
            dict(timestamp_us=int(self.t * 1e6),
                 roll=0.0, pitch=self.alpha, yaw=0.0,
                 rollspeed=0.0, pitchspeed=vz,
                 yawspeed=0.0,
                 lat=int(lat * 1e7), lon=int(lon * 1e7),
                 alt=alt_mm, vx=vx, vy=0.0, vz=vz,
                 ind_airspeed=vx,
                 true_airspeed=vx,
                 xacc=0.0, yacc=0.0, zacc=-9.81))
        # GLOBAL_POSITION_INT
        gpi = MAVLinkMessage(
            self.MSG_GLOBAL_POSITION_INT, "GLOBAL_POSITION_INT",
            dict(time_boot_ms=int(self.t * 1000),
                 lat=int(lat * 1e7), lon=int(lon * 1e7),
                 alt=alt_mm, relative_alt=alt_mm,
                 vx=int(vx * 100), vy=0, vz=int(vz * 100),
                 hdg=0))
        self._msg_log.append((self.t, hil, gpi, snap))

    def read_telemetry(self):
        if not self._msg_log:
            return None, None, None
        entry = self._msg_log[-1]
        return entry[1], entry[2], (entry[3] if len(entry) == 4 else None)


# ---------------------------------------------------------------------
def a_z_safe(pitch, vz):
    return vz

def V_safe(vx):
    return vx


# ---------------------------------------------------------------------
if __name__ == "__main__":
    print("MAVLink/PX4-compatible flying-boat vehicle wrapper")
    print("=" * 60)
    print(f"Supported MAV_CMD: NAV_WAYPOINT({MAV_CMD_NAV_WAYPOINT}), "
          f"NAV_LAND({MAV_CMD_NAV_LAND}), NAV_TAKEOFF({MAV_CMD_NAV_TAKEOFF}), "
          f"DO_CHANGE_SPEED({MAV_CMD_DO_CHANGE_SPEED}), "
          f"DO_SET_SERVO({MAV_CMD_DO_SET_SERVO})")
    print()
    ac = Aircraft()
    from ocean_real import load_buoy_default
    sea = load_buoy_default()
    veh = FlyingBoatVehicle(ac, sea)
    veh.reset()
    veh.arm()
    # Issue takeoff command
    veh.send_command(MAV_CMD_NAV_TAKEOFF, {"alt": 10.0, "speed": 13.0})
    print(f"Issued MAV_CMD_NAV_TAKEOFF: alt=10 m, speed=13 m/s")
    print()
    for i in range(40):
        veh.step()
        if i % 5 == 0:
            t = veh.t
            x, z = veh.x, veh.z
            Vx, Vz = veh.Vx, veh.Vz
            print(f"  t={t:5.2f} x={x:6.1f} z={z:5.2f} Vx={Vx:5.2f} "
                  f"Vz={Vz:+5.2f} alpha={math.degrees(veh.alpha):+5.1f} "
                  f"thr={veh.throttle:.2f}")