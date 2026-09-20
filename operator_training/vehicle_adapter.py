"""Authority-aware adapter over FlyingBoatVehicle.

The adapter enforces three rules from the design:

1. Each RUNNING tick commits to ONE control source (AUTO or HUMAN)
   before any servo is touched. The control source is decided by
   session.state, not by input arrival.
2. AUTO -> HUMAN is implemented by sending all four servos in one
   batch via SpatialControl.apply; the first servo call flips
   _active_cmd to MAV_CMD_DO_SET_SERVO and that is the desired side
   effect.
3. HUMAN -> AUTO must explicitly re-send the navigation command
   (MAV_CMD_NAV_LAND / _TAKEOFF / _WAYPOINT) so the navigation path
   in step() translates cmd -> setpoints again.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from aircraft import Aircraft
from mavlink_if import (
    MAV_CMD_NAV_LAND,
    MAV_CMD_NAV_TAKEOFF,
    MAV_CMD_NAV_WAYPOINT,
    FlyingBoatVehicle,
)
from ocean import Ocean
from ocean_directional import DirectionalOcean
from ollama_pilot import SpatialControl, observe

from .envelope import ControlEnvelope, ControlInput


@dataclass
class NavCommand:
    """Active navigation command issued to FlyingBoatVehicle."""

    kind: str  # "TAKEOFF" | "LAND" | "WAYPOINT"
    alt_m: float
    speed_m_s: float
    heading_deg: float = 0.0
    glide_deg: float = 8.0
    target_y_m: float = 0.0
    target_x_m: Optional[float] = None

    def send(self, vehicle: FlyingBoatVehicle) -> None:
        if not vehicle._armed:
            raise RuntimeError("vehicle is not armed")
        if self.kind == "TAKEOFF":
            vehicle.send_command(MAV_CMD_NAV_TAKEOFF, {
                "alt": float(self.alt_m),
                "speed": float(self.speed_m_s),
                "heading": float(math.radians(self.heading_deg)),
            })
        elif self.kind == "LAND":
            params = {
                "alt": float(self.alt_m),
                "speed": float(self.speed_m_s),
                "glide": float(self.glide_deg),
                "heading": float(math.radians(self.heading_deg)),
            }
            if self.target_x_m is not None:
                params["x"] = float(self.target_x_m)
            params["y"] = float(self.target_y_m)
            vehicle.send_command(MAV_CMD_NAV_LAND, params)
        elif self.kind == "WAYPOINT":
            params = {
                "alt": float(self.alt_m),
                "speed": float(self.speed_m_s),
                "heading": float(math.radians(self.heading_deg)),
            }
            if self.target_x_m is not None and self.target_y_m is not None:
                params["x"] = float(self.target_x_m)
                params["y"] = float(self.target_y_m)
            vehicle.send_command(MAV_CMD_NAV_WAYPOINT, params)
        else:  # pragma: no cover - validated upstream
            raise ValueError(f"unknown nav kind: {self.kind!r}")


class VehicleAdapter:
    """Thin wrapper exposing only what the session needs."""

    def __init__(self, vehicle: FlyingBoatVehicle, envelope: ControlEnvelope):
        self.vehicle = vehicle
        self.envelope = envelope

    # ----- construction -----
    @classmethod
    def build(cls, *, spatial: bool = True, seed: int = 42,
              aircraft: Aircraft | None = None,
              sea=None,
              envelope: ControlEnvelope | None = None) -> "VehicleAdapter":
        aircraft = aircraft or Aircraft()
        if sea is None:
            if spatial:
                sea = DirectionalOcean(Hs=0.8, Tp=6.0, theta_mean=0.0, seed=seed)
            else:
                sea = Ocean(Hs=0.8, Tp=6.0, seed=seed)
        vehicle = FlyingBoatVehicle(aircraft, sea, spatial=spatial, seed=seed)
        return cls(vehicle, envelope or ControlEnvelope.beginner())

    # ----- lifecycle -----
    def arm(self) -> None:
        self.vehicle.arm()

    def disarm(self) -> None:
        self.vehicle.disarm()

    def reset(self) -> None:
        self.vehicle.reset()

    def step(self, dt: float) -> None:
        self.vehicle.step(dt=dt)

    # ----- authority -----
    def apply_human(self, raw_input: ControlInput) -> None:
        """Apply human 4-axis control via SpatialControl (servo 1..4)."""
        SpatialControl(
            throttle=raw_input.throttle,
            pitch_deg=raw_input.pitch_deg,
            bank_deg=raw_input.bank_deg,
            rudder=raw_input.rudder,
        ).apply(self.vehicle)

    def apply_nav(self, nav: NavCommand) -> None:
        nav.send(self.vehicle)

    def is_servo_mode(self) -> bool:
        """True iff the last applied actuator command was a servo command."""
        from mavlink_if import MAV_CMD_DO_SET_SERVO
        return self.vehicle._active_cmd == MAV_CMD_DO_SET_SERVO

    # ----- observation -----
    def observe(self) -> dict:
        return observe(self.vehicle)

    @property
    def sim_t(self) -> float:
        return float(self.vehicle.t)

    @property
    def failed(self) -> bool:
        return bool(self.vehicle.damage.failed)

    @property
    def failure_reason(self) -> str:
        return str(self.vehicle.damage.failure_reason)

    @property
    def keel_clearance_m(self) -> float:
        obs = self.observe()
        clearance = obs.get("keel_clearance_m")
        if clearance is None:
            z = self.vehicle.z
            eta = self.vehicle.wave_elevation()
            return z - self.vehicle.hull.h_keel - eta
        return float(clearance)

    @property
    def wave_elevation_m(self) -> float:
        return float(self.vehicle.wave_elevation())
