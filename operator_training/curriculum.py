"""Curriculum definitions: phase machine, approach gate, touchdown rule.

Touchdown is the canonical event that ends a session: the hull bottom
makes contact with the water surface. Per the design review, this is
keel_clearance_m <= 0 (z - hull.h_keel - eta), which is also what
observe() exposes. The old manual_pilot.py used z - eta <= 0.05; that
threshold is preserved as `legacy_touchdown_altitude_m` for diagnosis
only and never used for course completion.

Approach gates define when AUTO LAND can be confirmed: a set of
predicates over telemetry. Default gate matches the existing
lateral_success thresholds (|y|<10, |Vy|<1.5, |bank|<10 deg) plus
the altitude and airspeed envelope from the design (25 m, 13 m/s,
8 deg glide).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Mapping


class CurriculumPhase(str, Enum):
    SETUP = "SETUP"
    READY = "READY"
    TAKEOFF = "TAKEOFF"
    CRUISE = "CRUISE"
    APPROACH = "APPROACH"
    TOUCHDOWN = "TOUCHDOWN"
    FINISHED = "FINISHED"
    ABORTED = "ABORTED"


@dataclass(frozen=True)
class TouchdownDetector:
    """Detect first hull-keel contact with the water surface.

    hull_clearance_m = z - hull_h_keel_m - eta
    """

    hull_h_keel_m: float
    min_sim_t_s: float = 1.0  # ignore contacts in the first second

    def clearance_m(self, *, z: float, eta: float) -> float:
        return float(z) - float(self.hull_h_keel_m) - float(eta)

    def is_contact(self, *, z: float, eta: float, sim_t: float) -> bool:
        if sim_t < self.min_sim_t_s:
            return False
        return self.clearance_m(z=z, eta=eta) <= 0.0

    @classmethod
    def from_vehicle(cls, vehicle) -> "TouchdownDetector":
        h_keel = getattr(getattr(vehicle, "hull", None), "h_keel", 0.0)
        return cls(hull_h_keel_m=float(h_keel))


@dataclass(frozen=True)
class ApproachGate:
    """Bounds that AUTO LAND can be confirmed inside.

    `predicates` are evaluated with the current telemetry dict and must
    return True to allow the gate to open. Built-in predicates cover
    altitude, airspeed, sink rate, bank, lateral offset/velocity, and
    damage / failure state.
    """

    altitude_lo_m: float = 15.0
    altitude_hi_m: float = 60.0
    airspeed_lo_m_s: float = 10.0
    airspeed_hi_m_s: float = 18.0
    sink_rate_max_m_s: float = 4.0
    bank_abs_deg: float = 12.0
    lateral_abs_m: float = 15.0
    lateral_speed_abs_m_s: float = 2.5
    require_not_failed: bool = True

    def predicate_altitude(self, obs: Mapping[str, float]) -> bool:
        z = obs.get("altitude_m")
        return z is not None and self.altitude_lo_m <= z <= self.altitude_hi_m

    def predicate_airspeed(self, obs: Mapping[str, float]) -> bool:
        v = obs.get("airspeed_m_s")
        if v is None:
            v = obs.get("forward_speed_m_s")
        return v is not None and self.airspeed_lo_m_s <= v <= self.airspeed_hi_m_s

    def predicate_sink(self, obs: Mapping[str, float]) -> bool:
        vz = obs.get("vertical_speed_m_s")
        return vz is not None and -vz <= self.sink_rate_max_m_s

    def predicate_bank(self, obs: Mapping[str, float]) -> bool:
        b = obs.get("bank_deg")
        return b is not None and abs(b) <= self.bank_abs_deg

    def predicate_lateral(self, obs: Mapping[str, float]) -> bool:
        y = obs.get("lateral_position_m")
        vy = obs.get("lateral_speed_m_s")
        if y is None:
            return True
        if abs(y) > self.lateral_abs_m:
            return False
        if vy is not None and abs(vy) > self.lateral_speed_abs_m_s:
            return False
        return True

    def predicate_health(self, obs: Mapping[str, float]) -> bool:
        if not self.require_not_failed:
            return True
        return not bool(obs.get("failed", False))

    def evaluate(self, obs: Mapping[str, float]) -> tuple[bool, list[str]]:
        checks = [
            ("altitude", self.predicate_altitude(obs)),
            ("airspeed", self.predicate_airspeed(obs)),
            ("sink_rate", self.predicate_sink(obs)),
            ("bank", self.predicate_bank(obs)),
            ("lateral", self.predicate_lateral(obs)),
            ("health", self.predicate_health(obs)),
        ]
        missing = [name for name, ok in checks if not ok]
        return (not missing, missing)


def default_approach_gate() -> ApproachGate:
    """Defaults matching design section 6.2 (~13 m/s, 25 m, 8 deg glide)."""
    return ApproachGate(
        altitude_lo_m=15.0,
        altitude_hi_m=60.0,
        airspeed_lo_m_s=11.0,
        airspeed_hi_m_s=15.0,
        sink_rate_max_m_s=3.0,
        bank_abs_deg=10.0,
    )


@dataclass(frozen=True)
class Curriculum:
    """Course definition: id, version, scenario, gate, touchdown rule."""

    curriculum_id: str
    curriculum_version: str
    scenario: str  # "takeoff" | "landing" | "manual_takeoff" | "manual_landing" | "hybrid"
    spatial: bool = True
    handover_altitude_m: float = 8.0
    handover_airspeed_m_s: float | None = None  # filled by session from aircraft
    handover_vertical_speed_m_s: float = 2.0
    handover_bank_deg: float = 10.0
    handover_duration_s: float = 1.0
    approach_gate: ApproachGate = field(default_factory=default_approach_gate)

    def to_dict(self) -> dict:
        return {
            "curriculum_id": self.curriculum_id,
            "curriculum_version": self.curriculum_version,
            "scenario": self.scenario,
            "spatial": self.spatial,
            "handover": {
                "altitude_m": self.handover_altitude_m,
                "airspeed_m_s": self.handover_airspeed_m_s,
                "vertical_speed_m_s": self.handover_vertical_speed_m_s,
                "bank_deg": self.handover_bank_deg,
                "duration_s": self.handover_duration_s,
            },
            "approach_gate": {
                "altitude_lo_m": self.approach_gate.altitude_lo_m,
                "altitude_hi_m": self.approach_gate.altitude_hi_m,
                "airspeed_lo_m_s": self.approach_gate.airspeed_lo_m_s,
                "airspeed_hi_m_s": self.approach_gate.airspeed_hi_m_s,
                "sink_rate_max_m_s": self.approach_gate.sink_rate_max_m_s,
                "bank_abs_deg": self.approach_gate.bank_abs_deg,
                "lateral_abs_m": self.approach_gate.lateral_abs_m,
                "lateral_speed_abs_m_s": self.approach_gate.lateral_speed_abs_m_s,
            },
        }
