"""Control envelope definitions and input validation.

Authority inputs are 4-axis (spatial) by default; longitudinal-only
inputs are also accepted for compat with non-spatial sessions.

The envelope is owned by the server. The browser receives the same
envelope via the capability handshake and refuses to send out-of-range
values; the server re-validates to defend against tampering or stale
clients.

Per design document (basic design, 2026-09-20), beginner curriculum
narrows pitch to -8..12 deg and bank to +/-25 deg; the extended
curriculum uses the full -8..15 / +/-45 deg envelope exposed by
FlyingBoatVehicle.send_servo.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping
import math


PITCH_BEGINNER_HI = 12.0
PITCH_EXTENDED_HI = 15.0
PITCH_LO = -8.0
BANK_BEGINNER = 25.0
BANK_EXTENDED = 45.0
RUDDER_LIMIT = 1.0


class EnvelopeViolation(ValueError):
    """Raised when an input is outside the session envelope."""


@dataclass(frozen=True)
class ControlInput:
    """Single-tick human input, already validated against the envelope."""

    throttle: float
    pitch_deg: float
    bank_deg: float
    rudder: float

    def as_dict(self) -> dict:
        return {
            "throttle": self.throttle,
            "pitch_deg": self.pitch_deg,
            "bank_deg": self.bank_deg,
            "rudder": self.rudder,
        }


@dataclass(frozen=True)
class ControlEnvelope:
    """Per-axis limits. All bounds inclusive. Throttle in [0, 1]."""

    throttle_lo: float = 0.0
    throttle_hi: float = 1.0
    pitch_lo: float = PITCH_LO
    pitch_hi: float = PITCH_BEGINNER_HI
    bank_abs: float = BANK_BEGINNER
    rudder_abs: float = RUDDER_LIMIT

    @classmethod
    def beginner(cls) -> "ControlEnvelope":
        return cls()

    @classmethod
    def extended(cls) -> "ControlEnvelope":
        return cls(
            pitch_hi=PITCH_EXTENDED_HI,
            bank_abs=BANK_EXTENDED,
        )

    def validate(self, raw: Mapping[str, float]) -> ControlInput:
        try:
            throttle = float(raw["throttle"])
            pitch_deg = float(raw["pitch_deg"])
            bank_deg = float(raw["bank_deg"])
            rudder = float(raw["rudder"])
        except (KeyError, TypeError, ValueError) as exc:
            raise EnvelopeViolation(f"missing or non-numeric axis: {exc}") from exc

        if not all(math.isfinite(v) for v in (throttle, pitch_deg, bank_deg, rudder)):
            raise EnvelopeViolation("non-finite axis value")

        if not (self.throttle_lo <= throttle <= self.throttle_hi):
            raise EnvelopeViolation(
                f"throttle {throttle} out of [{self.throttle_lo}, {self.throttle_hi}]"
            )
        if not (self.pitch_lo <= pitch_deg <= self.pitch_hi):
            raise EnvelopeViolation(
                f"pitch_deg {pitch_deg} out of [{self.pitch_lo}, {self.pitch_hi}]"
            )
        if abs(bank_deg) > self.bank_abs:
            raise EnvelopeViolation(
                f"bank_deg {bank_deg} exceeds |{self.bank_abs}|"
            )
        if abs(rudder) > self.rudder_abs:
            raise EnvelopeViolation(
                f"rudder {rudder} exceeds |{self.rudder_abs}|"
            )
        return ControlInput(
            throttle=throttle,
            pitch_deg=pitch_deg,
            bank_deg=bank_deg,
            rudder=rudder,
        )

    def clip(self, raw: ControlInput) -> ControlInput:
        """Hard-clip rather than reject. Used for auto-instruction matching."""
        return ControlInput(
            throttle=_clip(raw.throttle, self.throttle_lo, self.throttle_hi),
            pitch_deg=_clip(raw.pitch_deg, self.pitch_lo, self.pitch_hi),
            bank_deg=_clip(raw.bank_deg, -self.bank_abs, self.bank_abs),
            rudder=_clip(raw.rudder, -self.rudder_abs, self.rudder_abs),
        )


def _clip(value: float, lo: float, hi: float) -> float:
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value
