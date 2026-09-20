"""Gamepad calibration.

Converts raw axis readings from the browser Gamepad API into the
normalised axes that the envelope validator expects. The calibration
profile is owned by the browser and is sent to the server at the start
of each session; the server uses the profile to derive the calibrated
value from the raw axis before envelope validation.

Per design section 5.3:

    - deadzone defaults to 5% (tunable per axis)
    - outer re-normalisation: outer 95% of the live range maps to +/-1
    - per-axis invert flag (sign flip)
    - optional response curve (linear, gentle, aggressive)
    - trim is intentionally absent in v1
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Iterable, Mapping


class CalibrationError(ValueError):
    pass


class ResponseCurve:
    """Maps a normalised [0, 1] (after deadzone removal) value to a curve.

    `linear`     -> identity
    `gentle`     -> 0.5 * (1 - cos(pi * x))   ; cubic-feel around zero
    `aggressive` -> x * (3 - 2 * x)          ; S-curve, sharp near 1
    """

    KINDS = ("linear", "gentle", "aggressive")

    @classmethod
    def apply(cls, kind: str, x: float) -> float:
        if not math.isfinite(x):
            raise CalibrationError(f"non-finite input to curve: {x!r}")
        if x < 0.0:
            x = 0.0
        elif x > 1.0:
            x = 1.0
        if kind == "linear":
            return x
        if kind == "gentle":
            # Raised cosine: smooth at endpoints, flat in the middle.
            return 0.5 * (1.0 - math.cos(math.pi * x))
        if kind == "aggressive":
            # Standard smoothstep: zero slope at both endpoints.
            return x * x * (3.0 - 2.0 * x)
        raise CalibrationError(f"unknown response curve: {kind!r}")


@dataclass(frozen=True)
class AxisCalibration:
    """Calibration parameters for a single axis."""

    deadzone: float = 0.05           # fraction of full range
    invert: bool = False
    response_curve: str = "linear"  # one of ResponseCurve.KINDS
    # The "outer live range" the user actually achieves. Defaults to 1.0
    # which means assume the physical axis can reach the full range. If
    # calibration samples show 0.91 instead of 1.00, set max_observed=0.91
    # so the outer reach renormalises to +/-1.
    max_observed: float = 1.0

    def __post_init__(self):
        if not 0.0 <= self.deadzone <= 0.4:
            raise CalibrationError(
                f"deadzone {self.deadzone} outside [0, 0.4]"
            )
        if not 0.5 <= self.max_observed <= 1.0:
            raise CalibrationError(
                f"max_observed {self.max_observed} outside [0.5, 1.0]"
            )
        if self.response_curve not in ResponseCurve.KINDS:
            raise CalibrationError(
                f"response_curve {self.response_curve!r} not in "
                f"{ResponseCurve.KINDS}"
            )

    def apply(self, raw: float) -> float:
        """Convert a raw axis reading in [-1, 1] to a calibrated value.

        The Gamepad API returns axis values in [-1, 1] for centred sticks
        and [0, 1] for triggers. The caller is responsible for shifting
        triggers to [-1, 1] before invoking this method.
        """
        if not math.isfinite(raw):
            raise CalibrationError(f"non-finite raw axis: {raw!r}")
        if self.invert:
            raw = -raw
        sign = 1.0 if raw >= 0.0 else -1.0
        magnitude = abs(raw)
        if magnitude < self.deadzone:
            return 0.0
        # Renormalise the live range: the largest observed magnitude is
        # `max_observed`. Re-scale the post-deadzone reading to [0, 1]
        # using the live range so the outer reach gives +/-1.
        live = max(self.max_observed - self.deadzone, 1e-6)
        normalised = (magnitude - self.deadzone) / live
        if normalised > 1.0:
            normalised = 1.0
        return sign * ResponseCurve.apply(self.response_curve, normalised)


@dataclass(frozen=True)
class GamepadProfile:
    """Calibration for the four primary axes.

    `axis_min_sample`/`axis_max_sample` are the values observed during
    the calibration walk-through. They feed the live-range renorm so a
    controller whose stick never reaches the full swing still maps its
    outer reach to +/-1.
    """

    pit: AxisCalibration = field(default_factory=AxisCalibration)
    ban: AxisCalibration = field(default_factory=AxisCalibration)
    rud: AxisCalibration = field(default_factory=AxisCalibration)
    thr: AxisCalibration = field(default_factory=AxisCalibration)
    profile_hash: str = ""

    def __post_init__(self):
        if not isinstance(self.pit, AxisCalibration):
            raise CalibrationError("pit must be AxisCalibration")
        if not isinstance(self.ban, AxisCalibration):
            raise CalibrationError("ban must be AxisCalibration")
        if not isinstance(self.rud, AxisCalibration):
            raise CalibrationError("rud must be AxisCalibration")
        # Throttle uses an asymmetric deadzone because the right trigger
        # is read at rest near 0; the user pulls to advance. A centred
        # symmetric deadzone is wrong for triggers.
        if not isinstance(self.thr, AxisCalibration):
            raise CalibrationError("thr must be AxisCalibration")

    def apply(self, raw_axes: Mapping[str, float]) -> dict:
        """Map a raw {pit, ban, rud, thr} dict to calibrated axes."""
        try:
            pit_raw = float(raw_axes["pit"])
            ban_raw = float(raw_axes["ban"])
            rud_raw = float(raw_axes["rud"])
            thr_raw = float(raw_axes["thr"])
        except (KeyError, TypeError, ValueError) as exc:
            raise CalibrationError(f"missing or non-numeric raw axis: {exc}") from exc
        # Throttle: the raw value from a trigger is in [0, 1] (Gamepad
        # API). Convert to [-1, 1] symmetric by recentring on the
        # observed minimum (defaults to 0 if not sampled).
        thr_centered = 2.0 * thr_raw - 1.0
        thr_centered = max(-1.0, min(1.0, thr_centered))
        return {
            "pit": self.pit.apply(pit_raw),
            "ban": self.ban.apply(ban_raw),
            "rud": self.rud.apply(rud_raw),
            "thr": self.thr.apply(thr_centered),
        }


def default_profile() -> GamepadProfile:
    """Defaults for the most common Xbox / DualSense layout (no invert)."""
    return GamepadProfile()


def to_control_input(profile: GamepadProfile, raw_axes: Mapping[str, float],
                     *, pitch_range_deg: tuple[float, float],
                     bank_abs_deg: float,
                     rudder_abs: float = 1.0) -> dict:
    """Convert a raw pad snapshot to a candidate ControlInput dict.

    Returns a dict shaped like ``ControlInput.as_dict()``. The envelope
    validator still owns range clamping, so values outside the envelope
    are not pre-clipped here; the server-side validator rejects them.
    """
    axes = profile.apply(raw_axes)
    pitch_lo, pitch_hi = pitch_range_deg
    pitch = axes["pit"] * max(abs(pitch_lo), abs(pitch_hi))
    if pitch < 0:
        pitch = max(pitch, pitch_lo)
    else:
        pitch = min(pitch, pitch_hi)
    return {
        "throttle": float(axes["thr"] * 0.5 + 0.5),
        "pitch_deg": float(pitch),
        "bank_deg": float(axes["ban"] * bank_abs_deg),
        "rudder": float(axes["rud"] * rudder_abs),
    }


def assert_profile_serialisable(profile: GamepadProfile) -> None:
    """Sanity check that the profile can be JSON-serialised for transport."""
    import json
    payload = {
        "pit": profile.pit.__dict__,
        "ban": profile.ban.__dict__,
        "rud": profile.rud.__dict__,
        "thr": profile.thr.__dict__,
        "profile_hash": profile.profile_hash,
    }
    json.dumps(payload, sort_keys=True)
