"""Deterministic input harness for headless tests.

FakePad holds axis values and a small set of named buttons. Tests
mutate the pad and call session.tick() with the snapshotted state.
This mirrors the browser pad snapshot contract (one input message per
tick; axes and buttons are atomic at read time).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class FakePad:
    throttle: float = 0.0
    pitch_deg: float = 0.0
    bank_deg: float = 0.0
    rudder: float = 0.0
    buttons: dict = field(default_factory=dict)

    def axis_dict(self) -> dict:
        return {
            "throttle": float(self.throttle),
            "pitch_deg": float(self.pitch_deg),
            "bank_deg": float(self.bank_deg),
            "rudder": float(self.rudder),
        }

    def press(self, name: str) -> "FakePad":
        self.buttons[name] = True
        return self

    def release(self, name: str) -> "FakePad":
        self.buttons.pop(name, None)
        return self

    def is_pressed(self, name: str) -> bool:
        return bool(self.buttons.get(name, False))

    def reset_axes(self) -> None:
        self.throttle = 0.0
        self.pitch_deg = 0.0
        self.bank_deg = 0.0
        self.rudder = 0.0
