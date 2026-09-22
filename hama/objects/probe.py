from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from hama.earth.types import LocalPosition


@dataclass
class EnvironmentalProbe:
    """Reference object proving products can consume the generic Earth contract."""

    object_id: str
    position: LocalPosition
    samples: int = 0
    last_environment: dict[str, Any] | None = None

    def step(self, world, dt_s: float) -> None:
        self.last_environment = world.sample_environment(self.position).to_dict()
        self.samples += 1

    def snapshot(self) -> dict[str, Any]:
        return {
            "id": self.object_id,
            "type": "environment_probe",
            "position_m": list(self.position.as_tuple()),
            "samples": self.samples,
            "environment": self.last_environment,
        }
