from __future__ import annotations

from collections import OrderedDict
from typing import Any, Protocol, runtime_checkable

from .types import EarthPatchConfig, EnvironmentSnapshot, LocalPosition


@runtime_checkable
class EnvironmentProvider(Protocol):
    def sample(self, position: LocalPosition, time_s: float) -> EnvironmentSnapshot: ...


@runtime_checkable
class SimulationObject(Protocol):
    object_id: str
    position: LocalPosition

    def step(self, world: "EarthWorld", dt_s: float) -> None: ...
    def snapshot(self) -> dict[str, Any]: ...


class EarthWorld:
    """Deterministic fixed-step local Earth patch.

    The world owns simulation time and environmental sampling. Product-specific
    physics live in SimulationObject implementations instead of the Earth core.
    """

    def __init__(self, config: EarthPatchConfig, environment: EnvironmentProvider):
        if not isinstance(environment, EnvironmentProvider):
            raise TypeError("environment does not implement EnvironmentProvider")
        self.config = config
        self.environment = environment
        self.time_s = 0.0
        self._objects: OrderedDict[str, SimulationObject] = OrderedDict()

    @property
    def objects(self) -> tuple[SimulationObject, ...]:
        return tuple(self._objects.values())

    def add_object(self, obj: SimulationObject) -> None:
        if not isinstance(obj, SimulationObject):
            raise TypeError("object does not implement SimulationObject")
        if not obj.object_id or obj.object_id in self._objects:
            raise ValueError(f"invalid or duplicate object_id: {obj.object_id!r}")
        self._objects[obj.object_id] = obj

    def sample_environment(self, position: LocalPosition) -> EnvironmentSnapshot:
        return self.environment.sample(position, self.time_s)

    def step(self, dt_s: float) -> None:
        dt_s = float(dt_s)
        if not 0.0 < dt_s <= 60.0:
            raise ValueError("dt_s must be in (0, 60]")
        for obj in tuple(self._objects.values()):
            obj.step(self, dt_s)
        self.time_s += dt_s

    def run(self, duration_s: float, dt_s: float) -> None:
        duration_s = float(duration_s)
        dt_s = float(dt_s)
        if duration_s < 0 or dt_s <= 0:
            raise ValueError("duration_s must be nonnegative and dt_s positive")
        whole_steps = int(duration_s // dt_s)
        for _ in range(whole_steps):
            self.step(dt_s)
        remainder = duration_s - whole_steps * dt_s
        if remainder > 1e-12:
            self.step(remainder)

    def snapshot(self) -> dict[str, Any]:
        return {
            "schema": "hama-earth-world/v0.1",
            "patch": self.config.name,
            "time_s": self.time_s,
            "objects": [obj.snapshot() for obj in self._objects.values()],
        }
