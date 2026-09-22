from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Mapping

from .types import EarthPatchConfig, GeoPoint, LocalPosition

SCHEMA = "hama-earth-scenario/v0.1"


@dataclass(frozen=True)
class ScenarioObjectSpec:
    object_id: str
    object_type: str
    position: LocalPosition
    parameters: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ScenarioSpec:
    name: str
    patch: EarthPatchConfig
    environment_provider: str
    environment_parameters: Mapping[str, Any]
    objects: tuple[ScenarioObjectSpec, ...]
    duration_s: float
    dt_s: float

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ScenarioSpec":
        if data.get("schema") != SCHEMA:
            raise ValueError(f"scenario schema must be {SCHEMA}")
        world = data["world"]
        origin = world["origin"]
        start = datetime.fromisoformat(str(world["start_time"]).replace("Z", "+00:00"))
        patch = EarthPatchConfig(
            name=str(data["name"]),
            origin=GeoPoint(origin["lat_deg"], origin["lon_deg"], origin.get("altitude_m", 0.0)),
            extent_m=tuple(world["extent_m"]),
            start_time=start,
            seed=world.get("seed", 0),
        )
        run = data.get("run", {})
        duration_s = float(run.get("duration_s", 1.0))
        dt_s = float(run.get("dt_s", 0.05))
        if duration_s < 0 or not 0 < dt_s <= 60:
            raise ValueError("invalid run duration or timestep")
        env = data["environment"]
        objects = []
        seen = set()
        for raw in data.get("objects", []):
            object_id = str(raw["id"])
            if not object_id or object_id in seen:
                raise ValueError(f"invalid or duplicate object id: {object_id!r}")
            seen.add(object_id)
            pos = raw.get("position_m", (0.0, 0.0, 0.0))
            objects.append(ScenarioObjectSpec(
                object_id=object_id,
                object_type=str(raw["type"]),
                position=LocalPosition(*pos),
                parameters=dict(raw.get("parameters", {})),
            ))
        return cls(
            name=str(data["name"]), patch=patch,
            environment_provider=str(env["provider"]),
            environment_parameters=dict(env.get("parameters", {})),
            objects=tuple(objects), duration_s=duration_s, dt_s=dt_s,
        )


def load_scenario(path: str | Path) -> ScenarioSpec:
    with Path(path).open("r", encoding="utf-8") as fh:
        return ScenarioSpec.from_dict(json.load(fh))
