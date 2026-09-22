from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import math
from typing import Any, Mapping


def _finite(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


@dataclass(frozen=True)
class GeoPoint:
    lat_deg: float
    lon_deg: float
    altitude_m: float = 0.0

    def __post_init__(self):
        lat = _finite(self.lat_deg, "lat_deg")
        lon = _finite(self.lon_deg, "lon_deg")
        alt = _finite(self.altitude_m, "altitude_m")
        if not -90.0 <= lat <= 90.0:
            raise ValueError("lat_deg must be in [-90, 90]")
        if not -180.0 <= lon <= 180.0:
            raise ValueError("lon_deg must be in [-180, 180]")
        object.__setattr__(self, "lat_deg", lat)
        object.__setattr__(self, "lon_deg", lon)
        object.__setattr__(self, "altitude_m", alt)


@dataclass(frozen=True)
class LocalPosition:
    """Local tangent-plane position: x north, y east, z up, in metres."""

    x_m: float = 0.0
    y_m: float = 0.0
    z_m: float = 0.0

    def __post_init__(self):
        object.__setattr__(self, "x_m", _finite(self.x_m, "x_m"))
        object.__setattr__(self, "y_m", _finite(self.y_m, "y_m"))
        object.__setattr__(self, "z_m", _finite(self.z_m, "z_m"))

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.x_m, self.y_m, self.z_m)


@dataclass(frozen=True)
class EarthPatchConfig:
    name: str
    origin: GeoPoint
    extent_m: tuple[float, float, float]
    start_time: datetime
    seed: int = 0

    def __post_init__(self):
        if not self.name.strip():
            raise ValueError("patch name must not be empty")
        if len(self.extent_m) != 3:
            raise ValueError("extent_m must contain three values")
        extent = tuple(_finite(v, "extent_m") for v in self.extent_m)
        if any(v <= 0 for v in extent):
            raise ValueError("extent_m values must be positive")
        if self.start_time.tzinfo is None or self.start_time.utcoffset() is None:
            raise ValueError("start_time must be timezone-aware")
        object.__setattr__(self, "extent_m", extent)
        object.__setattr__(self, "seed", int(self.seed))


@dataclass(frozen=True)
class EnvironmentSnapshot:
    air_temperature_k: float
    air_pressure_pa: float
    air_density_kg_m3: float
    wind_m_s: tuple[float, float, float]
    surface_elevation_m: float = 0.0
    water_depth_m: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        for name in ("air_temperature_k", "air_pressure_pa", "air_density_kg_m3", "surface_elevation_m"):
            object.__setattr__(self, name, _finite(getattr(self, name), name))
        if self.air_temperature_k <= 0 or self.air_pressure_pa <= 0 or self.air_density_kg_m3 <= 0:
            raise ValueError("atmospheric values must be positive")
        if len(self.wind_m_s) != 3:
            raise ValueError("wind_m_s must contain three values")
        object.__setattr__(self, "wind_m_s", tuple(_finite(v, "wind_m_s") for v in self.wind_m_s))
        if self.water_depth_m is not None:
            depth = _finite(self.water_depth_m, "water_depth_m")
            if depth < 0:
                raise ValueError("water_depth_m must be nonnegative")
            object.__setattr__(self, "water_depth_m", depth)

    def to_dict(self) -> dict[str, Any]:
        return {
            "air_temperature_k": self.air_temperature_k,
            "air_pressure_pa": self.air_pressure_pa,
            "air_density_kg_m3": self.air_density_kg_m3,
            "wind_m_s": list(self.wind_m_s),
            "surface_elevation_m": self.surface_elevation_m,
            "water_depth_m": self.water_depth_m,
            "metadata": dict(self.metadata),
        }
