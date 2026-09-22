from __future__ import annotations

from typing import Any, Mapping

from .types import EnvironmentSnapshot, LocalPosition


class KmtCoastalEnvironment:
    """Adapter exposing KMT's existing ISA/wind and PM-wave models as Earth data.

    The current ocean model is one-dimensional: local x is passed to Ocean.eta
    and y is ignored. This limitation is explicit so a later 2-D/real-data
    provider can replace it without changing products or scenarios.
    """

    def __init__(self, *, wind_m_s=(0.0, 0.0, 0.0), gust_rms_m_s=0.0,
                 Hs_m=0.3, Tp_s=6.0, water_depth_m=50.0, seed=0):
        from atmosphere import Atmosphere, AtmosphereConfig
        from ocean import Ocean

        self.atmosphere = Atmosphere(AtmosphereConfig(
            wind=tuple(wind_m_s), gust_rms=float(gust_rms_m_s)), seed=int(seed))
        self.ocean = Ocean(Hs=float(Hs_m), Tp=float(Tp_s),
                           depth=float(water_depth_m), seed=int(seed))

    @classmethod
    def from_parameters(cls, parameters: Mapping[str, Any], seed: int = 0) -> "KmtCoastalEnvironment":
        allowed = {"wind_m_s", "gust_rms_m_s", "Hs_m", "Tp_s", "water_depth_m"}
        unknown = set(parameters) - allowed
        if unknown:
            raise ValueError(f"unknown kmt-coastal parameters: {sorted(unknown)}")
        return cls(seed=seed, **dict(parameters))

    def sample(self, position: LocalPosition, time_s: float) -> EnvironmentSnapshot:
        altitude = position.z_m
        wind = self.atmosphere.wind(time_s, altitude=altitude)
        eta = float(self.ocean.eta(position.x_m, time_s)[0])
        return EnvironmentSnapshot(
            air_temperature_k=self.atmosphere.temperature(altitude),
            air_pressure_pa=self.atmosphere.pressure(altitude),
            air_density_kg_m3=self.atmosphere.density(altitude),
            wind_m_s=tuple(float(v) for v in wind),
            surface_elevation_m=eta,
            water_depth_m=self.ocean.depth,
            metadata={"provider": "kmt-coastal", "ocean_dimensions": 1},
        )
