"""Simplified atmosphere: exponential density and seeded smooth gusts.

Coordinates are world x/y/z (z up); wind components point toward motion.
This is a synthetic disturbance model, not a weather forecast or ISA model.
"""
from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class AtmosphereConfig:
    wind: tuple = (0.0, 0.0, 0.0)  # m/s, world coordinates
    gust_rms: float = 0.0          # temporal RMS per component, m/s
    gust_period: float = 6.0       # characteristic seconds
    density_scale_height: float = 8500.0  # metres

    def __post_init__(self):
        wind = np.asarray(self.wind, dtype=float)
        if wind.shape != (3,) or not np.isfinite(wind).all():
            raise ValueError('wind must contain three finite components')
        if (not all(math.isfinite(v) for v in (
                self.gust_rms, self.gust_period, self.density_scale_height))
                or self.gust_rms < 0 or self.gust_period <= 0
                or self.density_scale_height <= 0):
            raise ValueError('invalid atmosphere parameters')


class Atmosphere:
    def __init__(self, config=None, seed=0):
        self.config = config or AtmosphereConfig()
        rng = np.random.default_rng(seed)
        self.phase = rng.uniform(0, 2 * math.pi, (3, 4))
        self.frequency = rng.uniform(0.5, 1.5, (3, 4)) * 2 * math.pi / self.config.gust_period

    def wind(self, t):
        # Analytical process: repeated reads do not consume randomness.
        gust = np.sin(self.frequency * t + self.phase).sum(axis=1)
        return np.asarray(self.config.wind) + self.config.gust_rms * math.sqrt(2 / 4) * gust

    def density(self, altitude):
        return 1.225 * math.exp(-max(0.0, altitude) / self.config.density_scale_height)
