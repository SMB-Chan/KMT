"""ISA troposphere and a logarithmic surface-layer wind.

Coordinates are world x/y/z (z up); wind components point toward motion.
Mean wind in the config is the value at z_ref (default 10 m). Gusts are a
seeded analytic process, not a weather forecast. Two harmonic gust models
are available: "sum4" (default, four equal-weight sinusoids per axis) and
"dryden" (eight log-spaced harmonics per axis weighted by the square root
of a Dryden-like spectrum, giving more energy at low frequency). Both are
deterministic for a given seed and scale the temporal RMS to gust_rms.
"""
from dataclasses import dataclass
import math
import numpy as np

from aircraft import RHO, G

GUST_MODELS = ("sum4", "dryden")
DRYDEN_N_HARMONICS = 8
DRYDEN_F_RANGE = (0.25, 8.0)   # multiples of 1/gust_period

ISA_T0 = 288.15
ISA_LAPSE = 0.0065
ISA_P0 = 101325.0
ISA_R = 287.05287
ISA_TROPOPAUSE = 11000.0
ISA_EXPONENT = G / (ISA_LAPSE * ISA_R) - 1.0


def isa_temperature(altitude):
    h = min(max(float(altitude), 0.0), ISA_TROPOPAUSE)
    return ISA_T0 - ISA_LAPSE * h


def isa_pressure(altitude):
    return ISA_P0 * (isa_temperature(altitude) / ISA_T0) ** (G / (ISA_LAPSE * ISA_R))


def isa_density(altitude):
    """ICAO ISA troposphere, pinned to RHO at sea level."""
    return RHO * (isa_temperature(altitude) / ISA_T0) ** ISA_EXPONENT


@dataclass(frozen=True)
class AtmosphereConfig:
    wind: tuple = (0.0, 0.0, 0.0)  # m/s at z_ref, world coordinates
    gust_rms: float = 0.0          # temporal RMS per component, m/s
    gust_period: float = 6.0       # characteristic seconds
    density_scale_height: float = 8500.0  # retained for config checks; density uses ISA
    z_ref: float = 10.0            # height where `wind` is specified, m
    z0: float = 0.001              # aerodynamic roughness over water, m
    gust_model: str = "sum4"       # "sum4" (default) or "dryden"

    def __post_init__(self):
        wind = np.asarray(self.wind, dtype=float)
        if wind.shape != (3,) or not np.isfinite(wind).all():
            raise ValueError('wind must contain three finite components')
        if (not all(math.isfinite(v) for v in (
                self.gust_rms, self.gust_period, self.density_scale_height,
                self.z_ref, self.z0))
                or self.gust_rms < 0 or self.gust_period <= 0
                or self.density_scale_height <= 0
                or self.z0 <= 0 or self.z_ref <= self.z0
                or self.gust_model not in GUST_MODELS):
            raise ValueError('invalid atmosphere parameters')


class Atmosphere:
    def __init__(self, config=None, seed=0):
        self.config = config or AtmosphereConfig()
        rng = np.random.default_rng(seed)
        if self.config.gust_model == "dryden":
            # Log-spaced harmonics weighted by the sqrt of a Dryden-like
            # temporal spectrum: w(f) ~ (1 + (f/f0)^2)^(-5/12), f0 = 1/gust_period.
            f = np.geomspace(DRYDEN_F_RANGE[0], DRYDEN_F_RANGE[1],
                             DRYDEN_N_HARMONICS)
            self.phase = rng.uniform(0, 2 * math.pi, (3, DRYDEN_N_HARMONICS))
            self.frequency = (np.tile(f, (3, 1)) * 2 * math.pi
                              / self.config.gust_period)
            self.weight = np.tile((1.0 + f ** 2) ** (-5.0 / 12.0), (3, 1))
        else:
            self.phase = rng.uniform(0, 2 * math.pi, (3, 4))
            self.frequency = rng.uniform(0.5, 1.5, (3, 4)) * 2 * math.pi / self.config.gust_period
            self.weight = np.ones((3, 4))
        # Temporal RMS of sum_i w_i sin(w_i t + phi_i) with independent
        # uniform phases is sqrt(sum_i w_i^2 / 2); normalise to unit RMS.
        self._gust_scale = np.sqrt(
            2.0 / (self.weight ** 2).sum(axis=1, keepdims=True))[:, 0]

    def temperature(self, altitude):
        return isa_temperature(altitude)

    def pressure(self, altitude):
        return isa_pressure(altitude)

    def density(self, altitude):
        return isa_density(altitude)

    def shear_factor(self, altitude):
        z = max(float(altitude), self.config.z0)
        return math.log(z / self.config.z0) / math.log(self.config.z_ref / self.config.z0)

    def wind(self, t, altitude=None):
        # Analytical process: repeated reads do not consume randomness.
        gust = (self.weight * np.sin(self.frequency * t + self.phase)).sum(axis=1)
        mean = np.asarray(self.config.wind, dtype=float)
        factor = 1.0 if altitude is None else self.shear_factor(altitude)
        gust = self.config.gust_rms * self._gust_scale * gust * factor
        return np.array([mean[0] * factor, mean[1] * factor, mean[2]], dtype=float) + gust
