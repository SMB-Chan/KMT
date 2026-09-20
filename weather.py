"""Deterministic weather layer on top of the ISA atmosphere.

Models the synoptic deviations from ISA (temperature/pressure offsets),
humidity (moist-air density) and hydrometeors (rain/snow, cloud layers,
fog), plus their consequences for the drone:

* density altitude -- hot, low-pressure or humid air is lighter; lift and
  propeller thrust both scale with rho, so heat waves and lows directly
  degrade takeoff and climb performance;
* precipitation -- falling drops exchange momentum with the airframe
  (extra drag), wet the propeller (efficiency loss) and cut visibility;
* icing -- supercooled droplets (freezing temperature + liquid water)
  accrete on the wing: added mass, roughness drag, CL_max loss and a
  fouled propeller. Accretion peaks near -8 C and stops below ~-28 C
  where clouds glaciate;
* downdrafts -- storm cells add a mean sinking-air column (negative wind
  z), which robs climb performance;
* lightning -- hazard indicator only; strikes and failures are not
  simulated.

`Weather` implements the `Atmosphere` interface (temperature / pressure /
density / shear_factor / wind) so it drops into spatial_dynamics or env in
place of a bare Atmosphere, and adds `effects()` / `step()` for the
aircraft-specific penalties. Everything is analytic and deterministic for
a given seed; steady-state weather (no time-varying fronts) keeps the
gust process the only time-dependent component.
"""
from dataclasses import asdict, dataclass
import math

import numpy as np

from atmosphere import (ISA_R, Atmosphere, AtmosphereConfig,
                        isa_pressure, isa_temperature)

# ----- physical constants and tunable coefficients -----
R_V = 461.495            # specific gas constant of water vapour, J/(kg K)
T_FREEZE = 273.15        # K
ICING_T_PEAK = 265.15    # K (-8 C): maximum accretion rate
ICING_T_LO = 245.15      # K (-28 C): below this clouds glaciate, no icing
ICING_BETA = 0.45        # droplet collection efficiency on the wing
ICE_MASS_MAX = 12.0      # kg, cap on accumulated ice
ICE_CD0_PER_KG = 0.004   # roughness drag per kg of ice (whole-aircraft CD0)
ICE_CL_LOSS_PER_KG = 0.035   # CL_max fraction lost per kg of ice
ICE_PROP_LOSS_PER_KG = 0.02  # propeller efficiency fraction lost per kg
CL_MAX_FACTOR_FLOOR = 0.5
PROP_FACTOR_FLOOR = 0.6

RAIN_PROP_LOSS_MAX = 0.10       # propeller efficiency loss in torrential rain
RAIN_PROP_LWC_REF = 2.0e-6      # kg/m3 LWC where the loss saturates
SNOW_FALL_SPEED = 1.0           # m/s, aggregate flakes
SNOW_ICING_FACTOR = 0.4         # snow accretes less than supercooled drops

FOG_VIS_MAX = 1000.0     # m, visibility below which fog water is counted
                         # (rain-reduced visibility above this is not fog)
FOG_VIS_REF = 200.0      # m, reference visibility of FOG_LWC_REF
FOG_LWC_REF = 0.15e-3    # kg/m3 at FOG_VIS_REF (simple empirical law)
FOG_CEILING = 300.0      # m, fog water tapers linearly to zero here


# ----- moist-air and hydrometeor physics -----
def saturation_vapor_pressure(T):
    """Magnus formula in Pa; over water >= 0 C, over ice below."""
    t = float(T) - T_FREEZE
    if t >= 0.0:
        return 611.2 * math.exp(17.62 * t / (243.12 + t))
    return 611.2 * math.exp(22.46 * t / (272.62 + t))


def moist_density(p, T, rh):
    """Density of moist air in kg/m3.

    Water vapour displaces heavier dry air, so humid air is *lighter*
    than dry air at the same pressure and temperature.
    """
    T = float(T)
    if T <= 0.0 or p <= 0.0 or not math.isfinite(p) or not math.isfinite(T):
        raise ValueError('moist_density requires positive finite p and T')
    rh = min(max(float(rh), 0.0), 1.0)
    e = min(rh * saturation_vapor_pressure(T), p)
    return (p - e) / (ISA_R * T) + e / (R_V * T)


def precip_fall_speed(rain_mm_h, snow=False):
    """Mass-weighted terminal fall speed in m/s (empirical)."""
    if snow:
        return SNOW_FALL_SPEED if rain_mm_h > 0 else 0.0
    R = max(float(rain_mm_h), 0.0)
    return min(9.0, 2.5 * R ** 0.24) if R > 0.0 else 0.0


def liquid_water_content(rain_mm_h, snow=False):
    """Precipitation water content in kg/m3 from the rate (mm/h).

    Rate / (3.6e6 * fall speed) converts a mm/h flux into the suspended
    mass per volume that produces it.
    """
    v = precip_fall_speed(rain_mm_h, snow)
    if v <= 0.0:
        return 0.0
    return max(float(rain_mm_h), 0.0) / 3.6e6 / v


def fog_lwc(visibility_m, humidity):
    """Near-surface fog water content in kg/m3 (simple visibility law)."""
    if visibility_m >= FOG_VIS_MAX or humidity < 0.95:
        return 0.0
    return FOG_LWC_REF * math.sqrt(FOG_VIS_REF / max(float(visibility_m), 50.0))


def icing_temperature_factor(T):
    """Accretion efficiency 0..1: ramps up from 0 C, peaks near -8 C,
    and falls to zero by -28 C where supercooled water no longer exists."""
    T = float(T)
    if T >= T_FREEZE or T <= ICING_T_LO:
        return 0.0
    if T >= ICING_T_PEAK:
        return (T_FREEZE - T) / (T_FREEZE - ICING_T_PEAK)
    return (T - ICING_T_LO) / (ICING_T_PEAK - ICING_T_LO)


def icing_rate(T, lwc, airspeed, area, snow=False):
    """Messinger-style accretion rate in kg/s onto `area` (m2)."""
    f = icing_temperature_factor(T)
    if f <= 0.0 or lwc <= 0.0 or area <= 0.0:
        return 0.0
    rate = ICING_BETA * float(lwc) * max(float(airspeed), 0.0) * float(area) * f
    return rate * (SNOW_ICING_FACTOR if snow else 1.0)


# ----- configuration -----
@dataclass(frozen=True)
class WeatherConfig:
    temp_offset_K: float = 0.0        # synoptic offset from ISA temperature
    pressure_offset_Pa: float = 0.0   # high/low pressure system
    humidity: float = 0.5             # relative humidity 0..1 (held with height)
    rain_mm_h: float = 0.0            # precipitation rate, liquid-equivalent
    snow: bool = False                # frozen precipitation
    visibility_m: float = 20000.0     # slant visibility; fog when < ~1 km
    cloud_base_m: float = 2500.0      # cloud layer bounds
    cloud_top_m: float = 4000.0
    cloud_cover: float = 0.0          # 0..1, scales the in-cloud water content
    cloud_lwc_g_m3: float = 0.30      # in-cloud liquid water content, g/m3
    downdraft_m_s: float = 0.0        # mean vertical wind (negative = sinking)
    lightning_risk: float = 0.0       # 0..1 hazard indicator (no strike sim)

    def __post_init__(self):
        vals = (self.temp_offset_K, self.pressure_offset_Pa, self.humidity,
                self.rain_mm_h, self.visibility_m, self.cloud_base_m,
                self.cloud_top_m, self.cloud_cover, self.cloud_lwc_g_m3,
                self.downdraft_m_s, self.lightning_risk)
        if not all(math.isfinite(float(v)) for v in vals):
            raise ValueError('weather parameters must be finite')
        if not 0.0 <= self.humidity <= 1.0:
            raise ValueError('humidity must be within [0, 1]')
        if self.rain_mm_h < 0.0 or self.visibility_m <= 0.0:
            raise ValueError('rain_mm_h must be >= 0 and visibility_m > 0')
        if not 0.0 <= self.cloud_base_m < self.cloud_top_m:
            raise ValueError('cloud layer requires 0 <= base < top')
        if not 0.0 <= self.cloud_cover <= 1.0 or self.cloud_lwc_g_m3 < 0.0:
            raise ValueError('cloud_cover must be within [0, 1] and lwc >= 0')
        if not 0.0 <= self.lightning_risk <= 1.0:
            raise ValueError('lightning_risk must be within [0, 1]')


@dataclass(frozen=True)
class WeatherPreset:
    name: str
    weather: WeatherConfig
    atmosphere: AtmosphereConfig   # recommended wind/gusts for the preset
    description_ja: str


PRESETS = {p.name: p for p in (
    WeatherPreset('clear', WeatherConfig(), AtmosphereConfig(),
                  '晴天・ISA標準 (影響なし)'),
    WeatherPreset('heat_wave',
                  WeatherConfig(temp_offset_K=15.0, humidity=0.25,
                                visibility_m=15000.0),
                  AtmosphereConfig(wind=(0.0, 0.0, 0.0), gust_rms=0.3),
                  '猛暑+15K: 空気密度が約5%低下し揚力・推力が落ちる'),
    WeatherPreset('low_pressure',
                  WeatherConfig(temp_offset_K=2.0, pressure_offset_Pa=-2500.0,
                                humidity=0.6, visibility_m=12000.0,
                                cloud_base_m=1500.0, cloud_top_m=3500.0,
                                cloud_cover=0.3),
                  AtmosphereConfig(wind=(-4.0, -2.0, 0.0), gust_rms=0.8),
                  '低気圧接近: 気圧-25hPaで密度が約4%低下'),
    WeatherPreset('overcast',
                  WeatherConfig(temp_offset_K=-1.0, humidity=0.85,
                                visibility_m=8000.0, cloud_base_m=800.0,
                                cloud_top_m=2000.0, cloud_cover=0.7),
                  AtmosphereConfig(wind=(-5.0, 2.0, 0.0), gust_rms=1.0),
                  '曇天: 雲底800m、飛行経路には影響小'),
    WeatherPreset('fog',
                  WeatherConfig(temp_offset_K=-1.0, humidity=1.0,
                                visibility_m=150.0, cloud_base_m=0.0,
                                cloud_top_m=200.0, cloud_cover=1.0,
                                cloud_lwc_g_m3=0.10),
                  AtmosphereConfig(wind=(-1.0, 0.0, 0.0), gust_rms=0.2),
                  '濃霧: 視程150m、霧粒による僅かな付着と視界喪失'),
    WeatherPreset('rain',
                  WeatherConfig(temp_offset_K=-2.0, pressure_offset_Pa=-800.0,
                                humidity=0.95, rain_mm_h=8.0,
                                visibility_m=4000.0, cloud_base_m=600.0,
                                cloud_top_m=2500.0, cloud_cover=0.9),
                  AtmosphereConfig(wind=(-7.0, 3.0, 0.0), gust_rms=1.5),
                  '降雨8mm/h: プロペラ濡れ・雨粒抗力・視程4km'),
    WeatherPreset('storm',
                  WeatherConfig(temp_offset_K=-4.0, pressure_offset_Pa=-2500.0,
                                humidity=1.0, rain_mm_h=35.0,
                                visibility_m=1500.0, cloud_base_m=300.0,
                                cloud_top_m=6000.0, cloud_cover=1.0,
                                cloud_lwc_g_m3=0.60, downdraft_m_s=-3.0,
                                lightning_risk=0.8),
                  AtmosphereConfig(wind=(-12.0, 5.0, 0.0), gust_rms=3.5,
                                   gust_model='dryden'),
                  '雷雨: 豪雨35mm/h・下降気流-3m/s・強乱流・雷害リスク0.8'),
    WeatherPreset('snow',
                  WeatherConfig(temp_offset_K=-25.0, pressure_offset_Pa=-500.0,
                                humidity=0.9, rain_mm_h=5.0, snow=True,
                                visibility_m=2000.0, cloud_base_m=400.0,
                                cloud_top_m=2000.0, cloud_cover=1.0),
                  AtmosphereConfig(wind=(-4.0, -2.0, 0.0), gust_rms=1.0),
                  '降雪-10C: 着氷条件 (雪は氷として付着、効率は低下)'),
)}


def get_preset(name):
    try:
        return PRESETS[str(name)]
    except KeyError:
        raise ValueError(
            f'unknown weather preset {name!r}; choose from {sorted(PRESETS)}')


# ----- effect snapshot -----
@dataclass(frozen=True)
class WeatherEffects:
    t_s: float
    altitude_m: float
    airspeed_m_s: float
    condition: str
    temperature_K: float
    pressure_Pa: float
    density_kg_m3: float
    wind_m_s: tuple
    lwc_kg_m3: float          # total (precip + cloud + fog)
    precip_lwc_kg_m3: float
    cloud_lwc_kg_m3: float
    fog_lwc_kg_m3: float
    icing_rate_kg_s: float
    ice_mass_kg: float
    prop_factor: float        # propeller efficiency multiplier in (0, 1]
    extra_cd: float           # drag-coefficient increment on the wing area
    cl_max_factor: float      # CL_max multiplier in (0, 1]
    visibility_m: float
    lightning_risk: float


class Weather:
    """Atmosphere-interface weather model with aircraft-effect terms."""

    def __init__(self, config=None, atmosphere=None, seed=0):
        if config is None:
            config = WeatherConfig()
        if not isinstance(config, WeatherConfig):
            raise TypeError('config must be a WeatherConfig')
        if atmosphere is not None and not isinstance(atmosphere, AtmosphereConfig):
            raise TypeError('atmosphere must be an AtmosphereConfig or None')
        self.config = config
        self._atm = Atmosphere(atmosphere, seed=seed)
        self.ice_mass_kg = 0.0

    # ----- Atmosphere interface (drop-in replacement) -----
    def temperature(self, altitude):
        return isa_temperature(altitude) + self.config.temp_offset_K

    def pressure(self, altitude):
        return isa_pressure(altitude) + self.config.pressure_offset_Pa

    def density(self, altitude):
        return moist_density(self.pressure(altitude), self.temperature(altitude),
                             self.config.humidity)

    def shear_factor(self, altitude):
        return self._atm.shear_factor(altitude)

    def wind(self, t, altitude=None):
        w = self._atm.wind(t, altitude)
        if self.config.downdraft_m_s:
            w[2] = w[2] + self.config.downdraft_m_s
        return w

    # ----- hydrometeors -----
    def precip_lwc(self, altitude):
        """Falling precipitation water content; rain lives below cloud base."""
        c = self.config
        if c.rain_mm_h <= 0.0 or altitude > c.cloud_base_m:
            return 0.0
        return liquid_water_content(c.rain_mm_h, c.snow)

    def cloud_lwc(self, altitude):
        c = self.config
        if c.cloud_base_m <= altitude <= c.cloud_top_m:
            return c.cloud_cover * c.cloud_lwc_g_m3 * 1e-3
        return 0.0

    def fog_lwc(self, altitude):
        if altitude >= FOG_CEILING:
            return 0.0
        base = fog_lwc(self.config.visibility_m, self.config.humidity)
        return base * (1.0 - altitude / FOG_CEILING)

    def total_lwc(self, altitude):
        return (self.precip_lwc(altitude) + self.cloud_lwc(altitude)
                + self.fog_lwc(altitude))

    # ----- aircraft-specific terms -----
    def prop_rain_factor(self, altitude):
        lwc = self.precip_lwc(altitude)
        if lwc <= 0.0:
            return 1.0
        loss = RAIN_PROP_LOSS_MAX * min(1.0, lwc / RAIN_PROP_LWC_REF)
        return 1.0 - loss

    def rain_drag_coeff(self, altitude, airspeed, rho):
        """Momentum-exchange drag increment (CD units on the wing area)."""
        lwc = self.precip_lwc(altitude)
        if lwc <= 0.0 or rho <= 0.0:
            return 0.0
        v = max(float(airspeed), 1.0)
        vt = precip_fall_speed(self.config.rain_mm_h, self.config.snow)
        return lwc * (v + vt) / (0.5 * rho * v)

    def ice_penalties(self):
        """(extra_cd, cl_max_factor, prop_factor) for the current ice load."""
        m = self.ice_mass_kg
        return (ICE_CD0_PER_KG * m,
                max(CL_MAX_FACTOR_FLOOR, 1.0 - ICE_CL_LOSS_PER_KG * m),
                max(PROP_FACTOR_FLOOR, 1.0 - ICE_PROP_LOSS_PER_KG * m))

    def icing_rate(self, altitude, airspeed, area):
        return icing_rate(self.temperature(altitude), self.total_lwc(altitude),
                          airspeed, area, snow=self.config.snow)

    def condition_label(self, altitude):
        c = self.config
        labels = []
        if c.lightning_risk >= 0.5:
            labels.append('storm')
        elif c.rain_mm_h > 0.0:
            labels.append('snow' if c.snow else 'rain')
        if c.visibility_m < 1000.0:
            labels.append('fog')
        if (icing_temperature_factor(self.temperature(altitude)) > 0.0
                and self.total_lwc(altitude) > 0.0):
            labels.append('icing')
        if not labels:
            if c.cloud_cover >= 0.5:
                labels.append('overcast')
            elif c.temp_offset_K >= 10.0:
                labels.append('heat')
            else:
                labels.append('clear')
        return '+'.join(labels)

    def effects(self, t, altitude, airspeed, area=None):
        """Snapshot of every weather term acting on the aircraft."""
        rho = self.density(altitude)
        wind = self.wind(t, altitude)
        ice_cd, cl_f, ice_prop = self.ice_penalties()
        rate = self.icing_rate(altitude, airspeed, area) if area else 0.0
        prop = max(PROP_FACTOR_FLOOR,
                   self.prop_rain_factor(altitude) * ice_prop)
        return WeatherEffects(
            t_s=float(t), altitude_m=float(altitude),
            airspeed_m_s=float(airspeed),
            condition=self.condition_label(altitude),
            temperature_K=self.temperature(altitude),
            pressure_Pa=self.pressure(altitude), density_kg_m3=rho,
            wind_m_s=tuple(float(v) for v in wind),
            lwc_kg_m3=self.total_lwc(altitude),
            precip_lwc_kg_m3=self.precip_lwc(altitude),
            cloud_lwc_kg_m3=self.cloud_lwc(altitude),
            fog_lwc_kg_m3=self.fog_lwc(altitude),
            icing_rate_kg_s=rate, ice_mass_kg=self.ice_mass_kg,
            prop_factor=prop,
            extra_cd=ice_cd + self.rain_drag_coeff(altitude, airspeed, rho),
            cl_max_factor=cl_f,
            visibility_m=self.config.visibility_m,
            lightning_risk=self.config.lightning_risk)

    # ----- state -----
    def step(self, dt, altitude, airspeed, area):
        """Advance ice accretion by dt; returns the new ice mass in kg."""
        if not math.isfinite(float(dt)) or dt < 0.0:
            raise ValueError('dt must be finite and >= 0')
        rate = self.icing_rate(altitude, airspeed, area)
        self.ice_mass_kg = min(ICE_MASS_MAX, self.ice_mass_kg + rate * dt)
        return self.ice_mass_kg

    def reset(self):
        self.ice_mass_kg = 0.0

    def summary(self, t=0.0, altitude=30.0, airspeed=11.3, area=None):
        """JSON-friendly report of the current conditions (report/telemetry)."""
        e = self.effects(t, altitude, airspeed, area=area)
        return dict(condition=e.condition,
                    temperature_C=e.temperature_K - T_FREEZE,
                    pressure_hPa=e.pressure_Pa / 100.0,
                    density_kg_m3=e.density_kg_m3,
                    humidity=self.config.humidity,
                    visibility_m=e.visibility_m,
                    wind_m_s=list(e.wind_m_s),
                    lwc_g_m3=e.lwc_kg_m3 * 1e3,
                    icing_rate_kg_h=e.icing_rate_kg_s * 3600.0,
                    ice_mass_kg=e.ice_mass_kg,
                    prop_factor=e.prop_factor,
                    extra_cd=e.extra_cd,
                    cl_max_factor=e.cl_max_factor,
                    lightning_risk=e.lightning_risk)


def weather_asdict(config):
    return asdict(config)
