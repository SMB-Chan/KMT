"""Regression tests for the deterministic weather layer.

Covers: moist-air density (weather.py), precipitation/cloud/fog water
profiles, Messinger-style icing accretion and its penalties, the preset
roster, Weather as a drop-in Atmosphere replacement, the guarded effect
branches in spatial_dynamics.integrate, and the wiring in env.py,
mavlink_if.py and fly_ollama.py. Every default path (weather=None or no
weather configured) must reproduce the previous behaviour exactly.
"""
import contextlib
import io
import json
import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from aircraft import RHO, Aircraft
from atmosphere import (ISA_R, Atmosphere, AtmosphereConfig,
                        isa_pressure, isa_temperature)
from dynamics import HullContact, HullDrag
from env import EnvConfig, FlyingBoatEnv
from fly_ollama import parser, run
from mavlink_if import FlyingBoatVehicle
from ocean import Ocean
from ollama_pilot import PilotError
from spatial_dynamics import integrate
from weather import (CL_MAX_FACTOR_FLOOR, FOG_CEILING, FOG_LWC_REF,
                     FOG_VIS_REF, FOG_VIS_MAX, ICE_CD0_PER_KG,
                     ICE_CL_LOSS_PER_KG, ICE_MASS_MAX, ICE_PROP_LOSS_PER_KG,
                     ICING_T_LO, ICING_T_PEAK, PROP_FACTOR_FLOOR, PRESETS,
                     T_FREEZE, Weather, WeatherConfig, get_preset,
                     icing_rate, icing_temperature_factor,
                     liquid_water_content, moist_density, precip_fall_speed,
                     saturation_vapor_pressure, weather_asdict)

# Synthetic cold in-cloud conditions: icing at low altitude without
# depending on any preset's cloud base.
ICE_CFG = WeatherConfig(temp_offset_K=-25.0, rain_mm_h=5.0, snow=True,
                        cloud_base_m=0.0, cloud_top_m=2000.0,
                        cloud_cover=1.0)


def scene(seed=42):
    """Standard spatial-integration scene (aircraft, hull, sea surface)."""
    ac = Aircraft()
    hull = HullContact()
    hd = HullDrag(Bwl=ac.geom.Bwl, Lwl=ac.geom.Lwl)
    sea = Ocean(Hs=1.5, Tp=6.0, seed=seed)
    surf = lambda x, y, t: float(sea.eta(np.array([x]), t)[0])
    return ac, hull, hd, surf


def iced_weather(area, seconds=120.0, dt=0.1, seed=1):
    """Weather with accumulated ice after `seconds` at 30 m in the cloud."""
    w = Weather(ICE_CFG, AtmosphereConfig(), seed=seed)
    for _ in range(int(round(seconds / dt))):
        w.step(dt, 30.0, 11.0, area)
    return w


class MoistAirTests(unittest.TestCase):
    def test_dry_limit_is_ideal_gas(self):
        p, T = 101325.0, 288.15
        self.assertAlmostEqual(moist_density(p, T, 0.0), p / (ISA_R * T),
                               places=12)

    def test_humid_air_is_lighter(self):
        p, T = 101325.0, 293.15
        rhos = [moist_density(p, T, rh) for rh in (0.0, 0.25, 0.5, 1.0)]
        self.assertTrue(all(a > b for a, b in zip(rhos, rhos[1:])))

    def test_hot_and_low_pressure_air_is_lighter(self):
        p, T, rh = 101325.0, 288.15, 0.5
        self.assertLess(moist_density(p, T + 15.0, rh),
                        moist_density(p, T, rh))
        self.assertLess(moist_density(p - 2500.0, T, rh),
                        moist_density(p, T, rh))

    def test_saturation_pressure_branches(self):
        # Water/ice branches meet at 0 C with the Magnus constant 611.2 Pa.
        self.assertAlmostEqual(saturation_vapor_pressure(T_FREEZE), 611.2,
                               places=9)
        warm = saturation_vapor_pressure(293.15)
        cold = saturation_vapor_pressure(253.15)
        self.assertGreater(warm, 611.2)
        self.assertLess(cold, 611.2)
        self.assertGreater(warm, cold)

    def test_default_weather_density_near_isa(self):
        # Opt-in moist air (default rh=0.5) stays within ~0.4% of ISA at SL.
        rho = Weather().density(0.0)
        self.assertAlmostEqual(rho, RHO, delta=0.005)
        self.assertLess(rho, RHO)

    def test_humidity_clamped_and_invalid_inputs_raise(self):
        p, T = 90000.0, 280.0
        self.assertEqual(moist_density(p, T, 5.0), moist_density(p, T, 1.0))
        self.assertEqual(moist_density(p, T, -2.0), moist_density(p, T, 0.0))
        for bad_p, bad_T in ((-1.0, T), (0.0, T), (p, -10.0), (p, 0.0),
                             (p, float("nan")), (float("inf"), T)):
            with self.assertRaises(ValueError):
                moist_density(bad_p, bad_T, 0.5)


class HydrometeorTests(unittest.TestCase):
    def test_rain_fall_speed_increasing_and_capped(self):
        self.assertEqual(precip_fall_speed(0.0), 0.0)
        self.assertAlmostEqual(precip_fall_speed(1.0), 2.5, places=12)
        self.assertLess(precip_fall_speed(8.0), precip_fall_speed(35.0))
        self.assertEqual(precip_fall_speed(1.0e6), 9.0)

    def test_snow_fall_speed(self):
        self.assertEqual(precip_fall_speed(5.0, snow=True), 1.0)
        self.assertEqual(precip_fall_speed(0.0, snow=True), 0.0)

    def test_liquid_water_content_from_rate(self):
        v8 = precip_fall_speed(8.0)
        self.assertAlmostEqual(liquid_water_content(8.0),
                               8.0 / 3.6e6 / v8, places=15)
        self.assertEqual(liquid_water_content(0.0), 0.0)
        self.assertGreater(liquid_water_content(35.0),
                           liquid_water_content(8.0))
        # Snow falls slowly, so the same rate suspends far more water.
        self.assertGreater(liquid_water_content(5.0, snow=True),
                           liquid_water_content(5.0))

    def test_precip_only_below_cloud_base(self):
        w = Weather(get_preset('rain').weather)
        base = w.config.cloud_base_m
        self.assertGreater(w.precip_lwc(30.0), 0.0)
        self.assertGreater(w.precip_lwc(base), 0.0)
        self.assertEqual(w.precip_lwc(base + 1.0), 0.0)
        self.assertEqual(Weather().precip_lwc(30.0), 0.0)

    def test_cloud_layer_bounds_and_cover(self):
        w = Weather(get_preset('overcast').weather)
        c = w.config
        inside = c.cloud_cover * c.cloud_lwc_g_m3 * 1e-3
        self.assertAlmostEqual(w.cloud_lwc(1000.0), inside, places=15)
        self.assertGreater(w.cloud_lwc(c.cloud_base_m), 0.0)
        self.assertGreater(w.cloud_lwc(c.cloud_top_m), 0.0)
        self.assertEqual(w.cloud_lwc(c.cloud_base_m - 1.0), 0.0)
        self.assertEqual(w.cloud_lwc(c.cloud_top_m + 1.0), 0.0)

    def test_fog_visibility_humidity_and_ceiling_laws(self):
        w = Weather(get_preset('fog').weather)
        base = FOG_LWC_REF * math.sqrt(FOG_VIS_REF / w.config.visibility_m)
        self.assertAlmostEqual(w.fog_lwc(0.0), base, places=15)
        self.assertAlmostEqual(w.fog_lwc(FOG_CEILING / 2.0), base / 2.0,
                               places=15)
        self.assertEqual(w.fog_lwc(FOG_CEILING), 0.0)
        self.assertEqual(w.fog_lwc(FOG_CEILING + 100.0), 0.0)
        # Rain-reduced visibility above FOG_VIS_MAX is not fog water.
        for name in ('rain', 'storm'):
            self.assertEqual(Weather(get_preset(name).weather).fog_lwc(0.0),
                             0.0)
        # Humidity gate at 0.95.
        dry = Weather(WeatherConfig(visibility_m=100.0, humidity=0.9))
        wet = Weather(WeatherConfig(visibility_m=100.0, humidity=0.95))
        self.assertEqual(dry.fog_lwc(0.0), 0.0)
        self.assertGreater(wet.fog_lwc(0.0), 0.0)

    def test_total_lwc_is_the_sum(self):
        cfg = WeatherConfig(rain_mm_h=10.0, visibility_m=100.0,
                            humidity=1.0, cloud_base_m=20.0,
                            cloud_top_m=500.0, cloud_cover=1.0)
        w = Weather(cfg)
        z = 20.0  # at the cloud base: precip, cloud and fog coexist
        self.assertGreater(w.precip_lwc(z), 0.0)
        self.assertGreater(w.cloud_lwc(z), 0.0)
        self.assertGreater(w.fog_lwc(z), 0.0)
        self.assertAlmostEqual(
            w.total_lwc(z),
            w.precip_lwc(z) + w.cloud_lwc(z) + w.fog_lwc(z), places=18)


class IcingLawTests(unittest.TestCase):
    def test_temperature_factor_profile(self):
        f = icing_temperature_factor
        self.assertEqual(f(T_FREEZE), 0.0)
        self.assertEqual(f(T_FREEZE + 10.0), 0.0)
        self.assertEqual(f(ICING_T_PEAK), 1.0)
        self.assertEqual(f(ICING_T_LO), 0.0)
        self.assertEqual(f(ICING_T_LO - 10.0), 0.0)
        # Midpoints of the two linear ramps.
        self.assertAlmostEqual(f((T_FREEZE + ICING_T_PEAK) / 2.0), 0.5,
                               places=12)
        self.assertAlmostEqual(f((ICING_T_PEAK + ICING_T_LO) / 2.0), 0.5,
                               places=12)

    def test_temperature_factor_monotone_on_both_ramps(self):
        f = icing_temperature_factor
        up = [f(T_FREEZE - 0.5 * i) for i in range(1, 17)]  # 0 -> -8 C
        self.assertTrue(all(a < b for a, b in zip(up, up[1:])))
        down = [f(ICING_T_PEAK - 0.5 * i) for i in range(0, 41)]  # -8 -> -28 C
        self.assertTrue(all(a > b for a, b in zip(down, down[1:])))

    def test_icing_rate_scaling_and_snow_factor(self):
        T, lwc, V, S = ICING_T_PEAK, 0.5e-3, 50.0, 22.5
        base = icing_rate(T, lwc, V, S)
        self.assertAlmostEqual(base, 0.45 * lwc * V * S, places=12)
        self.assertAlmostEqual(icing_rate(T, lwc, 2 * V, S), 2 * base,
                               places=12)
        self.assertAlmostEqual(icing_rate(T, lwc, V, 2 * S), 2 * base,
                               places=12)
        self.assertAlmostEqual(icing_rate(T, lwc, V, S, snow=True),
                               0.4 * base, places=12)
        self.assertEqual(icing_rate(T, 0.0, V, S), 0.0)
        self.assertEqual(icing_rate(T, lwc, V, 0.0), 0.0)
        self.assertEqual(icing_rate(T_FREEZE + 1.0, lwc, V, S), 0.0)
        self.assertEqual(icing_rate(ICING_T_LO - 1.0, lwc, V, S), 0.0)

    def test_step_accumulates_monotonically_and_caps(self):
        w = Weather(ICE_CFG, AtmosphereConfig(), seed=1)
        masses = [w.step(0.1, 30.0, 11.0, 22.5) for _ in range(1200)]
        self.assertTrue(all(a <= b for a, b in zip(masses, masses[1:])))
        self.assertAlmostEqual(masses[-1], 1.4278, delta=0.01)
        for _ in range(40000):
            w.step(0.1, 30.0, 11.0, 22.5)
        self.assertEqual(w.ice_mass_kg, ICE_MASS_MAX)

    def test_step_rejects_invalid_dt_and_reset_clears(self):
        w = Weather(ICE_CFG, AtmosphereConfig(), seed=1)
        w.step(1.0, 30.0, 11.0, 22.5)
        self.assertGreater(w.ice_mass_kg, 0.0)
        for bad in (-0.1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                w.step(bad, 30.0, 11.0, 22.5)
        w.reset()
        self.assertEqual(w.ice_mass_kg, 0.0)

    def test_ice_penalties_formulas_and_floors(self):
        w = Weather()
        self.assertEqual(w.ice_penalties(), (0.0, 1.0, 1.0))
        w.ice_mass_kg = 2.0
        cd, cl_f, prop = w.ice_penalties()
        self.assertAlmostEqual(cd, ICE_CD0_PER_KG * 2.0, places=15)
        self.assertAlmostEqual(cl_f, 1.0 - ICE_CL_LOSS_PER_KG * 2.0,
                               places=15)
        self.assertAlmostEqual(prop, 1.0 - ICE_PROP_LOSS_PER_KG * 2.0,
                               places=15)
        w.ice_mass_kg = 100.0  # beyond the physical cap: floors apply
        _, cl_f, prop = w.ice_penalties()
        self.assertEqual(cl_f, CL_MAX_FACTOR_FLOOR)
        self.assertEqual(prop, PROP_FACTOR_FLOOR)


class WeatherConfigTests(unittest.TestCase):
    def test_invalid_values_raise(self):
        for kwargs in (dict(humidity=1.5), dict(humidity=-0.1),
                       dict(rain_mm_h=-1.0), dict(visibility_m=0.0),
                       dict(visibility_m=-5.0),
                       dict(cloud_base_m=3000.0, cloud_top_m=2000.0),
                       dict(cloud_base_m=-1.0), dict(cloud_cover=1.5),
                       dict(cloud_cover=-0.1), dict(cloud_lwc_g_m3=-1.0),
                       dict(lightning_risk=1.5), dict(lightning_risk=-0.1),
                       dict(temp_offset_K=float("nan")),
                       dict(pressure_offset_Pa=float("inf")),
                       dict(downdraft_m_s=float("nan"))):
            with self.assertRaises(ValueError):
                WeatherConfig(**kwargs)

    def test_default_config_is_neutral(self):
        w = Weather()
        e = w.effects(0.0, 30.0, 11.0, area=22.5)
        self.assertEqual(e.condition, "clear")
        self.assertEqual(e.prop_factor, 1.0)
        self.assertEqual(e.extra_cd, 0.0)
        self.assertEqual(e.cl_max_factor, 1.0)
        self.assertEqual(e.lwc_kg_m3, 0.0)
        self.assertEqual(e.icing_rate_kg_s, 0.0)
        self.assertEqual(e.lightning_risk, 0.0)
        self.assertAlmostEqual(e.temperature_K, isa_temperature(30.0),
                               places=12)

    def test_asdict_is_json_friendly(self):
        d = weather_asdict(get_preset("storm").weather)
        self.assertEqual(d["rain_mm_h"], 35.0)
        self.assertEqual(d["snow"], False)
        self.assertEqual(json.loads(json.dumps(d)), d)

    def test_constructor_type_checks(self):
        with self.assertRaises(TypeError):
            Weather("storm")
        with self.assertRaises(TypeError):
            Weather(WeatherConfig(), Atmosphere(AtmosphereConfig()))
        Weather(WeatherConfig(), None, seed=3)  # atmosphere optional


class PresetTests(unittest.TestCase):
    NAMES = ["clear", "fog", "heat_wave", "low_pressure", "overcast",
             "rain", "snow", "storm"]

    def test_roster_and_lookup(self):
        self.assertEqual(sorted(PRESETS), self.NAMES)
        for name in self.NAMES:
            self.assertEqual(get_preset(name).name, name)
        with self.assertRaises(ValueError):
            get_preset("blizzard")

    def test_presets_valid_and_bounded(self):
        for name in self.NAMES:
            p = get_preset(name)
            w = Weather(p.weather, p.atmosphere, seed=3)
            e = w.effects(1.0, 30.0, 11.0, area=22.5)
            self.assertTrue(isinstance(p.description_ja, str)
                            and p.description_ja)
            self.assertGreater(e.prop_factor, 0.0)
            self.assertLessEqual(e.prop_factor, 1.0)
            self.assertGreaterEqual(e.extra_cd, 0.0)
            self.assertGreater(e.cl_max_factor, 0.0)
            self.assertLessEqual(e.cl_max_factor, 1.0)
            self.assertTrue(math.isfinite(e.density_kg_m3))
            self.assertTrue(all(math.isfinite(v) for v in e.wind_m_s))

    def test_density_ordering_follows_synoptic_laws(self):
        rho = {n: Weather(get_preset(n).weather,
                          get_preset(n).atmosphere).density(0.0)
               for n in ("clear", "heat_wave", "low_pressure", "snow")}
        self.assertLess(rho["heat_wave"], rho["clear"])
        self.assertLess(rho["low_pressure"], rho["clear"])
        self.assertGreater(rho["snow"], rho["clear"])

    def test_condition_labels(self):
        cond = {n: Weather(get_preset(n).weather,
                           get_preset(n).atmosphere).condition_label(30.0)
                for n in self.NAMES}
        self.assertEqual(cond["clear"], "clear")
        self.assertEqual(cond["heat_wave"], "heat")
        self.assertEqual(cond["overcast"], "overcast")
        self.assertEqual(cond["fog"], "fog")
        self.assertEqual(cond["rain"], "rain")
        self.assertEqual(cond["storm"], "storm")
        self.assertEqual(cond["snow"], "snow+icing")

    def test_storm_downdraft_shifts_wind_z_only(self):
        p = get_preset("storm")
        calm = replace(p.weather, downdraft_m_s=0.0)
        w_storm = Weather(p.weather, p.atmosphere, seed=9)
        w_calm = Weather(calm, p.atmosphere, seed=9)
        for t in (0.0, 0.7, 3.3, 11.1):
            a = w_storm.wind(t, 30.0)
            b = w_calm.wind(t, 30.0)
            self.assertEqual(a[0], b[0])
            self.assertEqual(a[1], b[1])
            self.assertAlmostEqual(a[2], b[2] + p.weather.downdraft_m_s,
                                   places=12)
            self.assertLess(a[2], b[2])

    def test_storm_prop_loss_and_rain_drag(self):
        w = Weather(get_preset("storm").weather,
                    get_preset("storm").atmosphere)
        e = w.effects(0.0, 30.0, 11.0, area=22.5)  # below the 300 m base
        self.assertGreater(e.prop_factor, 0.90)
        self.assertLess(e.prop_factor, 0.95)
        self.assertGreater(e.extra_cd, 0.0)
        self.assertGreater(e.precip_lwc_kg_m3, 0.0)
        self.assertEqual(e.fog_lwc_kg_m3, 0.0)
        # Above the cloud base there is no falling rain.
        self.assertEqual(w.prop_rain_factor(1000.0), 1.0)


class AtmosphereInterfaceTests(unittest.TestCase):
    def test_drop_in_wind_bitwise_when_downdraft_zero(self):
        cfg = AtmosphereConfig(wind=(3.0, -2.0, 0.5), gust_rms=0.8,
                               gust_model="dryden")
        a = Atmosphere(cfg, seed=7)
        w = Weather(WeatherConfig(), cfg, seed=7)
        for t in (0.0, 0.31, 1.7, 5.5):
            for h in (0.0, 10.0, 30.0, 120.0):
                np.testing.assert_array_equal(w.wind(t, h), a.wind(t, h))
                self.assertEqual(w.shear_factor(h), a.shear_factor(h))

    def test_temperature_pressure_offsets(self):
        w = Weather(WeatherConfig(temp_offset_K=-6.5,
                                  pressure_offset_Pa=-1200.0))
        for h in (0.0, 100.0, 500.0):
            self.assertAlmostEqual(w.temperature(h),
                                   isa_temperature(h) - 6.5, places=12)
            self.assertAlmostEqual(w.pressure(h),
                                   isa_pressure(h) - 1200.0, places=9)
            self.assertAlmostEqual(
                w.density(h),
                moist_density(w.pressure(h), w.temperature(h),
                              w.config.humidity), places=15)

    def test_wind_returns_fresh_arrays(self):
        w = Weather(WeatherConfig(downdraft_m_s=-2.0))
        v1 = w.wind(0.0, 10.0)
        v1[:] = 99.0
        v2 = w.wind(0.0, 10.0)
        self.assertTrue(np.all(np.abs(v2) < 50.0))

    def test_determinism_same_seed(self):
        cfg = AtmosphereConfig(wind=(-4.0, 2.0, 0.0), gust_rms=1.2)
        wc = get_preset("rain").weather
        w1 = Weather(wc, cfg, seed=13)
        w2 = Weather(wc, cfg, seed=13)
        for t in (0.0, 1.23, 9.9):
            np.testing.assert_array_equal(w1.wind(t, 30.0),
                                          w2.wind(t, 30.0))
        e1 = w1.effects(2.0, 30.0, 11.0, area=22.5)
        e2 = w2.effects(2.0, 30.0, 11.0, area=22.5)
        self.assertEqual(e1, e2)


class _StubWeather:
    """Minimal weather stand-in for validation branches."""

    def __init__(self, prop_factor=1.0, extra_cd=0.0, cl_max_factor=1.0):
        self._fx = SimpleNamespace(prop_factor=prop_factor,
                                   extra_cd=extra_cd,
                                   cl_max_factor=cl_max_factor)

    def effects(self, t, altitude, airspeed, area=None):
        return self._fx


class IntegrateWeatherTests(unittest.TestCase):
    KW = dict(dt=0.05, t=1.0, pitch=0.05, throttle=0.8,
              bank_command=0.0, rudder_command=0.0)
    STATE = [0.0, 0.0, 30.0, 9.0, 0.0, 0.0, 0.0, 0.0]

    def test_weather_none_is_bitwise_identical(self):
        ac, hull, hd, surf = scene()
        atm = Atmosphere(AtmosphereConfig(wind=(2.0, -1.0, 0.0),
                                          gust_rms=0.6), seed=5)
        s1, f1 = integrate(ac, hull, hd, atm, surf, self.STATE, **self.KW)
        s2, f2 = integrate(ac, hull, hd, atm, surf, self.STATE,
                           weather=None, **self.KW)
        np.testing.assert_array_equal(s1, s2)
        for k in ("T", "L", "D", "N_water"):
            self.assertEqual(f1[k], f2[k])

    def test_neutral_weather_is_bitwise_identical(self):
        ac, hull, hd, surf = scene()
        cfg = AtmosphereConfig(wind=(2.0, -1.0, 0.0), gust_rms=0.6)
        atm = Atmosphere(cfg, seed=5)
        w = Weather(WeatherConfig(), cfg, seed=5)
        s1, f1 = integrate(ac, hull, hd, atm, surf, self.STATE, **self.KW)
        s2, f2 = integrate(ac, hull, hd, atm, surf, self.STATE, weather=w,
                           **self.KW)
        np.testing.assert_array_equal(s1, s2)
        for k in ("T", "L", "D", "N_water"):
            self.assertEqual(f1[k], f2[k])

    def test_ice_caps_cl_at_stall(self):
        ac, hull, hd, surf = scene()
        w = iced_weather(ac.geom.S)
        _, cl_f, prop = w.ice_penalties()
        self.assertLess(cl_f, 1.0)
        # Single substep from an identical state: alpha and q match, so the
        # lift ratio isolates the CL_max cap exactly.
        kw = dict(dt=0.001, t=1.0, pitch=0.25, throttle=1.0,
                  bank_command=0.0, rudder_command=0.0,
                  extra_mass=w.ice_mass_kg)
        _, fi = integrate(ac, hull, hd, w, surf, self.STATE, weather=w, **kw)
        _, fc = integrate(ac, hull, hd, w, surf, self.STATE, **kw)
        self.assertAlmostEqual(fi["L"] / fc["L"], cl_f, places=9)
        self.assertAlmostEqual(fi["T"] / fc["T"], prop, places=9)
        # Roughness drag shows below stall, where the CL cap is not active
        # (at stall the capped CL also cuts induced drag, so net D may fall).
        kw2 = dict(kw, pitch=0.05)
        _, gi = integrate(ac, hull, hd, w, surf, self.STATE, weather=w, **kw2)
        _, gc = integrate(ac, hull, hd, w, surf, self.STATE, **kw2)
        self.assertEqual(gi["L"], gc["L"])
        self.assertGreater(gi["D"], gc["D"])

    def test_rain_adds_drag_and_robs_thrust_only(self):
        ac, hull, hd, surf = scene()
        p = get_preset("rain")
        w = Weather(p.weather, p.atmosphere, seed=2)
        kw = dict(dt=0.001, t=1.0, pitch=0.05, throttle=1.0,
                  bank_command=0.0, rudder_command=0.0)
        _, fw = integrate(ac, hull, hd, w, surf, self.STATE, weather=w, **kw)
        _, fb = integrate(ac, hull, hd, w, surf, self.STATE, **kw)
        self.assertGreater(fw["D"], fb["D"])
        self.assertLess(fw["T"], fb["T"])
        # No ice -> CL is untouched; with identical states lift is bitwise.
        self.assertEqual(fw["L"], fb["L"])

    def test_invalid_effects_raise(self):
        ac, hull, hd, surf = scene()
        atm = Atmosphere(AtmosphereConfig(), seed=1)
        for stub in (_StubWeather(prop_factor=0.0),
                     _StubWeather(prop_factor=1.5),
                     _StubWeather(extra_cd=-0.1),
                     _StubWeather(cl_max_factor=0.0),
                     _StubWeather(cl_max_factor=1.2)):
            with self.assertRaises(ValueError):
                integrate(ac, hull, hd, atm, surf, self.STATE,
                          weather=stub, **self.KW)


class EnvWeatherTests(unittest.TestCase):
    def test_weather_requires_spatial(self):
        with self.assertRaises(ValueError):
            FlyingBoatEnv(Aircraft(), EnvConfig(weather=WeatherConfig()))

    def test_episode_conditions_weather_field(self):
        env = FlyingBoatEnv(Aircraft(), EnvConfig())
        env.reset(seed=1)
        self.assertIsNone(env.episode_conditions["weather"])
        env = FlyingBoatEnv(Aircraft(),
                            EnvConfig(spatial=True, weather=ICE_CFG))
        env.reset(seed=1)
        conds = env.episode_conditions["weather"]
        self.assertEqual(conds["rain_mm_h"], 5.0)
        self.assertTrue(conds["snow"])

    def test_icing_accumulates_through_steps(self):
        cfg = EnvConfig(spatial=True, max_steps=6, weather=ICE_CFG)
        env = FlyingBoatEnv(Aircraft(), cfg)
        env.reset(seed=3)
        self.assertIs(env._atmosphere, env._weather)
        zero = np.zeros(4, dtype=np.float32)
        prev = 0.0
        for _ in range(6):
            _, _, _, info = env.step(zero)
            self.assertGreater(info["ice_mass_kg"], prev)
            self.assertEqual(info["ice_mass_kg"], env._ice_mass)
            self.assertIn("icing", info["weather"]["condition"])
            prev = info["ice_mass_kg"]
        self.assertGreater(prev, 0.0)

    def test_no_weather_keys_without_configuration(self):
        cfg = EnvConfig(spatial=True, max_steps=3,
                        atmosphere=AtmosphereConfig())
        env = FlyingBoatEnv(Aircraft(), cfg)
        env.reset(seed=3)
        self.assertIsNone(env._weather)
        _, _, _, info = env.step(np.zeros(4, dtype=np.float32))
        self.assertNotIn("ice_mass_kg", info)
        self.assertNotIn("weather", info)

    def test_storm_determinism_and_downdraft_wiring(self):
        p = get_preset("storm")
        calm = replace(p.weather, downdraft_m_s=0.0)
        states, zs = [], []
        for wc in (p.weather, p.weather, calm):
            env = FlyingBoatEnv(Aircraft(), EnvConfig(
                spatial=True, max_steps=8, weather=wc,
                atmosphere=p.atmosphere))
            env.reset(seed=11)
            zs.append(float(env._atmosphere.wind(0.0, 10.0)[2]))
            for _ in range(8):
                env.step(np.zeros(4, dtype=np.float32))
            states.append(env._state.copy())
        np.testing.assert_array_equal(states[0], states[1])
        self.assertAlmostEqual(zs[0], zs[2] - 3.0, places=12)


class MavlinkWeatherTests(unittest.TestCase):
    def sea(self):
        return Ocean(Hs=0.3, Tp=6.0, seed=42)

    def test_type_and_spatial_guards(self):
        with self.assertRaises(TypeError):
            FlyingBoatVehicle(Aircraft(), self.sea(), weather="storm")
        with self.assertRaises(TypeError):
            FlyingBoatVehicle(Aircraft(), self.sea(), spatial=True,
                              weather=get_preset("storm"))
        with self.assertRaises(ValueError):
            FlyingBoatVehicle(Aircraft(), self.sea(),
                              weather=WeatherConfig())

    def test_ice_accumulates_and_reaches_snapshot(self):
        v = FlyingBoatVehicle(Aircraft(), self.sea(), spatial=True,
                              atmosphere=AtmosphereConfig(),
                              weather=ICE_CFG, seed=5)
        self.assertIs(v.atmosphere, v.weather)
        v.arm()
        for _ in range(20):
            v.step(action=[0.6, 2.0, 0.0, 0.0])
        self.assertGreater(v.ice_mass, 0.0)
        snap = v.read_telemetry()[2]
        self.assertEqual(snap["ice_mass_kg"], v.ice_mass)
        self.assertIn("icing", snap["weather"]["condition"])
        self.assertGreater(snap["weather"]["ice_mass_kg"], 0.0)

    def test_reset_clears_ice(self):
        v = FlyingBoatVehicle(Aircraft(), self.sea(), spatial=True,
                              weather=ICE_CFG, seed=5)
        v.arm()
        for _ in range(20):
            v.step(action=[0.6, 2.0, 0.0, 0.0])
        self.assertGreater(v.ice_mass, 0.0)
        v.reset(7)
        self.assertEqual(v.ice_mass, 0.0)
        self.assertIsNotNone(v.weather)

    def test_default_vehicle_has_no_weather_keys(self):
        v = FlyingBoatVehicle(Aircraft(), self.sea(), spatial=True,
                              atmosphere=AtmosphereConfig(), seed=5)
        self.assertIsNone(v.weather)
        v.arm()
        v.step(action=[0.6, 2.0, 0.0, 0.0])
        snap = v.read_telemetry()[2]
        self.assertNotIn("ice_mass_kg", snap)
        self.assertNotIn("weather", snap)
        self.assertEqual(v.ice_mass, 0.0)


class RunnerWeatherTests(unittest.TestCase):
    def pilot(self):
        pilot = Mock()
        pilot.check_model.return_value = {"name": "mock"}
        pilot.decide.side_effect = PilotError("offline")
        return pilot

    def test_parser_rejects_unknown_preset(self):
        with self.assertRaises(SystemExit):
            with contextlib.redirect_stderr(io.StringIO()):
                parser().parse_args(["--weather", "blizzard"])

    def test_weather_requires_spatial_flag(self):
        args = parser().parse_args(["--weather", "storm"])
        with self.assertRaises(ValueError):
            run(args, self.pilot())

    def test_preset_supplies_wind_gust_and_weather(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "flight"
            args = parser().parse_args(
                ["--spatial", "--weather", "storm", "--duration", ".2",
                 "--interval", ".1", "--output", str(target)])
            with contextlib.redirect_stdout(io.StringIO()):
                summary = run(args, self.pilot())
            self.assertGreaterEqual(summary["fallback_decisions"], 1)
            cfg = json.loads((target / "config.json").read_text())
            self.assertEqual(cfg["atmosphere"]["wind"], [-12.0, 5.0, 0.0])
            self.assertEqual(cfg["atmosphere"]["gust_rms"], 3.5)
            self.assertEqual(cfg["atmosphere"]["gust_model"], "dryden")
            self.assertEqual(cfg["weather"]["rain_mm_h"], 35.0)
            self.assertEqual(cfg["weather"]["downdraft_m_s"], -3.0)
            rows = [json.loads(line) for line in
                    (target / "trajectory.jsonl").read_text().splitlines()]
            snap = rows[-1]["snapshot"]
            self.assertEqual(snap["weather"]["lightning_risk"], 0.8)
            self.assertEqual(snap["ice_mass_kg"], 0.0)  # warm storm: no ice

    def test_explicit_wind_overrides_preset(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "flight"
            args = parser().parse_args(
                ["--spatial", "--weather", "storm", "--wind", "0", "0", "0",
                 "--gust-rms", "0.5", "--duration", ".2", "--interval", ".1",
                 "--output", str(target)])
            with contextlib.redirect_stdout(io.StringIO()):
                run(args, self.pilot())
            cfg = json.loads((target / "config.json").read_text())
            self.assertEqual(cfg["atmosphere"]["wind"], [0.0, 0.0, 0.0])
            self.assertEqual(cfg["atmosphere"]["gust_rms"], 0.5)
            self.assertEqual(cfg["atmosphere"]["gust_model"], "sum4")
            self.assertEqual(cfg["weather"]["rain_mm_h"], 35.0)


class TrainWeatherTests(unittest.TestCase):
    def train_args(self, **kw):
        base = dict(tag=None, spatial=True, weather=None,
                    wind=(0., 0., 0.), gust_rms=0.0,
                    randomize_conditions=False, directional=False,
                    theta_mean_deg=0.0, spread_s=10, seed_per_episode=False,
                    hs=1.5, tp=6.0)
        base.update(kw)
        return SimpleNamespace(**base)

    def test_config_adopts_preset_wind_and_gusts(self):
        import train as train_mod
        cfg = train_mod._config_for("takeoff", self.train_args(weather="snow"))
        p = get_preset("snow")
        self.assertEqual(cfg.weather, p.weather)
        self.assertEqual(cfg.atmosphere.wind, tuple(p.atmosphere.wind))
        self.assertEqual(cfg.atmosphere.gust_rms, p.atmosphere.gust_rms)

    def test_explicit_wind_overrides_preset(self):
        import train as train_mod
        cfg = train_mod._config_for(
            "takeoff", self.train_args(weather="storm", wind=(1., 2., 3.),
                                       gust_rms=0.9))
        self.assertEqual(cfg.weather, get_preset("storm").weather)
        self.assertEqual(cfg.atmosphere.wind, (1., 2., 3.))
        self.assertEqual(cfg.atmosphere.gust_rms, 0.9)
        self.assertEqual(cfg.atmosphere.gust_model, "sum4")

    def test_default_config_unchanged(self):
        import train as train_mod
        cfg = train_mod._config_for("takeoff", self.train_args())
        self.assertIsNone(cfg.weather)
        self.assertEqual(cfg.atmosphere, AtmosphereConfig())

    def test_default_tag_includes_weather(self):
        import train as train_mod
        tag = train_mod._default_tag(
            "takeoff", self.train_args(weather="storm"))
        self.assertEqual(tag, "takeoff_spatial_storm")

    def test_main_requires_spatial_for_weather(self):
        import train as train_mod
        for argv in (["--weather", "storm"], ["--weather", "blizzard"]):
            with self.assertRaises(SystemExit):
                with contextlib.redirect_stderr(io.StringIO()):
                    train_mod.main(argv)
