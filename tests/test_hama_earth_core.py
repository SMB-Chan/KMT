import unittest
from datetime import datetime, timezone

from hama.earth import EarthPatchConfig, EarthWorld, EnvironmentSnapshot, GeoPoint, LocalPosition, ScenarioSpec
from hama.objects import EnvironmentalProbe


class ConstantEnvironment:
    def sample(self, position, time_s):
        return EnvironmentSnapshot(
            air_temperature_k=288.15, air_pressure_pa=101325.0,
            air_density_kg_m3=1.225, wind_m_s=(1.0, 2.0, 0.0),
            surface_elevation_m=0.1, water_depth_m=20.0,
            metadata={"time_s": time_s})


def patch():
    return EarthPatchConfig(
        "test", GeoPoint(35.0, 135.0), (1000, 1000, 200),
        datetime.now(timezone.utc), 7)


class EarthCoreTests(unittest.TestCase):
    def test_world_runs_generic_object(self):
        world = EarthWorld(patch(), ConstantEnvironment())
        probe = EnvironmentalProbe("p1", LocalPosition(0, 0, 10))
        world.add_object(probe)
        world.run(0.2, 0.05)
        self.assertAlmostEqual(world.time_s, 0.2)
        self.assertEqual(probe.samples, 4)
        self.assertEqual(probe.last_environment["wind_m_s"], [1.0, 2.0, 0.0])

    def test_duplicate_ids_rejected(self):
        world = EarthWorld(patch(), ConstantEnvironment())
        world.add_object(EnvironmentalProbe("p1", LocalPosition()))
        with self.assertRaises(ValueError):
            world.add_object(EnvironmentalProbe("p1", LocalPosition()))

    def test_scenario_contract(self):
        spec = ScenarioSpec.from_dict({
            "schema": "hama-earth-scenario/v0.1", "name": "demo",
            "world": {
                "origin": {"lat_deg": 34.5, "lon_deg": 133.8},
                "extent_m": [100, 100, 50],
                "start_time": "2026-09-22T06:00:00Z",
                "seed": 1
            },
            "environment": {"provider": "constant", "parameters": {}},
            "run": {"duration_s": 2, "dt_s": 0.1},
            "objects": [
                {"id": "probe", "type": "environment_probe",
                 "position_m": [1, 2, 3]}
            ]
        })
        self.assertEqual(spec.patch.origin.lat_deg, 34.5)
        self.assertEqual(spec.objects[0].position.as_tuple(), (1.0, 2.0, 3.0))

    def test_requires_timezone(self):
        with self.assertRaises(ValueError):
            EarthPatchConfig(
                "bad", GeoPoint(0, 0), (1, 1, 1), datetime(2026, 1, 1))


if __name__ == "__main__":
    unittest.main()
