from __future__ import annotations

import argparse
import json

from hama.earth.kmt_coastal import KmtCoastalEnvironment
from hama.earth.scenario import load_scenario
from hama.earth.world import EarthWorld
from hama.objects import EnvironmentalProbe


def build_world(spec):
    if spec.environment_provider != "kmt-coastal":
        raise ValueError(f"unsupported environment provider: {spec.environment_provider}")
    environment = KmtCoastalEnvironment.from_parameters(
        spec.environment_parameters, seed=spec.patch.seed)
    world = EarthWorld(spec.patch, environment)
    for obj in spec.objects:
        if obj.object_type != "environment_probe":
            raise ValueError(f"unsupported object type: {obj.object_type}")
        world.add_object(EnvironmentalProbe(obj.object_id, obj.position))
    return world


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run a HAMA Earth Sandbox scenario")
    parser.add_argument("scenario")
    args = parser.parse_args(argv)
    spec = load_scenario(args.scenario)
    world = build_world(spec)
    world.run(spec.duration_s, spec.dt_s)
    print(json.dumps(world.snapshot(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
