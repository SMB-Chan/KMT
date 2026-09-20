"""Held-out native-inner FlightJev on the 84 kg airframe."""
import hashlib
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from aircraft import Aircraft
from atmosphere import AtmosphereConfig
from dynamics import hull_force
from flight_jev import FlightJev
from mavlink_if import FlyingBoatVehicle, MAV_CMD_NAV_LAND, MAV_CMD_NAV_TAKEOFF
from ocean_directional import DirectionalOcean
from ollama_pilot import observe, stabilize_lateral

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SEEDS = list(range(9500, 9505))
WINDS = (-3.0, 3.0)


def run_case(scenario, seed, wind):
    Hs = 0.3 if scenario == 'takeoff' else 1.5
    v = FlyingBoatVehicle(
        Aircraft(), DirectionalOcean(Hs=Hs, Tp=6, seed=seed), spatial=True,
        atmosphere=AtmosphereConfig(wind=(0, wind, 0), gust_rms=0.5), seed=seed)
    if scenario == 'landing':
        v.z = 25
        v.Vx = 13 * math.cos(math.radians(8))
        v.Vz = -13 * math.sin(math.radians(8))
    v.arm()
    v.send_command(MAV_CMD_NAV_TAKEOFF if scenario == 'takeoff' else MAV_CMD_NAV_LAND,
                   dict(alt=8 if scenario == 'takeoff' else 0, speed=8.5))
    pilot = FlightJev(spatial=True)
    mission = dict(scenario=scenario, target_altitude_m=8 if scenario == 'takeoff' else 0,
                   target_speed_m_s=8.5)
    status = 'time_limit'
    for _ in range(300 if scenario == 'takeoff' else 600):
        control, _ = pilot.decide(observe(v), mission)
        stabilize_lateral(v, control).apply(v)
        v.step()
        snap = v.read_telemetry()[2]
        if v.damage.failed:
            status = 'damage_failure'
            break
        if snap['N_water'] > 8 * v.ac.W:
            status = 'impact_failure'
            break
        if abs(v.y) > 50:
            status = 'lateral_limit'
            break
        if scenario == 'takeoff' and v.z >= 8 and v.Vx >= 8.5 and v.lateral_success():
            status = 'success'
            break
        if scenario == 'landing' and v.z - v.hull.h_keel < v.wave_elevation():
            contact = hull_force(v.z, v.Vx, v.Vz, v.wave_elevation(), v.hull)
            status = ('success' if v.lateral_success() and abs(v.Vz) < 1.5
                      and max(contact.N, snap['N_water']) < 3 * v.ac.W else 'hard_landing')
            break
    return dict(scenario=scenario, seed=seed, wind_y=wind, status=status, t=v.t, z=v.z, Vx=v.Vx, Vz=v.Vz)


def main():
    rows = []
    for scenario in ('takeoff', 'landing'):
        for seed in SEEDS:
            for wind in WINDS:
                row = run_case(scenario, seed, wind)
                rows.append(row)
                print(scenario, seed, wind, row['status'], round(row['t'], 2), flush=True)
    scores = {}
    for scenario in ('takeoff', 'landing'):
        group = [r for r in rows if r['scenario'] == scenario]
        scores[scenario] = f"{sum(r['status']=='success' for r in group)}/{len(group)}"
    report = dict(note='flight_jev_v1 heuristic System One over native setpoints',
                  scores=scores, rows=rows,
                  source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in sorted(ROOT.glob('*.py'))})
    (OUT / 'validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(scores), flush=True)


if __name__ == '__main__':
    main()
