"""Env-var performance envelope for native / FlightJev inner control."""
import hashlib
import json
import math
import os
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
OUT = Path(os.environ.get('HAMA_OUTPUT', Path(__file__).resolve().parent))


def floats(name, default):
    text = os.environ.get(name, default)
    return [float(part) for part in text.split(',') if part.strip()]


def ints(name, default):
    text = os.environ.get(name, default)
    return [int(part) for part in text.split(',') if part.strip()]


def run_case(pilot_name, scenario, seed, wind, Hs):
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
    jev = None
    if pilot_name.startswith('jev'):
        cal = os.environ.get('HAMA_JEV_CALIBRATOR', '')
        jev = FlightJev.load(cal, spatial=True) if cal else FlightJev(spatial=True)
    mission = dict(scenario=scenario, target_altitude_m=8 if scenario == 'takeoff' else 0,
                   target_speed_m_s=8.5)
    status = 'time_limit'
    for _ in range(300 if scenario == 'takeoff' else 600):
        if jev is not None:
            control, _ = jev.decide(observe(v), mission)
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
    return dict(pilot=pilot_name, scenario=scenario, seed=seed, wind_y=wind, Hs=Hs,
                status=status, t=v.t, z=v.z, Vx=v.Vx, Vz=v.Vz, y=v.y)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pilots = os.environ.get('HAMA_PILOT', 'native,jev').split(',')
    scenarios = os.environ.get('HAMA_SCENARIO', 'takeoff,landing').split(',')
    hs_list = floats('HAMA_HS', '0.3,1.5,2.5,3.5')
    winds = floats('HAMA_WIND_Y', '-5,-3,3,5')
    seeds = ints('HAMA_SEEDS', '9500,9501,9502')
    rows = []
    for pilot in pilots:
        for scenario in scenarios:
            hs_used = [h for h in hs_list if not (scenario == 'takeoff' and h < 0)]
            for Hs in hs_used:
                for seed in seeds:
                    for wind in winds:
                        row = run_case(pilot.strip(), scenario.strip(), seed, wind, Hs)
                        rows.append(row)
                        print(row['pilot'], row['scenario'], Hs, seed, wind,
                              row['status'], round(row['t'], 2), flush=True)
    scores = {}
    for row in rows:
        key = f"{row['pilot']}_{row['scenario']}_Hs{row['Hs']}_wy{row['wind_y']}"
        scores.setdefault(key, []).append(row['status'] == 'success')
    summary = {key: f"{sum(v)}/{len(v)}" for key, v in scores.items()}
    report = dict(
        env={name: os.environ.get(name) for name in (
            'HAMA_PILOT', 'HAMA_SCENARIO', 'HAMA_HS', 'HAMA_WIND_Y', 'HAMA_SEEDS',
            'HAMA_JEV_CALIBRATOR', 'HAMA_OUTPUT')},
        defaults=dict(pilots=pilots, scenarios=scenarios, Hs=hs_list, wind_y=winds, seeds=seeds),
        scores=summary, rows=rows,
        source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in sorted(ROOT.glob('*.py'))})
    (OUT / 'envelope.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
