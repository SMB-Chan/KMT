"""Native 10-cond after ground-effect landing settle law."""
import hashlib
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from aircraft import Aircraft
from atmosphere import AtmosphereConfig
from dynamics import hull_force
from mavlink_if import FlyingBoatVehicle, MAV_CMD_NAV_LAND, MAV_CMD_NAV_TAKEOFF
from ocean_directional import DirectionalOcean

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SEEDS = list(range(9500, 9505))
WINDS = (-3.0, 3.0)


def run_case(scenario, seed, wind, Hs, duration):
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
    status = 'time_limit'
    for _ in range(int(round(duration / v.dt))):
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
    return dict(scenario=scenario, seed=seed, wind_y=wind, Hs=Hs, status=status,
                t=v.t, z=v.z, Vx=v.Vx, Vz=v.Vz, y=v.y)


def score(rows):
    return f"{sum(r['status']=='success' for r in rows)}/{len(rows)}"


def main():
    rows = []
    for scenario, Hs, duration in (('takeoff', 0.3, 15.0), ('takeoff', 1.5, 15.0),
                                   ('landing', 1.5, 30.0)):
        for seed in SEEDS:
            for wind in WINDS:
                row = run_case(scenario, seed, wind, Hs, duration)
                rows.append(row)
                print(scenario, Hs, seed, wind, row['status'], round(row['t'], 2),
                      round(row['Vz'], 2), flush=True)
    grouped = {}
    for row in rows:
        grouped.setdefault(f"{row['scenario']}_Hs{row['Hs']}", []).append(row)
    report = dict(
        note='landing_setpoint idles and dumps lift in ground effect; aero/hull unchanged',
        scores={key: score(group) for key, group in grouped.items()},
        rows=rows,
        source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in sorted(ROOT.glob('*.py'))})
    (OUT / 'validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report['scores'], indent=2), flush=True)


if __name__ == '__main__':
    main()
