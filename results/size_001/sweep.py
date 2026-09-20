"""Similar-airframe size sweep. Default 15 m design is λ=1."""
import hashlib
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from aircraft import Aircraft, scaled_aircraft
from atmosphere import AtmosphereConfig
from damage import SprayModel
from dynamics import HullContact, HullDrag, hull_force
from mavlink_if import FlyingBoatVehicle, MAV_CMD_NAV_LAND, MAV_CMD_NAV_TAKEOFF
from ocean_directional import DirectionalOcean

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SCALES = (0.5, 0.75, 1.0, 1.25, 1.5)
SEEDS = (9500, 9501, 9502)
WINDS = (-3.0, 3.0)


def outfit(vehicle, lam):
    vehicle.hull = HullContact(h_keel=0.30 * lam, A_wp=0.55 * 2.6 * lam ** 2)
    vehicle.hd = HullDrag(Bwl=0.55 * lam, Lwl=2.6 * lam, S_wet=2.5 * lam ** 2)
    vehicle.spray = SprayModel(prop_z_offset=0.90 * lam, D_prop=vehicle.ac.prop.D_prop,
                               critical_clearance=1.5 * lam)
    vehicle.reset(vehicle._wind_seed)


def lateral_ok(v, lam):
    return abs(v.y) < 10 * lam and abs(v.Vy) < 1.5 * math.sqrt(lam) and abs(v.bank) < math.radians(10)


def run_case(lam, scenario, seed, wind):
    ac = scaled_aircraft(lam)
    v = FlyingBoatVehicle(
        ac, DirectionalOcean(Hs=1.5 * lam, Tp=6, seed=seed), spatial=True,
        atmosphere=AtmosphereConfig(wind=(0, wind, 0), gust_rms=0.5), seed=seed)
    outfit(v, lam)
    z_goal, v_goal = 8.0 * lam, 8.5 * math.sqrt(lam)
    if scenario == 'landing':
        v.z = 25.0 * lam
        speed = 13.0 * math.sqrt(lam)
        v.Vx = speed * math.cos(math.radians(8))
        v.Vz = -speed * math.sin(math.radians(8))
    v.arm()
    v.send_command(MAV_CMD_NAV_TAKEOFF if scenario == 'takeoff' else MAV_CMD_NAV_LAND,
                   dict(alt=z_goal if scenario == 'takeoff' else 0, speed=v_goal))
    duration = (15 if scenario == 'takeoff' else 30) * math.sqrt(lam)
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
        if abs(v.y) > 50 * lam:
            status = 'lateral_limit'
            break
        if scenario == 'takeoff' and v.z >= z_goal and v.Vx >= v_goal and lateral_ok(v, lam):
            status = 'success'
            break
        if scenario == 'landing' and v.z - v.hull.h_keel < v.wave_elevation():
            contact = hull_force(v.z, v.Vx, v.Vz, v.wave_elevation(), v.hull)
            vz_lim = 1.5 * math.sqrt(lam)
            status = ('success' if lateral_ok(v, lam) and abs(v.Vz) < vz_lim
                      and max(contact.N, snap['N_water']) < 3 * v.ac.W else 'hard_landing')
            break
    return dict(scale=lam, span_m=ac.geom.b, mass_kg=ac.mass.total,
                T_over_W=ac.prop.T_static / ac.W, V_stall=ac.V_stall,
                scenario=scenario, seed=seed, wind_y=wind, status=status,
                t=v.t, z=v.z, Vx=v.Vx, Vz=v.Vz, y=v.y)


def main():
    rows = []
    for lam in SCALES:
        ac = scaled_aircraft(lam)
        print('λ', lam, 'b', ac.geom.b, 'm', round(ac.mass.total, 1),
              'T/W', round(ac.prop.T_static / ac.W, 3), 'Vs', round(ac.V_stall, 2), flush=True)
        for scenario in ('takeoff', 'landing'):
            for seed in SEEDS:
                for wind in WINDS:
                    row = run_case(lam, scenario, seed, wind)
                    rows.append(row)
                    print(lam, scenario, seed, wind, row['status'], round(row['t'], 2), flush=True)
    scores = {}
    for row in rows:
        key = f"λ{row['scale']}_{row['scenario']}"
        scores.setdefault(key, []).append(row['status'] == 'success')
    summary = {k: f"{sum(v)}/{len(v)}" for k, v in scores.items()}
    report = dict(scores=summary, rows=rows,
                  source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in sorted(ROOT.glob('*.py'))})
    (OUT / 'size.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
