"""Propeller mounting-height design sweep. Default SprayModel is not changed."""
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from aircraft import Aircraft, G, RHO_W
from atmosphere import AtmosphereConfig
from damage import SprayModel
from dynamics import HullContact
from flight_diagnostics import takeoff_progress
from mavlink_if import FlyingBoatVehicle, MAV_CMD_NAV_TAKEOFF
from ocean_directional import DirectionalOcean

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
HEIGHTS = (0.30, 0.50, 0.70, 0.90, 1.10, 1.30, 1.50)
SEEDS = list(range(9500, 9505))
WINDS = (-3.0, 3.0)


def hydrostatic(offset, eta=0.0):
    ac, hull = Aircraft(), HullContact()
    z = eta + hull.h_keel - ac.W / (RHO_W * G * hull.A_wp)
    spray = SprayModel(prop_z_offset=offset, D_prop=ac.prop.D_prop)
    return dict(z_cg_m=z,
                prop_clearance_m=spray.prop_top_clearance(z, eta),
                prop_bottom_clearance_m=z + offset - spray.D_prop / 2.0 - eta,
                T_factor=spray.spray_factor(z, eta, 0.0),
                static_thrust_N=ac.prop.T_static * spray.spray_factor(z, eta, 0.0))


def trial(offset, seed, wind, duration=15.0):
    vehicle = FlyingBoatVehicle(
        Aircraft(), DirectionalOcean(Hs=0.3, Tp=6, seed=seed),
        spatial=True, atmosphere=AtmosphereConfig(wind=(0, wind, 0), gust_rms=0.5),
        seed=seed)
    vehicle.spray.prop_z_offset = offset
    vehicle.arm()
    vehicle.send_command(MAV_CMD_NAV_TAKEOFF, dict(alt=8, speed=8.5))
    trajectory, status = [], 'time_limit'
    steps = int(round(duration / vehicle.dt))
    for _ in range(steps):
        vehicle.step()
        snap = vehicle.read_telemetry()[2]
        trajectory.append(snap)
        if vehicle.damage.failed:
            status = 'damage_failure'
            break
        if snap['N_water'] > 8 * vehicle.ac.W:
            status = 'impact_failure'
            break
        if abs(vehicle.y) > 50:
            status = 'lateral_limit'
            break
        if vehicle.z >= 8 and vehicle.Vx >= 8.5 and vehicle.lateral_success():
            status = 'success'
            break
    return dict(prop_z_offset_m=offset, seed=seed, wind_y=wind, status=status,
                t=vehicle.t, z=vehicle.z, Vx=vehicle.Vx, y=vehicle.y,
                T_factor=trajectory[-1]['T_factor'],
                prop_clearance_m=trajectory[-1]['prop_clearance_m'],
                prop_bottom_clearance_m=trajectory[-1]['prop_bottom_clearance_m'],
                diagnostics=takeoff_progress(trajectory))


def main():
    static = [dict(prop_z_offset_m=h, **hydrostatic(h)) for h in HEIGHTS]
    print('static', json.dumps(static, indent=2), flush=True)
    screening = []
    for offset in HEIGHTS:
        row = trial(offset, 9500, -3.0)
        screening.append(row)
        print(offset, row['status'], round(row['Vx'], 2), round(row['T_factor'], 3), flush=True)
    first_success = next((row['prop_z_offset_m'] for row in screening if row['status'] == 'success'), None)
    held_out = []
    if first_success is not None:
        for seed in SEEDS:
            for wind in WINDS:
                row = trial(first_success, seed, wind)
                held_out.append(row)
                print('held', seed, wind, row['status'], round(row['Vx'], 2), flush=True)
    report = dict(
        note='design sweep of SprayModel.prop_z_offset; production default 0.30 m is unchanged',
        Hs=0.3, Tp=6, gust_rms=0.5, duration=15, controller='native_takeoff',
        heights_m=list(HEIGHTS), static_calm_water=static,
        screening_seed9500_wind_m3=screening,
        first_success_offset_m=first_success, held_out_10=held_out,
        held_out_success=(f"{sum(r['status']=='success' for r in held_out)}/{len(held_out)}"
                          if held_out else 'n/a'),
        source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in sorted(ROOT.glob('*.py'))})
    # JSON key with minus is invalid as kwarg above; write via dict update instead if needed.
    (OUT / 'sweep.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
