"""Held-out takeoff/landing after adopting SprayModel.prop_z_offset=0.90 m."""
import hashlib
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from aircraft import Aircraft
from atmosphere import AtmosphereConfig
from dynamics import hull_force
from flight_diagnostics import takeoff_progress
from mavlink_if import FlyingBoatVehicle, MAV_CMD_NAV_LAND, MAV_CMD_NAV_TAKEOFF
from ocean_directional import DirectionalOcean
from ollama_pilot import SpatialControl, observe
from policy_pilot import PolicyPilot

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SEEDS = list(range(9500, 9505))
WINDS = (-3.0, 3.0)
TAKEOFF_MODEL = ROOT / 'results/takeoff_preview_best_policy.npz'
LANDING_MODEL = ROOT / 'results/landing_preview_best_policy.npz'


def vehicle(seed, wind, Hs):
    return FlyingBoatVehicle(
        Aircraft(), DirectionalOcean(Hs=Hs, Tp=6, seed=seed), spatial=True,
        atmosphere=AtmosphereConfig(wind=(0, wind, 0), gust_rms=0.5), seed=seed)


def apply_control(v, controller, scenario, pilot):
    if controller == 'native':
        return
    command, _ = pilot.decide(observe(v), dict(scenario=scenario))
    bank, rudder = v.lateral_setpoint()
    SpatialControl(command.throttle, command.pitch_deg, math.degrees(bank), rudder).apply(v)


def run_case(scenario, controller, seed, wind, Hs, duration):
    v = vehicle(seed, wind, Hs)
    if scenario == 'landing':
        v.z = 25
        v.Vx = 13 * math.cos(math.radians(8))
        v.Vz = -13 * math.sin(math.radians(8))
    v.arm()
    v.send_command(MAV_CMD_NAV_TAKEOFF if scenario == 'takeoff' else MAV_CMD_NAV_LAND,
                   dict(alt=8 if scenario == 'takeoff' else 0, speed=8.5))
    pilot = PolicyPilot(TAKEOFF_MODEL if scenario == 'takeoff' else LANDING_MODEL)
    trajectory, status = [], 'time_limit'
    for _ in range(int(round(duration / v.dt))):
        apply_control(v, controller, scenario, pilot)
        v.step()
        snap = v.read_telemetry()[2]
        trajectory.append(snap)
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
    result = dict(scenario=scenario, controller=controller, seed=seed, wind_y=wind, Hs=Hs,
                  status=status, t=v.t, z=v.z, Vx=v.Vx, Vz=v.Vz, y=v.y,
                  T_factor=trajectory[-1]['T_factor'],
                  prop_clearance_m=trajectory[-1]['prop_clearance_m'],
                  prop_z_offset_m=v.spray.prop_z_offset)
    if scenario == 'takeoff':
        result['diagnostics'] = takeoff_progress(trajectory)
    return result


def score(rows):
    n = len(rows)
    return f"{sum(r['status']=='success' for r in rows)}/{n}"


def main():
    rows = []
    suites = [
        ('takeoff', 'native', 0.3, 15.0),
        ('takeoff', 'rl_with_lateral_feedback', 0.3, 15.0),
        ('landing', 'native', 1.5, 30.0),
        ('landing', 'rl_with_lateral_feedback', 1.5, 30.0),
        ('takeoff', 'native', 1.5, 15.0),
    ]
    for scenario, controller, Hs, duration in suites:
        for seed in SEEDS:
            for wind in WINDS:
                row = run_case(scenario, controller, seed, wind, Hs, duration)
                rows.append(row)
                print(scenario, controller, Hs, seed, wind, row['status'],
                      round(row['Vx'], 2), flush=True)
    grouped = {}
    for row in rows:
        key = f"{row['scenario']}_{row['controller']}_Hs{row['Hs']}"
        grouped.setdefault(key, []).append(row)
    report = dict(
        note='production SprayModel.prop_z_offset adopted at 0.90 m from takeoff_spray_001',
        prop_z_offset_m=0.90,
        seeds=SEEDS,
        scores={key: score(group) for key, group in grouped.items()},
        rows=rows,
        models=dict(takeoff=str(TAKEOFF_MODEL), landing=str(LANDING_MODEL),
                    takeoff_sha256=hashlib.sha256(TAKEOFF_MODEL.read_bytes()).hexdigest(),
                    landing_sha256=hashlib.sha256(LANDING_MODEL.read_bytes()).hexdigest()),
        source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in sorted(ROOT.glob('*.py'))})
    (OUT / 'validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report['scores'], indent=2), flush=True)


if __name__ == '__main__':
    main()
