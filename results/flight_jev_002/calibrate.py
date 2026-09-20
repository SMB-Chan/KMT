"""Fit FlightJev noul calibrators from native rollouts."""
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from aircraft import Aircraft
from atmosphere import AtmosphereConfig
from flight_jev import FlightJev, features
from mavlink_if import FlyingBoatVehicle, MAV_CMD_NAV_LAND, MAV_CMD_NAV_TAKEOFF
from ocean_directional import DirectionalOcean
from ollama_pilot import observe, stabilize_lateral
from system_one import Logistic

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
TRAIN_SEEDS = list(range(9500, 9504))
TEST_SEEDS = [9504]
WINDS = (-3.0, 3.0)


def rollout(scenario, seed, wind):
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
    rows = []
    for _ in range(300 if scenario == 'takeoff' else 600):
        observation = observe(v)
        control, meta = pilot.decide(observation, mission)
        stabilize_lateral(v, control).apply(v)
        v.step()
        snap = v.read_telemetry()[2]
        rows.append(dict(observation=observation, z=v.z, Vx=v.Vx, Vz=v.Vz, t=v.t,
                         heuristic=meta['answers']))
        if v.z >= 8 and v.Vx >= 8.5 and scenario == 'takeoff':
            break
        if scenario == 'landing' and v.z - v.hull.h_keel < v.wave_elevation():
            break
        if snap['N_water'] > 8 * v.ac.W or abs(v.y) > 50 or v.damage.failed:
            break
    return rows


def label_rotate(rows, index):
    t0 = rows[index]['t']
    for row in rows[index:]:
        if row['t'] > t0 + 2.0:
            break
        if row['Vx'] >= 7.0:
            return 1.0
    return 0.0


def label_balloon(rows, index):
    t0 = rows[index]['t']
    for row in rows[index:]:
        if row['t'] > t0 + 1.0:
            break
        if row['z'] < 1.5 and row['Vz'] > -0.2:
            return 1.0
    return 0.0


def pack(scenario, seeds, label_fn):
    x, y, heuristic = [], [], []
    for seed in seeds:
        for wind in WINDS:
            rows = rollout(scenario, seed, wind)
            key = 'rotate_soon' if scenario == 'takeoff' else 'ballooning'
            for i, row in enumerate(rows):
                x.append(features(row['observation']))
                y.append(label_fn(rows, i))
                heuristic.append(row['heuristic'][key]['noul'])
    return np.array(x), np.array(y), np.array(heuristic)


def brier(p, y):
    return float(np.mean((p - y) ** 2))


def main():
    tx, ty, th = pack('takeoff', TRAIN_SEEDS, label_rotate)
    lx, ly, lh = pack('landing', TRAIN_SEEDS, label_balloon)
    rotate = Logistic.fit(tx, ty)
    balloon = Logistic.fit(lx, ly)
    np.savez(OUT / 'calibrator.npz', rotate_soon=rotate.weights, balloon_soon=balloon.weights)
    vx, vy, vh = pack('takeoff', TEST_SEEDS, label_rotate)
    bx, by, bh = pack('landing', TEST_SEEDS, label_balloon)
    report = dict(
        train=dict(takeoff_n=int(len(ty)), landing_n=int(len(ly))),
        holdout=dict(
            rotate_soon=dict(heuristic_brier=brier(vh, vy), learned_brier=brier(rotate.predict(vx), vy)),
            balloon_soon=dict(heuristic_brier=brier(bh, by), learned_brier=brier(balloon.predict(bx), by))),
        source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in sorted(ROOT.glob('*.py'))})
    (OUT / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report['holdout'], indent=2), flush=True)


if __name__ == '__main__':
    main()
