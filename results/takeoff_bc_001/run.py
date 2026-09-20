"""Takeoff BC from native setpoint, then longitudinal RL on the 84 kg airframe."""
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from aircraft import Aircraft
from atmosphere import AtmosphereConfig
from env import EnvConfig, FlyingBoatEnv
from mavlink_if import FlyingBoatVehicle, MAV_CMD_NAV_TAKEOFF
from ocean_directional import DirectionalOcean
from ollama_pilot import SpatialControl, observe
from policy import ActorCritic
from policy_pilot import PolicyPilot
import train

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
train.OUT = str(OUT)
SEEDS = list(range(9500, 9505))
WINDS = (-3.0, 3.0)
VAL = list(range(18400, 18420))
PREV = ROOT / 'results/new_airframe_long_001/takeoff_best_policy.npz'


def cfg():
    return EnvConfig(scenario='takeoff', directional=True, max_steps=300, dt=0.05, Hs=0.3, Tp=6.0)


def env_rollout(path, seed):
    env = FlyingBoatEnv(Aircraft(), cfg())
    policy = ActorCritic(11, 2, np.array([0., -1.]), np.ones(2), hidden=64, seed=0)
    policy.load(str(path))
    state = env.reset(seed)
    info = {}
    for _ in range(300):
        state, _, done, info = env.step(policy.act(state, deterministic=True)[0])
        if done:
            break
    return bool(info.get('success'))


def vehicle_case(path, seed, wind):
    v = FlyingBoatVehicle(
        Aircraft(), DirectionalOcean(Hs=0.3, Tp=6, seed=seed), spatial=True,
        atmosphere=AtmosphereConfig(wind=(0, wind, 0), gust_rms=0.5), seed=seed)
    v.arm()
    v.send_command(MAV_CMD_NAV_TAKEOFF, dict(alt=8, speed=8.5))
    pilot = PolicyPilot(path, spatial=False)
    status = 'time_limit'
    for _ in range(300):
        command, _ = pilot.decide(observe(v), dict(scenario='takeoff'))
        bank, rudder = v.lateral_setpoint()
        SpatialControl(command.throttle, command.pitch_deg, math.degrees(bank), rudder).apply(v)
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
        if v.z >= 8 and v.Vx >= 8.5 and v.lateral_success():
            status = 'success'
            break
    return dict(seed=seed, wind_y=wind, status=status, t=v.t, z=v.z, Vx=v.Vx, y=v.y)


def table(path):
    rows = []
    for seed in SEEDS:
        for wind in WINDS:
            row = vehicle_case(path, seed, wind)
            rows.append(row)
            print(path.name, seed, wind, row['status'], round(row['t'], 2), flush=True)
    return rows


def main():
    _, history = train.train(
        cfg(), episodes=400, max_updates=4, seed=8400, tag='takeoff',
        seed_per_episode=True, log_every=20, bc_episodes=24)
    train.plot_history(history, 'takeoff')
    candidates = {'final': OUT / 'takeoff_policy.npz'}
    best = OUT / 'takeoff_best_policy.npz'
    if best.exists():
        candidates['best'] = best
    validation = {}
    for name, path in candidates.items():
        n = sum(env_rollout(path, seed) for seed in VAL)
        validation[name] = f"{n}/20"
        print('val', name, validation[name], flush=True)
    selected = max(validation, key=lambda n: int(validation[n].split('/')[0]))
    if validation.get('best') == validation.get('final'):
        selected = 'best' if 'best' in candidates else 'final'
    path = candidates[selected]
    new_rows = table(path)
    old_rows = table(PREV) if PREV.exists() else []
    report = dict(
        note='takeoff BC from native then RL; vehicle uses lateral assist',
        selected=selected, validation=validation,
        train_success=f"{sum(history['success'])}/400",
        vehicle_bc_rl=f"{sum(r['status']=='success' for r in new_rows)}/10",
        vehicle_prev_long=f"{sum(r['status']=='success' for r in old_rows)}/10" if old_rows else 'n/a',
        native='10/10',
        new_rows=new_rows, old_rows=old_rows,
        source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in sorted(ROOT.glob('*.py'))})
    (OUT / 'report.json').write_text(json.dumps(report, indent=2, default=float) + '\n')
    print(report['vehicle_bc_rl'], 'prev', report['vehicle_prev_long'], flush=True)


if __name__ == '__main__':
    main()
