"""Train a spatial takeoff policy and evaluate it off the training seeds."""
import hashlib
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from aircraft import Aircraft
from atmosphere import AtmosphereConfig
from env import EnvConfig, FlyingBoatEnv
from flight_diagnostics import takeoff_progress
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
VAL_SEEDS = list(range(18000, 18020))


def config():
    return EnvConfig(
        spatial=True, scenario='takeoff', directional=True, max_steps=300, dt=0.05,
        Hs=0.3, Tp=6.0,
        atmosphere=AtmosphereConfig(wind=(0.0, 3.0, 0.0), gust_rms=0.5))


def jsonable(value):
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def env_rollout(path, seed):
    env = FlyingBoatEnv(Aircraft(), config())
    policy = ActorCritic(env.state_dim, env.action_dim,
                         np.array([0., -1., -1., -1.]), np.ones(4), hidden=64, seed=0)
    policy.load(str(path))
    state = env.reset(seed)
    reward = 0.0
    info = {}
    for _ in range(env.cfg.max_steps):
        state, step_reward, done, info = env.step(policy.act(state, deterministic=True)[0])
        reward += step_reward
        if done:
            break
    return dict(seed=seed, success=bool(info.get('success')), reward=float(reward),
                t=float(info.get('t', env._t)), z=float(env._z), Vx=float(env._Vx), y=float(env._y))


def vehicle_case(path, seed, wind, assist):
    vehicle = FlyingBoatVehicle(
        Aircraft(), DirectionalOcean(Hs=0.3, Tp=6, seed=seed), spatial=True,
        atmosphere=AtmosphereConfig(wind=(0, wind, 0), gust_rms=0.5), seed=seed)
    vehicle.arm()
    vehicle.send_command(MAV_CMD_NAV_TAKEOFF, dict(alt=8, speed=8.5))
    pilot = PolicyPilot(path, spatial=True)
    trajectory, status = [], 'time_limit'
    for _ in range(300):
        command, _ = pilot.decide(observe(vehicle), dict(scenario='takeoff'))
        if assist:
            bank, rudder = vehicle.lateral_setpoint()
            command = SpatialControl(command.throttle, command.pitch_deg,
                                     math.degrees(bank), rudder)
        command.apply(vehicle)
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
    return dict(seed=seed, wind_y=wind, assist=assist, status=status,
                t=vehicle.t, z=vehicle.z, Vx=vehicle.Vx, y=vehicle.y,
                diagnostics=takeoff_progress(trajectory))


def score(rows, key='success'):
    if key == 'success':
        n = sum(row.get('success') or row.get('status') == 'success' for row in rows)
        return f"{n}/{len(rows)}"
    return f"{sum(row['status']=='success' for row in rows)}/{len(rows)}"


def main():
    cfg = config()
    model, history = train.train(
        cfg, episodes=400, max_updates=4, seed=8000, tag='takeoff',
        seed_per_episode=True, log_every=20)
    train.plot_history(history, 'takeoff')
    candidates = {'final': OUT / 'takeoff_policy.npz'}
    best_path = OUT / 'takeoff_best_policy.npz'
    if best_path.exists():
        candidates['best'] = best_path
    validation = {}
    for name, path in candidates.items():
        rows = [env_rollout(path, seed) for seed in VAL_SEEDS]
        validation[name] = dict(rows=rows, success=score(rows),
                                mean_reward=float(np.mean([row['reward'] for row in rows])))
        print('val', name, validation[name]['success'],
              round(validation[name]['mean_reward'], 1), flush=True)
    selected = max(validation, key=lambda name: (
        sum(row['success'] for row in validation[name]['rows']),
        validation[name]['mean_reward']))
    selected_path = candidates[selected]
    vehicle_rows = []
    for assist in (False, True):
        for seed in SEEDS:
            for wind in WINDS:
                row = vehicle_case(selected_path, seed, wind, assist)
                row['candidate'] = selected
                vehicle_rows.append(row)
                print('veh', selected, 'assist' if assist else 'raw', seed, wind,
                      row['status'], round(row['Vx'], 2), round(row['z'], 2), flush=True)
    raw = [row for row in vehicle_rows if not row['assist']]
    assisted = [row for row in vehicle_rows if row['assist']]
    report = dict(
        note='spatial takeoff RL; vehicle 10-cond is held out from training and model selection',
        train=dict(episodes=400, seed=8000, seed_per_episode=True, Hs=0.3, Tp=6,
                   wind=(0, 3, 0), gust_rms=0.5, directional=True, max_steps=300),
        validation_seeds=VAL_SEEDS,
        validation={name: dict(success=item['success'], mean_reward=item['mean_reward'])
                    for name, item in validation.items()},
        selected=selected,
        vehicle=dict(raw=score(raw, 'status'), assisted=score(assisted, 'status'), rows=vehicle_rows),
        history=history,
        config=asdict(cfg),
        source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in sorted(ROOT.glob('*.py'))},
        model_sha256=hashlib.sha256(selected_path.read_bytes()).hexdigest())
    (OUT / 'report.json').write_text(json.dumps(jsonable(report), indent=2) + '\n')
    print('selected', selected, 'raw', report['vehicle']['raw'],
          'assisted', report['vehicle']['assisted'], flush=True)


if __name__ == '__main__':
    main()
