"""Train spatial takeoff and landing policies on the 84 kg float airframe."""
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from aircraft import Aircraft
from atmosphere import AtmosphereConfig
from dynamics import hull_force
from env import EnvConfig, FlyingBoatEnv
from mavlink_if import FlyingBoatVehicle, MAV_CMD_NAV_LAND, MAV_CMD_NAV_TAKEOFF
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


def config(scenario):
    takeoff = scenario == 'takeoff'
    return EnvConfig(
        spatial=True, scenario=scenario, directional=True, dt=0.05,
        max_steps=300 if takeoff else 600,
        Hs=0.3 if takeoff else 1.5, Tp=6.0,
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
    return value


def env_rollout(path, scenario, seed):
    cfg = config(scenario)
    env = FlyingBoatEnv(Aircraft(), cfg)
    policy = ActorCritic(env.state_dim, env.action_dim,
                         np.array([0., -1., -1., -1.]), np.ones(4), hidden=64, seed=0)
    policy.load(str(path))
    state = env.reset(seed)
    reward, info = 0.0, {}
    for _ in range(cfg.max_steps):
        state, step_reward, done, info = env.step(policy.act(state, deterministic=True)[0])
        reward += step_reward
        if done:
            break
    return dict(seed=seed, success=bool(info.get('success')), reward=float(reward),
                t=float(info.get('t', env._t)), z=float(env._z), Vx=float(env._Vx), y=float(env._y))


def vehicle_case(path, scenario, seed, wind, assist):
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
    pilot = PolicyPilot(path, spatial=True)
    status = 'time_limit'
    steps = 300 if scenario == 'takeoff' else 600
    for _ in range(steps):
        command, _ = pilot.decide(observe(v), dict(scenario=scenario))
        if assist:
            bank, rudder = v.lateral_setpoint()
            command = SpatialControl(command.throttle, command.pitch_deg,
                                     math.degrees(bank), rudder)
        command.apply(v)
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
    return dict(scenario=scenario, seed=seed, wind_y=wind, assist=assist, status=status,
                t=v.t, z=v.z, Vx=v.Vx, Vz=v.Vz, y=v.y)


def score(rows):
    return f"{sum(r.get('success') or r.get('status')=='success' for r in rows)}/{len(rows)}"


def select_and_eval(scenario, train_seed, val_seeds):
    cfg = config(scenario)
    _, history = train.train(
        cfg, episodes=400, max_updates=4, seed=train_seed, tag=scenario,
        seed_per_episode=True, log_every=20)
    train.plot_history(history, scenario)
    candidates = {'final': OUT / f'{scenario}_policy.npz'}
    best = OUT / f'{scenario}_best_policy.npz'
    if best.exists():
        candidates['best'] = best
    validation = {}
    for name, path in candidates.items():
        rows = [env_rollout(path, scenario, seed) for seed in val_seeds]
        validation[name] = dict(
            success=score(rows),
            mean_reward=float(np.mean([row['reward'] for row in rows])),
            n_success=sum(row['success'] for row in rows))
        print('val', scenario, name, validation[name]['success'],
              round(validation[name]['mean_reward'], 1), flush=True)
    selected = max(validation, key=lambda name: (
        validation[name]['n_success'], validation[name]['mean_reward']))
    path = candidates[selected]
    vehicle_rows = []
    for assist in (False, True):
        for seed in SEEDS:
            for wind in WINDS:
                row = vehicle_case(path, scenario, seed, wind, assist)
                row['candidate'] = selected
                vehicle_rows.append(row)
                print('veh', scenario, 'assist' if assist else 'raw', seed, wind,
                      row['status'], round(row['t'], 2), flush=True)
    raw = [row for row in vehicle_rows if not row['assist']]
    assisted = [row for row in vehicle_rows if row['assist']]
    return dict(
        selected=selected,
        train_success=f"{sum(history['success'])}/400",
        validation={name: {k: v for k, v in item.items() if k != 'n_success'}
                    for name, item in validation.items()},
        vehicle=dict(raw=score(raw), assisted=score(assisted), rows=vehicle_rows),
        smoothed_success_end=float(history['smoothed_success'][-1]),
        model_sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def main():
    report = dict(
        note='spatial RL on 84 kg float airframe; vehicle 10-cond held out from train/val',
        mass_kg=Aircraft().mass.total,
        takeoff=select_and_eval('takeoff', 8000, list(range(18000, 18020))),
        landing=select_and_eval('landing', 8100, list(range(18100, 18120))),
        source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in sorted(ROOT.glob('*.py'))})
    (OUT / 'report.json').write_text(json.dumps(jsonable(report), indent=2) + '\n')
    print('takeoff', report['takeoff']['vehicle'], 'landing', report['landing']['vehicle'], flush=True)


if __name__ == '__main__':
    main()
