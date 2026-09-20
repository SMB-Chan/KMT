"""Longitudinal RL on the 84 kg airframe; vehicle eval uses lateral assist."""
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
OLD = {
    'takeoff': ROOT / 'results/takeoff_preview_best_policy.npz',
    'landing': ROOT / 'results/landing_preview_best_policy.npz',
}


def config(scenario):
    takeoff = scenario == 'takeoff'
    return EnvConfig(
        scenario=scenario, directional=True, dt=0.05,
        max_steps=300 if takeoff else 600,
        Hs=0.3 if takeoff else 1.5, Tp=6.0)


def env_rollout(path, scenario, seed):
    cfg = config(scenario)
    env = FlyingBoatEnv(Aircraft(), cfg)
    policy = ActorCritic(11, 2, np.array([0., -1.]), np.ones(2), hidden=64, seed=0)
    policy.load(str(path))
    state = env.reset(seed)
    reward, info = 0.0, {}
    for _ in range(cfg.max_steps):
        state, step_reward, done, info = env.step(policy.act(state, deterministic=True)[0])
        reward += step_reward
        if done:
            break
    return dict(seed=seed, success=bool(info.get('success')), reward=float(reward))


def vehicle_case(path, scenario, seed, wind):
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
    pilot = PolicyPilot(path, spatial=False)
    status = 'time_limit'
    for _ in range(300 if scenario == 'takeoff' else 600):
        command, _ = pilot.decide(observe(v), dict(scenario=scenario))
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
        if scenario == 'takeoff' and v.z >= 8 and v.Vx >= 8.5 and v.lateral_success():
            status = 'success'
            break
        if scenario == 'landing' and v.z - v.hull.h_keel < v.wave_elevation():
            contact = hull_force(v.z, v.Vx, v.Vz, v.wave_elevation(), v.hull)
            status = ('success' if v.lateral_success() and abs(v.Vz) < 1.5
                      and max(contact.N, snap['N_water']) < 3 * v.ac.W else 'hard_landing')
            break
    return dict(seed=seed, wind_y=wind, status=status, t=v.t, z=v.z, Vx=v.Vx, Vz=v.Vz, y=v.y)


def score(rows):
    return f"{sum(r.get('success') or r.get('status')=='success' for r in rows)}/{len(rows)}"


def train_one(scenario, train_seed, val_seeds):
    cfg = config(scenario)
    bc = 24 if scenario == 'landing' else 0
    _, history = train.train(
        cfg, episodes=400, max_updates=4, seed=train_seed, tag=scenario,
        seed_per_episode=True, log_every=20, bc_episodes=bc)
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
        print('val', scenario, name, validation[name]['success'], flush=True)
    selected = max(validation, key=lambda n: (
        validation[n]['n_success'], validation[n]['mean_reward']))
    return candidates[selected], selected, history, validation


def vehicle_table(path, scenario):
    rows = []
    for seed in SEEDS:
        for wind in WINDS:
            row = vehicle_case(path, scenario, seed, wind)
            rows.append(row)
            print('veh', scenario, path.name, seed, wind, row['status'],
                  round(row['t'], 2), flush=True)
    return rows


def main():
    report = dict(note='longitudinal RL on 84 kg airframe; vehicle uses lateral assist',
                  mass_kg=Aircraft().mass.total, scenarios={})
    for scenario, seed, val in (('takeoff', 8200, range(18200, 18220)),
                                ('landing', 8300, range(18300, 18320))):
        path, selected, history, validation = train_one(scenario, seed, list(val))
        new_rows = vehicle_table(path, scenario)
        old_rows = vehicle_table(OLD[scenario], scenario) if OLD[scenario].exists() else []
        report['scenarios'][scenario] = dict(
            selected=selected,
            train_success=f"{sum(history['success'])}/400",
            validation={k: dict(success=v['success'], mean_reward=v['mean_reward'])
                        for k, v in validation.items()},
            vehicle_new=score(new_rows),
            vehicle_old_policy=score(old_rows) if old_rows else 'n/a',
            new_rows=new_rows, old_rows=old_rows,
            model_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    (OUT / 'report.json').write_text(json.dumps(report, indent=2, default=float) + '\n')
    for sc, block in report['scenarios'].items():
        print(sc, 'new', block['vehicle_new'], 'old', block['vehicle_old_policy'], flush=True)


if __name__ == '__main__':
    main()
