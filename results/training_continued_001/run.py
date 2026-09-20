"""Continue preview policies, select on validation, then evaluate held-out waves."""
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path
import shutil
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from aircraft import Aircraft
from env import EnvConfig, FlyingBoatEnv
from policy import ActorCritic
import train
OUT = Path(__file__).resolve().parent
train.OUT = str(OUT)

def load(path, seed=7000):
    p = ActorCritic(11, 2, np.array([0., -1.]), np.array([1., 1.]), hidden=64, seed=seed)
    p.load(str(path))
    return p

def evaluate(path, cfg, seeds):
    p = load(path)
    env = FlyingBoatEnv(Aircraft(), cfg)
    rows = []
    for seed in seeds:
        s = env.reset(seed=seed)
        reward = 0.
        for step in range(cfg.max_steps):
            a, _, _ = p.act(s, deterministic=True)
            s, r, done, info = env.step(a)
            reward += r
            if done:
                break
        rows.append(dict(seed=seed, success=bool(info.get('success', False)), reward=float(reward), steps=step+1))
    return dict(successes=sum(r['success'] for r in rows), trials=len(rows), mean_reward=float(np.mean([r['reward'] for r in rows])), rows=rows)

def save(report):
    (OUT/'report.json').write_text(json.dumps(report, indent=2)+'\n')

report = dict(episodes_per_scenario=400, training_seeds=[7000,7399], validation_seeds=[17000,17029], test_seeds=[27000,27049], lr_body=3e-5, lr_head=5e-5, max_updates=4, damage_model=False, scenarios={})
for scenario in ('takeoff','landing'):
    cfg = EnvConfig(scenario=scenario, max_steps=300 if scenario=='takeoff' else 600)
    baseline = ROOT/'results'/f'{scenario}_preview_best_policy.npz'
    entry = dict(config=asdict(cfg), baseline=str(baseline), baseline_sha256=hashlib.sha256(baseline.read_bytes()).hexdigest())
    report['scenarios'][scenario]=entry
    save(report)
    _, history = train.train(cfg, episodes=400, max_updates=4, lr_body=3e-5, lr_head=5e-5, seed=7000, log_every=40, tag=scenario, seed_per_episode=True, init_policy=load(baseline), bc_episodes=0)
    (OUT/f'{scenario}_history.json').write_text(json.dumps(history, indent=2)+'\n')
    train.plot_history(history, scenario)
    candidates = {'baseline':baseline, 'training_best':OUT/f'{scenario}_best_policy.npz', 'final':OUT/f'{scenario}_policy.npz'}
    entry['validation']={name:evaluate(path,cfg,range(17000,17030)) for name,path in candidates.items()}
    winner=max(candidates, key=lambda name:(entry['validation'][name]['successes'],entry['validation'][name]['mean_reward']))
    entry['selected']=winner
    shutil.copyfile(candidates[winner], OUT/f'{scenario}_selected_policy.npz')
    entry['test']={'baseline':evaluate(baseline,cfg,range(27000,27050)), 'selected':evaluate(OUT/f'{scenario}_selected_policy.npz',cfg,range(27000,27050))}
    save(report)
    print(scenario, 'selected', winner, 'test', {k:v['successes'] for k,v in entry['test'].items()}, flush=True)
report['source_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob('*.py')}
report['runner_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
save(report)
