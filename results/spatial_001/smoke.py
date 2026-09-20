"""Short spatial training and trajectory checks, not a performance benchmark."""
import json
import sys
from dataclasses import asdict
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from atmosphere import AtmosphereConfig
from aircraft import Aircraft
from env import EnvConfig, FlyingBoatEnv
import train
OUT = Path(__file__).resolve().parent
train.OUT = str(OUT)
report = {}
for scenario in ('takeoff', 'landing'):
    cfg = EnvConfig(spatial=True, scenario=scenario, directional=True,
                    max_steps=300 if scenario=='takeoff' else 600,
                    atmosphere=AtmosphereConfig(wind=(0, 3, 0), gust_rms=0.5))
    model, history = train.train(cfg, episodes=2, max_updates=1,
                                tag=scenario, seed=8100, seed_per_episode=True)
    env = FlyingBoatEnv(Aircraft(), cfg)
    state = env.reset(9100)
    for _ in range(cfg.max_steps):
        state, reward, done, info = env.step(model.act(state, deterministic=True)[0])
        assert np.isfinite(state).all() and np.isfinite(reward)
        if done:
            break
    (OUT/f'{scenario}_trajectory.json').write_text(json.dumps(env.trajectory(), indent=2)+'\n')
    report[scenario] = dict(config=asdict(cfg), history=history,
                            state_dim=env.state_dim, action_dim=env.action_dim,
                            evaluation_seed=9100, final=info)
(OUT/'smoke.json').write_text(json.dumps(report, indent=2)+'\n')
