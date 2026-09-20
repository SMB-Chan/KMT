"""Reproducible four-axis training and held-out evaluation (simulation only)."""
import argparse
import copy
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
from aircraft import Aircraft
from env import EnvConfig, FlyingBoatEnv
import train as trainer


def evaluate(policy, cfg, seeds):
    env = FlyingBoatEnv(Aircraft(), cfg)
    rows = []
    for seed in seeds:
        state = env.reset(seed=seed)
        total = 0.0
        for _ in range(cfg.max_steps):
            action = (policy.act(state, deterministic=True)[0] if policy is not None
                      else trainer.teacher_action(state, cfg, env.ac))
            state, reward, done, info = env.step(action)
            total += reward
            if done:
                break
        rows.append(dict(conditions=env.episode_conditions, reward=float(total),
                         success=bool(info.get('success', False)), steps=env._steps))
    return dict(success_rate=float(np.mean([r['success'] for r in rows])), episodes=rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--episodes', type=int, default=40)
    p.add_argument('--bc-episodes', type=int, default=8)
    p.add_argument('--bc-epochs', type=int, default=80)
    p.add_argument('--eval-seed', type=int, default=10040)
    p.add_argument('--validation-seed', type=int, default=20000)
    p.add_argument('--eval-episodes', type=int, default=10)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--scenario', choices=['takeoff', 'landing'], default='takeoff')
    p.add_argument('--output', type=Path, default=Path('results/robust_spatial'))
    args = p.parse_args()
    if min(args.episodes, args.eval_episodes) < 1 or args.bc_episodes < 0:
        p.error('episode counts must be positive (BC may be zero)')
    training_seeds = set(range(args.seed, args.seed + max(args.episodes, args.bc_episodes)))
    eval_seeds = set(range(args.eval_seed, args.eval_seed + args.eval_episodes))
    validation_seeds = set(range(args.validation_seed, args.validation_seed + args.eval_episodes))
    if training_seeds & (eval_seeds | validation_seeds) or eval_seeds & validation_seeds:
        p.error('training, validation and evaluation seeds must not overlap')
    if args.bc_epochs < 1:
        p.error('bc-epochs must be positive')
    args.output.mkdir(parents=True, exist_ok=True)
    trainer.OUT = str(args.output)
    cfg = EnvConfig(spatial=True, directional=True, randomize_conditions=True,
                    Hs=0.8, scenario=args.scenario,
                    max_steps=300 if args.scenario == 'takeoff' else 600)
    policy, history = trainer.train(cfg, episodes=args.episodes, seed=args.seed,
                                    tag=args.scenario, bc_episodes=args.bc_episodes,
                                    bc_epochs=args.bc_epochs, max_updates=4, log_every=1)
    # Select using separate validation flights, never the comparison test seeds.
    candidates = {"rl": policy}
    if args.bc_episodes:
        bc_policy = copy.deepcopy(policy)
        bc_policy.load(str(args.output / f"{args.scenario}_bc_policy.npz"))
        candidates["bc"] = bc_policy
    validation = {name: evaluate(model, cfg, sorted(validation_seeds))
                  for name, model in candidates.items()}
    selected = max(validation, key=lambda name: (
        validation[name]['success_rate'],
        np.mean([row['reward'] for row in validation[name]['episodes']])))
    candidates[selected].save(str(args.output / f"{args.scenario}_selected_policy.npz"))
    first_eval = args.eval_seed
    report = dict(config=asdict(cfg), seed=args.seed, train_episodes=args.episodes,
                  bc_episodes=args.bc_episodes, bc_epochs=args.bc_epochs, history=history,
                  validation=validation, selected=selected,
                  evaluation=evaluate(candidates[selected], cfg, range(first_eval, first_eval + args.eval_episodes)),
                  teacher_baseline=evaluate(None, cfg, range(first_eval, first_eval + args.eval_episodes)),
                  limitations='Reduced-order simulator; randomized conditions are not calibrated weather or evidence of real-flight safety.')
    (args.output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    trainer.plot_history(history, args.scenario)
    print(json.dumps(report['evaluation'], indent=2))


if __name__ == '__main__':
    main()
