"""Train an autopilot for the flying-boat drone.

Implements an A2C-style on-policy loop with:
    * short rollouts (length = episode or fixed horizon)
    * GAE(lambda) advantage estimation
    * clipped surrogate + value MSE + entropy bonus
    * episode-level reward statistics for learning curves

Trains two policies:
    * takeoff  (start on water)
    * landing  (start in air at altitude)
"""
from __future__ import annotations

import math
import time
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from aircraft import Aircraft
from env     import FlyingBoatEnv, EnvConfig
from policy  import ActorCritic

OUT = "results"
os.makedirs(OUT, exist_ok=True)


# ---------------------------------------------------------------------
def gae(rewards, values, dones, gamma: float, lam: float):
    """Generalised Advantage Estimation.

    rewards, values, dones: 1-D arrays of length T
    """
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float32)
    gae_t = 0.0
    next_v = 0.0
    for t in reversed(range(T)):
        if dones[t]:
            next_v = 0.0
            gae_t  = 0.0
        nonterminal = 0.0 if dones[t] else 1.0
        delta = rewards[t] + gamma * next_v * nonterminal - values[t]
        gae_t = delta + gamma * lam * nonterminal * gae_t
        adv[t] = gae_t
        next_v = values[t]
    ret = adv + values
    return adv, ret


# ---------------------------------------------------------------------
def train(env_cfg: EnvConfig, episodes: int = 400, max_updates: int = 8,
          lr_body: float = 3e-4, lr_head: float = 5e-4,
          gamma: float = 0.99, lam: float = 0.95,
          seed: int = 0, log_every: int = 10, tag: str = "policy",
          seed_per_episode: bool = False, init_policy=None):
    ac = Aircraft()
    env = FlyingBoatEnv(ac, env_cfg)
    state_dim = env.state_dim
    action_low  = np.array([0.0, -1.0], dtype=np.float32)
    action_high = np.array([1.0,  1.0], dtype=np.float32)
    if init_policy is None:
        policy = ActorCritic(state_dim, env.action_dim, action_low, action_high,
                             hidden=64, seed=seed,
                             init_action=_init_bias_for(env_cfg))
    else:
        policy = init_policy

    history = {
        "episode_reward": [], "episode_len": [],
        "success": [], "pol_loss": [], "v_loss": [], "entropy": [],
        "smoothed_reward": [], "smoothed_success": [],
    }

    # Sliding-window statistics
    win = 30
    sr_buf = []
    rr_buf = []

    # EMA
    ema_r = 0.0
    ema_s = 0.0
    ema_a = 0.5

    start_time = time.time()
    for ep in range(episodes):
        ep_seed = (seed + ep) if seed_per_episode else seed
        s = env.reset(seed=ep_seed)

        states, actions, log_ps, values = [], [], [], []
        rewards, dones = [], []
        ep_reward = 0.0
        ep_len = 0

        # Collect one episode
        for t in range(env_cfg.max_steps):
            a, lp, v = policy.act(s)
            s_next, r, done, info = env.step(a)
            states.append(s); actions.append(a); log_ps.append(lp)
            values.append(v); rewards.append(r); dones.append(done)
            ep_reward += r
            ep_len += 1
            s = s_next
            if done:
                break

        success = info.get("success", False)

        # Compute GAE for the rollout
        adv, ret = gae(np.array(rewards, dtype=np.float32),
                       np.array(values, dtype=np.float32),
                       np.array(dones, dtype=np.float32),
                       gamma, lam)
        # Normalise advantages
        adv = (adv - adv.mean()) / (adv.std() + 1e-6)

        batch = dict(
            state=np.array(states, dtype=np.float32),
            action=np.array(actions, dtype=np.float32),
            log_p=np.array(log_ps, dtype=np.float32),
            value=np.array(values, dtype=np.float32),
            advantage=adv.astype(np.float32),
            target_v=ret.astype(np.float32),
        )

        # Update policy multiple times on the same rollout
        last = {"pol_loss": 0.0, "v_loss": 0.0, "entropy": 0.0}
        for _ in range(max_updates):
            last = policy.update(batch, lr_body=lr_body, lr_head=lr_head)

        history["episode_reward"].append(ep_reward)
        history["episode_len"].append(ep_len)
        history["success"].append(1 if success else 0)
        history["pol_loss"].append(last["pol_loss"])
        history["v_loss"].append(last["v_loss"])
        history["entropy"].append(last["entropy"])

        sr_buf.append(1 if success else 0)
        rr_buf.append(ep_reward)
        if len(sr_buf) > win:
            sr_buf.pop(0); rr_buf.pop(0)
        ema_r = ema_a * ema_r + (1 - ema_a) * ep_reward
        ema_s = ema_a * ema_s + (1 - ema_a) * (1 if success else 0)
        history["smoothed_reward"].append(ema_r)
        history["smoothed_success"].append(ema_s)

        if ep % log_every == 0 or ep == episodes - 1:
            elapsed = time.time() - start_time
            sm_r = np.mean(rr_buf)
            sm_s = np.mean(sr_buf)
            print(f"[{tag}] ep {ep:4d} | "
                  f"rew {ep_reward:+7.1f} | sm {sm_r:+7.1f} | "
                  f"succ {sm_s*100:5.1f}% | "
                  f"pol {last['pol_loss']:+.3f} v {last['v_loss']:.3f} "
                  f"H {last['entropy']:.3f} | {elapsed:.1f}s")
        # Best checkpoint on windowed success (training can collapse
        # late, e.g. landing went 100% -> 0% after ep ~260)
        if ep == 0:
            best_sm_s = -1.0
        if float(np.mean(sr_buf)) > best_sm_s and len(sr_buf) >= win:
            best_sm_s = float(np.mean(sr_buf))
            policy.save(f"{OUT}/{tag}_best_policy.npz")

    policy.save(f"{OUT}/{tag}_policy.npz")
    return policy, history


# ---------------------------------------------------------------------
def plot_history(history: dict, tag: str):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    eps = np.arange(len(history["episode_reward"]))
    axes[0, 0].plot(eps, history["episode_reward"], alpha=0.3, label="reward")
    axes[0, 0].plot(eps, history["smoothed_reward"], color="C0", label="EMA")
    axes[0, 0].set_xlabel("Episode"); axes[0, 0].set_ylabel("Reward")
    axes[0, 0].set_title(f"{tag} -- episode reward"); axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(eps, history["success"], alpha=0.3, label="outcome")
    axes[0, 1].plot(eps, history["smoothed_success"], color="C2", label="EMA")
    axes[0, 1].set_ylim(-0.05, 1.05)
    axes[0, 1].set_xlabel("Episode"); axes[0, 1].set_ylabel("Success rate")
    axes[0, 1].set_title(f"{tag} -- success"); axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(eps, history["pol_loss"], label="pol loss")
    axes[1, 0].plot(eps, history["v_loss"],   label="v loss")
    axes[1, 0].plot(eps, history["entropy"],  label="entropy")
    axes[1, 0].set_xlabel("Episode"); axes[1, 0].set_ylabel("Loss")
    axes[1, 0].set_title(f"{tag} -- losses"); axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(eps, history["episode_len"], color="C3")
    axes[1, 1].set_xlabel("Episode"); axes[1, 1].set_ylabel("Episode length")
    axes[1, 1].set_title(f"{tag} -- episode length")
    axes[1, 1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{OUT}/{tag}_learning.png", dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------
def _init_bias_for(env_cfg: EnvConfig):
    """Normalized-action init bias per scenario.

    The actor bias is in normalized action units while the physical
    pitch envelope lives in EnvConfig; the takeoff rotation bias (+4 deg)
    is derived from the current envelope so it survives retunes (a
    hardcoded u-bias silently meant +1.5 deg after the -8 deg widening,
    and takeoff never scored). Landing is biased to descend so the
    first episodes reach the water.
    """
    lo, hi = env_cfg.pitch_lo, env_cfg.pitch_hi
    if env_cfg.scenario == "takeoff":
        return [0.98, (math.radians(4.0) - (hi + lo) / 2.0)
                / ((hi - lo) / 2.0)]
    return [-0.9, -0.7]


def _default_tag(scenario: str, args) -> str:
    if args.tag:
        return args.tag
    tag = scenario
    if args.directional:
        tag += f"_dir{args.theta_mean_deg:g}"
    if args.seed_per_episode:
        tag += "_rand"
    return tag


def _config_for(scenario: str, args) -> EnvConfig:
    # Landing from 25 m needs ~30 s even at full nose-down authority
    steps = 300 if scenario == "takeoff" else 600
    return EnvConfig(scenario=scenario, max_steps=steps, dt=0.05,
                     Hs=args.hs, Tp=args.tp,
                     directional=args.directional,
                     theta_mean_deg=args.theta_mean_deg,
                     spread_s=args.spread_s)


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(description="Train flying-boat autopilots")
    p.add_argument("--scenario", default="both",
                   choices=("takeoff", "landing", "both"))
    p.add_argument("--episodes", type=int, default=400)
    p.add_argument("--seed", type=int, default=None,
                   help="policy/wave seed (default: 0 takeoff, 1 landing)")
    p.add_argument("--seed-per-episode", action="store_true",
                   help="vary wave seed each episode (seed+ep); "
                        "recommended for robust claims")
    p.add_argument("--hs", type=float, default=1.5)
    p.add_argument("--tp", type=float, default=6.0)
    p.add_argument("--directional", action="store_true")
    p.add_argument("--theta-mean-deg", type=float, default=0.0)
    p.add_argument("--spread-s", type=int, default=10)
    p.add_argument("--tag", default=None,
                   help="output tag (default: scenario [+ _dir<deg>] [+ _rand])")
    args = p.parse_args(argv)
    scenarios = ("takeoff", "landing") if args.scenario == "both" \
        else (args.scenario,)
    for sc in scenarios:
        seed = args.seed if args.seed is not None else (0 if sc == "takeoff" else 1)
        tag = _default_tag(sc, args)
        print("=" * 60)
        print(f"Training autopilot for {sc} (tag={tag})")
        print("=" * 60)
        _, hist = train(_config_for(sc, args), episodes=args.episodes,
                        max_updates=4, seed=seed, tag=tag,
                        seed_per_episode=args.seed_per_episode)
        plot_history(hist, tag)


if __name__ == "__main__":
    main()