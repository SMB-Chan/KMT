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
          seed_per_episode: bool = False):
    ac = Aircraft()
    env = FlyingBoatEnv(ac, env_cfg)
    state_dim = env.state_dim
    action_low  = np.array([0.0, -1.0], dtype=np.float32)
    action_high = np.array([1.0,  1.0], dtype=np.float32)
    policy = ActorCritic(state_dim, env.action_dim, action_low, action_high,
                         hidden=64, seed=seed)

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
if __name__ == "__main__":
    print("=" * 60)
    print("Training autopilot for take-off")
    print("=" * 60)
    cfg_to = EnvConfig(scenario="takeoff", max_steps=300, dt=0.05)
    policy_to, hist_to = train(cfg_to, episodes=400, max_updates=4,
                               seed=0, tag="takeoff", seed_per_episode=False)
    plot_history(hist_to, "takeoff")

    print("\n" + "=" * 60)
    print("Training autopilot for landing")
    print("=" * 60)
    cfg_ld = EnvConfig(scenario="landing", max_steps=400, dt=0.05)
    policy_ld, hist_ld = train(cfg_ld, episodes=400, max_updates=4,
                               seed=1, tag="landing", seed_per_episode=False)
    plot_history(hist_ld, "landing")