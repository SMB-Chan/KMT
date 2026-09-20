"""Vectorized rollout collection for the FlyingBoatEnv.

Instead of stepping one environment at a time, this module runs N
parallel environments with different wave seeds.  Rollouts are returned
as a single flat batch which is suitable for A2C / PPO updates.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
import numpy as np

from aircraft import Aircraft
from env     import FlyingBoatEnv, EnvConfig


class VectorizedEnv:
    """N parallel FlyingBoatEnv instances, optionally with different seeds."""

    def __init__(self, n: int, cfg: EnvConfig, base_seed: int = 0):
        if n <= 0:
            raise ValueError("n must be positive")
        self.n = n
        self.cfg = cfg
        self.base_seed = base_seed
        self.envs = [FlyingBoatEnv(Aircraft(), cfg) for _ in range(n)]

    def reset(self):
        seeds = [self.base_seed + i for i in range(self.n)]
        states = np.stack([env.reset(seed=s) for env, s in zip(self.envs, seeds)])
        self._states = states.copy()
        self._done = np.zeros(self.n, dtype=bool)
        self._infos = [{} for _ in range(self.n)]
        return states

    @property
    def action_dim(self) -> int:
        return self.envs[0].action_dim

    def step(self, actions):
        """actions : (n, action_dim)  returns state, reward, done, infos."""
        if np.shape(actions) != (self.n, self.action_dim):
            raise ValueError(
                f"actions must have shape ({self.n}, {self.action_dim})")
        states, rewards, dones = [], [], []
        infos = []
        for i, (env, a) in enumerate(zip(self.envs, actions)):
            if self._done[i]:
                s, r, d, info = self._states[i], 0.0, True, self._infos[i]
            else:
                s, r, d, info = env.step(a)
                self._states[i], self._done[i], self._infos[i] = s, d, info
            states.append(s); rewards.append(r); dones.append(d)
            infos.append(info)
        return (np.stack(states), np.array(rewards, dtype=np.float32),
                np.array(dones, dtype=bool), infos)

    def rollout(self, policy, horizon: int | None = None,
                deterministic: bool = False):
        """Collect one episode per env (until done or horizon)."""
        if horizon is None:
            horizon = self.cfg.max_steps
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        s = self.reset()
        trajectories = [{"states": [], "actions": [], "log_ps": [],
                         "values": [], "rewards": [], "dones": []}
                        for _ in range(self.n)]
        done_mask = np.zeros(self.n, dtype=bool)
        for t in range(horizon):
            a, lp, v = policy.act_batch(s, deterministic=deterministic)
            s_next, r, d, infos = self.step(a)
            for i in range(self.n):
                if done_mask[i]:
                    continue
                trajectories[i]["states"].append(s[i])
                trajectories[i]["actions"].append(a[i])
                trajectories[i]["log_ps"].append(lp[i])
                trajectories[i]["values"].append(v[i])
                trajectories[i]["rewards"].append(float(r[i]))
                trajectories[i]["dones"].append(bool(d[i]))
            s = s_next
            done_mask |= d
            if done_mask.all():
                break
        return trajectories, infos

    def rollout_total_reward(self, policy, horizon: int | None = None) -> np.ndarray:
        """Per-env total reward for the next rollout."""
        traj, _ = self.rollout(policy, horizon)
        return np.array([sum(t["rewards"]) for t in traj], dtype=np.float32)
