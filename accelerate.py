"""Accelerated training and evaluation cycle.

Runs:
    1. Vectorised A2C training on N parallel envs (different wave seeds).
    2. Periodic evaluation with checkpointing.
    3. Sea-state sweep (Hs x success rate) to map the operational envelope.
    4. Continuous MAVLink-driven flight on the real NDBC ocean.
"""
from __future__ import annotations

import math
import time
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from aircraft     import Aircraft
from ocean        import Ocean
from ocean_real   import load_buoy_default, RealOcean
from env          import FlyingBoatEnv, EnvConfig
from policy       import ActorCritic
from vectorized   import VectorizedEnv
from damage       import SprayModel, IngressModel, DamageState
from mavlink_if   import FlyingBoatVehicle, MAV_CMD_NAV_TAKEOFF, MAV_CMD_NAV_LAND

OUT = "results"
os.makedirs(OUT, exist_ok=True)


# ---------------------------------------------------------------------
def gae(rewards, values, dones, gamma=0.99, lam=0.95):
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
    return adv, adv + values


# ---------------------------------------------------------------------
def train_vectorised(scenario: str = "takeoff",
                     n_envs: int = 8,
                     episodes: int = 200,
                     max_updates: int = 4,
                     dt: float = 0.05,
                     seed: int = 0,
                     tag: str = "policy"):
    """Train using N parallel envs (different wave seeds)."""
    cfg = EnvConfig(scenario=scenario, max_steps=300, dt=dt)
    ac = Aircraft()
    venv = VectorizedEnv(n_envs, cfg, base_seed=seed)
    state_dim = venv.envs[0].state_dim
    action_dim = venv.envs[0].action_dim
    action_low  = np.array([0.0, -1.0], dtype=np.float32)
    action_high = np.array([1.0,  1.0], dtype=np.float32)
    policy = ActorCritic(state_dim, action_dim, action_low, action_high,
                         hidden=64, seed=seed)

    history = {"episode_reward": [], "success": [], "smoothed_reward": [],
               "smoothed_success": [], "wallclock": []}
    ema_r = 0.0; ema_s = 0.0; ema_a = 0.5
    start = time.time()

    for ep in range(episodes):
        s = venv.reset()
        # Collect N parallel rollouts (one episode each)
        traj = [{"states": [], "actions": [], "log_ps": [],
                 "values": [], "rewards": [], "dones": []}
                for _ in range(n_envs)]
        ep_done = [False] * n_envs
        ep_reward = np.zeros(n_envs)
        ep_success = np.zeros(n_envs, dtype=bool)
        last_info = [None] * n_envs

        for t in range(cfg.max_steps):
            a, lp, v = policy.act_batch(s)
            s_next, r, d, infos = venv.step(a)
            for i in range(n_envs):
                if ep_done[i]:
                    continue
                traj[i]["states"].append(s[i].copy())
                traj[i]["actions"].append(a[i])
                traj[i]["log_ps"].append(float(lp[i]))
                traj[i]["values"].append(float(v[i]))
                traj[i]["rewards"].append(float(r[i]))
                traj[i]["dones"].append(bool(d[i]))
                ep_reward[i] += float(r[i])
                last_info[i] = infos[i]
                if d[i]:
                    ep_done[i] = True
                    ep_success[i] = bool(infos[i].get("success", False))
            s = s_next
            if all(ep_done):
                break

        # Update policy on each completed trajectory
        n_updates = 0
        avg_loss = 0.0
        for i in range(n_envs):
            if len(traj[i]["states"]) == 0:
                continue
            adv, ret = gae(np.array(traj[i]["rewards"], dtype=np.float32),
                           np.array(traj[i]["values"], dtype=np.float32),
                           np.array(traj[i]["dones"], dtype=np.float32))
            adv = (adv - adv.mean()) / (adv.std() + 1e-6)
            batch = dict(
                state=np.array(traj[i]["states"], dtype=np.float32),
                action=np.array(traj[i]["actions"], dtype=np.float32),
                log_p=np.array(traj[i]["log_ps"], dtype=np.float32),
                value=np.array(traj[i]["values"], dtype=np.float32),
                advantage=adv.astype(np.float32),
                target_v=ret.astype(np.float32),
            )
            for _ in range(max_updates):
                losses = policy.update(batch)
                avg_loss += losses["loss"]
                n_updates += 1

        mean_rew = ep_reward.mean()
        succ_rate = ep_success.mean()
        ema_r = ema_a * ema_r + (1 - ema_a) * mean_rew
        ema_s = ema_a * ema_s + (1 - ema_a) * succ_rate
        history["episode_reward"].append(mean_rew)
        history["success"].append(succ_rate)
        history["smoothed_reward"].append(ema_r)
        history["smoothed_success"].append(ema_s)
        history["wallclock"].append(time.time() - start)

        if ep % 5 == 0 or ep == episodes - 1:
            elapsed = time.time() - start
            print(f"[{tag}] ep {ep:4d} | "
                  f"rew {mean_rew:+7.1f} | sm {ema_r:+7.1f} | "
                  f"succ {ema_s*100:5.1f}% | "
                  f"{elapsed:5.1f}s", flush=True)

    policy.save(f"{OUT}/{tag}_policy.npz")
    return policy, history


# ---------------------------------------------------------------------
def sea_state_sweep(policy, scenario: str = "takeoff",
                    Hs_list=None, Tp_list=None, n_seeds: int = 3,
                    n_envs: int = 16):
    """Map takeoff / landing success rate across sea states."""
    if Hs_list is None:
        Hs_list = [0.3, 0.5, 0.8, 1.0, 1.2, 1.5, 1.8, 2.2]
    if Tp_list is None:
        Tp_list = [4.0, 5.0, 6.0, 8.0]
    if n_seeds <= 0 or n_envs <= 0:
        raise ValueError("n_seeds and n_envs must be positive")
    results = {}
    print("Sea-state sweep:")
    for Hs in Hs_list:
        for Tp in Tp_list:
            cfg = EnvConfig(scenario=scenario, max_steps=300, dt=0.05, Hs=Hs, Tp=Tp)
            rewards, successes = [], []
            for start in range(0, n_seeds, n_envs):
                venv = VectorizedEnv(min(n_envs, n_seeds - start), cfg,
                                     base_seed=start)
                traj, infos = venv.rollout(policy, deterministic=True)
                rewards.extend(sum(t["rewards"]) for t in traj)
                successes.extend(bool(info.get("success", False)) for info in infos)
            rew = np.asarray(rewards)
            succ = np.asarray(successes)
            results[(Hs, Tp)] = (float(rew.mean()), float(succ.mean()))
            print(f"  Hs={Hs:.1f} m  Tp={Tp:.1f} s  "
                  f"rew={rew.mean():+7.1f}  succ={succ.mean()*100:5.1f}%",
                  flush=True)
    return results


# ---------------------------------------------------------------------
def plot_sea_envelope(results):
    """Heat-map of success rate over (Hs, Tp)."""
    Hs_list = sorted({k[0] for k in results})
    Tp_list = sorted({k[1] for k in results})
    grid = np.zeros((len(Hs_list), len(Tp_list)))
    for i, Hs in enumerate(Hs_list):
        for j, Tp in enumerate(Tp_list):
            _, succ = results[(Hs, Tp)]
            grid[i, j] = succ
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(grid, origin="lower", aspect="auto",
                   cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(len(Tp_list)))
    ax.set_xticklabels([f"{Tp:.0f}" for Tp in Tp_list])
    ax.set_yticks(range(len(Hs_list)))
    ax.set_yticklabels([f"{Hs:.1f}" for Hs in Hs_list])
    ax.set_xlabel("Peak period Tp, s")
    ax.set_ylabel("Significant wave height Hs, m")
    ax.set_title("Takeoff success rate -- operational envelope")
    for i in range(len(Hs_list)):
        for j in range(len(Tp_list)):
            ax.text(j, i, f"{grid[i, j]*100:.0f}%",
                    ha="center", va="center",
                    color="black", fontsize=9)
    cb = fig.colorbar(im, ax=ax); cb.set_label("success rate")
    fig.tight_layout()
    fig.savefig(f"{OUT}/sea_envelope.png", dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------
def plot_history(history: dict, tag: str):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    eps = np.arange(len(history["episode_reward"]))
    axes[0].plot(eps, history["episode_reward"], alpha=0.3, label="reward")
    axes[0].plot(eps, history["smoothed_reward"], color="C0", label="EMA")
    axes[0].set_xlabel("Episode"); axes[0].set_ylabel("Reward")
    axes[0].set_title(f"{tag} reward"); axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[1].plot(eps, history["success"], alpha=0.3, label="success")
    axes[1].plot(eps, history["smoothed_success"], color="C2", label="EMA")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].set_xlabel("Episode"); axes[1].set_ylabel("Success rate")
    axes[1].set_title(f"{tag} success"); axes[1].legend(); axes[1].grid(True, alpha=0.3)
    axes[2].plot(history["wallclock"], eps)
    axes[2].set_xlabel("Wall clock, s"); axes[2].set_ylabel("Episodes")
    axes[2].set_title(f"{tag} throughput")
    axes[2].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{OUT}/{tag}_learning.png", dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------
def mavlink_sea_sweep(policy, scenario: str = "takeoff",
                      Hs_list=None, Tp_list=None, n_seeds: int = 3,
                      duration: float = 15.0):
    """Sweep using MAVLink + damage model for realistic envelope."""
    if Hs_list is None:
        Hs_list = [0.3, 0.5, 0.8, 1.0, 1.3, 1.6, 2.0]
    if Tp_list is None:
        Tp_list = [4.0, 6.0, 8.0]
    results = {}
    print("\nMAVLink + damage-model sea-state sweep:")
    n = int(duration / 0.05)
    for Hs in Hs_list:
        for Tp in Tp_list:
            succs = []
            for sd in range(n_seeds):
                sea = Ocean(Hs=Hs, Tp=Tp, seed=sd)
                ac = Aircraft()
                veh = FlyingBoatVehicle(ac, sea)
                veh.reset()
                veh.arm()
                if scenario == "takeoff":
                    veh.send_command(MAV_CMD_NAV_TAKEOFF,
                                     {"alt": 10.0, "speed": 13.0})
                else:
                    veh.reset()
                    veh.x, veh.z = 0.0, 25.0
                    gs = math.radians(8.0)
                    veh.Vx = 13.0 * math.cos(gs)
                    veh.Vz = -13.0 * math.sin(gs)
                    veh.t = 0.0
                    veh._msg_log.clear()
                    veh.arm()
                    veh.send_command(MAV_CMD_NAV_LAND,
                                     {"alt": 0.0, "glide": 8.0})
                observer = FlyingBoatEnv(ac, EnvConfig(scenario=scenario))
                observer._sea = sea
                success = False
                for _ in range(n):
                    observer._x, observer._z = veh.x, veh.z
                    observer._Vx, observer._Vz = veh.Vx, veh.Vz
                    observer._t, observer._prev_throttle = veh.t, veh.throttle
                    action, _, _ = policy.act(observer._build_state(), deterministic=True)
                    veh.step(action=action)
                    if veh.damage.failed:
                        break
                    eta = float(sea.eta(np.array([veh.x]), veh.t)[0])
                    if scenario == "takeoff":
                        success = veh.z >= 8.0 and veh.Vx >= 8.5
                        if success:
                            break
                    elif veh.z < eta + veh.hull.h_keel:
                        snap = veh.read_telemetry()[2]
                        success = abs(veh.Vz) < 1.5 and snap["N_water"] < 3 * ac.W
                        break
                succs.append(success and not veh.damage.failed)
            results[(Hs, Tp)] = float(np.mean(succs))
            print(f"  Hs={Hs:.1f}  Tp={Tp:.1f}  succ={results[(Hs, Tp)]*100:5.1f}%",
                  flush=True)
    return results


def plot_damage_envelope(results):
    """Operational envelope with damage model."""
    Hs_list = sorted({k[0] for k in results})
    Tp_list = sorted({k[1] for k in results})
    grid = np.zeros((len(Hs_list), len(Tp_list)))
    for i, Hs in enumerate(Hs_list):
        for j, Tp in enumerate(Tp_list):
            grid[i, j] = results[(Hs, Tp)]
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(grid, origin="lower", aspect="auto",
                   cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(len(Tp_list)))
    ax.set_xticklabels([f"{Tp:.0f}" for Tp in Tp_list])
    ax.set_yticks(range(len(Hs_list)))
    ax.set_yticklabels([f"{Hs:.1f}" for Hs in Hs_list])
    ax.set_xlabel("Peak period Tp, s")
    ax.set_ylabel("Significant wave height Hs, m")
    ax.set_title("Operational envelope (MAVLink + damage model)")
    for i in range(len(Hs_list)):
        for j in range(len(Tp_list)):
            ax.text(j, i, f"{grid[i, j]*100:.0f}%",
                    ha="center", va="center",
                    color="black", fontsize=9)
    cb = fig.colorbar(im, ax=ax); cb.set_label("success rate")
    fig.tight_layout()
    fig.savefig(f"{OUT}/envelope_with_damage.png", dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------
if __name__ == "__main__":
    t0 = time.time()
    # 1. Train takeoff policy (vectorised, 8 parallel envs)
    print("=" * 60)
    print("Accelerated cycle")
    print("=" * 60)
    policy_to, hist_to = train_vectorised(
        scenario="takeoff", n_envs=8, episodes=150, max_updates=4,
        dt=0.05, seed=0, tag="takeoff_v2")
    plot_history(hist_to, "takeoff_v2")

    # 2. Sea-state sweep with the trained policy
    print("\n" + "=" * 60)
    print("Sea-state sweep (RL env, no damage model)")
    print("=" * 60)
    sweep = sea_state_sweep(policy_to, scenario="takeoff",
                            Hs_list=[0.3, 0.5, 0.8, 1.0, 1.3, 1.6, 2.0],
                            Tp_list=[4.0, 6.0, 8.0], n_envs=16)
    plot_sea_envelope(sweep)

    # 3. MAVLink + damage-model sweep
    mavlink_sweep = mavlink_sea_sweep(policy_to, scenario="takeoff",
                                      Hs_list=[0.3, 0.5, 0.8, 1.0,
                                               1.3, 1.6, 2.0],
                                      Tp_list=[4.0, 6.0, 8.0],
                                      n_seeds=2)
    plot_damage_envelope(mavlink_sweep)

    elapsed = time.time() - t0
    print(f"\nTotal wall-clock: {elapsed:.1f} s")