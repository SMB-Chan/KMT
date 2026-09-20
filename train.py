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
def landing_teacher_action(state, cfg: EnvConfig):
    """Normalized env action from the wave-preview landing setpoint."""
    from mavlink_if import LowLevelController
    z = float(state[0]) * 10.0
    Vx = float(state[1]) * 15.0
    Vz = float(state[2]) * 15.0
    eta = float(state[3]) * 1.5
    preview = np.asarray(state[8:8 + len(cfg.preview_dx_m)], dtype=float) * 1.5
    ctl = LowLevelController(pitch_lo=cfg.pitch_lo, pitch_hi=cfg.pitch_hi)
    pitch, thr = ctl.landing_setpoint(
        {"z": z, "Vx": Vx, "Vz": Vz, "eta": eta, "wave_preview": preview},
        0.0, math.radians(8.0))
    mid = (cfg.pitch_hi + cfg.pitch_lo) / 2.0
    half = (cfg.pitch_hi - cfg.pitch_lo) / 2.0
    return np.array([float(thr), float((pitch - mid) / half)], dtype=np.float32)


def takeoff_teacher_action(state, cfg: EnvConfig, ac=None):
    """Normalized env action from the two-stage takeoff setpoint."""
    from mavlink_if import LowLevelController, rotate_speed
    z = float(state[0]) * 10.0
    Vx = float(state[1]) * 15.0
    Vz = float(state[2]) * 15.0
    ctl = LowLevelController(pitch_lo=cfg.pitch_lo, pitch_hi=cfg.pitch_hi)
    pitch, thr = ctl.takeoff_setpoint(
        {"z": z, "Vx": Vx, "Vz": Vz}, cfg.z_goal, cfg.V_lo,
        rotate_speed(ac) if ac is not None else 7.0)
    mid = (cfg.pitch_hi + cfg.pitch_lo) / 2.0
    half = (cfg.pitch_hi - cfg.pitch_lo) / 2.0
    return np.array([float(thr), float((pitch - mid) / half)], dtype=np.float32)


def teacher_action(state, cfg: EnvConfig, ac=None):
    longitudinal = (landing_teacher_action(state, cfg) if cfg.scenario == "landing"
                    else takeoff_teacher_action(state, cfg, ac))
    if not cfg.spatial:
        return longitudinal
    lateral = state[8 + len(cfg.preview_dx_m):]
    y, vy = float(lateral[0]) * 10, float(lateral[1]) * 15
    heading = math.atan2(float(lateral[3]), float(lateral[4]))
    desired_vy = float(np.clip(-0.55 * y, -5, 5))
    bank = float(np.clip(0.22 * (desired_vy - vy), -math.pi / 6, math.pi / 6))
    rudder = float(np.clip(-1.5 * heading - 0.04 * y - 0.10 * vy, -1, 1))
    return np.concatenate((longitudinal, [bank / math.radians(45), rudder])).astype(np.float32)


def _bc_pretrain(policy, env, episodes: int, epochs: int, seed: int):
    states, actions = [], []
    for ep in range(episodes):
        s = env.reset(seed=seed + ep)
        for _ in range(env.cfg.max_steps):
            a = teacher_action(s, env.cfg, getattr(env, "ac", None))
            states.append(s)
            actions.append(a)
            s, _, done, _ = env.step(a)
            if done:
                break
    if not states:
        return
    xs = np.array(states, dtype=np.float32)
    ys = np.array(actions, dtype=np.float32)
    rng = np.random.default_rng(seed)
    losses = []
    for _ in range(epochs):
        order = rng.permutation(len(xs))
        for start in range(0, len(xs), 128):
            idx = order[start:start + 128]
            policy.imitate(xs[idx], ys[idx], lr=0.01, action_space=True)
        prediction = policy.act_batch(xs, deterministic=True)[0]
        losses.append(float(np.mean(((prediction - ys) / policy.action_range) ** 2)))
    return {"samples": len(xs), "epochs": epochs, "normalized_action_mse": losses}


def train(env_cfg: EnvConfig, episodes: int = 400, max_updates: int = 8,
          lr_body: float = 3e-4, lr_head: float = 5e-4,
          gamma: float = 0.99, lam: float = 0.95,
          seed: int = 0, log_every: int = 10, tag: str = "policy",
          seed_per_episode: bool = False, init_policy=None,
          bc_episodes: int = 0, bc_epochs: int = 8):
    ac = Aircraft()
    env = FlyingBoatEnv(ac, env_cfg)
    state_dim = env.state_dim
    action_low = np.array([0.0] + [-1.0] * (env.action_dim - 1), dtype=np.float32)
    action_high = np.ones(env.action_dim, dtype=np.float32)
    if init_policy is None:
        policy = ActorCritic(state_dim, env.action_dim, action_low, action_high,
                             hidden=64, seed=seed,
                             init_action=_init_bias_for(env_cfg))
    else:
        policy = init_policy
    if bc_episodes < 0 or bc_epochs < 0:
        raise ValueError("bc_episodes and bc_epochs must be nonnegative")
    bc_report = None
    if bc_episodes > 0:
        bc_report = _bc_pretrain(policy, env, bc_episodes, bc_epochs, seed)
        policy.save(f"{OUT}/{tag}_bc_policy.npz")

    history = {
        "bc_report": bc_report,
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
        ep_seed = (seed + ep) if (seed_per_episode or env_cfg.randomize_conditions) else seed
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
    extra = [0.0, 0.0] if env_cfg.spatial else []
    lo, hi = env_cfg.pitch_lo, env_cfg.pitch_hi
    if env_cfg.scenario == "takeoff":
        return [0.98, (math.radians(4.0) - (hi + lo) / 2.0)
                / ((hi - lo) / 2.0)] + extra
    return [-0.9, -0.7] + extra


def _default_tag(scenario: str, args) -> str:
    if args.tag:
        return args.tag
    tag = scenario
    if getattr(args, "spatial", False):
        tag += "_spatial"
    if getattr(args, "weather", None):
        tag += f"_{args.weather}"
    if args.directional:
        tag += f"_dir{args.theta_mean_deg:g}"
    if args.seed_per_episode:
        tag += "_rand"
    return tag


def _config_for(scenario: str, args) -> EnvConfig:
    # Landing from 25 m needs ~30 s even at full nose-down authority
    steps = 300 if scenario == "takeoff" else 600
    from atmosphere import AtmosphereConfig
    wind = tuple(getattr(args, "wind", (0, 0, 0)))
    gust_rms = getattr(args, "gust_rms", 0.0)
    gust_model = "sum4"
    weather_cfg = None
    preset_name = getattr(args, "weather", None)
    if preset_name:
        from weather import get_preset
        preset = get_preset(preset_name)
        weather_cfg = preset.weather
        if not any(wind) and not gust_rms:
            # Adopt the preset's recommended wind/gusts when untouched.
            wind = tuple(preset.atmosphere.wind)
            gust_rms = preset.atmosphere.gust_rms
            gust_model = preset.atmosphere.gust_model
    return EnvConfig(randomize_conditions=getattr(args, "randomize_conditions", False),
                     spatial=getattr(args, "spatial", False),
                     atmosphere=AtmosphereConfig(wind=wind,
                                                  gust_rms=gust_rms,
                                                  gust_model=gust_model),
                     weather=weather_cfg,
                     scenario=scenario, max_steps=steps, dt=0.05,
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
    p.add_argument("--randomize-conditions", action="store_true", help="sample reproducible wave and wind conditions per episode")
    p.add_argument("--hs", type=float, default=1.5)
    p.add_argument("--tp", type=float, default=6.0)
    p.add_argument("--directional", action="store_true")
    p.add_argument("--theta-mean-deg", type=float, default=0.0)
    p.add_argument("--spread-s", type=int, default=10)
    p.add_argument("--tag", default=None,
                   help="output tag (default: scenario [+ _dir<deg>] [+ _rand])")
    p.add_argument("--bc-episodes", type=int, default=None,
                   help="landing BC episodes from wave-preview flare "
                        "(default: 24 landing, 0 takeoff)")
    p.add_argument("--spatial", action="store_true", help="enable lateral motion, bank/yaw control and atmosphere")
    p.add_argument("--wind", type=float, nargs=3, default=(0., 0., 0.), metavar=("WX", "WY", "WZ"))
    p.add_argument("--gust-rms", type=float, default=0.0)
    from weather import PRESETS
    p.add_argument("--weather", choices=sorted(PRESETS), default=None,
                   help="weather preset (spatial only); brings recommended "
                        "wind/gusts unless --wind/--gust-rms are set")
    args = p.parse_args(argv)
    if not args.spatial and (any(args.wind) or args.gust_rms):
        p.error("--wind and --gust-rms require --spatial")
    if args.weather and not args.spatial:
        p.error("--weather requires --spatial")
    scenarios = ("takeoff", "landing") if args.scenario == "both" \
        else (args.scenario,)
    for sc in scenarios:
        seed = args.seed if args.seed is not None else (0 if sc == "takeoff" else 1)
        tag = _default_tag(sc, args)
        print("=" * 60)
        print(f"Training autopilot for {sc} (tag={tag})")
        print("=" * 60)
        bc = args.bc_episodes
        if bc is None:
            bc = 24 if not args.spatial else 0
        _, hist = train(_config_for(sc, args), episodes=args.episodes,
                        max_updates=4, seed=seed, tag=tag,
                        seed_per_episode=args.seed_per_episode,
                        bc_episodes=bc)
        plot_history(hist, tag)


if __name__ == "__main__":
    main()