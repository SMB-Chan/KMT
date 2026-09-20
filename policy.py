"""Tiny numpy-based Actor-Critic.

Architecture:
    Input(state_dim)  ->  Linear(64) -> ReLU -> Linear(64) -> ReLU
                                                  |
                                                  +-- Linear(action_dim)  (mean)
                                                  +-- Linear(1)          (value)
    learnable per-action log-std parameter (state-independent).

Action sampling:
    u ~ N(mean, exp(log_std)^2)
    a = tanh(u) * scale + offset  (then clipped to action bounds by env)
"""
from __future__ import annotations

import math
import numpy as np


# ---------------------------------------------------------------------
def kaiming_init(fan_in: int, shape, rng=None):
    """Kaiming-He initialisation for a weight matrix."""
    bound = math.sqrt(2.0 / fan_in)
    rng = rng if rng is not None else np.random.default_rng()
    return rng.uniform(-bound, bound, size=shape).astype(np.float32)


def zeros_init(shape):
    return np.zeros(shape, dtype=np.float32)


# ---------------------------------------------------------------------
class MLP:
    """A tiny multi-layer ReLU MLP with manual back-prop."""

    def __init__(self, layer_sizes, rng=None):
        rng = rng or np.random.default_rng(0)
        self.sizes = list(layer_sizes)
        self.params = []        # list of (W, b, name)
        for i, (din, dout) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            W = kaiming_init(din, (din, dout), rng)
            b = zeros_init((dout,))
            self.params.append({"W": W, "b": b, "name": f"layer{i}"})

    def forward(self, x):
        """Return list of pre-activations and activations."""
        h = [x]
        z = []
        for layer in self.params:
            x = x @ layer["W"] + layer["b"]
            z.append(x)
            x = np.maximum(x, 0.0)        # ReLU
            h.append(x)
        return h, z

    def backward(self, h, z, grad_out):
        """Backprop through ReLU MLP given gradient of loss wrt final h."""
        grads = []
        g = grad_out
        for layer, z_l, h_l in zip(reversed(self.params),
                                   reversed(z),
                                   reversed(h[:-1])):
            d_relu = (z_l > 0).astype(np.float32)
            g = g * d_relu
            dW = h_l.T @ g
            db = g.sum(axis=0)
            grads.insert(0, {"W": dW, "b": db})
            g = g @ layer["W"].T
        return grads

    def get_flat(self):
        out = []
        for layer in self.params:
            out.append(layer["W"].ravel())
            out.append(layer["b"].ravel())
        return np.concatenate(out)

    def set_flat(self, flat):
        idx = 0
        for layer in self.params:
            nW = layer["W"].size
            nb = layer["b"].size
            layer["W"] = flat[idx:idx + nW].reshape(layer["W"].shape).astype(np.float32)
            idx += nW
            layer["b"] = flat[idx:idx + nb].astype(np.float32)
            idx += nb

    def flat_dim(self):
        return sum(p["W"].size + p["b"].size for p in self.params)


# ---------------------------------------------------------------------
class ActorCritic:
    """Actor-Critic policy with shared body.

    state_dim, action_dim, action_low/high, hidden=64.
    """

    def __init__(self, state_dim: int, action_dim: int,
                 action_low: np.ndarray, action_high: np.ndarray,
                 hidden: int = 64, seed: int = 0,
                 init_action: np.ndarray | None = None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_low = np.asarray(action_low, dtype=np.float32)
        self.action_high = np.asarray(action_high, dtype=np.float32)
        self.action_range = (self.action_high - self.action_low) / 2.0
        self.action_mid = (self.action_high + self.action_low) / 2.0
        rng = np.random.default_rng(seed)
        self.rng = rng
        # Shared body  -> [state_dim, hidden, hidden]
        self.body = MLP([state_dim, hidden, hidden], rng=rng)
        # Actor mean head  -> hidden -> action_dim
        self.actor_head_W = kaiming_init(hidden, (hidden, action_dim), rng)
        self.actor_head_b = zeros_init((action_dim,))
        # Critic head     -> hidden -> 1
        self.critic_head_W = kaiming_init(hidden, (hidden, 1), rng)
        self.critic_head_b = zeros_init((1,))
        # Learnable log-std (state-independent) for action distribution
        self.log_std = -1.5 * np.ones(action_dim, dtype=np.float32)
        # Bias the initial action. Default is takeoff-oriented
        # ('throttle ~ 1.0, pitch ~ +4 deg'); landing passes
        # init_action=[-0.9, -0.7] (throttle ~0.05, pitch ~-5 deg) so the
        # first episodes reach the water and produce a learning signal.
        # a[0] in [0,1] => a = 0.5*tanh(u) + 0.5 ;  u = atanh(2*a - 1)
        # a[1] in [-1,1] => a = tanh(u)
        if init_action is None:
            bias_u = np.array([math.atanh(0.98), math.atanh(-0.05)],
                              dtype=np.float32)   # throttle ~ 0.99, pitch ~ 4.1 deg
        else:
            a_n = np.clip(np.asarray(init_action, dtype=np.float32),
                          -0.98, 0.98)
            bias_u = np.arctanh(a_n).astype(np.float32)
        self.actor_head_b[:min(2, action_dim)] = bias_u[:min(2, action_dim)]

    # -------- helpers --------
    def _body_forward(self, x):
        """Return (h_last_hidden, value_mean)."""
        h, _ = self.body.forward(x)
        h_last = h[-1]                              # (batch, hidden)
        value = h_last @ self.critic_head_W + self.critic_head_b   # (batch, 1)
        mean = h_last @ self.actor_head_W + self.actor_head_b     # (batch, action_dim)
        return h, h_last, mean, value

    # -------- sampling --------
    def act(self, state: np.ndarray, deterministic: bool = False):
        """Sample action. Returns (action, log_prob, value)."""
        a, lp, v = self.act_batch(np.asarray(state).reshape(1, -1),
                                  deterministic=deterministic)
        return a[0], float(lp[0]), float(v[0])

    def act_batch(self, states: np.ndarray, deterministic: bool = False):
        """Vectorised batch sampling.  states : (B, state_dim)."""
        s = states.astype(np.float32)
        h, h_last, mean, value = self._body_forward(s)
        std = np.exp(self.log_std)
        if deterministic:
            u = mean
        else:
            u = mean + std * self.rng.standard_normal(mean.shape).astype(np.float32)
        a_raw = np.tanh(u)
        a = a_raw * self.action_range + self.action_mid
        a = np.clip(a, self.action_low, self.action_high)
        log_p, _ = self._action_log_prob(a, mean)
        return (a.astype(np.float32),
                log_p.astype(np.float32),
                value.squeeze(-1).astype(np.float32))

    def _action_log_prob(self, actions, mean):
        # Reconstruct the representable action consistently at sampling/update.
        raw = np.clip((actions - self.action_mid) / self.action_range,
                      -1.0 + 1e-6, 1.0 - 1e-6)
        u = np.arctanh(raw)
        log_gauss = -0.5 * ((u - mean) ** 2 / np.exp(2 * self.log_std)
                            + 2 * self.log_std + math.log(2 * math.pi))
        log_det = np.log1p(-raw ** 2) + np.log(self.action_range)
        return (log_gauss - log_det).sum(axis=-1), u

    def value_batch(self, states: np.ndarray) -> np.ndarray:
        """Compute value for a batch of states."""
        s = states.astype(np.float32)
        h, _ = self.body.forward(s)
        h_last = h[-1]
        value = h_last @ self.critic_head_W + self.critic_head_b
        return value.squeeze(-1).astype(np.float32)

    # -------- losses --------
    def policy_loss(self, advantages, old_log_p, log_p):
        # PPO-style clipped surrogate for stability
        ratio = np.exp(log_p - old_log_p)
        eps = 0.2
        unclipped = ratio * advantages
        clipped = np.clip(ratio, 1 - eps, 1 + eps) * advantages
        return -np.minimum(unclipped, clipped).mean()

    # -------- update --------
    def update(self, batch, lr_body=3e-4, lr_head=3e-4, lr_std=3e-4,
               clip_grad: float = 5.0):
        """Single-batch update from a collected rollout.

        batch: dict with arrays
            state  (T, state_dim) float32
            action (T, action_dim) float32   -- already scaled to action range
            log_p  (T,)
            value  (T,)
            advantage (T,)
            target_v  (T,)
        """
        states = batch["state"]
        advantages = batch["advantage"]
        target_v = batch["target_v"]
        old_log_p = batch["log_p"]
        actions = batch["action"]

        h, h_last, mean, value = self._body_forward(states)
        log_p, u = self._action_log_prob(actions, mean)
        ratio = np.exp(np.clip(log_p - old_log_p, -60.0, 60.0))
        eps = 0.2
        surr1 = ratio * advantages
        surr2 = np.clip(ratio, 1 - eps, 1 + eps) * advantages
        pol_loss = -np.minimum(surr1, surr2).mean()

        # ---- Value loss ----
        v_loss = ((value.squeeze(-1) - target_v) ** 2).mean()

        # ---- Entropy bonus (encourage exploration) ----
        std = np.exp(self.log_std)
        ent = (0.5 * (np.log(2 * math.pi * std ** 2) + 1.0)).sum()

        loss = pol_loss + 0.5 * v_loss - 0.01 * ent

        # ---- backward ----
        # d(loss)/d(value)
        d_v = (value.squeeze(-1) - target_v) / value.shape[0]
        d_v = d_v.reshape(-1, 1)
        # Critic head
        d_critic_W = h_last.T @ d_v
        d_critic_b = d_v.sum(axis=0)
        # Actor head
        active = ((advantages >= 0) & (ratio <= 1 + eps)) | \
                 ((advantages < 0) & (ratio >= 1 - eps))
        d_log_p = -advantages * ratio * active / advantages.shape[0]
        std2 = std ** 2
        d_mean = d_log_p[:, None] * (u - mean) / std2               # (T, A)
        d_actor_W = h_last.T @ d_mean
        d_actor_b = d_mean.sum(axis=0)

        # gradient w.r.t. log_std
        # d log_p / d log_std = -1 + (u - mean)^2 / std^2
        d_log_std = (d_log_p[:, None]
                     * (-1.0 + ((u - mean) ** 2) / std2)).sum(axis=0)
        d_log_std = d_log_std - 0.01  # derivative of Gaussian entropy

        # Body gradient:  (d_critic_W from critic head + d_actor_W from actor head)
        d_h = (d_v @ self.critic_head_W.T
               + d_mean @ self.actor_head_W.T)
        # Now back-prop through MLP body
        # We need body grads w.r.t. h_last = z2 (the last hidden pre-activation).
        # Our MLP.backward expects grad_out wrt the final activation.
        body_grads = self.body.backward(h, [z for z in h[1:]], d_h)

        # ---- gradient clip & update ----
        gradients = [g[k] for g in body_grads for k in ("W", "b")]
        gradients += [d_critic_W, d_critic_b, d_actor_W, d_actor_b, d_log_std]
        norm = math.sqrt(sum(float(np.sum(g.astype(np.float64) ** 2)) for g in gradients))
        if not math.isfinite(norm):
            raise FloatingPointError("nonfinite policy gradient")
        if norm > clip_grad:
            for g in gradients:
                g *= clip_grad / (norm + 1e-12)
        for layer, g in zip(self.body.params, body_grads):
            layer["W"] -= lr_body * g["W"]
            layer["b"] -= lr_body * g["b"]
        self.critic_head_W -= lr_head * d_critic_W
        self.critic_head_b -= lr_head * d_critic_b
        self.actor_head_W  -= lr_head * d_actor_W
        self.actor_head_b  -= lr_head * d_actor_b
        self.log_std       -= lr_std  * d_log_std
        np.clip(self.log_std, -5.0, 2.0, out=self.log_std)

        return {"pol_loss": float(pol_loss),
                "v_loss":   float(v_loss),
                "entropy":  float(ent),
                "loss":     float(loss)}

    # -------- state-dict --------
    def save(self, path: str):
        np.savez(path,
                 body_W0=self.body.params[0]["W"],
                 body_W1=self.body.params[1]["W"],
                 body_b0=self.body.params[0]["b"],
                 body_b1=self.body.params[1]["b"],
                 actor_head_W=self.actor_head_W,
                 actor_head_b=self.actor_head_b,
                 critic_head_W=self.critic_head_W,
                 critic_head_b=self.critic_head_b,
                 log_std=self.log_std)

    def load(self, path: str):
        z = np.load(path)
        self.body.params[0]["W"] = z["body_W0"]
        self.body.params[0]["b"] = z["body_b0"]
        self.body.params[1]["W"] = z["body_W1"]
        self.body.params[1]["b"] = z["body_b1"]
        self.actor_head_W  = z["actor_head_W"]
        self.actor_head_b  = z["actor_head_b"]
        self.critic_head_W = z["critic_head_W"]
        self.critic_head_b = z["critic_head_b"]
        self.log_std       = z["log_std"]
