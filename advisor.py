"""Phi advisor (LLM) + short-horizon simulation loop for
curriculum generation.

The advisor takes the current state and proposes (throttle, pitch)
commands.  A short-horizon rollout evaluates the proposal.  Phi sees
the result and refines its proposal.  After N refinement rounds, the
final (state, action) pair is added to the curriculum dataset.

By default the advisor uses Ollama's `phi3.5` model via the local HTTP
API on port 11434.  If Ollama is unavailable, a deterministic stub
advisor (rule-based) is used as a fallback so the rest of the
pipeline keeps working.
"""
from __future__ import annotations

import math
import json
import time
import urllib.request
import urllib.error
import numpy as np

# ---------------------------------------------------------------------
PHASE_LABELS_TAKEOFF = ["water_taxi", "hump", "climb", "cruise"]
PHASE_LABELS_LANDING = ["approach", "flare", "touchdown"]


def phase_from_state(state: np.ndarray, scenario: str) -> str:
    """Heuristic phase label from the normalised state vector."""
    z = float(state[0]) * 10.0
    Vx = float(state[1]) * 15.0
    Vz = float(state[2]) * 15.0
    if scenario == "takeoff":
        if z < 0.6 and Vx < 5.5:
            return "hump"
        if z < 1.0:
            return "water_taxi"
        if Vz < 0.4:
            return "climb"
        return "cruise"
    # landing
    if z < 0.5:
        return "touchdown"
    if z < 1.5:
        return "flare"
    return "approach"


# ---------------------------------------------------------------------
class OllamaPhi:
    """Thin client for Ollama's /api/generate endpoint."""

    def __init__(self, model: str = "phi3.5:latest",
                 host: str = "http://localhost:11434",
                 timeout: float = 60.0):
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout
        self.last_response: str = ""

    def ask(self, prompt: str) -> str:
        body = json.dumps({
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.0, "num_predict": 200},
        }).encode("utf-8")
        req = urllib.request.Request(
            f"{self.host}/api/generate",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                self.last_response = data.get("response", "")
                return self.last_response
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            return f"[ollama error: {e}]"

    def ping(self) -> bool:
        try:
            with urllib.request.urlopen(f"{self.host}/api/tags",
                                        timeout=3) as r:
                return r.status == 200
        except Exception:
            return False


# ---------------------------------------------------------------------
#  Action parsing -- Phi is asked to output JSON {"throttle":..,"pitch_deg":..}
# ---------------------------------------------------------------------
def parse_action(text: str, pitch_lo_deg: float = -3.0,
                 pitch_hi_deg: float = 12.0) -> tuple[float, float]:
    """Extract (throttle, pitch_rad) from Phi free-form text."""
    import re
    throttle = None
    pitch_deg = None
    # 1) Try the whole text as JSON
    for candidate in (text,
                      # strip markdown fences
                      re.sub(r"^```(?:json)?\s*|\s*```$", "",
                             text.strip(), flags=re.MULTILINE),
                      # extract the first {...} block
                      None):
        if candidate is None:
            m = re.search(r"\{[^{}]*\}", text, flags=re.DOTALL)
            if not m:
                break
            candidate = m.group(0)
        try:
            j = json.loads(candidate)
            throttle = float(j.get("throttle", j.get("thr", None)))
            pitch_deg = float(j.get("pitch_deg",
                                    j.get("pitch", j.get("alpha_deg", None))))
            break
        except Exception:
            continue
    # 2) Regex fallback
    if throttle is None:
        m = re.search(r"throttle\s*[:=]\s*([-+]?\d+\.?\d*)", text,
                      flags=re.IGNORECASE)
        if m:
            throttle = float(m.group(1))
    if pitch_deg is None:
        m = re.search(r"pitch\s*[:=]\s*([-+]?\d+\.?\d*)", text,
                      flags=re.IGNORECASE)
        if m:
            pitch_deg = float(m.group(1))
    if throttle is None:
        throttle = 0.5
    if pitch_deg is None:
        pitch_deg = 0.0
    throttle = float(np.clip(throttle, 0.0, 1.0))
    pitch_deg = float(np.clip(pitch_deg, pitch_lo_deg, pitch_hi_deg))
    return throttle, math.radians(pitch_deg)


# ---------------------------------------------------------------------
#  Stub advisor (used when Ollama is offline) -- rule-based but
#  mirrors the recommendations described in the user's session log.
# ---------------------------------------------------------------------
class StubAdvisor:
    """Rule-based fallback that mimics Phi's per-phase recommendations."""
    def __init__(self, scenario: str):
        self.scenario = scenario

    def recommend(self, state: np.ndarray,
                  refinement_history=None) -> tuple[float, float]:
        phase = phase_from_state(state, self.scenario)
        if self.scenario == "takeoff":
            if phase == "water_taxi":
                return 1.00, math.radians(4.0)
            if phase == "hump":
                return 1.00, math.radians(0.0)
            if phase == "climb":
                return 0.85, math.radians(8.0)
            return 0.55, math.radians(2.0)
        # landing
        if phase == "approach":
            return 0.00, math.radians(-5.0)
        if phase == "flare":
            return 0.05, math.radians(0.0)
        return 0.00, math.radians(2.0)

    def ask(self, prompt: str) -> str:
        return json.dumps(self.recommend_from_text(prompt))

    def recommend_from_text(self, prompt: str) -> dict:
        # We can't extract state from text reliably here; the caller
        # should use recommend() directly.  Return a generic action.
        return {"throttle": 0.7, "pitch_deg": 3.0}


# ---------------------------------------------------------------------
#  Phi advisor wrapper
# ---------------------------------------------------------------------
class PhiAdvisor:
    """Combines Phi (via Ollama) with a rule-based stub as fallback.

    `recommend(state)` returns a (throttle, pitch) tuple in physical units.
    Behind the scenes it formats a prompt, queries Phi, parses the
    response, and falls back to the stub if Phi fails.
    """

    PITCH_LO_DEG = -8.0
    PITCH_HI_DEG = 12.0

    def __init__(self, scenario: str, model: str = "phi3.5:latest"):
        self.scenario = scenario
        self.phi = OllamaPhi(model=model)
        self.stub = StubAdvisor(scenario)
        self.use_phi = self.phi.ping()
        self.calls = 0

    def recommend(self, state: np.ndarray,
                  refinement_history: list | None = None) -> tuple[float, float]:
        z, Vx, Vz = float(state[0]) * 10.0, float(state[1]) * 15.0, float(state[2]) * 15.0
        eta, Veta = float(state[3]) * 1.5, float(state[4]) * 1.5 / 6.0
        hull_in_water = bool(state[5] > 0.5)
        Vx_ratio = float(state[6])
        phase = phase_from_state(state, self.scenario)

        if not self.use_phi:
            return self.stub.recommend(state)

        prompt = (
            "You are advising a fixed-wing flying-boat drone autopilot. "
            f"Scenario = {self.scenario}. "
            f"Current phase (heuristic) = {phase}. "
            f"State: altitude z = {z:+.2f} m, forward speed Vx = {Vx:+.2f} m/s, "
            f"vertical speed Vz = {Vz:+.2f} m/s, wave elevation eta = {eta:+.2f} m, "
            f"wave rate Veta = {Veta:+.2f} m/s, hull_in_water = {hull_in_water}, "
            f"Vx/Vstall = {Vx_ratio:+.2f}. "
            "Recommend the next (throttle, pitch_deg) command as JSON "
            "with keys throttle (0..1) and pitch_deg (in degrees). "
            "Reply with only the JSON object."
        )
        if refinement_history:
            prompt += "\nPrevious attempts and short-horizon outcomes:\n"
            for i, h in enumerate(refinement_history[-3:]):
                prompt += (f"  attempt {i}: thr={h['throttle']:.2f} "
                           f"pitch={math.degrees(h['pitch']):.1f} deg -> "
                           f"{h['outcome']}\n")
            prompt += "Refine your recommendation to improve the outcome."

        text = self.phi.ask(prompt)
        self.calls += 1
        if text.startswith("[ollama error"):
            return self.stub.recommend(state)
        thr, pit = parse_action(text,
                                pitch_lo_deg=self.PITCH_LO_DEG,
                                pitch_hi_deg=self.PITCH_HI_DEG)
        return thr, pit


# ---------------------------------------------------------------------
#  Short-horizon simulation
# ---------------------------------------------------------------------
def short_simulate(env, action: tuple[float, float],
                   horizon_steps: int = 10):
    """Apply action for `horizon_steps` control steps and return the
    summary metrics used by the advisor to refine its proposal.
    """
    from env import FlyingBoatEnv
    if not isinstance(env, FlyingBoatEnv):
        raise TypeError("env must be a FlyingBoatEnv")
    saved = (env._x, env._z, env._Vx, env._Vz,
             env._t, env._steps, env._prev_throttle,
             env._done, env._state.copy(), list(env._traj))
    z0 = env._z
    Vx0 = env._Vx
    Vz0 = env._Vz
    # Env clips to its envelope; record what is actually applied so the
    # curriculum matches the simulation (e.g. -5 deg -> -3 deg floor).
    lo, hi = env.cfg.pitch_lo, env.cfg.pitch_hi
    mid, half = (hi + lo) / 2.0, (hi - lo) / 2.0
    thr_applied = float(np.clip(action[0], 0.0, 1.0))
    pit_applied = mid + float(np.clip((action[1] - mid) / half,
                                      -1.0, 1.0)) * half
    clipped = not (math.isclose(thr_applied, action[0]) and
                   math.isclose(pit_applied, action[1]))
    outcome = {"throttle": thr_applied, "pitch": pit_applied,
               "clipped": clipped}
    # Env takes normalized pitch in [-1, 1]; advisor uses physical radians.
    norm_pitch = float(np.clip((pit_applied - mid) / half, -1.0, 1.0))
    for _ in range(horizon_steps):
        a = np.array([thr_applied, norm_pitch], dtype=np.float32)
        s, r, done, info = env.step(a)
        if done:
            break
    zf = env._z
    Vxf = env._Vx
    Vzf = env._Vz
    dz = zf - z0
    dVx = Vxf - Vx0
    dVz = Vzf - Vz0
    outcome.update({
        "dz": dz, "dVx": dVx, "dVz": dVz,
        "z_f": zf, "Vx_f": Vxf, "Vz_f": Vzf,
        "done": bool(done), "info": info,
    })
    # Plain-text outcome for the prompt
    outcome["outcome"] = (
        f"dz={dz:+.2f} m, dVx={dVx:+.2f} m/s, dVz={dVz:+.2f} m/s "
        f"({'done' if done else 'still flying'})"
    )
    # Restore env state
    (env._x, env._z, env._Vx, env._Vz,
     env._t, env._steps, env._prev_throttle,
     env._done, env._state, traj) = saved
    env._traj = traj
    return outcome


# ---------------------------------------------------------------------
#  Curriculum generation loop
# ---------------------------------------------------------------------
def generate_curriculum(scenario: str, n_examples: int = 24,
                        n_refinements: int = 2,
                        horizon_steps: int = 12,
                        seeds: list[int] | None = None,
                        model: str = "phi3.5:latest",
                        use_phi: bool = True):
    """Generate (state, action) examples by iterative Phi + simulation.

    Returns a list of dicts:
        {state, action_throttle, action_pitch, phase, score, refined}
    """
    from env import FlyingBoatEnv, EnvConfig
    from aircraft import Aircraft

    advisor = PhiAdvisor(scenario, model=model) if use_phi else \
              StubAdvisor(scenario)
    if seeds is None:
        seeds = list(range(100, 100 + n_examples))

    curriculum = []
    for i, seed in enumerate(seeds[:n_examples]):
        env = FlyingBoatEnv(Aircraft(),
                            EnvConfig(scenario=scenario,
                                      max_steps=200, dt=0.05))
        s = env.reset(seed=seed)
        history = []
        # Initial proposal
        thr, pit = advisor.recommend(s, refinement_history=history)
        # Score the initial proposal (so even with no refinements we have data)
        initial_outcome = short_simulate(env, (thr, pit), horizon_steps)
        thr, pit = initial_outcome["throttle"], initial_outcome["pitch"]
        score = initial_outcome["dz"] * 10.0 + initial_outcome["dVx"] * 1.0 \
                - abs(initial_outcome["dVz"]) * 2.0
        best = (thr, pit, score)
        # Iterative refinement
        for r in range(n_refinements):
            outcome = short_simulate(env, (thr, pit), horizon_steps)
            thr, pit = outcome["throttle"], outcome["pitch"]
            history.append({"throttle": thr, "pitch": pit,
                            "outcome": outcome["outcome"]})
            score = outcome["dz"] * 10.0 + outcome["dVx"] * 1.0 \
                    - abs(outcome["dVz"]) * 2.0
            if best[2] is None or score > best[2]:
                best = (thr, pit, score)
            if r < n_refinements - 1:
                thr, pit = advisor.recommend(s, refinement_history=history)
        # Final score with the best proposal
        final_outcome = short_simulate(env, best[:2], horizon_steps)
        phase = phase_from_state(s, scenario)
        curriculum.append({
            "seed": seed,
            "state": s.tolist(),
            "phase": phase,
            "action_throttle": float(best[0]),
            "action_pitch_deg": float(math.degrees(best[1])),
            "score": float(best[2]) if best[2] is not None else 0.0,
            "outcome": final_outcome,
            "refined": bool(use_phi and n_refinements > 0),
        })
        print(f"[curriculum] {i+1:2d}/{n_examples} "
              f"seed={seed} phase={phase} "
              f"thr={best[0]:.2f} pitch={math.degrees(best[1]):+5.1f} deg "
              f"score={(best[2] if best[2] is not None else 0):+.2f}",
              flush=True)
    return curriculum


# ---------------------------------------------------------------------
if __name__ == "__main__":
    print("Phi advisor status:", end=" ")
    a = PhiAdvisor("takeoff")
    print("online" if a.use_phi else "offline (using stub)")
    for s in [0, 1]:
        cur = generate_curriculum(scenario="takeoff", n_examples=2,
                                  n_refinements=1, seeds=[200+s, 201+s])
        for ex in cur:
            print(ex["phase"], ex["action_throttle"],
                  ex["action_pitch_deg"])