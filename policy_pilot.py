"""Deterministic saved RL policy adapter for the vehicle runner."""
from pathlib import Path
import hashlib
import math
import numpy as np
from policy import ActorCritic
from aircraft import Aircraft
from env import EnvConfig, pitch_from_normalized
from ollama_pilot import Control, SpatialControl


class PolicyPilot:
    def __init__(self, path, spatial=False):
        self.path = Path(path)
        self.spatial = spatial
        actions = 4 if spatial else 2
        self.model = ActorCritic(19 if spatial else 11, actions,
                                np.array([0.] + [-1.] * (actions - 1)),
                                np.ones(actions), hidden=64, seed=0)
        self.model.load(str(self.path))

    def check_model(self):
        return dict(name=self.path.name, source='local_rl', spatial=self.spatial,
                    sha256=hashlib.sha256(self.path.read_bytes()).hexdigest())

    def decide(self, obs, mission, previous=None):
        state = [obs['altitude_m'] / 10, obs['forward_speed_m_s'] / 15,
                 obs['vertical_speed_m_s'] / 15, obs['wave_elevation_m'] / 1.5,
                 obs['wave_rate_m_s'] * 6 / 1.5,
                 float(obs['keel_clearance_m'] < 0),
                 obs['forward_speed_m_s'] / Aircraft().V_stall
                 if mission['scenario'] == 'takeoff' else max(0., 1 - obs['altitude_m'] / 8),
                 obs['applied_throttle'],
                 *(v / 1.5 for v in obs['wave_preview_m'])]
        if self.spatial:
            heading = math.radians(obs['heading_deg'])
            state += [obs['lateral_position_m'] / 10, obs['lateral_speed_m_s'] / 15,
                      obs['bank_deg'] / 45, math.sin(heading), math.cos(heading),
                      *(v / 15 for v in obs['wind_m_s'])]
        action = self.model.act(np.asarray(state, dtype=np.float32), deterministic=True)[0]
        values = [float(action[0]),
                  math.degrees(pitch_from_normalized(action[1], EnvConfig.pitch_lo,
                                                     EnvConfig.pitch_hi))]
        control = (SpatialControl(*values, float(45 * action[2]), float(action[3]))
                   if self.spatial else Control(*values))
        return control, dict(source='local_rl', normalized_action=action.tolist())
