"""Flight-specific System One layer on existing observations and setpoints."""
import math
from ollama_pilot import Control, SpatialControl
from mavlink_if import LowLevelController
from system_one import Logistic, choice_from_logits, score_answer, noul_answer

TAKEOFF_PHASES = ('taxi', 'rotate', 'climb')
LANDING_PHASES = ('approach', 'flare', 'settle')


def _logit(distance, scale):
    return max(-8.0, min(8.0, distance / scale))


def features(observation):
    return (
        1.0,
        float(observation['altitude_m']) / 10.0,
        float(observation['forward_speed_m_s']) / 15.0,
        float(observation['vertical_speed_m_s']) / 5.0,
        float(observation['keel_clearance_m']) / 10.0,
        float(observation['wave_elevation_m']) / 1.5,
        float(observation.get('applied_throttle', 0.0)),
    )


def evaluate(observation, mission, calibrator=None):
    z = float(observation['altitude_m'])
    vx = float(observation['forward_speed_m_s'])
    vz = float(observation['vertical_speed_m_s'])
    eta = float(observation['wave_elevation_m'])
    clearance = float(observation['keel_clearance_m'])
    scenario = mission['scenario']
    if scenario == 'takeoff':
        phase = choice_from_logits(TAKEOFF_PHASES, (
            _logit(7.0 - vx, 1.5),
            _logit(min(vx - 7.0, 2.0 - z), 1.5),
            _logit(z - 2.0, 1.5)))
        progress = score_answer(min(z / 8.0, 1.0) * min(vx / 8.5, 1.0),
                                min(1.0, abs(z - 8.0) / 8.0 + abs(vx - 8.5) / 8.5))
    else:
        phase = choice_from_logits(LANDING_PHASES, (
            _logit(z - 5.0, 1.5),
            _logit(min(5.0 - z, z - 1.0), 1.0),
            _logit(1.0 - z, 0.6)))
        progress = score_answer(max(0.0, 1.0 - z / 25.0), min(1.0, abs(vz) / 1.5))
    ground = noul_answer(1.0 / (1.0 + math.exp((z - eta - 1.5) / 0.4)))
    balloon = noul_answer(1.0 / (1.0 + math.exp((-vz - 0.2) / 0.15)) * (1.0 if z < 2 else 0.0)
                          if scenario == 'landing' else 0.0)
    if calibrator is not None:
        x = features(observation)
        if scenario == 'landing' and 'balloon_soon' in calibrator:
            balloon = noul_answer(calibrator['balloon_soon'].predict(x))
        if scenario == 'takeoff' and 'rotate_soon' in calibrator:
            answers_rotate = noul_answer(calibrator['rotate_soon'].predict(x))
        else:
            answers_rotate = noul_answer(0.0 if scenario != 'takeoff' else
                                         1.0 / (1.0 + math.exp((7.0 - vx) / 1.0)))
    else:
        answers_rotate = noul_answer(0.0 if scenario != 'takeoff' else
                                     1.0 / (1.0 + math.exp((7.0 - vx) / 1.0)))
    return {
        'phase': phase,
        'progress': progress,
        'in_ground_effect': ground,
        'ballooning': balloon,
        'rotate_soon': answers_rotate,
        'clearance_m': clearance,
    }


def compose(observation, mission, answers, spatial=False):
    ctl = LowLevelController()
    state = dict(z=observation['altitude_m'], Vx=observation['forward_speed_m_s'],
                 Vz=observation['vertical_speed_m_s'], eta=observation['wave_elevation_m'],
                 wave_preview=observation['wave_preview_m'])
    target_alt = mission.get('target_altitude_m', 8 if mission['scenario'] == 'takeoff' else 0)
    target_speed = mission.get('target_speed_m_s', 8.5)
    if mission['scenario'] == 'takeoff':
        pitch, throttle = ctl.takeoff_setpoint(
            state, target_alt, target_speed, mission.get('v_rotate_m_s', 7.0))
    else:
        pitch, throttle = ctl.landing_setpoint(state, 0.0, math.radians(8))
        balloon = answers['ballooning']
        if balloon.noul >= 0.7 and balloon.confidence >= 0.5:
            pitch, throttle = math.radians(-4.0), 0.0
    if spatial:
        return SpatialControl(float(throttle), math.degrees(pitch), 0.0, 0.0)
    return Control(float(throttle), math.degrees(pitch))


class FlightJev:
    def __init__(self, spatial=False, calibrator=None):
        self.spatial = spatial
        self.calibrator = calibrator or {}

    @classmethod
    def load(cls, path, spatial=False):
        import numpy as np
        bundle = np.load(path, allow_pickle=False)
        calibrator = {name: Logistic(bundle[name]) for name in bundle.files}
        return cls(spatial=spatial, calibrator=calibrator)

    def check_model(self):
        source = 'calibrated_system_one' if self.calibrator else 'heuristic_system_one'
        return dict(name='flight_jev_v1', source=source, spatial=self.spatial,
                    calibrated=sorted(self.calibrator))

    def decide(self, observation, mission, previous=None):
        answers = evaluate(observation, mission, self.calibrator)
        control = compose(observation, mission, answers, self.spatial)
        payload = {key: (value.__dict__ if hasattr(value, '__dict__') else value)
                   for key, value in answers.items()}
        return control, dict(source='flight_jev', answers=payload)
