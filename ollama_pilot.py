"""Local Ollama pilot for the longitudinal simulator (no hardware transport)."""
from __future__ import annotations

from dataclasses import dataclass, asdict
import json
from http.client import HTTPException
import math
import time
from urllib.error import URLError
from urllib.parse import urlsplit
from urllib.request import Request, build_opener, ProxyHandler, HTTPRedirectHandler

from ocean_directional import wave_preview


CONTROL_SCHEMA = {
    'type': 'object',
    'properties': {
        'throttle': {'type': 'number', 'minimum': 0, 'maximum': 1},
        'pitch_deg': {'type': 'number', 'minimum': -8, 'maximum': 15},
    },
    'required': ['throttle', 'pitch_deg'],
    'additionalProperties': False,
}
SYSTEM_PROMPT = '''You pilot an 80 kg flying boat in a longitudinal simulation.
Return ONLY JSON {"throttle": number, "pitch_deg": number}.
Limits: throttle 0..1; body pitch -8..15 degrees (positive nose up).
Stall speed is 6.4 m/s.
Takeoff goal: reach 8 m altitude with forward speed above 8.5 m/s.
Landing goal: touch the water with sink rate below 1.5 m/s.
Wave height is not altitude clearance.
wave_preview_m is encounter-time surface elevation at 5, 15, 30 m ahead;
use it to time flare with wave phase.
Derive throttle and pitch from the current observation, mission and
previous control; do not repeat a fixed value. No text or code.'''


class PilotError(RuntimeError):
    """Local service or model response cannot be used as a control."""


@dataclass(frozen=True)
class Control:
    throttle: float
    pitch_deg: float

    @classmethod
    def parse(cls, text):
        try:
            data = json.loads(text)
        except (ValueError, TypeError) as exc:
            raise PilotError('model did not return valid JSON') from exc
        if not isinstance(data, dict) or set(data) != {'throttle', 'pitch_deg'}:
            raise PilotError('expected exactly throttle and pitch_deg')
        for name, low, high in [('throttle', 0, 1), ('pitch_deg', -8, 15)]:
            value = data[name]
            if (type(value) not in (int, float) or not math.isfinite(value)
                    or not low <= value <= high):
                raise PilotError(f'{name} is not a finite number in [{low}, {high}]')
        return cls(float(data['throttle']), float(data['pitch_deg']))

    def apply(self, vehicle):
        # Direct servo mode bypasses navigation setpoints; vehicle.step still
        # enforces disarmed/failure thrust shutdown.
        vehicle.send_servo(1, 1000 + 1000 * self.throttle)
        vehicle.send_servo(2, 1500 + 500 * self.pitch_deg / 15)


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise PilotError('Ollama redirects are not supported')


class OllamaPilot:
    def __init__(self, model='phi3.5:latest', base_url='http://127.0.0.1:11434',
                 timeout=60.0, seed=0):
        url = urlsplit(base_url)
        if (url.scheme != 'http' or url.hostname not in ('127.0.0.1', 'localhost', '::1')
                or url.username or url.password or url.path not in ('', '/')
                or url.query or url.fragment):
            raise ValueError('Ollama endpoint must be a local HTTP origin')
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError('timeout must be positive and finite')
        if not isinstance(model, str) or not model.strip():
            raise ValueError('model is required')
        self.model, self.base_url = model, base_url.rstrip('/')
        self.timeout, self.seed = timeout, seed
        self.opener = build_opener(ProxyHandler({}), NoRedirect())

    def _request(self, path, payload=None):
        data = None if payload is None else json.dumps(payload, allow_nan=False).encode()
        request = Request(self.base_url + path, data=data,
                          headers={'Content-Type': 'application/json'})
        try:
            with self.opener.open(request, timeout=self.timeout) as response:
                body = response.read(1_048_577)
            if len(body) > 1_048_576:
                raise PilotError('Ollama response exceeds 1 MiB')
            result = json.loads(body)
            if not isinstance(result, dict):
                raise PilotError('invalid Ollama response envelope')
            if result.get('error'):
                raise PilotError(str(result['error']))
            return result
        except (URLError, TimeoutError, OSError, ValueError, HTTPException) as exc:
            raise PilotError(f'local Ollama request failed: {exc}') from exc

    def check_model(self):
        models = self._request('/api/tags').get('models', [])
        if not isinstance(models, list) or any(not isinstance(m, dict) for m in models):
            raise PilotError('invalid local model inventory')
        expected = self.model if ':' in self.model else self.model + ':latest'
        for model in models:
            if model.get('name') == expected or model.get('model') == expected:
                return model
        raise PilotError(f'{expected} is not installed locally; run ollama pull {expected}')

    def decide(self, observation, mission, previous=None):
        payload = {
            'model': self.model, 'stream': False, 'format': CONTROL_SCHEMA,
            'messages': [{'role': 'system', 'content': SYSTEM_PROMPT},
                         {'role': 'user', 'content': json.dumps({
                             'mission': mission, 'observation': observation,
                             'previous_control': asdict(previous) if previous else None,
                         }, allow_nan=False)}],
            'options': {'temperature': 0, 'seed': self.seed, 'num_predict': 64,
                        'num_ctx': 2048},
            'keep_alive': '5m',
        }
        started = time.monotonic()
        result = self._request('/api/chat', payload)
        if result.get('done') is not True or result.get('done_reason') == 'length':
            raise PilotError('model response was incomplete')
        message = result.get('message')
        if not isinstance(message, dict) or not isinstance(message.get('content'), str):
            raise PilotError('missing assistant content')
        control = Control.parse(message['content'])
        return control, {'latency_s': time.monotonic() - started,
                         'raw_response': message['content'],
                         'eval_count': result.get('eval_count'),
                         'total_duration_ns': result.get('total_duration')}


def observe(vehicle):
    eta = float(vehicle.sea.eta([vehicle.x], vehicle.t)[0])
    wave_rate = (float(vehicle.sea.eta([vehicle.x], vehicle.t + vehicle.dt)[0]) - eta) / vehicle.dt
    preview = wave_preview(vehicle.sea, vehicle.x, vehicle.t, vehicle.Vx)
    return dict(t=vehicle.t, x_m=vehicle.x, altitude_m=vehicle.z,
                forward_speed_m_s=vehicle.Vx, vertical_speed_m_s=vehicle.Vz,
                wave_elevation_m=eta, wave_rate_m_s=wave_rate,
                wave_preview_m=[float(v) for v in preview],
                keel_clearance_m=vehicle.z - vehicle.hull.h_keel - eta,
                water_mass_kg=vehicle.damage.water_mass, failed=vehicle.damage.failed)


def fallback_control(vehicle, scenario, target_alt, target_speed):
    state = {'z': vehicle.z, 'Vx': vehicle.Vx, 'Vz': vehicle.Vz}
    if scenario == 'takeoff':
        pitch, throttle = vehicle.ctl.takeoff_setpoint(state, target_alt, target_speed)
    else:
        pitch, throttle = vehicle.ctl.landing_setpoint(state, 0, math.radians(8))
    return Control(float(throttle), math.degrees(pitch))
