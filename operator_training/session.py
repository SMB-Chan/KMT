"""Single-owner training state machine. All times passed by caller are monotonic."""
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import uuid
import numpy as np
from aircraft import Aircraft
from atmosphere import AtmosphereConfig
from dynamics import hull_force
from mavlink_if import (FlyingBoatVehicle, MAV_CMD_NAV_TAKEOFF,
                        MAV_CMD_NAV_WAYPOINT, MAV_CMD_NAV_LAND)
from ocean_directional import DirectionalOcean
from ollama_pilot import SpatialControl, observe

DT = .05
LIMITS = {'throttle': [0, 1], 'pitch_deg': [-8, 12], 'bank_deg': [-25, 25], 'rudder': [-1, 1]}
ROOT = Path(__file__).resolve().parents[1]


def control_value(data):
    if not isinstance(data, dict) or set(data) != set(LIMITS):
        raise ValueError('four control axes required')
    for key, (lo, hi) in LIMITS.items():
        v = data[key]
        if type(v) not in (int, float) or not math.isfinite(v) or not lo <= v <= hi:
            raise ValueError(f'invalid {key}')
    return SpatialControl(**data)


class Session:
    def __init__(self, config, now=0., log_root=None):
        if not isinstance(config, dict) or set(config) - {'mode', 'exercise', 'hs', 'seed', 'input_device', 'calibration'}:
            raise ValueError('unknown configuration')
        mode, exercise = config.get('mode', 'hybrid'), config.get('exercise', 'takeoff')
        hs, seed = config.get('hs', .3), config.get('seed', 42)
        if mode not in ('hybrid', 'manual', 'observe') or exercise not in ('takeoff', 'cruise', 'landing'):
            raise ValueError('invalid mode / exercise')
        if type(hs) not in (int, float) or hs not in (0, .3, .6):
            raise ValueError('introductory sea states: 0, 0.3, 0.6 m')
        if type(seed) is not int or not 0 <= seed <= 999999:
            raise ValueError('invalid seed')
        if config.get('input_device', 'keyboard') not in ('keyboard', 'gamepad'):
            raise ValueError('invalid input device')
        self.id = uuid.uuid4().hex
        self.config = dict(config, mode=mode, exercise=exercise, hs=hs, seed=seed)
        self.vehicle = FlyingBoatVehicle(Aircraft(), DirectionalOcean(Hs=hs, Tp=6, seed=seed),
                                        spatial=True, atmosphere=AtmosphereConfig(), seed=seed)
        v = self.vehicle
        if exercise != 'takeoff':
            v.z, v.Vx, v.Vz = (25., 13., 0.) if exercise == 'cruise' else (25., 13*math.cos(math.radians(8)), -13*math.sin(math.radians(8)))
        self.lifecycle, self.phase = 'READY', {'takeoff': 'TAKEOFF', 'cruise': 'CRUISE', 'landing': 'APPROACH'}[exercise]
        self.authority = 'HUMAN' if mode == 'manual' else 'AUTO'
        self.reason = '開始確認待ち'
        self.tick = 0
        self.seq = -1
        self.applied_seq = None
        self.input = SpatialControl(0, 0, 0, 0)
        self.last_input = self.last_heartbeat = now
        self.stable = self.matched = 0.
        self.pending_land = False
        self.events = {}
        self.finished = False
        self.metrics = {'peak_sampled_load_w': 0., 'interventions': 0, 'paused_count': 0}
        self.log_dir = None
        self.files = {}
        if log_root is not None:
            self.log_dir = Path(log_root) / self.id
            self.log_dir.mkdir(parents=True)
            sources = list(ROOT.glob('*.py')) + list((ROOT / 'operator_training').glob('*.py'))
            manifest = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
            (self.log_dir / 'config.json').write_text(json.dumps(dict(schema=1, config=self.config, limits=LIMITS,
                source_sha256=manifest, numpy_version=np.__version__), ensure_ascii=False, indent=2))
            for name in ('inputs', 'ticks', 'events'):
                self.files[name] = (self.log_dir / f'{name}.jsonl').open('a', buffering=1)

    def record(self, name, data):
        if name not in self.files:
            return
        try:
            self.files[name].write(json.dumps(data, ensure_ascii=False, allow_nan=False) + '\n')
        except (OSError, ValueError):
            self.lifecycle, self.reason = 'PAUSED', 'RECORDING_ERROR'
            raise

    def pause(self, reason):
        if self.lifecycle == 'RUNNING':
            self.lifecycle, self.reason = 'PAUSED', reason
            self.metrics['paused_count'] += 1
            self.matched = 0.
            self.record('events', dict(type='pause', reason=reason, tick=self.tick))

    def accept_input(self, seq, data, now):
        if type(seq) is not int or seq < 0:
            raise ValueError('invalid sequence')
        value = control_value(data)
        if seq <= self.seq:
            return False
        self.seq, self.input, self.last_input = seq, value, now
        if self.lifecycle == 'RUNNING':
            self.record('inputs', dict(seq=seq, received_monotonic=now, control=asdict(value)))
        return True

    def applied(self):
        v = self.vehicle
        return dict(throttle=float(v.throttle), pitch_deg=math.degrees(v.alpha),
                    bank_deg=math.degrees(v.bank_command), rudder=float(v.rudder_command))

    def input_matches(self):
        return all(abs(asdict(self.input)[k] - self.applied()[k]) <= tol
                   for k, tol in zip(LIMITS, (.05, 2., 3., .1)))

    def gate(self):
        v = self.vehicle
        reasons = []
        # Experimental introductory gate, NOT a certified capture envelope.
        tests = [(20 <= v.z <= 30, '高度20–30 m'), (11 <= observe(v)['airspeed_m_s'] <= 15, '対気速度11–15 m/s'),
                 (-2.5 <= v.Vz <= -.5, '降下率0.5–2.5 m/s'), (abs(v.y) <= 10, '横ずれ±10 m'),
                 (abs(math.degrees(v.heading)) <= 10, '北向き±10°'),
                 (abs(math.degrees(v.bank)) <= 8, 'バンク±8°'), (not v.damage.failed, '機体正常')]
        for ok, label in tests:
            if not ok:
                reasons.append(label)
        return reasons

    def _auto_command(self):
        cmd = {'TAKEOFF': MAV_CMD_NAV_TAKEOFF, 'CRUISE': MAV_CMD_NAV_WAYPOINT,
               'APPROACH': MAV_CMD_NAV_LAND}[self.phase]
        self.vehicle.send_command(cmd, dict(alt=0 if self.phase == 'APPROACH' else 25,
                                           speed=13, glide=8, heading=0, y=0))

    def event(self, event_id, action, now):
        if not isinstance(event_id, str) or not 1 <= len(event_id) <= 80:
            raise ValueError('invalid event id')
        if event_id in self.events:
            return self.events[event_id]
        ok, reason = True, ''
        fresh = now - self.last_input <= .25 and now - self.last_heartbeat <= .5
        if self.finished:
            return dict(type='ack', event_id=event_id, action=action, accepted=False, reason='試行は終了済みです', tick=self.tick)
        if action == 'start' and self.lifecycle == 'READY':
            if not fresh or self.seq < 0:
                ok, reason = False, '新しい入力が必要です'
            else:
                self.vehicle.arm()
                if self.authority == 'AUTO':
                    self._auto_command()
                self.lifecycle, self.reason = 'RUNNING', ''
        elif action == 'pause':
            self.pause('USER_PAUSE')
        elif action == 'resume' and self.lifecycle == 'PAUSED':
            if self.reason in ('SERVER_OR_RECORDING_ERROR', 'RECORDING_ERROR', 'STATE_ERROR') or not fresh or (self.authority == 'HUMAN' and not self.input_matches()):
                ok, reason = False, '通信確認と適用値への入力合わせが必要です'
            else:
                self.lifecycle, self.reason = 'RUNNING', ''
        elif action == 'take_control' and self.lifecycle == 'RUNNING' and self.authority == 'AUTO':
            if not fresh or self.config['mode'] == 'observe' or self.stable < 1 or self.matched < .5:
                ok, reason = False, '安定飛行1秒・入力一致0.5秒が必要です'
            else:
                self.authority = 'HUMAN'
                self.phase = 'CRUISE'
                self.metrics['interventions'] += 1
        elif action == 'request_auto_land' and self.lifecycle == 'RUNNING':
            self.pending_land = True
        elif action == 'confirm_auto_land' and self.lifecycle == 'RUNNING' and self.pending_land:
            missing = self.gate()
            if missing:
                ok, reason = False, '未達: ' + ' / '.join(missing)
            else:
                self.authority, self.phase, self.pending_land = 'AUTO', 'APPROACH', False
                self.metrics['interventions'] += 1
                self._auto_command()
        elif action == 'cancel':
            self.pending_land = False
        elif action == 'abort':
            self.finish('ABORTED', 'USER_ABORT')
        else:
            ok, reason = False, 'この状態では実行できません'
        ack = dict(type='ack', event_id=event_id, action=action, accepted=ok, reason=reason, tick=self.tick)
        if len(self.events) >= 4096:
            raise ValueError('event limit reached; create a new session')
        self.events[event_id] = ack
        self.record('events', ack)
        return ack

    def advance(self, now):
        if self.lifecycle != 'RUNNING':
            return
        if now - self.last_heartbeat > .5 or (self.authority == 'HUMAN' and now - self.last_input > .25):
            self.pause('INPUT_TIMEOUT')
            return
        v = self.vehicle
        self.matched = self.matched + DT if self.input_matches() and now-self.last_input <= .25 else 0
        if self.authority == 'HUMAN':
            self.input.apply(v)
            self.applied_seq = self.seq
        else:
            self.applied_seq = None
        v.step(DT)
        self.tick += 1
        obs = observe(v)
        stable = v.z >= 8 and obs['airspeed_m_s'] >= 1.3*v.ac.V_stall and abs(v.Vz) < 2 and abs(v.bank) < math.radians(10)
        self.stable = self.stable + DT if stable else 0.
        if self.authority == 'AUTO' and self.phase == 'TAKEOFF' and v.z >= 24:
            self.phase = 'CRUISE'
            self._auto_command()
        snap = v.read_telemetry()[2] or {}
        load = float(snap.get('N_water', 0))/v.ac.W
        self.metrics['peak_sampled_load_w'] = max(self.metrics['peak_sampled_load_w'], load)
        # Avoid unbounded legacy telemetry accumulation; tick logs retain history.
        if len(v._msg_log) > 2:
            del v._msg_log[:-2]
        if v.damage.failed or load > 8 or abs(v.y) > 500 or v.z > 150:
            self.finish('FINISHED', 'FAILURE_OR_LIMIT')
        elif self.phase != 'TAKEOFF' and v.z-v.hull.h_keel < v.wave_elevation():
            contact = hull_force(v.z, v.Vx, v.Vz, v.wave_elevation(), v.hull)
            good = abs(v.Vz) < 1.5 and max(contact.N/v.ac.W, load) < 3 and v.lateral_success()
            self.finish('FINISHED', 'TOUCHDOWN_OK' if good else 'HARD_LANDING')
        elif self.phase == 'TAKEOFF' and self.authority == 'HUMAN' and self.stable >= 1:
            self.phase = 'CRUISE'
        if v.t >= 600:
            self.finish('FINISHED', 'TIME_LIMIT')
        self.record('ticks', self.snapshot())

    def finish(self, state, reason):
        self.lifecycle, self.reason, self.finished = state, reason, True
        if self.log_dir:
            tmp = self.log_dir / 'summary.tmp'
            tmp.write_text(json.dumps(self.summary(), ensure_ascii=False, indent=2))
            tmp.replace(self.log_dir / 'summary.json')

    def summary(self):
        return dict(session_id=self.id, state=self.lifecycle, reason=self.reason, sim_time_s=self.vehicle.t,
                    metrics=self.metrics, mode=self.config['mode'], input_device=self.config.get('input_device'),
                    note='実験用。初接触までの評価。荷重は20Hz標本。実機技能の認定ではない。')

    def snapshot(self):
        v = self.vehicle
        return dict(type='state', session_id=self.id, tick=self.tick, sim_time_s=v.t,
                    lifecycle=self.lifecycle, phase=self.phase, authority=self.authority, reason=self.reason,
                    applied_control=self.applied(), position=[v.x, v.y, v.z],
                    attitude=[v.alpha, v.bank, v.heading], observation=observe(v),
                    requested_control=asdict(self.input), last_input_seq=self.seq, applied_input_seq=self.applied_seq,
                    stable_s=self.stable, matched_s=self.matched, pending_land=self.pending_land,
                    gate_missing=self.gate() if self.pending_land else [], metrics=dict(self.metrics))

    def spectrum(self):
        s = self.vehicle.sea
        terms = []
        # Same coefficients, phases and clock as the physical contact surface.
        for i in range(s.n_freq):
            for j in range(s.n_dir):
                terms.append([float(s.k[i]*s.cos_t[j]), float(s.k[i]*s.sin_t[j]),
                              float(s.omega[i]), float(s.amps[i, j]), float(s.phases[i, j])])
        return terms

    def close(self):
        for f in self.files.values():
            f.close()
