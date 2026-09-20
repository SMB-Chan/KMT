"""Core tests deliberately require no FastAPI or physical gamepad."""
import json
import math
from pathlib import Path
import tempfile
import unittest
import numpy as np
from operator_training.session import Session, control_value, DT

ZERO = dict(throttle=0, pitch_deg=0, bank_deg=0, rudder=0)


class OperatorTrainingTests(unittest.TestCase):
    def session(self, **kwargs):
        return Session(dict(mode='manual', exercise='cruise', hs=0, **kwargs))

    def start(self, s, now=0):
        s.accept_input(0, ZERO, now)
        s.last_heartbeat = now
        return s.event('start', 'start', now)

    def test_config_and_finite_controls(self):
        for val in (float('nan'), True, 26):
            with self.assertRaises(ValueError):
                control_value(dict(ZERO, bank_deg=val))
        for cfg in ({'mode': 'unknown'}, {'hs': 5}, {'seed': True}, {'unknown': 0}):
            with self.assertRaises(ValueError):
                Session(cfg)

    def test_input_monotonic_and_duplicate_events(self):
        s = self.session()
        self.assertTrue(s.accept_input(2, ZERO, 0))
        self.assertFalse(s.accept_input(1, dict(ZERO, throttle=1), .1))
        self.assertEqual(s.input.throttle, 0)
        a = s.event('one', 'start', 0)
        b = s.event('one', 'abort', 0)
        self.assertEqual(a, b)
        self.assertEqual(s.lifecycle, 'RUNNING')

    def test_timeout_freezes_physics_and_damage(self):
        s = self.session()
        self.start(s)
        damage = repr(s.vehicle.damage)
        s.advance(.3)
        self.assertEqual(s.lifecycle, 'PAUSED')
        self.assertEqual(s.vehicle.t, 0)
        self.assertEqual(repr(s.vehicle.damage), damage)
        s.advance(5)
        self.assertEqual(s.tick, 0)

    def test_auto_ignores_direct_human_input(self):
        s = Session(dict(mode='hybrid', exercise='takeoff', hs=0))
        s.accept_input(0, dict(ZERO, pitch_deg=-8, bank_deg=25), 0)
        s.event('start', 'start', 0)
        s.advance(.05)
        self.assertNotEqual(s.vehicle.alpha, math.radians(-8))
        self.assertNotEqual(s.vehicle.bank_command, math.radians(25))

    def test_start_and_resume_need_fresh_input_and_alignment(self):
        s = self.session()
        self.assertFalse(s.event('early', 'start', 0)['accepted'])
        self.start(s)
        s.advance(.05)
        s.pause('TEST')
        s.accept_input(1, dict(ZERO, throttle=1), .1)
        self.assertFalse(s.event('bad-resume', 'resume', .1)['accepted'])
        s.accept_input(2, s.applied(), .1)
        self.assertTrue(s.event('good-resume', 'resume', .1)['accepted'])

    def test_takeover_guard_and_atomic_apply(self):
        s = Session(dict(mode='hybrid', exercise='cruise', hs=0))
        self.start(s)
        self.assertFalse(s.event('early', 'take_control', 0)['accepted'])
        s.stable, s.matched = 1, .5
        self.assertTrue(s.event('ready', 'take_control', 0)['accepted'])
        s.accept_input(1, dict(ZERO, throttle=.6, pitch_deg=3, bank_deg=10), .05)
        s.advance(.05)
        self.assertEqual(s.authority, 'HUMAN')
        self.assertAlmostEqual(s.vehicle.alpha, math.radians(3))
        self.assertAlmostEqual(s.vehicle.bank_command, math.radians(10))

    def test_landing_gate_requires_request_and_geometry(self):
        s = self.session()
        self.start(s)
        self.assertFalse(s.event('no-request', 'confirm_auto_land', 0)['accepted'])
        s.event('request', 'request_auto_land', 0)
        self.assertFalse(s.event('level-flight', 'confirm_auto_land', 0)['accepted'])
        s.vehicle.Vz = -1.8
        self.assertTrue(s.event('descent', 'confirm_auto_land', 0)['accepted'])
        self.assertEqual((s.authority, s.phase), ('AUTO', 'APPROACH'))

    def test_spectrum_matches_contact_surface(self):
        s = Session(dict(hs=.3))
        terms = s.spectrum()
        for x,y,t in [(0,0,0),(18,-5,2.3),(-100,19,40)]:
            render = sum(a*math.cos(kx*x+ky*y-w*t+p) for kx,ky,w,a,p in terms)
            true = float(s.vehicle.sea.eta([x],[y],t)[0,0])
            self.assertAlmostEqual(render, true, places=10)

    def test_logging_and_summary(self):
        with tempfile.TemporaryDirectory() as path:
            s = Session(dict(mode='manual', exercise='cruise', hs=0), log_root=path)
            self.start(s)
            for i in range(1, 5):
                s.accept_input(i, dict(ZERO, throttle=.6), i*DT)
                s.last_heartbeat = i*DT
                s.advance(i*DT)
            s.event('end', 'abort', .2)
            s.close()
            root = Path(path)/s.id
            self.assertEqual(len((root/'ticks.jsonl').read_text().splitlines()), 4)
            data = json.loads((root/'summary.json').read_text())
            self.assertEqual(data['reason'], 'USER_ABORT')
            self.assertEqual(data['sim_time_s'], .2)
            self.assertIn('source_sha256', json.loads((root/'config.json').read_text()))

    def test_explicit_pause_stops_clock(self):
        s = self.session()
        self.start(s)
        s.advance(.05)
        s.event('pause', 'pause', .05)
        for now in [1, 5, 100]:
            s.advance(now)
        self.assertEqual(s.vehicle.t, .05)
