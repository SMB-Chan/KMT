import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock
from urllib.error import URLError

from aircraft import Aircraft
from ocean import Ocean
from mavlink_if import FlyingBoatVehicle
from ollama_pilot import Control, OllamaPilot, PilotError, observe
from fly_ollama import parser, run


class OllamaTests(unittest.TestCase):
    def test_prompt_prescribes_no_fixed_command(self):
        from ollama_pilot import SYSTEM_PROMPT
        for canned in ["about 4 degrees", "about -5 degrees",
                       "full throttle", "low throttle"]:
            self.assertNotIn(canned, SYSTEM_PROMPT)
        self.assertIn("Derive throttle and pitch", SYSTEM_PROMPT)

    def test_control_schema_rejects_unusable_outputs(self):
        for text in ['no JSON', '[]', '{"throttle":1}',
                     '{"throttle":true,"pitch_deg":0}',
                     '{"throttle":1,"pitch_deg":NaN}',
                     '{"throttle":2,"pitch_deg":0}',
                     '{"throttle":1,"pitch_deg":16}',
                     '{"throttle":1,"pitch_deg":0,"command":"arm"}']:
            with self.subTest(text=text), self.assertRaises(PilotError):
                Control.parse(text)
        self.assertEqual(Control.parse('{"throttle":1,"pitch_deg":-5}'), Control(1, -5))

    def test_control_applied_without_rl_pitch_clipping(self):
        v = FlyingBoatVehicle(Aircraft(), Ocean(Hs=0))
        v.arm()
        Control(.1, -5).apply(v); v.step()
        self.assertAlmostEqual(v.throttle, .1)
        self.assertAlmostEqual(__import__('math').degrees(v.alpha), -5)
        v.disarm(); Control(1, 15).apply(v); v.step()
        self.assertEqual(v.read_telemetry()[2]['T'], 0)

    def test_local_endpoint_only(self):
        for host in ['https://example.com', 'http://example.com', 'http://localhost/path',
                     'http://user@localhost', 'http://localhost?x=1']:
            with self.assertRaises(ValueError): OllamaPilot(base_url=host)

    def test_schema_and_prompt_are_sent(self):
        pilot = OllamaPilot()
        pilot._request = Mock(return_value={'done': True, 'message': {'content': '{"throttle":0.8,"pitch_deg":4}'}})
        control, metadata = pilot.decide({'altitude_m': 0}, {'scenario':'takeoff'})
        self.assertEqual(control, Control(.8, 4))
        payload = pilot._request.call_args.args[1]
        self.assertFalse(payload['stream'])
        self.assertEqual(payload['format']['required'], ['throttle', 'pitch_deg'])
        self.assertEqual(payload['model'], 'phi3.5:latest')
        self.assertGreaterEqual(metadata['latency_s'], 0)

    def test_incomplete_response_rejected(self):
        pilot = OllamaPilot()
        for result in [{'done':False}, {'done':True, 'done_reason':'length'}, {'done':True, 'message':None}]:
            pilot._request = Mock(return_value=result)
            with self.assertRaises(PilotError): pilot.decide({}, {})

    def test_transport_failure_is_pilot_error(self):
        pilot = OllamaPilot()
        for error in [TimeoutError('slow'), URLError('offline')]:
            pilot.opener.open = Mock(side_effect=error)
            with self.assertRaises(PilotError): pilot.check_model()

    def test_missing_model_is_reported(self):
        pilot = OllamaPilot(model='phi3.5')
        pilot._request = Mock(return_value={'models':[]})
        with self.assertRaisesRegex(PilotError, 'not installed'): pilot.check_model()
        pilot._request.return_value = {'models':[{'name':'phi3.5:latest'}]}
        self.assertEqual(pilot.check_model()['name'], 'phi3.5:latest')

    def test_http_payload_and_decoding(self):
        pilot = OllamaPilot()
        response = Mock()
        response.read.return_value = b'{"models":[]}'
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        pilot.opener.open = Mock(return_value=response)
        self.assertEqual(pilot._request('/api/tags'), {'models':[]})
        request = pilot.opener.open.call_args.args[0]
        self.assertEqual(request.full_url, 'http://127.0.0.1:11434/api/tags')
        self.assertEqual(pilot.opener.open.call_args.kwargs['timeout'], 60)

    def test_runner_hold_cadence_logs_and_fallback(self):
        pilot = Mock()
        pilot.check_model.return_value = {'name':'phi3.5:latest'}
        pilot.decide.side_effect = [(Control(.9, 4), {'latency_s':.1}), PilotError('timeout')]
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp)/'flight'
            args = parser().parse_args(['--duration','.2','--interval','.1','--output',str(target)])
            with contextlib.redirect_stdout(io.StringIO()): summary = run(args, pilot)
            self.assertEqual(summary['physics_steps'], 4)
            self.assertEqual(summary['decisions'], 2)
            self.assertEqual(summary['fallback_decisions'], 1)
            rows = [json.loads(x) for x in (target/'trajectory.jsonl').read_text().splitlines()]
            self.assertEqual([x['source'] for x in rows], ['ollama','ollama','fallback','fallback'])
            self.assertEqual(rows[0]['applied_throttle'], .9)
            self.assertEqual(json.loads((target/'summary.json').read_text())['status'], 'time_limit')

    def test_strict_aborts_before_applying_fallback(self):
        pilot = Mock()
        pilot.check_model.return_value = {'name':'phi3.5:latest'}
        pilot.decide.side_effect = PilotError('invalid')
        with tempfile.TemporaryDirectory() as temp:
            args = parser().parse_args(['--strict','--output', str(Path(temp)/'strict')])
            with contextlib.redirect_stdout(io.StringIO()): summary = run(args, pilot)
            self.assertEqual(summary['status'], 'pilot_error')
            self.assertEqual(summary['physics_steps'], 0)

    def test_observation_includes_wave_clearance(self):
        v = FlyingBoatVehicle(Aircraft(), Ocean(Hs=0))
        obs = observe(v)
        self.assertAlmostEqual(obs['keel_clearance_m'], v.z - v.hull.h_keel)
        self.assertEqual(obs['wave_elevation_m'], 0)


if __name__ == '__main__':
    unittest.main()
