import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from teacher_student import StudentPilot, load_demonstrations
from fly_ollama import parser, run
from ollama_pilot import observe
from mavlink_if import FlyingBoatVehicle
from aircraft import Aircraft
from ocean import Ocean


class StudentTests(unittest.TestCase):
    def test_fit_save_reload(self):
        p = StudentPilot()
        x = np.random.default_rng(4).normal(size=(20, 13))
        y = np.tile([.3, -.5], (20, 1))
        before = np.mean((p.predict(x)-y)**2)
        p.fit(x, y, epochs=300)
        self.assertLess(np.mean((p.predict(x)-y)**2), before*.1)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'student.npz'; p.save(path)
            np.testing.assert_allclose(StudentPilot.load(path).predict(x), p.predict(x))
            with self.assertRaises(FileExistsError): p.save(path)

    def test_only_teacher_records_are_used(self):
        obs = observe(FlyingBoatVehicle(Aircraft(), Ocean()))
        mission = dict(scenario='takeoff', target_altitude_m=8, target_speed_m_s=8.5)
        record = dict(observation=obs, mission=mission, control={'throttle':1, 'pitch_deg':4})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'decisions.jsonl'
            path.write_text('\n'.join(json.dumps(dict(record, source=s)) for s in ('ollama','fallback','student','aborted')))
            x, y, sources, skipped = load_demonstrations([path])
            self.assertEqual(x.shape, (1,13))
            self.assertEqual(skipped, 3)

    def test_offline_flight_steps_without_ollama(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp)/'student.npz'; StudentPilot().save(checkpoint)
            args = parser().parse_args(['--student',str(checkpoint), '--duration','.15',
                                        '--interval','.05','--strict','--output',str(Path(tmp)/'flight')])
            with contextlib.redirect_stdout(io.StringIO()): summary = run(args)
            self.assertEqual(summary['decisions'], 3)
            self.assertEqual(summary['fallback_decisions'], 0)
            self.assertEqual(summary['model'], 'phi-student')
