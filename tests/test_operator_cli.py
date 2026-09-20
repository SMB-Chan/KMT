"""Smoke tests for the operator_training CLI entry point."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[1]


class CliSmokeTests(unittest.TestCase):
    def test_smoke_subcommand(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "smoke"
            # Use enough ticks to actually complete handover; the smoke
            # driver returns exit 1 if handover did not complete in the
            # tick budget.
            result = subprocess.run(
                [sys.executable, "-m", "operator_training", "smoke",
                 str(out), "--ticks", "200"],
                cwd=str(WORKSPACE),
                capture_output=True, text=True, timeout=120,
            )
            self.assertEqual(result.returncode, 0,
                             f"stderr: {result.stderr}")
            payload = json.loads(result.stdout.strip())
            self.assertTrue(payload["handover_requested"])
            self.assertTrue(payload["handover_completed"])
            self.assertTrue((out / "config.json").exists())
            self.assertTrue((out / "ticks.jsonl").exists())

    def test_replay_subcommand_passes_after_smoke(self):
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "src"
            tgt = Path(td) / "tgt"
            # Run smoke first to produce the source
            subprocess.run(
                [sys.executable, "-m", "operator_training", "smoke",
                 str(src), "--ticks", "200"],
                cwd=str(WORKSPACE), check=True,
                capture_output=True, timeout=120,
            )
            result = subprocess.run(
                [sys.executable, "-m", "operator_training", "replay",
                 str(src), str(tgt)],
                cwd=str(WORKSPACE),
                capture_output=True, text=True, timeout=120,
            )
            self.assertEqual(result.returncode, 0,
                             f"stderr: {result.stderr}")
            payload = json.loads(result.stdout.strip())
            self.assertTrue(payload["passed"], payload)
            self.assertGreater(payload["ticks_compared"], 0)

    def test_headless_keys_subcommand(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "keys"
            result = subprocess.run(
                [sys.executable, "-m", "operator_training", "headless-keys",
                 str(out), "--ticks", "100"],
                cwd=str(WORKSPACE),
                capture_output=True, text=True, timeout=60,
            )
            self.assertEqual(result.returncode, 0,
                             f"stderr: {result.stderr}")
            payload = json.loads(result.stdout.strip())
            self.assertEqual(payload["lifecycle_end"], "RUNNING")
            self.assertEqual(payload["authority_end"], "HUMAN")
            self.assertGreater(payload["sim_t_end"], 0.0)

    def test_unknown_subcommand_fails(self):
        result = subprocess.run(
            [sys.executable, "-m", "operator_training", "no-such-cmd"],
            cwd=str(WORKSPACE),
            capture_output=True, text=True, timeout=30,
        )
        self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
