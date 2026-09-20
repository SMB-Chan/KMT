"""Tests for operator_training.flight_report.

These tests build synthetic session directories (config.json + JSONL
streams) so they exercise the metric computation against known data
without depending on a live Session.
"""
import json
import tempfile
import unittest
from pathlib import Path

from operator_training.flight_report import (
    FlightMetrics,
    compare_metrics,
    compute_metrics,
    render_markdown,
    write_long_csv,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True))
            f.write("\n")


def _make_session(tmp: Path, *, ticks: list[dict], inputs: list[dict],
                  events: list[dict], summary: dict,
                  config: dict | None = None) -> Path:
    sd = tmp / "session"
    sd.mkdir(parents=True, exist_ok=True)
    cfg = {
        "session_id": "synthetic",
        "curriculum_id": "hybrid_baseline",
        "curriculum_version": "1",
        "scenario": "takeoff",
        "seed": 42,
        "spatial": True,
        "control_envelope": {"pitch_lo": -8.0, "pitch_hi": 15.0,
                              "bank_abs": 25.0, "rudder_abs": 1.0},
    }
    if config:
        cfg.update(config)
    (sd / "config.json").write_text(json.dumps(cfg, indent=2, sort_keys=True))
    _write_jsonl(sd / "ticks.jsonl", ticks)
    _write_jsonl(sd / "inputs.jsonl", inputs)
    _write_jsonl(sd / "events.jsonl", events)
    (sd / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    return sd


def _tick(sim_t: float, *, altitude: float = 1.0, airspeed: float = 5.0,
          forward: float = 5.0, bank: float = 0.0, pitch_cmd: float = 4.0,
          throttle: float = 0.8, rudder: float = 0.0,
          water_kg: float = 0.0, failed: bool = False,
          vertical: float = 0.0, keel: float = 0.1,
          authority: str = "AUTO") -> dict:
    return {
        "kind": "tick",
        "session_id": "synthetic",
        "tick": int(sim_t * 20),
        "sim_t": sim_t,
        "authority": authority,
        "assist": False,
        "applied_seq": None,
        "control": {
            "throttle": throttle,
            "pitch_deg": pitch_cmd,
            "bank_deg": bank,
            "rudder": rudder,
        },
        "state": {
            "altitude_m": altitude,
            "airspeed_m_s": airspeed,
            "forward_speed_m_s": forward,
            "vertical_speed_m_s": vertical,
            "bank_deg": bank,
            "heading_deg": 0.0,
            "lateral_position_m": 0.0,
            "lateral_speed_m_s": 0.0,
            "wave_elevation_m": 0.0,
            "keel_clearance_m": keel,
        },
        "damage": {"failed": failed, "water_kg": water_kg},
    }


def _input(seq: int, *, applied: bool, authority: str = "AUTO") -> dict:
    return {
        "kind": "input",
        "session_id": "synthetic",
        "seq": seq,
        "receive_t": float(seq) * 0.05,
        "applied": applied,
        "reason": f"authority={authority.lower()}" if not applied else "ok",
        "raw": {"throttle": 0.7, "pitch_deg": 5.0,
                "bank_deg": 0.0, "rudder": 0.0},
        "request": {"throttle": 0.7, "pitch_deg": 5.0,
                    "bank_deg": 0.0, "rudder": 0.0} if applied else None,
    }


class ComputeMetricsTests(unittest.TestCase):
    def test_empty_session_returns_zero_metrics(self):
        with tempfile.TemporaryDirectory() as td:
            sd = _make_session(
                Path(td), ticks=[], inputs=[], events=[],
                summary={"lifecycle_end": "FINISHED", "phase_end": "TOUCHDOWN",
                         "authority_end": "AUTO", "sim_t_end": 0.0,
                         "tick_count": 0},
            )
            m = compute_metrics(sd)
            self.assertEqual(m.tick_count, 0)
            self.assertEqual(m.max_altitude_m, 0.0)
            self.assertEqual(m.outcome, "incomplete")
            self.assertTrue(m.session_dir.endswith("session"))

    def test_successful_takeoff_metrics(self):
        with tempfile.TemporaryDirectory() as td:
            sd = _make_session(
                Path(td),
                ticks=[
                    _tick(0.0, altitude=0.1, airspeed=0.0, forward=0.0,
                          pitch_cmd=4.0, throttle=1.0, bank=0.0,
                          keel=-0.05),
                    _tick(1.0, altitude=0.5, airspeed=4.0, forward=4.0,
                          pitch_cmd=6.0, throttle=1.0, bank=1.0,
                          keel=0.3),
                    _tick(2.0, altitude=2.0, airspeed=8.0, forward=8.0,
                          pitch_cmd=8.0, throttle=1.0, bank=0.5,
                          keel=1.5),
                    _tick(3.0, altitude=5.0, airspeed=12.0, forward=12.0,
                          pitch_cmd=10.0, throttle=0.9, bank=0.3,
                          keel=4.8),
                    _tick(4.0, altitude=10.0, airspeed=14.0, forward=14.0,
                          pitch_cmd=8.0, throttle=0.7, bank=0.2,
                          keel=9.8),
                ],
                inputs=[
                    _input(1, applied=True, authority="HUMAN"),
                    _input(2, applied=True, authority="HUMAN"),
                    _input(3, applied=False),
                ],
                events=[
                    {"kind": "ready", "session_id": "synthetic", "sim_t": 0.0},
                    {"kind": "start_takeoff", "session_id": "synthetic",
                     "sim_t": 0.0, "authority": "AUTO"},
                ],
                summary={
                    "lifecycle_end": "FINISHED",
                    "phase_end": "TOUCHDOWN",
                    "authority_end": "AUTO",
                    "sim_t_end": 4.0,
                    "tick_count": 5,
                    "failure_reason": None,
                },
            )
            m = compute_metrics(sd)
            self.assertEqual(m.outcome, "success")
            self.assertAlmostEqual(m.max_altitude_m, 10.0, places=4)
            self.assertAlmostEqual(m.max_airspeed_m_s, 14.0, places=4)
            self.assertAlmostEqual(m.max_bank_deg, 1.0, places=4)
            self.assertAlmostEqual(m.max_pitch_deg, 10.0, places=4)
            self.assertAlmostEqual(m.min_keel_clearance_m, -0.05, places=4)
            self.assertEqual(m.tick_count, 5)
            self.assertEqual(m.inputs_attempted, 3)
            self.assertEqual(m.inputs_applied, 2)
            self.assertEqual(m.inputs_rejected, 1)

    def test_failure_classification(self):
        with tempfile.TemporaryDirectory() as td:
            sd = _make_session(
                Path(td),
                ticks=[
                    _tick(0.0, failed=False),
                    _tick(1.0, failed=True, water_kg=0.6),
                ],
                inputs=[], events=[],
                summary={
                    "lifecycle_end": "FINISHED",
                    "phase_end": "TAKEOFF",
                    "authority_end": "AUTO",
                    "sim_t_end": 1.0,
                    "tick_count": 2,
                    "failure_reason": "sling",
                },
            )
            m = compute_metrics(sd)
            self.assertEqual(m.outcome, "failure")
            self.assertEqual(m.failure_reason, "sling")
            self.assertTrue(m.damage_failed)

    def test_authority_time_share(self):
        with tempfile.TemporaryDirectory() as td:
            sd = _make_session(
                Path(td),
                ticks=[
                    _tick(0.0, authority="AUTO"),
                    _tick(1.0, authority="AUTO"),
                    _tick(2.0, authority="HUMAN"),
                    _tick(3.0, authority="HUMAN"),
                ],
                inputs=[], events=[],
                summary={
                    "lifecycle_end": "RUNNING", "phase_end": "CRUISE",
                    "authority_end": "HUMAN", "sim_t_end": 3.0,
                    "tick_count": 4,
                },
            )
            m = compute_metrics(sd)
            # Tick i covers the interval [sim_t[i-1], sim_t[i]].
            # Tick 0 is the start point (interval length 0).
            # Tick 1 is AUTO over [0, 1] -> AUTO = 1.0.
            # Tick 2 is HUMAN over [1, 2] -> HUMAN = 1.0.
            # Tick 3 is HUMAN over [2, 3] -> HUMAN = 1.0.
            self.assertAlmostEqual(m.time_auto_s, 1.0, places=4)
            self.assertAlmostEqual(m.time_human_s, 2.0, places=4)

    def test_jerk_only_measures_human_authority(self):
        with tempfile.TemporaryDirectory() as td:
            # AUTO guidance flips pitch sharply between ticks (huge
            # "jerk" if we counted it); HUMAN steps are smooth.
            sd = _make_session(
                Path(td),
                ticks=[
                    _tick(0.0, pitch_cmd=0.0, authority="AUTO"),
                    _tick(0.05, pitch_cmd=15.0, authority="AUTO"),
                    _tick(0.10, pitch_cmd=0.0, authority="AUTO"),
                    _tick(0.15, pitch_cmd=15.0, authority="AUTO"),
                    _tick(0.20, pitch_cmd=0.0, authority="HUMAN"),
                    _tick(0.25, pitch_cmd=1.0, authority="HUMAN"),
                    _tick(0.30, pitch_cmd=2.0, authority="HUMAN"),
                ],
                inputs=[], events=[],
                summary={
                    "lifecycle_end": "RUNNING", "phase_end": "TAKEOFF",
                    "authority_end": "HUMAN", "sim_t_end": 0.30,
                    "tick_count": 7,
                },
            )
            m = compute_metrics(sd)
            self.assertLess(m.pitch_jerk_rms, 200.0,
                            "jerk RMS should ignore AUTO ticks")


class MarkdownRenderTests(unittest.TestCase):
    def test_markdown_contains_key_sections(self):
        with tempfile.TemporaryDirectory() as td:
            sd = _make_session(
                Path(td),
                ticks=[
                    _tick(0.0, altitude=0.1, airspeed=0.5),
                    _tick(1.0, altitude=2.0, airspeed=8.0),
                ],
                inputs=[],
                events=[],
                summary={
                    "lifecycle_end": "FINISHED",
                    "phase_end": "TOUCHDOWN",
                    "authority_end": "AUTO",
                    "sim_t_end": 1.0,
                    "tick_count": 2,
                },
            )
            m = compute_metrics(sd)
            md = render_markdown(m, sd)
            self.assertIn("# Flight Report", md)
            self.assertIn("## Outcome", md)
            self.assertIn("## Aircraft performance", md)
            self.assertIn("## Authority & pilot inputs", md)
            self.assertIn("## Damage & integrity", md)
            self.assertIn("## Altitude profile", md)
            self.assertIn("SUCCESS", md)
            self.assertIn("`takeoff`", md)


class CsvExportTests(unittest.TestCase):
    def test_long_csv_has_uniform_columns(self):
        with tempfile.TemporaryDirectory() as td:
            sd = _make_session(
                Path(td),
                ticks=[
                    _tick(0.0),
                    _tick(0.05),
                    _tick(0.10),
                ],
                inputs=[], events=[],
                summary={
                    "lifecycle_end": "RUNNING", "phase_end": "TAKEOFF",
                    "authority_end": "AUTO", "sim_t_end": 0.10,
                    "tick_count": 3,
                },
            )
            csv_path = Path(td) / "series.csv"
            n = write_long_csv(sd, csv_path)
            self.assertEqual(n, 3)
            with csv_path.open() as f:
                lines = f.read().strip().split("\n")
            self.assertEqual(len(lines), 4)  # header + 3 rows
            header = lines[0].split(",")
            for line in lines[1:]:
                self.assertEqual(len(line.split(",")), len(header))


class CompareTests(unittest.TestCase):
    def test_compare_renders_table(self):
        with tempfile.TemporaryDirectory() as td:
            a = _make_session(
                Path(td) / "a",
                ticks=[_tick(0.0, altitude=1.0)],
                inputs=[],
                events=[],
                summary={"lifecycle_end": "FINISHED", "phase_end": "TOUCHDOWN",
                         "authority_end": "AUTO", "sim_t_end": 1.0,
                         "tick_count": 1},
                config={"session_id": "session-a", "scenario": "takeoff"},
            )
            b = _make_session(
                Path(td) / "b",
                ticks=[_tick(0.0, altitude=2.0)],
                inputs=[],
                events=[],
                summary={"lifecycle_end": "FINISHED", "phase_end": "TOUCHDOWN",
                         "authority_end": "AUTO", "sim_t_end": 1.0,
                         "tick_count": 1, "failure_reason": "sling"},
                config={"session_id": "session-b", "scenario": "landing"},
            )
            ma, mb = compute_metrics(a), compute_metrics(b)
            md = compare_metrics([ma, mb])
            self.assertIn("Session ID", md)
            self.assertIn("session-a", md)
            self.assertIn("session-b", md)
            self.assertIn("Metric", md)


if __name__ == "__main__":
    unittest.main()
