"""Deterministic replay of a recorded session.

Reads ``config.json``, ``inputs.jsonl`` and ``events.jsonl`` from a
session directory and replays the tick sequence in a fresh Session. The
resulting ``ticks.jsonl`` is compared field-by-field with the recorded
``ticks.jsonl`` to verify bit-equal (within a configurable tolerance)
replay.

This is the acceptance test for design section 9.3 "保存状態再生と元
ログの指標一致". Without replay equality the recorded results cannot be
trusted as evidence of operator performance.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Optional

from .envelope import ControlEnvelope, ControlInput
from .recording import FileSink
from .schemas import API_VERSION
from .session import (
    Authority,
    Curriculum,
    Lifecycle,
    PendingRequest,
    Session,
    SessionConfig,
)


DEFAULT_TOLERANCES = {
    "altitude_m": 1e-6,
    "forward_speed_m_s": 1e-6,
    "vertical_speed_m_s": 1e-6,
    "lateral_position_m": 1e-6,
    "lateral_speed_m_s": 1e-6,
    "bank_deg": 1e-6,
    "heading_deg": 1e-6,
    "airspeed_m_s": 1e-6,
    "keel_clearance_m": 1e-6,
    "wave_elevation_m": 1e-6,
}


@dataclass
class ReplayReport:
    session_id: str
    ticks_compared: int
    mismatches: list = field(default_factory=list)
    inputs_replayed: int = 0
    end_reason_match: bool = True

    @property
    def passed(self) -> bool:
        return not self.mismatches and self.end_reason_match

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "ticks_compared": self.ticks_compared,
            "inputs_replayed": self.inputs_replayed,
            "end_reason_match": self.end_reason_match,
            "mismatches": self.mismatches,
            "passed": self.passed,
        }


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_config(root: Path) -> dict:
    """Load a previously written config.json."""
    return _read_json(root / "config.json")


def replay_session(*, source_root: Path, target_root: Path,
                   sink=None) -> tuple[Session, ReplayReport]:
    """Replay the inputs in source_root into a fresh session under target_root.

    The replay session uses the same seed, envelope, curriculum, and
    physics instance layout as the original. Returns the new Session
    (with its own recording under target_root) and the comparison report.
    """
    cfg = load_config(source_root)
    envelope_dict = cfg["control_envelope"]
    envelope = ControlEnvelope(
        throttle_lo=float(envelope_dict["throttle_lo"]),
        throttle_hi=float(envelope_dict["throttle_hi"]),
        pitch_lo=float(envelope_dict["pitch_lo"]),
        pitch_hi=float(envelope_dict["pitch_hi"]),
        bank_abs=float(envelope_dict["bank_abs"]),
        rudder_abs=float(envelope_dict["rudder_abs"]),
    )
    curriculum_dict = cfg.get("curriculum") or {}
    handover = curriculum_dict.get("handover") or {}
    approach_gate_dict = curriculum_dict.get("approach_gate") or {}
    from .curriculum import ApproachGate
    approach_gate = ApproachGate(
        altitude_lo_m=float(approach_gate_dict.get("altitude_lo_m", 15.0)),
        altitude_hi_m=float(approach_gate_dict.get("altitude_hi_m", 60.0)),
        airspeed_lo_m_s=float(approach_gate_dict.get("airspeed_lo_m_s", 10.0)),
        airspeed_hi_m_s=float(approach_gate_dict.get("airspeed_hi_m_s", 18.0)),
        sink_rate_max_m_s=float(approach_gate_dict.get("sink_rate_max_m_s", 4.0)),
        bank_abs_deg=float(approach_gate_dict.get("bank_abs_deg", 12.0)),
        lateral_abs_m=float(approach_gate_dict.get("lateral_abs_m", 15.0)),
        lateral_speed_abs_m_s=float(approach_gate_dict.get("lateral_speed_abs_m_s", 2.5)),
    )
    airspeed_value = handover.get("airspeed_m_s")
    curriculum = Curriculum(
        curriculum_id=cfg["curriculum_id"],
        curriculum_version=cfg["curriculum_version"],
        scenario=cfg["scenario"],
        handover_altitude_m=float(handover.get("altitude_m", 8.0)),
        handover_airspeed_m_s=(
            float(airspeed_value) if airspeed_value is not None else None
        ),
        handover_vertical_speed_m_s=float(handover.get("vertical_speed_m_s", 2.0)),
        handover_bank_deg=float(handover.get("bank_deg", 10.0)),
        handover_duration_s=float(handover.get("duration_s", 1.0)),
        approach_gate=approach_gate,
    )
    seed = int(cfg.get("seed", 42))
    session_id = cfg.get("session_id") or "replay"
    config = SessionConfig(
        session_id=session_id,
        curriculum=curriculum,
        envelope=envelope,
        seed=seed,
    )
    sink = sink or FileSink(root=target_root)
    sess = Session.create(config=config, sink=sink, seed=seed)
    return sess, _drive_replay(sess, source_root=source_root)


def _drive_replay(sess: Session, *, source_root: Path) -> ReplayReport:
    inputs = _read_jsonl(source_root / "inputs.jsonl")
    expected_ticks = _read_jsonl(source_root / "ticks.jsonl")
    expected_summary = _read_json(source_root / "summary.json")
    report = ReplayReport(
        session_id=sess.config.session_id,
        ticks_compared=0,
    )

    sess.ready()
    sess.start_takeoff()

    # Index expected ticks by tick number for sequential comparison.
    expected_by_tick = {int(row["tick"]): row for row in expected_ticks}

    # Index inputs by seq for O(1) lookup when replaying tick-by-tick.
    seq_to_input: dict = {}
    for row in inputs:
        seq = int(row.get("seq", -1))
        if seq >= 0:
            seq_to_input[seq] = row

    expected_tick_numbers = sorted(expected_by_tick.keys())
    last_seen_seq = -1

    for tick_no in expected_tick_numbers:
        expected_row = expected_by_tick[tick_no]
        applied_seq = expected_row.get("applied_seq")
        # Replay exactly one tick. If the original applied an input at
        # this tick, replay that input; otherwise step with no input.
        human_input = None
        raw_input = None
        chosen_seq = -1
        if applied_seq is not None and int(applied_seq) >= 0:
            chosen_seq = int(applied_seq)
            in_row = seq_to_input.get(chosen_seq)
            if in_row is not None:
                raw = in_row.get("request") or in_row.get("raw")
                if isinstance(raw, dict):
                    # Replay: do NOT envelope-validate. The original
                    # session applied whatever the operator requested
                    # (the envelope is enforced upstream at the
                    # transport layer, not during replay). Mismatches
                    # in the envelope range would otherwise mask the
                    # intended input.
                    try:
                        from .envelope import ControlInput
                        human_input = ControlInput(
                            throttle=float(raw.get("throttle", 0.0)),
                            pitch_deg=float(raw.get("pitch_deg", 0.0)),
                            bank_deg=float(raw.get("bank_deg", 0.0)),
                            rudder=float(raw.get("rudder", 0.0)),
                        )
                    except (TypeError, ValueError):
                        human_input = None
                    raw_input = raw
                    report.inputs_replayed += 1
        last_seen_seq = max(last_seen_seq, chosen_seq)
        sess.tick(dt=0.05, human_input=human_input,
                  input_seq=chosen_seq, raw_input=raw_input)

        # Read the just-written tick row from disk and compare.
        if sess.recording.sink is not None and hasattr(sess.recording.sink, "_root"):
            tick_file = sess.recording.sink._root / "ticks.jsonl"
            if tick_file.exists():
                rows = _read_jsonl(tick_file)
                if rows:
                    actual_row = rows[-1]
                    _compare_tick(actual_row, expected_row, report)

        if sess.state.lifecycle in (Lifecycle.FINISHED, Lifecycle.ABORTED):
            break

    # End reason comparison.
    if expected_summary:
        actual_summary_path = (
            (sess.recording.sink._root / "summary.json")
            if hasattr(sess.recording.sink, "_root") else None
        )
        if actual_summary_path and actual_summary_path.exists():
            actual_summary = _read_json(actual_summary_path)
            report.end_reason_match = (
                actual_summary.get("end_reason") == expected_summary.get("end_reason")
            )
    return report


def _compare_tick(actual: dict, expected: dict, report: ReplayReport) -> None:
    report.ticks_compared += 1
    for key, tol in DEFAULT_TOLERANCES.items():
        a = actual.get("state", {}).get(key)
        e = expected.get("state", {}).get(key)
        if a is None or e is None:
            continue
        if not math.isfinite(a) or not math.isfinite(e):
            continue
        if abs(a - e) > tol:
            report.mismatches.append({
                "tick": int(actual["tick"]),
                "field": f"state.{key}",
                "actual": a,
                "expected": e,
                "tolerance": tol,
            })
    # Authority and assist must match exactly.
    if actual.get("authority") != expected.get("authority"):
        report.mismatches.append({
            "tick": int(actual["tick"]),
            "field": "authority",
            "actual": actual.get("authority"),
            "expected": expected.get("authority"),
        })
    if actual.get("assist") != expected.get("assist"):
        report.mismatches.append({
            "tick": int(actual["tick"]),
            "field": "assist",
            "actual": actual.get("assist"),
            "expected": expected.get("assist"),
        })
