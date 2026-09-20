"""JSONL recording with bounded buffering.

Per design document section 9.3:

    inputs.jsonl    - received seq, monotonic receive time, calibrated request
    ticks.jsonl     - applied seq, applied control, normalized state, damage
    events.jsonl    - authority transitions, stop reasons, rejections, comms
    config.json     - curriculum version, source SHA, envelope, thresholds
    summary.json    - outcomes, end reason, validation scope

The writer is bounded to absorb bursts without dropping data; when the
buffer is full it blocks the caller rather than silently dropping rows.
This matches the "ログの欠落は黙認しない" principle from section 7.2.

summary.json is written via a tempfile + rename so a crash mid-write
leaves either the old summary or no summary, never a torn file.
"""
from __future__ import annotations

import json
import os
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Iterable, Optional, Protocol


class RecordingSink(Protocol):
    """Abstract destination for jsonl rows. Implementations: file, null."""

    def open(self) -> None: ...
    def write_row(self, stream: str, row: dict) -> None: ...
    def write_json(self, name: str, payload: dict) -> None: ...
    def close(self) -> None: ...


class NullSink:
    """Discards every row. Useful for in-memory tests."""

    def open(self) -> None:
        return None

    def write_row(self, stream: str, row: dict) -> None:
        return None

    def write_json(self, name: str, payload: dict) -> None:
        return None

    def close(self) -> None:
        return None


@dataclass
class FileSink:
    """Buffered JSONL sink with bounded queue and per-stream files."""

    root: Path
    queue_capacity: int = 4096

    _root: Optional[Path] = field(default=None, init=False, repr=False)
    _handles: dict = field(default_factory=dict, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def open(self) -> None:
        self._root = Path(self.root)
        self._root.mkdir(parents=True, exist_ok=True)

    def _path(self, stream: str) -> Path:
        assert self._root is not None
        return self._root / f"{stream}.jsonl"

    def _handle(self, stream: str) -> IO[str]:
        if stream not in self._handles:
            path = self._path(stream)
            path.parent.mkdir(parents=True, exist_ok=True)
            self._handles[stream] = path.open("a", encoding="utf-8")
        return self._handles[stream]

    def write_row(self, stream: str, row: dict) -> None:
        with self._lock:
            handle = self._handle(stream)
            handle.write(json.dumps(row, allow_nan=False, sort_keys=True))
            handle.write("\n")
            handle.flush()

    def write_json(self, name: str, payload: dict) -> None:
        assert self._root is not None
        path = self._root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic write: temp file in the same directory, then os.replace.
        fd, tmp_path = tempfile.mkstemp(prefix=".tmp_", dir=str(path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, allow_nan=False, sort_keys=True, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_path, path)
        except Exception:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise

    def close(self) -> None:
        with self._lock:
            for handle in self._handles.values():
                handle.close()
            self._handles.clear()


@dataclass
class Recording:
    """High-level recorder that frames the contract from section 9.3."""

    session_id: str
    sink: RecordingSink
    config: dict

    _sink_open: bool = field(default=False, init=False, repr=False)

    def open(self) -> None:
        if self._sink_open:
            return
        self.sink.open()
        self._sink_open = True
        self.sink.write_json("config.json", self._envelope_config())

    def _envelope_config(self) -> dict:
        envelope = self.config.get("control_envelope", {})
        curriculum = self.config.get("curriculum") or {}
        approach_gate = curriculum.get("approach_gate") or {}
        handover = curriculum.get("handover") or {}
        return {
            "session_id": self.session_id,
            "control_envelope": envelope,
            "curriculum_id": self.config.get("curriculum_id"),
            "curriculum_version": self.config.get("curriculum_version"),
            "scenario": self.config.get("scenario"),
            "wave": self.config.get("wave"),
            "wind_m_s": self.config.get("wind_m_s"),
            "seed": self.config.get("seed"),
            "spatial": self.config.get("spatial", True),
            "curriculum": {
                "handover": handover,
                "approach_gate": approach_gate,
            },
        }

    def record_input(self, *, seq: int, receive_t: float,
                     applied: bool, reason: str,
                     raw: dict, request: Optional[dict]) -> None:
        self.sink.write_row("inputs", {
            "kind": "input",
            "session_id": self.session_id,
            "seq": int(seq),
            "receive_t": float(receive_t),
            "applied": bool(applied),
            "reason": reason,
            "raw": raw,
            "request": request,
        })

    def record_tick(self, *, tick: int, sim_t: float,
                    seq: Optional[int], authority: str,
                    assist: bool, control: dict, state: dict,
                    damage: dict) -> None:
        self.sink.write_row("ticks", {
            "kind": "tick",
            "session_id": self.session_id,
            "tick": int(tick),
            "sim_t": float(sim_t),
            "applied_seq": None if seq is None else int(seq),
            "authority": authority,
            "assist": bool(assist),
            "control": control,
            "state": state,
            "damage": damage,
        })

    def record_event(self, *, sim_t: float, kind: str, **payload) -> None:
        row = {
            "kind": kind,
            "session_id": self.session_id,
            "sim_t": float(sim_t),
        }
        row.update(payload)
        self.sink.write_row("events", row)

    def finalize(self, *, summary: dict) -> None:
        self.sink.write_json("summary.json", summary)
        self.sink.close()
        self._sink_open = False
