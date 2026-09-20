"""API contract types and JSON Schema generation.

These are the wire types exchanged over WebSocket and HTTP. They are
declared as dataclasses so the same definition feeds the validator,
the documentation generator, and the recorded log lines (every wire
message is JSON-compatible by construction).

Schema version is pinned to ``API_VERSION``. Any breaking change
bumps the major version; clients and sessions refuse mismatches.

Per design section 7.2 the messages exchanged are:

    input      - human 4-axis request with seq + epoch
    telemetry  - server snapshot of vehicle + session state
    event      - lifecycle / authority / stop reasons with event_id
    ack        - accept / reject decision for an event
    hello      - capability handshake from server at WS open
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from enum import Enum
from typing import Any, Iterable, Mapping


API_VERSION = 1
MAX_MESSAGE_BYTES = 16 * 1024
MAX_INPUT_HZ = 60


class MessageKind(str, Enum):
    HELLO = "hello"
    INPUT = "input"
    TELEMETRY = "telemetry"
    EVENT = "event"
    ACK = "ack"
    STOP = "stop"
    PING = "ping"
    PONG = "pong"


class EventKind(str, Enum):
    READY = "ready"
    START_TAKEOFF = "start_takeoff"
    START_MANUAL = "start_manual"
    OFFER_MANUAL = "offer_manual"
    HANDOVER_REQUESTED = "handover_requested"
    HANDOVER_DECLINED = "handover_declined"
    HANDOVER_ACCEPTED = "handover_accepted"
    AUTO_LAND_REQUESTED = "auto_land_requested"
    AUTO_LAND_DECLINED = "auto_land_declined"
    AUTO_LAND_REQUEST_CANCELLED = "auto_land_request_cancelled"
    AUTO_LAND_ACCEPTED = "auto_land_accepted"
    PAUSE = "pause"
    RESUME = "resume"
    ABORT = "abort"
    FINISH = "finish"
    INPUT_REJECTED = "input_rejected"
    TOUCHDOWN = "touchdown"


class DeclineReason(str, Enum):
    NOT_RUNNING = "not_running"
    NOT_OFFERED = "not_offered"
    ALREADY_HUMAN = "already_human"
    ALREADY_AUTO = "already_auto"
    INPUT_MISMATCH = "input_mismatch"
    GATE_FAILED = "gate_failed"


# ---------------------------------------------------------------------
# Wire types
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class ControlAxes:
    throttle: float
    pitch_deg: float
    bank_deg: float
    rudder: float

    def validate(self) -> None:
        for name, value in asdict(self).items():
            if not math.isfinite(value):
                raise SchemaError(f"{name} not finite: {value!r}")


@dataclass(frozen=True)
class InputMessage:
    """Client -> server human input."""

    v: int
    type: str
    session_id: str
    epoch: int
    seq: int
    control: ControlAxes
    receive_t: float = 0.0       # filled by server on arrival
    profile_hash: str = ""       # calibration profile identifier

    def to_wire(self) -> dict:
        d = {
            "v": self.v,
            "type": self.type,
            "session_id": self.session_id,
            "epoch": self.epoch,
            "seq": self.seq,
            "control": asdict(self.control),
            "profile_hash": self.profile_hash,
        }
        if self.receive_t:
            d["receive_t"] = self.receive_t
        return d


@dataclass(frozen=True)
class TelemetryMessage:
    """Server -> client periodic snapshot."""

    v: int
    type: str
    tick: int
    sim_time_s: float
    lifecycle: str
    phase: str
    authority: str
    pending: str
    last_input_seq: int
    applied_control: ControlAxes
    position_neu_m: dict
    velocity_neu_m_s: dict
    attitude: dict
    airspeed_m_s: float
    wave_clearance_m: float
    damage: dict
    warnings: list
    assist_active: bool = False

    def to_wire(self) -> dict:
        return {
            "v": self.v,
            "type": self.type,
            "tick": self.tick,
            "sim_time_s": self.sim_time_s,
            "lifecycle": self.lifecycle,
            "phase": self.phase,
            "authority": self.authority,
            "pending": self.pending,
            "assist_active": self.assist_active,
            "last_input_seq": self.last_input_seq,
            "applied_control": asdict(self.applied_control),
            "position_neu_m": dict(self.position_neu_m),
            "velocity_neu_m_s": dict(self.velocity_neu_m_s),
            "attitude": dict(self.attitude),
            "airspeed_m_s": self.airspeed_m_s,
            "wave_clearance_m": self.wave_clearance_m,
            "damage": dict(self.damage),
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class EventMessage:
    v: int
    type: str
    event_id: str
    kind: str
    sim_time_s: float
    session_id: str
    payload: dict = field(default_factory=dict)

    def to_wire(self) -> dict:
        return {
            "v": self.v,
            "type": self.type,
            "event_id": self.event_id,
            "kind": self.kind,
            "sim_time_s": self.sim_time_s,
            "session_id": self.session_id,
            "payload": dict(self.payload),
        }


@dataclass(frozen=True)
class AckMessage:
    v: int
    type: str
    event_id: str
    accepted: bool
    reason: str
    applied_tick: int
    session_id: str

    def to_wire(self) -> dict:
        return {
            "v": self.v,
            "type": self.type,
            "event_id": self.event_id,
            "accepted": self.accepted,
            "reason": self.reason,
            "applied_tick": self.applied_tick,
            "session_id": self.session_id,
        }


@dataclass(frozen=True)
class HelloMessage:
    """Server -> client at WS open."""

    v: int
    type: str
    session_id: str
    epoch: int
    control_envelope: dict
    curriculum: dict
    initial_telemetry: dict

    def to_wire(self) -> dict:
        return {
            "v": self.v,
            "type": self.type,
            "session_id": self.session_id,
            "epoch": self.epoch,
            "control_envelope": dict(self.control_envelope),
            "curriculum": dict(self.curriculum),
            "initial_telemetry": dict(self.initial_telemetry),
        }


class SchemaError(ValueError):
    pass


# ---------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------
def finite_float(value: Any, *, name: str) -> float:
    try:
        f = float(value)
    except (TypeError, ValueError) as exc:
        raise SchemaError(f"{name} is not numeric: {value!r}") from exc
    if not math.isfinite(f):
        raise SchemaError(f"{name} not finite: {value!r}")
    return f


def finite_int(value: Any, *, name: str) -> int:
    try:
        i = int(value)
    except (TypeError, ValueError) as exc:
        raise SchemaError(f"{name} is not integer: {value!r}") from exc
    return i


def parse_input_message(raw: Mapping[str, Any]) -> InputMessage:
    if not isinstance(raw, Mapping):
        raise SchemaError("input payload must be an object")
    if raw.get("v") != API_VERSION:
        raise SchemaError(f"unsupported version: {raw.get('v')!r}")
    if raw.get("type") != MessageKind.INPUT.value:
        raise SchemaError(f"wrong type for input: {raw.get('type')!r}")
    control_raw = raw.get("control")
    if not isinstance(control_raw, Mapping):
        raise SchemaError("control must be an object")
    control = ControlAxes(
        throttle=finite_float(control_raw.get("throttle"), name="throttle"),
        pitch_deg=finite_float(control_raw.get("pitch_deg"), name="pitch_deg"),
        bank_deg=finite_float(control_raw.get("bank_deg"), name="bank_deg"),
        rudder=finite_float(control_raw.get("rudder"), name="rudder"),
    )
    control.validate()
    return InputMessage(
        v=API_VERSION,
        type=MessageKind.INPUT.value,
        session_id=str(raw.get("session_id", "")),
        epoch=finite_int(raw.get("epoch", 0), name="epoch"),
        seq=finite_int(raw.get("seq", -1), name="seq"),
        control=control,
        receive_t=finite_float(raw.get("receive_t", 0.0), name="receive_t"),
        profile_hash=str(raw.get("profile_hash", "")),
    )


# ---------------------------------------------------------------------
# JSON Schema generation (lightweight, sufficient for documentation and
# contract tests).
# ---------------------------------------------------------------------
def _schema_for_dataclass(cls: type) -> dict:
    props = {}
    required = []
    for f in fields(cls):
        if f.metadata.get("wire_skip"):
            continue
        required.append(f.name)
        t = f.type
        if t in (int, "int"):
            props[f.name] = {"type": "integer"}
        elif t in (float, "float"):
            props[f.name] = {"type": "number"}
        elif t in (bool, "bool"):
            props[f.name] = {"type": "boolean"}
        elif t in (str, "str"):
            props[f.name] = {"type": "string"}
        elif t is dict or t == "dict" or t == "Dict":
            props[f.name] = {"type": "object"}
        elif t is list or t == "list" or t == "List":
            props[f.name] = {"type": "array"}
        else:
            props[f.name] = {"type": "object"}
        # Provide explicit ranges for the ControlAxes fields where the
        # server-side envelope is authoritative. The actual envelope is
        # broadcast via the hello message.
        if cls is ControlAxes:
            if f.name == "throttle":
                props[f.name]["minimum"] = 0.0
                props[f.name]["maximum"] = 1.0
            elif f.name == "pitch_deg":
                props[f.name]["minimum"] = -8.0
                props[f.name]["maximum"] = 15.0
            elif f.name == "bank_deg":
                props[f.name]["minimum"] = -45.0
                props[f.name]["maximum"] = 45.0
            elif f.name == "rudder":
                props[f.name]["minimum"] = -1.0
                props[f.name]["maximum"] = 1.0
    return {
        "type": "object",
        "required": required,
        "properties": props,
        "additionalProperties": False,
    }


def json_schema() -> dict:
    """Generate a JSON Schema bundle for the wire types."""
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "version": API_VERSION,
        "messages": {
            "input": _schema_for_dataclass(InputMessage),
            "control_axes": _schema_for_dataclass(ControlAxes),
            "telemetry": _schema_for_dataclass(TelemetryMessage),
            "event": _schema_for_dataclass(EventMessage),
            "ack": _schema_for_dataclass(AckMessage),
            "hello": _schema_for_dataclass(HelloMessage),
        },
        "enums": {
            "message_kind": [k.value for k in MessageKind],
            "event_kind": [k.value for k in EventKind],
            "decline_reason": [k.value for k in DeclineReason],
        },
        "limits": {
            "max_message_bytes": MAX_MESSAGE_BYTES,
            "max_input_hz": MAX_INPUT_HZ,
        },
    }


def wire_size(payload: Mapping[str, Any]) -> int:
    """Bytes a payload occupies when serialised as the canonical wire form."""
    return len(json.dumps(payload, allow_nan=False).encode("utf-8"))


def assert_wire_size(payload: Mapping[str, Any]) -> None:
    n = wire_size(payload)
    if n > MAX_MESSAGE_BYTES:
        raise SchemaError(
            f"message size {n}B exceeds {MAX_MESSAGE_BYTES}B"
        )
