"""Browser cockpit operator training runtime.

P0 scope (this package, headless):

    envelope     - control envelope definitions and validation
    recording    - JSONL writers for inputs / ticks / events
    curriculum   - phase definitions, approach gates, touchdown rules
    fake_pad     - deterministic input harness for tests
    vehicle_adapter - authority-aware wrapper over FlyingBoatVehicle
    session      - lifecycle, authority, handover logic
    runtime      - fixed-step tick scheduler with watchdog
    calibration  - gamepad calibration (deadzone, invert, response curve)
    schemas      - wire types and JSON Schema for the WS / HTTP contract
    replay       - deterministic replay comparator

Authority model: exactly one of AUTO or HUMAN per RUNNING tick.
ASSIST is a flag on HUMAN authority (lateral or pitch overridden by
controller while the operator retains primary responsibility).
"""
from .envelope import ControlEnvelope, ControlInput, EnvelopeViolation
from .recording import Recording, RecordingSink, NullSink
from .curriculum import (
    ApproachGate,
    Curriculum,
    CurriculumPhase,
    TouchdownDetector,
    default_approach_gate,
)
from .fake_pad import FakePad
from .vehicle_adapter import NavCommand, VehicleAdapter
from .session import (
    Authority,
    AutoLandDeclinedReason,
    HandoverDeclinedReason,
    Lifecycle,
    PendingRequest,
    Phase,
    Session,
    SessionConfig,
    SessionState,
)
from .runtime import FixedStepScheduler, SimOverrun, Watchdog
from .calibration import (
    AxisCalibration,
    CalibrationError,
    GamepadProfile,
    ResponseCurve,
    assert_profile_serialisable,
    default_profile,
    to_control_input,
)
from .schemas import (
    API_VERSION,
    MAX_INPUT_HZ,
    MAX_MESSAGE_BYTES,
    AckMessage,
    ControlAxes,
    DeclineReason,
    EventKind,
    EventMessage,
    HelloMessage,
    InputMessage,
    MessageKind,
    SchemaError,
    TelemetryMessage,
    assert_wire_size,
    json_schema,
    parse_input_message,
    wire_size,
)
from .replay import ReplayReport, load_config, replay_session
from .flight_report import (
    FlightMetrics,
    compute_metrics,
    compare_metrics,
    render_markdown,
    write_long_csv,
)
from .server import (
    OperatorServer,
    ServerConfig,
    ServerError,
    OriginNotAllowed,
    HostNotAllowed,
    SessionNotFound,
    SessionAlreadyTaken,
    SessionRegistry,
    capability_envelope,
    capabilities_payload,
)

__all__ = [
    "API_VERSION",
    "MAX_INPUT_HZ",
    "MAX_MESSAGE_BYTES",
    "AckMessage",
    "ApproachGate",
    "Assert",
    "Authority",
    "AutoLandDeclinedReason",
    "AxisCalibration",
    "CalibrationError",
    "ControlAxes",
    "ControlEnvelope",
    "ControlInput",
    "Curriculum",
    "CurriculumPhase",
    "DeclineReason",
    "EnvelopeViolation",
    "EventKind",
    "EventMessage",
    "FakePad",
    "FixedStepScheduler",
    "FlightMetrics",
    "GamepadProfile",
    "HelloMessage",
    "HandoverDeclinedReason",
    "InputMessage",
    "Lifecycle",
    "MessageKind",
    "NavCommand",
    "NullSink",
    "PendingRequest",
    "Phase",
    "Recording",
    "RecordingSink",
    "ReplayReport",
    "ResponseCurve",
    "SchemaError",
    "Session",
    "SessionConfig",
    "SessionState",
    "SimOverrun",
    "TelemetryMessage",
    "TouchdownDetector",
    "VehicleAdapter",
    "Watchdog",
    "assert_profile_serialisable",
    "assert_wire_size",
    "compute_metrics",
    "compare_metrics",
    "default_approach_gate",
    "default_profile",
    "json_schema",
    "load_config",
    "parse_input_message",
    "render_markdown",
    "replay_session",
    "to_control_input",
    "wire_size",
    "write_long_csv",
    "OperatorServer",
    "ServerConfig",
    "ServerError",
    "OriginNotAllowed",
    "HostNotAllowed",
    "SessionNotFound",
    "SessionAlreadyTaken",
    "SessionRegistry",
    "capability_envelope",
    "capabilities_payload",
]
