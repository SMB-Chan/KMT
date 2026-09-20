import json
import math
import unittest

from operator_training.schemas import (
    API_VERSION,
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


def _valid_input_dict(**overrides) -> dict:
    base = {
        "v": API_VERSION,
        "type": "input",
        "session_id": "abc",
        "epoch": 2,
        "seq": 7,
        "control": {
            "throttle": 0.5,
            "pitch_deg": 1.0,
            "bank_deg": -2.0,
            "rudder": 0.0,
        },
        "profile_hash": "",
    }
    base.update(overrides)
    return base


class InputParseTests(unittest.TestCase):
    def test_parse_valid(self):
        raw = _valid_input_dict()
        msg = parse_input_message(raw)
        self.assertEqual(msg.session_id, "abc")
        self.assertEqual(msg.seq, 7)
        self.assertEqual(msg.control.pitch_deg, 1.0)

    def test_reject_wrong_version(self):
        with self.assertRaises(SchemaError):
            parse_input_message(_valid_input_dict(v=2))

    def test_reject_wrong_type(self):
        with self.assertRaises(SchemaError):
            parse_input_message(_valid_input_dict(type="telemetry"))

    def test_reject_non_finite_control(self):
        bad = _valid_input_dict(control={
            "throttle": float("nan"), "pitch_deg": 0.0,
            "bank_deg": 0.0, "rudder": 0.0,
        })
        with self.assertRaises(SchemaError):
            parse_input_message(bad)

    def test_reject_missing_control_field(self):
        bad = _valid_input_dict(control={
            "throttle": 0.5, "pitch_deg": 0.0, "bank_deg": 0.0,
            # rudder missing
        })
        with self.assertRaises(SchemaError):
            parse_input_message(bad)

    def test_reject_non_object_payload(self):
        with self.assertRaises(SchemaError):
            parse_input_message("not an object")


class WireSizeTests(unittest.TestCase):
    def test_well_under_limit(self):
        raw = _valid_input_dict()
        size = wire_size(raw)
        self.assertLess(size, MAX_MESSAGE_BYTES)

    def test_size_limit_enforced(self):
        raw = _valid_input_dict()
        raw["control"]["throttle"] = 0.5
        # Inflate profile_hash to push past the limit.
        raw["profile_hash"] = "x" * (MAX_MESSAGE_BYTES + 1)
        with self.assertRaises(SchemaError):
            assert_wire_size(raw)


class JsonSchemaTests(unittest.TestCase):
    def test_schema_emitted(self):
        schema = json_schema()
        self.assertEqual(schema["version"], API_VERSION)
        self.assertIn("input", schema["messages"])
        self.assertIn("control_axes", schema["messages"])

    def test_schema_is_json_serialisable(self):
        schema = json_schema()
        s = json.dumps(schema)
        self.assertGreater(len(s), 100)


class MessageKindTests(unittest.TestCase):
    def test_event_kinds_listed(self):
        self.assertIn("offer_manual", [k.value for k in EventKind])
        self.assertIn("handover_accepted", [k.value for k in EventKind])
        self.assertIn("auto_land_accepted", [k.value for k in EventKind])
        self.assertIn("start_manual", [k.value for k in EventKind])

    def test_decline_reasons_listed(self):
        self.assertIn("input_mismatch", [k.value for k in DeclineReason])
        self.assertIn("gate_failed", [k.value for k in DeclineReason])


class WireRoundTripTests(unittest.TestCase):
    def test_input_to_wire_round_trip(self):
        msg = InputMessage(
            v=API_VERSION,
            type=MessageKind.INPUT.value,
            session_id="s",
            epoch=1,
            seq=2,
            control=ControlAxes(throttle=0.5, pitch_deg=1.0,
                                bank_deg=-1.0, rudder=0.0),
        )
        wire = msg.to_wire()
        self.assertEqual(wire["type"], "input")
        self.assertEqual(wire["control"]["throttle"], 0.5)

    def test_event_to_wire_includes_payload(self):
        msg = EventMessage(
            v=API_VERSION, type=MessageKind.EVENT.value,
            event_id="evt-1", kind=EventKind.HANDOVER_ACCEPTED.value,
            sim_time_s=1.5, session_id="s",
            payload={"tolerance_s": 0.5},
        )
        wire = msg.to_wire()
        self.assertEqual(wire["kind"], "handover_accepted")
        self.assertEqual(wire["payload"]["tolerance_s"], 0.5)

    def test_ack_to_wire(self):
        msg = AckMessage(
            v=API_VERSION, type=MessageKind.ACK.value,
            event_id="evt-2", accepted=False,
            reason=DeclineReason.INPUT_MISMATCH.value,
            applied_tick=0, session_id="s",
        )
        wire = msg.to_wire()
        self.assertFalse(wire["accepted"])
        self.assertEqual(wire["reason"], "input_mismatch")


if __name__ == "__main__":
    unittest.main()
