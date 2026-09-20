"""Tests for the asyncio HTTP + WebSocket server."""
import asyncio
import json
import os
import socket
import tempfile
import time
import unittest
from pathlib import Path

from operator_training.envelope import (
    BANK_BEGINNER,
    ControlEnvelope,
    PITCH_BEGINNER_HI,
)
from operator_training.curriculum import Curriculum
from operator_training.server import (
    OperatorServer,
    ServerConfig,
    SessionAlreadyTaken,
    SessionNotFound,
)
from operator_training.session import (
    Authority,
    PendingRequest,
    SessionConfig,
)
from operator_training.ws_client import WSClient, http_get, http_post
from operator_training.ws_frames import (
    OP_TEXT,
    encode_frame,
    encode_text,
    FrameDecoder,
)


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


async def _wait_for_event(ws, kind: str, *, event_id: str | None = None,
                          timeout: float = 2.0):
    """Drain messages until one of the given type / event_id arrives."""
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        try:
            remaining = max(0.05, deadline - asyncio.get_event_loop().time())
            msg = await ws.recv_message(timeout=remaining)
        except Exception:
            return None
        if msg is None:
            return None
        try:
            obj = json.loads(msg)
        except json.JSONDecodeError:
            continue
        if obj.get("type") != kind:
            continue
        if event_id is not None and obj.get("event_id") != event_id:
            continue
        return obj
    return None


class WSFrameRoundTripTests(unittest.TestCase):
    def test_encode_decode_small(self):
        payload = b"hello"
        encoded = encode_text("hello")
        dec = FrameDecoder()
        dec.feed(encoded)
        frame = dec.pop()
        self.assertEqual(frame.opcode, OP_TEXT)
        self.assertEqual(frame.payload, payload)

    def test_decode_split_chunks(self):
        encoded = encode_text("x" * 200)
        dec = FrameDecoder()
        dec.feed(encoded[:50])
        self.assertIsNone(dec.pop())
        dec.feed(encoded[50:])
        frame = dec.pop()
        self.assertEqual(frame.payload, b"x" * 200)

    def test_oversized_payload_rejected(self):
        dec = FrameDecoder(max_text_payload=10)
        encoded = encode_text("x" * 32)
        with self.assertRaises(ValueError):
            dec.feed(encoded)
            dec.pop()


class _ServerFixture:
    """Spin up an OperatorServer on a free port for the test."""

    def __init__(self, recordings_root: Path | None = None):
        self.server: OperatorServer | None = None
        self.port: int = 0
        self.recordings_root = recordings_root

    async def __aenter__(self):
        cfg = ServerConfig(
            allowed_origins=("http://localhost", "http://127.0.0.1"),
            allowed_hosts=("localhost", "127.0.0.1"),
            recordings_root=self.recordings_root,
            require_csrf=False,
            tick_hz=20.0,
            heartbeat_s=10.0,
            input_max_age_s=0.25,
        )
        self.server = OperatorServer(
            config=cfg,
            envelope=ControlEnvelope.beginner(),
            curriculum=Curriculum(
                curriculum_id="hybrid_baseline",
                curriculum_version="1",
                scenario="hybrid",
                handover_altitude_m=4.0,
                handover_airspeed_m_s=8.0,
                handover_duration_s=0.5,
            ),
        )
        self.port = _free_port()
        await self.server.start(host="127.0.0.1", port=self.port)
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if self.server is not None:
            await self.server.stop()


class HttpCapabilityTests(unittest.TestCase):
    def test_capabilities_shape(self):
        async def run():
            async with _ServerFixture() as fix:
                status, headers, body = await http_get(
                    "127.0.0.1", fix.port, "/api/capabilities",
                )
                self.assertTrue(status.startswith("HTTP/1.1 200"))
                payload = json.loads(body.decode("utf-8"))
                self.assertEqual(payload["v"], 1)
                self.assertIn("control_envelope", payload)
                self.assertEqual(payload["control_envelope"]["pitch_hi"],
                                 PITCH_BEGINNER_HI)
                self.assertEqual(payload["control_envelope"]["bank_abs"],
                                 BANK_BEGINNER)
                self.assertEqual(payload["limits"]["max_input_hz"], 60)
                self.assertIn("schemas", payload)
        asyncio.run(run())

    def test_health_endpoint(self):
        async def run():
            async with _ServerFixture() as fix:
                status, _, body = await http_get("127.0.0.1", fix.port, "/health")
                self.assertTrue(status.startswith("HTTP/1.1 200"))
                self.assertIn(b"ok", body)
        asyncio.run(run())

    def test_unknown_path_returns_404(self):
        async def run():
            async with _ServerFixture() as fix:
                status, _, _ = await http_get("127.0.0.1", fix.port, "/no/such/path")
                self.assertTrue(status.startswith("HTTP/1.1 404"))
        asyncio.run(run())


class HttpSessionTests(unittest.TestCase):
    def test_create_session_returns_id(self):
        async def run():
            with tempfile.TemporaryDirectory() as td:
                async with _ServerFixture(recordings_root=Path(td)) as fix:
                    body = json.dumps({
                        "seed": 42, "scenario": "hybrid",
                    }).encode("utf-8")
                    status, headers, resp = await http_post(
                        "127.0.0.1", fix.port, "/api/sessions", body,
                    )
                    self.assertTrue(status.startswith("HTTP/1.1 201"),
                                    status)
                    payload = json.loads(resp.decode("utf-8"))
                    self.assertIn("session_id", payload)
                    self.assertEqual(payload["epoch"], 0)
                    self.assertEqual(payload["control_envelope"]["pitch_hi"],
                                     PITCH_BEGINNER_HI)
        asyncio.run(run())

    def test_create_session_with_invalid_json(self):
        async def run():
            async with _ServerFixture() as fix:
                status, _, _ = await http_post(
                    "127.0.0.1", fix.port, "/api/sessions",
                    b"not json",
                )
                self.assertTrue(status.startswith("HTTP/1.1 400"))
        asyncio.run(run())

    def test_summary_for_missing_session(self):
        async def run():
            async with _ServerFixture() as fix:
                status, _, _ = await http_get(
                    "127.0.0.1", fix.port, "/api/sessions/no-such/summary",
                )
                self.assertTrue(status.startswith("HTTP/1.1 404"))
        asyncio.run(run())


class OriginAndHostTests(unittest.TestCase):
    def test_disallowed_origin_rejected(self):
        async def run():
            async with _ServerFixture() as fix:
                status, _, body = await http_get(
                    "127.0.0.1", fix.port, "/api/capabilities",
                    headers={"Origin": "http://evil.example.com"},
                )
                self.assertTrue(status.startswith("HTTP/1.1 403"), status)
                self.assertIn(b"origin", body)
        asyncio.run(run())

    def test_disallowed_host_rejected(self):
        async def run():
            cfg = ServerConfig(
                allowed_origins=("http://localhost", "http://127.0.0.1"),
                allowed_hosts=("only-this-host.example",),
            )
            server = OperatorServer(
                config=cfg, envelope=ControlEnvelope.beginner(),
                curriculum=Curriculum(
                    curriculum_id="x", curriculum_version="1", scenario="hybrid",
                ),
            )
            port = _free_port()
            await server.start(host="127.0.0.1", port=port)
            try:
                status, _, _ = await http_get(
                    "127.0.0.1", port, "/api/capabilities",
                )
                self.assertTrue(status.startswith("HTTP/1.1 403"))
            finally:
                await server.stop()
        asyncio.run(run())


class WSRoundTripTests(unittest.TestCase):
    def test_hello_then_telemetry(self):
        async def run():
            with tempfile.TemporaryDirectory() as td:
                async with _ServerFixture(recordings_root=Path(td)) as fix:
                    # Create session
                    body = json.dumps({"seed": 42, "scenario": "hybrid"}).encode()
                    _, _, resp = await http_post(
                        "127.0.0.1", fix.port, "/api/sessions", body,
                    )
                    session_id = json.loads(resp.decode())["session_id"]
                    async with WSClient(host="127.0.0.1", port=fix.port,
                                        path=f"/ws/sessions/{session_id}",
                                        origin="http://127.0.0.1") as ws:
                        hello = json.loads(await ws.recv_message())
                        self.assertEqual(hello["type"], "hello")
                        self.assertEqual(hello["session_id"], session_id)
                        self.assertEqual(hello["epoch"], 1)
                        self.assertIn("control_envelope", hello)
                        # The session auto-starts on first connect
                        # is NOT done by default; verify a telemetry
                        # message arrives after start_takeoff event.
                        await ws.send_text(json.dumps({
                            "v": 1, "type": "event", "event_id": "e1",
                            "kind": "start_takeoff",
                            "session_id": session_id,
                            "epoch": 1, "sim_time_s": 0.0,
                            "payload": {},
                        }))
                        ack = json.loads(await ws.recv_message())
                        self.assertEqual(ack["type"], "ack")
                        self.assertTrue(ack["accepted"])
                        # First telemetry tick
                        tel = json.loads(await ws.recv_message())
                        self.assertEqual(tel["type"], "telemetry")
                        self.assertEqual(tel["authority"], "AUTO")
                        self.assertEqual(tel["phase"], "TAKEOFF")
        asyncio.run(run())

    def test_duplicate_connection_rejected(self):
        async def run():
            with tempfile.TemporaryDirectory() as td:
                async with _ServerFixture(recordings_root=Path(td)) as fix:
                    body = json.dumps({"seed": 42}).encode()
                    _, _, resp = await http_post(
                        "127.0.0.1", fix.port, "/api/sessions", body,
                    )
                    session_id = json.loads(resp.decode())["session_id"]
                    async with WSClient(host="127.0.0.1", port=fix.port,
                                        path=f"/ws/sessions/{session_id}",
                                        origin="http://127.0.0.1") as ws1:
                        hello1 = json.loads(await ws1.recv_message())
                        self.assertEqual(hello1["epoch"], 1)
                        async with WSClient(host="127.0.0.1", port=fix.port,
                                            path=f"/ws/sessions/{session_id}",
                                            origin="http://127.0.0.1") as ws2:
                            # Second connection should bump epoch and
                            # close the first one with a close frame.
                            hello2 = json.loads(await ws2.recv_message())
                            self.assertEqual(hello2["epoch"], 2)
                            # ws1 may receive a close frame or just EOF.
                            try:
                                tail = await ws1.recv_message(timeout=2.0)
                            except Exception:
                                tail = None
                            self.assertIsNone(tail)
        asyncio.run(run())

    def test_envelope_violation_rejected(self):
        async def run():
            with tempfile.TemporaryDirectory() as td:
                async with _ServerFixture(recordings_root=Path(td)) as fix:
                    body = json.dumps({"seed": 42}).encode()
                    _, _, resp = await http_post(
                        "127.0.0.1", fix.port, "/api/sessions", body,
                    )
                    session_id = json.loads(resp.decode())["session_id"]
                    async with WSClient(host="127.0.0.1", port=fix.port,
                                        path=f"/ws/sessions/{session_id}",
                                        origin="http://127.0.0.1") as ws:
                        await ws.recv_message()  # hello
                        # Send an input that violates the envelope
                        # (pitch > 12 deg).
                        await ws.send_text(json.dumps({
                            "v": 1, "type": "input",
                            "session_id": session_id,
                            "epoch": 1, "seq": 1,
                            "control": {
                                "throttle": 0.5,
                                "pitch_deg": 25.0,
                                "bank_deg": 0.0,
                                "rudder": 0.0,
                            },
                        }))
                        ack = json.loads(await ws.recv_message())
                        self.assertEqual(ack["type"], "ack")
                        self.assertFalse(ack["accepted"])
                        self.assertIn("envelope", ack["reason"])
        asyncio.run(run())

    def test_stale_epoch_rejected(self):
        async def run():
            with tempfile.TemporaryDirectory() as td:
                async with _ServerFixture(recordings_root=Path(td)) as fix:
                    body = json.dumps({"seed": 42}).encode()
                    _, _, resp = await http_post(
                        "127.0.0.1", fix.port, "/api/sessions", body,
                    )
                    session_id = json.loads(resp.decode())["session_id"]
                    async with WSClient(host="127.0.0.1", port=fix.port,
                                        path=f"/ws/sessions/{session_id}",
                                        origin="http://127.0.0.1") as ws:
                        hello = json.loads(await ws.recv_message())
                        self.assertEqual(hello["epoch"], 1)
                        # Send an input with epoch=0 (stale)
                        await ws.send_text(json.dumps({
                            "v": 1, "type": "input",
                            "session_id": session_id,
                            "epoch": 0, "seq": 1,
                            "control": {
                                "throttle": 0.5, "pitch_deg": 0.0,
                                "bank_deg": 0.0, "rudder": 0.0,
                            },
                        }))
                        ack = json.loads(await ws.recv_message())
                        self.assertFalse(ack["accepted"])
                        self.assertEqual(ack["reason"], "stale epoch")
        asyncio.run(run())

    def test_input_rate_limit(self):
        async def run():
            with tempfile.TemporaryDirectory() as td:
                async with _ServerFixture(recordings_root=Path(td)) as fix:
                    body = json.dumps({"seed": 42}).encode()
                    _, _, resp = await http_post(
                        "127.0.0.1", fix.port, "/api/sessions", body,
                    )
                    session_id = json.loads(resp.decode())["session_id"]
                    async with WSClient(host="127.0.0.1", port=fix.port,
                                        path=f"/ws/sessions/{session_id}",
                                        origin="http://127.0.0.1") as ws:
                        await ws.recv_message()  # hello
                        # Fire 80 inputs as fast as we can. We expect
                        # ACK rate-limit rejections once we exceed 60.
                        sent = 0
                        rejected = 0
                        for i in range(80):
                            await ws.send_text(json.dumps({
                                "v": 1, "type": "input",
                                "session_id": session_id,
                                "epoch": 1, "seq": i + 1,
                                "control": {
                                    "throttle": 0.5, "pitch_deg": 0.0,
                                    "bank_deg": 0.0, "rudder": 0.0,
                                },
                            }))
                            sent += 1
                        # Drain ACKs / telemetry for ~2s.
                        deadline = time.monotonic() + 2.0
                        while time.monotonic() < deadline:
                            msg = await ws.recv_message(timeout=0.2)
                            if msg is None:
                                break
                            obj = json.loads(msg)
                            if obj.get("type") == "ack":
                                if not obj.get("accepted"):
                                    rejected += 1
                        self.assertGreater(rejected, 0,
                                           "expected some rate-limit rejections")
        asyncio.run(run())

    def test_handover_request_event(self):
        async def run():
            with tempfile.TemporaryDirectory() as td:
                async with _ServerFixture(recordings_root=Path(td)) as fix:
                    body = json.dumps({"seed": 42}).encode()
                    _, _, resp = await http_post(
                        "127.0.0.1", fix.port, "/api/sessions", body,
                    )
                    session_id = json.loads(resp.decode())["session_id"]
                    async with WSClient(host="127.0.0.1", port=fix.port,
                                        path=f"/ws/sessions/{session_id}",
                                        origin="http://127.0.0.1") as ws:
                        hello = json.loads(await ws.recv_message())
                        self.assertEqual(hello["type"], "hello")
                        rec = fix.server.registry.get(session_id)
                        rec.session.start_takeoff()
                        rec.session.state.pending = "OFFER_MANUAL"
                        rec.session.state.match_elapsed_s = 0.0
                        rec.session.state.offer_manual_elapsed_s = 0.5
                        # Send take_control event
                        await ws.send_text(json.dumps({
                            "v": 1, "type": "event",
                            "event_id": "tc1",
                            "kind": "take_control",
                            "session_id": session_id,
                            "epoch": hello["epoch"], "sim_time_s": 0.0,
                            "payload": {},
                        }))
                        ack = await _wait_for_event(ws, "ack", event_id="tc1")
                        self.assertTrue(ack["accepted"], f"ack={ack}")
                        # Send a matching input - session is now in
                        # OFFER_MANUAL pending, so eval_handover will
                        # run on each subsequent tick.
                        for seq in range(1, 30):
                            auto = rec.session._capture_auto_setpoints()
                            from operator_training.envelope import ControlInput
                            ci = ControlInput(auto["throttle"], auto["pitch_deg"],
                                              auto["bank_deg"], auto["rudder"])
                            await ws.send_text(json.dumps({
                                "v": 1, "type": "input",
                                "session_id": session_id,
                                "epoch": hello["epoch"], "seq": seq,
                                "control": {
                                    "throttle": ci.throttle,
                                    "pitch_deg": ci.pitch_deg,
                                    "bank_deg": ci.bank_deg,
                                    "rudder": ci.rudder,
                                },
                            }))
                            await asyncio.sleep(0.05)
                            # Look for telemetry with HUMAN authority
                            tel = await _wait_for_event(
                                ws, "telemetry", timeout=0.5,
                            )
                            if tel and tel.get("authority") == "HUMAN":
                                return
                        self.fail("handover did not complete in time")
        asyncio.run(run())


class SessionRegistryTests(unittest.TestCase):
    def test_get_missing_raises(self):
        from operator_training.server import SessionRegistry
        reg = SessionRegistry()
        with self.assertRaises(SessionNotFound):
            reg.get("nope")


class WavePatchStreamingTests(unittest.TestCase):
    def test_wave_patch_broadcast(self):
        import math as _math
        from operator_training.wave_patch import (
            WAVE_PATCH_VERSION,
            decode_wave_patch,
            decode_wave_patch_header,
        )
        async def run():
            with tempfile.TemporaryDirectory() as td:
                async with _ServerFixture(recordings_root=Path(td)) as fix:
                    # Reduce the wave patch spec to keep the wire small.
                    fix.server.config.wave_patch_size = 9
                    fix.server.config.wave_patch_extent_m = 16.0
                    fix.server.config.wave_patch_hz = 50.0
                    body = json.dumps({"seed": 42}).encode()
                    _, _, resp = await http_post(
                        "127.0.0.1", fix.port, "/api/sessions", body,
                    )
                    session_id = json.loads(resp.decode())["session_id"]
                    async with WSClient(host="127.0.0.1", port=fix.port,
                                        path=f"/ws/sessions/{session_id}",
                                        origin="http://127.0.0.1") as ws:
                        await ws.recv_message()  # hello
                        # Wait for a binary wave patch frame.
                        blob = None
                        deadline = time.monotonic() + 3.0
                        while time.monotonic() < deadline and blob is None:
                            blob = await ws.recv_binary(timeout=0.5)
                        self.assertIsNotNone(blob,
                                              "no wave patch received in 3s")
                        head, payload = blob.split(b"\n", 1)
                        meta = decode_wave_patch_header(head)
                        self.assertEqual(meta["type"], "wave_patch")
                        self.assertEqual(meta["v"], WAVE_PATCH_VERSION)
                        self.assertEqual(meta["spec"]["size"], 9)
                        frame = decode_wave_patch(payload)
                        self.assertEqual(frame.spec.size, 9)
                        self.assertEqual(len(frame.heights), 81)
                        # Heights must all be finite.
                        for h in frame.heights:
                            self.assertTrue(_math.isfinite(h))
        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
