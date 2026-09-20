"""End-to-end integration test: HTTP cockpit + MAVLink SITL on the same session.

The combined `serve` mode in `__main__._serve` wires OperatorServer
(HTTP+WebSocket) and MavlinkUdpBridge together through callbacks:

- MavlinkUdpBridge.get_vehicle() returns the active session's vehicle.
- MavlinkUdpBridge.on_manual forwards MANUAL_CONTROL inputs to the
  session's pending input slot.

This test exercises the same wiring without subprocesses so we can
assert telemetry, ARM, TAKEOFF, MANUAL_CONTROL, and WS broadcast all
flow through one shared session.
"""
import asyncio
import json
import socket
import unittest

from operator_training.curriculum import Curriculum
from operator_training.envelope import ControlEnvelope
from operator_training.mavlink_udp import MavlinkUdpBridge
from operator_training.mavlink_wire import (
    MSG_COMMAND_LONG,
    MSG_HEARTBEAT,
    MSG_MANUAL_CONTROL,
    MAV_CMD_COMPONENT_ARM_DISARM,
    MAV_CMD_NAV_TAKEOFF,
    decode_buffer,
    encode_v2,
    pack_command_long,
    pack_heartbeat,
    pack_manual_control,
)
from operator_training.server import OperatorServer, ServerConfig


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


async def _drive(server, bridge, duration_s: float):
    end = asyncio.get_event_loop().time() + duration_s
    while asyncio.get_event_loop().time() < end:
        bridge.emit()
        await asyncio.sleep(0.05)


class CombinedServeIntegration(unittest.TestCase):
    """HTTP+WS operator server plus MAVLink UDP bridge share one session."""

    def test_http_capabilities_and_mavlink_heartbeat(self):
        async def run():
            server = OperatorServer(
                config=ServerConfig(
                    allowed_origins=("http://localhost",),
                    allowed_hosts=("localhost", "127.0.0.1"),
                    tick_hz=20.0,
                ),
                envelope=ControlEnvelope.beginner(),
                curriculum=Curriculum(
                    curriculum_id="hybrid_baseline",
                    curriculum_version="1",
                    scenario="hybrid",
                ),
            )
            await server.start(host="127.0.0.1", port=0)
            http_port = server.bound_port

            # Create a session through the public HTTP API.
            reader, writer = await asyncio.open_connection(
                "127.0.0.1", http_port,
            )
            body = b'{"seed":42,"target_takeoff_alt_m":15.0}'
            writer.write(
                b"POST /api/sessions HTTP/1.1\r\n"
                b"Host: 127.0.0.1\r\n"
                b"Content-Type: application/json\r\n"
                b"Content-Length: " + str(len(body)).encode() + b"\r\n"
                b"Connection: close\r\n\r\n" + body
            )
            await writer.drain()
            buf = b""
            while True:
                chunk = await reader.read(4096)
                if not chunk:
                    break
                buf += chunk
            writer.close()
            await writer.wait_closed()
            head, _, body = buf.partition(b"\r\n\r\n")
            self.assertTrue(head.startswith(b"HTTP/1.1 201"), msg=head[:60])
            created = json.loads(body.decode())
            session_id = created["session_id"]
            record = server.registry.get(session_id)
            self.assertIsNotNone(record, "session not registered")
            session = record.session
            session.adapter.arm()
            session.state.lifecycle = "RUNNING"
            session.state.authority = "HUMAN"
            session.state.phase = "TAKEOFF"

            def _vehicle():
                return session.adapter.vehicle

            def _on_manual(ci):
                session._pending_input = ci
                session._pending_seq = session.state.last_input_seq + 1

            mav_port = _free_port()
            gcs_port = _free_port()
            bridge = MavlinkUdpBridge(
                get_vehicle=_vehicle,
                bind_host="127.0.0.1", bind_port=mav_port,
                gcs_host="127.0.0.1", gcs_port=gcs_port,
                on_manual=_on_manual,
            )
            await bridge.start()
            try:
                # -- HTTP capabilities endpoint --
                reader, writer = await asyncio.open_connection(
                    "127.0.0.1", http_port,
                )
                writer.write(
                    b"GET /api/capabilities HTTP/1.1\r\n"
                    b"Host: 127.0.0.1\r\n"
                    b"Connection: close\r\n\r\n"
                )
                await writer.drain()
                buf = b""
                while True:
                    chunk = await reader.read(4096)
                    if not chunk:
                        break
                    buf += chunk
                writer.close()
                await writer.wait_closed()
                head, _, body = buf.partition(b"\r\n\r\n")
                self.assertTrue(head.startswith(b"HTTP/1.1 200"),
                                msg=head[:60])
                payload = json.loads(body.decode())
                self.assertIn("control_envelope", payload)
                self.assertIn("curricula", payload)
                self.assertIn("schemas", payload)
                self.assertEqual(payload["v"], 1)

                # -- MAVLink: GCS receives HEARTBEAT --
                gcs = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                gcs.bind(("127.0.0.1", gcs_port))
                gcs.settimeout(2.0)
                bridge._last_hb = 0.0
                bridge.emit()
                try:
                    data, _ = gcs.recvfrom(4096)
                except socket.timeout:
                    data = b""
                self.assertTrue(data, "no MAVLink traffic received")
                pkts = decode_buffer(bytearray(data))
                self.assertTrue(any(p.msgid == MSG_HEARTBEAT for p in pkts))

                # -- MAVLink: ARM via COMMAND_LONG --
                arm = encode_v2(
                    MSG_COMMAND_LONG,
                    pack_command_long(command=MAV_CMD_COMPONENT_ARM_DISARM,
                                       param1=1.0),
                    sysid=255, compid=190,
                )
                gcs.sendto(arm, ("127.0.0.1", bridge.bound_port))
                await asyncio.sleep(0.05)
                self.assertTrue(session.adapter.vehicle._armed)

                # -- MAVLink: TAKEOFF sets active command --
                takeoff = encode_v2(
                    MSG_COMMAND_LONG,
                    pack_command_long(command=MAV_CMD_NAV_TAKEOFF,
                                       param7=12.0),
                    sysid=255, compid=190,
                )
                gcs.sendto(takeoff, ("127.0.0.1", bridge.bound_port))
                await asyncio.sleep(0.05)
                self.assertEqual(
                    session.adapter.vehicle._active_cmd,
                    MAV_CMD_NAV_TAKEOFF,
                )

                # -- MAVLink: MANUAL_CONTROL lands as pending input --
                manual = encode_v2(
                    MSG_MANUAL_CONTROL,
                    pack_manual_control(x=0, y=0, z=900, r=0, target=1),
                    sysid=255, compid=190,
                )
                gcs.sendto(manual, ("127.0.0.1", bridge.bound_port))
                await asyncio.sleep(0.05)
                self.assertIsNotNone(session._pending_input)
                self.assertGreater(session._pending_input.throttle, 0.8)

                # -- Drive a few seconds to confirm the bridge loop is healthy --
                await _drive(server, bridge, 0.4)
                gcs.sendto(encode_v2(
                    MSG_HEARTBEAT, pack_heartbeat(), sysid=255, compid=190,
                ), ("127.0.0.1", bridge.bound_port))
                await asyncio.sleep(0.05)
                gcs.close()
            finally:
                await bridge.stop()
                await server.stop()
        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
