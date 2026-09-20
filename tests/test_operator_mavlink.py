import asyncio
import socket
import unittest

from operator_training.mavlink_udp import MavlinkUdpBridge, build_sitl_vehicle
from operator_training.mavlink_wire import (
    MAV_CMD_COMPONENT_ARM_DISARM,
    MAV_CMD_NAV_TAKEOFF,
    MSG_COMMAND_ACK,
    MSG_COMMAND_LONG,
    MSG_HEARTBEAT,
    MSG_MANUAL_CONTROL,
    decode_buffer,
    encode_v1,
    encode_v2,
    pack_command_long,
    pack_heartbeat,
    pack_manual_control,
    unpack_command_long,
    x25crc,
)


class WireTests(unittest.TestCase):
    def test_v2_round_trip_heartbeat(self):
        payload = pack_heartbeat(base_mode=128, system_status=4)
        blob = encode_v2(MSG_HEARTBEAT, payload, seq=7)
        self.assertEqual(blob[0], 0xFD)
        pkts = decode_buffer(bytearray(blob))
        self.assertEqual(len(pkts), 1)
        self.assertEqual(pkts[0].msgid, MSG_HEARTBEAT)
        self.assertEqual(pkts[0].seq, 7)
        self.assertEqual(pkts[0].payload, payload)

    def test_v1_round_trip(self):
        payload = pack_heartbeat()
        blob = encode_v1(MSG_HEARTBEAT, payload, seq=3)
        pkts = decode_buffer(bytearray(blob))
        self.assertEqual(pkts[0].msgid, MSG_HEARTBEAT)
        self.assertEqual(pkts[0].magic, 0xFE)

    def test_bad_crc_is_skipped(self):
        blob = bytearray(encode_v2(MSG_HEARTBEAT, pack_heartbeat()))
        blob[-1] ^= 0xFF
        pkts = decode_buffer(blob)
        self.assertEqual(pkts, [])

    def test_command_long_pack_unpack(self):
        raw = pack_command_long(command=400, param1=1.0, param7=25.0)
        d = unpack_command_long(raw)
        self.assertEqual(d["command"], 400)
        self.assertAlmostEqual(d["param1"], 1.0)
        self.assertAlmostEqual(d["param7"], 25.0)

    def test_x25_known_vector(self):
        self.assertEqual(x25crc(b""), 0xFFFF)
        self.assertNotEqual(x25crc(b"\x00"), 0xFFFF)


class UdpSitlTests(unittest.TestCase):
    def test_heartbeat_and_arm_takeoff(self):
        async def run():
            veh = build_sitl_vehicle(seed=1)
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.bind(("127.0.0.1", 0))
            sock.setblocking(False)
            gcs_port = sock.getsockname()[1]
            bridge = MavlinkUdpBridge(
                get_vehicle=lambda: veh,
                bind_host="127.0.0.1", bind_port=0,
                gcs_host="127.0.0.1", gcs_port=gcs_port,
            )
            await bridge.start()
            try:
                bridge._last_hb = 0.0
                bridge.emit()
                await asyncio.sleep(0.05)
                data, _ = sock.recvfrom(4096)
                pkts = decode_buffer(bytearray(data))
                self.assertTrue(any(p.msgid == MSG_HEARTBEAT for p in pkts))
                arm = encode_v2(
                    MSG_COMMAND_LONG,
                    pack_command_long(command=MAV_CMD_COMPONENT_ARM_DISARM, param1=1.0),
                    sysid=255, compid=190,
                )
                sock.sendto(arm, ("127.0.0.1", bridge.bound_port))
                await asyncio.sleep(0.05)
                self.assertTrue(veh._armed)
                takeoff = encode_v2(
                    MSG_COMMAND_LONG,
                    pack_command_long(command=MAV_CMD_NAV_TAKEOFF, param7=20.0),
                    sysid=255, compid=190,
                )
                sock.sendto(takeoff, ("127.0.0.1", bridge.bound_port))
                await asyncio.sleep(0.05)
                self.assertEqual(veh._active_cmd, MAV_CMD_NAV_TAKEOFF)
                vx0 = veh.Vx
                for _ in range(80):
                    veh.step(dt=0.05)
                self.assertGreater(veh.Vx, vx0)
                self.assertGreater(veh.throttle, 0.5)
            finally:
                await bridge.stop()
                sock.close()
        asyncio.run(run())

    def test_manual_control_sets_servo(self):
        async def run():
            veh = build_sitl_vehicle(seed=2)
            veh.arm()
            seen = []
            bridge = MavlinkUdpBridge(
                get_vehicle=lambda: veh,
                bind_host="127.0.0.1", bind_port=0,
                gcs_host="127.0.0.1", gcs_port=9,
                on_manual=seen.append,
            )
            await bridge.start()
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                pkt = encode_v2(
                    MSG_MANUAL_CONTROL,
                    pack_manual_control(x=0, y=0, z=800, r=0, target=1),
                    sysid=255, compid=190,
                )
                sock.sendto(pkt, ("127.0.0.1", bridge.bound_port))
                await asyncio.sleep(0.05)
                self.assertEqual(len(seen), 1)
                self.assertGreater(seen[0].throttle, 0.7)
                self.assertTrue(bridge.apply_manual_if_fresh(veh))
                self.assertGreater(veh.throttle, 0.7)
            finally:
                await bridge.stop()
                sock.close()
        asyncio.run(run())


class CliMavlinkTests(unittest.TestCase):
    def test_mavlink_duration_exits(self):
        import subprocess
        import sys
        r = subprocess.run(
            [sys.executable, "-m", "operator_training", "mavlink",
             "--duration", "0.4", "--port", "0"],
            capture_output=True, text=True, timeout=8,
        )
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("MAVLink SITL", r.stdout)


if __name__ == "__main__":
    unittest.main()
