"""UDP MAVLink SITL so QGroundControl / MAVProxy / Mission Planner can fly KMT.

Binds a local UDP port (default 14551) and streams HEARTBEAT + telemetry to
the GCS listen address (default 127.0.0.1:14550, QGC's auto-connect port).
Incoming COMMAND_LONG / MANUAL_CONTROL are applied to FlyingBoatVehicle.
"""
from __future__ import annotations

import asyncio
import logging
import math
import time
from typing import Callable, Optional

from aircraft import Aircraft
from mavlink_if import (
    MAV_CMD_DO_CHANGE_SPEED,
    MAV_CMD_DO_SET_SERVO,
    MAV_CMD_NAV_LAND,
    MAV_CMD_NAV_TAKEOFF,
    MAV_CMD_NAV_WAYPOINT,
    FlyingBoatVehicle,
)
from ocean_directional import DirectionalOcean

from .envelope import ControlEnvelope, ControlInput
from .mavlink_wire import (
    INT16_MAX,
    MAV_AUTOPILOT_GENERIC,
    MAV_CMD_COMPONENT_ARM_DISARM,
    MAV_CMD_DO_CHANGE_SPEED as WIRE_DO_CHANGE_SPEED,
    MAV_CMD_DO_SET_MODE,
    MAV_CMD_DO_SET_SERVO as WIRE_DO_SET_SERVO,
    MAV_CMD_GET_HOME_POSITION,
    MAV_CMD_NAV_LAND as WIRE_NAV_LAND,
    MAV_CMD_NAV_TAKEOFF as WIRE_NAV_TAKEOFF,
    MAV_CMD_NAV_WAYPOINT as WIRE_NAV_WAYPOINT,
    MAV_CMD_REQUEST_AUTOPILOT_CAPABILITIES,
    MAV_CMD_REQUEST_MESSAGE,
    MAV_CMD_REQUEST_PROTOCOL_VERSION,
    MAV_COMP_ID_AUTOPILOT1,
    MAV_MODE_FLAG_GUIDED_ENABLED,
    MAV_MODE_FLAG_MANUAL_INPUT_ENABLED,
    MAV_MODE_FLAG_SAFETY_ARMED,
    MAV_MODE_FLAG_STABILIZE_ENABLED,
    MAV_RESULT_ACCEPTED,
    MAV_RESULT_DENIED,
    MAV_RESULT_FAILED,
    MAV_RESULT_UNSUPPORTED,
    MAV_STATE_ACTIVE,
    MAV_STATE_CRITICAL,
    MAV_STATE_STANDBY,
    MAV_TYPE_FIXED_WING,
    MSG_ATTITUDE,
    MSG_COMMAND_LONG,
    MSG_GPS_RAW_INT,
    MSG_GLOBAL_POSITION_INT,
    MSG_HEARTBEAT,
    MSG_HOME_POSITION,
    MSG_LOCAL_POSITION_NED,
    MSG_MANUAL_CONTROL,
    MSG_MISSION_CLEAR_ALL,
    MSG_MISSION_REQUEST_LIST,
    MSG_PARAM_REQUEST_LIST,
    MSG_PARAM_REQUEST_READ,
    MSG_PARAM_VALUE,
    MSG_REQUEST_DATA_STREAM,
    MSG_SET_MODE,
    MSG_STATUSTEXT,
    MSG_SYS_STATUS,
    MSG_VFR_HUD,
    decode_buffer,
    encode_v2,
    pack_attitude,
    pack_command_ack,
    pack_global_position_int,
    pack_gps_raw_int,
    pack_heartbeat,
    pack_home_position,
    pack_local_position_ned,
    pack_mission_ack,
    pack_mission_count,
    pack_param_value,
    pack_statustext,
    pack_sys_status,
    pack_vfr_hud,
    unpack_command_long,
    unpack_manual_control,
    unpack_set_mode,
)


log = logging.getLogger("operator_training.mavlink")

SYSID = 1
COMPID = MAV_COMP_ID_AUTOPILOT1


def _axis(value: int, scale: float) -> float:
    if value == INT16_MAX:
        return 0.0
    return max(-1.0, min(1.0, value / 1000.0)) * scale


class MavlinkUdpBridge:
    """One UDP socket, one vehicle (or a getter that returns the live one)."""

    def __init__(self, *, get_vehicle: Callable[[], Optional[FlyingBoatVehicle]],
                 bind_host: str = "127.0.0.1", bind_port: int = 14551,
                 gcs_host: str = "127.0.0.1", gcs_port: int = 14550,
                 envelope: ControlEnvelope | None = None,
                 on_manual: Callable[[ControlInput], None] | None = None,
                 on_arm: Callable[[bool], bool] | None = None,
                 on_nav: Callable[[int, dict], bool] | None = None):
        self.get_vehicle = get_vehicle
        self.bind_host = bind_host
        self.bind_port = bind_port
        self.gcs = (gcs_host, gcs_port) if gcs_port else None
        self.envelope = envelope or ControlEnvelope.beginner()
        self.on_manual = on_manual
        self.on_arm = on_arm
        self.on_nav = on_nav
        self._transport: Optional[asyncio.DatagramTransport] = None
        self._peers: set[tuple] = set()
        self._seq = 0
        self._rx = bytearray()
        self._last_hb = 0.0
        self._last_slow = 0.0
        self._home: Optional[tuple] = None
        self._manual: Optional[ControlInput] = None
        self._manual_until = 0.0

    @property
    def bound_port(self) -> int:
        if self._transport is None:
            return self.bind_port
        sock = self._transport.get_extra_info("sockname")
        return int(sock[1]) if sock else self.bind_port

    async def start(self) -> None:
        loop = asyncio.get_running_loop()
        self._transport, _ = await loop.create_datagram_endpoint(
            lambda: _Proto(self),
            local_addr=(self.bind_host, self.bind_port),
        )

    async def stop(self) -> None:
        if self._transport is not None:
            self._transport.close()
            self._transport = None

    def feed(self, data: bytes, addr: tuple) -> None:
        self._peers.add(addr)
        self._rx.extend(data)
        for pkt in decode_buffer(self._rx):
            if pkt.sysid == SYSID and pkt.compid == COMPID:
                continue
            self._dispatch(pkt)

    def emit(self) -> None:
        veh = self.get_vehicle()
        if veh is None or self._transport is None:
            return
        now = time.monotonic()
        t_ms = int(veh.t * 1000)
        if now - self._last_hb >= 1.0:
            self._last_hb = now
            self._send_heartbeat(veh)
            self._send(MSG_SYS_STATUS, pack_sys_status())
        if now - self._last_slow >= 0.2:
            self._last_slow = now
            self._send_nav(veh, t_ms)
        self._send_fast(veh, t_ms)

    def apply_manual_if_fresh(self, veh: FlyingBoatVehicle) -> bool:
        if self._manual is None or time.monotonic() > self._manual_until:
            return False
        from mavlink_if import MAV_CMD_DO_SET_SERVO
        if veh._active_cmd != MAV_CMD_DO_SET_SERVO:
            pass
        pwm_thr = int(1000 + 1000 * self._manual.throttle)
        pwm_pit = int(1500 + 500 * (self._manual.pitch_deg / 15.0))
        pwm_ban = int(1500 + 500 * (self._manual.bank_deg / 45.0))
        pwm_rud = int(1500 + 500 * self._manual.rudder)
        veh.send_servo(1, pwm_thr)
        veh.send_servo(2, pwm_pit)
        if veh.spatial:
            veh.send_servo(3, pwm_ban)
            veh.send_servo(4, pwm_rud)
        return True

    def _targets(self) -> list[tuple]:
        out = set(self._peers)
        if self.gcs is not None:
            out.add(self.gcs)
        return list(out)

    def _send(self, msgid: int, payload: bytes) -> None:
        if self._transport is None:
            return
        pkt = encode_v2(msgid, payload, sysid=SYSID, compid=COMPID, seq=self._seq)
        self._seq = (self._seq + 1) & 0xFF
        for addr in self._targets():
            try:
                self._transport.sendto(pkt, addr)
            except OSError:
                pass

    def _send_heartbeat(self, veh: FlyingBoatVehicle) -> None:
        armed = bool(veh._armed)
        failed = bool(veh.damage.failed)
        base = MAV_MODE_FLAG_STABILIZE_ENABLED | MAV_MODE_FLAG_MANUAL_INPUT_ENABLED
        if armed:
            base |= MAV_MODE_FLAG_SAFETY_ARMED | MAV_MODE_FLAG_GUIDED_ENABLED
        status = MAV_STATE_CRITICAL if failed else (
            MAV_STATE_ACTIVE if armed else MAV_STATE_STANDBY
        )
        self._send(MSG_HEARTBEAT, pack_heartbeat(
            mav_type=MAV_TYPE_FIXED_WING,
            autopilot=MAV_AUTOPILOT_GENERIC,
            base_mode=base,
            system_status=status,
        ))

    def _send_fast(self, veh: FlyingBoatVehicle, t_ms: int) -> None:
        self._send(MSG_ATTITUDE, pack_attitude(
            t_ms=t_ms, roll=float(veh.bank), pitch=float(veh.alpha),
            yaw=float(veh.heading), rollspeed=float(veh.rollspeed),
            pitchspeed=float(veh.pitchspeed), yawspeed=float(veh.yawspeed),
        ))
        gs = math.hypot(veh.Vx, veh.Vy)
        heading = int(math.degrees(veh.heading) % 360)
        self._send(MSG_VFR_HUD, pack_vfr_hud(
            airspeed=float(math.hypot(veh.Vx, veh.Vz)),
            groundspeed=float(gs), heading=heading,
            throttle=int(max(0, min(100, veh.throttle * 100))),
            alt=float(veh.z), climb=float(veh.Vz),
        ))

    def _send_nav(self, veh: FlyingBoatVehicle, t_ms: int) -> None:
        lat = veh.origin_lat + veh.x / 111000.0
        lon = veh.origin_lon + veh.y / (111000.0 * math.cos(math.radians(veh.origin_lat)))
        lat_e7 = int(lat * 1e7)
        lon_e7 = int(lon * 1e7)
        alt_mm = int(veh.z * 1000)
        hdg = int(math.degrees(veh.heading) % 360 * 100)
        gs = math.hypot(veh.Vx, veh.Vy)
        self._send(MSG_GLOBAL_POSITION_INT, pack_global_position_int(
            t_ms=t_ms, lat_e7=lat_e7, lon_e7=lon_e7, alt_mm=alt_mm,
            vx_cms=int(veh.Vx * 100), vy_cms=int(veh.Vy * 100),
            vz_cms=int(-veh.Vz * 100), hdg_cdeg=hdg,
        ))
        self._send(MSG_GPS_RAW_INT, pack_gps_raw_int(
            t_us=int(veh.t * 1e6), lat_e7=lat_e7, lon_e7=lon_e7, alt_mm=alt_mm,
            vel_cms=int(gs * 100), cog_cdeg=hdg,
        ))
        self._send(MSG_LOCAL_POSITION_NED, pack_local_position_ned(
            t_ms=t_ms, x=float(veh.x), y=float(veh.y), z=float(-veh.z),
            vx=float(veh.Vx), vy=float(veh.Vy), vz=float(-veh.Vz),
        ))
        if self._home is None and veh._armed:
            self._home = (lat_e7, lon_e7, alt_mm)
            self._send(MSG_HOME_POSITION, pack_home_position(
                lat_e7=lat_e7, lon_e7=lon_e7, alt_mm=alt_mm,
            ))

    def _dispatch(self, pkt) -> None:
        if pkt.msgid == MSG_COMMAND_LONG:
            try:
                cmd = unpack_command_long(pkt.payload)
            except ValueError:
                return
            if cmd["target_system"] not in (0, SYSID):
                return
            self._handle_command(cmd)
        elif pkt.msgid == MSG_MANUAL_CONTROL:
            mc = unpack_manual_control(pkt.payload)
            if mc and mc["target"] in (0, SYSID):
                self._handle_manual(mc)
        elif pkt.msgid == MSG_SET_MODE:
            unpack_set_mode(pkt.payload)
        elif pkt.msgid == MSG_MISSION_REQUEST_LIST:
            self._send(44, pack_mission_count(0))
        elif pkt.msgid == MSG_MISSION_CLEAR_ALL:
            self._send(47, pack_mission_ack())
        elif pkt.msgid in (MSG_PARAM_REQUEST_LIST, MSG_PARAM_REQUEST_READ):
            self._send(MSG_PARAM_VALUE, pack_param_value("SYSID_THISMAV", float(SYSID), 0, 1))
        elif pkt.msgid == MSG_REQUEST_DATA_STREAM:
            pass

    def _handle_command(self, cmd: dict) -> None:
        command = int(cmd["command"])
        veh = self.get_vehicle()
        result = MAV_RESULT_UNSUPPORTED
        if command == MAV_CMD_COMPONENT_ARM_DISARM:
            result = self._arm(cmd["param1"] >= 0.5)
        elif command in (WIRE_NAV_TAKEOFF, WIRE_NAV_LAND, WIRE_NAV_WAYPOINT,
                         WIRE_DO_CHANGE_SPEED, WIRE_DO_SET_SERVO,
                         MAV_CMD_DO_SET_MODE):
            result = self._nav(command, cmd)
        elif command in (MAV_CMD_REQUEST_AUTOPILOT_CAPABILITIES,
                         MAV_CMD_REQUEST_PROTOCOL_VERSION,
                         MAV_CMD_REQUEST_MESSAGE, MAV_CMD_GET_HOME_POSITION):
            result = MAV_RESULT_ACCEPTED
            if command == MAV_CMD_GET_HOME_POSITION and self._home:
                self._send(MSG_HOME_POSITION, pack_home_position(
                    lat_e7=self._home[0], lon_e7=self._home[1], alt_mm=self._home[2],
                ))
        self._send(77, pack_command_ack(command, result))
        if result == MAV_RESULT_ACCEPTED and command == MAV_CMD_COMPONENT_ARM_DISARM:
            self._send(MSG_STATUSTEXT, pack_statustext(
                "ARMED" if veh is not None and veh._armed else "DISARMED",
            ))

    def _arm(self, want_arm: bool) -> int:
        if self.on_arm is not None:
            return MAV_RESULT_ACCEPTED if self.on_arm(want_arm) else MAV_RESULT_DENIED
        veh = self.get_vehicle()
        if veh is None:
            return MAV_RESULT_FAILED
        try:
            if want_arm:
                veh.arm()
            else:
                veh.disarm()
            return MAV_RESULT_ACCEPTED
        except RuntimeError:
            return MAV_RESULT_DENIED

    def _nav(self, command: int, cmd: dict) -> int:
        if self.on_nav is not None:
            ok = self.on_nav(command, cmd)
            return MAV_RESULT_ACCEPTED if ok else MAV_RESULT_DENIED
        veh = self.get_vehicle()
        if veh is None:
            return MAV_RESULT_FAILED
        if not veh._armed:
            return MAV_RESULT_DENIED
        try:
            if command == WIRE_NAV_TAKEOFF:
                alt = float(cmd["param7"] or 25.0)
                veh.send_command(MAV_CMD_NAV_TAKEOFF, {"alt": alt, "speed": 13.0})
            elif command == WIRE_NAV_LAND:
                veh.send_command(MAV_CMD_NAV_LAND, {"alt": 0.0, "glide": 8.0})
            elif command == WIRE_NAV_WAYPOINT:
                veh.send_command(MAV_CMD_NAV_WAYPOINT, {
                    "alt": float(cmd["param7"] or veh.z),
                    "speed": 13.0,
                })
            elif command == WIRE_DO_CHANGE_SPEED:
                speed = float(cmd["param2"] or cmd["param1"] or 13.0)
                veh.send_command(MAV_CMD_DO_CHANGE_SPEED, {"speed": speed})
            elif command == WIRE_DO_SET_SERVO:
                veh.send_command(MAV_CMD_DO_SET_SERVO, {
                    "servo": int(cmd["param1"]), "pwm": float(cmd["param2"]),
                })
            else:
                return MAV_RESULT_UNSUPPORTED
            return MAV_RESULT_ACCEPTED
        except (ValueError, RuntimeError):
            return MAV_RESULT_FAILED

    def _handle_manual(self, mc: dict) -> None:
        env = self.envelope
        thr_src = mc["z"]
        throttle = 0.0 if thr_src == INT16_MAX else max(0.0, min(1.0, thr_src / 1000.0))
        nx = 0.0 if mc["x"] == INT16_MAX else max(-1.0, min(1.0, -mc["x"] / 1000.0))
        pitch = nx * env.pitch_hi if nx >= 0.0 else nx * (-env.pitch_lo)
        raw = {
            "throttle": throttle,
            "pitch_deg": pitch,
            "bank_deg": _axis(mc["y"], env.bank_abs),
            "rudder": _axis(mc["r"], env.rudder_abs),
        }
        try:
            ci = self.envelope.validate(raw)
        except Exception:
            return
        self._manual = ci
        self._manual_until = time.monotonic() + 0.4
        if self.on_manual is not None:
            self.on_manual(ci)


class _Proto(asyncio.DatagramProtocol):
    def __init__(self, bridge: MavlinkUdpBridge):
        self.bridge = bridge

    def datagram_received(self, data: bytes, addr) -> None:
        self.bridge.feed(data, addr)


def build_sitl_vehicle(*, seed: int = 42, spatial: bool = True) -> FlyingBoatVehicle:
    ac = Aircraft()
    sea = DirectionalOcean(Hs=0.8, Tp=6.0, theta_mean=0.0, seed=seed)
    return FlyingBoatVehicle(ac, sea, spatial=spatial, seed=seed)


async def run_sitl(*, bind_host: str = "127.0.0.1", bind_port: int = 14551,
                   gcs_host: str = "127.0.0.1", gcs_port: int = 14550,
                   tick_hz: float = 20.0, seed: int = 42,
                   duration_s: float | None = None) -> int:
    veh = build_sitl_vehicle(seed=seed)
    bridge = MavlinkUdpBridge(
        get_vehicle=lambda: veh,
        bind_host=bind_host, bind_port=bind_port,
        gcs_host=gcs_host, gcs_port=gcs_port,
    )
    await bridge.start()
    dt = 1.0 / tick_hz
    print(f"MAVLink SITL udp:{bind_host}:{bridge.bound_port} -> {gcs_host}:{gcs_port}",
          flush=True)
    print("QGroundControl: auto-connect UDP 14550, or add link udp:127.0.0.1:"
          f"{bridge.bound_port}", flush=True)
    print("MAVProxy: mavproxy.py --master=udp:127.0.0.1:"
          f"{bridge.bound_port} --out=udp:127.0.0.1:{gcs_port}", flush=True)
    print("Arm from the GCS, then Takeoff / joystick (MANUAL_CONTROL).", flush=True)
    t0 = time.monotonic()
    try:
        while True:
            if duration_s is not None and (time.monotonic() - t0) >= duration_s:
                break
            if veh._armed:
                bridge.apply_manual_if_fresh(veh)
                veh.step(dt=dt)
            bridge.emit()
            await asyncio.sleep(dt)
    finally:
        await bridge.stop()
    return 0
