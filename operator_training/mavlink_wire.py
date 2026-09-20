"""Minimal MAVLink v1/v2 codec (stdlib only).

pymavlink does not build on the project's Python 3.14 image, so the
SITL path encodes the common dialect subset QGroundControl / MAVProxy
need: HEARTBEAT, SYS_STATUS, ATTITUDE, GLOBAL_POSITION_INT, GPS_RAW_INT,
VFR_HUD, COMMAND_LONG/ACK, MANUAL_CONTROL, PARAM_VALUE, MISSION_COUNT.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass


STX_V1 = 0xFE
STX_V2 = 0xFD

MSG_HEARTBEAT = 0
MSG_SYS_STATUS = 1
MSG_SYSTEM_TIME = 2
MSG_PARAM_REQUEST_READ = 20
MSG_PARAM_REQUEST_LIST = 21
MSG_PARAM_VALUE = 22
MSG_GPS_RAW_INT = 24
MSG_ATTITUDE = 30
MSG_LOCAL_POSITION_NED = 32
MSG_GLOBAL_POSITION_INT = 33
MSG_MISSION_ITEM = 39
MSG_MISSION_REQUEST = 40
MSG_MISSION_CURRENT = 42
MSG_MISSION_REQUEST_LIST = 43
MSG_MISSION_COUNT = 44
MSG_MISSION_CLEAR_ALL = 45
MSG_MISSION_ACK = 47
MSG_SET_MODE = 11
MSG_REQUEST_DATA_STREAM = 66
MSG_MANUAL_CONTROL = 69
MSG_RC_CHANNELS_OVERRIDE = 70
MSG_VFR_HUD = 74
MSG_COMMAND_INT = 75
MSG_COMMAND_LONG = 76
MSG_COMMAND_ACK = 77
MSG_AUTOPILOT_VERSION = 148
MSG_HOME_POSITION = 242
MSG_STATUSTEXT = 253

CRC_EXTRA = {
    MSG_HEARTBEAT: 50,
    MSG_SYS_STATUS: 124,
    MSG_SYSTEM_TIME: 137,
    MSG_SET_MODE: 89,
    MSG_PARAM_REQUEST_READ: 214,
    MSG_PARAM_REQUEST_LIST: 159,
    MSG_PARAM_VALUE: 220,
    MSG_GPS_RAW_INT: 24,
    MSG_ATTITUDE: 39,
    MSG_LOCAL_POSITION_NED: 185,
    MSG_GLOBAL_POSITION_INT: 104,
    MSG_MISSION_ITEM: 254,
    MSG_MISSION_REQUEST: 230,
    MSG_MISSION_CURRENT: 28,
    MSG_MISSION_REQUEST_LIST: 132,
    MSG_MISSION_COUNT: 221,
    MSG_MISSION_CLEAR_ALL: 232,
    MSG_MISSION_ACK: 153,
    MSG_REQUEST_DATA_STREAM: 148,
    MSG_MANUAL_CONTROL: 243,
    MSG_RC_CHANNELS_OVERRIDE: 124,
    MSG_VFR_HUD: 20,
    MSG_COMMAND_INT: 158,
    MSG_COMMAND_LONG: 152,
    MSG_COMMAND_ACK: 143,
    MSG_AUTOPILOT_VERSION: 178,
    MSG_HOME_POSITION: 104,
    MSG_STATUSTEXT: 83,
}

MAV_TYPE_FIXED_WING = 1
MAV_AUTOPILOT_GENERIC = 0
MAV_STATE_STANDBY = 3
MAV_STATE_ACTIVE = 4
MAV_STATE_CRITICAL = 5
MAV_COMP_ID_AUTOPILOT1 = 1

MAV_MODE_FLAG_CUSTOM_MODE_ENABLED = 1
MAV_MODE_FLAG_AUTO_ENABLED = 4
MAV_MODE_FLAG_GUIDED_ENABLED = 8
MAV_MODE_FLAG_STABILIZE_ENABLED = 16
MAV_MODE_FLAG_MANUAL_INPUT_ENABLED = 64
MAV_MODE_FLAG_SAFETY_ARMED = 128

MAV_RESULT_ACCEPTED = 0
MAV_RESULT_DENIED = 2
MAV_RESULT_UNSUPPORTED = 3
MAV_RESULT_FAILED = 4

MAV_CMD_NAV_WAYPOINT = 16
MAV_CMD_NAV_LAND = 21
MAV_CMD_NAV_TAKEOFF = 22
MAV_CMD_DO_SET_MODE = 176
MAV_CMD_DO_CHANGE_SPEED = 178
MAV_CMD_DO_SET_SERVO = 183
MAV_CMD_COMPONENT_ARM_DISARM = 400
MAV_CMD_GET_HOME_POSITION = 410
MAV_CMD_REQUEST_MESSAGE = 512
MAV_CMD_REQUEST_PROTOCOL_VERSION = 519
MAV_CMD_REQUEST_AUTOPILOT_CAPABILITIES = 520

MAV_SEVERITY_INFO = 6
PARAM_TYPE_REAL32 = 9
GPS_FIX_3D = 3
INT16_MAX = 32767


def x25crc(data: bytes, crc: int = 0xFFFF) -> int:
    for b in data:
        tmp = b ^ (crc & 0xFF)
        tmp ^= (tmp << 4) & 0xFF
        crc = ((crc >> 8) ^ (tmp << 8) ^ (tmp << 3) ^ (tmp >> 4)) & 0xFFFF
    return crc


def _crc_extra(msgid: int) -> int:
    extra = CRC_EXTRA.get(msgid)
    if extra is None:
        raise ValueError(f"no crc extra for msgid {msgid}")
    return extra


@dataclass(frozen=True)
class MavlinkPacket:
    magic: int
    seq: int
    sysid: int
    compid: int
    msgid: int
    payload: bytes

    @property
    def is_v2(self) -> bool:
        return self.magic == STX_V2


def encode_v2(msgid: int, payload: bytes, *, sysid: int = 1, compid: int = 1,
              seq: int = 0) -> bytes:
    if len(payload) > 255:
        raise ValueError("payload too long")
    header = bytes([
        STX_V2, len(payload), 0, 0, seq & 0xFF, sysid & 0xFF, compid & 0xFF,
        msgid & 0xFF, (msgid >> 8) & 0xFF, (msgid >> 16) & 0xFF,
    ])
    body = header[1:] + payload
    crc = x25crc(body)
    crc = x25crc(bytes([_crc_extra(msgid)]), crc)
    return header + payload + struct.pack("<H", crc)


def encode_v1(msgid: int, payload: bytes, *, sysid: int = 1, compid: int = 1,
              seq: int = 0) -> bytes:
    if msgid > 255 or len(payload) > 255:
        raise ValueError("v1 msgid/payload out of range")
    header = bytes([
        STX_V1, len(payload), seq & 0xFF, sysid & 0xFF, compid & 0xFF, msgid,
    ])
    body = header[1:] + payload
    crc = x25crc(body)
    crc = x25crc(bytes([_crc_extra(msgid)]), crc)
    return header + payload + struct.pack("<H", crc)


def decode_buffer(buf: bytearray) -> list[MavlinkPacket]:
    """Consume complete frames from *buf*; leave a partial frame in place."""
    out: list[MavlinkPacket] = []
    while True:
        pkt, consumed = _pop_one(buf)
        if consumed == 0:
            break
        del buf[:consumed]
        if pkt is not None:
            out.append(pkt)
    return out


def _pop_one(buf: bytearray) -> tuple[MavlinkPacket | None, int]:
    n = len(buf)
    i = 0
    while i < n and buf[i] not in (STX_V1, STX_V2):
        i += 1
    if i:
        return None, i
    if n < 8:
        return None, 0
    magic = buf[0]
    if magic == STX_V1:
        if n < 6:
            return None, 0
        plen, seq, sysid, compid, msgid = buf[1], buf[2], buf[3], buf[4], buf[5]
        total = 6 + plen + 2
        if n < total:
            return None, 0
        payload = bytes(buf[6:6 + plen])
        crc_got = struct.unpack_from("<H", buf, 6 + plen)[0]
        crc = x25crc(bytes(buf[1:6 + plen]))
        try:
            crc = x25crc(bytes([_crc_extra(msgid)]), crc)
        except ValueError:
            return None, total
        if crc != crc_got:
            return None, 1
        return MavlinkPacket(magic, seq, sysid, compid, msgid, payload), total
    if n < 12:
        return None, 0
    plen = buf[1]
    incompat = buf[2]
    seq, sysid, compid = buf[4], buf[5], buf[6]
    msgid = buf[7] | (buf[8] << 8) | (buf[9] << 16)
    sig = 13 if incompat & 0x01 else 0
    total = 10 + plen + 2 + sig
    if n < total:
        return None, 0
    payload = bytes(buf[10:10 + plen])
    crc_got = struct.unpack_from("<H", buf, 10 + plen)[0]
    crc = x25crc(bytes(buf[1:10 + plen]))
    try:
        crc = x25crc(bytes([_crc_extra(msgid)]), crc)
    except ValueError:
        return None, total
    if crc != crc_got:
        return None, 1
    return MavlinkPacket(magic, seq, sysid, compid, msgid, payload), total


def pack_heartbeat(*, custom_mode: int = 0, mav_type: int = MAV_TYPE_FIXED_WING,
                   autopilot: int = MAV_AUTOPILOT_GENERIC, base_mode: int = 0,
                   system_status: int = MAV_STATE_STANDBY) -> bytes:
    return struct.pack("<IBBBBB", custom_mode, mav_type, autopilot, base_mode,
                       system_status, 3)


def pack_sys_status(*, voltage_mv: int = 22100, load: int = 200,
                    sensors: int = 0x1F) -> bytes:
    return struct.pack("<IIIHHhbHHHHHH", sensors, sensors, sensors, load,
                       voltage_mv, -1, -1, 0, 0, 0, 0, 0, 0)


def pack_attitude(*, t_ms: int, roll: float, pitch: float, yaw: float,
                  rollspeed: float = 0.0, pitchspeed: float = 0.0,
                  yawspeed: float = 0.0) -> bytes:
    return struct.pack("<Iffffff", t_ms, roll, pitch, yaw,
                       rollspeed, pitchspeed, yawspeed)


def pack_global_position_int(*, t_ms: int, lat_e7: int, lon_e7: int,
                             alt_mm: int, vx_cms: int, vy_cms: int,
                             vz_cms: int, hdg_cdeg: int) -> bytes:
    return struct.pack("<IiiiihhhH", t_ms, lat_e7, lon_e7, alt_mm, alt_mm,
                       vx_cms, vy_cms, vz_cms, hdg_cdeg & 0xFFFF)


def pack_gps_raw_int(*, t_us: int, lat_e7: int, lon_e7: int, alt_mm: int,
                     vel_cms: int, cog_cdeg: int, sats: int = 12) -> bytes:
    return struct.pack("<QBiiiHHHHB", t_us, GPS_FIX_3D, lat_e7, lon_e7, alt_mm,
                       80, 120, vel_cms & 0xFFFF, cog_cdeg & 0xFFFF, sats)


def pack_vfr_hud(*, airspeed: float, groundspeed: float, heading: int,
                 throttle: int, alt: float, climb: float) -> bytes:
    return struct.pack("<ffHHff", airspeed, groundspeed, heading & 0xFFFF,
                       throttle & 0xFFFF, alt, climb)


def pack_local_position_ned(*, t_ms: int, x: float, y: float, z: float,
                            vx: float, vy: float, vz: float) -> bytes:
    return struct.pack("<Iffffff", t_ms, x, y, z, vx, vy, vz)


def pack_command_ack(command: int, result: int) -> bytes:
    return struct.pack("<HB", command, result)


def pack_statustext(text: str, severity: int = MAV_SEVERITY_INFO) -> bytes:
    raw = text.encode("ascii", "replace")[:50]
    raw = raw + b"\x00" * (50 - len(raw))
    return bytes([severity]) + raw


def pack_mission_count(count: int = 0, sysid: int = 255, compid: int = 0) -> bytes:
    return struct.pack("<HBB", count, sysid, compid)


def pack_mission_ack(sysid: int = 255, compid: int = 0, result: int = 0) -> bytes:
    return struct.pack("<BBB", sysid, compid, result)


def pack_param_value(name: str, value: float, index: int, count: int) -> bytes:
    ident = name.encode("ascii")[:16]
    ident = ident + b"\x00" * (16 - len(ident))
    return struct.pack("<fHH", value, count, index) + ident + bytes([PARAM_TYPE_REAL32])


def pack_home_position(*, lat_e7: int, lon_e7: int, alt_mm: int) -> bytes:
    q = struct.pack("<ffff", 1.0, 0.0, 0.0, 0.0)
    return struct.pack("<iii", lat_e7, lon_e7, alt_mm) + struct.pack("<fff", 0.0, 0.0, 0.0) + q + struct.pack("<fff", 0.0, 0.0, 0.0)


def pack_command_long(*, command: int, param1: float = 0, param2: float = 0,
                      param3: float = 0, param4: float = 0, param5: float = 0,
                      param6: float = 0, param7: float = 0,
                      target_system: int = 1, target_component: int = 1,
                      confirmation: int = 0) -> bytes:
    return struct.pack("<fffffffHBBB", param1, param2, param3, param4, param5,
                       param6, param7, command, target_system, target_component,
                       confirmation)


def pack_manual_control(*, x: int = 0, y: int = 0, z: int = 0, r: int = 0,
                        buttons: int = 0, target: int = 1) -> bytes:
    return struct.pack("<hhhhHB", x, y, z, r, buttons, target)


def unpack_command_long(payload: bytes) -> dict:
    if len(payload) < 33:
        raise ValueError("COMMAND_LONG truncated")
    p1, p2, p3, p4, p5, p6, p7, command, target_system, target_component, confirmation = struct.unpack_from(
        "<fffffffHBBB", payload, 0,
    )
    return {
        "param1": p1, "param2": p2, "param3": p3, "param4": p4,
        "param5": p5, "param6": p6, "param7": p7,
        "command": command, "target_system": target_system,
        "target_component": target_component, "confirmation": confirmation,
    }


def unpack_manual_control(payload: bytes) -> dict | None:
    if len(payload) < 11:
        return None
    x, y, z, r, buttons, target = struct.unpack_from("<hhhhHB", payload, 0)
    return {"x": x, "y": y, "z": z, "r": r, "buttons": buttons, "target": target}


def unpack_set_mode(payload: bytes) -> dict | None:
    if len(payload) < 6:
        return None
    custom_mode, target_system, base_mode = struct.unpack_from("<IBB", payload, 0)
    return {"custom_mode": custom_mode, "target_system": target_system,
            "base_mode": base_mode}
