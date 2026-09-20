"""WebSocket frame encoder/decoder (RFC 6455) using only the stdlib.

This module implements just enough of RFC 6455 to serve the operator
training WS endpoint. It supports text frames (the only kind we send
or receive - JSON payloads) and ping/pong for keep-alive. Fragmentation
is rejected - the wire protocol requires one logical message per
WS frame, which is how the canonical client in this project works.
"""
from __future__ import annotations

import base64
import hashlib
import os
import struct
from dataclasses import dataclass
from typing import Optional


WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
OP_CONT = 0x0
OP_TEXT = 0x1
OP_BINARY = 0x2
OP_CLOSE = 0x8
OP_PING = 0x9
OP_PONG = 0xA


@dataclass
class WSFrame:
    opcode: int
    payload: bytes

    @property
    def is_text(self) -> bool:
        return self.opcode == OP_TEXT

    @property
    def is_close(self) -> bool:
        return self.opcode == OP_CLOSE

    @property
    def is_ping(self) -> bool:
        return self.opcode == OP_PING

    @property
    def is_pong(self) -> bool:
        return self.opcode == OP_PONG


def compute_accept_key(sec_websocket_key: str) -> str:
    digest = hashlib.sha1((sec_websocket_key + WS_GUID).encode("ascii")).digest()
    return base64.b64encode(digest).decode("ascii")


def encode_frame(opcode: int, payload: bytes, *, mask: bool = False) -> bytes:
    """Encode a single WS frame."""
    if opcode not in (OP_TEXT, OP_BINARY, OP_PING, OP_PONG, OP_CLOSE):
        raise ValueError(f"unsupported opcode: {opcode:#x}")
    header = bytearray()
    header.append(0x80 | opcode)  # FIN=1
    length = len(payload)
    mask_bit = 0x80 if mask else 0x00
    if length < 126:
        header.append(mask_bit | length)
    elif length <= 0xFFFF:
        header.append(mask_bit | 126)
        header.extend(struct.pack(">H", length))
    else:
        header.append(mask_bit | 127)
        header.extend(struct.pack(">Q", length))
    if mask:
        mask_key = os.urandom(4)
        header.extend(mask_key)
        masked = bytes(b ^ mask_key[i % 4] for i, b in enumerate(payload))
        return bytes(header) + masked
    return bytes(header) + payload


def encode_text(payload: str) -> bytes:
    return encode_frame(OP_TEXT, payload.encode("utf-8"))


def encode_close(code: int = 1000, reason: str = "") -> bytes:
    body = struct.pack(">H", code)
    if reason:
        body += reason.encode("utf-8")
    return encode_frame(OP_CLOSE, body)


def encode_pong(payload: bytes) -> bytes:
    return encode_frame(OP_PONG, payload)


class FrameDecoder:
    """Incremental WS frame decoder.

    Feed it raw bytes via feed(); consume completed frames via pop().
    Two payload limits apply: text/ping/pong/close use `max_text_payload`,
    binary frames use `max_binary_payload`. This matches the API
    contract where text messages are capped at 16 KiB but binary wave
    patches (~17 KiB at the default 65x65 grid) need more headroom.
    """

    def __init__(self, max_text_payload: int = 16 * 1024,
                 max_binary_payload: int = 256 * 1024):
        self._buf = bytearray()
        self._max_text = max_text_payload
        self._max_binary = max_binary_payload

    def feed(self, data: bytes) -> None:
        self._buf.extend(data)

    def pop(self) -> Optional[WSFrame]:
        if len(self._buf) < 2:
            return None
        b1, b2 = self._buf[0], self._buf[1]
        fin = bool(b1 & 0x80)
        opcode = b1 & 0x0F
        masked = bool(b2 & 0x80)
        length = b2 & 0x7F
        idx = 2
        if length == 126:
            if len(self._buf) < idx + 2:
                return None
            length = struct.unpack(">H", bytes(self._buf[idx:idx + 2]))[0]
            idx += 2
        elif length == 127:
            if len(self._buf) < idx + 8:
                return None
            length = struct.unpack(">Q", bytes(self._buf[idx:idx + 8]))[0]
            idx += 8
        if length > self._max_text:
            # check whether this is a binary frame; allow larger payloads
            # for the wave patch channel.
            if (opcode == 0x2 and length > self._max_binary):
                raise ValueError(
                    f"binary frame payload {length} exceeds "
                    f"{self._max_binary}"
                )
            if opcode != 0x2 and length > self._max_text:
                raise ValueError(
                    f"frame payload {length} exceeds {self._max_text}"
                )
        if masked:
            if len(self._buf) < idx + 4:
                return None
            mask_key = bytes(self._buf[idx:idx + 4])
            idx += 4
        else:
            mask_key = None
        if len(self._buf) < idx + length:
            return None
        payload = bytes(self._buf[idx:idx + length])
        if mask_key is not None:
            payload = bytes(b ^ mask_key[i % 4] for i, b in enumerate(payload))
        del self._buf[:idx + length]
        if not fin:
            raise ValueError("fragmented frames are not supported")
        return WSFrame(opcode=opcode, payload=payload)
