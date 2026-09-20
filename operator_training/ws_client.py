"""Async WebSocket client for testing the operator training server.

This client implements just enough of RFC 6455 to talk to OperatorServer
without pulling in the `websockets` library. It speaks text frames
only and parses incoming JSON into dicts.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import os
import secrets
import socket
import struct
import urllib.parse
from dataclasses import dataclass, field
from typing import Optional

from .ws_frames import (
    OP_CLOSE,
    OP_PING,
    OP_PONG,
    OP_TEXT,
    compute_accept_key,
    encode_text,
    FrameDecoder,
)


@dataclass
class WSClient:
    """Minimal asyncio WS client.

    Connect, send_text, recv_message. Single connection only.
    """

    host: str = "127.0.0.1"
    port: int = 8765
    path: str = "/"
    origin: Optional[str] = None
    extra_headers: dict = field(default_factory=dict)

    reader: Optional[asyncio.StreamReader] = field(default=None, init=False)
    writer: Optional[asyncio.StreamWriter] = field(default=None, init=False)
    decoder: FrameDecoder = field(
        default_factory=lambda: FrameDecoder(
            max_text_payload=16 * 1024, max_binary_payload=256 * 1024,
        ),
        init=False,
    )
    _connected: bool = field(default=False, init=False)

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.close()

    async def connect(self) -> None:
        sec_key = base64.b64encode(secrets.token_bytes(16)).decode("ascii")
        host_header = f"{self.host}:{self.port}"
        req_lines = [
            f"GET {self.path} HTTP/1.1",
            f"Host: {host_header}",
            "Upgrade: websocket",
            "Connection: Upgrade",
            f"Sec-WebSocket-Key: {sec_key}",
            "Sec-WebSocket-Version: 13",
        ]
        if self.origin:
            req_lines.append(f"Origin: {self.origin}")
        for k, v in self.extra_headers.items():
            req_lines.append(f"{k}: {v}")
        request = ("\r\n".join(req_lines) + "\r\n\r\n").encode("ascii")
        self.reader, self.writer = await asyncio.open_connection(self.host, self.port)
        self.writer.write(request)
        await self.writer.drain()

        status_line = await self.reader.readline()
        if not status_line.startswith(b"HTTP/1.1 101"):
            raise ConnectionError(f"bad ws upgrade: {status_line!r}")
        # Read headers until blank line.
        accept_seen = False
        while True:
            line = await self.reader.readline()
            if line in (b"\r\n", b"", b"\n"):
                break
            if line.lower().startswith(b"sec-websocket-accept:"):
                expected = compute_accept_key(sec_key).encode("ascii")
                got = line.split(b":", 1)[1].strip()
                if got != expected:
                    raise ConnectionError(f"bad accept: {got!r}")
                accept_seen = True
        if not accept_seen:
            raise ConnectionError("missing Sec-WebSocket-Accept")
        self._connected = True

    async def send_text(self, payload: str) -> None:
        if not self._connected:
            raise RuntimeError("not connected")
        self.writer.write(encode_text(payload))
        await self.writer.drain()

    async def send_close(self, code: int = 1000) -> None:
        from .ws_frames import encode_close
        body = struct.pack(">H", code)
        self.writer.write(encode_close(code))
        await self.writer.drain()

    async def recv_message(self, *, timeout: float = 5.0) -> Optional[str]:
        """Receive the next TEXT frame, silently skipping non-text ones."""
        if not self._connected:
            return None
        try:
            data = await asyncio.wait_for(self.reader.read(4096), timeout=timeout)
        except asyncio.TimeoutError:
            return None
        if not data:
            self._connected = False
            return None
        self.decoder.feed(data)
        # Drain control / binary frames; loop until we see a text frame
        # or the buffer is empty.
        while True:
            frame = self.decoder.pop()
            if frame is None:
                # Need more bytes to complete a frame.
                try:
                    data = await asyncio.wait_for(self.reader.read(4096),
                                                  timeout=timeout)
                except asyncio.TimeoutError:
                    return None
                if not data:
                    self._connected = False
                    return None
                self.decoder.feed(data)
                continue
            if frame.is_close:
                self._connected = False
                return None
            if frame.is_ping:
                from .ws_frames import encode_pong
                self.writer.write(encode_pong(frame.payload))
                await self.writer.drain()
                continue
            if frame.is_pong:
                continue
            if (frame.opcode & 0x0F) == 0x2:
                # Binary frame (wave patch). Drop it; callers that need
                # the payload should use recv_binary().
                continue
            if frame.is_text:
                return frame.payload.decode("utf-8")
            return None

    async def recv_binary(self, *, timeout: float = 5.0) -> Optional[bytes]:
        """Receive the next BINARY frame."""
        if not self._connected:
            return None
        try:
            data = await asyncio.wait_for(self.reader.read(4096), timeout=timeout)
        except asyncio.TimeoutError:
            return None
        if not data:
            self._connected = False
            return None
        self.decoder.feed(data)
        while True:
            frame = self.decoder.pop()
            if frame is None:
                try:
                    data = await asyncio.wait_for(self.reader.read(4096),
                                                  timeout=timeout)
                except asyncio.TimeoutError:
                    return None
                if not data:
                    self._connected = False
                    return None
                self.decoder.feed(data)
                continue
            if frame.is_close:
                self._connected = False
                return None
            if frame.is_ping:
                from .ws_frames import encode_pong
                self.writer.write(encode_pong(frame.payload))
                await self.writer.drain()
                continue
            if (frame.opcode & 0x0F) == 0x2:
                return frame.payload
            return None

    async def close(self) -> None:
        if not self._connected:
            return
        try:
            await self.send_close()
        except Exception:
            pass
        # Give the server a moment to drain in-flight messages
        # before we close the transport. asyncio.sleep(0) yields
        # to the event loop so the server's pending writes can land.
        try:
            await asyncio.sleep(0)
        except Exception:
            pass
        try:
            self.writer.close()
            await self.writer.wait_closed()
        except Exception:
            pass
        self._connected = False


async def http_get(host: str, port: int, path: str,
                   headers: Optional[dict] = None) -> tuple:
    reader, writer = await asyncio.open_connection(host, port)
    hdrs = headers or {}
    request = (f"GET {path} HTTP/1.1\r\n"
               f"Host: {host}:{port}\r\n"
               f"Connection: close\r\n"
               + "".join(f"{k}: {v}\r\n" for k, v in hdrs.items())
               + "\r\n").encode("ascii")
    writer.write(request)
    await writer.drain()
    status_line = await reader.readline()
    headers_data = {}
    while True:
        line = await reader.readline()
        if line in (b"\r\n", b"\n", b""):
            break
        k, _, v = line.decode("ascii").partition(":")
        headers_data[k.strip().lower()] = v.strip()
    body = await reader.read(int(headers_data.get("content-length", "0") or "0"))
    writer.close()
    await writer.wait_closed()
    return status_line.decode("ascii").rstrip(), headers_data, body


async def http_post(host: str, port: int, path: str, body: bytes,
                    headers: Optional[dict] = None) -> tuple:
    reader, writer = await asyncio.open_connection(host, port)
    hdrs = dict(headers or {})
    hdrs.setdefault("Content-Type", "application/json")
    hdrs.setdefault("Content-Length", str(len(body)))
    request = (f"POST {path} HTTP/1.1\r\n"
               f"Host: {host}:{port}\r\n"
               f"Connection: close\r\n"
               + "".join(f"{k}: {v}\r\n" for k, v in hdrs.items())
               + "\r\n").encode("ascii")
    writer.write(request + body)
    await writer.drain()
    status_line = await reader.readline()
    headers_data = {}
    while True:
        line = await reader.readline()
        if line in (b"\r\n", b"\n", b""):
            break
        k, _, v = line.decode("ascii").partition(":")
        headers_data[k.strip().lower()] = v.strip()
    resp_body = await reader.read(int(headers_data.get("content-length", "0") or "0"))
    writer.close()
    await writer.wait_closed()
    return status_line.decode("ascii").rstrip(), headers_data, resp_body
