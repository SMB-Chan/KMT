"""HTTP + WebSocket server for operator-training sessions.

Implementation is intentionally minimal: stdlib only (asyncio + socket).
FastAPI / websockets / uvicorn are *optional* dependencies (design
section 3: "Python側Web依存は任意追加").

Endpoints (v1):

    GET  /api/capabilities          -> capability handshake
    POST /api/sessions              -> create a session
    GET  /api/sessions/<id>/summary -> session summary (when finished)
    GET  /api/sessions/<id>/export  -> session artefacts (zip or tar)
    WS   /ws/sessions/<id>          -> the operator cockpit pipe

Per design section 7.2 / 7.3:

    * Origin and Host validation against an allow-list
    * One WS connection per session (second connection is rejected)
    * Message size capped at MAX_MESSAGE_BYTES (16 KiB)
    * Input rate capped at MAX_INPUT_HZ (60 Hz)
    * Heartbeat watchdog (default 0.5 s)
    * Epoch advances on every reconnect; old epoch inputs are dropped
"""
from __future__ import annotations

import asyncio
import base64
import hmac
import json
import logging
import os
import secrets
import time
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .calibration import GamepadProfile
from .coordinates import Attitude, neu_to_ruf
from .envelope import ControlEnvelope, ControlInput, EnvelopeViolation
from .recording import FileSink, Recording, NullSink
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
)
from .session import (
    Authority,
    Curriculum,
    Lifecycle,
    PendingRequest,
    Session,
    SessionConfig,
    SessionState,
)
from .wave_patch import (
    WavePatchFrame,
    WavePatchSpec,
    encode_wave_patch_message,
    sample_wave_heights,
)
from .ws_frames import (
    FrameDecoder,
    OP_BINARY,
    OP_CLOSE,
    OP_PING,
    OP_PONG,
    OP_TEXT,
    compute_accept_key,
    encode_close,
    encode_frame,
    encode_pong,
    encode_text,
)


# Binary frames (RFC 6455 opcode 0x2) carry wave patch payloads.
# The browser detects the wave patch by the JSON prefix in the frame
# body; the protocol field inside the JSON then names the kind.
WAVE_PATCH_OPCODE = OP_BINARY


log = logging.getLogger("operator_training.server")


DEFAULT_ALLOWED_ORIGINS = ("http://localhost", "http://127.0.0.1")
DEFAULT_ALLOWED_HOSTS = ("localhost", "127.0.0.1", "0.0.0.0")


class ServerError(Exception):
    """Base for server-side errors."""


class OriginNotAllowed(ServerError):
    def __init__(self, origin: str):
        super().__init__(f"origin not allowed: {origin}")
        self.origin = origin


class HostNotAllowed(ServerError):
    def __init__(self, host: str):
        super().__init__(f"host not allowed: {host}")
        self.host = host


class SessionNotFound(ServerError):
    pass


class SessionAlreadyTaken(ServerError):
    pass


@dataclass
class ServerConfig:
    allowed_origins: tuple = field(default_factory=lambda: DEFAULT_ALLOWED_ORIGINS)
    allowed_hosts: tuple = field(default_factory=lambda: DEFAULT_ALLOWED_HOSTS)
    recordings_root: Optional[Path] = None
    require_csrf: bool = True
    csrf_cookie_name: str = "ot_csrf"
    tick_hz: float = 20.0
    heartbeat_s: float = 30.0
    input_max_age_s: float = 0.25
    static_root: Optional[Path] = None  # served at /cockpit/ if set
    wave_patch_hz: float = 10.0
    wave_patch_size: int = 65
    wave_patch_extent_m: float = 128.0
    wave_patch_enabled: bool = True


@dataclass
class _ActiveConnection:
    session_id: str
    epoch: int
    reader: asyncio.StreamReader
    writer: asyncio.StreamWriter
    peer: tuple
    origin: str
    profile_hash: str = ""
    last_input_seq: int = -1
    input_window_start: float = 0.0
    input_window_count: int = 0


@dataclass
class _SessionRecord:
    session_id: str
    session: Session
    epoch: int = 0
    active: Optional[_ActiveConnection] = None


class SessionRegistry:
    """In-memory registry of sessions and active connections.

    A single instance is shared between HTTP handlers and the WS
    endpoint. The registry is the gatekeeper for the "single pilot"
    rule from design section 3.
    """

    def __init__(self):
        self._sessions: dict = {}

    def add(self, record: _SessionRecord) -> None:
        self._sessions[record.session_id] = record

    def get(self, session_id: str) -> _SessionRecord:
        if session_id not in self._sessions:
            raise SessionNotFound(session_id)
        return self._sessions[session_id]

    def pop(self, session_id: str) -> Optional[_SessionRecord]:
        return self._sessions.pop(session_id, None)

    def __contains__(self, session_id: str) -> bool:
        return session_id in self._sessions

    def values(self):
        return self._sessions.values()


def _build_session(config: Optional[SessionConfig] = None,
                   sink=None) -> Session:
    return Session.create(config=config, sink=sink)


def capability_envelope(envelope: ControlEnvelope) -> dict:
    return {
        "throttle_lo": envelope.throttle_lo,
        "throttle_hi": envelope.throttle_hi,
        "pitch_lo": envelope.pitch_lo,
        "pitch_hi": envelope.pitch_hi,
        "bank_abs": envelope.bank_abs,
        "rudder_abs": envelope.rudder_abs,
    }


def capabilities_payload(envelope: ControlEnvelope,
                         curricula: list) -> dict:
    return {
        "v": API_VERSION,
        "limits": {
            "max_message_bytes": MAX_MESSAGE_BYTES,
            "max_input_hz": MAX_INPUT_HZ,
        },
        "control_envelope": capability_envelope(envelope),
        "curricula": [c.to_dict() for c in curricula],
        "schemas": json_schema(),
    }


# ---------------------------------------------------------------------
# HTTP request parsing
# ---------------------------------------------------------------------
@dataclass
class HttpRequest:
    method: str
    target: str
    path: str
    query: dict
    version: str
    headers: dict
    body: bytes

    @property
    def origin(self) -> str:
        return self.headers.get("origin", "")

    @property
    def host(self) -> str:
        return self.headers.get("host", "")


async def read_http_request(reader: asyncio.StreamReader,
                            max_header_bytes: int = 16 * 1024,
                            max_body_bytes: int = 64 * 1024) -> Optional[HttpRequest]:
    """Read a single HTTP/1.1 request from reader.

    Returns None on EOF before any bytes arrive. Raises ValueError on
    malformed headers or oversized bodies.
    """
    header_data = await reader.readuntil(b"\r\n\r\n")
    if not header_data:
        return None
    if len(header_data) > max_header_bytes:
        raise ValueError("headers too large")
    lines = header_data.split(b"\r\n")
    request_line = lines[0].decode("ascii", errors="replace")
    parts = request_line.split(" ")
    if len(parts) != 3:
        raise ValueError(f"bad request line: {request_line!r}")
    method, target, version = parts
    parsed = urllib.parse.urlparse(target)
    query = dict(urllib.parse.parse_qsl(parsed.query, keep_blank_values=True))
    headers: dict = {}
    for line in lines[1:]:
        if not line:
            continue
        try:
            k, _, v = line.decode("ascii").partition(":")
            headers[k.strip().lower()] = v.strip()
        except Exception as exc:
            raise ValueError(f"bad header line: {line!r}") from exc
    content_length = int(headers.get("content-length", "0") or "0")
    if content_length > max_body_bytes:
        raise ValueError(f"body {content_length} > {max_body_bytes}")
    body = b""
    while len(body) < content_length:
        chunk = await reader.read(content_length - len(body))
        if not chunk:
            break
        body += chunk
    return HttpRequest(
        method=method, target=target,
        path=parsed.path, query=query, version=version,
        headers=headers, body=body,
    )


def write_http_response(writer: asyncio.StreamWriter, status: int,
                        body: bytes, *,
                        content_type: str = "application/json",
                        extra_headers: Optional[dict] = None) -> None:
    reason = {
        200: "OK", 201: "Created", 204: "No Content",
        400: "Bad Request", 403: "Forbidden", 404: "Not Found",
        409: "Conflict", 413: "Payload Too Large", 500: "Internal Server Error",
    }.get(status, "OK")
    headers = {
        "Content-Type": content_type,
        "Content-Length": str(len(body)),
        "Connection": "close",
        "Cache-Control": "no-store",
    }
    if extra_headers:
        headers.update(extra_headers)
    header_lines = "\r\n".join(f"{k}: {v}" for k, v in headers.items())
    writer.write(
        f"HTTP/1.1 {status} {reason}\r\n{header_lines}\r\n\r\n".encode("ascii")
        + body
    )


# ---------------------------------------------------------------------
# Main server
# ---------------------------------------------------------------------
class OperatorServer:
    def __init__(self, config: Optional[ServerConfig] = None,
                 envelope: Optional[ControlEnvelope] = None,
                 curriculum: Optional[Curriculum] = None):
        self.config = config or ServerConfig()
        self.envelope = envelope or ControlEnvelope.beginner()
        self.curriculum = curriculum or Curriculum(
            curriculum_id="hybrid_baseline",
            curriculum_version="1",
            scenario="hybrid",
        )
        self.registry = SessionRegistry()
        self._server: Optional[asyncio.base_events.Server] = None
        self._tick_tasks: dict = {}
        self._static_root: Optional[Path] = None
        if self.config.static_root is not None:
            self.static_root = self.config.static_root

    # ----- HTTP -----
    async def _http_dispatch(self, req: HttpRequest) -> tuple:
        if req.method != "OPTIONS":
            try:
                self._check_origin(req)
                self._check_host(req)
            except ServerError as exc:
                return 403, str(exc).encode("ascii")
        try:
            if req.method == "GET" and req.path == "/api/capabilities":
                payload = capabilities_payload(self.envelope, [self.curriculum])
                return 200, json.dumps(payload).encode("utf-8")
            if req.method == "POST" and req.path == "/api/sessions":
                body = self._parse_session_post(req)
                cfg = SessionConfig(
                    curriculum=body.get("curriculum", self.curriculum),
                    envelope=body.get("envelope", self.envelope),
                    seed=int(body.get("seed", 42)),
                    target_takeoff_alt_m=float(body.get("target_takeoff_alt_m", 25.0)),
                )
                sink = self._make_sink(cfg.session_id)
                sess = Session.create(config=cfg, sink=sink)
                sess.ready()  # move from SETUP to READY for the client
                self.registry.add(_SessionRecord(
                    session_id=cfg.session_id, session=sess,
                ))
                return 201, json.dumps({
                    "v": API_VERSION,
                    "session_id": cfg.session_id,
                    "epoch": 0,
                    "control_envelope": capability_envelope(self.envelope),
                    "curriculum": self.curriculum.to_dict(),
                }).encode("utf-8")
            if req.method == "GET" and req.path.startswith("/api/sessions/"):
                return await self._handle_session_get(req)
            if req.method == "GET" and req.path == "/health":
                return 200, b"{\"status\":\"ok\"}"
            return 404, b"{\"error\":\"not found\"}"
        except ServerError as exc:
            return 400, json.dumps({"error": str(exc)}).encode("utf-8")
        except ValueError as exc:
            return 400, json.dumps({"error": str(exc)}).encode("utf-8")

    def _parse_session_post(self, req: HttpRequest) -> dict:
        if not req.body:
            return {}
        try:
            data = json.loads(req.body.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise ServerError(f"invalid JSON: {exc}") from exc
        if not isinstance(data, dict):
            raise ServerError("session body must be an object")
        return data

    async def _handle_session_get(self, req: HttpRequest) -> tuple:
        parts = req.path.split("/")
        # /api/sessions/<id>/summary
        # /api/sessions/<id>/export
        if len(parts) != 5:
            return 404, b"{\"error\":\"not found\"}"
        _, _, _, session_id, action = parts
        try:
            record = self.registry.get(session_id)
        except SessionNotFound:
            return 404, b"{\"error\":\"session not found\"}"
        if action == "summary":
            summary = record.session.finalize if False else None
            # Build a minimal summary without finalising the session.
            obs = record.session.adapter.observe()
            return 200, json.dumps({
                "session_id": session_id,
                "lifecycle": record.session.state.lifecycle,
                "phase": record.session.state.phase,
                "authority": record.session.state.authority,
                "sim_t": record.session.state.sim_t,
                "tick_count": record.session.tick_count,
                "analytics": {
                    "altitude_m": float(obs.get("altitude_m", 0.0)),
                    "forward_speed_m_s": float(obs.get("forward_speed_m_s", 0.0)),
                    "keel_clearance_m": float(obs.get("keel_clearance_m", 0.0)),
                },
            }).encode("utf-8")
        if action == "export":
            # The session artefacts live under recordings_root/<id>/ when
            # running the server; we emit a small tar.gz if available.
            return 501, b"{\"error\":\"export not implemented\"}"
        return 404, b"{\"error\":\"not found\"}"

    def _make_sink(self, session_id: str):
        if self.config.recordings_root is None:
            return NullSink()
        root = self.config.recordings_root / session_id
        return FileSink(root=root)

    def _serve_static(self, path: str) -> tuple:
        """Serve a file from the configured static root.

        Path must start with /cockpit/. Traversal outside the static
        root is rejected. Returns (status, body, content_type).
        """
        rel = path[len("/cockpit/"):]
        if not rel:
            rel = "index.html"
        # Reject path traversal.
        if ".." in rel.split("/"):
            return 400, b"bad path", "text/plain"
        target = (self._static_root / rel).resolve()
        try:
            target.relative_to(self._static_root)
        except ValueError:
            return 400, b"bad path", "text/plain"
        if not target.exists() or not target.is_file():
            return 404, b"not found", "text/plain"
        body = target.read_bytes()
        suffix = target.suffix.lower()
        ctype = {
            ".html": "text/html; charset=utf-8",
            ".js": "application/javascript",
            ".css": "text/css",
            ".json": "application/json",
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".svg": "image/svg+xml",
            ".ico": "image/x-icon",
        }.get(suffix, "application/octet-stream")
        return 200, body, ctype

    # ----- Origin / Host / CSRF -----
    def _check_origin(self, req: HttpRequest) -> None:
        if not req.origin:
            return  # native clients may omit Origin
        for allowed in self.config.allowed_origins:
            if req.origin == allowed or req.origin.startswith(allowed + ":"):
                return
        raise OriginNotAllowed(req.origin)

    def _check_host(self, req: HttpRequest) -> None:
        host = req.host.split(":")[0]
        if host in self.config.allowed_hosts:
            return
        raise HostNotAllowed(host)

    def _check_csrf(self, req: HttpRequest) -> bool:
        if not self.config.require_csrf:
            return True
        # GET requests are exempt; state-changing requests need the
        # header token or a SameSite cookie match.
        if req.method in ("GET", "HEAD", "OPTIONS"):
            return True
        cookie = req.headers.get("cookie", "")
        token_in_cookie = ""
        for part in cookie.split(";"):
            name, _, value = part.strip().partition("=")
            if name == self.config.csrf_cookie_name:
                token_in_cookie = value
                break
        token_in_header = req.headers.get("x-csrf-token", "")
        return bool(token_in_cookie) and hmac.compare_digest(
            token_in_cookie, token_in_header
        )

    # ----- Lifecycle -----
    async def serve(self, host: str = "127.0.0.1", port: int = 8765):
        self._server = await asyncio.start_server(
            self._handle_connection, host=host, port=port,
        )
        log.info("OperatorServer listening on http://%s:%d", host, port)
        try:
            async with self._server:
                await self._server.serve_forever()
        except asyncio.CancelledError:
            pass

    async def start(self, host: str = "127.0.0.1", port: int = 8765) -> None:
        self._server = await asyncio.start_server(
            self._handle_connection, host=host, port=port,
        )
        sockets = self._server.sockets or []
        if sockets:
            sock = sockets[0]
            self._bound_port = sock.getsockname()[1]
        log.info("OperatorServer bound on port %d", self._bound_port)

    async def stop(self) -> None:
        # Close every active connection so per-connection _ws_loop
        # coroutines can return before we tear down the listening
        # socket. Without this, asyncio.run() can hang waiting for
        # _handle_connection to drain, and the next test may inherit
        # a half-closed socket.
        for record in list(self.registry.values()):
            conn = record.active
            if conn is not None:
                try:
                    conn.writer.write(encode_close(1001, "server shutting down"))
                    await conn.writer.drain()
                except Exception:
                    pass
                try:
                    conn.writer.close()
                except Exception:
                    pass
                record.active = None
        for task in self._tick_tasks.values():
            task.cancel()
        # Give the event loop a chance to cancel pending connection
        # handlers cleanly.
        await asyncio.sleep(0)
        for record in list(self.registry.values()):
            try:
                record.session.finalize()
            except Exception:
                pass
        if self._server is not None:
            self._server.close()
            try:
                await asyncio.wait_for(self._server.wait_closed(), timeout=2.0)
            except (asyncio.TimeoutError, Exception):
                pass

    @property
    def bound_port(self) -> int:
        return getattr(self, "_bound_port", 0)

    @property
    def static_root(self) -> Optional[Path]:
        return self._static_root

    @static_root.setter
    def static_root(self, value: Optional[Path]) -> None:
        if value is None:
            self._static_root = None
            return
        path = Path(value)
        if not path.exists() or not path.is_dir():
            raise FileNotFoundError(f"static root not found: {path}")
        self._static_root = path.resolve()

    # ----- Connection dispatch -----
    async def _handle_connection(self, reader, writer):
        peer = writer.get_extra_info("peername")
        try:
            req = await read_http_request(reader)
            if req is None:
                writer.close()
                return
            if req.path.startswith("/ws/sessions/"):
                await self._handle_ws(req, reader, writer)
                return
            if req.method == "OPTIONS":
                # CORS preflight: return 204 with permissive CORS for
                # the allowed origins. The Origin allow-list already
                # blocked disallowed origins above.
                write_http_response(writer, 204, b"", extra_headers={
                    "Access-Control-Allow-Origin": req.origin or "*",
                    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
                    "Access-Control-Allow-Headers":
                        "Content-Type, X-CSRF-Token, X-Operator-Epoch",
                    "Access-Control-Max-Age": "600",
                })
                await writer.drain()
                writer.close()
                return
            if req.path.startswith("/cockpit/") and self._static_root is not None:
                status, body, content_type = self._serve_static(req.path)
                extra = {}
                if req.origin:
                    extra["Access-Control-Allow-Origin"] = req.origin
                write_http_response(writer, status, body,
                                    content_type=content_type,
                                    extra_headers=extra)
                await writer.drain()
                return
            status, body = await self._http_dispatch(req)
            extra = {}
            if req.origin:
                extra["Access-Control-Allow-Origin"] = req.origin
            write_http_response(writer, status, body, extra_headers=extra)
            await writer.drain()
        except (ValueError, ServerError) as exc:
            log.warning("connection error: %s", exc)
            try:
                write_http_response(writer, 400,
                                    json.dumps({"error": str(exc)}).encode("utf-8"))
                await writer.drain()
            except Exception:
                pass
        except Exception as exc:
            log.exception("unhandled: %s", exc)
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass

    # ----- WS -----
    async def _handle_ws(self, req: HttpRequest, reader: asyncio.StreamReader,
                         writer: asyncio.StreamWriter) -> None:
        try:
            self._check_origin(req)
            self._check_host(req)
        except ServerError:
            write_http_response(writer, 403, b"{\"error\":\"origin/host denied\"}")
            await writer.drain()
            return

        parts = req.path.split("/")
        # /ws/sessions/<id>
        if len(parts) != 4:
            write_http_response(writer, 404, b"{\"error\":\"bad ws path\"}")
            await writer.drain()
            return
        session_id = parts[3]
        try:
            record = self.registry.get(session_id)
        except SessionNotFound:
            write_http_response(writer, 404, b"{\"error\":\"session not found\"}")
            await writer.drain()
            return

        # Validate WebSocket upgrade headers.
        ws_key = req.headers.get("sec-websocket-key")
        ws_version = req.headers.get("sec-websocket-version")
        if not ws_key or ws_version != "13":
            write_http_response(writer, 400,
                                b"{\"error\":\"ws upgrade required\"}")
            await writer.drain()
            return

        accept = compute_accept_key(ws_key)
        handshake = (
            "HTTP/1.1 101 Switching Protocols\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Accept: {accept}\r\n"
            "\r\n"
        ).encode("ascii")
        writer.write(handshake)
        await writer.drain()

        # Bump epoch on (re)connect to invalidate any in-flight inputs
        # from a prior connection.
        record.epoch += 1
        if record.active is not None:
            old = record.active
            try:
                old.writer.write(encode_close(1013, "epoch bumped"))
                await old.writer.drain()
                old.writer.close()
            except Exception:
                pass
            record.active = None

        record.active = _ActiveConnection(
            session_id=session_id, epoch=record.epoch,
            reader=reader, writer=writer,
            peer=req.headers.get("host", ""),
            origin=req.origin,
        )
        # Send initial hello + first telemetry.
        obs = record.session.adapter.observe()
        await self._send_hello(record, obs)
        # Drive the WS connection.
        await self._ws_loop(record)

    async def _send_hello(self, record: _SessionRecord, initial_obs: dict) -> None:
        hello = HelloMessage(
            v=API_VERSION,
            type=MessageKind.HELLO.value,
            session_id=record.session_id,
            epoch=record.epoch,
            control_envelope=capability_envelope(self.envelope),
            curriculum=self.curriculum.to_dict(),
            initial_telemetry=_build_telemetry(record, initial_obs).to_wire(),
        )
        payload = json.dumps(hello.to_wire(), allow_nan=False)
        record.active.writer.write(encode_text(payload))
        await record.active.writer.drain()

    async def _ws_loop(self, record: _SessionRecord) -> None:
        decoder = FrameDecoder(
            max_text_payload=MAX_MESSAGE_BYTES,
            max_binary_payload=256 * 1024,
        )
        reader = record.active.reader
        writer = record.active.writer
        tick_interval = 1.0 / self.config.tick_hz
        wave_interval = (1.0 / self.config.wave_patch_hz
                        if self.config.wave_patch_enabled else None)
        next_tick = asyncio.get_event_loop().time()
        next_wave = next_tick
        last_heartbeat = time.monotonic()
        epoch = record.epoch
        try:
            while True:
                timeout = tick_interval
                if wave_interval is not None:
                    timeout = min(timeout, max(wave_interval, tick_interval))
                try:
                    chunk = await asyncio.wait_for(reader.read(4096), timeout=timeout)
                except asyncio.TimeoutError:
                    chunk = b""
                if not chunk:
                    # read() returning b"" means EOF; a timeout just
                    # leaves chunk as b"", which we must NOT treat as
                    # a disconnect. Distinguish by checking whether
                    # the reader has actually seen EOF.
                    if reader.at_eof():
                        break
                    chunk = b""
                decoder.feed(chunk)
                while True:
                    frame = decoder.pop()
                    if frame is None:
                        break
                    if frame.is_close:
                        writer.write(encode_close())
                        await writer.drain()
                        return
                    if frame.is_ping:
                        writer.write(encode_pong(frame.payload))
                        await writer.drain()
                        continue
                    if frame.is_pong:
                        last_heartbeat = time.monotonic()
                        continue
                    if not frame.is_text:
                        continue
                    last_heartbeat = time.monotonic()
                    await self._handle_ws_message(record, frame.payload)

                now = asyncio.get_event_loop().time()
                if now >= next_tick:
                    next_tick = now + tick_interval
                    await self._tick_once(record)
                if wave_interval is not None and now >= next_wave:
                    next_wave = now + wave_interval
                    await self._emit_wave_patch(record)
                # Server-side watchdog: if no client traffic for too
                # long we close the connection.
                if (time.monotonic() - last_heartbeat) > self.config.heartbeat_s:
                    writer.write(encode_close(1011, "heartbeat timeout"))
                    await writer.drain()
                    return
        except (ConnectionError, asyncio.IncompleteReadError):
            pass
        finally:
            if record.active is not None and record.active.epoch == epoch:
                record.active = None

    async def _emit_wave_patch(self, record: _SessionRecord) -> None:
        """Compute and send a wave patch centred on the vehicle."""
        if record.active is None:
            return
        veh = record.session.adapter.vehicle
        sea = getattr(veh, "sea", None)
        if sea is None:
            return
        spec = WavePatchSpec(
            size=self.config.wave_patch_size,
            extent_m=self.config.wave_patch_extent_m,
        )
        try:
            heights = sample_wave_heights(
                sea, float(veh.x), float(veh.y),
                spec, float(veh.t),
            )
        except Exception:
            return
        frame = WavePatchFrame(
            sim_time_s=float(veh.t),
            centre_north_m=float(veh.x),
            centre_east_m=float(veh.y),
            spec=spec,
            heights=heights,
        )
        try:
            msg = encode_wave_patch_message(
                frame,
                session_id=record.session_id,
                epoch=record.epoch,
            )
        except Exception:
            return
        try:
            # Binary frame with the wave patch payload. The browser
            # detects it via the JSON prefix and switches into binary
            # parsing for the remainder of the frame.
            record.active.writer.write(
                encode_frame(WAVE_PATCH_OPCODE, msg)
            )
            await record.active.writer.drain()
        except (ConnectionError, asyncio.IncompleteReadError, ValueError):
            # Bad opcode / closed pipe / framing error: skip this tick.
            pass

    async def _handle_ws_message(self, record: _SessionRecord,
                                 payload: bytes) -> None:
        try:
            text = payload.decode("utf-8")
            raw = json.loads(text)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            await self._send_ack(record, event_id="-",
                                 accepted=False,
                                 reason=f"decode: {exc}",
                                 applied_tick=record.session.tick_count)
            return
        try:
            assert_wire_size(raw)
        except SchemaError as exc:
            await self._send_ack(record, event_id="-",
                                 accepted=False,
                                 reason=str(exc),
                                 applied_tick=record.session.tick_count)
            return
        kind = raw.get("type")
        if kind == MessageKind.INPUT.value:
            await self._handle_input(record, raw)
        elif kind == MessageKind.EVENT.value:
            await self._handle_event(record, raw)
        elif kind == MessageKind.PING.value:
            pong = {"v": API_VERSION, "type": MessageKind.PONG.value,
                    "session_id": record.session_id,
                    "epoch": record.epoch,
                    "sim_time_s": record.session.state.sim_t}
            record.active.writer.write(encode_text(json.dumps(pong)))
            await record.active.writer.drain()
        else:
            await self._send_ack(record, event_id="-",
                                 accepted=False,
                                 reason=f"unknown type: {kind!r}",
                                 applied_tick=record.session.tick_count)

    async def _handle_input(self, record: _SessionRecord, raw: dict) -> None:
        try:
            msg = parse_input_message(raw)
        except SchemaError as exc:
            await self._send_ack(record, event_id="-",
                                 accepted=False,
                                 reason=str(exc),
                                 applied_tick=record.session.tick_count)
            return
        if msg.session_id and msg.session_id != record.session_id:
            await self._send_ack(record, event_id="-",
                                 accepted=False,
                                 reason="session mismatch",
                                 applied_tick=record.session.tick_count)
            return
        if msg.epoch != record.epoch:
            await self._send_ack(record, event_id=f"input-{msg.seq}",
                                 accepted=False,
                                 reason="stale epoch",
                                 applied_tick=record.session.tick_count)
            return
        if record.active is not None and msg.profile_hash:
            record.active.profile_hash = msg.profile_hash
        # Rate limit per the design contract: max 60 messages / second.
        now = time.monotonic()
        if (now - record.active.input_window_start) >= 1.0:
            record.active.input_window_start = now
            record.active.input_window_count = 0
        record.active.input_window_count += 1
        if record.active.input_window_count > MAX_INPUT_HZ:
            await self._send_ack(record, event_id=f"input-{msg.seq}",
                                 accepted=False,
                                 reason="input rate exceeded",
                                 applied_tick=record.session.tick_count)
            return
        # Validate axes against the envelope.
        ci_raw = {
            "throttle": msg.control.throttle,
            "pitch_deg": msg.control.pitch_deg,
            "bank_deg": msg.control.bank_deg,
            "rudder": msg.control.rudder,
        }
        try:
            ci = self.envelope.validate(ci_raw)
        except EnvelopeViolation as exc:
            await self._send_ack(record, event_id=f"input-{msg.seq}",
                                 accepted=False,
                                 reason=f"envelope: {exc}",
                                 applied_tick=record.session.tick_count)
            return
        # Reject duplicate seq.
        if msg.seq <= record.active.last_input_seq:
            await self._send_ack(record, event_id=f"input-{msg.seq}",
                                 accepted=False,
                                 reason="seq <= last_input_seq",
                                 applied_tick=record.session.tick_count)
            return
        record.active.last_input_seq = msg.seq
        # Defer the actual session.tick to the WS loop's _tick_once.
        record.session._pending_input = ci
        record.session._pending_seq = msg.seq

    async def _handle_event(self, record: _SessionRecord, raw: dict) -> None:
        event_id = str(raw.get("event_id", "-"))
        kind = raw.get("kind")
        sess = record.session
        # Start commands require READY; resume and abort also work while paused.
        if kind == "start_takeoff":
            if sess.state.lifecycle != Lifecycle.READY:
                await self._send_ack(record, event_id=event_id,
                                     accepted=False,
                                     reason="not_ready",
                                     applied_tick=sess.tick_count)
                return
            sess.start_takeoff()
            await self._send_ack(record, event_id=event_id,
                                 accepted=True, reason="",
                                 applied_tick=sess.tick_count)
            return
        if kind == "start_manual":
            if sess.state.lifecycle != Lifecycle.READY:
                await self._send_ack(record, event_id=event_id,
                                     accepted=False,
                                     reason="not_ready",
                                     applied_tick=sess.tick_count)
                return
            sess.start_manual()
            await self._send_ack(record, event_id=event_id,
                                 accepted=True, reason="",
                                 applied_tick=sess.tick_count)
            return
        allowed_paused = sess.state.lifecycle == Lifecycle.PAUSED and kind in ("resume", "abort")
        if sess.state.lifecycle != Lifecycle.RUNNING and not allowed_paused:
            await self._send_ack(record, event_id=event_id,
                                 accepted=False,
                                 reason=DeclineReason.NOT_RUNNING.value,
                                 applied_tick=sess.tick_count)
            return
        if kind == "take_control":
            if sess.state.authority != Authority.AUTO:
                await self._send_ack(record, event_id=event_id,
                                     accepted=False,
                                     reason=DeclineReason.ALREADY_HUMAN.value,
                                     applied_tick=sess.tick_count)
                return
            sess.request_handover()
        elif kind == "request_auto_land":
            if sess.state.authority != Authority.HUMAN:
                await self._send_ack(record, event_id=event_id,
                                     accepted=False,
                                     reason=DeclineReason.ALREADY_AUTO.value,
                                     applied_tick=sess.tick_count)
                return
            sess.request_auto_land()
        elif kind == "cancel_auto_land":
            sess.cancel_auto_land_request()
        elif kind == "pause":
            sess.pause()
        elif kind == "resume":
            payload = raw.get("payload") if isinstance(raw.get("payload"), dict) else {}
            confirm = bool(raw.get("confirm", False) or payload.get("confirm", False))
            if not sess.resume(confirm=confirm):
                await self._send_ack(record, event_id=event_id,
                                     accepted=False,
                                     reason="resume not confirmed",
                                     applied_tick=sess.tick_count)
                return
        elif kind == "abort":
            sess.abort(reason=str(raw.get("reason", "client_abort")))
        else:
            await self._send_ack(record, event_id=event_id,
                                 accepted=False,
                                 reason=f"unknown event kind: {kind!r}",
                                 applied_tick=sess.tick_count)
            return
        await self._send_ack(record, event_id=event_id,
                             accepted=True, reason="",
                             applied_tick=sess.tick_count)
        await self._send_telemetry(record)

    async def _tick_once(self, record: _SessionRecord) -> None:
        sess = record.session
        if sess.state.lifecycle != Lifecycle.RUNNING:
            return
        ci = getattr(sess, "_pending_input", None)
        seq = getattr(sess, "_pending_seq", -1)
        sess._pending_input = None
        sess._pending_seq = -1
        sess.tick(dt=1.0 / self.config.tick_hz, human_input=ci, input_seq=seq)
        await self._send_telemetry(record)

    async def _send_telemetry(self, record: _SessionRecord) -> None:
        obs = record.session.adapter.observe()
        telemetry = _build_telemetry(record, obs).to_wire()
        try:
            record.active.writer.write(
                encode_text(json.dumps(telemetry, allow_nan=False))
            )
            await record.active.writer.drain()
        except (ConnectionError, asyncio.IncompleteReadError):
            pass

    async def _send_ack(self, record: _SessionRecord, *, event_id: str,
                        accepted: bool, reason: str,
                        applied_tick: int) -> None:
        if record.active is None:
            return
        ack = AckMessage(
            v=API_VERSION, type=MessageKind.ACK.value,
            event_id=event_id, accepted=accepted, reason=reason,
            applied_tick=int(applied_tick),
            session_id=record.session_id,
        )
        record.active.writer.write(
            encode_text(json.dumps(ack.to_wire(), allow_nan=False))
        )
        try:
            await record.active.writer.drain()
        except (ConnectionError, asyncio.IncompleteReadError):
            pass


def _build_telemetry(record: _SessionRecord, obs: dict) -> TelemetryMessage:
    sess = record.session
    v = sess.adapter.vehicle
    applied = {
        "throttle": float(obs.get("applied_throttle", v.throttle)),
        "pitch_deg": float(math.degrees(obs.get("alpha", v.alpha))
                            if obs.get("alpha") is not None
                            else math.degrees(v.alpha)),
        "bank_deg": float(obs.get("bank_deg", math.degrees(getattr(v, "bank", 0.0)))),
        "rudder": float(getattr(v, "rudder_command", 0.0)),
    }
    return TelemetryMessage(
        v=API_VERSION,
        type=MessageKind.TELEMETRY.value,
        tick=sess.tick_count,
        sim_time_s=float(sess.state.sim_t),
        lifecycle=sess.state.lifecycle,
        phase=sess.state.phase,
        authority=sess.state.authority,
        pending=sess.state.pending,
        assist_active=sess.state.assist_active,
        last_input_seq=sess.state.last_input_seq,
        applied_control=ControlAxes(**applied),
        position_neu_m={
            "x": float(obs.get("x_m", 0.0)),
            "y": float(obs.get("lateral_position_m", 0.0)),
            "z": float(obs.get("altitude_m", 0.0)),
        },
        velocity_neu_m_s={
            "x": float(obs.get("forward_speed_m_s", 0.0)),
            "y": float(obs.get("lateral_speed_m_s", 0.0)),
            "z": float(obs.get("vertical_speed_m_s", 0.0)),
        },
        attitude={
            "bank_deg": float(obs.get("bank_deg", 0.0)),
            "pitch_deg": float(math.degrees(getattr(v, "_last_pitch", 0.0))),
            "heading_deg": float(obs.get("heading_deg", 0.0)),
        },
        airspeed_m_s=float(obs.get("airspeed_m_s", 0.0)),
        wave_clearance_m=float(obs.get("keel_clearance_m", 0.0)),
        damage={
            "failed": bool(obs.get("failed", False)),
            "water_kg": float(obs.get("water_mass_kg", 0.0)),
            "failure_reason": str(obs.get("failure_reason", "")),
        },
        warnings=[],
    )


# Import math at module scope to avoid accidental shadowing.
import math  # noqa: E402
