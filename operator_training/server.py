"""Same-origin, single-operator server. Never expose without access control."""
import asyncio
from contextlib import asynccontextmanager, suppress
import hashlib
import hmac
import json
import os
from pathlib import Path
import secrets
import time
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware
from operator_training.session import Session, LIMITS, ROOT, DT

LOG_ROOT = ROOT / 'var' / 'operator_sessions'
DIST = ROOT / 'web' / 'operator' / 'dist'
SECRET = secrets.token_bytes(32)


def issue_cookie():
    nonce = secrets.token_hex(24)
    return nonce + '.' + hmac.new(SECRET, nonce.encode(), hashlib.sha256).hexdigest()


def valid_cookie(value):
    if not value or '.' not in value:
        return False
    nonce, signature = value.split('.', 1)
    return hmac.compare_digest(signature, hmac.new(SECRET, nonce.encode(), hashlib.sha256).hexdigest())


def valid_origin(ws):
    # Explicit external origins may be supplied for a reverse proxy.
    allowed = {x.strip() for x in os.getenv('OPERATOR_ORIGINS', '').split(',') if x.strip()}
    host = ws.headers.get('host', '')
    expected = ('https' if ws.url.scheme == 'wss' else 'http') + '://' + host
    return ws.headers.get('origin') in allowed | {expected}


class Runtime:
    def __init__(self):
        self.session = None
        self.owner = None
        self.connected = False
        self.epoch = 0
        self.last_error = ''

    async def run(self):
        deadline = time.monotonic()
        while True:
            await asyncio.sleep(max(0, deadline-time.monotonic()))
            now = time.monotonic()
            s = self.session
            try:
                if s:
                    if now-deadline > 2*DT:
                        s.pause('SIM_OVERRUN')
                    s.advance(now)
            except Exception as exc:
                self.last_error = type(exc).__name__
                if s:
                    s.lifecycle, s.reason = 'PAUSED', 'SERVER_OR_RECORDING_ERROR'
            deadline = (now if now-deadline > 2*DT else deadline) + DT


runtime = Runtime()


@asynccontextmanager
async def lifespan(app):
    task = asyncio.create_task(runtime.run())
    yield
    task.cancel()
    with suppress(asyncio.CancelledError):
        await task
    if runtime.session:
        runtime.session.close()


app = FastAPI(lifespan=lifespan)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=os.getenv(
    'OPERATOR_HOSTS', 'localhost,127.0.0.1,*.e2b.app,testserver').split(','))


@app.middleware('http')
async def headers(request: Request, call_next):
    response = await call_next(request)
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['Referrer-Policy'] = 'same-origin'
    response.headers['Cache-Control'] = 'no-store'
    if not valid_cookie(request.cookies.get('operator_owner')):
        response.set_cookie('operator_owner', issue_cookie(), httponly=True,
                            secure=request.url.scheme == 'https', samesite='strict')
    return response


@app.get('/api/capabilities')
async def capabilities():
    return dict(version=1, limits=LIMITS, modes=['observe', 'hybrid', 'manual'],
                dt=DT, experimental=True, sea_states=[0, .3, .6])


@app.get('/api/summary')
async def summary(request: Request):
    if request.cookies.get('operator_owner') != runtime.owner or not runtime.session:
        return JSONResponse({'error': 'no owned session'}, status_code=403)
    return runtime.session.summary()


@app.websocket('/ws')
async def socket(ws: WebSocket):
    owner = ws.cookies.get('operator_owner')
    if not valid_origin(ws) or not valid_cookie(owner) or runtime.connected:
        await ws.close(code=1008)
        return
    if runtime.owner and owner != runtime.owner:
        await ws.close(code=1008)
        return
    await ws.accept()
    runtime.owner, runtime.connected = owner, True
    runtime.epoch += 1
    epoch = runtime.epoch
    if runtime.session:
        runtime.session.seq = -1
        runtime.session.matched = 0
        runtime.session.last_input = -1e30
    outgoing = asyncio.Queue(maxsize=64)

    async def send_loop():
        while True:
            s = runtime.session
            while not outgoing.empty():
                await asyncio.wait_for(ws.send_json(outgoing.get_nowait()), 1)
            if s:
                try:
                    state = s.snapshot()
                    state.update(epoch=epoch)
                    await asyncio.wait_for(ws.send_json(state), 1)
                except (ValueError, OSError):
                    s.pause('STATE_ERROR')
            await asyncio.sleep(DT)

    await ws.send_json(dict(type='hello', epoch=epoch, limits=LIMITS,
                           session_id=runtime.session.id if runtime.session else None,
                           config=runtime.session.config if runtime.session else None,
                           spectrum=runtime.session.spectrum() if runtime.session else []))
    sender = asyncio.create_task(send_loop())
    count, window = 0, time.monotonic()
    try:
        while True:
            raw = await asyncio.wait_for(ws.receive_text(), 2)
            now = time.monotonic()
            if now-window >= 1:
                count, window = 0, now
            count += 1
            if len(raw.encode()) > 16384 or count > 80 or sender.done():
                raise ValueError('message/rate limit or blocked sender')
            msg = json.loads(raw)
            if not isinstance(msg, dict) or type(msg.get('v')) is not int or msg.get('v') != 1 or type(msg.get('epoch')) is not int or msg['epoch'] != epoch:
                raise ValueError('invalid protocol/epoch')
            kind = msg.get('type')
            if kind == 'create':
                if runtime.session and runtime.session.lifecycle == 'RUNNING':
                    raise ValueError('pause before replacing a session')
                if runtime.session:
                    if not runtime.session.finished:
                        runtime.session.finish('ABORTED', 'REPLACED')
                    runtime.session.close()
                runtime.session = Session(msg.get('config'), now, LOG_ROOT)
                await outgoing.put(dict(type='created', session_id=runtime.session.id, epoch=epoch,
                                        config=runtime.session.config, spectrum=runtime.session.spectrum()))
            elif kind == 'heartbeat':
                if runtime.session:
                    runtime.session.last_heartbeat = now
                outgoing.put_nowait(dict(type='pong', client_time=msg.get('client_time')))
            elif runtime.session and msg.get('session_id') == runtime.session.id:
                s = runtime.session
                if kind == 'input':
                    s.accept_input(msg.get('seq'), msg.get('control'), now)
                elif kind == 'event':
                    outgoing.put_nowait(s.event(msg.get('event_id'), msg.get('action'), now))
                else:
                    raise ValueError('unknown message')
            else:
                raise ValueError('session mismatch')
    except (WebSocketDisconnect, asyncio.TimeoutError, ValueError, TypeError, OSError, asyncio.QueueFull):
        pass
    finally:
        sender.cancel()
        with suppress(asyncio.CancelledError, Exception):
            await sender
        if runtime.session:
            try:
                runtime.session.pause('CONNECTION_LOST')
            except OSError:
                pass
        runtime.connected = False
        with suppress(Exception):
            await ws.close()


@app.get('/')
async def index():
    if not (DIST / 'index.html').exists():
        return JSONResponse({'error': 'Build frontend: cd web/operator && npm ci && npm run build'}, status_code=503)
    return FileResponse(DIST / 'index.html')


if (DIST / 'assets').exists():
    app.mount('/assets', StaticFiles(directory=DIST / 'assets'), name='assets')
