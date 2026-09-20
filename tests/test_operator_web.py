"""Optional transport tests: install requirements-operator.txt and httpx."""
import tempfile
import unittest
try:
    from fastapi.testclient import TestClient
    from starlette.websockets import WebSocketDisconnect
    import operator_training.server as server
    AVAILABLE = True
except ImportError:
    AVAILABLE = False


@unittest.skipUnless(AVAILABLE, 'optional FastAPI/httpx dependencies not installed')
class OperatorWebTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.old_root = server.LOG_ROOT
        server.LOG_ROOT = self.tmp.name
        server.runtime = server.Runtime()
        self.client = TestClient(server.app)
        self.client.__enter__()
        self.client.get('/api/capabilities')

    def tearDown(self):
        self.client.__exit__(None, None, None)
        server.LOG_ROOT = self.old_root
        self.tmp.cleanup()

    def connect(self):
        return self.client.websocket_connect('/ws', headers={'origin': 'http://testserver'})

    def receive_type(self, ws, kind):
        for _ in range(100):
            m = ws.receive_json()
            if m['type'] == kind:
                return m
        self.fail('message not received')

    def test_cross_origin_rejected(self):
        with self.assertRaises(WebSocketDisconnect):
            with self.client.websocket_connect('/ws', headers={'origin': 'https://evil.example'}):
                pass

    def test_cookie_required(self):
        self.client.cookies.clear()
        with self.assertRaises(WebSocketDisconnect):
            with self.connect():
                pass

    def test_control_and_reconnect_epoch(self):
        with self.connect() as ws:
            hello = ws.receive_json()
            epoch = hello['epoch']
            ws.send_json(dict(v=1, epoch=epoch, type='create', config=dict(mode='manual', exercise='cruise', hs=0)))
            created = self.receive_type(ws, 'created')
            sid = created['session_id']
            ws.send_json(dict(v=1, epoch=epoch, session_id=sid, type='input', seq=0,
                control=dict(throttle=.5, pitch_deg=2, bank_deg=0, rudder=0)))
            ws.send_json(dict(v=1, epoch=epoch, session_id=sid, type='event', event_id='start', action='start'))
            self.assertTrue(self.receive_type(ws, 'ack')['accepted'])
            with self.assertRaises(WebSocketDisconnect):
                with self.connect():
                    pass
        self.assertEqual(server.runtime.session.lifecycle, 'PAUSED')
        with self.connect() as ws:
            hello = ws.receive_json()
            self.assertGreater(hello['epoch'], epoch)
            self.assertEqual(hello['session_id'], sid)
            self.assertEqual(server.runtime.session.lifecycle, 'PAUSED')
            self.assertEqual(server.runtime.session.seq, -1)

    def test_invalid_input_disconnects_and_pauses(self):
        with self.connect() as ws:
            epoch = ws.receive_json()['epoch']
            ws.send_json(dict(v=1,epoch=epoch,type='create',config={}))
            sid = self.receive_type(ws,'created')['session_id']
            ws.send_json(dict(v=1,epoch=epoch,type='input',session_id=sid,seq=0,control={'throttle': True}))
            with self.assertRaises(WebSocketDisconnect):
                while True:
                    ws.receive_json()

    def test_summary_owned_and_http_security_headers(self):
        response = self.client.get('/api/capabilities')
        self.assertEqual(response.headers['x-content-type-options'],'nosniff')
        self.assertEqual(self.client.get('/api/summary').status_code,403)
        with self.connect() as ws:
            epoch = ws.receive_json()['epoch']
            ws.send_json(dict(v=1,epoch=epoch,type='create',config={}))
            self.receive_type(ws,'created')
            self.assertEqual(self.client.get('/api/summary').status_code,200)
            self.client.cookies.clear()
            self.assertEqual(self.client.get('/api/summary').status_code,403)
