"""Regression checks for commands received while physics is paused."""
import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from operator_training.server import OperatorServer
from operator_training.session import Lifecycle


class PausedCommandsTests(unittest.TestCase):
    def test_resume_and_abort_are_accepted_while_paused(self):
        async def run():
            for kind in ('resume', 'abort'):
                server = OperatorServer.__new__(OperatorServer)
                server._send_ack = AsyncMock()
                server._send_telemetry = AsyncMock()
                session = SimpleNamespace(
                    state=SimpleNamespace(lifecycle=Lifecycle.PAUSED),
                    tick_count=7, resume=Mock(return_value=True), abort=Mock(),
                )
                record = SimpleNamespace(session=session)
                await server._handle_event(record, {
                    'event_id': 'regression', 'kind': kind,
                    'payload': {'confirm': True},
                })
                self.assertTrue(server._send_ack.call_args.kwargs['accepted'])
                server._send_telemetry.assert_awaited_once_with(record)
                if kind == 'resume':
                    session.resume.assert_called_once_with(confirm=True)
                else:
                    session.abort.assert_called_once_with(reason='client_abort')
        asyncio.run(run())
