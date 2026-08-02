import asyncio
import io
import msgpack
import signal
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pie_client import PieClient
from pie_client_cli import engine


class ClosedWebSocket:
    def __aiter__(self):
        return self

    async def __anext__(self):
        raise StopAsyncIteration


class RespondingWebSocket:
    def __init__(self, client):
        self.client = client
        self.sent = []

    async def send(self, encoded):
        message = msgpack.unpackb(encoded, raw=False)
        self.sent.append(message)
        future = self.client.pending_requests.get(message["corr_id"])
        if future is None:
            raise AssertionError("response future was not registered before send")
        if not future.done():
            future.set_result((True, "ok"))


class NeverEventProcess:
    process_id = "12345678-test-process"

    async def recv(self):
        await asyncio.Event().wait()


class PythonClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_upload_registers_pending_request_before_sending_chunks(self):
        client = PieClient("ws://example.invalid")
        client.ws = RespondingWebSocket(client)

        result = await client._upload_chunked(
            b"hello",
            {
                "type": "add_program",
                "program_hash": "hash",
                "manifest": "name = 'demo'",
                "force_overwrite": False,
            },
        )

        self.assertEqual(result, "ok")
        self.assertEqual(client.pending_requests, {})
        self.assertEqual(len(client.ws.sent), 1)

    async def test_listener_rejects_pending_requests_when_connection_ends(self):
        client = PieClient("ws://example.invalid")
        client.ws = ClosedWebSocket()
        future = asyncio.get_running_loop().create_future()
        client.pending_requests[1] = future

        await client._listen_to_server()

        self.assertEqual(client.pending_requests, {})
        with self.assertRaises(ConnectionError):
            await future

    async def test_stream_output_detaches_on_stdin_eof_from_monitor_thread(self):
        original_sigint = signal.getsignal(signal.SIGINT)
        with mock.patch("sys.stdin", io.StringIO("")):
            await asyncio.wait_for(
                engine._stream_inferlet_output_async(NeverEventProcess(), mock.Mock()),
                timeout=1,
            )
        self.assertEqual(signal.getsignal(signal.SIGINT), original_sigint)


if __name__ == "__main__":
    unittest.main()
