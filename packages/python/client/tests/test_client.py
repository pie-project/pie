import asyncio
import io
import msgpack
import signal
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pie_client import PieClient, inferlet, program_file
from pie_client_cli import engine


class ClosedWebSocket:
    def __aiter__(self):
        return self

    async def __anext__(self):
        raise StopAsyncIteration


class RespondingWebSocket:
    def __init__(self, client, result="ok"):
        self.client = client
        self.result = result
        self.sent = []

    async def send(self, encoded):
        message = msgpack.unpackb(encoded, raw=False)
        self.sent.append(message)
        future = self.client.pending_requests.get(message["corr_id"])
        if future is None:
            raise AssertionError("response future was not registered before send")
        if not future.done():
            future.set_result((True, self.result))


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
                "file": "demo.wasm",
                "version": None,
                "force_overwrite": False,
            },
        )

        self.assertEqual(result, "ok")
        self.assertEqual(client.pending_requests, {})
        self.assertEqual(len(client.ws.sent), 1)

    async def test_install_sends_the_file_name_and_version_and_returns_the_id(self):
        client = PieClient("ws://example.invalid")
        client.ws = RespondingWebSocket(client, result="demo@1.2.0")

        installed = await client.install_program_bytes(b"source", "demo.py", "1.2.0")

        self.assertEqual(installed, "demo@1.2.0")
        [frame] = client.ws.sent
        self.assertEqual(frame["type"], "add_program")
        self.assertEqual(frame["file"], "demo.py")
        self.assertEqual(frame["version"], "1.2.0")
        self.assertFalse(frame["force_overwrite"])
        self.assertNotIn("manifest", frame)

    def test_a_program_is_named_by_its_file_or_its_directory(self):
        self.assertEqual(program_file("/a/b/text_completion.wasm"), "text-completion.wasm")
        self.assertEqual(program_file("/a/beam_search_py/main.py"), "beam-search-py.py")
        self.assertEqual(program_file("/a/beam-search-js/index.mjs"), "beam-search-js.js")
        with self.assertRaises(ValueError):
            program_file("/a/b/notes.txt")

    async def test_listener_rejects_pending_requests_when_connection_ends(self):
        client = PieClient("ws://example.invalid")
        client.ws = ClosedWebSocket()
        future = asyncio.get_running_loop().create_future()
        client.pending_requests[1] = future

        await client._listen_to_server()

        self.assertEqual(client.pending_requests, {})
        with self.assertRaises(ConnectionError):
            await future

    async def test_listener_relays_the_reason_the_server_gave_before_closing(self):
        class RefusingWebSocket:
            def __init__(self):
                self.frames = iter(['{"type":"error","message":"admission rejected: cluster saturated"}'])

            def __aiter__(self):
                return self

            async def __anext__(self):
                try:
                    return next(self.frames)
                except StopIteration:
                    raise StopAsyncIteration

        client = PieClient("ws://example.invalid")
        client.ws = RefusingWebSocket()
        future = asyncio.get_running_loop().create_future()
        client.pending_requests[1] = future

        await client._listen_to_server()

        with self.assertRaises(ConnectionError) as raised:
            await future
        self.assertIn("admission rejected: cluster saturated", str(raised.exception))

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
