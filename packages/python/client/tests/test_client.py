import asyncio
import io
import blake3
import msgpack
import signal
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pie_client import PieClient, inferlet, program_file
from pie_client.client import Process
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
            },
        )

        self.assertEqual(result, "ok")
        self.assertEqual(client.pending_requests, {})
        self.assertEqual(len(client.ws.sent), 1)

    async def test_install_sends_the_file_name_and_hash_and_returns_the_build(self):
        pinned = "demo@" + blake3.blake3(b"source").hexdigest()
        client = PieClient("ws://example.invalid")
        client.ws = RespondingWebSocket(client, result=pinned)

        installed = await client.install_program_bytes(b"source", "demo.py")

        self.assertEqual(installed, pinned)
        [frame] = client.ws.sent
        self.assertEqual(frame["type"], "add_program")
        self.assertEqual(frame["file"], "demo.py")
        self.assertEqual(frame["program_hash"], blake3.blake3(b"source").hexdigest())
        self.assertEqual(
            set(frame),
            {"type", "corr_id", "program_hash", "file", "chunk_index", "total_chunks", "chunk_data"},
        )

    async def test_reload_weights_sends_an_absolute_path_and_returns_the_answer(self):
        client = PieClient("ws://example.invalid")
        client.ws = RespondingWebSocket(client, result='{"version": 3}')

        self.assertEqual(await client.reload_weights("ckpt", version=3), {"version": 3})
        [frame] = client.ws.sent
        path = str(Path("ckpt").absolute())
        self.assertEqual(frame, {"type": "reload_weights", "corr_id": 1, "path": path, "version": 3})

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




def process_event(process_id, event, value):
    return msgpack.packb(
        {"type": "process_event", "process_id": process_id, "event": event, "value": value},
        use_bin_type=True,
    )


class ServedClientTests(unittest.IsolatedAsyncioTestCase):
    async def serve(self, frames, **options):
        from websockets.asyncio.server import serve

        async def handler(ws):
            for frame in frames:
                await ws.send(frame)
            await ws.wait_closed()

        server = await serve(handler, "127.0.0.1", 0)
        self.addAsyncCleanup(server.wait_closed)
        self.addCleanup(server.close)
        client = PieClient(f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}", **options)
        processes = {}
        for process_id in ("big", "small", "bad"):
            client.process_event_queues[process_id] = asyncio.Queue()
            processes[process_id] = Process(client, process_id)
        await client.connect()
        self.addAsyncCleanup(client.close)
        return processes

    async def test_a_large_return_and_a_small_one_both_arrive(self):
        big = "x" * (5 << 20)
        p = await self.serve([process_event("big", "return", big), process_event("small", "return", "ok")])
        small, large = await asyncio.wait_for(asyncio.gather(p["small"].result(), p["big"].result()), 10)
        self.assertEqual((small, len(large)), ("ok", len(big)))

    async def test_a_message_past_max_message_size_names_the_limit(self):
        p = await self.serve([process_event("big", "return", "x" * 4096)], max_message_size=1024)
        with self.assertRaisesRegex(RuntimeError, "max_message_size=1024"):
            await asyncio.wait_for(p["big"].result(), 10)

    async def test_a_message_that_fails_to_process_fails_only_its_process(self):
        broken = msgpack.packb(
            {"type": "file", "process_id": "bad", "file_hash": "h", "chunk_index": 0, "total_chunks": 2, "chunk_data": None},
            use_bin_type=True,
        )
        p = await self.serve([broken, process_event("small", "return", "ok")])
        self.assertEqual(await asyncio.wait_for(p["small"].result(), 10), "ok")
        with self.assertRaisesRegex(RuntimeError, "could not process a `file` message"):
            await asyncio.wait_for(p["bad"].result(), 10)

if __name__ == "__main__":
    unittest.main()
