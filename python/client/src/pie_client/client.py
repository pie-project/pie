import asyncio
import base64
import json
import msgpack
import websockets
import blake3

from enum import Enum
from typing import Any
from dataclasses import dataclass
from pathlib import Path

from .inferlet import Inferlet
from .crypto import ParsedPrivateKey


class Event(Enum):
    """Events received from a process."""

    Stdout = "stdout"
    Stderr = "stderr"
    Message = "message"
    File = "file"
    Return = "return"
    Error = "error"


class ReceivedFile(bytes):
    """One complete file an inferlet sent, with the name it suggested.

    A `bytes` SUBCLASS, deliberately. The file event has always carried the
    payload and nothing else, so every program written against this client
    does `event, data = await proc.recv()` and then writes `data`. Returning a
    tuple or a dataclass here would break all of them to add one field.
    Subclassing means the payload is still the payload -- `len(data)`,
    `data[:8]`, `f.write(data)` all unchanged -- and `data.name` is the new
    fact beside it.

    `name` is `None` for `session.send-file`, which carries no name, and a
    string for `session.send-file-as`, `send-frames` and `send-pcm`, which do.
    It is a SUGGESTION off the wire: sanitise it before joining it to a
    directory (`file_name()` does).
    """

    name: str | None

    def __new__(cls, data: bytes, name: str | None = None):
        self = super().__new__(cls, data)
        self.name = name
        return self

    def file_name(self, fallback: str = "file.bin") -> str:
        """A name safe to join onto a directory.

        The directory part, `..` and NUL are stripped and `fallback` stands in
        when nothing usable is left. The server sanitises on the way out too;
        this is the check that counts, because a client does not get to assume
        the server it is talking to is the one that wrote that code.
        """
        raw = self.name or ""
        tail = raw.replace("\\", "/").rsplit("/", 1)[-1]
        base = tail.replace("\0", "").strip()
        if base in ("", ".", ".."):
            return fallback
        return base

    def save(self, directory, fallback: str = "file.bin") -> Path:
        """Write this file into `directory` under its own name, and return the
        path. Creates the directory if it is not there."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / self.file_name(fallback)
        path.write_bytes(self)
        return path


class Process:
    """Represents a running process on the server."""

    def __init__(self, client, process_id: str):
        self.client = client
        self.process_id = process_id
        self.event_queue = self.client.process_event_queues.get(process_id)
        if self.event_queue is None:
            raise Exception(
                f"Internal error: No event queue for process {process_id}"
            )
        self.output: list[str] = []

    async def signal(self, message: str):
        """Send a string message to the process (fire-and-forget)."""
        await self.client.signal_process(self.process_id, message)

    async def transfer_file(self, file_bytes: bytes):
        """Transfer a file to the process (fire-and-forget, chunked)."""
        await self.client._transfer_file(self.process_id, file_bytes)

    async def recv(self) -> tuple[Event, "str | ReceivedFile"]:
        """
        Receive an event from the process. Blocks until an event is available.
        Returns a tuple of (Event, value): a string for the text events, and a
        `ReceivedFile` -- bytes carrying the name the inferlet suggested --
        for `Event.File`.
        """
        if self.event_queue is None:
            raise Exception("Event queue is not available for this process.")
        event_str, value = await self.event_queue.get()
        return Event(event_str), value

    async def result(self) -> Any:
        """Drain the process to its end and return what it returned: parsed
        JSON when the return is JSON, else the string. Stdout and messages
        are collected on `self.output`; an error raises `RuntimeError`."""
        while True:
            event, value = await self.recv()
            if event in (Event.Stdout, Event.Message):
                self.output.append(value)
            elif event == Event.Return:
                try:
                    return json.loads(value) if value else None
                except json.JSONDecodeError:
                    return value
            elif event == Event.Error:
                raise RuntimeError(value)

    async def terminate(self):
        """Request termination of the process."""
        await self.client.terminate_process(self.process_id)



class PieClient:
    """
    An asynchronous client for interacting with the Pie WebSocket server.
    This client is designed to be used as an async context manager.
    """

    def __init__(self, server_uri: str, identity: str = "default/bench"):
        """
        Initialize the client.
        :param server_uri: The WebSocket server URI (e.g., "ws://127.0.0.1:8080").
            A URI with no path gets the gateway's `/v1/ws` upgrade path appended.
        :param identity: Value for the `x-pie-identity` trust-edge header the
            gateway requires (`gateway/src/ingress/identity.rs`). A deployment
            terminates identity at an edge proxy; a direct client must supply it
            or the upgrade is rejected before the socket opens.
        """
        self.server_uri = server_uri
        self.identity = identity
        self.ws = None
        self.listener_task = None
        self.corr_id_counter = 0
        self.pending_requests = {}
        self.process_event_queues = {}
        self.pending_downloads = {}  # For reassembling file chunks

        # Buffer for early events to prevent race conditions.
        self.orphan_events = {}

    async def __aenter__(self):
        """Enter the async context, establishing the connection."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit the async context, closing the connection cleanly."""
        await self.close()

    async def connect(self):
        """Establish a WebSocket connection and start the background listener."""
        from urllib.parse import urlsplit
        uri = self.server_uri
        if not urlsplit(uri).path.strip("/"):
            uri = uri.rstrip("/") + "/v1/ws"
        self.ws = await websockets.connect(
            uri, additional_headers={"x-pie-identity": self.identity})
        self.listener_task = asyncio.create_task(self._listen_to_server())

    async def _listen_to_server(self):
        """Background task to receive and process all incoming server messages."""
        try:
            async for raw_msg in self.ws:
                if isinstance(raw_msg, bytes):
                    try:
                        message = msgpack.unpackb(raw_msg, raw=False)
                        await self._process_server_message(message)
                    except msgpack.UnpackException:
                        pass
        except (
            websockets.ConnectionClosedOK,
            websockets.ConnectionClosedError,
            Exception,
        ):
            pass
        finally:
            self._fail_connection_waiters(ConnectionError("WebSocket connection closed"))

    async def _process_server_message(self, message: dict):
        """Route incoming server messages based on their type."""
        msg_type = message.get("type")

        if msg_type == "response":
            corr_id = message.get("corr_id")
            if corr_id in self.pending_requests:
                future = self.pending_requests.pop(corr_id)
                future.set_result((message.get("ok"), message.get("result")))

        elif msg_type == "process_event":
            process_id = message.get("process_id")
            event = message.get("event")
            value = message.get("value", "")
            event_tuple = (event, value)

            if process_id in self.process_event_queues:
                await self.process_event_queues[process_id].put(event_tuple)
                # Clean up on terminal events
                if event in ("return", "error"):
                    del self.process_event_queues[process_id]
            else:
                # Queue doesn't exist yet, buffer the event
                if process_id not in self.orphan_events:
                    self.orphan_events[process_id] = []
                self.orphan_events[process_id].append(event_tuple)

        elif msg_type == "file":
            await self._handle_file_chunk(message)

    async def _handle_file_chunk(self, message: dict):
        """Processes a chunk of a file sent from the server."""
        file_hash = message.get("file_hash")
        process_id = message.get("process_id")
        chunk_index = message.get("chunk_index")
        total_chunks = message.get("total_chunks")

        if process_id not in self.process_event_queues:
            return

        if file_hash not in self.pending_downloads:
            if chunk_index != 0:
                return
            self.pending_downloads[file_hash] = {
                "buffer": bytearray(),
                "total_chunks": total_chunks,
                "process_id": process_id,
                # The name rides on every chunk; the first one is the one
                # that is guaranteed to have been seen, so it is the one
                # remembered. `None` when the inferlet named nothing.
                "name": message.get("name"),
            }

        download = self.pending_downloads[file_hash]
        download["buffer"].extend(message.get("chunk_data"))

        if chunk_index == total_chunks - 1:
            completed_file = ReceivedFile(
                bytes(download["buffer"]), download.get("name")
            )
            computed_hash = blake3.blake3(completed_file).hexdigest()
            if computed_hash == file_hash:
                if process_id in self.process_event_queues:
                    await self.process_event_queues[process_id].put(
                        ("file", completed_file)
                    )
            del self.pending_downloads[file_hash]

    async def close(self):
        """Gracefully close the WebSocket connection and shut down background tasks."""
        if self.ws:
            try:
                await self.ws.close()
            except Exception:
                pass
        if self.listener_task:
            try:
                self.listener_task.cancel()
                await self.listener_task
            except asyncio.CancelledError:
                pass
        self._fail_connection_waiters(ConnectionError("WebSocket connection closed"))

    def _fail_connection_waiters(self, exc: Exception):
        """Reject all waiters that depend on the websocket staying open."""
        self._fail_pending_requests(exc)
        self._fail_process_event_queues(exc)

    def _fail_pending_requests(self, exc: Exception):
        """Reject all requests still waiting for a server response."""
        pending = list(self.pending_requests.items())
        self.pending_requests.clear()
        for _, future in pending:
            if not future.done():
                future.set_exception(exc)

    def _fail_process_event_queues(self, exc: Exception):
        """Wake process recv() waiters when the connection goes away."""
        message = str(exc) or exc.__class__.__name__
        queues = list(self.process_event_queues.items())
        self.process_event_queues.clear()
        for _, queue in queues:
            queue.put_nowait((Event.Error.value, message))

    def _get_next_corr_id(self):
        """Generate a unique correlation ID for a request."""
        self.corr_id_counter += 1
        return self.corr_id_counter

    async def _send_msg_and_wait(self, msg: dict) -> tuple[bool, str]:
        """Send a message that expects a Response and wait for it."""
        corr_id = self._get_next_corr_id()
        msg["corr_id"] = corr_id
        future = asyncio.get_event_loop().create_future()
        self.pending_requests[corr_id] = future
        encoded = msgpack.packb(msg, use_bin_type=True)
        try:
            await self.ws.send(encoded)
        except Exception:
            self.pending_requests.pop(corr_id, None)
            raise
        try:
            return await future
        finally:
            self.pending_requests.pop(corr_id, None)

    # =========================================================================
    # Authentication
    # =========================================================================

    async def authenticate(
        self, username: str, private_key: ParsedPrivateKey | None = None
    ) -> None:
        """
        Authenticate the client with the server using public key authentication.

        :param username: The username to authenticate as.
        :param private_key: The private key for signing the challenge.
                           Required if the server has authentication enabled.
        :raises Exception: If authentication fails.
        """
        msg = {"type": "auth_identify", "username": username}
        ok, result = await self._send_msg_and_wait(msg)

        if not ok:
            raise Exception(f"Username '{username}' rejected by server: {result}")

        if result in (
            "Authenticated (Engine disabled authentication)",
            # Gateway-fronted sessions arrive pre-authenticated (the edge
            # supplied the identity via `x-pie-identity`); the engine answers
            # identify with this instead of a challenge.
            "Already authenticated",
        ):
            return

        if private_key is None:
            raise Exception(
                "Server requires public key authentication but no private key provided"
            )

        try:
            challenge = base64.b64decode(result)
        except Exception as e:
            raise Exception(f"Failed to decode challenge from server: {e}")

        signature_bytes = private_key.sign(challenge)
        signature_b64 = base64.b64encode(signature_bytes).decode("utf-8")

        msg = {"type": "auth_prove", "signature": signature_b64}
        ok, result = await self._send_msg_and_wait(msg)

        if not ok:
            raise Exception(
                f"Signature verification failed for username '{username}': {result}"
            )

    # =========================================================================
    # Queries
    # =========================================================================

    async def query(self, subject: str, record: str) -> tuple[bool, str]:
        """Send a generic query to the server."""
        msg = {"type": "query", "subject": subject, "record": record}
        return await self._send_msg_and_wait(msg)

    async def check_program(self, inferlet: str) -> bool:
        """Whether the server holds `inferlet`, given as `name@version`."""
        if "@" not in inferlet:
            raise ValueError("Version required: use 'name@version' format")
        name, version = inferlet.rsplit("@", 1)
        msg = {"type": "check_program", "name": name, "version": version}
        ok, result = await self._send_msg_and_wait(msg)
        if not ok:
            raise Exception(f"CheckProgram failed: {result}")
        return result == "true"

    async def _upload_chunked(self, data_bytes: bytes, msg_template: dict):
        """Internal helper to handle generic chunked uploads with response."""
        chunk_size = 256 * 1024
        total_size = len(data_bytes)
        total_chunks = (
            (total_size + chunk_size - 1) // chunk_size if total_size > 0 else 1
        )

        corr_id = self._get_next_corr_id()
        msg_template["corr_id"] = corr_id
        msg_template["total_chunks"] = total_chunks
        future = asyncio.get_event_loop().create_future()
        self.pending_requests[corr_id] = future

        try:
            if total_size == 0:
                msg = msg_template.copy()
                msg.update({"chunk_index": 0, "chunk_data": b""})
                await self.ws.send(msgpack.packb(msg, use_bin_type=True))
            else:
                for chunk_index in range(total_chunks):
                    start = chunk_index * chunk_size
                    end = min(start + chunk_size, total_size)
                    msg = msg_template.copy()
                    msg.update(
                        {"chunk_index": chunk_index, "chunk_data": data_bytes[start:end]}
                    )
                    await self.ws.send(msgpack.packb(msg, use_bin_type=True))
        except Exception:
            self.pending_requests.pop(corr_id, None)
            raise
        try:
            ok, result = await future
        finally:
            self.pending_requests.pop(corr_id, None)

        if not ok:
            raise Exception(f"Upload failed: {result}")
        return result

    async def install_program(self, wasm_path: str | Path, manifest_path: str | Path, force_overwrite: bool = False):
        """Install a program from files: a component or a script, and its
        manifest."""
        await self.install_program_bytes(
            Path(wasm_path).read_bytes(), Path(manifest_path).read_text(), force_overwrite
        )

    async def install_program_bytes(self, program_bytes: bytes, manifest: str, force_overwrite: bool = False):
        """Install a program to the server in chunks: its artifact (a
        component, or a script's source when the manifest names a
        `[runtime] language`) and its manifest TOML."""
        program_hash = blake3.blake3(program_bytes).hexdigest()
        template = {
            "type": "add_program",
            "program_hash": program_hash,
            "manifest": manifest,
            "force_overwrite": force_overwrite,
        }
        await self._upload_chunked(program_bytes, template)

    # =========================================================================
    # File Transfer (fire-and-forget, no response expected)
    # =========================================================================

    async def _transfer_file(self, process_id: str, file_bytes: bytes):
        """Transfer a file to a process (fire-and-forget, chunked)."""
        file_hash = blake3.blake3(file_bytes).hexdigest()
        chunk_size = 256 * 1024
        total_size = len(file_bytes)
        total_chunks = (
            (total_size + chunk_size - 1) // chunk_size if total_size > 0 else 1
        )

        if total_size == 0:
            msg = {
                "type": "transfer_file",
                "process_id": process_id,
                "file_hash": file_hash,
                "chunk_index": 0,
                "total_chunks": 1,
                "chunk_data": b"",
            }
            await self.ws.send(msgpack.packb(msg, use_bin_type=True))
        else:
            for chunk_index in range(total_chunks):
                start = chunk_index * chunk_size
                end = min(start + chunk_size, total_size)
                msg = {
                    "type": "transfer_file",
                    "process_id": process_id,
                    "file_hash": file_hash,
                    "chunk_index": chunk_index,
                    "total_chunks": total_chunks,
                    "chunk_data": file_bytes[start:end],
                }
                await self.ws.send(msgpack.packb(msg, use_bin_type=True))

    # =========================================================================
    # Process Lifecycle
    # =========================================================================

    async def launch_process(
        self,
        inferlet: str,
        input: dict | list | None = None,
        capture_outputs: bool = True,
    ) -> Process:
        """Launch a process. Returns a Process object for interaction.

        :param inferlet: The inferlet name (e.g., "text-completion@0.1.0").
        :param input: A dict of input parameters, serialized to JSON.
        :param capture_outputs: If True, process outputs are streamed to the client.
        :return: A Process object for the launched inferlet.
        """
        msg = {
            "type": "launch_process",
            "inferlet": inferlet,
            "input": json.dumps(input if input is not None else {}),
            "capture_outputs": capture_outputs,
        }
        ok, result = await self._send_msg_and_wait(msg)

        if not ok:
            raise Exception(f"Failed to launch process: {result}")

        process_id = result
        queue = asyncio.Queue()
        self.process_event_queues[process_id] = queue
        # Replay any orphan events
        if process_id in self.orphan_events:
            for event_tuple in self.orphan_events.pop(process_id):
                await queue.put(event_tuple)

        return Process(self, process_id)

    async def run(self, program: "Inferlet | str", /, **kwargs: Any) -> Process:
        """Launch an inferlet with keyword arguments as its input.

        A decorated function (`@inferlet`) is installed first if the server
        does not hold its exact source yet; a string is an installed
        program's `name@version` (or bare name). Returns the `Process`;
        `await process.result()` waits for what it returns.
        """
        if isinstance(program, Inferlet):
            if not await self.check_program(program.program):
                await self.install_program_bytes(
                    program.source.encode("utf-8"), program.manifest_toml()
                )
            program = program.program
        return await self.launch_process(program, input=kwargs)

    async def attach_process(self, process_id: str) -> Process:
        """Attach to an existing process.

        :param process_id: The UUID of the process to attach to.
        :return: A Process object for the attached process.
        """
        msg = {
            "type": "attach_process",
            "process_id": process_id,
        }
        ok, result = await self._send_msg_and_wait(msg)

        if not ok:
            raise Exception(f"Failed to attach to process: {result}")

        queue = asyncio.Queue()
        self.process_event_queues[process_id] = queue
        if process_id in self.orphan_events:
            for event_tuple in self.orphan_events.pop(process_id):
                await queue.put(event_tuple)

        return Process(self, process_id)

    async def list_processes(self) -> list[dict]:
        """Get a list of running process stats (dicts with id, username, program, arguments, elapsed_secs)."""
        msg = {"type": "list_processes"}
        ok, result = await self._send_msg_and_wait(msg)
        if ok:
            try:
                return json.loads(result)
            except (json.JSONDecodeError, TypeError):
                return []
        raise Exception(f"List processes failed: {result}")

    async def ping(self) -> None:
        """Ping the server to check connectivity."""
        msg = {"type": "ping"}
        ok, result = await self._send_msg_and_wait(msg)
        if not ok:
            raise Exception(f"Ping failed: {result}")

    async def signal_process(self, process_id: str, message: str):
        """Send a signal/message to a running process (fire-and-forget)."""
        msg = {
            "type": "signal_process",
            "process_id": process_id,
            "message": message,
        }
        await self.ws.send(msgpack.packb(msg, use_bin_type=True))

    async def terminate_process(self, process_id: str) -> None:
        """Request the server to terminate a running process."""
        msg = {"type": "terminate_process", "process_id": process_id}
        ok, result = await self._send_msg_and_wait(msg)
        if not ok:
            raise Exception(f"Failed to terminate process: {result}")
