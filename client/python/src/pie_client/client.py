import asyncio
import base64
import json
import msgpack
import websockets
import blake3

from enum import Enum
from dataclasses import dataclass
from pathlib import Path

from .crypto import ParsedPrivateKey


class Event(Enum):
    """Events received from a process."""

    Stdout = "stdout"
    Stderr = "stderr"
    Message = "message"
    File = "file"
    Return = "return"
    Error = "error"


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

    async def signal(self, message: str):
        """Send a string message to the process (fire-and-forget)."""
        await self.client.signal_process(self.process_id, message)

    async def transfer_file(self, file_bytes: bytes):
        """Transfer a file to the process (fire-and-forget, chunked)."""
        await self.client._transfer_file(self.process_id, file_bytes)

    async def recv(self) -> tuple[Event, str | bytes]:
        """
        Receive an event from the process. Blocks until an event is available.
        Returns a tuple of (Event, value), where value can be a string or bytes.
        """
        if self.event_queue is None:
            raise Exception("Event queue is not available for this process.")
        event_str, value = await self.event_queue.get()
        return Event(event_str), value

    async def terminate(self):
        """Request termination of the process."""
        await self.client.terminate_process(self.process_id)



class PieClient:
    """
    An asynchronous client for interacting with the Pie WebSocket server.
    This client is designed to be used as an async context manager.
    """

    def __init__(self, server_uri: str):
        """
        Initialize the client.
        :param server_uri: The WebSocket server URI (e.g., "ws://127.0.0.1:8080").
        """
        self.server_uri = server_uri
        self.ws = None
        self.listener_task = None
        self.corr_id_counter = 0
        self.pending_requests = {}
        self.process_event_queues = {}
        self.pending_downloads = {}  # For reassembling file chunks

        # Buffer for early events to prevent race conditions.
        self.orphan_events = {}

    # Also keep old name for backward compat
    @property
    def inst_event_queues(self):
        return self.process_event_queues

    async def __aenter__(self):
        """Enter the async context, establishing the connection."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit the async context, closing the connection cleanly."""
        await self.close()

    async def connect(self):
        """Establish a WebSocket connection and start the background listener."""
        self.ws = await websockets.connect(self.server_uri)
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
            }

        download = self.pending_downloads[file_hash]
        download["buffer"].extend(message.get("chunk_data"))

        if chunk_index == total_chunks - 1:
            completed_file = bytes(download["buffer"])
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
        await self.ws.send(encoded)
        return await future

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

        if result == "Authenticated (Engine disabled authentication)":
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

    async def auth_by_token(self, token: str) -> None:
        """Authenticate using an internal token (backend/shell ↔ engine)."""
        msg = {"type": "auth_by_token", "token": token}
        ok, result = await self._send_msg_and_wait(msg)
        if not ok:
            raise Exception(f"Internal authentication failed: {result}")

    # =========================================================================
    # Queries
    # =========================================================================

    async def query(self, subject: str, record: str) -> tuple[bool, str]:
        """Send a generic query to the server."""
        msg = {"type": "query", "subject": subject, "record": record}
        return await self._send_msg_and_wait(msg)

    async def check_program(
        self,
        inferlet: str,
        wasm_path: str | Path | None = None,
        manifest_path: str | Path | None = None,
    ) -> bool:
        """Check if a program exists on the server.

        Args:
            inferlet: The inferlet name (e.g., "text-completion@0.1.0").
            wasm_path: Optional path to the WASM binary file for hash verification.
            manifest_path: Optional path to the manifest TOML file for hash verification.
        """
        if (wasm_path is None) != (manifest_path is None):
            raise ValueError(
                "wasm_path and manifest_path must both be provided or both be None"
            )

        name = inferlet
        version = None
        if "@" in inferlet:
            name, version = inferlet.rsplit("@", 1)

        wasm_hash = None
        manifest_hash = None
        if wasm_path and manifest_path:
            wasm_bytes = Path(wasm_path).read_bytes()
            manifest_content = Path(manifest_path).read_text()
            wasm_hash = blake3.blake3(wasm_bytes).hexdigest()
            manifest_hash = blake3.blake3(manifest_content.encode()).hexdigest()

        msg = {
            "type": "check_program",
            "name": name,
        }
        if version is not None:
            msg["version"] = version
        if wasm_hash is not None:
            msg["wasm_hash"] = wasm_hash
        if manifest_hash is not None:
            msg["manifest_hash"] = manifest_hash

        ok, result = await self._send_msg_and_wait(msg)
        if ok:
            return result == "true"
        raise Exception(f"CheckProgram failed: {result}")

    # =========================================================================
    # Program Upload
    # =========================================================================

    async def _upload_chunked(self, data_bytes: bytes, msg_template: dict):
        """Internal helper to handle generic chunked uploads with response."""
        data_hash = msg_template.get("program_hash") or msg_template.get("file_hash")

        chunk_size = 256 * 1024
        total_size = len(data_bytes)
        total_chunks = (
            (total_size + chunk_size - 1) // chunk_size if total_size > 0 else 1
        )

        corr_id = self._get_next_corr_id()
        msg_template["corr_id"] = corr_id
        msg_template["total_chunks"] = total_chunks

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

        future = asyncio.get_event_loop().create_future()
        self.pending_requests[corr_id] = future
        ok, result = await future

        if not ok:
            raise Exception(f"Upload failed: {result}")
        return result

    async def install_program(self, wasm_path: str | Path, manifest_path: str | Path, force_overwrite: bool = False):
        """Install a program to the server in chunks."""
        program_bytes = Path(wasm_path).read_bytes()
        manifest = Path(manifest_path).read_text()
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
        arguments: list[str] | None = None,
        capture_outputs: bool = True,
    ) -> Process:
        """Launch a process. Returns a Process object for interaction.

        :param inferlet: The inferlet name (e.g., "text-completion@0.1.0").
        :param arguments: Command-line arguments to pass to the inferlet.
        :param capture_outputs: If True, process outputs are streamed to the client.
        :return: A Process object for the launched inferlet.
        """
        msg = {
            "type": "launch_process",
            "inferlet": inferlet,
            "arguments": arguments or [],
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

    async def list_processes(self) -> list[str]:
        """Get a list of running process UUID strings."""
        msg = {"type": "list_processes"}
        ok, result = await self._send_msg_and_wait(msg)
        if ok:
            try:
                return json.loads(result)
            except (json.JSONDecodeError, TypeError):
                return [result] if result else []
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

    async def launch_daemon(
        self,
        inferlet: str,
        port: int,
        arguments: list[str] | None = None,
    ) -> None:
        """Launch a daemon inferlet that listens on a specific port."""
        msg = {
            "type": "launch_daemon",
            "port": port,
            "inferlet": inferlet,
            "arguments": arguments or [],
        }
        ok, result = await self._send_msg_and_wait(msg)
        if not ok:
            raise Exception(f"Failed to launch daemon: {result}")
