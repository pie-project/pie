import asyncio
import base64
import msgpack
import websockets
import blake3

from enum import Enum
from dataclasses import dataclass

from .crypto import ParsedPrivateKey


class Event(Enum):
    """Enumeration for events received from an instance."""

    Message = 0
    Completed = 1
    Aborted = 2
    Exception = 3
    ServerError = 4
    OutOfResources = 5
    Blob = 6  # Represents a binary data blob
    Stdout = 7  # Streaming stdout output
    Stderr = 8  # Streaming stderr output


@dataclass
class InstanceInfo:
    """Information about a running instance."""

    id: str
    arguments: list[str]
    status: str  # "Attached", "Detached", or "Finished"
    username: str = ""
    elapsed_secs: int = 0
    kv_pages_used: int = 0


class Instance:
    """Represents a running instance of a program on the server."""

    def __init__(self, client, instance_id: str):
        self.client = client
        self.instance_id = instance_id
        self.event_queue = self.client.inst_event_queues.get(instance_id)
        if self.event_queue is None:
            raise Exception(
                f"Internal error: No event queue for instance {instance_id}"
            )

    async def send(self, message: str):
        """Send a string message to the instance."""
        await self.client.signal_instance(self.instance_id, message)

    async def upload_blob(self, blob_bytes: bytes):
        """Upload a blob of binary data to the instance."""
        await self.client.upload_blob(self.instance_id, blob_bytes)

    async def recv(self) -> tuple[Event, str | bytes]:
        """
        Receive an event from the instance. Blocks until an event is available.
        Returns a tuple of (Event, message), where message can be a string or bytes.
        """
        if self.event_queue is None:
            raise Exception("Event queue is not available for this instance.")
        event_code, msg = await self.event_queue.get()

        event = Event(event_code)
        return event, msg

    async def terminate(self):
        """Request termination of the instance."""
        await self.client.terminate_instance(self.instance_id)


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
        self.pending_launch_requests = {}
        self.pending_attach_requests = {}
        self.pending_list_requests = {}
        self.inst_event_queues = {}
        self.pending_downloads = {}  # For reassembling blob chunks

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
                future.set_result((message.get("successful"), message.get("result")))

        elif msg_type == "challenge":
            # Auth challenge response - treat like a regular response
            corr_id = message.get("corr_id")
            if corr_id in self.pending_requests:
                future = self.pending_requests.pop(corr_id)
                # successful=True with challenge as the result
                future.set_result((True, message.get("challenge")))

        elif msg_type == "instance_launch_result":
            corr_id = message.get("corr_id")
            if corr_id in self.pending_launch_requests:
                future = self.pending_launch_requests.pop(corr_id)
                successful = message.get("successful")
                instance_id = message.get("message")
                if successful:
                    # Create event queue before resolving to prevent race condition
                    queue = asyncio.Queue()
                    self.inst_event_queues[instance_id] = queue
                    # Replay any orphan events
                    if instance_id in self.orphan_events:
                        early_events = self.orphan_events.pop(instance_id)
                        for event_tuple in early_events:
                            await queue.put(event_tuple)
                future.set_result((successful, instance_id))

        elif msg_type == "instance_attach_result":
            corr_id = message.get("corr_id")
            if corr_id in self.pending_attach_requests:
                future, instance_id = self.pending_attach_requests.pop(corr_id)
                successful = message.get("successful")
                result_msg = message.get("message")
                if successful:
                    # Create event queue before resolving to prevent race condition
                    queue = asyncio.Queue()
                    self.inst_event_queues[instance_id] = queue
                    # Replay any orphan events
                    if instance_id in self.orphan_events:
                        early_events = self.orphan_events.pop(instance_id)
                        for event_tuple in early_events:
                            await queue.put(event_tuple)
                future.set_result((successful, result_msg))

        elif msg_type == "live_instances":
            corr_id = message.get("corr_id")
            if corr_id in self.pending_list_requests:
                future = self.pending_list_requests.pop(corr_id)
                instances_raw = message.get("instances", [])
                instances = [
                    InstanceInfo(
                        id=inst.get("id"),
                        arguments=inst.get("arguments", []),
                        status=inst.get("status", "Unknown"),
                        username=inst.get("username", ""),
                        elapsed_secs=inst.get("elapsed_secs", 0),
                        kv_pages_used=inst.get("kv_pages_used", 0),
                    )
                    for inst in instances_raw
                ]
                future.set_result(instances)

        elif msg_type == "instance_event":
            instance_id = message.get("instance_id")
            event_tuple = (message.get("event"), message.get("message"))

            if instance_id in self.inst_event_queues:
                await self.inst_event_queues[instance_id].put(event_tuple)
            else:
                # Queue doesn't exist yet, buffer the event
                if instance_id not in self.orphan_events:
                    self.orphan_events[instance_id] = []
                self.orphan_events[instance_id].append(event_tuple)

        elif msg_type == "streaming_output":
            instance_id = message.get("instance_id")
            output = message.get("output", {})
            if instance_id in self.inst_event_queues:
                # output is {"Stdout": text} or {"Stderr": text}
                if "Stdout" in output:
                    await self.inst_event_queues[instance_id].put(
                        (Event.Stdout.value, output["Stdout"])
                    )
                elif "Stderr" in output:
                    await self.inst_event_queues[instance_id].put(
                        (Event.Stderr.value, output["Stderr"])
                    )

        elif msg_type == "download_blob":
            await self._handle_blob_chunk(message)

    async def _handle_blob_chunk(self, message: dict):
        """Processes a chunk of a blob sent from the server, ensuring sequential order."""
        blob_hash = message.get("blob_hash")
        instance_id = message.get("instance_id")
        chunk_index = message.get("chunk_index")
        total_chunks = message.get("total_chunks")

        if instance_id not in self.inst_event_queues:
            return  # Ignore blobs for unknown/terminated instances

        # Initialize download on the first chunk (index 0)
        if blob_hash not in self.pending_downloads:
            if chunk_index != 0:

                return
            self.pending_downloads[blob_hash] = {
                "buffer": bytearray(),
                "total_chunks": total_chunks,
                "next_chunk_index": 1,
                "instance_id": instance_id,
            }

        download = self.pending_downloads[blob_hash]

        # Validate chunk consistency and order
        if (
            total_chunks != download["total_chunks"]
            or chunk_index != download["next_chunk_index"] - 1
        ):
            error_msg = (
                "Chunk count mismatch"
                if total_chunks != download["total_chunks"]
                else "Out-of-order chunk"
            )

            del self.pending_downloads[blob_hash]
            return

        download["buffer"].extend(message.get("chunk_data"))
        download["next_chunk_index"] += 1

        # If all chunks are received, finalize the download
        if download["next_chunk_index"] == download["total_chunks"]:
            completed_blob = bytes(download["buffer"])
            computed_hash = blake3.blake3(completed_blob).hexdigest()
            if computed_hash == blob_hash:
                await self.inst_event_queues[instance_id].put(
                    (Event.Blob.value, completed_blob)
                )

            del self.pending_downloads[blob_hash]

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
                pass  # Expected on cancellation

    def _get_next_corr_id(self):
        """Generate a unique correlation ID for a request."""
        self.corr_id_counter += 1
        return self.corr_id_counter

    async def _send_msg_and_wait(self, msg: dict) -> tuple[bool, str]:
        """Send a message that expects a response and wait for it."""
        corr_id = self._get_next_corr_id()
        msg["corr_id"] = corr_id
        future = asyncio.get_event_loop().create_future()
        self.pending_requests[corr_id] = future
        encoded = msgpack.packb(msg, use_bin_type=True)
        await self.ws.send(encoded)
        return await future

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
        # Send identification request
        msg = {"type": "identification", "username": username}
        successful, result = await self._send_msg_and_wait(msg)

        if not successful:
            raise Exception(f"Username '{username}' rejected by server: {result}")

        # Check if server has disabled authentication
        if result == "Authenticated (Engine disabled authentication)":

            return

        # Server returned a challenge - we need to sign it
        if private_key is None:
            raise Exception(
                "Server requires public key authentication but no private key provided"
            )

        # Decode the base64-encoded challenge
        try:
            challenge = base64.b64decode(result)
        except Exception as e:
            raise Exception(f"Failed to decode challenge from server: {e}")

        # Sign the challenge
        signature_bytes = private_key.sign(challenge)
        signature_b64 = base64.b64encode(signature_bytes).decode("utf-8")

        # Send the signature
        msg = {"type": "signature", "signature": signature_b64}
        successful, result = await self._send_msg_and_wait(msg)

        if not successful:
            raise Exception(
                f"Signature verification failed for username '{username}': {result}"
            )

    async def internal_authenticate(self, token: str) -> None:
        """
        Authenticate the client with the server using an internal token.
        This is used for internal communication (backend <-> engine, shell <-> engine).

        :param token: The internal authentication token.
        :raises Exception: If authentication fails.
        """
        msg = {"type": "internal_authenticate", "token": token}
        successful, result = await self._send_msg_and_wait(msg)
        if not successful:
            raise Exception(f"Internal authentication failed: {result}")

    async def identify(self, username: str) -> tuple[bool, str]:
        """
        [DEPRECATED] Use authenticate() instead.
        Legacy method for simple username identification.
        """
        msg = {"type": "identification", "username": username}
        successful, result = await self._send_msg_and_wait(msg)
        return successful, result

    async def query(self, subject: str, record: str) -> tuple[bool, str]:
        """Send a generic query to the server."""
        msg = {"type": "query", "subject": subject, "record": record}
        return await self._send_msg_and_wait(msg)

    async def program_exists(self, program_hash: str) -> bool:
        """Check if a program with the given hash exists on the server."""
        successful, result = await self.query("program_exists", program_hash)
        if successful:
            return result == "true"
        raise Exception(f"Query for program_exists failed: {result}")

    async def _upload_chunked(self, data_bytes: bytes, msg_template: dict):
        """Internal helper to handle generic chunked uploads."""
        data_hash = msg_template.get("program_hash") or msg_template.get("blob_hash")
        upload_type = msg_template["type"]

        chunk_size = 256 * 1024
        total_size = len(data_bytes)
        # An empty upload is still one chunk of zero bytes
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
        successful, result = await future

        if not successful:
            raise Exception(f"{upload_type.replace('_', ' ').title()} failed: {result}")

        return result

    async def upload_program(self, program_bytes: bytes):
        """Upload a program to the server in chunks."""
        program_hash = blake3.blake3(program_bytes).hexdigest()
        template = {"type": "upload_program", "program_hash": program_hash}
        await self._upload_chunked(program_bytes, template)

    async def upload_blob(self, instance_id: str, blob_bytes: bytes):
        """Upload a blob of data to a specific instance in chunks."""
        blob_hash = blake3.blake3(blob_bytes).hexdigest()
        template = {
            "type": "upload_blob",
            "instance_id": instance_id,
            "blob_hash": blob_hash,
        }
        await self._upload_chunked(blob_bytes, template)

    async def launch_instance(
        self,
        program_hash: str,
        arguments: list[str] | None = None,
        detached: bool = False,
    ) -> Instance:
        """Launch an instance of a program."""
        corr_id = self._get_next_corr_id()
        msg = {
            "type": "launch_instance",
            "corr_id": corr_id,
            "program_hash": program_hash,
            "arguments": arguments or [],
            "detached": detached,
        }

        future = asyncio.get_event_loop().create_future()
        self.pending_launch_requests[corr_id] = future
        encoded = msgpack.packb(msg, use_bin_type=True)
        await self.ws.send(encoded)

        successful, instance_id = await future

        if successful:
            return Instance(self, instance_id)
        raise Exception(f"Failed to launch instance: {instance_id}")

    async def launch_instance_from_registry(
        self, inferlet: str, arguments: list[str] | None = None, detached: bool = False
    ) -> Instance:
        """
        Launch an instance of an inferlet from the registry.

        The inferlet parameter can be:
        - Full name with version: "std/text-completion@0.1.0"
        - Without namespace (defaults to "std"): "text-completion@0.1.0"
        - Without version (defaults to "latest"): "std/text-completion" or "text-completion"

        :param inferlet: The inferlet name (e.g., "std/text-completion@0.1.0").
        :param arguments: Command-line arguments to pass to the inferlet.
        :param detached: If True, the instance runs in detached mode.
        :return: An Instance object for the launched inferlet.
        """
        corr_id = self._get_next_corr_id()
        msg = {
            "type": "launch_instance_from_registry",
            "corr_id": corr_id,
            "inferlet": inferlet,
            "arguments": arguments or [],
            "detached": detached,
        }

        future = asyncio.get_event_loop().create_future()
        self.pending_launch_requests[corr_id] = future
        encoded = msgpack.packb(msg, use_bin_type=True)
        await self.ws.send(encoded)

        successful, instance_id = await future

        if successful:
            return Instance(self, instance_id)
        raise Exception(f"Failed to launch instance from registry: {instance_id}")

    async def attach_instance(self, instance_id: str) -> Instance:
        """
        Attach to an existing detached instance.

        :param instance_id: The UUID of the instance to attach to.
        :return: An Instance object for the attached instance.
        :raises Exception: If attachment fails.
        """
        corr_id = self._get_next_corr_id()
        msg = {
            "type": "attach_instance",
            "corr_id": corr_id,
            "instance_id": instance_id,
        }

        future = asyncio.get_event_loop().create_future()
        self.pending_attach_requests[corr_id] = (future, instance_id)
        encoded = msgpack.packb(msg, use_bin_type=True)
        await self.ws.send(encoded)

        successful, result = await future

        if successful:
            return Instance(self, instance_id)
        raise Exception(f"Failed to attach to instance: {result}")

    async def list_instances(self) -> list[InstanceInfo]:
        """
        Get a list of all running instances on the server.

        :return: List of InstanceInfo objects.
        """
        corr_id = self._get_next_corr_id()
        msg = {"type": "list_instances", "corr_id": corr_id}

        future = asyncio.get_event_loop().create_future()
        self.pending_list_requests[corr_id] = future
        encoded = msgpack.packb(msg, use_bin_type=True)
        await self.ws.send(encoded)

        return await future

    async def ping(self) -> None:
        """
        Ping the server to check connectivity.

        :raises Exception: If ping fails.
        """
        msg = {"type": "ping"}
        successful, result = await self._send_msg_and_wait(msg)
        if not successful:
            raise Exception(f"Ping failed: {result}")

    async def signal_instance(self, instance_id: str, message: str):
        """Send a signal/message to a running instance (fire-and-forget)."""
        msg = {
            "type": "signal_instance",
            "instance_id": instance_id,
            "message": message,
        }
        await self.ws.send(msgpack.packb(msg, use_bin_type=True))

    async def terminate_instance(self, instance_id: str) -> None:
        """
        Request the server to terminate a running instance.

        :param instance_id: The UUID of the instance to terminate.
        :raises Exception: If termination fails.
        """
        msg = {"type": "terminate_instance", "instance_id": instance_id}
        successful, result = await self._send_msg_and_wait(msg)
        if not successful:
            raise Exception(f"Failed to terminate instance: {result}")

    async def launch_server_instance(
        self,
        program_hash: str,
        port: int,
        arguments: list[str] | None = None,
    ) -> None:
        """
        Launch a server inferlet that listens on a specific port.

        Server inferlets implement the wasi:http/incoming-handler interface and
        handle incoming HTTP requests. Unlike regular inferlets, they are long-running
        and create a new WASM instance for each incoming request.

        :param program_hash: The hash of the uploaded program.
        :param port: The TCP port to listen on.
        :param arguments: Command-line arguments to pass to the inferlet.
        :raises Exception: If launch fails.
        """
        msg = {
            "type": "launch_server_instance",
            "port": port,
            "program_hash": program_hash,
            "arguments": arguments or [],
        }
        successful, result = await self._send_msg_and_wait(msg)
        if not successful:
            raise Exception(f"Failed to launch server instance: {result}")

