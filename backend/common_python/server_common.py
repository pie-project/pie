"""Shared server utilities for PIE backends.

This module hosts the transport loop, registration logic, and configuration
helpers that are agnostic to the underlying compute backend. Individual
backends provide their own handler classes and model loading routines while
reusing this shared infrastructure.
"""

from __future__ import annotations

import enum
import os
import random
import struct
import sys
import queue
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Type

import msgpack
import msgspec
import torch
import zmq
from platformdirs import user_cache_dir
from websockets.sync.client import connect

from message import (
    DownloadAdapterRequest,
    EmbedImageRequest,
    ForwardPassRequest,
    HandshakeRequest,
    HeartbeatRequest,
    InitializeAdapterRequest,
    QueryRequest,
    UpdateAdapterRequest,
    UploadAdapterRequest,
)


class HandlerId(enum.Enum):
    """Enumeration of handler message types."""

    HANDSHAKE = 0
    HEARTBEAT = 1
    QUERY = 2
    FORWARD_PASS = 3
    EMBED_IMAGE = 4
    INITIALIZE_ADAPTER = 5
    UPDATE_ADAPTER = 6
    UPLOAD_HANDLER = 7
    DOWNLOAD_HANDLER = 8


def resolve_cache_dir(cache_dir: str | None) -> str:
    """Resolve the cache directory using CLI arg > env var > default."""

    return cache_dir or os.environ.get("PIE_HOME") or str(Path(user_cache_dir("pie")))


def build_config(**kwargs: Any) -> Dict[str, Any]:
    """Normalize server configuration dictionary and resolve cache directory."""
    config = {k: v for k, v in kwargs.items() if v is not None}

    # Resolve the cache directory
    config["cache_dir"] = resolve_cache_dir(config.get("cache_dir"))

    # Check that either `max_num_kv_pages` or `gpu_mem_headroom` is set
    if "max_num_kv_pages" not in config and "gpu_mem_headroom" not in config:
        terminate(
            "Config must contain either 'max_num_kv_pages' or 'gpu_mem_headroom'."
        )

    # Check that if `gpu_mem_headroom` is set, then CUDA must be available
    if "gpu_mem_headroom" in config:
        if not torch.cuda.is_available():
            terminate("'gpu_mem_headroom' is set but CUDA is not available.")
        if "cuda" not in config["device"]:
            terminate("'gpu_mem_headroom' is set but device is not a CUDA device.")

    return config


def print_config(config: Dict[str, Any]) -> None:
    """Utility to print configuration in a consistent format."""

    print("--- Configuration ---")
    for key, value in config.items():
        print(f"{key}: {value}")
    print("----------------------")


def start_service(
    *,
    config: Dict[str, Any],
    handler_cls: Type,
    register_with_controller: bool = True,
) -> None:
    """Spin up the backend service using the provided handler implementation."""

    if config["controller_host"] in ["127.0.0.1", "localhost"]:
        unique_id = random.randint(1000, 9999)
        endpoint = f"ipc:///tmp/pie-service-{unique_id}"
        real_endpoint = endpoint
    else:
        endpoint = f"tcp://{config['host']}:{config['port']}"
        real_endpoint = f"tcp://*:{config['port']}"

    handler = handler_cls(
        config=config,
    )

    context = zmq.Context()
    socket = context.socket(zmq.ROUTER)
    socket.bind(real_endpoint)

    heartbeat_request_queue = queue.Queue()
    work_request_queue = queue.Queue()
    response_queue = queue.Queue()

    # Thread Structure:
    #
    # +-----------------------------+
    # |     zmq_listen_thread       |  <-- Receives requests (ZMQ socket)
    # +-----------------------------+
    #   |                         |
    #   | (heartbeat req queue)   | (work req queue)
    #   v                         v
    # +------------------+    +---------------+
    # | heartbeat_thread |    | worker_thread |
    # +------------------+    +---------------+
    #           |                |
    #           +-------+--------+
    #                   | (response queue)
    #                   v
    # +-----------------------------+
    # |    zmq_response_thread      |  --> Sends responses (ZMQ socket)
    # +-----------------------------+

    threading.Thread(
        target=heartbeat_thread,
        args=(heartbeat_request_queue, response_queue, handler),
        daemon=True,
    ).start()
    threading.Thread(
        target=worker_thread,
        args=(work_request_queue, response_queue, handler),
        daemon=True,
    ).start()
    threading.Thread(
        target=zmq_response_thread, args=(response_queue, socket), daemon=True
    ).start()
    threading.Thread(
        target=zmq_listen_thread,
        args=(heartbeat_request_queue, work_request_queue, socket),
        daemon=True,
    ).start()

    if register_with_controller:
        threading.Thread(
            target=register_thread,
            args=(config, endpoint),
            daemon=True,
        ).start()

    try:
        # Block forever until receiving keyboard interrupt
        while True:
            threading.Event().wait()
    except KeyboardInterrupt:
        print("\nShutting down server...")
    finally:
        socket.close()
        context.term()
        print("Server shutdown complete.")


def register_thread(config: Dict[str, Any], endpoint: str) -> None:
    """Register this service with the controller."""

    controller_addr = f"ws://{config['controller_host']}:{config['controller_port']}"
    try:
        with connect(controller_addr) as websocket:
            auth_msg = msgpack.packb(
                {
                    "type": "authenticate",
                    "corr_id": 0,
                    "token": config["auth_token"],
                },
                use_bin_type=True,
            )
            if auth_msg is not None:
                websocket.send(auth_msg)
            auth_response = msgpack.unpackb(websocket.recv(), raw=False)
            if not auth_response.get("successful"):
                print(
                    f"Authentication failed: {auth_response.get('result', 'Unknown error')}"
                )
                sys.exit(1)

            reg_msg = msgpack.packb(
                {
                    "type": "attach_remote_service",
                    "corr_id": 0,
                    "endpoint": endpoint,
                    "service_name": config["model"],
                    "service_type": "model",
                },
                use_bin_type=True,
            )
            if reg_msg is not None:
                websocket.send(reg_msg)
            reg_response = msgpack.unpackb(websocket.recv(), raw=False)
            if not reg_response.get("successful"):
                print(
                    f"Controller registration failed: {reg_response.get('result', 'Unknown error')}"
                )
                sys.exit(1)

            print(f"Registered with controller at {controller_addr}")

    except (ConnectionRefusedError, TimeoutError) as exc:
        print(f"Failed to connect to the controller at {controller_addr}.")
        print(f"Error: {exc}")
        print("Please ensure the controller is running and accessible. Terminating.")
        os._exit(1)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"An unexpected error occurred during registration: {exc}. Terminating.")
        os._exit(1)


def heartbeat_thread(
    heartbeat_request_queue: queue.Queue, response_queue: queue.Queue, handler: Any
) -> None:
    """Heartbeat thread that responds to heartbeat requests to the controller. And if no
    heartbeat is received for the timeout period, terminates the program."""

    heartbeat_timeout = 15
    last_heartbeat_time = time.monotonic()

    try:
        while True:
            time.sleep(1)

            if time.monotonic() - last_heartbeat_time > heartbeat_timeout:
                print(
                    f"[!] Heartbeat timeout after {heartbeat_timeout}s, exiting",
                    file=sys.stderr,
                )
                os._exit(1)

            if heartbeat_request_queue.empty():
                continue

            while not heartbeat_request_queue.empty():
                client_identity, corr_id_bytes, handler_id_bytes, reqs = (
                    heartbeat_request_queue.get()
                )
                resps = handler.heartbeat(reqs)
                response_queue.put(
                    (client_identity, corr_id_bytes, handler_id_bytes, resps)
                )

            last_heartbeat_time = time.monotonic()
    except Exception as exc:  # pylint: disable=broad-exception-caught
        terminate(f"Unhandled error occurred in the heartbeat thread: {exc}")


def worker_thread(
    work_request_queue: queue.Queue, response_queue: queue.Queue, handler: Any
) -> None:
    """Worker thread that processes incoming requests from the controller."""

    try:
        while True:
            client_identity, corr_id_bytes, handler_id_bytes, handler_id, reqs = (
                work_request_queue.get()
            )

            resps = []
            match handler_id:
                case HandlerId.HANDSHAKE.value:
                    resps = handler.handshake(reqs)
                case HandlerId.QUERY.value:
                    resps = handler.query(reqs)
                case HandlerId.FORWARD_PASS.value:
                    resps = handler.forward_pass(reqs)
                case HandlerId.EMBED_IMAGE.value:
                    handler.embed_image(reqs)
                case HandlerId.INITIALIZE_ADAPTER.value:
                    handler.initialize_adapter(reqs)
                case HandlerId.UPDATE_ADAPTER.value:
                    handler.update_adapter(reqs)
                case HandlerId.UPLOAD_HANDLER.value:
                    handler.upload_handler(reqs)
                case HandlerId.DOWNLOAD_HANDLER.value:
                    resps = handler.download_handler(reqs)
                case HandlerId.HEARTBEAT.value:
                    raise RuntimeError(
                        "Heartbeat should not be handled by the worker thread"
                    )
                case _:
                    print(f"[!] Unknown handler ID: {handler_id}", file=sys.stderr)

            if resps:
                response_queue.put(
                    (client_identity, corr_id_bytes, handler_id_bytes, resps)
                )
    except Exception as exc:  # pylint: disable=broad-exception-caught
        terminate(f"Unhandled error occurred in the worker thread: {exc}")


def zmq_response_thread(response_queue: queue.Queue, socket: zmq.Socket) -> None:
    """Thread that sends responses to the controller."""

    msgpack_encoder = msgspec.msgpack.Encoder()
    try:
        while True:
            client_identity, corr_id_bytes, handler_id_bytes, resps = (
                response_queue.get()
            )
            response_msg = [client_identity, corr_id_bytes, handler_id_bytes] + [
                msgpack_encoder.encode(r) for r in resps
            ]
            socket.send_multipart(response_msg)
    except zmq.error.ZMQError as exc:
        # Terminate the thread if the context is terminated or the socket is not valid
        # This is a normal shutdown scenario initiated by `start_service`
        if exc.errno in {zmq.ETERM, zmq.ENOTSOCK}:
            return
        terminate(f"Unhandled error occurred in the ZMQ response thread: {exc}")
    except Exception as exc:  # pylint: disable=broad-exception-caught
        terminate(f"Unhandled error occurred in the ZMQ response thread: {exc}")


def zmq_listen_thread(
    heartbeat_request_queue: queue.Queue,
    work_request_queue: queue.Queue,
    socket: zmq.Socket,
) -> None:
    """Thread that listens for incoming requests from the controller and
    dispatches them to the appropriate handler."""

    decoders = {
        HandlerId.HANDSHAKE.value: msgspec.msgpack.Decoder(HandshakeRequest),
        HandlerId.HEARTBEAT.value: msgspec.msgpack.Decoder(HeartbeatRequest),
        HandlerId.QUERY.value: msgspec.msgpack.Decoder(QueryRequest),
        HandlerId.FORWARD_PASS.value: msgspec.msgpack.Decoder(ForwardPassRequest),
        HandlerId.EMBED_IMAGE.value: msgspec.msgpack.Decoder(EmbedImageRequest),
        HandlerId.INITIALIZE_ADAPTER.value: msgspec.msgpack.Decoder(
            InitializeAdapterRequest
        ),
        HandlerId.UPDATE_ADAPTER.value: msgspec.msgpack.Decoder(UpdateAdapterRequest),
        HandlerId.UPLOAD_HANDLER.value: msgspec.msgpack.Decoder(UploadAdapterRequest),
        HandlerId.DOWNLOAD_HANDLER.value: msgspec.msgpack.Decoder(
            DownloadAdapterRequest
        ),
    }

    try:
        while True:
            # Block until a message is received
            message = socket.recv_multipart()

            if len(message) < 3:
                print(f"[!] Received invalid message: {message}", file=sys.stderr)
                continue

            client_identity, corr_id_bytes, handler_id_bytes = message[:3]
            try:
                # corr_id extracted but not used
                _ = struct.unpack(">I", corr_id_bytes)[0]
                handler_id = struct.unpack(">I", handler_id_bytes)[0]
                reqs = [decoders[handler_id].decode(m) for m in message[3:]]
            except (struct.error, KeyError, msgspec.DecodeError) as exc:
                print(
                    f"[!] Error decoding request header or payload: {exc}",
                    file=sys.stderr,
                )
                continue

            if not reqs:
                print("[!] Received empty request body", file=sys.stderr)
                continue

            # Dispatch the heartbeat request to the heartbeat thread and all other requests
            # to the worker thread
            if handler_id == HandlerId.HEARTBEAT.value:
                heartbeat_request_queue.put(
                    (client_identity, corr_id_bytes, handler_id_bytes, reqs)
                )
            else:
                work_request_queue.put(
                    (client_identity, corr_id_bytes, handler_id_bytes, handler_id, reqs)
                )
    except zmq.error.ZMQError as exc:
        # Terminate the thread if the context is terminated or the socket is not valid
        # This is a normal shutdown scenario initiated by `start_service`
        if exc.errno in {zmq.ETERM, zmq.ENOTSOCK}:
            return
        terminate(f"Unhandled error occurred in the ZMQ listen loop: {exc}")
    except Exception as exc:  # pylint: disable=broad-exception-caught
        terminate(f"Unhandled error occurred in the ZMQ listen loop: {exc}")


def terminate(msg: str) -> None:
    """Terminate the program with a message."""
    print(f"\n[!!!] {msg} Terminating.", file=sys.stderr)
    traceback.print_exc()
    os._exit(1)


__all__ = [
    "HandlerId",
    "build_config",
    "print_config",
    "resolve_cache_dir",
    "zmq_listen_thread",
    "zmq_response_thread",
    "worker_thread",
    "heartbeat_thread",
    "register_thread",
    "start_service",
    "terminate",
]
