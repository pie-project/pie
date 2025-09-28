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
import threading
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Type

import msgpack
import msgspec
import torch
import zmq
from platformdirs import user_cache_dir
from websockets.sync.client import connect

from config.common import ModelInfo
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


@dataclass
class ServerConfig:
    """Lightweight view of server configuration options."""

    model: str
    host: str
    port: int
    controller_host: str
    controller_port: int
    auth_token: str | None
    cache_dir: str | None
    kv_page_size: int
    max_dist_size: int
    max_num_kv_pages: int
    max_num_embeds: int
    max_num_adapters: int
    max_adapter_rank: int
    device: str
    dtype: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return self.__dict__.copy()


def resolve_cache_dir(cache_dir: str | None) -> str:
    """Resolve the cache directory using CLI arg > env var > default."""

    return cache_dir or os.environ.get("PIE_HOME") or str(Path(user_cache_dir("pie")))


def build_config(**kwargs: Any) -> Dict[str, Any]:
    """Normalize server configuration dictionary and resolve cache directory."""

    config = dict(kwargs)
    config["cache_dir"] = resolve_cache_dir(config.get("cache_dir"))
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
    model: Any,
    model_info: ModelInfo,
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
        model=model,
        model_info=model_info,
        kv_page_size=config["kv_page_size"],
        max_dist_size=config["max_dist_size"],
        max_num_kv_pages=config["max_num_kv_pages"],
        max_num_embeds=config["max_num_embeds"],
        max_batch_tokens=config["max_batch_tokens"],
        max_num_adapters=config["max_num_adapters"],
        max_adapter_rank=config["max_adapter_rank"],
        dtype=getattr(torch, config["dtype"]),
        device=config["device"],
    )

    context = zmq.Context()
    socket = context.socket(zmq.ROUTER)
    socket.bind(real_endpoint)

    threading.Thread(target=run_zmq_server, args=(socket, handler), daemon=True).start()

    if register_with_controller:
        threading.Thread(
            target=register,
            args=(config, endpoint),
            daemon=True,
        ).start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down server...")
    finally:
        socket.close()
        context.term()
        print("Server shutdown complete.")


def register(config: Dict[str, Any], endpoint: str) -> None:
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


def run_zmq_server(socket: zmq.Socket, handler: Any) -> None:
    """Core ZMQ service loop dispatching requests to the handler.

    Exits the program if a heartbeat is not received for 60 seconds or if any
    exception occurs.
    """
    # Heartbeat timeout and timer setup (60 seconds for FlashInfer JIT compilation)
    heartbeat_timeout = 60  # seconds
    last_heartbeat_time = time.monotonic()

    msgpack_encoder = msgspec.msgpack.Encoder()
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

    poller = zmq.Poller()
    poller.register(socket, zmq.POLLIN)

    try:
        while True:
            # Check for heartbeat timeout before waiting for a message
            if time.monotonic() - last_heartbeat_time > heartbeat_timeout:
                print(
                    f"[!] Heartbeat timeout after {heartbeat_timeout}s, exiting",
                    file=sys.stderr,
                )
                os._exit(1)  # Use os._exit for immediate termination from a thread

            # Poll for 1 second to remain responsive to the heartbeat check
            events = dict(poller.poll(timeout=1000))
            if socket in events:
                message = socket.recv_multipart()
            else:
                # Poller timed out, loop again to re-check the heartbeat timer
                continue

            if len(message) < 3:
                print(f"[!] Received invalid message: {message}", file=sys.stderr)
                continue

            client_identity, corr_id_bytes, handler_id_bytes = message[:3]
            try:
                _ = struct.unpack(">I", corr_id_bytes)[
                    0
                ]  # corr_id extracted but not used
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

            resps = []
            match handler_id:
                case HandlerId.HANDSHAKE.value:
                    resps = handler.handshake(reqs)
                case HandlerId.HEARTBEAT.value:
                    # Update heartbeat timer when heartbeat is received
                    last_heartbeat_time = time.monotonic()
                    resps = handler.heartbeat(reqs)
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
                case _:
                    print(f"[!] Unknown handler ID: {handler_id}", file=sys.stderr)

            if resps:
                response_msg = [client_identity, corr_id_bytes, handler_id_bytes] + [
                    msgpack_encoder.encode(r) for r in resps
                ]
                socket.send_multipart(response_msg)

    except (zmq.ZMQError, OSError, ValueError, RuntimeError, KeyError) as exc:
        print(
            f"\n[!!!] Unhandled error occurred in the ZMQ server loop: {exc}",
            file=sys.stderr,
        )
        traceback.print_exc()
        os._exit(1)


__all__ = [
    "HandlerId",
    "ServerConfig",
    "build_config",
    "print_config",
    "resolve_cache_dir",
    "run_zmq_server",
    "start_service",
    "register",
]
