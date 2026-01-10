"""
Pycrust-based server for PIE backend communication.

This module provides the RPC endpoint that handles requests from the Rust runtime
using pycrust (iceoryx2 shared memory IPC).
"""

from __future__ import annotations

import random
import signal
import threading

import msgspec
from websockets.sync.client import connect

from pycrust import RpcEndpoint

from .runtime import Runtime


def start_server(
    host: str,
    port: int,
    auth_token: str,
    service: Runtime,
    *,
    run_tests: bool = False,
    log_queue: object | None = None,
):
    """Spin up the backend service using pycrust RPC.

    Args:
        host: Controller host to register with
        port: Controller port to register with
        auth_token: Authentication token for controller
        service: Runtime implementation to handle requests
        run_tests: If True, spawn embedded test client after server starts
        log_queue: Optional queue for logging to controller
    """

    import os

    # Use PID + time-based unique ID to avoid collisions across forked processes
    unique_id = f"{os.getpid()}-{int(random.random() * 1000000)}"
    service_name = f"pie-backend-{unique_id}"

    # Event to signal shutdown
    shutdown_event = threading.Event()

    # Create pycrust RPC endpoint
    endpoint = RpcEndpoint(service_name)

    # Register RPC methods
    endpoint.register("handshake", service.handshake_rpc)
    endpoint.register("query", service.query_rpc)
    endpoint.register("fire_batch", service.fire_batch)
    endpoint.register("embed_image", service.embed_image_rpc)
    endpoint.register("initialize_adapter", service.initialize_adapter_rpc)
    endpoint.register("update_adapter", service.update_adapter_rpc)
    endpoint.register("upload_adapter", service.upload_adapter_rpc)
    endpoint.register("download_adapter", service.download_adapter_rpc)

    # Start registration thread (connects to controller via WebSocket)
    reg_t = threading.Thread(
        target=register_thread,
        args=(host, port, auth_token, service_name, shutdown_event, log_queue),
        daemon=True,
    )
    reg_t.start()

    # Spawn embedded test client if requested
    test_t = None
    if run_tests:
        from .test_client import run_embedded_tests

        test_t = threading.Thread(
            target=run_embedded_tests,
            args=(service_name,),
            daemon=True,
        )
        test_t.start()

    # Setup shutdown signal handlers
    # IMPORTANT: We raise KeyboardInterrupt instead of just setting an event,
    # because the pycrust worker loop (in Rust) only exits on KeyboardInterrupt.
    # This ensures iceoryx2 resources are properly cleaned up on SIGTERM.
    def shutdown_handler(signum, _frame):
        if not shutdown_event.is_set():
            msg = f"\nReceived signal {signum}, shutting down server..."
            if log_queue:
                log_queue.put({"level": "DEBUG", "message": msg})
            shutdown_event.set()
            # Raise KeyboardInterrupt to make pycrust worker exit cleanly
            # This allows iceoryx2 to cleanup its semaphores properly
            raise KeyboardInterrupt()

    signal.signal(signal.SIGTERM, shutdown_handler)
    signal.signal(signal.SIGINT, shutdown_handler)

    try:
        if log_queue:
            log_queue.put(
                {
                    "level": "DEBUG",
                    "message": f"Starting pycrust endpoint: {service_name}",
                }
            )

        # This blocks until shutdown (Ctrl+C or SIGTERM)
        endpoint.listen()

    except KeyboardInterrupt:
        pass
    finally:
        if log_queue:
            log_queue.put(
                {
                    "level": "DEBUG",
                    "message": "Waiting for background threads to finish...",
                }
            )

        # Shutdown the runtime (stops worker ranks via NCCL)
        try:
            service.shutdown()
        except Exception as e:
            if log_queue:
                log_queue.put(
                    {"level": "ERROR", "message": f"Error during service shutdown: {e}"}
                )

        # Stop remaining threads
        if reg_t and reg_t.is_alive():
            reg_t.join(timeout=2.0)
        if test_t and test_t.is_alive():
            test_t.join(timeout=2.0)

        if log_queue:
            log_queue.put({"level": "DEBUG", "message": "Server shutdown complete."})


def register_thread(
    host: str,
    port: int,
    auth_token: str,
    service_name: str,
    shutdown_event: threading.Event,
    log_queue: object | None = None,
) -> None:
    """Register this service with the controller.

    Args:
        host: Controller host
        port: Controller port
        auth_token: Authentication token
        service_name: Pycrust service name (used by Rust runtime to connect)
        shutdown_event: Event to signal shutdown
        log_queue: Optional queue for logging
    """
    import time

    # Wait for pycrust endpoint to start listening before registering.
    # The endpoint.listen() call happens in the main thread after this
    # thread starts, so we need a small delay to ensure it's ready.
    time.sleep(0.5)

    encoder = msgspec.msgpack.Encoder()
    decoder = msgspec.msgpack.Decoder()

    controller_addr = f"ws://{host}:{port}"
    try:
        with connect(controller_addr) as websocket:
            # Authenticate
            auth_msg = encoder.encode(
                {
                    "type": "internal_authenticate",
                    "corr_id": 0,
                    "token": auth_token,
                }
            )

            if auth_msg is not None:
                websocket.send(auth_msg)

            auth_response = decoder.decode(websocket.recv())
            if not auth_response.get("successful"):
                err_msg = f"Authentication failed: {auth_response.get('result', 'Unknown error')}"
                if log_queue:
                    log_queue.put({"level": "ERROR", "message": err_msg})
                shutdown_event.set()
                return

            # Register service with pycrust service name as endpoint
            reg_msg = encoder.encode(
                {
                    "type": "attach_remote_service",
                    "corr_id": 0,
                    "endpoint": service_name,  # Pycrust service name
                    "service_name": f"service-{random.randint(1000000, 9999999)}",
                    "service_type": "model",
                }
            )

            if reg_msg is not None:
                websocket.send(reg_msg)

            reg_response = decoder.decode(websocket.recv())
            if not reg_response.get("successful"):
                err_msg = f"Controller registration failed: {reg_response.get('result', 'Unknown error')}"
                if log_queue:
                    log_queue.put({"message": err_msg, "level": "ERROR"})
                shutdown_event.set()
                return

            success_msg = f"Registered with controller at {controller_addr} (service: {service_name})"
            if log_queue:
                log_queue.put({"message": success_msg, "level": "DEBUG"})

            # Keep connection alive until shutdown
            while not shutdown_event.is_set():
                shutdown_event.wait(timeout=1.0)

    except (ConnectionRefusedError, TimeoutError) as exc:
        err_msg = (
            f"Failed to connect to the controller at {controller_addr}. Error: {exc}"
        )
        if log_queue:
            log_queue.put({"message": err_msg, "level": "ERROR"})
        shutdown_event.set()

    except (OSError, ValueError, RuntimeError) as exc:
        err_msg = (
            f"An unexpected error occurred during registration: {exc}. Terminating."
        )
        if log_queue:
            log_queue.put({"message": err_msg, "level": "ERROR"})
        shutdown_event.set()
