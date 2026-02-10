"""
RPC server for PIE backend communication.

This module provides the RPC loop that handles requests from the Rust runtime
via IPC. Each group leader runs a poll loop on its RpcServer instance.
"""

from __future__ import annotations

import threading

import msgpack

from .backend import Backend

# Status codes for RPC dispatch (must match Rust)
STATUS_OK = 0
STATUS_METHOD_NOT_FOUND = 1
STATUS_INVALID_PARAMS = 2
STATUS_INTERNAL_ERROR = 3


def poll_rpc_server(
    server, service: Backend, stop_event: threading.Event, poll_timeout_ms: int = 100
) -> None:
    """Poll an RpcServer and process requests.

    This is the main worker loop that polls a Python-hosted RpcServer
    for incoming requests from the Rust runtime. The server was created
    via `_pie.RpcServer.create()` and its `server_name` was passed to
    Rust through the Config so Rust can connect as an RpcClient.

    Args:
        server: _pie.RpcServer instance
        service: Backend instance to dispatch calls to
        stop_event: Event to signal shutdown
        poll_timeout_ms: How long to block waiting for requests (ms)
    """
    # Method dispatch table
    methods = {
        "query": service.query_rpc,
        "fire_batch": service.fire_batch,
        "embed_image": service.embed_image_rpc,
        "initialize_adapter": service.initialize_adapter_rpc,
        "update_adapter": service.update_adapter_rpc,
        "upload_adapter": service.upload_adapter_rpc,
        "download_adapter": service.download_adapter_rpc,
    }

    try:
        while not stop_event.is_set():
            # Poll the RPC server (releases GIL while waiting)
            request = server.poll_blocking(poll_timeout_ms)
            if request is None:
                continue  # Timeout, try again

            request_id, method, payload = request

            try:
                # Unpack args
                args = msgpack.unpackb(payload)

                # Get handler
                fn = methods.get(method)
                if fn is None:
                    response = msgpack.packb(f"Method not found: {method}")
                    server.respond(request_id, response)
                    continue

                # Call handler
                if isinstance(args, dict):
                    result = fn(**args)
                elif isinstance(args, (list, tuple)):
                    result = fn(*args)
                else:
                    result = fn(args)

                # Pack and respond
                response = msgpack.packb(result)
                server.respond(request_id, response)

            except Exception as e:
                import traceback

                tb = traceback.format_exc()
                print(f"[RPC Server Error] {method}: {e}\n{tb}")
                response = msgpack.packb(str(e))
                server.respond(request_id, response)
    finally:
        # Ensure cleanup when thread stops
        print("[RPC Worker] Shutting down Backend...")
        service.shutdown()


def start_rpc_worker(
    server, service: Backend, thread_name: str = "pie-rpc-worker"
) -> tuple[threading.Thread, threading.Event]:
    """Start an RPC worker thread that polls the RpcServer.

    Args:
        server: _pie.RpcServer instance
        service: Backend instance to dispatch calls to
        thread_name: Name for the worker thread (for debugging)

    Returns:
        tuple (thread, stop_event) where thread is already started.
    """
    stop_event = threading.Event()

    def worker():
        poll_rpc_server(server, service, stop_event)

    thread = threading.Thread(target=worker, name=thread_name, daemon=True)
    thread.start()
    return thread, stop_event
