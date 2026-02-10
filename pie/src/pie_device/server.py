"""
FFI server for PIE backend communication.

This module provides the RPC endpoint that handles requests from the Rust runtime
using direct FFI calls via PyO3.
"""

from __future__ import annotations

import queue
import threading

import msgpack

from .runtime import Runtime

# Status codes for FFI dispatch (must match Rust)
STATUS_OK = 0
STATUS_METHOD_NOT_FOUND = 1
STATUS_INVALID_PARAMS = 2
STATUS_INTERNAL_ERROR = 3


def poll_ffi_queue(
    ffi_queue, service: Runtime, stop_event: threading.Event, poll_timeout_ms: int = 100
) -> None:
    """Poll the Rust FfiQueue and process requests.

    This is the new high-performance worker loop that polls a Rust queue
    directly without Python queue overhead. Should be called from a dedicated
    Python thread that owns all CUDA state.

    Args:
        ffi_queue: _pie.FfiQueue instance from start_server_with_ffi
        service: Runtime instance to dispatch calls to
        stop_event: Event to signal shutdown
        poll_timeout_ms: How long to block waiting for requests (ms)
    """
    # Method dispatch table
    methods = {
        "handshake": service.handshake_rpc,
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
            # Poll the Rust queue (releases GIL while waiting)
            request = ffi_queue.poll_blocking(poll_timeout_ms)
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
                    ffi_queue.respond(request_id, response)
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
                ffi_queue.respond(request_id, response)

            except Exception as e:
                import traceback

                tb = traceback.format_exc()
                print(f"[FFI Queue Error] {method}: {e}\n{tb}")
                response = msgpack.packb(str(e))
                ffi_queue.respond(request_id, response)
    finally:
        # Ensure cleanup when thread stops
        print("[FFI Worker] Shutting down Runtime...")
        service.shutdown()


def start_ffi_worker(
    ffi_queue, service: Runtime, thread_name: str = "pie-ffi-worker"
) -> tuple[threading.Thread, threading.Event]:
    """Start the FFI worker thread that polls the Rust queue.

    Args:
        ffi_queue: _pie.FfiQueue instance
        service: Runtime instance to dispatch calls to
        thread_name: Name for the worker thread (for debugging)

    Returns:
        tuple (thread, stop_event) where thread is already started.
    """
    stop_event = threading.Event()

    def worker():
        poll_ffi_queue(ffi_queue, service, stop_event)

    thread = threading.Thread(target=worker, name=thread_name, daemon=True)
    thread.start()
    return thread, stop_event
