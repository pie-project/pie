"""Inferlet submission and event streaming for Pie.

Client-side module for submitting inferlets (local or from registry)
and streaming their output events.
"""

import sys
import asyncio
from pathlib import Path
from typing import Optional, Any


def submit_and_wait(
    client_config: dict,
    inferlet_path: Path,
    manifest_path: Path,
    arguments: list[str],
    server_handle: Any = None,
    backend_processes: list | None = None,
    on_event: Optional[callable] = None,
) -> None:
    """Submit a local inferlet and wait for it to finish.

    Args:
        client_config: Client configuration with host, port, internal_auth_token
        inferlet_path: Path to the .wasm inferlet file
        manifest_path: Path to the manifest TOML file
        arguments: Arguments to pass to the inferlet
        server_handle: Optional server handle for process monitoring
        backend_processes: Optional list of backend processes to monitor
        on_event: Optional callback: (event_type: str, message: str) -> None
    """
    asyncio.run(
        _submit_local_async(
            client_config,
            inferlet_path,
            manifest_path,
            arguments,
            server_handle,
            backend_processes,
            on_event,
        )
    )


def submit_from_registry_and_wait(
    client_config: dict,
    inferlet_name: str,
    arguments: list[str],
    server_handle: Any = None,
    backend_processes: list | None = None,
    on_event: Optional[callable] = None,
) -> None:
    """Submit an inferlet from the registry and wait for it to finish.

    Args:
        client_config: Client configuration with host, port, internal_auth_token
        inferlet_name: Inferlet name (e.g., "text-completion@0.1.0")
        arguments: Arguments to pass to the inferlet
        server_handle: Optional server handle for process monitoring
        backend_processes: Optional list of backend processes to monitor
        on_event: Optional callback: (event_type: str, message: str) -> None
    """
    asyncio.run(
        _submit_registry_async(
            client_config,
            inferlet_name,
            arguments,
            server_handle,
            backend_processes,
            on_event,
        )
    )


# =============================================================================
# Async Implementations
# =============================================================================


async def _submit_local_async(
    client_config: dict,
    inferlet_path: Path,
    manifest_path: Path,
    arguments: list[str],
    server_handle,
    backend_processes,
    on_event,
) -> None:
    """Async implementation for local inferlet submission."""
    import tomllib
    from pie_client import PieClient

    def emit(event_type: str, msg: str):
        if on_event:
            on_event(event_type, msg)
        else:
            print(msg)

    if not inferlet_path.exists():
        raise FileNotFoundError(f"Inferlet not found: {inferlet_path}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    uri = _build_uri(client_config)
    token = client_config.get("internal_auth_token")

    monitor_task = _start_monitor(server_handle, backend_processes)

    try:
        async with PieClient(uri) as client:
            await client.auth_by_token(token)

            # Parse manifest for inferlet name
            manifest = tomllib.loads(manifest_path.read_text())
            name = manifest["package"]["name"]
            version = manifest["package"]["version"]
            inferlet_name = f"{name}@{version}"
            emit("info", f"Inferlet: {inferlet_name}")

            # Install if needed
            if not await client.program_exists(inferlet_name, inferlet_path, manifest_path):
                emit("info", "Installing inferlet...")
                await client.install_program(inferlet_path, manifest_path)
            else:
                emit("info", "Inferlet already cached on server.")

            # Launch and stream
            emit("info", f"Launching {inferlet_path.name}...")
            instance = await client.launch_process(
                inferlet_name,
                arguments=arguments,
                capture_outputs=True,
            )
            emit("info", f"Instance launched: {instance.instance_id}")

            await _stream_events(instance, monitor_task, emit)
    finally:
        _cancel_monitor(monitor_task)


async def _submit_registry_async(
    client_config: dict,
    inferlet_name: str,
    arguments: list[str],
    server_handle,
    backend_processes,
    on_event,
) -> None:
    """Async implementation for registry inferlet submission."""
    from pie_client import PieClient

    def emit(event_type: str, msg: str):
        if on_event:
            on_event(event_type, msg)
        else:
            print(msg)

    uri = _build_uri(client_config)
    token = client_config.get("internal_auth_token")

    monitor_task = _start_monitor(server_handle, backend_processes)

    try:
        async with PieClient(uri) as client:
            await client.auth_by_token(token)

            instance = await client.launch_process_from_registry(
                inferlet=inferlet_name,
                arguments=arguments,
                capture_outputs=True,
            )

            await _stream_events(instance, monitor_task, emit)
    finally:
        _cancel_monitor(monitor_task)


# =============================================================================
# Shared Helpers
# =============================================================================


def _build_uri(client_config: dict) -> str:
    """Build WebSocket URI from client config."""
    host = client_config.get("host", "127.0.0.1")
    port = client_config.get("port", 8080)
    return f"ws://{host}:{port}"


def _start_monitor(server_handle, backend_processes) -> asyncio.Task | None:
    """Start process health monitor if processes are provided."""
    if not backend_processes:
        return None
    return asyncio.create_task(
        _monitor_processes_task(server_handle, backend_processes)
    )


def _cancel_monitor(monitor_task: asyncio.Task | None):
    """Cancel the monitor task if running."""
    if monitor_task and not monitor_task.done():
        monitor_task.cancel()


async def _stream_events(instance, monitor_task, emit) -> None:
    """Stream events from an inferlet instance until completion.

    This is the shared event loop used by both local and registry submissions.
    """
    from pie_client import Event

    while True:
        recv_task = asyncio.create_task(instance.recv())
        tasks = [recv_task]
        if monitor_task:
            tasks.append(monitor_task)

        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

        if monitor_task in done:
            recv_task.cancel()
            monitor_task.result()  # Propagate any exception

        if recv_task not in done:
            continue
        event, message = recv_task.result()

        if event == Event.Stdout:
            print(message, end="", flush=True)
        elif event == Event.Stderr:
            print(message, end="", file=sys.stderr, flush=True)
        elif event == Event.Message:
            emit("message", f"[Message] {message}")
        elif event == Event.Completed:
            emit("completed", f"{message}")
            break
        elif event == Event.Aborted:
            emit("aborted", f"⚠️ Instance aborted: {message}")
            break
        elif event == Event.Exception:
            emit("exception", f"❌ Instance exception: {message}")
            break
        elif event == Event.ServerError:
            emit("error", f"❌ Server error: {message}")
            break
        elif event == Event.OutOfResources:
            emit("error", f"❌ Out of resources: {message}")
            break
        elif event == Event.Blob:
            emit("blob", f"[Received blob: {len(message)} bytes]")
        else:
            emit("unknown", f"[Unknown event {event}]: {message}")


async def _monitor_processes_task(server_handle, backend_processes):
    """Async task to monitor backend process health."""
    if not backend_processes:
        return

    while True:
        # Check SpawnContext processes
        for ctx in backend_processes:
            if hasattr(ctx, "processes"):
                for p in ctx.processes:
                    if not p.is_alive() and p.exitcode != 0:
                        raise RuntimeError(
                            f"Worker process {p.pid} died (exit code {p.exitcode})"
                        )

        if server_handle and hasattr(server_handle, "is_running"):
            if not server_handle.is_running():
                raise RuntimeError("Engine process died")

        await asyncio.sleep(1.0)
