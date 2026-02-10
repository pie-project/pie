"""Inferlet submission and event streaming for Pie.

Provides `run()` — a synchronous, blocking call that connects to a Pie engine,
optionally installs a local WASM program, launches the inferlet, and streams
its output until completion.

Moved from pie_cli.inferlet — now part of the engine package.
"""

import sys
import asyncio
from pathlib import Path
from typing import Any, Callable, Optional


def run(
    client_config: dict,
    inferlet_name: str,
    arguments: list[str],
    *,
    wasm_path: Path | None = None,
    manifest_path: Path | None = None,
    server_handle: Any = None,
    backend_processes: list | None = None,
    on_event: Optional[Callable[[str, str], None]] = None,
) -> None:
    """Run an inferlet and block until it completes.

    Connects to the Pie engine, optionally installs a local WASM program
    (if wasm_path and manifest_path are provided), launches the process,
    and streams output events until termination.

    Args:
        client_config: Dict with ``host``, ``port``, ``internal_auth_token``.
        inferlet_name: Inferlet identifier (e.g. ``"text-completion@0.1.0"``).
        arguments: Command-line arguments forwarded to the inferlet.
        wasm_path: Optional local ``.wasm`` binary. If provided together with
            ``manifest_path``, the program is installed before launching.
        manifest_path: Optional ``Pie.toml`` manifest (must pair with ``wasm_path``).
        server_handle: Optional engine process handle for health monitoring.
        backend_processes: Optional backend process list for health monitoring.
        on_event: Callback ``(event_type, message) -> None``. Defaults to ``print``.
    """
    asyncio.run(
        _run_async(
            client_config,
            inferlet_name,
            arguments,
            wasm_path=wasm_path,
            manifest_path=manifest_path,
            server_handle=server_handle,
            backend_processes=backend_processes,
            on_event=on_event,
        )
    )

# =============================================================================
# Internals
# =============================================================================


async def _run_async(
    client_config: dict,
    inferlet_name: str,
    arguments: list[str],
    *,
    wasm_path: Path | None,
    manifest_path: Path | None,
    server_handle,
    backend_processes,
    on_event,
) -> None:
    from pie_client import PieClient

    def emit(kind: str, msg: str):
        if on_event:
            on_event(kind, msg)
        else:
            print(msg)

    uri = _build_uri(client_config)
    token = client_config.get("internal_auth_token")
    monitor = _start_monitor(server_handle, backend_processes)

    try:
        async with PieClient(uri) as client:
            await client.auth_by_token(token)

            # Install from local path if provided
            if wasm_path is not None and manifest_path is not None:
                if not wasm_path.exists():
                    raise FileNotFoundError(f"Inferlet not found: {wasm_path}")
                if not manifest_path.exists():
                    raise FileNotFoundError(f"Manifest not found: {manifest_path}")

                if not await client.check_program(
                    inferlet_name, wasm_path, manifest_path
                ):
                    emit("info", "Installing inferlet...")
                    await client.install_program(wasm_path, manifest_path)
                else:
                    emit("info", "Inferlet already cached on server.")

            # Launch and stream
            emit("info", f"Launching {inferlet_name}...")
            process = await client.launch_process(
                inferlet_name,
                arguments=arguments,
                capture_outputs=True,
            )
            emit("info", f"Process started: {process.process_id}")

            await _stream_events(process, monitor, emit)
    finally:
        _cancel_monitor(monitor)


# =============================================================================
# Event streaming
# =============================================================================


async def _stream_events(process, monitor_task, emit) -> None:
    """Stream events from a process until it returns or errors."""
    from pie_client import Event

    while True:
        recv_task = asyncio.create_task(process.recv())
        tasks = [recv_task]
        if monitor_task:
            tasks.append(monitor_task)

        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

        if monitor_task in done:
            recv_task.cancel()
            monitor_task.result()

        if recv_task not in done:
            continue

        event, value = recv_task.result()

        if event == Event.Stdout:
            print(value, end="", flush=True)
        elif event == Event.Stderr:
            print(value, end="", file=sys.stderr, flush=True)
        elif event == Event.Message:
            emit("message", f"[Message] {value}")
        elif event == Event.Return:
            emit("completed", f"{value}")
            break
        elif event == Event.Error:
            emit("error", f"❌ {value}")
            break
        elif event == Event.File:
            emit("file", f"[Received file: {len(value)} bytes]")


# =============================================================================
# Helpers
# =============================================================================


def _build_uri(client_config: dict) -> str:
    host = client_config.get("host", "127.0.0.1")
    port = client_config.get("port", 8080)
    return f"ws://{host}:{port}"


def _start_monitor(server_handle, backend_processes) -> asyncio.Task | None:
    if not backend_processes:
        return None
    return asyncio.create_task(
        _monitor_processes(server_handle, backend_processes)
    )


def _cancel_monitor(task: asyncio.Task | None):
    if task and not task.done():
        task.cancel()


async def _monitor_processes(server_handle, backend_processes):
    """Poll backend process health. Raises on unexpected death."""
    while True:
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
