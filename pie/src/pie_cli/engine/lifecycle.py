"""Unified engine lifecycle for Pie.

Consolidates the duplicated start → work → shutdown pattern from
serve.py, run.py, and http.py into a single `run()` function.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional

import typer
from rich.console import Console

from pie_cli.config.schema import EngineConfig


class RunMode(Enum):
    """Engine operating mode."""

    SERVE = "serve"
    RUN = "run"
    HTTP = "http"


@dataclass
class InferletSpec:
    """Specification for an inferlet to run."""

    name: str | None = None
    wasm_path: Path | None = None
    manifest_path: Path | None = None
    arguments: list[str] = field(default_factory=list)
    http_port: int | None = None


def run(
    config: EngineConfig,
    mode: RunMode,
    *,
    inferlet: InferletSpec | None = None,
    monitor: bool = False,
    console: Console | None = None,
    on_status: Callable[[str], None] | None = None,
) -> None:
    """Unified engine lifecycle: start → work → shutdown.

    This replaces the 3 nearly-identical try/except blocks in serve/run/http.

    Args:
        config: Typed engine configuration.
        mode: Whether to serve, run, or launch HTTP.
        inferlet: Inferlet specification (required for RUN and HTTP modes).
        monitor: Launch TUI monitor (SERVE mode only).
        console: Rich console for output.
        on_status: Status callback for engine startup progress.
    """
    from pie_cli.engine.process import start, check, terminate

    if console is None:
        console = Console()

    engine_config, model_configs = config.to_legacy_dicts()

    server_handle = None
    backend_processes = []

    try:
        server_handle, backend_processes = start(
            engine_config, model_configs,
            console=console, on_status=on_status,
        )
        console.print()

        if mode == RunMode.SERVE:
            _run_serve(config, server_handle, backend_processes, monitor, console)
        elif mode == RunMode.RUN:
            _run_inferlet(config, server_handle, backend_processes, inferlet, console)
        elif mode == RunMode.HTTP:
            _run_http(config, server_handle, backend_processes, inferlet, console)

        # Normal shutdown
        console.print()
        with console.status("[dim]Shutting down...[/dim]"):
            terminate(server_handle, backend_processes)
        console.print("[green]✓[/green] Shutdown complete")

    except KeyboardInterrupt:
        console.print()
        with console.status("[dim]Shutting down...[/dim]"):
            terminate(server_handle, backend_processes)
        console.print("[green]✓[/green] Shutdown complete")
    except Exception as e:
        console.print(f"[red]✗[/red] {e}")
        terminate(server_handle, backend_processes)
        raise typer.Exit(1)


# =============================================================================
# Mode implementations
# =============================================================================


def _run_serve(
    config: EngineConfig,
    server_handle: Any,
    backend_processes: list,
    monitor: bool,
    console: Console,
) -> None:
    """SERVE mode: either launch TUI monitor or poll indefinitely."""
    from pie_cli.engine.process import check

    if monitor:
        from pie_cli.monitor.app import LLMMonitorApp
        from pie_cli.monitor.provider import PieMetricsProvider

        model = config.primary_model
        provider = PieMetricsProvider(
            host=config.host,
            port=config.port,
            internal_token=server_handle.internal_token,
            config={
                "model": model.name or model.hf_repo,
                "tp_size": model.tensor_parallel_size or len(model.device),
                "max_batch": model.max_batch_tokens or 32,
            },
        )
        provider.start()

        app = LLMMonitorApp(provider=provider)
        app.run()

        provider.stop()
    else:
        try:
            while True:
                if not check(backend_processes):
                    console.print("[red]A backend process died. Shutting down.[/red]")
                    break

                if server_handle and hasattr(server_handle, "is_running"):
                    if not server_handle.is_running():
                        console.print("[red]Engine process died. Shutting down.[/red]")
                        break

                time.sleep(1.0)
        except KeyboardInterrupt:
            pass


def _run_inferlet(
    config: EngineConfig,
    server_handle: Any,
    backend_processes: list,
    spec: InferletSpec | None,
    console: Console,
) -> None:
    """RUN mode: run a one-shot inferlet and return on completion."""
    from pie_cli.engine import client as inferlet_client

    if spec is None:
        raise ValueError("InferletSpec required for RUN mode")

    client_config = {
        "host": config.host,
        "port": config.port,
        "internal_auth_token": server_handle.internal_token,
    }

    # Resolve inferlet name from manifest if using local path
    inferlet_name = spec.name
    if spec.wasm_path is not None and spec.manifest_path is not None:
        import tomllib

        manifest = tomllib.loads(spec.manifest_path.read_text())
        name = manifest["package"]["name"]
        version = manifest["package"]["version"]
        inferlet_name = f"{name}@{version}"

    inferlet_client.run(
        client_config,
        inferlet_name,
        spec.arguments,
        wasm_path=spec.wasm_path,
        manifest_path=spec.manifest_path,
        server_handle=server_handle,
        backend_processes=backend_processes,
    )


def _run_http(
    config: EngineConfig,
    server_handle: Any,
    backend_processes: list,
    spec: InferletSpec | None,
    console: Console,
) -> None:
    """HTTP mode: launch a server inferlet and wait indefinitely."""
    import asyncio
    from pie_client import PieClient

    if spec is None:
        raise ValueError("InferletSpec required for HTTP mode")
    if spec.http_port is None:
        raise ValueError("http_port required for HTTP mode")

    async def _launch():
        uri = f"ws://{config.host}:{config.port}"
        token = server_handle.internal_token

        async with PieClient(uri) as client:
            await client.auth_by_token(token)

            if spec.wasm_path is not None and spec.manifest_path is not None:
                import tomllib

                manifest = tomllib.loads(spec.manifest_path.read_text())
                name = manifest["package"]["name"]
                version = manifest["package"]["version"]
                inferlet_name = f"{name}@{version}"

                if not await client.check_program(inferlet_name, spec.wasm_path, spec.manifest_path):
                    console.print(f"[dim]Installing {spec.wasm_path.name}...[/dim]")
                    await client.install_program(spec.wasm_path, spec.manifest_path)

                console.print(f"[dim]Starting server on port {spec.http_port}...[/dim]")
                await client.launch_process(
                    inferlet_name,
                    arguments=["--port", str(spec.http_port)] + spec.arguments,
                    capture_outputs=False,
                )
            else:
                raise ValueError("HTTP mode requires --path and --manifest for server inferlets")

    asyncio.run(_launch())

    console.print(
        f"[green]✓[/green] Server inferlet listening on "
        f"[cyan]http://127.0.0.1:{spec.http_port}/[/cyan]"
    )
    console.print("[dim]Press Ctrl+C to stop[/dim]")
    console.print()

    # Wait indefinitely until interrupted
    while True:
        time.sleep(1)
