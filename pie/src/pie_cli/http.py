"""HTTP command implementation for Pie CLI.

Implements: pie http <inferlet> --port <port> [args]
Launches a server inferlet that listens on the specified port.
"""

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from . import serve as serve_module

console = Console()


def http(
    inferlet: Optional[str] = typer.Argument(
        None, help="Inferlet name from registry (e.g., 'std/http-server@0.1.0')"
    ),
    path: Optional[Path] = typer.Option(
        None, "--path", "-p", help="Path to a local .wasm server inferlet file"
    ),
    port: int = typer.Option(
        ..., "--port", help="TCP port for the server to listen on"
    ),
    config: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to TOML configuration file"
    ),
    log: Optional[Path] = typer.Option(None, "--log", help="Path to log file"),
    dummy: bool = typer.Option(
        False, "--dummy", help="Enable dummy mode (skip GPU weight loading, return random tokens)"
    ),
    arguments: Optional[list[str]] = typer.Argument(
        None, help="Arguments to pass to the inferlet"
    ),
) -> None:
    """Launch a server inferlet on a specific port.

    Server inferlets implement the wasi:http/incoming-handler interface and
    handle incoming HTTP requests. Unlike regular inferlets, they are long-running
    and create a new WASM instance for each incoming request.

    You can specify an inferlet either by registry name or by path (mutually exclusive):

    \b
    - By path: pie http --path ./server.wasm --port 8080
    - By registry: pie http std/http-server@0.1.0 --port 8080

    The server will run until interrupted with Ctrl+C.
    """
    # Validate at least one of inferlet or path is provided
    if inferlet is None and path is None:
        console.print("[red]✗[/red] Specify an inferlet name or --path")
        raise typer.Exit(1)

    # Handle the case where --path is used with -- separator
    if inferlet is not None and path is not None:
        arguments = [inferlet] + (arguments or [])
        inferlet = None

    # Verify inferlet exists if using path
    if path is not None and not path.exists():
        console.print(f"[red]✗[/red] File not found: {path}")
        raise typer.Exit(1)

    try:
        engine_config, model_configs = serve_module.load_config(
            config,
            log_dir=str(log.parent) if log else None,
            dummy_mode=dummy,
        )
    except typer.Exit:
        raise

    console.print()

    # Show run info
    inferlet_display = str(path) if path else inferlet
    lines = Text()
    lines.append(f"{'Inferlet':<15}", style="white")
    lines.append(f"{inferlet_display}\n", style="dim")
    lines.append(f"{'Port':<15}", style="white")
    lines.append(f"{port}\n", style="cyan bold")
    lines.append(f"{'Model':<15}", style="white")
    lines.append(f"{model_configs[0].get('hf_repo', 'unknown')}\n", style="dim")
    lines.append(f"{'Device':<15}", style="white")
    device = model_configs[0].get("device", ["unknown"])
    device_str = ", ".join(device) if isinstance(device, list) else device
    lines.append(device_str, style="dim")

    console.print(
        Panel(lines, title="Pie HTTP Server", title_align="left", border_style="dim")
    )
    console.print()

    server_handle = None
    backend_processes = []

    try:
        # Start engine and backends
        from pie import manager

        server_handle, backend_processes = manager.start_engine_and_backend(
            engine_config, model_configs, console=console
        )

        console.print()

        # Launch the server inferlet
        client_config = {
            "host": engine_config["host"],
            "port": engine_config["port"],
            "internal_auth_token": server_handle.internal_token,
        }

        if path is not None:
            _launch_server_inferlet(
                client_config, path, port, arguments or []
            )
        else:
            # TODO: Support launching from registry
            console.print("[red]✗[/red] Registry mode not yet supported for server inferlets")
            raise typer.Exit(1)

        console.print(f"[green]✓[/green] Server inferlet listening on [cyan]http://127.0.0.1:{port}/[/cyan]")
        console.print("[dim]Press Ctrl+C to stop[/dim]")
        console.print()

        # Wait indefinitely until interrupted
        import time
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        console.print()
        console.print("[yellow]![/yellow] Interrupted")
        with console.status("[dim]Shutting down...[/dim]"):
            manager.terminate_engine_and_backend(server_handle, backend_processes)
        console.print("[green]✓[/green] Server stopped")
        raise typer.Exit(0)
    except Exception as e:
        from pie import manager

        if isinstance(e, manager.EngineError):
            console.print(f"[red]✗[/red] {e}")
            manager.terminate_engine_and_backend(server_handle, backend_processes)
            raise typer.Exit(1)
        console.print(f"[red]✗[/red] Error: {e}")
        manager.terminate_engine_and_backend(server_handle, backend_processes)
        raise typer.Exit(1)


def _launch_server_inferlet(
    client_config: dict,
    path: Path,
    port: int,
    arguments: list[str],
) -> None:
    """Upload and launch a server inferlet from a local path."""
    import asyncio
    import blake3
    from pie_client import PieClient

    async def _launch():
        host = client_config["host"]
        engine_port = client_config["port"]
        uri = f"ws://{host}:{engine_port}"

        async with PieClient(uri) as client:
            await client.internal_authenticate(client_config["internal_auth_token"])

            # Read and hash the program
            program_bytes = path.read_bytes()
            program_hash = blake3.blake3(program_bytes).hexdigest()

            # Upload if needed
            if not await client.program_exists(program_hash):
                console.print(f"[dim]Uploading {path.name}...[/dim]")
                await client.upload_program(program_bytes)

            # Launch as server instance
            console.print(f"[dim]Starting server on port {port}...[/dim]")
            await client.launch_server_instance(program_hash, port, arguments)

    asyncio.run(_launch())
