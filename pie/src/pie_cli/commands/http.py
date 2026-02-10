"""HTTP command implementation for Pie CLI.

Implements: pie http <inferlet> --port <port> [args]
Launches a server inferlet that listens on the specified port.
"""

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from pie_cli.config import load_config
from pie_cli.display import inferlet_panel
from pie_cli.engine.lifecycle import run as engine_run, RunMode, InferletSpec

console = Console()


def http(
    inferlet: Optional[str] = typer.Argument(
        None, help="Inferlet name from registry (e.g., 'http-server@0.1.0')"
    ),
    path: Optional[Path] = typer.Option(
        None, "--path", "-p", help="Path to a local .wasm server inferlet file"
    ),
    port: int = typer.Option(
        ..., "--port", help="TCP port for the server to listen on"
    ),
    manifest: Optional[Path] = typer.Option(
        None, "--manifest", "-m", help="Path to the manifest TOML file (required with --path)"
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
    """Launch a server inferlet on a specific port."""
    # Validate inputs
    if inferlet is None and path is None:
        console.print("[red]✗[/red] Specify an inferlet name or --path")
        raise typer.Exit(1)

    if inferlet is not None and path is not None:
        arguments = [inferlet] + (arguments or [])
        inferlet = None

    if path is not None and not path.exists():
        console.print(f"[red]✗[/red] File not found: {path}")
        raise typer.Exit(1)

    if path is not None and manifest is None:
        console.print("[red]✗[/red] --manifest is required when using --path")
        raise typer.Exit(1)

    if manifest is not None and not manifest.exists():
        console.print(f"[red]✗[/red] Manifest not found: {manifest}")
        raise typer.Exit(1)

    try:
        cfg = load_config(config, dummy_mode=dummy)
    except (FileNotFoundError, ValueError) as e:
        console.print(f"[red]✗[/red] {e}")
        raise typer.Exit(1)

    console.print()
    inferlet_display = str(path) if path else inferlet
    inferlet_panel(
        cfg, inferlet_display,
        title="Pie HTTP Server", http_port=port, console=console,
    )
    console.print()

    spec = InferletSpec(
        name=inferlet,
        wasm_path=path,
        manifest_path=manifest,
        arguments=arguments or [],
        http_port=port,
    )

    engine_run(cfg, RunMode.HTTP, inferlet=spec, console=console)
