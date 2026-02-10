"""Serve command implementation for Pie CLI.

Implements: pie serve
Starts the Pie engine and optionally provides a real-time TUI monitor.
"""

from pathlib import Path

import typer
from rich.console import Console

from pie_cli.config import load_config
from pie_cli.display import engine_panel
from pie_cli.runtime import serve as runtime_serve

console = Console()


def serve(
    config: Path | None = typer.Option(
        None, "--config", "-c", help="Path to TOML configuration file"
    ),
    host: str | None = typer.Option(None, "--host", help="Override host address"),
    port: int | None = typer.Option(None, "--port", help="Override port"),
    no_auth: bool = typer.Option(False, "--no-auth", help="Disable authentication"),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose logging"
    ),
    monitor: bool = typer.Option(
        False, "--monitor", "-m", help="Launch real-time TUI monitor"
    ),
    dummy: bool = typer.Option(
        False, "--dummy", help="Enable dummy mode (skip GPU weight loading, return random tokens)"
    ),
) -> None:
    """Start the Pie engine and enter an interactive session."""
    try:
        cfg = load_config(
            config, host=host, port=port, no_auth=no_auth,
            verbose=verbose, dummy_mode=dummy,
        )
    except (FileNotFoundError, ValueError) as e:
        console.print(f"[red]✗[/red] {e}")
        if isinstance(e, FileNotFoundError):
            console.print("[dim]Run 'pie config init' first.[/dim]")
        raise typer.Exit(1)

    console.print()
    engine_panel(cfg, console=console)
    console.print()

    runtime_serve(cfg, monitor=monitor, console=console)
