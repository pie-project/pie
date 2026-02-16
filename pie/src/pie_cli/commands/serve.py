"""Serve command implementation for Pie CLI.

Implements: pie serve
Starts the Pie engine and optionally provides a real-time TUI monitor.
"""

import asyncio
from pathlib import Path

import typer
from rich.console import Console

from rich.panel import Panel
from rich.text import Text

from pie.config import load_config
from pie.server import Server

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

    model = cfg.primary_model
    lines = Text()
    lines.append(f"{'Host':<15}", style="white")
    lines.append(f"{cfg.host}:{cfg.port}\n", style="dim")
    lines.append(f"{'Model':<15}", style="white")
    lines.append(f"{model.hf_repo}\n", style="dim")
    lines.append(f"{'Device':<15}", style="white")
    lines.append(", ".join(model.device), style="dim")

    console.print()
    console.print(Panel(lines, title="Pie Engine", title_align="left", border_style="dim"))
    console.print()

    async def _run():
        server = Server(cfg)
        async with server:
            if monitor:
                from pie_cli.monitor.app import LLMMonitorApp
                from pie_cli.monitor.provider import PieMetricsProvider

                model_cfg = cfg.primary_model
                provider = PieMetricsProvider(
                    host=cfg.host,
                    port=cfg.port,
                    internal_token=server.token,
                    config={
                        "model": model_cfg.name or model_cfg.hf_repo,
                        "tp_size": model_cfg.tensor_parallel_size or len(model_cfg.device),
                        "max_batch": model_cfg.max_batch_tokens or 32,
                    },
                )
                provider.start()
                try:
                    LLMMonitorApp(provider=provider).run()
                finally:
                    provider.stop()
            else:
                await server.wait()

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        pass
    finally:
        console.print()
        console.print("[green]✓[/green] Shutdown complete")
