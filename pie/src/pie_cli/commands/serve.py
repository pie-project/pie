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
    no_snapshot: bool = typer.Option(
        False,
        "--no-snapshot",
        help="Disable the host-side Python snapshot optimization for this run "
             "(overrides python_snapshot in the config file).",
    ),
) -> None:
    """Start the Pie engine and enter an interactive session.

    To skip real weight loading and serve random tokens (testing), set
    `[model.<name>.driver].type = "dummy"` in the config TOML.
    """
    try:
        cfg = load_config(
            config, host=host, port=port, no_auth=no_auth,
            verbose=verbose,
        )
    except (FileNotFoundError, ValueError) as e:
        console.print(f"[red]✗[/red] {e}")
        if isinstance(e, FileNotFoundError):
            console.print("[dim]Run 'pie config init' first.[/dim]")
        raise typer.Exit(1)

    if no_snapshot:
        cfg.server.python_snapshot = False

    lines = Text()
    lines.append(f"{'Host':<15}", style="white")
    lines.append(f"{cfg.server.host}:{cfg.server.port}\n", style="dim")
    for i, model in enumerate(cfg.models):
        prefix = "Model" if i == 0 else ""
        lines.append(f"{prefix:<15}", style="white")
        lines.append(
            f"{model.name} ({model.hf_repo}) — {model.driver.type} on "
            f"{', '.join(model.driver.device)}\n",
            style="dim",
        )

    console.print()
    console.print(Panel(lines, title="Pie Engine", title_align="left", border_style="dim"))
    console.print()

    async def _run():
        server = Server(cfg)
        async with server:
            console.print(f"[bold green]✓[/bold green] Server ready at {server.url}")
            if monitor:
                from pie_cli.monitor.app import LLMMonitorApp
                from pie_cli.monitor.provider import PieMetricsProvider

                model_cfg = cfg.models[0]
                provider = PieMetricsProvider(
                    host=cfg.server.host,
                    port=cfg.server.port,
                    internal_token=server.token,
                    config={
                        "model": model_cfg.name,
                        "tp_size": model_cfg.driver.tensor_parallel_size,
                        "max_batch": 32,
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
