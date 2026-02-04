"""Run command implementation for Pie CLI.

Implements: pie run <inferlet> [args]
Runs an inferlet with a one-shot Pie engine instance.
"""

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from . import serve as serve_module

console = Console()


def run(
    inferlet: Optional[str] = typer.Argument(
        None, help="Inferlet name from registry (e.g., 'text-completion@0.1.0')"
    ),
    path: Optional[Path] = typer.Option(
        None, "--path", "-p", help="Path to a local .wasm inferlet file"
    ),
    manifest: Optional[Path] = typer.Option(
        None, "--manifest", "-m", help="Path to the manifest TOML file (required with --path)"
    ),
    config: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to TOML configuration file"
    ),
    port: Optional[int] = typer.Option(None, "--port", help="Override port"),
    log: Optional[Path] = typer.Option(None, "--log", help="Path to log file"),
    dummy: bool = typer.Option(
        False, "--dummy", help="Enable dummy mode (skip GPU weight loading, return random tokens)"
    ),
    arguments: Optional[list[str]] = typer.Argument(
        None, help="Arguments to pass to the inferlet"
    ),
) -> None:
    """Run an inferlet with a one-shot Pie engine.

    This command starts the Pie engine, runs the specified inferlet,
    waits for it to complete, and then shuts down.

    You can specify an inferlet either by registry name or by path (mutually exclusive):

    - By registry: pie run text-completion@0.1.0
    - By path: pie run --path ./my_inferlet.wasm
    """
    # Validate at least one of inferlet or path is provided
    if inferlet is None and path is None:
        console.print("[red]✗[/red] Specify an inferlet name or --path")
        raise typer.Exit(1)

    # Handle the case where --path is used with -- separator
    # Positional args after -- get captured as `inferlet` first, so
    # prepend it to `arguments` instead
    if inferlet is not None and path is not None:
        arguments = [inferlet] + (arguments or [])
        inferlet = None

    # Verify inferlet exists if using path
    if path is not None and not path.exists():
        console.print(f"[red]✗[/red] File not found: {path}")
        raise typer.Exit(1)

    # Manifest is required when using --path
    if path is not None and manifest is None:
        console.print("[red]✗[/red] --manifest is required when using --path")
        raise typer.Exit(1)

    # Verify manifest exists if provided
    if manifest is not None and not manifest.exists():
        console.print(f"[red]✗[/red] Manifest not found: {manifest}")
        raise typer.Exit(1)

    try:
        engine_config, model_configs = serve_module.load_config(
            config,
            port=port,
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
    lines.append(f"{'Model':<15}", style="white")
    lines.append(f"{model_configs[0].get('hf_repo', 'unknown')}\n", style="dim")
    lines.append(f"{'Device':<15}", style="white")
    device = model_configs[0].get("device", ["unknown"])
    device_str = ", ".join(device) if isinstance(device, list) else device
    lines.append(device_str, style="dim")

    console.print(Panel(lines, title="Pie Run", title_align="left", border_style="dim"))
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

        # Run the inferlet
        client_config = {
            "host": engine_config["host"],
            "port": engine_config["port"],
            "internal_auth_token": server_handle.internal_token,
        }

        if path is not None:
            manager.submit_inferlet_and_wait(
                client_config, path, manifest, arguments or [], server_handle, backend_processes
            )
        else:
            # Launch from registry
            manager.submit_inferlet_from_registry_and_wait(
                client_config,
                inferlet,
                arguments or [],
                server_handle,
                backend_processes,
            )

        # Cleanup
        console.print()
        with console.status("[dim]Shutting down...[/dim]"):
            manager.terminate_engine_and_backend(server_handle, backend_processes)
        console.print("[green]✓[/green] Complete")

    except KeyboardInterrupt:
        console.print()
        console.print("[yellow]![/yellow] Interrupted")
        with console.status("[dim]Shutting down...[/dim]"):
            manager.terminate_engine_and_backend(server_handle, backend_processes)
        raise typer.Exit(130)
    except Exception as e:
        from pie import manager

        if isinstance(e, manager.EngineError):
            console.print(f"[red]✗[/red] {e}")
            manager.terminate_engine_and_backend(server_handle, backend_processes)
            raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]✗[/red] Error: {e}")
        manager.terminate_engine_and_backend(server_handle, backend_processes)
        raise typer.Exit(1)
