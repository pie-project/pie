"""Serve command implementation for Pie CLI.

Implements: pie serve
Starts the Pie engine and optionally provides an interactive shell session.
"""

from pathlib import Path

import toml
import typer
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from pie import path as pie_path

console = Console()


def load_config(
    config_path: Path | None,
    host: str | None = None,
    port: int | None = None,
    enable_auth: bool | None = None,
    no_auth: bool = False,
    verbose: bool = False,
    cache_dir: str | None = None,
    log_dir: str | None = None,
    registry: str | None = None,
    dummy_mode: bool = False,
) -> tuple[dict, list[dict]]:
    """Load and merge configuration from file and CLI arguments.

    Returns:
        Tuple of (engine_config, model_configs)
    """
    # Load config file
    file_path = config_path or pie_path.get_default_config_path()
    if not file_path.exists():
        console.print(f"[red]✗[/red] Configuration not found at {file_path}")
        console.print("[dim]Run 'pie config init' first.[/dim]")
        raise typer.Exit(1)

    config = toml.loads(file_path.read_text())

    # Extract engine config from [engine] section
    engine_section = config.get("engine", {})

    # Build engine config with CLI overrides
    engine_config = {
        "host": host or engine_section.get("host", config.get("host", "127.0.0.1")),
        "port": port or engine_section.get("port", config.get("port", 8080)),
        "enable_auth": (
            False
            if no_auth
            else (
                enable_auth
                if enable_auth is not None
                else engine_section.get("enable_auth", config.get("enable_auth", True))
            )
        ),
        "cache_dir": cache_dir
        or engine_section.get(
            "cache_dir", config.get("cache_dir", str(pie_path.get_pie_home() / "cache"))
        ),
        "verbose": verbose
        or engine_section.get("verbose", config.get("verbose", False)),
        "log_dir": log_dir
        or engine_section.get(
            "log_dir", config.get("log_dir", str(pie_path.get_pie_home() / "logs"))
        ),
        "registry": registry
        or engine_section.get(
            "registry", config.get("registry", "https://registry.pie-project.org/")
        ),
        # Include telemetry configuration from [telemetry] section
        "telemetry": config.get("telemetry", {}),
    }

    model_configs = config.get("model", [])
    # Handle both [model] (dict) and [[model]] (list) formats
    if isinstance(model_configs, dict):
        model_configs = [model_configs]
    if not model_configs:
        console.print("[red]✗[/red] No model configuration found")
        raise typer.Exit(1)

    # Apply dummy_mode override from CLI
    if dummy_mode:
        for model_config in model_configs:
            model_config["dummy_mode"] = True

    return engine_config, model_configs


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
    cache_dir: str | None = typer.Option(
        None, "--cache-dir", help="Cache directory path"
    ),
    log_dir: str | None = typer.Option(None, "--log-dir", help="Log directory path"),
    interactive: bool = typer.Option(
        False, "--interactive", "-i", help="Enable interactive shell mode"
    ),
    monitor: bool = typer.Option(
        False, "--monitor", "-m", help="Launch real-time TUI monitor"
    ),
    dummy: bool = typer.Option(
        False, "--dummy", help="Enable dummy mode (skip GPU weight loading, return random tokens)"
    ),
) -> None:
    """Start the Pie engine and enter an interactive session.

    This command starts the Pie engine server along with configured backend
    services. In interactive mode, it provides a shell for running inferlets.
    """
    try:
        engine_config, model_configs = load_config(
            config,
            host=host,
            port=port,
            no_auth=no_auth,
            verbose=verbose,
            cache_dir=cache_dir,
            log_dir=log_dir,
            dummy_mode=dummy,
        )
    except typer.Exit:
        raise

    console.print()

    # Show config summary
    mode = "interactive" if interactive else "server"
    lines = Text()
    lines.append(f"{'Host':<15}", style="white")
    lines.append(f"{engine_config['host']}:{engine_config['port']}\n", style="dim")
    lines.append(f"{'Model':<15}", style="white")
    lines.append(f"{model_configs[0].get('hf_repo', 'unknown')}\n", style="dim")
    lines.append(f"{'Device':<15}", style="white")
    device = model_configs[0].get("device", ["unknown"])
    device_str = ", ".join(device) if isinstance(device, list) else device
    lines.append(device_str, style="dim")

    console.print(
        Panel(
            lines, title=f"Pie Engine ({mode})", title_align="left", border_style="dim"
        )
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

        if interactive:
            console.print("[dim]Type 'help' for commands, ↑/↓ for history[/dim]")
            console.print()
            manager.run_interactive_shell(engine_config, server_handle.internal_token)
        elif monitor:
            # Launch real-time TUI monitor
            from pie_cli.monitor.app import LLMMonitorApp
            from pie_cli.monitor.provider import PieMetricsProvider

            provider = PieMetricsProvider(
                host=engine_config.get("host", "127.0.0.1"),
                port=engine_config.get("port", 8000),
                internal_token=server_handle.internal_token,
                config={
                    "model": (
                        model_configs[0].get("name", "Unknown")
                        if model_configs
                        else "Unknown"
                    ),
                    "tp_size": engine_config.get("tp_size", 1),
                    "max_batch": engine_config.get("max_batch_size", 32),
                },
            )
            provider.start()

            app = LLMMonitorApp(provider=provider)
            app.run()

            provider.stop()
        else:
            import time

            try:
                # Keep running while processes are alive
                while True:
                    if not manager.check_backend_processes(backend_processes):
                        typer.echo("[red]A backend process died. Shutting down.[/red]")
                        break

                    if server_handle and hasattr(server_handle, "is_running"):
                        if not server_handle.is_running():
                            typer.echo("[red]Engine process died. Shutting down.[/red]")
                            break

                    time.sleep(1.0)
            except KeyboardInterrupt:
                pass

        # Cleanup
        console.print()
        with console.status("[dim]Shutting down...[/dim]"):
            manager.terminate_engine_and_backend(server_handle, backend_processes)
        console.print("[green]✓[/green] Shutdown complete")

    except KeyboardInterrupt:
        console.print()
        with console.status("[dim]Shutting down...[/dim]"):
            manager.terminate_engine_and_backend(server_handle, backend_processes)
        console.print("[green]✓[/green] Shutdown complete")
    except Exception as e:
        from pie import manager

        if isinstance(e, manager.EngineError):
            console.print(f"[red]✗[/red] {e}")
            raise typer.Exit(1)
        console.print(f"[red]✗[/red] Error: {e}")
        manager.terminate_engine_and_backend(server_handle, backend_processes)
        raise typer.Exit(1)
