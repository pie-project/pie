"""Run command implementation for Pie CLI.

Implements: pie run <inferlet> [args]
Runs an inferlet with a one-shot Pie engine instance.
"""

import asyncio
import sys
from pathlib import Path

import typer
from rich.console import Console

from rich.panel import Panel
from rich.text import Text

from pie.config import load_config
from pie.server import Server

console = Console()


def run(
    inferlet: str | None = typer.Argument(
        None, help="Inferlet name from registry (e.g., 'text-completion@0.1.0')"
    ),
    path: Path | None = typer.Option(
        None, "--path", "-p", help="Path to a local .wasm inferlet file"
    ),
    manifest: Path | None = typer.Option(
        None, "--manifest", "-m", help="Path to the manifest TOML file (required with --path)"
    ),
    config: Path | None = typer.Option(
        None, "--config", "-c", help="Path to TOML configuration file"
    ),
    port: int | None = typer.Option(None, "--port", help="Override port"),
    log: Path | None = typer.Option(None, "--log", help="Path to log file"),
    dummy: bool = typer.Option(
        False, "--dummy", help="Enable dummy mode (skip GPU weight loading, return random tokens)"
    ),
    arguments: list[str] | None = typer.Argument(
        None, help="Arguments to pass to the inferlet"
    ),
) -> None:
    """Run an inferlet with a one-shot Pie engine."""
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
        cfg = load_config(config, port=port, dummy_mode=dummy)
    except (FileNotFoundError, ValueError) as e:
        console.print(f"[red]✗[/red] {e}")
        raise typer.Exit(1)

    model = cfg.primary_model
    inferlet_display = str(path) if path else inferlet
    lines = Text()
    lines.append(f"{'Inferlet':<15}", style="white")
    lines.append(f"{inferlet_display}\n", style="dim")
    lines.append(f"{'Model':<15}", style="white")
    lines.append(f"{model.hf_repo}\n", style="dim")
    lines.append(f"{'Device':<15}", style="white")
    lines.append(", ".join(model.device), style="dim")

    console.print()
    console.print(Panel(lines, title="Pie Run", title_align="left", border_style="dim"))
    console.print()

    # Resolve program name
    name = inferlet
    force_overwrite = path is not None
    if path is not None and manifest is not None:
        import tomllib
        manifest_data = tomllib.loads(manifest.read_text())
        pkg_name = manifest_data["package"]["name"]
        version = manifest_data["package"]["version"]
        name = f"{pkg_name}@{version}"

    async def _run():
        from pie_client import Event

        async with Server(cfg) as server:
            client = await server.connect()

            # Install from local path if provided
            if path is not None and manifest is not None:
                print("Installing program (force overwrite)...")
                await client.install_program(path, manifest, force_overwrite=force_overwrite)

            # Resolve bare name to name@version if needed
            resolved = name
            if "@" not in resolved:
                resolved = await client.resolve_version(resolved, cfg.registry)

            # Launch and stream
            print(f"Launching {resolved}...")
            process = await client.launch_process(
                resolved,
                arguments=arguments or [],
                capture_outputs=True,
            )
            print(f"Process started: {process.process_id}")

            try:
                while True:
                    event, value = await process.recv()
                    if event == Event.Stdout:
                        print(value, end="", flush=True)
                    elif event == Event.Stderr:
                        print(value, end="", file=sys.stderr, flush=True)
                    elif event == Event.Message:
                        print(f"[Message] {value}")
                    elif event == Event.Return:
                        print(value)
                        break
                    elif event == Event.Error:
                        print(f"❌ {value}", file=sys.stderr)
                        break
                    elif event == Event.File:
                        print(f"[Received file: {len(value)} bytes]")
            except Exception as e:
                import traceback
                print(f"[RECV ERROR] {type(e).__name__}: {e}", file=sys.stderr, flush=True)
                traceback.print_exc(file=sys.stderr)

    asyncio.run(_run())
