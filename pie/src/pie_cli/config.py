"""Configuration CLI commands for Pie.

Implements: pie config init|show|set

Re-exports from pie.config for backward compatibility:
    Config, AuthConfig, TelemetryConfig, ModelConfig,
    load_config, DEFAULT_MODEL, create_default_config_content
"""

from pathlib import Path

import toml
import typer
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax

from pie import path as pie_path
from pie.config import (  # noqa: F401
    AuthConfig,
    TelemetryConfig,
    ModelConfig,
    Config,
    load_config,
    DEFAULT_MODEL,
    create_default_config_content,
)

console = Console()
app = typer.Typer(help="Manage configuration")


@app.command("init")
def config_init(
    path: str | None = typer.Option(None, "--path", help="Custom config path"),
) -> None:
    """Create a default config file."""
    config_path = Path(path) if path else pie_path.get_default_config_path()

    # Create parent directory if needed
    config_path.parent.mkdir(parents=True, exist_ok=True)

    # Create content
    content = create_default_config_content()

    config_path.write_text(content)
    console.print(f"[green]✓[/green] Configuration file created at {config_path}")

    # Check if default model exists
    try:
        from huggingface_hub import scan_cache_dir
        cache_info = scan_cache_dir()
        model_exists = False
        for repo in cache_info.repos:
            if repo.repo_id == DEFAULT_MODEL:
                model_exists = True
                break

        if not model_exists:
            console.print(
                f"[yellow]![/yellow] Default model '{DEFAULT_MODEL}' not found.\n"
                f"  Run [bold]pie model download {DEFAULT_MODEL}[/bold] to install."
            )
    except Exception:
        # Don't fail config init if cache scan fails
        pass


@app.command("show")
def config_show(
    path: str | None = typer.Option(None, "--path", help="Custom config path"),
) -> None:
    """Show the content of the config file."""
    config_path = Path(path) if path else pie_path.get_default_config_path()

    if not config_path.exists():
        console.print(f"[red]✗[/red] Configuration file not found at {config_path}")
        raise typer.Exit(1)

    content = config_path.read_text()
    syntax = Syntax(content, "toml", theme="monokai", line_numbers=False)
    display_path = str(config_path)
    try:
        display_path = f"~/{config_path.relative_to(Path.home())}"
    except ValueError:
        pass

    console.print(
        Panel(
            syntax,
            title=f"Configuration ({display_path})",
            title_align="left",
            border_style="dim",
        )
    )


@app.command("set")
def config_set(
    key: str = typer.Argument(..., help="Dot-path key (e.g., 'host', 'port', 'auth.enabled', 'model.0.hf_repo')"),
    value: str = typer.Argument(..., help="Value to set"),
    path: str | None = typer.Option(None, "--path", help="Custom config path"),
) -> None:
    """Set a config value by dot-path.

    Examples:

    \\b
        pie config set host 0.0.0.0
        pie config set port 9090
        pie config set auth.enabled true
        pie config set model.0.hf_repo meta-llama/Llama-3.2-1B
        pie config set model.0.device "cuda:0,cuda:1"
        pie config set telemetry.enabled true
    """
    config_path = Path(path) if path else pie_path.get_default_config_path()

    if not config_path.exists():
        console.print(f"[red]✗[/red] Configuration file not found at {config_path}")
        raise typer.Exit(1)

    config = toml.loads(config_path.read_text())

    # Parse the value into the appropriate type
    parsed_value = _parse_value(value)

    # Navigate dot-path and set value
    _set_nested(config, key, parsed_value)

    config_path.write_text(toml.dumps(config))
    console.print(f"[green]✓[/green] Set {key} = {parsed_value}")


def _parse_value(value: str):
    """Parse a string value into the appropriate Python type."""
    # Booleans
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False

    # Integers
    try:
        return int(value)
    except ValueError:
        pass

    # Floats
    try:
        return float(value)
    except ValueError:
        pass

    # Comma-separated list (for device lists like "cuda:0,cuda:1")
    if "," in value:
        return [v.strip() for v in value.split(",")]

    # String
    return value


def _set_nested(config: dict, key: str, value) -> None:
    """Set a value in a nested dict using dot-path notation.

    Handles TOML array-of-tables (e.g., model.0.hf_repo) by treating
    numeric path segments as list indices.
    """
    parts = key.split(".")
    obj = config

    for i, part in enumerate(parts[:-1]):
        # Check if this part is a list index
        try:
            idx = int(part)
            if isinstance(obj, list) and idx < len(obj):
                obj = obj[idx]
            else:
                console.print(f"[red]✗[/red] Index {idx} out of range for '{'.'.join(parts[:i])}'")
                raise typer.Exit(1)
        except ValueError:
            # Regular dict key
            if part not in obj:
                obj[part] = {}
            obj = obj[part]

    # Set the final value
    final_key = parts[-1]
    try:
        idx = int(final_key)
        if isinstance(obj, list) and idx < len(obj):
            obj[idx] = value
        else:
            console.print(f"[red]✗[/red] Index {idx} out of range")
            raise typer.Exit(1)
    except ValueError:
        obj[final_key] = value
