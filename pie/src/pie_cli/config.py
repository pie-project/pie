"""Configuration management commands for Pie CLI.

Implements: pie config init|update|show
"""

from pathlib import Path
from typing import Optional

import toml
import typer
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text

from pie_runtime import path as pie_path
from pie_runtime.config import create_default_config_content, DEFAULT_MODEL
from huggingface_hub import scan_cache_dir

console = Console()
app = typer.Typer(help="Manage configuration")


@app.command("init")
def config_init(
    path: Optional[str] = typer.Option(None, "--path", help="Custom config path"),
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
    path: Optional[str] = typer.Option(None, "--path", help="Custom config path"),
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


@app.command("update")
def config_update(
    # Engine configuration options
    host: Optional[str] = typer.Option(None, "--host", help="Network host to bind to"),
    port: Optional[int] = typer.Option(None, "--port", help="Network port to bind to"),
    enable_auth: Optional[bool] = typer.Option(
        None,
        "--enable-auth/--disable-auth",
        help="Enable/disable authentication",
    ),
    verbose: Optional[bool] = typer.Option(
        None,
        "--verbose/--no-verbose",
        help="Enable/disable verbose logging",
    ),
    registry: Optional[str] = typer.Option(
        None, "--registry", help="Inferlet registry URL"
    ),
    # Model configuration options (first model in array)
    hf_repo: Optional[str] = typer.Option(
        None, "--hf-repo", help="HuggingFace model repository"
    ),
    device: Optional[str] = typer.Option(
        None, "--device", help="Device assignment (e.g., 'cuda:0' or 'cuda:0,cuda:1')"
    ),
    activation_dtype: Optional[str] = typer.Option(
        None,
        "--activation-dtype",
        help="Activation dtype (e.g., 'bfloat16', 'float16')",
    ),
    weight_dtype: Optional[str] = typer.Option(
        None, "--weight-dtype", help="Weight dtype (e.g., 'bfloat16', 'float16')"
    ),
    kv_page_size: Optional[int] = typer.Option(
        None, "--kv-page-size", help="KV cache page size"
    ),
    max_batch_tokens: Optional[int] = typer.Option(
        None, "--max-batch-tokens", help="Maximum batch tokens"
    ),
    max_dist_size: Optional[int] = typer.Option(
        None, "--max-dist-size", help="Maximum distribution size"
    ),
    max_num_embeds: Optional[int] = typer.Option(
        None, "--max-num-embeds", help="Maximum number of embeddings"
    ),
    max_num_adapters: Optional[int] = typer.Option(
        None, "--max-num-adapters", help="Maximum number of adapters"
    ),
    max_adapter_rank: Optional[int] = typer.Option(
        None, "--max-adapter-rank", help="Maximum adapter rank"
    ),
    gpu_mem_utilization: Optional[float] = typer.Option(
        None, "--gpu-mem-utilization", help="GPU memory utilization (0.0-1.0)"
    ),
    telemetry_enabled: Optional[bool] = typer.Option(
        None,
        "--telemetry/--no-telemetry",
        help="Enable/disable OpenTelemetry tracing",
    ),
    telemetry_endpoint: Optional[str] = typer.Option(
        None, "--telemetry-endpoint", help="OTLP endpoint for traces"
    ),
    use_cuda_graphs: Optional[bool] = typer.Option(
        None,
        "--use-cuda-graphs/--no-use-cuda-graphs",
        help="Enable/disable CUDA graphs",
    ),
    path: Optional[str] = typer.Option(None, "--path", help="Custom config path"),
) -> None:
    """Update the entries of the config file."""
    config_path = Path(path) if path else pie_path.get_default_config_path()

    if not config_path.exists():
        console.print(f"[red]✗[/red] Configuration file not found at {config_path}")
        raise typer.Exit(1)

    config = toml.loads(config_path.read_text())

    # Track updates
    updates = []

    # Engine-level options
    engine_options = {
        "host": host,
        "port": port,
        "verbose": verbose,
        "registry": registry,
    }
    for key, value in engine_options.items():
        if value is not None:
            config[key] = value
            updates.append(f"{key}={value}")

    # Auth section options
    if enable_auth is not None:
        if "auth" not in config:
            config["auth"] = {}
        config["auth"]["enabled"] = enable_auth
        updates.append(f"auth.enabled={enable_auth}")

    # Model-level options (update first model)
    model_options = {
        "hf_repo": hf_repo,
        "device": device.split(",") if device else None,
        "activation_dtype": activation_dtype,
        "weight_dtype": weight_dtype,
        "kv_page_size": kv_page_size,
        "max_batch_tokens": max_batch_tokens,
        "max_dist_size": max_dist_size,
        "max_num_embeds": max_num_embeds,
        "max_num_adapters": max_num_adapters,
        "max_adapter_rank": max_adapter_rank,
        "gpu_mem_utilization": gpu_mem_utilization,
        "use_cuda_graphs": use_cuda_graphs,
    }

    model_updated = False
    for key, value in model_options.items():
        if value is not None:
            # Ensure model array exists
            if "model" not in config:
                config["model"] = [{}]
            elif not config["model"]:
                config["model"] = [{}]
            config["model"][0][key] = value
            updates.append(f"model.{key}={value}")
            model_updated = True

    # Telemetry section options
    if telemetry_enabled is not None or telemetry_endpoint is not None:
        if "telemetry" not in config:
            config["telemetry"] = {}
        if telemetry_enabled is not None:
            config["telemetry"]["enabled"] = telemetry_enabled
            updates.append(f"telemetry.enabled={telemetry_enabled}")
        if telemetry_endpoint is not None:
            config["telemetry"]["endpoint"] = telemetry_endpoint
            updates.append(f"telemetry.endpoint={telemetry_endpoint}")

    if not updates:
        console.print("[yellow]![/yellow] No configuration options provided")
        return

    config_path.write_text(toml.dumps(config))
    console.print(f"[green]✓[/green] Updated {len(updates)} option(s)")
    for update in updates:
        console.print(f"  [dim]{update}[/dim]")
