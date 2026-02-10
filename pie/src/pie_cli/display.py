"""Rich display helpers for Pie CLI.

Shared panel builders and output formatting used across command modules.
"""

from rich.console import Console
from rich.panel import Panel
from rich.text import Text


def engine_panel(config, *, title: str = "Pie Engine", console: Console | None = None) -> None:
    """Display a Rich panel summarizing the engine config.

    Used by serve, run, and http commands to show a consistent config summary.
    """
    if console is None:
        console = Console()

    model = config.primary_model
    lines = Text()
    lines.append(f"{'Host':<15}", style="white")
    lines.append(f"{config.host}:{config.port}\n", style="dim")
    lines.append(f"{'Model':<15}", style="white")
    lines.append(f"{model.hf_repo}\n", style="dim")
    lines.append(f"{'Device':<15}", style="white")
    lines.append(", ".join(model.device), style="dim")

    console.print(
        Panel(lines, title=title, title_align="left", border_style="dim")
    )


def inferlet_panel(
    config, inferlet_display: str, *,
    title: str = "Pie Run",
    http_port: int | None = None,
    console: Console | None = None,
) -> None:
    """Display a Rich panel for inferlet runs (run/http commands)."""
    if console is None:
        console = Console()

    model = config.primary_model
    lines = Text()
    lines.append(f"{'Inferlet':<15}", style="white")
    lines.append(f"{inferlet_display}\n", style="dim")

    if http_port is not None:
        lines.append(f"{'Port':<15}", style="white")
        lines.append(f"{http_port}\n", style="cyan bold")

    lines.append(f"{'Model':<15}", style="white")
    lines.append(f"{model.hf_repo}\n", style="dim")
    lines.append(f"{'Device':<15}", style="white")
    lines.append(", ".join(model.device), style="dim")

    console.print(
        Panel(lines, title=title, title_align="left", border_style="dim")
    )


def check_result(
    name: str, value: str, passed: bool, *, console: Console | None = None
) -> None:
    """Display a single health check result (for doctor command)."""
    if console is None:
        console = Console()

    icon = "[green]✓[/green]" if passed else "[red]✗[/red]"
    console.print(f"  {icon} {name}: {value}")
