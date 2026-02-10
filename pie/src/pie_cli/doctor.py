"""Doctor command for Pie Server CLI.

Implements: pie-server doctor
Runs environment health check to verify the system is ready for running.
"""

import importlib.metadata
import platform

import typer
from rich.console import Console
from rich.panel import Panel
from rich.text import Text


console = Console()


def _format_check(name: str, value: str, status: str, width: int = 20) -> Text:
    """Format a single check line."""
    line = Text()
    if status == "pass":
        line.append("✓ ", style="green")
    elif status == "warn":
        line.append("! ", style="yellow")
    elif status == "fail":
        line.append("✗ ", style="red")
    else:
        line.append("? ", style="dim")

    line.append(f"{name:<{width}}", style="white")
    line.append(value, style="dim")
    return line


def _check_system() -> list[tuple[str, str, str]]:
    """Check system info and GPUs."""
    results = []

    # Platform
    info = f"{platform.system()} {platform.release()} ({platform.machine()})"
    results.append(("Platform", info, "pass"))

    # GPUs
    try:
        import torch

        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                name = torch.cuda.get_device_name(i)
                results.append((f"GPU {i}", name, "pass"))
        elif torch.backends.mps.is_available():
            results.append(("GPU", "Apple Metal (MPS)", "pass"))
        else:
            results.append(("GPU", "None (CPU only)", "warn"))
    except ImportError:
        results.append(("GPU", "Unknown (PyTorch not installed)", "fail"))

    return results


def _check_libraries() -> list[tuple[str, str, str]]:
    """Check library installations."""
    results = []

    # pie_backend (merged into pie-server)
    try:
        import pie_backend  # noqa: F401

        results.append(("pie_backend", "OK (part of pie-server)", "pass"))
    except ImportError:
        results.append(("pie_backend", "Not installed", "warn"))

    # pie-client
    try:
        ver = importlib.metadata.version("pie-client")
        results.append(("pie-client", ver, "pass"))
    except importlib.metadata.PackageNotFoundError:
        results.append(("pie-client", "Not installed", "warn"))

    # PyTorch
    try:
        import torch

        results.append(("pytorch", torch.__version__, "pass"))
    except ImportError:
        results.append(("pytorch", "Not installed", "fail"))

    # FlashInfer
    try:
        import flashinfer  # noqa: F401

        ver = importlib.metadata.version("flashinfer-python")
        results.append(("flashinfer", ver, "pass"))
    except ImportError:
        results.append(("flashinfer", "Not installed", "warn"))
    except Exception as e:
        results.append(("flashinfer", f"Error: {e}", "warn"))

    # # FBGEMM_GPU
    # try:
    #     import fbgemm_gpu  # noqa: F401
    #     ver = importlib.metadata.version('fbgemm-gpu-genai')
    #     results.append(("fbgemm_gpu", ver, "pass"))
    # except ImportError:
    #     try:
    #         ver = importlib.metadata.version('fbgemm-gpu-genai')
    #         results.append(("fbgemm_gpu", ver, "pass"))
    #     except importlib.metadata.PackageNotFoundError:
    #         results.append(("fbgemm_gpu", "Not installed", "warn"))

    return results


def doctor() -> None:
    """Run environment health check.

    Checks platform, GPU availability, and required dependencies
    to verify the system is ready for running the Pie backend.
    """
    console.print()

    has_critical_failure = False

    # System checks
    system_checks = _check_system()
    lines = Text()
    for i, (name, value, status) in enumerate(system_checks):
        if i > 0:
            lines.append("\n")
        lines.append_text(_format_check(name, value, status))
    console.print(Panel(lines, title="System", title_align="left", border_style="dim"))

    # Library checks
    lib_checks = _check_libraries()
    if any(s == "fail" for _, _, s in lib_checks):
        has_critical_failure = True

    lines = Text()
    for i, (name, value, status) in enumerate(lib_checks):
        if i > 0:
            lines.append("\n")
        lines.append_text(_format_check(name, value, status))
    console.print(
        Panel(lines, title="Libraries", title_align="left", border_style="dim")
    )

    # Summary
    all_checks = system_checks + lib_checks
    total_pass = sum(1 for _, _, s in all_checks if s == "pass")
    total_warn = sum(1 for _, _, s in all_checks if s == "warn")

    console.print()
    if has_critical_failure:
        console.print(
            "[red]✗ Critical issues found. Please resolve before running Pie.[/red]"
        )
        raise typer.Exit(1)
    elif total_warn > 0:
        console.print(
            f"[yellow]![/yellow] Ready with warnings [dim]({total_pass} passed, {total_warn} warnings)[/dim]"
        )
    else:
        console.print(
            f"[green]✓[/green] All checks passed [dim]({total_pass} checks)[/dim]"
        )
    console.print()
