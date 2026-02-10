"""Model management commands for Pie CLI.

Implements: pie model list|download|remove
Uses HuggingFace Hub as the source for models.
"""

import typer
import typing
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)
from rich.text import Text


class TqdmProgress:
    """Adapter to map tqdm calls to a global rich Progress instance.

    Uses HF's single byte bar for progress display.
    """

    _progress: typing.Optional[Progress] = None
    _lock: typing.Optional["threading.Lock"] = None
    _known_total: int = 0  # Set by model_download via dry run

    def __init__(
        self,
        iterable=None,
        desc: str | None = None,
        total: float | None = None,
        unit: str = "it",
        unit_scale: bool = False,
        miniters: int | None = None,
        ncols: int | None = None,
        **kwargs,
    ):
        import threading

        self.iterable = iterable
        self.desc = desc
        self.total = total or 0
        self.unit = unit
        self.task_id = None
        self._is_byte_task = unit == "B"

        # Skip non-byte tasks (like "Fetching X files")
        if not self._is_byte_task:
            return

        # Create a task for this progress bar
        if self._progress is not None:
            # No description needed - header already shows "Downloading: repo_id"
            total_to_use = (
                TqdmProgress._known_total if TqdmProgress._known_total > 0 else None
            )
            self.task_id = self._progress.add_task(
                description="",
                total=total_to_use,
            )

    def update(self, n: float = 1) -> None:
        if self._progress is not None and self.task_id is not None:
            self._progress.update(self.task_id, advance=n)

    def close(self) -> None:
        if self._progress is not None and self.task_id is not None:
            self._progress.update(self.task_id, visible=False)

    def __iter__(self):
        if self.iterable:
            for item in self.iterable:
                yield item
                self.update(1)

    @classmethod
    def get_lock(cls):
        import threading

        if cls._lock is None:
            cls._lock = threading.Lock()
        return cls._lock

    @classmethod
    def set_lock(cls, lock):
        cls._lock = lock

    def refresh(self, nolock=False, lock_args=None):
        if self._progress:
            self._progress.refresh()

    def reset(self, total=None):
        if self._progress is not None and self.task_id is not None:
            self._progress.update(self.task_id, total=total, completed=0)

    @classmethod
    def write(cls, s, file=None, end="\n", nolock=False):
        console.print(s, end=end)

    def set_description(self, desc=None, refresh=True):
        # Ignore HF's description updates to keep UI clean
        pass

    def set_postfix(self, ordered_dict=None, refresh=True, **kwargs):
        pass

    def disable(self):
        pass



console = Console()
app = typer.Typer(help="Manage models from HuggingFace")


@app.command("list")
def model_list() -> None:
    """List locally cached HuggingFace models."""
    from pie_cli.model_utils import (
        get_hf_cache_dir,
        parse_repo_id_from_dirname,
        get_model_config,
        check_pie_compatibility,
    )

    cache_dir = get_hf_cache_dir()

    if not cache_dir.exists():
        console.print(
            Panel(
                "[dim]No HuggingFace cache found[/dim]",
                title="Models",
                title_align="left",
                border_style="dim",
            )
        )
        return

    # Collect models
    models: list[tuple[str, bool, str]] = []  # (repo_id, compatible, info)

    for entry in cache_dir.iterdir():
        if not entry.is_dir():
            continue
        repo_id = parse_repo_id_from_dirname(entry.name)
        if repo_id is None:
            continue

        config = get_model_config(cache_dir, repo_id)
        compatible, info = check_pie_compatibility(config)
        models.append((repo_id, compatible, info))

    if not models:
        console.print(
            Panel(
                "[dim]No models found[/dim]",
                title="Models",
                title_align="left",
                border_style="dim",
            )
        )
        return

    # Build display
    lines = Text()
    for i, (repo_id, compatible, info) in enumerate(sorted(models)):
        if i > 0:
            lines.append("\n")

        if compatible:
            lines.append("✓ ", style="green")
            lines.append(repo_id, style="white")
            lines.append(f" ({info})", style="dim")
        else:
            lines.append("○ ", style="dim")
            lines.append(repo_id, style="dim")
            lines.append(f" ({info})", style="dim")

    console.print(Panel(lines, title="Models", title_align="left", border_style="dim"))


@app.command("download")
def model_download(
    repo_id: str = typer.Argument(
        ..., help="HuggingFace repo ID (e.g., meta-llama/Llama-3.2-1B-Instruct)"
    )
) -> None:
    """Download a model from HuggingFace."""
    console.print()
    console.print(f"[bold]Downloading:[/bold] {repo_id}")

    try:
        # First, do a dry run to get total download size (silently)
        from huggingface_hub import snapshot_download
        from huggingface_hub.utils import disable_progress_bars, enable_progress_bars

        disable_progress_bars()
        try:
            with console.status("[dim]Calculating download size...[/dim]"):
                dry_run_info = snapshot_download(
                    repo_id,
                    local_files_only=False,
                    dry_run=True,
                )
                # Only count files that will actually be downloaded
                total_size = sum(
                    f.file_size
                    for f in dry_run_info
                    if f.file_size is not None and f.will_download
                )
        finally:
            enable_progress_bars()

        # Configuration for the progress bar
        progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            DownloadColumn(),
            TransferSpeedColumn(),
            TimeRemainingColumn(),
            console=console,
        )

        TqdmProgress._progress = progress
        TqdmProgress._known_total = total_size

        with progress:
            local_path = snapshot_download(
                repo_id,
                local_files_only=False,
                tqdm_class=TqdmProgress,
            )

        TqdmProgress._progress = None
        TqdmProgress._known_total = 0

        console.print(f"[green]✓[/green] Downloaded to {local_path}")

        # Check compatibility
        from pie_cli.model_utils import (
            get_hf_cache_dir,
            get_model_config,
            check_pie_compatibility,
        )

        cache_dir = get_hf_cache_dir()
        config = get_model_config(cache_dir, repo_id)
        compatible, info = check_pie_compatibility(config)

        console.print()
        if compatible:
            console.print(f"[green]✓[/green] Pie compatible (arch: {info})")
            console.print("Add to config.toml:")
            console.print(f'  hf_repo = "{repo_id}"')
        else:
            console.print(f"[yellow]![/yellow] Not Pie compatible ({info})")

    except Exception as e:
        console.print(f"[red]✗[/red] Download failed: {e}")
        raise typer.Exit(1)


@app.command("remove")
def model_remove(
    repo_id: str = typer.Argument(..., help="HuggingFace repo ID to remove")
) -> None:
    """Remove a locally cached model."""
    try:
        from huggingface_hub import scan_cache_dir

        cache_info = scan_cache_dir()
    except Exception as e:
        console.print(f"[red]✗[/red] Failed to scan cache: {e}")
        raise typer.Exit(1)

    # Find the repo in cache
    target_repo = None
    for repo in cache_info.repos:
        if repo.repo_id == repo_id:
            target_repo = repo
            break

    if target_repo is None:
        console.print(f"[red]✗[/red] Model '{repo_id}' not found in cache")
        raise typer.Exit(1)

    # Confirm deletion
    size_mb = target_repo.size_on_disk / (1024 * 1024)
    if not typer.confirm(f"Remove {repo_id} ({size_mb:.1f} MB)?"):
        console.print("[dim]Aborted.[/dim]")
        return

    # Delete using huggingface_hub
    try:
        delete_strategy = cache_info.delete_revisions(
            *[rev.commit_hash for rev in target_repo.revisions]
        )
        delete_strategy.execute()
        console.print(f"[green]✓[/green] Removed {repo_id}")
    except Exception as e:
        console.print(f"[red]✗[/red] Failed to remove: {e}")
        raise typer.Exit(1)
