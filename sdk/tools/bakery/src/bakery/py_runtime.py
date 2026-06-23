"""Python WASM runtime install/check helpers.

Pie's Python inferlets are built with `factored-componentize-py` — a
"factored" build that imports a `componentize-py-runtime` core module
from the host instead of bundling the whole interpreter into each WASM.
The host loads that core module from `$PIE_HOME/py-runtime/shared/*.wasm`,
so the runtime tarball must be present before any Python inferlet can
actually run.

This module is shared between bakery (which fetches at build time so
the WASM is immediately runnable) and the pie-worker engine (which
fetches lazily on startup). Both call paths converge on
`ensure_installed()`.
"""

from __future__ import annotations

import lzma
import os
import tarfile
from io import BytesIO
from pathlib import Path

import httpx

# Pinned to the version pie-worker expects. Bumping this is a coordinated
# change with the runtime/program/python module on the Rust side.
RUNTIME_URL = (
    "https://registry.pie-project.org/api/v1/runtimes/python3.14/0.4.0/download"
)


def get_pie_home() -> Path:
    """Resolve $PIE_HOME, falling back to ~/.pie."""
    if pie_home := os.environ.get("PIE_HOME"):
        return Path(pie_home)
    return Path.home() / ".pie"


def get_runtime_dir() -> Path:
    """The directory the host expects to find shared/*.wasm in."""
    return get_pie_home() / "py-runtime"


def is_installed() -> bool:
    """True if a usable py-runtime tree exists.

    We check for `shared/componentize-py-runtime.wasm` specifically — the
    one core module the host *must* link against. If that file is present
    the rest of the tree is almost certainly intact.
    """
    return (get_runtime_dir() / "shared" / "componentize-py-runtime.wasm").is_file()


def ensure_installed(*, quiet: bool = False) -> Path:
    """Install the Python WASM runtime if it isn't already.

    Returns the resolved runtime directory. Raises on download/extract
    failure — callers decide whether to surface the failure immediately
    or continue (e.g. bakery only needs it at run time).

    Args:
        quiet: If True, suppress the progress bar and "extracting" line.
            Used by the engine startup path where we don't want to clutter
            the terminal unless something interesting happens.
    """
    runtime_dir = get_runtime_dir()
    if is_installed():
        return runtime_dir

    pie_home = get_pie_home()
    pie_home.mkdir(parents=True, exist_ok=True)

    blob = _fetch(quiet=quiet)
    decompressed = lzma.decompress(blob)
    with tarfile.open(fileobj=BytesIO(decompressed), mode="r:") as tar:
        tar.extractall(path=pie_home)

    if not is_installed():
        raise RuntimeError(
            f"Python runtime download completed but {runtime_dir} is incomplete."
        )
    return runtime_dir


def _fetch(*, quiet: bool) -> bytes:
    """Download the runtime tarball, optionally with a progress bar.

    Rich is imported lazily so the module stays importable in minimal
    environments (e.g. test harnesses) even though both bakery and
    pie-worker already depend on it.
    """
    if quiet:
        with httpx.stream("GET", RUNTIME_URL, follow_redirects=True) as resp:
            resp.raise_for_status()
            return b"".join(resp.iter_bytes())

    from rich.console import Console
    from rich.progress import (
        BarColumn,
        DownloadColumn,
        Progress,
        TextColumn,
        TransferSpeedColumn,
    )

    console = Console()
    chunks = bytearray()
    with httpx.stream("GET", RUNTIME_URL, follow_redirects=True) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        with Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            DownloadColumn(),
            TransferSpeedColumn(),
        ) as progress:
            task = progress.add_task(
                "Downloading Python 3.14 runtime for Python inferlets",
                total=total or None,
            )
            for chunk in resp.iter_bytes():
                chunks.extend(chunk)
                progress.update(task, advance=len(chunk))
    console.print("[dim]Extracting runtime…[/dim]")
    return bytes(chunks)
