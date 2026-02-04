"""Inferlet subcommand implementation for Bakery.

This module implements the `bakery inferlet` subcommand group for
searching, querying, and publishing inferlets to the Pie Registry.
"""

import hashlib
from pathlib import Path
from typing import Annotated, Optional

from rich.table import Table
from rich.panel import Panel
from rich import box
from .console import console

import toml
import typer

from .config import get_token
from .registry import (
    REGISTRY_URL,
    RegistryClient,
    RegistryError,
    PublishStartRequest,
    PublishCommitRequest,
)


# Create the inferlet subcommand group
inferlet_app = typer.Typer(
    name="inferlet",
    help="Manage inferlet packages from the Pie Registry.",
    no_args_is_help=True,
)


def _format_size(size_bytes: int) -> str:
    """Format byte size as human-readable string."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"


def _format_downloads(downloads: int) -> str:
    """Format download count as human-readable string."""
    if downloads < 1000:
        return str(downloads)
    elif downloads < 1_000_000:
        return f"{downloads / 1000:.1f}k"
    else:
        return f"{downloads / 1_000_000:.1f}M"


@inferlet_app.command("search")
def search(
    query: Annotated[str, typer.Argument(help="Search query.")] = "",
    page: Annotated[int, typer.Option("--page", "-p", help="Page number.")] = 1,
    per_page: Annotated[int, typer.Option("--per-page", help="Results per page.")] = 20,
) -> None:
    """Search for inferlets in the registry."""
    try:
        with RegistryClient(base_url=REGISTRY_URL) as client:
            with console.status("Searching..."):
                result = client.search(
                    query=query, page=page, per_page=per_page
                )

            if result.total == 0:
                if query:
                    console.print(
                        f"[yellow]No inferlets found matching '{query}'[/yellow]"
                    )
                else:
                    console.print("[yellow]No inferlets found[/yellow]")
                return

            console.print(
                f"Found [bold]{result.total}[/bold] inferlet(s) (page {result.page}/{result.pages}):"
            )
            console.print()

            table = Table(box=None, padding=(0, 2))
            table.add_column("NAME", style="bold cyan")
            table.add_column("VERSION", style="green")
            table.add_column("DOWNLOADS", justify="right", style="magenta")
            table.add_column("DESCRIPTION", style="dim")

            for item in result.items:
                name = item.name
                version = item.latest_version or "-"
                downloads = _format_downloads(item.downloads)
                description = (item.description or "").split("\n")[0][:60]
                if item.description and len(item.description) > 60:
                    description += "..."

                table.add_row(name, version, downloads, description)

            console.print(table)

            if result.pages > 1:
                console.print()
                console.print(
                    f"[dim]Use --page to see more results (page {result.page}/{result.pages})[/dim]"
                )

    except RegistryError as e:
        console.print(f"[red]❌ Error: {e.detail}[/red]")
        raise typer.Exit(1)


@inferlet_app.command("info")
def info(
    name: Annotated[
        str,
        typer.Argument(
            help="Inferlet name (e.g., 'text-completion')."
        ),
    ],
) -> None:
    """Get detailed information about an inferlet."""
    try:
        with RegistryClient(base_url=REGISTRY_URL) as client:
            with console.status("Fetching info..."):
                detail = client.info(name)

            latest = detail.versions[0] if detail.versions else None

            # --- Header ---
            title_str = f"{detail.name}"
            if latest:
                title_str += f"@{latest.num}"

            console.print()
            console.print(f"[bold]{title_str}[/bold]")
            console.print()

            # --- Info ---
            if latest and latest.description:
                console.print(f"[dim]{latest.description.strip()}[/dim]")

            if latest:
                if latest.repository:
                    console.print(
                        f"[dim]url:[/dim] [blue link={latest.repository}]{latest.repository}[/blue link]"
                    )
                if latest.runtime:
                    runtime_str = ", ".join(f"{k}={v}" for k, v in latest.runtime.items())
                    console.print(f"[dim]runtime: {runtime_str}[/dim]")

            console.print()

            # --- Parameters Panel ---
            if latest and latest.parameters:
                param_table = Table(
                    show_header=False,
                    box=None,
                    padding=(0, 2),
                    expand=True,
                    pad_edge=False,
                )
                param_table.add_column("Name", style="white", no_wrap=True)
                param_table.add_column("Type", style="green", no_wrap=True)
                param_table.add_column("Description", style="dim", ratio=1)

                for param_name, param_info in latest.parameters.items():
                    param_type = param_info.get("type", "?")
                    optional = " (opt)" if param_info.get("optional") else ""
                    desc = param_info.get("description", "")
                    param_table.add_row(param_name, f"{param_type}{optional}", desc)

                console.print(
                    Panel(
                        param_table,
                        title="Parameters",
                        title_align="left",
                        box=box.ROUNDED,
                        border_style="dim",
                        expand=True,
                        padding=(0, 1),
                    )
                )

            # --- Dependencies Panel ---
            if latest and latest.dependencies:
                dep_table = Table(
                    show_header=False,
                    box=None,
                    padding=(0, 2),
                    expand=True,
                    pad_edge=False,
                )
                dep_table.add_column("Name", style="cyan", no_wrap=True)
                dep_table.add_column("Version", style="green", no_wrap=True)

                for dep_name, dep_version in latest.dependencies.items():
                    dep_table.add_row(dep_name, dep_version)

                console.print(
                    Panel(
                        dep_table,
                        title="Dependencies",
                        title_align="left",
                        box=box.ROUNDED,
                        border_style="dim",
                        expand=True,
                        padding=(0, 1),
                    )
                )

    except RegistryError as e:
        if e.status_code == 404:
            console.print(f"[red]❌ Inferlet '{name}' not found[/red]")
        else:
            console.print(f"[red]❌ Error: {e.detail}[/red]")
        raise typer.Exit(1)


@inferlet_app.command("publish")
def publish(
    directory: Annotated[
        Path, typer.Argument(help="Directory containing Pie.toml manifest.")
    ] = Path("."),
) -> None:
    """Publish the inferlet in the specified directory."""
    directory = directory.expanduser().resolve()

    # Check for Pie.toml
    manifest_path = directory / "Pie.toml"
    if not manifest_path.exists():
        console.print(f"[red]❌ No Pie.toml found in {directory}[/red]")
        raise typer.Exit(1)

    # Load the manifest
    try:
        manifest = toml.load(manifest_path)
    except toml.TomlDecodeError as e:
        console.print(f"[red]❌ Failed to parse Pie.toml: {e}[/red]")
        raise typer.Exit(1)

    # Extract package info
    package = manifest.get("package", {})
    name = package.get("name")
    version = package.get("version")
    description = package.get("description")

    if not name:
        console.print("[red]❌ Missing 'package.name' in Pie.toml[/red]")
        raise typer.Exit(1)

    if not version:
        console.print("[red]❌ Missing 'package.version' in Pie.toml[/red]")
        raise typer.Exit(1)

    # Extract extra metadata
    authors = package.get("authors")
    repository = package.get("repository")
    keywords = package.get("keywords")

    readme_filename = package.get("readme")
    readme_content = None
    if readme_filename:
        readme_path = directory / readme_filename
        if readme_path.exists():
            try:
                readme_content = readme_path.read_text(encoding="utf-8")
            except Exception as e:
                console.print(
                    f"[yellow]⚠️ Failed to read README file '{readme_filename}': {e}[/yellow]"
                )
        else:
            console.print(
                f"[yellow]⚠️ README file '{readme_filename}' specified in Pie.toml but not found[/yellow]"
            )

    # Get runtime requirements
    runtime = manifest.get("runtime")

    # Get parameters
    parameters = manifest.get("parameters")

    # Get dependencies
    dependencies = manifest.get("dependencies")

    # Find the .wasm artifact
    wasm_path = directory / f"{name}.wasm"
    if not wasm_path.exists():
        # Try common alternatives
        for pattern in [
            "*.wasm",
            "target/wasm32-wasip2/release/*.wasm",
            "target/*.wasm",
        ]:
            matches = list(directory.glob(pattern))
            if matches:
                wasm_path = matches[0]
                break

    if not wasm_path.exists():
        console.print(f"[red]❌ No .wasm artifact found. Expected: {name}.wasm[/red]")
        raise typer.Exit(1)

    console.print(f"📦 Publishing [bold cyan]{name}@{version}[/bold cyan]")
    console.print(f"   Artifact: [blue]{wasm_path.name}[/blue]")

    # Load the token
    token = get_token()
    if not token:
        console.print()
        console.print("[red]❌ Not authenticated. Run `bakery login` first.[/red]")
        raise typer.Exit(1)

    # Read and hash the artifact
    artifact_bytes = wasm_path.read_bytes()
    checksum = hashlib.sha256(artifact_bytes).hexdigest()
    size_bytes = len(artifact_bytes)

    console.print(f"   Size: {_format_size(size_bytes)}")
    console.print(f"   Checksum: [dim]{checksum[:16]}...[/dim]")
    console.print()

    try:
        with RegistryClient(token=token, base_url=REGISTRY_URL) as client:
            # Verify we're authenticated
            user = client.get_me()

            console.print(f"🔐 Publishing as: [bold]{user.login}[/bold]")
            console.print()

            # Start the publish process
            with console.status("[bold green]Publishing...[/bold green]") as status:
                status.update("[bold green]📤 Starting publish...[/bold green]")
                start_req = PublishStartRequest(
                    name=name,
                    version=version,
                    checksum=checksum,
                    size_bytes=size_bytes,
                    description=description,
                )
                start_resp = client.start_publish(start_req)

                status.update("[bold green]📤 Uploading artifact...[/bold green]")
                client.upload_artifact(start_resp.upload_url, artifact_bytes)

                status.update("[bold green]📤 Finalizing publish...[/bold green]")
                commit_req = PublishCommitRequest(
                    name=name,
                    version=version,
                    storage_path=start_resp.storage_path,
                    checksum=checksum,
                    size_bytes=size_bytes,
                    description=description,
                    runtime=runtime,
                    parameters=parameters,
                    dependencies=dependencies,
                    authors=authors,
                    keywords=keywords,
                    repository=repository,
                    readme=readme_content,
                )
                commit_resp = client.commit_publish(commit_req)

            console.print(
                Panel(
                    f"Published: [bold]{commit_resp.name}@{commit_resp.version}[/bold]\n\n"
                    f"Install with:\n"
                    f"   pie run {commit_resp.name}",
                    title="[green]✅ Published[/green]",
                    border_style="green",
                )
            )

    except RegistryError as e:
        console.print(f"[red]❌ Publish failed: {e.detail}[/red]")
        raise typer.Exit(1)

