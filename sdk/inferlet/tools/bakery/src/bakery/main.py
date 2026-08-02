"""Bakery CLI - Main entry point.

This module provides the main typer application with all subcommands.
"""

from pathlib import Path
from typing import Annotated, Optional

import typer

from . import build as build_cmd
from . import create as create_cmd
from . import inferlet as inferlet_cmd
from . import login as login_cmd

# Main application
app = typer.Typer(
    name="bakery",
    help="Pie Bakery - Build and publish inferlets",
    no_args_is_help=True,
)

# Inferlet subcommand group (from registry)
app.add_typer(inferlet_cmd.inferlet_app, name="inferlet")


# ============================================================================
# Path expansion callback for typer
# ============================================================================


def expand_path(path: Optional[Path]) -> Optional[Path]:
    """Expand ~ in paths."""
    if path is None:
        return None
    return path.expanduser()


# ============================================================================
# Login command (Registry authentication)
# ============================================================================


@app.command()
def login() -> None:
    """Authenticate with the Pie Registry using GitHub OAuth."""
    try:
        login_cmd.handle_login_command()
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


# ============================================================================
# Build command
# ============================================================================


@app.command()
def build(
    input_path: Annotated[
        Path,
        typer.Argument(help="Input file (.js, .ts) or directory with package.json."),
    ],
    output: Annotated[
        Path, typer.Option("-o", "--output", help="Output .wasm file path.")
    ],
    debug: Annotated[
        bool, typer.Option("--debug", help="Enable debug build (include source maps).")
    ] = False,
) -> None:
    """Build a JavaScript/TypeScript inferlet into a WebAssembly component."""
    try:
        build_cmd.handle_build_command(
            input_path=expand_path(input_path),
            output=expand_path(output),
            debug=debug,
        )
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


# ============================================================================
# Create command
# ============================================================================


@app.command()
def create(
    name: Annotated[str, typer.Argument(help="Name of the inferlet project.")],
    ts: Annotated[
        bool,
        typer.Option("--ts", "-t", help="Create a TypeScript project instead of Rust."),
    ] = False,
    output: Annotated[
        Optional[Path], typer.Option("-o", "--output", help="Output directory.")
    ] = None,
) -> None:
    """Create a new inferlet project (Rust by default, or TypeScript with --ts)."""
    try:
        create_cmd.handle_create_command(name=name, rust=not ts, output=output)
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


# ============================================================================
# Entry point
# ============================================================================


def main() -> None:
    """Main entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
