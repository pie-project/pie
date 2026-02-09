"""List command implementation for the Pie CLI.

This module implements the `pie-cli list` subcommand for querying
all running inferlet instances from a Pie engine.
"""

from pathlib import Path
from typing import Optional

import typer

from . import engine


def truncate_with_ellipsis(s: str, max_chars: int) -> str:
    """Truncate a string to max_chars, adding ellipsis if truncated."""
    if len(s) <= max_chars:
        return s
    if max_chars < 3:
        return s[:max_chars]
    return s[: max_chars - 3] + "..."


def format_arguments(args: list[str]) -> str:
    """Format arguments with proper quoting to preserve grouping."""
    formatted = []
    for arg in args:
        if not arg or any(c.isspace() for c in arg):
            # Quote if empty or contains whitespace
            escaped = arg.replace('"', '\\"')
            formatted.append(f'"{escaped}"')
        else:
            formatted.append(arg)
    return " ".join(formatted)


def handle_list_command(
    config: Optional[Path] = None,
    host: Optional[str] = None,
    port: Optional[int] = None,
    username: Optional[str] = None,
    private_key_path: Optional[Path] = None,
    full: bool = False,
    long: bool = False,
) -> None:
    """Handle the `pie-cli list` command.

    1. Creates a client configuration from config file and command-line arguments
    2. Connects to the Pie engine server
    3. Queries for all live instances
    4. Displays the list of running inferlet instances
    """
    client_config = engine.ClientConfig.create(
        config_path=config,
        host=host,
        port=port,
        username=username,
        private_key_path=private_key_path,
    )

    client = engine.connect_and_authenticate(client_config)

    try:
        instances = engine.list_processes(client)

        if not instances:
            typer.echo("✅ No running instances found.")
            return

        plural = "" if len(instances) == 1 else "s"
        typer.echo(f"✅ Found {len(instances)} running instance{plural}:")
        typer.echo()

        # Column widths
        id_width = 36 if full else (8 if long else 4)
        status_width = 8
        args_width = 60 if full else (80 if long else 84)

        # Print header
        typer.echo(
            f"{'ID':<{id_width}}  {'STATUS':<{status_width}}  "
            f"{'ARGUMENTS':<{args_width}}"
        )

        # Print separator
        typer.echo(f"{'-' * id_width}  {'-' * status_width}  " f"{'-' * args_width}")

        # Print each instance
        for inst in instances:
            # Format UUID based on display mode
            if full:
                uuid_display = inst.id
            elif long:
                uuid_display = inst.id[:8]
            else:
                uuid_display = inst.id[:4]

            # Format other fields
            status_display = inst.status
            args_str = format_arguments(inst.arguments)
            args_display = truncate_with_ellipsis(args_str, args_width)

            typer.echo(
                f"{uuid_display:<{id_width}}  {status_display:<{status_width}}  "
                f"{args_display:<{args_width}}"
            )

    finally:
        engine.close_client(client)
