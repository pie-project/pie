"""Attach command implementation for the Pie CLI.

This module implements the `pie-cli attach` subcommand for attaching
to a running inferlet instance on a Pie engine and streaming its output.
"""

from pathlib import Path
from typing import Optional

import typer

from . import engine


def handle_attach_command(
    instance_id_prefix: str,
    config: Optional[Path] = None,
    host: Optional[str] = None,
    port: Optional[int] = None,
    username: Optional[str] = None,
    private_key_path: Optional[Path] = None,
) -> None:
    """Handle the `pie-cli attach` command.

    1. Creates a client configuration from config file and command-line arguments
    2. Connects to the Pie engine server
    3. Queries for all live instances
    4. Matches the UUID prefix to find the target instance
    5. Attaches to the instance if a unique match is found
    6. Streams the inferlet output with signal handling:
       - Ctrl-C (SIGINT): Terminates the inferlet on the server
       - Ctrl-D (EOF): Detaches from the inferlet (continues running on server)
    7. Reports errors if no match or multiple matches are found
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
        # Query all running instances
        instances = engine.list_processes(client)

        # Find matching instances
        matching = [
            inst for inst in instances if inst.startswith(instance_id_prefix)
        ]

        if len(matching) == 0:
            raise ValueError(
                f"No instance found with ID prefix '{instance_id_prefix}'. "
                "Use `pie-cli list` to see running instances."
            )

        if len(matching) > 1:
            typer.echo(
                f"❌ The prefix '{instance_id_prefix}' is ambiguous. Multiple instances match:"
            )
            typer.echo()
            for inst in matching:
                typer.echo(f"  {inst}")
            typer.echo()
            typer.echo(
                "Please provide a more specific prefix to uniquely identify the instance."
            )
            raise ValueError("Ambiguous instance ID prefix")

        # Found exactly one match
        instance_id = matching[0]

        # Attach to the instance
        instance = engine.attach_process(client, instance_id)

        typer.echo(f"✅ Attached to instance {instance_id}")

        engine.stream_inferlet_output(instance, client)

    finally:
        engine.close_client(client)
