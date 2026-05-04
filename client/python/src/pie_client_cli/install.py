"""Install command implementation for the Pie CLI.

This module implements the `pie-cli install` subcommand for installing inferlets
to an existing running Pie engine instance without launching them.
"""

from pathlib import Path
from typing import Optional

import typer

from . import engine
from .submit import parse_manifest


def handle_install_command(
    path: Path,
    manifest: Path,
    config: Optional[Path] = None,
    host: Optional[str] = None,
    port: Optional[int] = None,
    username: Optional[str] = None,
    private_key_path: Optional[Path] = None,
    force: bool = False,
) -> None:
    """Handle the `pie-cli install` command.

    Installs an inferlet to the Pie engine without launching it.

    Steps:
    1. Creates a client configuration from config file and command-line arguments
    2. Connects to the Pie engine server
    3. Uploads the inferlet if not already on server (or --force is used)
    """
    if not path.exists():
        raise FileNotFoundError(f"Inferlet file not found: {path}")

    if not manifest.exists():
        raise FileNotFoundError(f"Manifest file not found: {manifest}")

    manifest_content = manifest.read_text()
    name, version = parse_manifest(manifest_content)
    inferlet_name = f"{name}@{version}"

    typer.echo(f"Inferlet: {inferlet_name}")

    client_config = engine.ClientConfig.create(
        config_path=config,
        host=host,
        port=port,
        username=username,
        private_key_path=private_key_path,
    )

    client = engine.connect_and_authenticate(client_config)

    try:
        if force or not engine.check_program(client, inferlet_name, str(path), str(manifest)):
            engine.install_program(client, str(path), str(manifest), force_overwrite=force)
            typer.echo("✅ Inferlet installed successfully.")
        else:
            typer.echo("Inferlet already exists on server.")
    finally:
        engine.close_client(client)
