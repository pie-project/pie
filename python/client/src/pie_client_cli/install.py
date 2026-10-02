"""Install command implementation for the Pie CLI.

This module implements the `pie-cli install` subcommand for installing inferlets
to an existing running Pie engine instance without launching them.
"""

from pathlib import Path
from typing import Optional

import typer

from . import engine


def handle_install_command(
    path: Path,
    version: Optional[str] = None,
    config: Optional[Path] = None,
    host: Optional[str] = None,
    port: Optional[int] = None,
    username: Optional[str] = None,
    force: bool = False,
) -> None:
    """Handle the `pie-cli install` command.

    Installs an inferlet (a `.wasm` component or a `.py` / `.js` script) to
    the Pie engine without launching it. The file names the program; the
    server answers with the `name@version` it installed.

    Steps:
    1. Creates a client configuration from config file and command-line arguments
    2. Connects to the Pie engine server
    3. Uploads the inferlet (replacing an installed version with --force)
    """
    if not path.exists():
        raise FileNotFoundError(f"Inferlet file not found: {path}")

    client_config = engine.ClientConfig.create(
        config_path=config,
        host=host,
        port=port,
        username=username,
    )

    client = engine.connect_and_authenticate(client_config)

    try:
        program = engine.install_program(client, path, version, force_overwrite=force)
        typer.echo(f"✅ Installed {program}.")
    finally:
        engine.close_client(client)
