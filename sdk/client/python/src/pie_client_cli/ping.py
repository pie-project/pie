"""Ping command implementation for the Pie CLI.

This module implements the `pie-cli ping` subcommand for checking the liveness
of a running Pie engine instance.
"""

import time
from pathlib import Path
from typing import Optional

import typer

from . import engine


def handle_ping_command(
    config: Optional[Path] = None,
    host: Optional[str] = None,
    port: Optional[int] = None,
    username: Optional[str] = None,
    private_key_path: Optional[Path] = None,
) -> None:
    """Handle the `pie-cli ping` command.

    1. Creates a client configuration from config file and command-line arguments
    2. Attempts to connect to the Pie engine server
    3. Reports success if connection and authentication succeed, or failure otherwise
    """
    client_config = engine.ClientConfig.create(
        config_path=config,
        host=host,
        port=port,
        username=username,
        private_key_path=private_key_path,
    )

    url = f"ws://{client_config.host}:{client_config.port}"
    typer.echo(f"🔍 Pinging Pie engine at {url}")

    client = engine.connect_and_authenticate(client_config)

    try:
        start_time = time.perf_counter()
        engine.ping(client)
        duration = time.perf_counter() - start_time

        typer.echo(
            f"✅ Pie engine is alive and responsive! (latency: {duration * 1000:.3f}ms)"
        )
    finally:
        engine.close_client(client)
