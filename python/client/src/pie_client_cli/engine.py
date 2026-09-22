"""Engine client utilities for the Pie CLI.

This module provides utilities for connecting to the Pie engine and streaming
inferlet output with signal handling.
"""

import asyncio
import os
import signal
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import typer

from pie_client import PieClient, Event, Process

from . import path as path_utils
from .config import ConfigFile


@dataclass
class ClientConfig:
    """Configuration for connecting to the Pie engine."""

    host: str
    port: int
    username: str

    @classmethod
    def create(
        cls,
        config_path: Optional[Path] = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
        username: Optional[str] = None,
    ) -> "ClientConfig":
        """Create a ClientConfig, merging command-line args with config file.

        Command-line arguments take precedence over config file values.
        """
        # Read config file if any parameter is missing
        config_file: Optional[ConfigFile] = None
        if host is None or port is None or username is None:
            config_file_path = config_path or path_utils.get_default_config_path()
            if config_file_path.exists():
                config_file = ConfigFile.load(config_file_path)
            elif config_path is not None:
                # User explicitly specified a config path that doesn't exist
                raise FileNotFoundError(f"Config file not found at '{config_path}'")
            # If default config doesn't exist, we'll use defaults below

        # Merge with defaults
        final_host = host or (config_file.host if config_file else None) or "127.0.0.1"
        final_port = port or (config_file.port if config_file else None) or 8080
        final_username = (
            username or (config_file.username if config_file else None) or os.getlogin()
        )

        return cls(host=final_host, port=final_port, username=final_username)


async def _connect_and_authenticate_async(client_config: ClientConfig) -> PieClient:
    """Connect to the engine and authenticate (async version)."""
    url = f"ws://{client_config.host}:{client_config.port}"

    client = PieClient(url)
    try:
        await client.connect()
    except Exception:
        raise ConnectionError(f"Could not connect to engine at {url}. Is it running?")

    try:
        await client.authenticate(client_config.username)
    except Exception as e:
        await client.close()
        raise ConnectionError(f"Failed to identify to the engine as '{client_config.username}': {e}")

    return client


def connect_and_authenticate(client_config: ClientConfig) -> PieClient:
    """Connect to the engine and authenticate (sync wrapper)."""
    return asyncio.get_event_loop().run_until_complete(
        _connect_and_authenticate_async(client_config)
    )


def _write_with_prefix(
    is_stderr: bool,
    content: str,
    short_id: str,
    at_line_start: bool,
) -> bool:
    """Write output with instance ID prefix at line starts.

    Returns the new at_line_start state.
    """
    if not content:
        return at_line_start

    writer = sys.stderr if is_stderr else sys.stdout
    lines = content.split("\n")
    first = True

    for line in lines:
        if not first:
            # We encountered a '\n' separator
            writer.write("\n")
            at_line_start = True
        first = False

        # Add prefix only if at line start and line is non-empty
        if line:
            if at_line_start:
                writer.write(f"[Instance {short_id}] ")
                at_line_start = False
            writer.write(line)

    writer.flush()
    return at_line_start


async def _stream_inferlet_output_async(
    instance: Process,
    client: PieClient,
) -> None:
    """Stream output from an inferlet with signal handling (async version).

    Behavior:
    - Ctrl-C (SIGINT): Sends terminate request to the server
    - Ctrl-D (EOF on stdin): Detaches from the inferlet (continues running)
    """
    instance_id = instance.process_id
    short_id = instance_id[: min(8, len(instance_id))]
    at_line_start_stdout = True
    at_line_start_stderr = True

    # Set up SIGINT handling
    sigint_received = asyncio.Event()
    original_handler = signal.getsignal(signal.SIGINT)

    def sigint_handler(signum, frame):
        sigint_received.set()

    signal.signal(signal.SIGINT, sigint_handler)

    # Set up stdin EOF detection in a separate thread
    eof_received = asyncio.Event()
    loop = asyncio.get_running_loop()

    def stdin_monitor():
        try:
            while True:
                data = sys.stdin.read(1)
                if not data:  # EOF
                    loop.call_soon_threadsafe(eof_received.set)
                    break
        except Exception:
            pass

    stdin_thread = threading.Thread(target=stdin_monitor, daemon=True)
    stdin_thread.start()

    try:
        while True:
            # Create tasks for the three events we're waiting on
            recv_task = asyncio.create_task(instance.recv())
            sigint_wait = asyncio.create_task(sigint_received.wait())
            eof_wait = asyncio.create_task(eof_received.wait())

            done, pending = await asyncio.wait(
                [recv_task, sigint_wait, eof_wait],
                return_when=asyncio.FIRST_COMPLETED,
            )

            # Cancel pending tasks
            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

            # Handle SIGINT (Ctrl-C)
            if sigint_wait in done:
                typer.echo(
                    f"\n[Instance {short_id}] Received Ctrl-C, terminating instance ..."
                )
                try:
                    await client.terminate_process(instance_id)
                except Exception as e:
                    typer.echo(
                        f"[Instance {short_id}] Failed to send terminate request: {e}",
                        err=True,
                    )
                return

            # Handle EOF (Ctrl-D)
            if eof_wait in done:
                typer.echo(f"\n[Instance {short_id}] Detached from instance ...")
                return

            # Handle instance event
            if recv_task in done:
                try:
                    event, message = recv_task.result()
                except Exception as e:
                    print(f"[Instance {short_id}] ReceiveError: {e}")
                    raise

                if event == Event.Message:
                    typer.echo(f"[Instance {short_id}] Message: {message}")

                elif event == Event.Return:
                    typer.echo(f"[Instance {short_id}] Completed: {message}")
                    return

                elif event == Event.Error:
                    typer.echo(f"[Instance {short_id}] Error: {message}")
                    raise RuntimeError(f"inferlet terminated with error: {message}")

                elif event == Event.Stdout:
                    at_line_start_stdout = _write_with_prefix(
                        False, message, short_id, at_line_start_stdout
                    )

                elif event == Event.Stderr:
                    at_line_start_stderr = _write_with_prefix(
                        True, message, short_id, at_line_start_stderr
                    )

                elif event == Event.File:
                    # Announced, not dropped. `message` is a `ReceivedFile`:
                    # bytes carrying the name the inferlet suggested, so the
                    # one line this can print is a useful one.
                    typer.echo(
                        f"[Instance {short_id}] File: "
                        f"{message.file_name('(unnamed)')} ({len(message)} bytes)"
                    )

    finally:
        # Restore original signal handler
        signal.signal(signal.SIGINT, original_handler)


def stream_inferlet_output(instance: Process, client: PieClient) -> None:
    """Stream output from an inferlet with signal handling (sync wrapper)."""
    asyncio.get_event_loop().run_until_complete(
        _stream_inferlet_output_async(instance, client)
    )


# Sync wrappers for common client operations
def ping(client: PieClient) -> None:
    """Ping the server (sync wrapper)."""
    asyncio.get_event_loop().run_until_complete(client.ping())


def list_processes(client: PieClient) -> list[str]:
    """List processes (sync wrapper)."""
    return asyncio.get_event_loop().run_until_complete(client.list_processes())


def terminate_process(client: PieClient, instance_id: str) -> None:
    """Terminate a process (sync wrapper)."""
    asyncio.get_event_loop().run_until_complete(client.terminate_process(instance_id))


def attach_process(client: PieClient, instance_id: str) -> Process:
    """Attach to a process (sync wrapper)."""
    return asyncio.get_event_loop().run_until_complete(
        client.attach_process(instance_id)
    )


def install_program(
    client: PieClient,
    wasm_path: str,
    manifest_path: str,
    force_overwrite: bool = False,
) -> None:
    """Install a program (sync wrapper).

    Args:
        client: The PieClient instance.
        wasm_path: Path to the WASM binary file.
        manifest_path: Path to the manifest TOML file.
        force_overwrite: If True, overwrite an existing program with the same name+version.
    """
    asyncio.get_event_loop().run_until_complete(
        client.install_program(wasm_path, manifest_path, force_overwrite=force_overwrite)
    )


def check_program(client: PieClient, inferlet: str) -> bool:
    """Whether the server holds `inferlet` (`name@version`); sync wrapper."""
    return asyncio.get_event_loop().run_until_complete(client.check_program(inferlet))


def launch_process(
    client: PieClient,
    inferlet: str,
    input: dict,
    capture_outputs: bool = True,
) -> Process:
    """Launch a process (sync wrapper).

    The inferlet must be in name@version format (e.g., "text-completion@0.1.0").
    """
    return asyncio.get_event_loop().run_until_complete(
        client.launch_process(inferlet, input, capture_outputs)
    )

def close_client(client: PieClient) -> None:
    """Close the client (sync wrapper)."""
    asyncio.get_event_loop().run_until_complete(client.close())
