"""Engine client utilities for the Pie CLI.

This module provides utilities for connecting to the Pie engine, authentication,
and streaming inferlet output with signal handling.
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

from pie_client import PieClient, Event, Instance, InstanceInfo
from pie_client.crypto import ParsedPrivateKey

from . import path as path_utils
from .config import ConfigFile


@dataclass
class ClientConfig:
    """Configuration for connecting to the Pie engine."""

    host: str
    port: int
    username: str
    private_key: Optional[ParsedPrivateKey]
    enable_auth: bool

    @classmethod
    def create(
        cls,
        config_path: Optional[Path] = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
        username: Optional[str] = None,
        private_key_path: Optional[Path] = None,
    ) -> "ClientConfig":
        """Create a ClientConfig, merging command-line args with config file.

        Command-line arguments take precedence over config file values.
        """
        # Read config file if any parameter is missing
        config_file: Optional[ConfigFile] = None
        if host is None or port is None or username is None or private_key_path is None:
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

        # Get enable_auth setting
        enable_auth = (
            config_file.enable_auth
            if config_file and config_file.enable_auth is not None
            else True
        )

        # Load private key if auth is enabled
        private_key: Optional[ParsedPrivateKey] = None
        if enable_auth:
            key_path = private_key_path
            if key_path is None and config_file and config_file.private_key_path:
                key_path = Path(config_file.private_key_path).expanduser()

            if key_path is None:
                raise ValueError(
                    "Private key is required when authentication is enabled. "
                    "Set private_key_path in config or use --private-key-path."
                )

            key_path = key_path.expanduser()

            # Check permissions on Unix
            if os.name == "posix":
                path_utils.check_private_key_permissions(key_path)

            # Read and parse the private key
            key_content = key_path.read_text()
            private_key = ParsedPrivateKey.parse(key_content)

        return cls(
            host=final_host,
            port=final_port,
            username=final_username,
            private_key=private_key,
            enable_auth=enable_auth,
        )


async def _connect_and_authenticate_async(client_config: ClientConfig) -> PieClient:
    """Connect to the engine and authenticate (async version)."""
    url = f"ws://{client_config.host}:{client_config.port}"

    client = PieClient(url)
    try:
        await client.connect()
    except Exception:
        raise ConnectionError(f"Could not connect to engine at {url}. Is it running?")

    try:
        await client.authenticate(client_config.username, client_config.private_key)
    except Exception as e:
        await client.close()
        if client_config.enable_auth:
            raise ConnectionError(
                f"Failed to authenticate with engine using the specified private key: {e}"
            )
        else:
            raise ConnectionError(
                f"Failed to authenticate with engine (client public key authentication disabled): {e}"
            )

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
    instance: Instance,
    client: PieClient,
) -> None:
    """Stream output from an inferlet with signal handling (async version).

    Behavior:
    - Ctrl-C (SIGINT): Sends terminate request to the server
    - Ctrl-D (EOF on stdin): Detaches from the inferlet (continues running)
    """
    instance_id = instance.instance_id
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

    def stdin_monitor():
        try:
            while True:
                data = sys.stdin.read(1)
                if not data:  # EOF
                    asyncio.get_event_loop().call_soon_threadsafe(eof_received.set)
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
                    await client.terminate_instance(instance_id)
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

                elif event == Event.Completed:
                    typer.echo(f"[Instance {short_id}] Completed: {message}")
                    return

                elif event in (
                    Event.Aborted,
                    Event.Exception,
                    Event.ServerError,
                    Event.OutOfResources,
                ):
                    typer.echo(f"[Instance {short_id}] {event.name}: {message}")
                    raise RuntimeError(f"inferlet terminated with status {event.name}")

                elif event == Event.Stdout:
                    at_line_start_stdout = _write_with_prefix(
                        False, message, short_id, at_line_start_stdout
                    )

                elif event == Event.Stderr:
                    at_line_start_stderr = _write_with_prefix(
                        True, message, short_id, at_line_start_stderr
                    )

                elif event == Event.Blob:
                    # Ignore binary blobs
                    pass

    finally:
        # Restore original signal handler
        signal.signal(signal.SIGINT, original_handler)


def stream_inferlet_output(instance: Instance, client: PieClient) -> None:
    """Stream output from an inferlet with signal handling (sync wrapper)."""
    asyncio.get_event_loop().run_until_complete(
        _stream_inferlet_output_async(instance, client)
    )


# Sync wrappers for common client operations
def ping(client: PieClient) -> None:
    """Ping the server (sync wrapper)."""
    asyncio.get_event_loop().run_until_complete(client.ping())


def list_instances(client: PieClient) -> list[InstanceInfo]:
    """List instances (sync wrapper)."""
    return asyncio.get_event_loop().run_until_complete(client.list_instances())


def terminate_instance(client: PieClient, instance_id: str) -> None:
    """Terminate an instance (sync wrapper)."""
    asyncio.get_event_loop().run_until_complete(client.terminate_instance(instance_id))


def attach_instance(client: PieClient, instance_id: str) -> Instance:
    """Attach to an instance (sync wrapper)."""
    return asyncio.get_event_loop().run_until_complete(
        client.attach_instance(instance_id)
    )


def upload_program(client: PieClient, program_bytes: bytes, manifest: str) -> None:
    """Upload a program (sync wrapper).

    Args:
        client: The PieClient instance.
        program_bytes: The WASM binary data.
        manifest: The manifest TOML content as a string.
    """
    asyncio.get_event_loop().run_until_complete(
        client.upload_program(program_bytes, manifest)
    )


def program_exists(
    client: PieClient,
    inferlet: str,
    wasm_hash: str | None = None,
    toml_hash: str | None = None,
) -> bool:
    """Check if a program exists (sync wrapper).

    The inferlet parameter can be:
    - Full name with version: "std/text-completion@0.1.0"
    - Without namespace (defaults to "std"): "text-completion@0.1.0"
    - Without version (defaults to "latest"): "std/text-completion" or "text-completion"

    Args:
        client: The Pie client.
        inferlet: The inferlet name (e.g., "std/text-completion@0.1.0").
        wasm_hash: Optional WASM binary hash to verify.
        toml_hash: Optional TOML manifest hash to verify.
            If hashes are provided, both must be specified together.
    """
    return asyncio.get_event_loop().run_until_complete(
        client.program_exists(inferlet, wasm_hash, toml_hash)
    )


def launch_instance(
    client: PieClient,
    inferlet: str,
    arguments: list[str],
    detached: bool = False,
) -> Instance:
    """Launch an instance (sync wrapper).

    This function performs a two-level search for the inferlet:
    1. First, it searches for the program among client-uploaded programs.
    2. If not found, it falls back to searching the registry.

    The inferlet parameter can be:
    - Full name with version: "std/text-completion@0.1.0"
    - Without namespace (defaults to "std"): "text-completion@0.1.0"
    - Without version (defaults to "latest"): "std/text-completion" or "text-completion"
    """
    return asyncio.get_event_loop().run_until_complete(
        client.launch_instance(inferlet, arguments, detached)
    )


def launch_instance_from_registry(
    client: PieClient,
    inferlet: str,
    arguments: list[str],
    detached: bool = False,
) -> Instance:
    """Launch an instance from the registry only (sync wrapper).

    Unlike `launch_instance`, this function searches only the registry and does not
    check client-uploaded programs. Use this when you explicitly want to launch
    an inferlet from the registry.

    The inferlet parameter can be:
    - Full name with version: "std/text-completion@0.1.0"
    - Without namespace (defaults to "std"): "text-completion@0.1.0"
    - Without version (defaults to "latest"): "std/text-completion" or "text-completion"
    """
    return asyncio.get_event_loop().run_until_complete(
        client.launch_instance_from_registry(inferlet, arguments, detached)
    )


def close_client(client: PieClient) -> None:
    """Close the client (sync wrapper)."""
    asyncio.get_event_loop().run_until_complete(client.close())
