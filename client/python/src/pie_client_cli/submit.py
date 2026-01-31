"""Submit command implementation for the Pie CLI.

This module implements the `pie-cli submit` subcommand for submitting inferlets
to an existing running Pie engine instance.
"""

import subprocess
import tempfile
import tomllib
from pathlib import Path
from typing import Optional

import blake3
import typer

from . import engine


def parse_manifest(manifest_content: str) -> tuple[str, str, str]:
    """Parse the manifest to extract namespace, name, and version.

    Args:
        manifest_content: The TOML manifest content as a string.

    Returns:
        A tuple of (namespace, name, version).

    Raises:
        ValueError: If the manifest is missing required fields or has invalid format.
    """
    manifest = tomllib.loads(manifest_content)

    package = manifest.get("package")
    if package is None:
        raise ValueError("Manifest missing [package] section")

    full_name = package.get("name")
    if full_name is None:
        raise ValueError("Manifest missing package.name field")

    version = package.get("version")
    if version is None:
        raise ValueError("Manifest missing package.version field")

    # Parse "namespace/name" format
    parts = full_name.split("/", 1)
    if len(parts) != 2:
        raise ValueError(
            f"Invalid package.name format '{full_name}': expected 'namespace/name'"
        )

    return parts[0], parts[1], version


def compose_components(program_bytes: bytes, library_paths: list[Path]) -> bytes:
    """Compose a program with multiple libraries using wac CLI.

    Uses `wac plug` command to link libraries into the main program.
    Libraries are linked sequentially in the order provided.

    Args:
        program_bytes: The main program WASM bytes.
        library_paths: List of paths to library WASM files.

    Returns:
        The composed WASM bytes.

    Raises:
        RuntimeError: If wac CLI is not available or composition fails.
    """
    # Check if wac is available
    try:
        result = subprocess.run(
            ["wac", "--version"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError("wac CLI is not available")
    except FileNotFoundError:
        raise RuntimeError(
            "wac CLI is not installed. Install it with: cargo install wac-cli"
        )

    socket_bytes = program_bytes

    for library_path in library_paths:
        if not library_path.exists():
            raise FileNotFoundError(f"Library file not found: {library_path}")

        # Read library bytes
        plug_bytes = library_path.read_bytes()

        # Compose using wac plug
        # wac plug --plug <library.wasm> <main.wasm> -o <output.wasm>
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            socket_file = temp_path / "socket.wasm"
            plug_file = temp_path / "plug.wasm"
            output_file = temp_path / "composed.wasm"

            socket_file.write_bytes(socket_bytes)
            plug_file.write_bytes(plug_bytes)

            result = subprocess.run(
                [
                    "wac",
                    "plug",
                    "--plug",
                    str(plug_file),
                    str(socket_file),
                    "-o",
                    str(output_file),
                ],
                capture_output=True,
                text=True,
            )

            if result.returncode != 0:
                raise RuntimeError(
                    f"wac plug failed for {library_path}:\n{result.stderr}"
                )

            socket_bytes = output_file.read_bytes()

    return socket_bytes


def handle_submit_command(
    inferlet: Optional[str] = None,
    path: Optional[Path] = None,
    manifest: Optional[Path] = None,
    config: Optional[Path] = None,
    host: Optional[str] = None,
    port: Optional[int] = None,
    username: Optional[str] = None,
    private_key_path: Optional[Path] = None,
    detached: bool = False,
    link: Optional[list[Path]] = None,
    arguments: Optional[list[str]] = None,
) -> None:
    """Handle the `pie-cli submit` command.

    You can specify an inferlet either by registry name or by path (mutually exclusive):

    - By registry: pie-client submit std/text-completion@0.1.0
    - By path: pie-client submit --path ./my_inferlet.wasm --manifest ./Pie.toml

    Steps:
    1. Creates a client configuration from config file and command-line arguments
    2. Connects to the Pie engine server
    3. If using path and libraries are specified, composes them with the inferlet using wac
    4. Uploads the composed inferlet if not already on server (path mode only)
    5. Launches the inferlet with the provided arguments
    6. In non-detached mode, streams the inferlet output with signal handling
    """
    # Validate at least one of inferlet or path is provided
    if inferlet is None and path is None:
        typer.echo("Error: Specify an inferlet name or --path", err=True)
        raise typer.Exit(1)

    # Handle the case where --path is used with -- separator
    # Positional args after -- get captured as `inferlet` first, so
    # prepend it to `arguments` instead
    if inferlet is not None and path is not None:
        arguments = [inferlet] + (arguments or [])
        inferlet = None

    # Validate manifest is provided when using --path
    if path is not None and manifest is None:
        typer.echo("Error: --manifest is required when using --path", err=True)
        raise typer.Exit(1)

    link = link or []
    arguments = arguments or []

    client_config = engine.ClientConfig.create(
        config_path=config,
        host=host,
        port=port,
        username=username,
        private_key_path=private_key_path,
    )

    client = engine.connect_and_authenticate(client_config)

    try:
        if path is not None:
            # Launch from local file path
            if not path.exists():
                raise FileNotFoundError(f"Inferlet file not found: {path}")

            if not manifest.exists():
                raise FileNotFoundError(f"Manifest file not found: {manifest}")

            inferlet_blob = path.read_bytes()
            manifest_content = manifest.read_text()

            # Parse the manifest to extract namespace, name, and version
            namespace, name, version = parse_manifest(manifest_content)
            typer.echo(f"Inferlet: {namespace}/{name}@{version}")

            # If libraries are specified, compose them with the main inferlet
            if link:
                final_blob = compose_components(inferlet_blob, link)
            else:
                final_blob = inferlet_blob

            # Calculate the hashes of the WASM blob and TOML manifest
            wasm_hash = blake3.blake3(final_blob).hexdigest()
            toml_hash = blake3.blake3(manifest_content.encode()).hexdigest()
            inferlet_name = f"{namespace}/{name}@{version}"

            # Upload the composed inferlet to the server (check both name and hashes match)
            if not engine.program_exists(client, inferlet_name, wasm_hash, toml_hash):
                engine.upload_program(client, final_blob, manifest_content)
                typer.echo("✅ Inferlet upload successful.")
            else:
                typer.echo("Inferlet already exists on server.")

            # Launch the instance
            inferlet_name = f"{namespace}/{name}@{version}"
            instance = engine.launch_instance(
                client,
                inferlet_name,
                arguments,
                detached,
            )
        else:
            # Launch from registry
            if link:
                typer.echo(
                    "Warning: --link option is ignored when launching from registry",
                    err=True,
                )

            typer.echo(f"Launching from registry: {inferlet}")

            instance = engine.launch_instance_from_registry(
                client,
                inferlet,
                arguments,
                detached,
            )

        typer.echo(f"✅ Inferlet launched with ID: {instance.instance_id}")

        if not detached:
            engine.stream_inferlet_output(instance, client)

    finally:
        engine.close_client(client)
