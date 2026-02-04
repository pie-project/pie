"""Submit command implementation for the Pie CLI.

This module implements the `pie-cli submit` subcommand for submitting inferlets
to an existing running Pie engine instance.
"""

import shutil
import subprocess
import tempfile
import tomllib
from pathlib import Path
from typing import Optional

import typer

from . import engine


def parse_manifest(manifest_content: str) -> tuple[str, str]:
    """Parse the manifest to extract name and version.

    Args:
        manifest_content: The TOML manifest content as a string.

    Returns:
        A tuple of (name, version).

    Raises:
        ValueError: If the manifest is missing required fields.
    """
    manifest = tomllib.loads(manifest_content)

    package = manifest.get("package")
    if package is None:
        raise ValueError("Manifest missing [package] section")

    name = package.get("name")
    if name is None:
        raise ValueError("Manifest missing package.name field")

    version = package.get("version")
    if version is None:
        raise ValueError("Manifest missing package.version field")

    return name, version


def compose_components(
    program_path: Path, library_paths: list[Path], output_path: Path
) -> None:
    """Compose a program with multiple libraries using wac CLI.

    Uses `wac plug` command to link libraries into the main program.
    Libraries are linked sequentially in the order provided.

    Args:
        program_path: Path to the main program WASM file.
        library_paths: List of paths to library WASM files.
        output_path: Path where the composed WASM binary will be written.

    Raises:
        RuntimeError: If wac CLI is not available or composition fails.
        FileNotFoundError: If program or library files are not found.
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

    if not program_path.exists():
        raise FileNotFoundError(f"Program file not found: {program_path}")

    # Start with the main program
    current_wasm = program_path

    for library_path in library_paths:
        if not library_path.exists():
            raise FileNotFoundError(f"Library file not found: {library_path}")

        # Compose using wac plug
        # wac plug --plug <library.wasm> <main.wasm> -o <output.wasm>
        with tempfile.NamedTemporaryFile(suffix=".wasm", delete=False) as tmp:
            tmp_output = Path(tmp.name)

        try:
            result = subprocess.run(
                [
                    "wac",
                    "plug",
                    "--plug",
                    str(library_path),
                    str(current_wasm),
                    "-o",
                    str(tmp_output),
                ],
                capture_output=True,
                text=True,
            )

            if result.returncode != 0:
                raise RuntimeError(
                    f"wac plug failed for {library_path}:\n{result.stderr}"
                )

            # Clean up previous temp file if it's not the original program
            if current_wasm != program_path:
                current_wasm.unlink()

            current_wasm = tmp_output
        except Exception:
            tmp_output.unlink(missing_ok=True)
            raise

    # Move the final composed binary to the output path
    if current_wasm != program_path:
        current_wasm.rename(output_path)
    else:
        # No libraries were linked, just copy the original
        shutil.copy(program_path, output_path)


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

    - By registry: pie-client submit text-completion@0.1.0
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

            manifest_content = manifest.read_text()

            # Parse the manifest to extract name and version
            name, version = parse_manifest(manifest_content)
            inferlet_name = f"{name}@{version}"

            typer.echo(f"Inferlet: {inferlet_name}")

            # If libraries are specified, compose them and always upload
            if link:
                with tempfile.NamedTemporaryFile(suffix=".wasm", delete=False) as tmp:
                    composed_path = Path(tmp.name)
                    try:
                        compose_components(path, link, composed_path)
                        engine.install_program(client, composed_path, manifest)
                    finally:
                        composed_path.unlink(missing_ok=True)
                typer.echo("✅ Inferlet installed successfully.")
            # No composition - check if program already exists before installing
            else:
                if not engine.program_exists(client, inferlet_name, path, manifest):
                    engine.install_program(client, path, manifest)
                    typer.echo("✅ Inferlet installed successfully.")
                else:
                    typer.echo("Inferlet already exists on server.")

            # Launch the instance
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
