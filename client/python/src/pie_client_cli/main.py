"""Pie CLI - Main entry point.

This module provides the main typer application with all subcommands.
"""

from pathlib import Path
from typing import Annotated, Optional

import typer

from . import abort as abort_cmd
from . import attach as attach_cmd
from . import config as config_cmd
from . import list as list_cmd
from . import ping as ping_cmd
from . import submit as submit_cmd

# Main application
app = typer.Typer(
    name="pie-cli",
    help="Programmable Inference Command Line Interface",
    no_args_is_help=True,
)

# Config subcommand group
config_app = typer.Typer(help="Manage configuration.")
app.add_typer(config_app, name="config")


# ============================================================================
# Path expansion callback for typer
# ============================================================================


def expand_path(path: Optional[Path]) -> Optional[Path]:
    """Expand ~ in paths."""
    if path is None:
        return None
    return path.expanduser()


# ============================================================================
# Common option types
# ============================================================================

ConfigOption = Annotated[
    Optional[Path],
    typer.Option("--config", help="Path to a custom TOML configuration file."),
]

HostOption = Annotated[
    Optional[str],
    typer.Option("--host", help="The network host to connect to."),
]

PortOption = Annotated[
    Optional[int],
    typer.Option("--port", help="The network port to connect to."),
]

UsernameOption = Annotated[
    Optional[str],
    typer.Option("--username", help="The username to use for authentication."),
]

PrivateKeyPathOption = Annotated[
    Optional[Path],
    typer.Option(
        "--private-key-path", help="Path to the private key file for authentication."
    ),
]


# ============================================================================
# Submit command
# ============================================================================


@app.command()
def submit(
    inferlet: Annotated[
        Optional[str],
        typer.Argument(
            help="Inferlet name from registry (e.g., 'std/text-completion@0.1.0')"
        ),
    ] = None,
    path: Annotated[
        Optional[Path],
        typer.Option("--path", "-p", help="Path to a local .wasm inferlet file"),
    ] = None,
    manifest: Annotated[
        Optional[Path],
        typer.Option(
            "--manifest",
            "-m",
            help="Path to the manifest TOML file (required when using --path)",
        ),
    ] = None,
    config: ConfigOption = None,
    host: HostOption = None,
    port: PortOption = None,
    username: UsernameOption = None,
    private_key_path: PrivateKeyPathOption = None,
    detached: Annotated[
        bool,
        typer.Option("-d", "--detached", help="Run the inferlet in detached mode."),
    ] = False,
    link: Annotated[
        Optional[list[Path]],
        typer.Option("-l", "--link", help="Paths to .wasm library files to link."),
    ] = None,
    arguments: Annotated[
        Optional[list[str]], typer.Argument(help="Arguments to pass to the inferlet.")
    ] = None,
) -> None:
    """Submit an inferlet to a running Pie engine.

    You can specify an inferlet either by registry name or by path (mutually exclusive):

    - By registry: pie-client submit std/text-completion@0.1.0
    - By path: pie-client submit --path ./my_inferlet.wasm --manifest ./Pie.toml
    """
    try:
        submit_cmd.handle_submit_command(
            inferlet=inferlet,
            path=expand_path(path),
            manifest=expand_path(manifest),
            config=expand_path(config),
            host=host,
            port=port,
            username=username,
            private_key_path=expand_path(private_key_path),
            detached=detached,
            link=[expand_path(p) for p in link] if link else None,
            arguments=arguments,
        )
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


# ============================================================================
# Ping command
# ============================================================================


@app.command()
def ping(
    config: ConfigOption = None,
    host: HostOption = None,
    port: PortOption = None,
    username: UsernameOption = None,
    private_key_path: PrivateKeyPathOption = None,
) -> None:
    """Check if the Pie engine is alive and responsive."""
    try:
        ping_cmd.handle_ping_command(
            config=expand_path(config),
            host=host,
            port=port,
            username=username,
            private_key_path=expand_path(private_key_path),
        )
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


# ============================================================================
# List command
# ============================================================================


@app.command("list")
def list_instances(
    config: ConfigOption = None,
    host: HostOption = None,
    port: PortOption = None,
    username: UsernameOption = None,
    private_key_path: PrivateKeyPathOption = None,
    full: Annotated[
        bool, typer.Option("--full", help="Display the full UUID.")
    ] = False,
    long: Annotated[
        bool, typer.Option("--long", help="Display the first 8 characters of the UUID.")
    ] = False,
) -> None:
    """List all running inferlet instances."""
    try:
        list_cmd.handle_list_command(
            config=expand_path(config),
            host=host,
            port=port,
            username=username,
            private_key_path=expand_path(private_key_path),
            full=full,
            long=long,
        )
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


# ============================================================================
# Attach command
# ============================================================================


@app.command()
def attach(
    instance_id_prefix: Annotated[
        str, typer.Argument(help="Prefix or full UUID of the instance to attach to.")
    ],
    config: ConfigOption = None,
    host: HostOption = None,
    port: PortOption = None,
    username: UsernameOption = None,
    private_key_path: PrivateKeyPathOption = None,
) -> None:
    """Attach to a running inferlet instance and stream its output."""
    try:
        attach_cmd.handle_attach_command(
            instance_id_prefix=instance_id_prefix,
            config=expand_path(config),
            host=host,
            port=port,
            username=username,
            private_key_path=expand_path(private_key_path),
        )
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


# ============================================================================
# Abort command
# ============================================================================


@app.command()
def abort(
    instance_id_prefix: Annotated[
        str, typer.Argument(help="Prefix or full UUID of the instance to terminate.")
    ],
    config: ConfigOption = None,
    host: HostOption = None,
    port: PortOption = None,
    username: UsernameOption = None,
    private_key_path: PrivateKeyPathOption = None,
) -> None:
    """Terminate a running inferlet instance."""
    try:
        abort_cmd.handle_abort_command(
            instance_id_prefix=instance_id_prefix,
            config=expand_path(config),
            host=host,
            port=port,
            username=username,
            private_key_path=expand_path(private_key_path),
        )
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


# ============================================================================
# Config subcommands
# ============================================================================


@config_app.command("init")
def config_init(
    enable_auth: Annotated[
        bool, typer.Option("--enable-auth/--no-auth", help="Enable authentication.")
    ] = True,
    path: Annotated[
        Optional[str],
        typer.Option("--path", help="Path where the config file should be saved."),
    ] = None,
) -> None:
    """Create a default config file."""
    try:
        config_cmd.handle_config_init(enable_auth=enable_auth, custom_path=path)
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


@config_app.command("update")
def config_update(
    host: Annotated[
        Optional[str], typer.Option("--host", help="Host to connect to.")
    ] = None,
    port: Annotated[
        Optional[int], typer.Option("--port", help="Port to connect to.")
    ] = None,
    username: Annotated[
        Optional[str], typer.Option("--username", help="Username for authentication.")
    ] = None,
    private_key_path: Annotated[
        Optional[str],
        typer.Option("--private-key-path", help="Path to private key file."),
    ] = None,
    enable_auth: Annotated[
        Optional[bool], typer.Option("--enable-auth", help="Enable authentication.")
    ] = None,
    path: Annotated[
        Optional[str], typer.Option("--path", help="Path to the config file to update.")
    ] = None,
) -> None:
    """Update the entries of the default config file."""
    try:
        config_cmd.handle_config_update(
            host=host,
            port=port,
            username=username,
            private_key_path=private_key_path,
            enable_auth=enable_auth,
            custom_path=path,
        )
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


@config_app.command("show")
def config_show(
    path: Annotated[
        Optional[str], typer.Option("--path", help="Path to the config file to show.")
    ] = None,
) -> None:
    """Show the content of the default config file."""
    try:
        config_cmd.handle_config_show(custom_path=path)
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
