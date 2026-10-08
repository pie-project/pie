"""Configuration management for the Pie CLI.

This module implements configuration file management including creation,
updating, and display of CLI settings.
"""

import getpass
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import toml
import typer

from . import path as path_utils


@dataclass
class ConfigFile:
    """Configuration file structure."""

    host: Optional[str] = None
    port: Optional[int] = None
    username: Optional[str] = None

    @classmethod
    def load(cls, path: Path) -> "ConfigFile":
        """Load configuration from a TOML file."""
        content = path.read_text()
        data = toml.loads(content)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def save(self, path: Path) -> None:
        """Save configuration to a TOML file."""
        # Filter out None values
        data = {k: v for k, v in asdict(self).items() if v is not None}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(toml.dumps(data))


def create_default_config_content() -> str:
    """Create the default content of the config file."""
    username = getpass.getuser()

    return f"""host = "127.0.0.1"
port = 8080
username = "{username}"
"""


def handle_config_init(custom_path: Optional[str] = None) -> None:
    """Create a default config file.

    Args:
        custom_path: Custom path for the config file (uses default if not specified).
    """
    config_path = (
        Path(custom_path) if custom_path else path_utils.get_default_config_path()
    )

    # Check if config file already exists
    if config_path.exists():
        if not typer.confirm(
            f"Configuration file already exists at '{config_path}'. Overwrite?",
            default=False,
        ):
            typer.echo("Aborting. Configuration file was not overwritten.")
            return

    # Create parent directories
    config_path.parent.mkdir(parents=True, exist_ok=True)

    config_path.write_text(create_default_config_content())

    typer.echo(f"✅ Created default configuration file at '{config_path}'")
    typer.echo(config_path.read_text())


def handle_config_update(
    host: Optional[str] = None,
    port: Optional[int] = None,
    username: Optional[str] = None,
    custom_path: Optional[str] = None,
) -> None:
    """Update the specified entries of the config file.

    Args:
        host: New host value.
        port: New port value.
        username: New username value.
        custom_path: Custom path for the config file (uses default if not specified).
    """
    config_path = (
        Path(custom_path) if custom_path else path_utils.get_default_config_path()
    )

    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found at '{config_path}'. "
            "Run `pie-cli config init` first."
        )

    # Load existing config
    config = ConfigFile.load(config_path)

    # Track updates
    updated = []

    if host is not None:
        updated.append(f'host = "{host}"')
        config.host = host
    if port is not None:
        updated.append(f"port = {port}")
        config.port = port
    if username is not None:
        updated.append(f'username = "{username}"')
        config.username = username

    if not updated:
        typer.echo("⚠️ No fields provided to update.")
        return

    # Save updated config
    config.save(config_path)

    typer.echo(f"✅ Updated configuration file at '{config_path}'")
    typer.echo("   Updated fields:")
    for field_update in updated:
        typer.echo(f"   - {field_update}")


def handle_config_show(custom_path: Optional[str] = None) -> None:
    """Show the content of the config file.

    Args:
        custom_path: Custom path for the config file (uses default if not specified).
    """
    config_path = (
        Path(custom_path) if custom_path else path_utils.get_default_config_path()
    )

    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found at '{config_path}'. "
            "Run `pie-cli config init` first."
        )

    content = config_path.read_text()
    typer.echo(f"📄 Configuration file at '{config_path}':")
    typer.echo(content)
