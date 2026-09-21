"""Configuration management for the Pie CLI.

This module implements configuration file management including creation,
updating, and display of CLI settings.
"""

import getpass
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import toml
import typer

from pie_client.crypto import ParsedPrivateKey

from . import path as path_utils


@dataclass
class ConfigFile:
    """Configuration file structure."""

    host: Optional[str] = None
    port: Optional[int] = None
    username: Optional[str] = None
    private_key_path: Optional[str] = None
    enable_auth: Optional[bool] = None

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


def find_ssh_key() -> Optional[str]:
    """Find an SSH key in the user's ~/.ssh directory.

    Searches for keys in order: id_ed25519, id_rsa, id_ecdsa.
    Returns the path as a string (with ~) if found, None otherwise.
    """
    ssh_dir = Path.home() / ".ssh"

    for key_name in ["id_ed25519", "id_rsa", "id_ecdsa"]:
        key_path = ssh_dir / key_name
        if key_path.exists():
            return f"~/.ssh/{key_name}"

    return None


def validate_private_key(key_path: str) -> bool:
    """Validate that a private key at the given path can be parsed.

    Prints warnings if validation fails but does not raise exceptions.
    Returns True if the key is valid, False otherwise.
    """
    expanded_path = Path(key_path).expanduser()

    if not expanded_path.exists():
        typer.echo()
        typer.echo(f"⚠️ Warning: Private key file not found at '{expanded_path}'")
        return False

    # Check permissions on Unix
    if os.name == "posix":
        try:
            path_utils.check_private_key_permissions(expanded_path)
        except PermissionError as e:
            typer.echo()
            typer.echo(f"⚠️ Warning: {e}")
            return False

    try:
        key_content = expanded_path.read_text()
        ParsedPrivateKey.parse(key_content)
        return True
    except ValueError as e:
        typer.echo()
        typer.echo(f"⚠️ Warning: Failed to parse private key at '{expanded_path}'")
        typer.echo(f"   Error: {e}")
        return False
    except IOError as e:
        typer.echo()
        typer.echo(f"⚠️ Warning: Failed to read private key file at '{expanded_path}'")
        typer.echo(f"   Error: {e}")
        return False


def create_default_config_content(private_key_path: str, enable_auth: bool) -> str:
    """Create the default content of the config file."""
    key_path = private_key_path if enable_auth else ""
    username = getpass.getuser()

    return f"""host = "127.0.0.1"
port = 8080
username = "{username}"
private_key_path = "{key_path}"
enable_auth = {str(enable_auth).lower()}
"""


def handle_config_init(
    enable_auth: bool = True, custom_path: Optional[str] = None
) -> None:
    """Create a default config file.

    Args:
        enable_auth: Whether to enable authentication.
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

    # Find SSH key or use default
    found_key_path = find_ssh_key()
    default_key_path = "~/.ssh/id_ed25519"

    # Create config file
    content = create_default_config_content(
        found_key_path or default_key_path, enable_auth
    )
    config_path.write_text(content)

    typer.echo(f"✅ Created default configuration file at '{config_path}'")
    typer.echo(content)

    # Print messages about the private key if authentication is enabled
    if enable_auth:
        if found_key_path:
            typer.echo(f"✅ Using private key found at '{found_key_path}'")
            typer.echo("   You can update the key path in the config file:")
            typer.echo("      `pie-cli config update --private-key-path <path>`")

            if not validate_private_key(found_key_path):
                typer.echo(
                    "   The configuration has been saved, but you'll need to "
                    "provide a valid key to connect."
                )
        else:
            typer.echo()
            typer.echo(
                f"⚠️ Warning: Private key not found in '~/.ssh', using default path: '{default_key_path}'"
            )
            typer.echo(
                "   Please take either of the following actions when using authentication:"
            )
            typer.echo("   1. Generate an SSH key pair by running `ssh-keygen`")
            typer.echo("   2. Update the key path in the config file:")
            typer.echo("      `pie-cli config update --private-key-path <path>`")


def handle_config_update(
    host: Optional[str] = None,
    port: Optional[int] = None,
    username: Optional[str] = None,
    private_key_path: Optional[str] = None,
    enable_auth: Optional[bool] = None,
    custom_path: Optional[str] = None,
) -> None:
    """Update the specified entries of the config file.

    Args:
        host: New host value.
        port: New port value.
        username: New username value.
        private_key_path: New private key path.
        enable_auth: New enable_auth value.
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
    updated_key_path = None

    if host is not None:
        updated.append(f'host = "{host}"')
        config.host = host
    if port is not None:
        updated.append(f"port = {port}")
        config.port = port
    if username is not None:
        updated.append(f'username = "{username}"')
        config.username = username
    if private_key_path is not None:
        updated.append(f'private_key_path = "{private_key_path}"')
        updated_key_path = private_key_path
        config.private_key_path = private_key_path
    if enable_auth is not None:
        updated.append(f"enable_auth = {str(enable_auth).lower()}")
        config.enable_auth = enable_auth

    if not updated:
        typer.echo("⚠️ No fields provided to update.")
        return

    # Save updated config
    config.save(config_path)

    typer.echo(f"✅ Updated configuration file at '{config_path}'")
    typer.echo("   Updated fields:")
    for field_update in updated:
        typer.echo(f"   - {field_update}")

    # Validate key if updated
    if updated_key_path is not None:
        if not validate_private_key(updated_key_path):
            typer.echo(
                "   The configuration has been saved, but you'll need to "
                "provide a valid key to connect."
            )


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
