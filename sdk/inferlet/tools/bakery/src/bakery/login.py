"""Login command implementation for Bakery.

This module implements the `bakery login` command for authenticating
with the Pie Registry using GitHub OAuth.
"""

import webbrowser
from urllib.parse import urlencode

import typer

from .config import ConfigFile
from . import path as path_utils
from .registry import REGISTRY_URL, RegistryClient, RegistryError


# Callback URL that shows the token for manual copy
OAUTH_CALLBACK_URL = "https://registry.pie-project.org/api/v1/auth/github/callback"


def handle_login_command(
    registry_url: str = REGISTRY_URL,
) -> None:
    """Handle the `bakery login` command.

    1. Opens browser for GitHub OAuth
    2. User authenticates and gets a token
    3. User pastes the token back into CLI
    4. Token is validated and saved to config

    Args:
        registry_url: Base URL for the registry API.
    """
    # Build the OAuth URL
    auth_url = f"{registry_url}/auth/github"
    params = {"redirect_uri": OAUTH_CALLBACK_URL}
    full_url = f"{auth_url}?{urlencode(params)}"

    typer.echo("🔐 Opening browser for GitHub authentication...")
    typer.echo()
    typer.echo(f"   If the browser doesn't open, visit:")
    typer.echo(f"   {full_url}")
    typer.echo()

    # Try to open the browser
    try:
        webbrowser.open(full_url)
    except Exception:
        pass  # Browser may not be available

    typer.echo("After authenticating with GitHub, you'll receive a token.")
    typer.echo("Please paste it below:")
    typer.echo()

    # Get the token from user
    token = typer.prompt("Token", hide_input=False)
    token = token.strip()

    if not token:
        typer.echo("❌ No token provided. Login cancelled.", err=True)
        raise typer.Exit(1)

    # Validate the token by making a request
    typer.echo()
    typer.echo("🔍 Validating token...")

    try:
        with RegistryClient(token=token, base_url=registry_url) as client:
            user = client.get_me()

            typer.echo()
            typer.echo(f"✅ Authenticated as: {user.login}")
            if user.name:
                typer.echo(f"   Name: {user.name}")
            if user.is_superuser:
                typer.echo("   🌟 Superuser (can publish to std)")

    except RegistryError as e:
        typer.echo(f"❌ Token validation failed: {e.detail}", err=True)
        raise typer.Exit(1)

    # Save the token to config
    config_path = path_utils.get_config_path()

    if config_path.exists():
        config = ConfigFile.load(config_path)
    else:
        config = ConfigFile()

    config.registry_token = token
    config.save(config_path)

    typer.echo()
    typer.echo(f"✅ Token saved to: {config_path}")
    typer.echo()
    typer.echo("You can now publish inferlets with:")
    typer.echo("   bakery inferlet publish")
