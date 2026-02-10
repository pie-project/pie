"""Authorization management commands for Pie CLI.

Implements: pie auth add|remove|list
"""

import sys
from datetime import datetime
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from pie_cli import path as pie_path

import toml


def load_authorized_users(auth_path) -> dict:
    """Load authorized users from file, or return empty dict if not exists.

    Expected format (compatible with Rust runtime):
        [users.username.keys]
        key_name = "ssh-ed25519..."
    """
    if not auth_path.exists():
        return {}
    data = toml.loads(auth_path.read_text())
    if "users" not in data:
        return {}
    return {u: k.get("keys", {}) for u, k in data["users"].items()}


def save_authorized_users(auth_path, users: dict) -> None:
    """Save authorized users to file in the format expected by the Rust runtime.

    Output format:
        [users.username.keys]
        key_name = "ssh-ed25519..."
    """
    auth_path.parent.mkdir(parents=True, exist_ok=True)
    formatted = {"users": {u: {"keys": k} for u, k in users.items()}}
    auth_path.write_text(toml.dumps(formatted))

console = Console()
app = typer.Typer(help="Manage authorized clients")


@app.command("add")
def auth_add(
    username: str = typer.Argument(..., help="Username of the user"),
    key_name: str | None = typer.Argument(
        None, help="Optional name for this key (defaults to timestamp)"
    ),
) -> None:
    """Add an authorized user and its public key.

    The public key is read from stdin and can be in OpenSSH, PKCS#8 PEM, or PKCS#1 PEM format.
    """
    # Generate key name if not provided
    if key_name is None:
        key_name = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")

    # Show prompts only in interactive mode
    if sys.stdin.isatty():
        console.print()
        console.print(f"[bold]Adding authorized user:[/bold] {username}")
        console.print(f"[dim]Key name: {key_name}[/dim]")
        console.print()
        console.print("[dim]Enter public key (paste, then Ctrl-D on a new line):[/dim]")
        console.print("[dim]Supported: RSA, ED25519, ECDSA in OpenSSH/PEM format[/dim]")
        console.print()

    # Read public key from stdin
    public_key = sys.stdin.read().strip()

    auth_path = pie_path.get_authorized_users_path()
    users = load_authorized_users(auth_path)

    # Create user entry if it doesn't exist
    user_created = False
    if username not in users:
        users[username] = {}
        user_created = True

    if not public_key:
        console.print()
        console.print("[yellow]![/yellow] No public key provided")
        save_authorized_users(auth_path, users)
        if user_created:
            console.print(f"[green]✓[/green] Created user '{username}' without keys")
        else:
            console.print(f"[dim]User '{username}' already exists[/dim]")
        return

    # Check if key name already exists
    if key_name in users[username]:
        console.print()
        console.print(f"[red]✗[/red] Key '{key_name}' already exists for '{username}'")
        raise typer.Exit(1)

    # Add the key (store the raw public key string)
    users[username][key_name] = public_key
    save_authorized_users(auth_path, users)

    console.print()
    if user_created:
        console.print(
            f"[green]✓[/green] Created user '{username}' with key '{key_name}'"
        )
    else:
        console.print(f"[green]✓[/green] Added key '{key_name}' to '{username}'")


@app.command("remove")
def auth_remove(
    username: str = typer.Argument(..., help="Username of the user"),
    key_name: str | None = typer.Argument(
        None, help="Optional name of the specific key to remove"
    ),
) -> None:
    """Remove an authorized user or a specific key.

    If key_name is provided, only that key is removed.
    If key_name is not provided, the entire user entry is removed.
    """
    auth_path = pie_path.get_authorized_users_path()

    if not auth_path.exists():
        console.print("[red]✗[/red] No authorized users file")
        raise typer.Exit(1)

    users = load_authorized_users(auth_path)

    if username not in users:
        console.print(f"[red]✗[/red] User '{username}' not found")
        raise typer.Exit(1)

    if key_name:
        # Remove specific key
        if key_name not in users[username]:
            console.print(f"[red]✗[/red] Key '{key_name}' not found for '{username}'")
            raise typer.Exit(1)

        del users[username][key_name]
        save_authorized_users(auth_path, users)
        console.print(f"[green]✓[/green] Removed key '{key_name}' from '{username}'")
    else:
        # Remove entire user - prompt for confirmation in interactive mode
        key_count = len(users[username])
        if sys.stdin.isatty():
            confirm = typer.confirm(f"Remove user '{username}' and {key_count} key(s)?")
            if not confirm:
                console.print("[dim]Cancelled.[/dim]")
                raise typer.Exit(1)

        del users[username]
        save_authorized_users(auth_path, users)
        console.print(f"[green]✓[/green] Removed user '{username}' and all keys")


@app.command("list")
def auth_list() -> None:
    """List all authorized users and their keys."""
    auth_path = pie_path.get_authorized_users_path()

    if not auth_path.exists():
        console.print(
            Panel(
                "[dim]No authorized users[/dim]",
                title="Authorized Users",
                title_align="left",
                border_style="dim",
            )
        )
        return

    users = load_authorized_users(auth_path)

    if not users:
        console.print(
            Panel(
                "[dim]No authorized users[/dim]",
                title="Authorized Users",
                title_align="left",
                border_style="dim",
            )
        )
        return

    lines = Text()
    for i, username in enumerate(sorted(users.keys())):
        if i > 0:
            lines.append("\n")
        user_keys = users[username]
        key_names = ", ".join(sorted(user_keys.keys())) if user_keys else "no keys"
        lines.append(f"{username:<20}", style="white")
        lines.append(f"{len(user_keys)} keys: {key_names}", style="dim")

    console.print(
        Panel(lines, title="Authorized Users", title_align="left", border_style="dim")
    )
    console.print(f"[dim]{auth_path}[/dim]")
