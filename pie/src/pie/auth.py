"""Authorization utilities for Pie."""

from pathlib import Path

import toml


def load_authorized_users(auth_path: Path) -> dict:
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


def save_authorized_users(auth_path: Path, users: dict) -> None:
    """Save authorized users to file in the format expected by the Rust runtime.

    Output format:
        [users.username.keys]
        key_name = "ssh-ed25519..."
    """
    auth_path.parent.mkdir(parents=True, exist_ok=True)
    formatted = {"users": {u: {"keys": k} for u, k in users.items()}}
    auth_path.write_text(toml.dumps(formatted))
