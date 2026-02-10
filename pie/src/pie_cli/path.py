"""Path utilities for Pie."""

import os
from pathlib import Path


def get_pie_home() -> Path:
    """Get the Pie home directory (~/.pie or PIE_HOME env var)."""
    if pie_home := os.environ.get("PIE_HOME"):
        return Path(pie_home)
    return Path.home() / ".pie"


def get_default_config_path() -> Path:
    """Get the default config file path (~/.pie/config.toml)."""
    return get_pie_home() / "config.toml"


def get_authorized_users_path() -> Path:
    """Get the authorized users file path (~/.pie/authorized_users.toml)."""
    return get_pie_home() / "authorized_users.toml"


def get_log_dir() -> Path:
    """Get the log directory ({PIE_HOME}/logs)."""
    return get_pie_home() / "logs"


def get_auth_dir() -> Path:
    """Get the auth directory ({PIE_HOME}/auth)."""
    return get_pie_home() / "auth"


def get_program_dir() -> Path:
    """Get the program/cache directory ({PIE_HOME}/programs)."""
    return get_pie_home() / "programs"
