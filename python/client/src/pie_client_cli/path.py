"""Path utilities for the Pie CLI.

This module provides functions for working with Pie-specific paths and directories.
"""

import os
from pathlib import Path


def get_pie_home() -> Path:
    """Get the Pie CLI home directory.

    Returns PIE_CLI_HOME environment variable if set, otherwise ~/.pie_cli.
    """
    if pie_home := os.environ.get("PIE_CLI_HOME"):
        return Path(pie_home)
    return Path.home() / ".pie"


def get_default_config_path() -> Path:
    """Get the path to the default configuration file."""
    return get_pie_home() / "cli_config.toml"


def expand_tilde(path: str) -> Path:
    """Expand ~ in a path string to the user's home directory."""
    return Path(path).expanduser()
