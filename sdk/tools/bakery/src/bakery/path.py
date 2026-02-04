"""Path utilities for Bakery."""

import os
from pathlib import Path


def get_bakery_home() -> Path:
    """Get the Bakery home directory.

    Returns BAKERY_HOME environment variable if set, otherwise ~/.pie/bakery.
    """
    if bakery_home := os.environ.get("BAKERY_HOME"):
        return Path(bakery_home)
    return Path.home() / ".pie" / "bakery"


def get_config_path() -> Path:
    """Get the path to the bakery configuration file."""
    return get_bakery_home() / "config.toml"


def get_sdk_root() -> Path | None:
    """Find the SDK root directory.

    Searches in order:
    1. PIE_SDK environment variable
    2. Walk up from current directory looking for sdk/

    Returns:
        Path to SDK root, or None if not found.
    """
    if pie_sdk := os.environ.get("PIE_SDK"):
        path = Path(pie_sdk)
        if path.exists():
            return path

    current_dir = Path.cwd()
    for parent in [current_dir] + list(current_dir.parents):
        sdk_path = parent / "sdk"
        if sdk_path.exists() and sdk_path.is_dir():
            return sdk_path
    return None


def get_inferlet_js_path() -> Path:
    """Get the path to the inferlet-js library.

    Raises:
        FileNotFoundError: If inferlet-js cannot be found.
    """
    if pie_sdk := os.environ.get("PIE_SDK"):
        path = Path(pie_sdk) / "javascript"
        if path.exists():
            return path

    current_dir = Path.cwd()
    for parent in [current_dir] + list(current_dir.parents):
        inferlet_js_path = parent / "sdk" / "javascript"
        if inferlet_js_path.exists() and (inferlet_js_path / "package.json").exists():
            return inferlet_js_path

    raise FileNotFoundError(
        "Could not find inferlet-js library. Please set PIE_SDK environment variable."
    )


def get_wit_path() -> Path:
    """Get the path to the WIT directory containing the exec world.

    Raises:
        FileNotFoundError: If WIT directory cannot be found.
    """
    if pie_sdk := os.environ.get("PIE_SDK"):
        path = Path(pie_sdk) / "rust" / "inferlet" / "wit"
        if path.exists() and (path / "world.wit").exists():
            return path
        path = Path(pie_sdk) / "interfaces"
        if path.exists():
            return path

    current_dir = Path.cwd()
    for parent in [current_dir] + list(current_dir.parents):
        wit_path = parent / "sdk" / "rust" / "inferlet" / "wit"
        if wit_path.exists() and (wit_path / "world.wit").exists():
            return wit_path
        wit_path = parent / "sdk" / "interfaces"
        if wit_path.exists():
            return wit_path

    raise FileNotFoundError(
        "Could not find WIT directory. Please set PIE_SDK environment variable."
    )


def get_inferlet_py_path() -> Path:
    """Get the path to the Python inferlet library.

    Raises:
        FileNotFoundError: If inferlet library cannot be found.
    """
    if pie_sdk := os.environ.get("PIE_SDK"):
        path = Path(pie_sdk) / "python"
        if path.exists():
            return path

    current_dir = Path.cwd()
    for parent in [current_dir] + list(current_dir.parents):
        inferlet_path = parent / "sdk" / "python"
        if inferlet_path.exists() and (inferlet_path / "pyproject.toml").exists():
            return inferlet_path

    raise FileNotFoundError(
        "Could not find inferlet library. Please set PIE_SDK environment variable."
    )

