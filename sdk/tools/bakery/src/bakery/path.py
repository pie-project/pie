"""Path utilities for Bakery.

Resolves the locations bakery needs at build time: the SDK root (where
``sdk/rust/inferlet/wit/`` and friends live), and the per-language
inferlet libraries.

Resolution order for each path:
  1. ``PIE_SDK`` env var (explicit override; users opt in)
  2. Walk upward from the current working directory looking for ``sdk/``
  3. Walk upward from this package's install location — covers the
     ``uv pip install -e sdk/tools/bakery`` case where bakery is editable
     inside the pie checkout but the user invokes ``pie build`` from
     somewhere else (e.g. their inferlet directory under /tmp).

If all three fail we raise a ``FileNotFoundError`` with a clear hint
about ``PIE_SDK``.
"""

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


def _candidate_roots() -> list[Path]:
    """Where to start the upward walk from.

    cwd first (developer working in their inferlet), then the directory
    bakery itself was installed from (covers the case where bakery is
    editable inside a pie checkout but invoked from outside).
    """
    return [Path.cwd(), Path(__file__).resolve().parent]


def _find_upward(anchor: Path) -> Path | None:
    """Walk upward from each candidate root looking for ``parent / anchor``.

    Returns the matching ``parent`` (the directory *containing* the
    anchor), so callers can take any sub-path of it.
    """
    for start in _candidate_roots():
        for parent in [start] + list(start.parents):
            if (parent / anchor).exists():
                return parent
    return None


def get_sdk_root() -> Path | None:
    """Find the SDK root directory.

    Searches in order: PIE_SDK env var, then upward from cwd / bakery
    install location for a ``sdk/rust/inferlet/wit`` anchor (specific
    enough to avoid matching unrelated ``sdk/`` directories).

    Returns:
        Path to ``sdk/``, or None if not found.
    """
    if pie_sdk := os.environ.get("PIE_SDK"):
        path = Path(pie_sdk)
        if path.exists():
            return path
    parent = _find_upward(Path("sdk/rust/inferlet/wit"))
    return parent / "sdk" if parent is not None else None


def get_inferlet_js_path() -> Path:
    """Get the path to the inferlet-js library.

    Raises:
        FileNotFoundError: If inferlet-js cannot be found.
    """
    if pie_sdk := os.environ.get("PIE_SDK"):
        path = Path(pie_sdk) / "javascript"
        if path.exists():
            return path
    parent = _find_upward(Path("sdk/javascript/package.json"))
    if parent is not None:
        return parent / "sdk" / "javascript"
    raise FileNotFoundError(
        "Could not find inferlet-js library. Please set PIE_SDK environment variable."
    )


def get_wit_path() -> Path:
    """Get the path to the WIT directory containing the exec world.

    Raises:
        FileNotFoundError: If WIT directory cannot be found.
    """
    if pie_sdk := os.environ.get("PIE_SDK"):
        wit = Path(pie_sdk) / "rust" / "inferlet" / "wit"
        if wit.exists() and (wit / "world.wit").exists():
            return wit
        legacy = Path(pie_sdk) / "interfaces"
        if legacy.exists():
            return legacy
    parent = _find_upward(Path("sdk/rust/inferlet/wit/world.wit"))
    if parent is not None:
        return parent / "sdk" / "rust" / "inferlet" / "wit"
    parent = _find_upward(Path("sdk/interfaces"))
    if parent is not None:
        return parent / "sdk" / "interfaces"
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
    parent = _find_upward(Path("sdk/python/pyproject.toml"))
    if parent is not None:
        return parent / "sdk" / "python"
    raise FileNotFoundError(
        "Could not find inferlet library. Please set PIE_SDK environment variable."
    )
