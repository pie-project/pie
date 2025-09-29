"""Utility for finding repository root and setting up import paths."""

import sys
from pathlib import Path


def find_repo_root(start_path: Path = None) -> Path:
    """
    Find the repository root by looking for marker files.

    Searches upward from start_path (or current file's directory) until it finds
    a directory containing typical repo root markers like .git, pyproject.toml, etc.

    Args:
        start_path: Path to start searching from. Defaults to caller's file location.

    Returns:
        Path to the repository root

    Raises:
        RuntimeError: If repo root cannot be found
    """
    if start_path is None:
        # Get the caller's file path
        import inspect
        frame = inspect.currentframe().f_back
        start_path = Path(frame.f_globals['__file__']).parent

    current = Path(start_path).resolve()

    # Markers that indicate we've found the repo root
    root_markers = {
        '.git',           # Git repository
        'pyproject.toml', # Python project
        'Cargo.toml',     # Rust project
        'pie-cli',        # PIE-specific marker
        '.claude'         # Claude config (PIE-specific)
    }

    # Walk up the directory tree
    for parent in [current] + list(current.parents):
        if any((parent / marker).exists() for marker in root_markers):
            return parent

    raise RuntimeError(f"Could not find repository root starting from {start_path}")


def setup_pie_imports() -> None:
    """
    Set up sys.path to enable PIE imports from anywhere in the repo.

    Adds the repo root, common_python, and backend-python directories to sys.path,
    allowing imports to work consistently regardless of working directory.
    """
    repo_root = find_repo_root()
    common_python_path = repo_root / "backend" / "common_python"
    backend_python_path = repo_root / "backend" / "backend-python"

    repo_root_str = str(repo_root)
    common_python_str = str(common_python_path)
    backend_python_str = str(backend_python_path)

    # Add paths if not already present
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)

    if common_python_str not in sys.path:
        sys.path.insert(0, common_python_str)

    if backend_python_str not in sys.path:
        sys.path.insert(0, backend_python_str)


if __name__ == "__main__":
    # Test the utility
    root = find_repo_root()
    print(f"Repo root: {root}")
    setup_pie_imports()
    print("Import paths set up successfully")
