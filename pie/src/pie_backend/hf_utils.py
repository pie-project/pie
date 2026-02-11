"""HuggingFace utilities for PIE backend.

This module provides utilities for:
- Resolving HuggingFace cache paths
- Loading model config from HuggingFace's config.json
- Architecture mapping from HuggingFace to PIE
"""

import json
from pathlib import Path

from huggingface_hub.constants import HF_HUB_CACHE

from .model import HF_TO_PIE_ARCH



def get_hf_cache_dir() -> Path:
    """Get the HuggingFace cache directory.

    Uses huggingface_hub's HF_HUB_CACHE which respects environment
    variable overrides (e.g., HF_HUB_CACHE, HF_HOME).
    """
    return Path(HF_HUB_CACHE)


def get_hf_snapshot_dir(repo_id: str) -> Path:
    """Get the snapshot directory for a HuggingFace model.

    Args:
        repo_id: HuggingFace repo ID (e.g., "meta-llama/Llama-3.2-1B-Instruct")

    Returns:
        Path to the snapshot directory containing model files

    Raises:
        ValueError: If the model is not found in cache
    """
    cache_dir = get_hf_cache_dir()

    # Convert repo_id to cache dirname format: org/repo -> models--org--repo
    dirname = "models--" + repo_id.replace("/", "--")
    model_cache = cache_dir / dirname

    if not model_cache.exists():
        raise ValueError(
            f"Model '{repo_id}' not found in HuggingFace cache. "
            f"Run 'pie model download {repo_id}' first."
        )

    # Find snapshot directory
    snapshots_dir = model_cache / "snapshots"
    if not snapshots_dir.exists():
        raise ValueError(f"No snapshots found for '{repo_id}'")

    # Get the most recent snapshot (usually there's only one)
    # HF uses commit hashes for snapshot dirs, we take the first one
    snapshots = sorted(snapshots_dir.iterdir())
    if not snapshots:
        raise ValueError(f"No snapshots found for '{repo_id}'")

    return snapshots[0]


def load_hf_config(snapshot_dir: Path) -> dict:
    """Load and parse config.json from HuggingFace model snapshot.

    Args:
        snapshot_dir: Path to the HuggingFace snapshot directory

    Returns:
        Parsed config dictionary with normalized field names
    """
    config_path = snapshot_dir / "config.json"
    if not config_path.exists():
        raise ValueError(f"config.json not found in {snapshot_dir}")

    with open(config_path) as f:
        config = json.load(f)

    return config




def get_safetensor_files(snapshot_dir: Path) -> list[str]:
    """Get list of safetensor files in a snapshot directory.

    Returns the filenames (not full paths) of all *.safetensors files.
    """
    files = []
    for f in snapshot_dir.iterdir():
        if f.suffix == ".safetensors":
            files.append(f.name)
    return sorted(files)


def parse_repo_id_from_dirname(dirname: str) -> str | None:
    """Parse HuggingFace repo ID from cache directory name.

    HF cache uses format: models--{org}--{repo}
    Returns: org/repo or None if not a valid model directory
    """
    if not dirname.startswith("models--"):
        return None
    parts = dirname[8:].split("--")  # Remove "models--" prefix
    if len(parts) == 2:
        return f"{parts[0]}/{parts[1]}"
    elif len(parts) == 1:
        return parts[0]  # No org, just repo name
    return None


def check_pie_compatibility(config: dict | None) -> tuple[bool, str]:
    """Check if a model is compatible with Pie.

    Returns: (is_compatible, arch_name or reason)
    """
    if config is None:
        return False, "no config"

    model_type = config.get("model_type", "")
    if model_type in HF_TO_PIE_ARCH:
        return True, HF_TO_PIE_ARCH[model_type]

    return False, f"unsupported type: {model_type}"

