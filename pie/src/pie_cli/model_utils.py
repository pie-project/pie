"""Model utilities for Pie."""

import json
from pathlib import Path

from huggingface_hub.constants import HF_HUB_CACHE


# Mapping from HuggingFace model_type to Pie architecture
HF_TO_PIE_ARCH = {
    "llama": "llama3",
    "qwen2": "qwen2",
    "qwen3": "qwen3",
    "gptoss": "gptoss",
    "gpt_oss": "gptoss",  # HuggingFace config may use underscore variant
    "gemma2": "gemma2",
}



def get_hf_cache_dir() -> Path:
    """Get the HuggingFace cache directory.

    Uses huggingface_hub's HF_HUB_CACHE which respects environment
    variable overrides (e.g., HF_HUB_CACHE, HF_HOME).
    """
    return Path(HF_HUB_CACHE)


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


def get_model_config(cache_dir: Path, repo_id: str) -> dict | None:
    """Get config.json from cached model snapshot."""
    # Convert repo_id to cache dirname format
    dirname = "models--" + repo_id.replace("/", "--")
    model_cache = cache_dir / dirname

    if not model_cache.exists():
        return None

    # Find snapshot directory
    snapshots_dir = model_cache / "snapshots"
    if not snapshots_dir.exists():
        return None

    # Use first available snapshot
    snapshots = list(snapshots_dir.iterdir())
    if not snapshots:
        return None

    config_path = snapshots[0] / "config.json"
    if not config_path.exists():
        return None

    try:
        with open(config_path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
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
