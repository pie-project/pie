"""HuggingFace utilities for PIE backend.

This module provides utilities for:
- Resolving HuggingFace cache paths
- Loading model config from HuggingFace's config.json
- Loading tokenizer from HuggingFace's tokenizer.json
- Architecture mapping from HuggingFace to PIE
"""

import json
from pathlib import Path

from huggingface_hub.constants import HF_HUB_CACHE


# Mapping from HuggingFace model_type to PIE architecture
HF_TO_PIE_ARCH = {
    "llama": "llama3",
    "qwen2": "qwen2",
    "qwen3": "qwen3",
    "olmo3": "olmo3",
    "gptoss": "gptoss",
    "gpt_oss": "gptoss",  # HuggingFace config may use underscore variant
    "gemma2": "gemma2",
    "gemma3_text": "gemma3",
    "mistral3": "mistral3",  # Ministral models (e.g., Ministral-3-3B-Instruct-2512)
}



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


def load_hf_tokenizer(snapshot_dir: Path) -> dict:
    """Load tokenizer from HuggingFace format.

    Parses tokenizer.json and tokenizer_config.json to construct
    the PIE tokenizer format:
    - merge_table: dict[int, bytes] (rank -> token bytes)
    - special_tokens: dict[str, int]
    - split_regex: str
    - chat_template: str (Jinja format)

    Args:
        snapshot_dir: Path to the HuggingFace snapshot directory

    Returns:
        Tokenizer configuration dict in PIE format
    """
    result = {
        "type": "bpe",
        "num_vocab": 0,
        "merge_table": {},
        "special_tokens": {},
        "split_regex": "",
        "escape_non_printable": False,
        "sentencepiece_space": False,
    }

    # Load tokenizer.json for vocabulary
    tokenizer_path = snapshot_dir / "tokenizer.json"
    if tokenizer_path.exists():
        with open(tokenizer_path) as f:
            tokenizer_data = json.load(f)

        # Extract vocabulary from model.vocab
        model = tokenizer_data.get("model", {})
        vocab = model.get("vocab", {})

        # Build merge_table: rank -> bytes
        # HF vocab is {token_string: rank}, we need {rank: bytes}
        merge_table = {}
        for token_str, rank in vocab.items():
            try:
                # Encode token string to bytes
                token_bytes = token_str.encode("utf-8")
                merge_table[rank] = token_bytes
            except UnicodeEncodeError:
                # Skip tokens that can't be encoded
                continue

        # Auto-detect escape_non_printable
        # GPT-2, RoBERTa, and Qwen tokenizers use a byte-level mapping where
        # non-printable bytes are mapped to unicode characters starting at U+0100.
        # The key indicator is a standalone single-character token "Ā" (U+0100)
        # at a LOW token ID (< 256), which represents byte 0x00 in these tokenizers.
        # If "Ā" exists but has a HIGH token ID (like 239503 in Gemma 2), it's just
        # a linguistic character, not a byte-level mapping token.
        # Mistral Tekken starts byte mapping at 1000.
        A_macron_id = vocab.get("\u0100")  # U+0100 = Ā
        if A_macron_id is not None and (A_macron_id < 256 or A_macron_id == 1000):
            result["escape_non_printable"] = True

        # Auto-detect SentencePiece tokenizers
        # SentencePiece uses ▁ (U+2581) to encode spaces at word boundaries.
        # Check if any of the first 1000 tokens contain this character.
        sentencepiece_detected = any("▁" in tok for tok in list(vocab.keys())[:1000])
        result["sentencepiece_space"] = sentencepiece_detected

        result["merge_table"] = merge_table

        # Calculate true vocab size based on max index to avoid "id out of range" errors
        # Start with base vocab
        max_id = 0
        if vocab:
            max_id = max(vocab.values())

        # Extract added tokens for special tokens and update max_id
        added_tokens = tokenizer_data.get("added_tokens", [])
        for token_info in added_tokens:
            tid = token_info.get("id")
            if tid is not None:
                max_id = max(max_id, tid)

            # Add ALL added tokens to special_tokens, not just those marked special=True
            # This ensures we handle tokens like <think> (which might be special=False) correctly
            # during detokenization.
            content = token_info.get("content", "")
            if content and tid is not None:
                result["special_tokens"][content] = tid

        result["num_vocab"] = max_id + 1

        # Extract pre_tokenizer pattern (split_regex)
        pre_tokenizer = tokenizer_data.get("pre_tokenizer", {})
        pt_type = pre_tokenizer.get("type", "")
        
        if pt_type == "Sequence":
            # Common case: Sequence of pretokenizers (GPT-style, Qwen)
            pretokenizers = pre_tokenizer.get("pretokenizers", [])
            for pt in pretokenizers:
                if pt.get("type") == "Split" and "pattern" in pt:
                    pattern = pt["pattern"]
                    if isinstance(pattern, dict) and "Regex" in pattern:
                        result["split_regex"] = pattern["Regex"]
                        break
        elif pt_type == "Split":
            # Direct Split pretokenizer (Gemma 2)
            pattern = pre_tokenizer.get("pattern", {})
            if isinstance(pattern, dict):
                if "Regex" in pattern:
                    result["split_regex"] = pattern["Regex"]
                elif "String" in pattern:
                    # Simple string split (e.g., " ") - escape for regex
                    import re
                    escaped = re.escape(pattern["String"])
                    # Create a regex that splits on the pattern but keeps content
                    # For space-based splitting, match non-space runs
                    if pattern["String"] == " ":
                        result["split_regex"] = r"[^ ]+"
                    else:
                        result["split_regex"] = f"[^{escaped}]+"
        
        # Fallback: if no split_regex found, use a sensible default
        # This matches Unicode word characters and non-space sequences
        if not result.get("split_regex"):
            result["split_regex"] = r"[^\s]+"

    return result


def get_safetensor_files(snapshot_dir: Path) -> list[str]:
    """Get list of safetensor files in a snapshot directory.

    Returns the filenames (not full paths) of all *.safetensors files.
    """
    files = []
    for f in snapshot_dir.iterdir():
        if f.suffix == ".safetensors":
            files.append(f.name)
    return sorted(files)
