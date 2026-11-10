#!/usr/bin/env python3
"""Fetch a checkpoint's config.json and special tokens as an adoption fixture.

    scripts/adopt/fixture.py <hf-repo> <catalog-model-id>

writes crates/poem-compiler/tests/poem_compiler/adopted/<id>.json, which
`every_adopting_family_reproduces_its_listed_models` adopts and compares,
trace for trace, with the model the catalog lists under that id. Reads
HF_TOKEN for gated repositories; fetches nothing but small JSON files (for a
diffusers pipeline, model_index.json and each component's config.json).
"""

import json
import os
import pathlib
import sys
import urllib.error
import urllib.request

FIXTURES = pathlib.Path(__file__).resolve().parents[2] / "crates/poem-compiler/tests/poem_compiler/adopted"
SPECIAL = ("bos_token", "eos_token", "pad_token", "unk_token")

# A diffusers component's tensors are prefixed by its role, as
# crates/checkpoint/src/file/diffusers.rs `prefix_of` names it.
ROLES = {
    "transformer": "dit",
    "transformer_2": "dit2",
    "text_encoder": "te",
    "text_encoder_2": "te2",
    "vae": "vae",
    "video_vae": "vae",
    "image_encoder": "ie",
    "audio_vae": "avae",
    "vocoder": "voc",
}


def pipeline_config(repo: str, index: dict) -> dict:
    """A pipeline's configuration as `pie model adopt` hands it to a hook:
    each weighted component's config.json under its role, beside the index."""
    out = {"pipeline": index}
    scheduler = fetch(repo, "scheduler/scheduler_config.json")
    if scheduler is not None:
        out["scheduler"] = scheduler
    for folder, entry in index.items():
        if folder.startswith("_") or not isinstance(entry, list) or folder == "scheduler":
            continue
        if folder.startswith("tokenizer") or folder.startswith("feature_"):
            continue
        config = fetch(repo, f"{folder}/config.json")
        if config is not None:
            out[ROLES.get(folder, folder)] = config
    return out


def fetch(repo: str, file: str):
    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    request = urllib.request.Request(f"{endpoint}/{repo}/resolve/main/{file}")
    token = os.environ.get("HF_TOKEN")
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return None
        raise


def special_tokens(tokenizer_config) -> dict:
    out = {}
    for key in SPECIAL:
        value = (tokenizer_config or {}).get(key)
        if isinstance(value, dict):
            value = value.get("content")
        if isinstance(value, str):
            out[key] = value
    return out


def finite(value):
    """`value` as JSON can hold it: Python's Infinity as the largest double,
    NaN as null, as `pie model adopt` reads them."""
    if isinstance(value, float):
        if value != value:
            return None
        if value in (float("inf"), float("-inf")):
            return 1.7976931348623157e308 if value > 0 else -1.7976931348623157e308
        return value
    if isinstance(value, dict):
        return {k: finite(v) for k, v in value.items()}
    if isinstance(value, list):
        return [finite(v) for v in value]
    return value


# A per-tensor table (modelopt's `quantized_layers`, compressed-tensors'
# `config_groups` targets) can run to megabytes. No hook reads one that
# large, so a fixture records where it was left out instead.
TRIM_ABOVE = 64 << 10


def trimmed(value, at="", dropped=None):
    """`value` with every entry whose JSON exceeds TRIM_ABOVE left out, and
    the dotted paths of what was."""
    dropped = [] if dropped is None else dropped
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            path = f"{at}.{key}" if at else key
            if len(json.dumps(item)) > TRIM_ABOVE and not isinstance(item, (str, int, float)):
                if isinstance(item, dict) and len(item) < 64:
                    out[key] = trimmed(item, path, dropped)[0]
                else:
                    dropped.append(path)
            else:
                out[key] = item
        return out, dropped
    return value, dropped


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    repo, model_id = sys.argv[1:]
    index = fetch(repo, "model_index.json")
    if index is not None:
        config = pipeline_config(repo, index)
        tokenizer = fetch(repo, "tokenizer/tokenizer_config.json")
    else:
        config = fetch(repo, "config.json")
        tokenizer = fetch(repo, "tokenizer_config.json")
    if config is None:
        sys.exit(f"{repo} has no config.json")
    config, dropped = trimmed(config)
    fixture = {
        "source": repo,
        "config": config,
        "special": special_tokens(tokenizer),
    }
    if dropped:
        fixture["trimmed"] = dropped
    FIXTURES.mkdir(parents=True, exist_ok=True)
    path = FIXTURES / f"{model_id}.json"
    path.write_text(json.dumps(finite(fixture), indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(path)


if __name__ == "__main__":
    main()
