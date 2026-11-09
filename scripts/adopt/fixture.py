#!/usr/bin/env python3
"""Fetch a checkpoint's config.json and special tokens as an adoption fixture.

    scripts/adopt/fixture.py <hf-repo> <catalog-model-id>

writes crates/poem-compiler/tests/poem_compiler/adopted/<id>.json, which
`every_adopting_family_reproduces_its_listed_models` adopts and compares,
trace for trace, with the model the catalog lists under that id. Reads
HF_TOKEN for gated repositories; fetches nothing but two small JSON files.
"""

import json
import os
import pathlib
import sys
import urllib.error
import urllib.request

FIXTURES = pathlib.Path(__file__).resolve().parents[2] / "crates/poem-compiler/tests/poem_compiler/adopted"
SPECIAL = ("bos_token", "eos_token", "pad_token", "unk_token")


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


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    repo, model_id = sys.argv[1:]
    config = fetch(repo, "config.json")
    if config is None:
        sys.exit(f"{repo} has no config.json")
    fixture = {
        "source": repo,
        "config": config,
        "special": special_tokens(fetch(repo, "tokenizer_config.json")),
    }
    FIXTURES.mkdir(parents=True, exist_ok=True)
    path = FIXTURES / f"{model_id}.json"
    path.write_text(json.dumps(fixture, indent=2, sort_keys=True) + "\n")
    print(path)


if __name__ == "__main__":
    main()
