#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import urllib.error
import urllib.request
from typing import Any

REPOSITORY = pathlib.Path(__file__).resolve().parents[2]
FIXTURES = REPOSITORY / "crates/poem-compiler/tests/poem_compiler/adopted"
SPECIAL_TOKENS = ("bos_token", "eos_token", "pad_token", "unk_token")
COMPONENT_ROLES = {
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
WEIGHTLESS_PREFIXES = ("_", "tokenizer", "feature_")
TRIM_ABOVE_BYTES = 64 << 10
FLOAT_LIMIT = 1.7976931348623157e308


class Hub:
    def __init__(self, endpoint: str, token: str | None) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.token = token

    @classmethod
    def from_environment(cls) -> Hub:
        return cls(os.environ.get("HF_ENDPOINT", "https://huggingface.co"), os.environ.get("HF_TOKEN"))

    def json(self, repo: str, path: str) -> Any | None:
        request = urllib.request.Request(f"{self.endpoint}/{repo}/resolve/main/{path}")
        if self.token:
            request.add_header("Authorization", f"Bearer {self.token}")
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            if error.code == 404:
                return None
            raise


def pipeline_config(hub: Hub, repo: str, index: dict[str, Any]) -> dict[str, Any]:
    config: dict[str, Any] = {"pipeline": index}
    scheduler = hub.json(repo, "scheduler/scheduler_config.json")
    if scheduler is not None:
        config["scheduler"] = scheduler
    for folder, entry in index.items():
        if not isinstance(entry, list) or folder == "scheduler" or folder.startswith(WEIGHTLESS_PREFIXES):
            continue
        component = hub.json(repo, f"{folder}/config.json")
        if component is not None:
            config[COMPONENT_ROLES.get(folder, folder)] = component
    return config


def special_tokens(tokenizer_config: dict[str, Any] | None) -> dict[str, str]:
    tokens: dict[str, str] = {}
    for key in SPECIAL_TOKENS:
        value = (tokenizer_config or {}).get(key)
        if isinstance(value, dict):
            value = value.get("content")
        if isinstance(value, str):
            tokens[key] = value
    return tokens


def trimmed(value: Any, at: str, dropped: list[str]) -> Any:
    if not isinstance(value, dict):
        return value
    kept: dict[str, Any] = {}
    for key, item in value.items():
        path = f"{at}.{key}" if at else key
        if isinstance(item, (dict, list)) and len(json.dumps(item)) > TRIM_ABOVE_BYTES:
            if isinstance(item, dict) and len(item) < 64:
                kept[key] = trimmed(item, path, dropped)
            else:
                dropped.append(path)
        else:
            kept[key] = item
    return kept


def finite(value: Any) -> Any:
    if isinstance(value, float):
        if math.isnan(value):
            return None
        if math.isinf(value):
            return math.copysign(FLOAT_LIMIT, value)
        return value
    if isinstance(value, dict):
        return {key: finite(item) for key, item in value.items()}
    if isinstance(value, list):
        return [finite(item) for item in value]
    return value


def fixture(hub: Hub, repo: str) -> dict[str, Any]:
    index = hub.json(repo, "model_index.json")
    if index is not None:
        config = pipeline_config(hub, repo, index)
        tokenizer = hub.json(repo, "tokenizer/tokenizer_config.json")
    else:
        config = hub.json(repo, "config.json")
        tokenizer = hub.json(repo, "tokenizer_config.json")
    if config is None:
        raise SystemExit(f"{repo} has no config.json")
    dropped: list[str] = []
    record: dict[str, Any] = {
        "source": repo,
        "config": trimmed(config, "", dropped),
        "special": special_tokens(tokenizer),
    }
    if dropped:
        record["trimmed"] = dropped
    return finite(record)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Record a release's configuration as an adoption fixture for a model the catalog lists."
    )
    parser.add_argument("repo", help="Hugging Face repository, owner/name")
    parser.add_argument("model_id", help="catalog model id the release is listed as")
    parser.add_argument("--out", type=pathlib.Path, default=FIXTURES, help="fixture directory")
    args = parser.parse_args()

    record = fixture(Hub.from_environment(), args.repo)
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"{args.model_id}.json"
    path.write_text(json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(path)


if __name__ == "__main__":
    main()
