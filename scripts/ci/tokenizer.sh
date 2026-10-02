#!/usr/bin/env bash
# Qwen3.5's tokenizer files into <dir>, for the random-weight miniature.
set -euo pipefail
dir="${1:?dir}"; mkdir -p "$dir"
for f in tokenizer.json tokenizer_config.json chat_template.jinja; do
  [ -s "$dir/$f" ] || curl -sSL -o "$dir/$f" "https://huggingface.co/Qwen/Qwen3.5-0.8B/resolve/main/$f"
done
