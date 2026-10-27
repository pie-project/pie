#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
(cd examples && cargo build -q --release --target wasm32-wasip3 -p text-completion)
mkdir -p tests/browser/inferlets
cp examples/target/wasm32-wasip3/release/text_completion.wasm tests/browser/inferlets/
mkdir -p tests/browser/languages
for lang in python javascript; do
  src="${PIE_HOME:-$HOME/.pie}/languages/$lang.wasm"
  if [ -f "$src" ]; then cp "$src" tests/browser/languages/; else echo "languages: no $src (*/inferlet/language/build.sh builds it); $lang twins will not run"; fi
done
