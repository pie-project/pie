#!/usr/bin/env bash
# Builds every inferlet in tests/inferlets for wasm32-wasip2 and puts the ones
# web/site/matrix.json names beside the page (web/site/inferlets/).
set -euo pipefail
cd "$(dirname "$0")/../.."
(cd tests/inferlets && cargo build -q --release --target wasm32-wasip2)
mkdir -p web/site/inferlets
for name in $(python3 -c 'import json; print(" ".join(sorted({e["inferlet"] for e in json.load(open("web/site/matrix.json"))})))'); do
  cp "tests/inferlets/target/wasm32-wasip2/release/${name//-/_}.wasm" web/site/inferlets/
  cp "tests/inferlets/$name/Pie.toml" "web/site/inferlets/$name.Pie.toml"
done
