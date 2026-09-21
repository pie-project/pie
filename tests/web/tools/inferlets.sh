#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
(cd tests/inferlets && cargo build -q --release --target wasm32-wasip2)
mkdir -p tests/web/inferlets
for name in $(python3 -c 'import json; print(" ".join(sorted({e["inferlet"] for e in json.load(open("tests/web/matrix.json"))})))'); do
  case "$name" in
    *-js|*-py)
      twin="tests/inferlets/$name/target/${name//-/_}.wasm"
      if [ -f "$twin" ]; then cp "$twin" tests/web/inferlets/; else echo "inferlets: $name not built (bakery), skipped"; continue; fi ;;
    *) cp "tests/inferlets/target/wasm32-wasip2/release/${name//-/_}.wasm" tests/web/inferlets/ ;;
  esac
  cp "tests/inferlets/$name/Pie.toml" "tests/web/inferlets/$name.Pie.toml"
done
