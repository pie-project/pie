#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
(cd examples && cargo build -q --release --target wasm32-wasip2)
mkdir -p tests/browser/inferlets
for name in $(python3 -c 'import json; print(" ".join(sorted({e["inferlet"] for e in json.load(open("tests/browser/matrix.json"))})))'); do
  case "$name" in
    *-js) cp "examples/$name/index.js" "tests/browser/inferlets/${name//-/_}.js" ;;
    *-py) cp "examples/$name/main.py" "tests/browser/inferlets/${name//-/_}.py" ;;
    *) cp "examples/target/wasm32-wasip2/release/${name//-/_}.wasm" tests/browser/inferlets/ ;;
  esac
  cp "examples/$name/Pie.toml" "tests/browser/inferlets/$name.Pie.toml"
done
# The script twins run under their language component, which
# the page installs from bytes: copy the ones the examples workspace built.
mkdir -p tests/browser/languages
for lang in python javascript; do
  src="${PIE_HOME:-$HOME/.pie}/languages/$lang.wasm"
  if [ -f "$src" ]; then cp "$src" tests/browser/languages/; else echo "languages: no $src (*/inferlet/language/build.sh builds it); $lang twins will not run"; fi
done
