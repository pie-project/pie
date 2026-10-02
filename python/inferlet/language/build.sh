#!/usr/bin/env bash
# Build the Python language component: the one wasm every Python inferlet
# runs in. It bundles CPython, the `inferlet` Python package and the WIT
# world's bindings, snapshotted after import so a program starts warm.
#
#   python/inferlet/language/build.sh [OUT]
#
#   OUT               defaults to $PIE_HOME/languages/python.wasm
#                     (~/.pie/languages/python.wasm), where `pie` looks.
#   COMPONENTIZE_PY   the componentize-py to run (default: the one on PATH).
#                     componentize-py >= 0.25 is required: the world is
#                     component-model async.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd "$here/../../.." && pwd)"
out="${1:-${PIE_HOME:-$HOME/.pie}/languages/python.wasm}"
cpy="${COMPONENTIZE_PY:-componentize-py}"

if ! command -v "$cpy" >/dev/null 2>&1; then
  echo "componentize-py not found; \`uv tool install componentize-py\` (>= 0.25) installs it" >&2
  exit 1
fi

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
cp "$here/app.py" "$work/app.py"
cp -R "$root/python/inferlet/src/inferlet" "$work/inferlet"

case "$out" in
  /*) ;;
  *) out="$PWD/$out" ;;
esac
mkdir -p "$(dirname "$out")"

echo "== componentize ($("$cpy" --version))"
(cd "$work" && "$cpy" -d "$root/crates/inferlet/wit" -w inferlet componentize -p "$work" -o "$out" app)
ls -la "$out"
