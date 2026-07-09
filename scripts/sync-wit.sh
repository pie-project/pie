#!/usr/bin/env bash
#
# One-way sync of the canonical WIT interface into the vendored SDK copies.
#
# Source of truth: interface/inferlet/  (the single `pie:inferlet` package —
# world.wit + the sibling interface files + a vendored deps/ tree of the
# wasi 0.3 packages the wasmtime host implements).
#
# Vendored copies (DO NOT hand-edit):
#   - sdk/rust/inferlet/wit/
#   - sdk/tools/bakery/src/bakery/wit/
#
# Each copy is a full mirror of the package: the interface *.wit files + world.wit
# are copied directly, and deps/ (the vendored wasi 0.3 wit) is copied verbatim.
# Manual three-way editing of these copies has regressed twice
# (see commit 94043eb1); run this instead.
#
# Usage:
#   scripts/sync-wit.sh          # sync the copies in place
#   scripts/sync-wit.sh --check  # verify copies are in sync (CI); non-zero on drift
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$ROOT/interface/inferlet"

# Vendored copies: each entry is the `wit/` directory of a consumer.
COPIES=(
  "$ROOT/sdk/rust/inferlet/wit"
  "$ROOT/sdk/tools/bakery/src/bakery/wit"
)

sync_one() {
  local wit_dir="$1"
  rm -rf "$wit_dir"
  mkdir -p "$wit_dir"
  # The package source: every top-level interface file + world.wit + the
  # vendored wasi 0.3 dependency tree.
  cp "$SRC"/*.wit "$wit_dir"/
  cp -r "$SRC"/deps "$wit_dir"/deps
}

for wit_dir in "${COPIES[@]}"; do
  sync_one "$wit_dir"
done

if [[ "${1:-}" == "--check" ]]; then
  if ! git -C "$ROOT" diff --quiet -- "${COPIES[@]}"; then
    echo "error: vendored WIT copies are out of sync with interface/inferlet." >&2
    echo "Run scripts/sync-wit.sh and commit the result." >&2
    git -C "$ROOT" --no-pager diff --stat -- "${COPIES[@]}" >&2
    exit 1
  fi
  echo "WIT copies are in sync."
fi
