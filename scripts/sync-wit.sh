#!/usr/bin/env bash
#
# One-way sync of the canonical WIT interface into the vendored SDK copy.
#
# Source of truth: crates/inferlet-api/wit/  (the single `pie:inferlet` package —
# world.wit + the sibling interface files + a vendored deps/ tree of the
# wasi 0.3 packages the wasmtime host implements).
#
# Vendored copy (DO NOT hand-edit):
#   - sdk/inferlet/tools/bakery/src/bakery/wit/
#
# It is the LAST one. The Rust guest used to need a second: `inferlet` held the
# `wit_bindgen::generate!` site, and `generate!`'s `path` is a filesystem path
# resolved at macro expansion, so a published `.crate` could only reach a `wit/`
# inside its own package directory. Moving the generator to `inferlet-api` —
# the crate that owns the WIT — made that path `"wit"` and the bindings a
# cargo dependency, so the copy had nothing left to do. bakery's copy stays
# because bakery is a Python package: it can vendor a directory, not link a rlib.
#
# The copy is a full mirror of the package: the interface *.wit files + world.wit
# are copied directly, and deps/ (the vendored wasi 0.3 wit) is copied verbatim.
# Manual three-way editing of this copy has regressed twice
# (see commit 94043eb1); run this instead.
#
# Usage:
#   scripts/sync-wit.sh          # sync the copy in place
#   scripts/sync-wit.sh --check  # verify the copy is in sync (CI); non-zero on drift
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$ROOT/crates/inferlet-api/wit"

# Vendored copies: each entry is the `wit/` directory of a consumer. One left;
# kept as a list because that is what makes adding the next one a one-line
# change rather than a rewrite.
COPIES=(
  "$ROOT/sdk/inferlet/tools/bakery/src/bakery/wit"
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
    echo "error: vendored WIT copies are out of sync with crates/inferlet-api/wit." >&2
    echo "Run scripts/sync-wit.sh and commit the result." >&2
    git -C "$ROOT" --no-pager diff --stat -- "${COPIES[@]}" >&2
    exit 1
  fi
  echo "WIT copies are in sync."
fi
