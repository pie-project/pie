#!/usr/bin/env bash
# Enforces pie-engine's major_refactor.md layer order:
#
#   driver (L0) -> store (L1) -> scheduler (L2) -> pipeline (L3)
#                                                 -> inferlet/server (L4)
#
# Each layer may only import layers strictly below it (plus external/leaf
# crates such as `pie_ptir`, `wasmtime`, `tokio`, ...). There is NO accepted
# exception at L0/L1/L2/L3 — the only documented upward exception anywhere
# in the crate is `inferlet::host::session` -> `server` (guest-to-client I/O
# facade, see `inferlet/host.rs`'s module doc), which is L4 and outside
# the scope of this script. This script fails if any of L0..L3 contains a
# reference to `crate::<forbidden-layer>`, including fully-qualified inline
# calls, and separately fails if the pre-refactor `ptir`/`api`/`inference`
# module paths have reappeared anywhere in the crate.
set -uo pipefail

cd "$(dirname "$0")/.." || exit 1

ENGINE_SRC="runtime/engine/src"
fail=0

# check_layer <layer-name> <path...> -- <forbidden-module...>
check_layer() {
  local layer="$1"
  shift
  local paths=()
  while [ "$1" != "--" ]; do
    paths+=("$1")
    shift
  done
  shift # drop the --
  local forbidden=("$@")

  local pattern
  pattern=$(printf '%s|' "${forbidden[@]}")
  pattern="crate::(${pattern%|})(::|;|,|\\s+as\\s+)"

  local matches
  matches=$(grep -rnE "$pattern" "${paths[@]}" 2>/dev/null || true)
  if [ -n "$matches" ]; then
    echo "::error::[$layer] forbidden upward import(s) found:"
    echo "$matches"
    fail=1
  else
    echo "[$layer] OK — no upward imports of {${forbidden[*]}}"
  fi
}

check_layer "L0 driver" \
  "$ENGINE_SRC/driver.rs" "$ENGINE_SRC/driver" -- \
  store scheduler pipeline inferlet server inference ptir api

check_layer "L1 store" \
  "$ENGINE_SRC/store.rs" "$ENGINE_SRC/store" -- \
  scheduler pipeline inferlet server inference ptir api

check_layer "L2 scheduler" \
  "$ENGINE_SRC/scheduler.rs" "$ENGINE_SRC/scheduler" -- \
  pipeline inferlet server ptir api inference

check_layer "L3 pipeline" \
  "$ENGINE_SRC/pipeline.rs" "$ENGINE_SRC/pipeline" -- \
  inferlet server api inference ptir

# Legacy module paths must never reappear, anywhere in the crate. This is a
# bare (non-`use`-scoped) check on purpose: even a *doc comment* mentioning
# `crate::ptir`/`crate::api`/`crate::inference` would mean a stale reference
# to a module that no longer exists, so there is no legitimate false positive
# to guard against here (unlike the `pie_ptir` IR crate import, which is a
# different token and is not matched by `crate::ptir`).
echo
echo "Legacy module path check (crate::{ptir,api,inference})"
legacy_matches=$(grep -rnE "crate::(ptir|api|inference)(::|;|,|\s+as\s+)?\b" "$ENGINE_SRC" 2>/dev/null || true)
if [ -n "$legacy_matches" ]; then
  echo "::error::Legacy crate::{ptir,api,inference} path reference(s) found:"
  echo "$legacy_matches"
  fail=1
else
  echo "OK — no crate::{ptir,api,inference} references"
fi

echo
echo "Legacy module directory/file check (src/{ptir,api,inference,inference.rs})"
for legacy in ptir api inference inference.rs; do
  if [ -e "$ENGINE_SRC/$legacy" ]; then
    echo "::error::Legacy module path $ENGINE_SRC/$legacy still exists"
    fail=1
  fi
done
if [ "$fail" -eq 0 ]; then
  echo "OK — no legacy ptir/api/inference module paths"
fi

if [ "$fail" -ne 0 ]; then
  echo
  echo "::error::pie-engine layering gate FAILED — see violations above."
  exit 1
fi

echo
echo "pie-engine layering gate PASSED."
exit 0
