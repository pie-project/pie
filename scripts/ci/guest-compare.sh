#!/usr/bin/env bash
# The guest steps' results byte for byte across three forms, on one GPU:
#   whole      - the steps as emitted: what native serving creates
#   compacted  - PIE_TEST_GUEST_COMPACT=1: each step compacted by naga, as the tab does
#   rolled     - PIE_TEST_GUEST_COMPACT=1 PIE_TEST_GUEST_ROLLED=1: also the pivot
#                finish's block rolled, the tab's form whole
# Native runs whole; compaction and the roll are the tab's (cfg wasm32), so a
# native run takes them through the test-limits knobs, which the engine-wgpu
# tests build with. One tree, one target.
# The engine-wgpu suites that run guest programs write what each pass reads
# back under PIE_SUITE_GUEST_DUMP (engine-gpu/suite/runtime/mod.rs); one test
# thread keeps the passes in one order. Needs PIE_SLANGC (slangc 2026.17).
#
#   bash scripts/ci/guest-compare.sh <work dir>
set -euo pipefail
work=${1:?a work dir}
repo=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
rev=${REV:-origin/sc/gpu-pool-seg-forms-rolled}
filters="runtime::outputs runtime::guest_state runtime::roles runtime::slots runtime::scores runtime::hosted runtime::one_owner"
mkdir -p "$work"
git -C "$repo" fetch -q origin
[ -d "$work/tree" ] || git -C "$repo" worktree add -q --detach "$work/tree" "$rev"
run() { # <dump name> [env...]
  local name=$1; shift
  (cd "$work/tree" && env "$@" PIE_SUITE_GUEST_DUMP="$work/dump/$name" CARGO_TARGET_DIR="$work/target" \
    cargo test -q -p engine-wgpu --features wgpu --test engine_wgpu -- --test-threads=1 $filters) \
    > "$work/$name.log" 2>&1 || true
  echo "$name: $(grep 'test result' "$work/$name.log" || echo 'no result; see the log')"
}
rm -rf "$work/dump"
run whole
run compacted PIE_TEST_GUEST_COMPACT=1
run rolled PIE_TEST_GUEST_COMPACT=1 PIE_TEST_GUEST_ROLLED=1
for name in whole compacted rolled; do echo "$name: $(ls "$work/dump/$name" | wc -l) dumps"; done
diff -r "$work/dump/whole" "$work/dump/compacted" && echo "whole == compacted, byte for byte"
diff -r "$work/dump/whole" "$work/dump/rolled" && echo "whole == rolled, byte for byte"
