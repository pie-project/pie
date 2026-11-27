#!/usr/bin/env bash
# The guest steps' results byte for byte across three forms, on one GPU:
#   compacted  - sc/gpu-pool-seg-forms (each step compacted by naga)
#   whole      - the same tree with PIE_TEST_GUEST_WHOLE=1 (steps as emitted)
#   rolled     - sc/gpu-pool-seg-forms-rolled (the pivot finish's block rolled)
# The engine-wgpu suites that run guest programs write what each pass reads
# back under PIE_SUITE_GUEST_DUMP (engine-gpu/suite/runtime/mod.rs); one test
# thread keeps the passes in one order. Needs PIE_SLANGC (slangc 2026.17).
#
#   bash scripts/ci/guest-compare.sh <work dir>
set -euo pipefail
work=${1:?a work dir}
repo=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
base=${BASE_REV:-origin/sc/gpu-pool-seg-forms}
rolled=${ROLLED_REV:-origin/sc/gpu-pool-seg-forms-rolled}
filters="runtime::outputs runtime::guest_state runtime::roles runtime::slots runtime::scores runtime::hosted runtime::one_owner"
mkdir -p "$work"
git -C "$repo" fetch -q origin
for side in base rolled; do
  rev=$base; [ $side = rolled ] && rev=$rolled
  [ -d "$work/$side" ] || git -C "$repo" worktree add -q --detach "$work/$side" "$rev"
done
run() { # <tree> <dump name> [env...]
  local tree=$1 name=$2; shift 2
  (cd "$work/$tree" && env "$@" PIE_SUITE_GUEST_DUMP="$work/dump/$name" CARGO_TARGET_DIR="$work/target-$tree" \
    cargo test -q -p engine-wgpu --features wgpu --test engine_wgpu -- --test-threads=1 $filters) \
    > "$work/$name.log" 2>&1 || true
  echo "$name: $(grep 'test result' "$work/$name.log" || echo 'no result; see the log')"
}
rm -rf "$work/dump"
run base compacted
run base whole PIE_TEST_GUEST_WHOLE=1
run rolled rolled
for name in compacted whole rolled; do echo "$name: $(ls "$work/dump/$name" | wc -l) dumps"; done
diff -r "$work/dump/compacted" "$work/dump/whole" && echo "compacted == whole, byte for byte"
diff -r "$work/dump/compacted" "$work/dump/rolled" && echo "compacted == rolled, byte for byte"
