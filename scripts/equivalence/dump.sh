#!/usr/bin/env bash
# dump.sh OUT [filter...]: everything an IR change can move, from the tree
# this is run in (the repository root), into OUT.
#
#   OUT/shard.txt            every deployment's shard plan at tp 1/2/4/8
#   OUT/xla/                 every XLA fire's StableHLO (dry coverage)
#   OUT/cerebras-phase/      every Cerebras fire's CSL, phase mode
#   OUT/cerebras-table/      table mode
#   OUT/cerebras-whole/      table mode, whole fire, pools kept
#   OUT/<name>.txt           each coverage run's lines, timings stripped
#
# The coverage examples share a binary name, so each is built and copied out
# first. The dumps are tens of GB in all: give OUT room, or pass a filter.
set -euo pipefail
out=$1
shift
mkdir -p "$out/bin"
target=${CARGO_TARGET_DIR:-target}
cargo build --release -q -p pie-poem-compiler --example shard_dump
cp "$target"/release/examples/shard_dump "$out/bin/shard_dump"
cargo build --release -q -p engine-xla --example coverage
cp "$target"/release/examples/coverage "$out/bin/xla"
cargo build --release -q -p engine-cerebras --example coverage
cp "$target"/release/examples/coverage "$out/bin/cerebras"

run() {
    local name=$1 exe=$2
    shift 2
    rm -rf "${out:?}/$name"
    mkdir -p "$out/$name"
    env "$@" "$out/bin/$exe" --dump "$out/$name" "${FILTERS[@]}" |
        sed -E 's/ +[0-9.]+s  / /' >"$out/$name.txt"
}

FILTERS=("$@")
"$out/bin/shard_dump" "$out/shard.txt" &
run xla xla &
run cerebras-phase cerebras PIE_CEREBRAS_TABLE=0 &
run cerebras-table cerebras PIE_CEREBRAS_TABLE=1 &
run cerebras-whole cerebras PIE_CEREBRAS_TABLE=1 PIE_CEREBRAS_WHOLE_FIRE=1 PIE_CEREBRAS_KEEP_POOLS=1 &
wait
tail -n1 "$out"/*.txt
