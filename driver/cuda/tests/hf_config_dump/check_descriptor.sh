#!/usr/bin/env bash
# Round-trips every corpus config through the *new* path and diffs against the
# old one.
#
#   config.json --[C++ parse_hf_config]-------------------> golden
#   config.json --[Rust normalize]--> pie.model/1 --[C++ read]--> must equal it
#
# This is what says `parse_hf_config` can be deleted: both halves of the
# replacement, checked end to end against the thing being replaced, on 28 real
# configs and 27 synthetic ones.
#
#   ./check_descriptor.sh
#
# Needs `cargo build -p pie-model-config --bin descriptor` (done automatically)
# and a C++ compiler; see build.sh for how nlohmann is found.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "$here/../../../.." && pwd)"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

"$here/build.sh" "$work/dump_hf_config" >/dev/null
"$here/build.sh" --descriptor "$work/dump_from_descriptor" >/dev/null
(cd "$repo" && cargo build -q -p pie-model-config --bin descriptor)
descriptor="$repo/target/debug/descriptor"

pass=0; fail=0
for config in "$here"/corpus/*.json; do
    name="$(basename "$config" .json)"
    if ! "$descriptor" "$config" > "$work/$name.descriptor.json" 2>"$work/$name.err"; then
        echo "FAIL $name: the normalizer refused it"; sed 's/^/    /' "$work/$name.err"
        fail=$((fail+1)); continue
    fi
    "$work/dump_from_descriptor" "$work/$name.descriptor.json" > "$work/$name.actual" || true
    if diff -q "$work/$name.actual" "$here/golden/$name.json" >/dev/null; then
        pass=$((pass+1))
    else
        echo "FAIL $name"
        diff "$here/golden/$name.json" "$work/$name.actual" | head -12 | sed 's/^/    /'
        fail=$((fail+1))
    fi
done
echo "$pass matched, $fail differed"
[ "$fail" -eq 0 ]
