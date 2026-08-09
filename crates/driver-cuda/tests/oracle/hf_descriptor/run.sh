#!/bin/bash
# Regenerate the golden for tests/hf_descriptor_parity.rs — gate-hf-config's
# read side.
#
# The pipeline is `check_descriptor.sh`'s, pointed at a hash:
#
#   corpus config.json --[Rust normalize]--> pie.model/1
#                      --[C++ descriptor.cpp read]--> 134-field dump
#
# over all 56 corpus configs, flattened to type-tagged rows (floats as f32
# BIT PATTERNS — nlohmann's shortest-repr and Rust's `{}` disagree on text,
# not on value), plus three refusal cases: a foreign version, a missing key,
# an unknown rope scaling. The Rust side replays the same pipeline with its
# own reader and must produce the identical transcript.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../../../.." && pwd)"
DUMPDIR="$ROOT/crates/driver-cuda/csrc/tests/hf_config_dump"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

# build.sh probes for a FetchContent copy of nlohmann; point it directly at
# the one the driver's own build tree carries so the probe cannot miss.
if [[ -z "${NLOHMANN_INCLUDE:-}" ]]; then
    hpp="$(find "$ROOT/target" -path '*json-src/single_include/nlohmann/json.hpp' -print -quit 2>/dev/null || true)"
    if [[ -n "$hpp" ]]; then
        export NLOHMANN_INCLUDE="$(dirname "$(dirname "$hpp")")"
    fi
fi
"$DUMPDIR/build.sh" --descriptor "$WORK/dump_from_descriptor" >/dev/null
(cd "$ROOT" && cargo build -q -p model --features config --bin descriptor)
DESCRIPTOR="$ROOT/target/debug/descriptor"

OUT="$WORK/out.txt"
: > "$OUT"

flatten() {
    python3 - "$1" "$2" <<'PY'
import json, struct, sys

name, path = sys.argv[1], sys.argv[2]
doc = json.load(open(path))

def tag(v):
    if v is None: return "null"
    if isinstance(v, bool): return f"b:{int(v)}"
    if isinstance(v, int): return f"i:{v}"
    if isinstance(v, float):
        return f"f:{struct.unpack('<I', struct.pack('<f', v))[0]}"
    return f"s:{v}"

def walk(prefix, v, out):
    if isinstance(v, dict):
        for k in sorted(v):
            walk(f"{prefix}.{k}" if prefix else k, v[k], out)
    elif isinstance(v, list):
        out.append((f"{prefix}.len", f"i:{len(v)}"))
        for i, e in enumerate(v):
            walk(f"{prefix}.{i}", e, out)
    else:
        out.append((prefix, tag(v)))

rows = []
walk("", doc, rows)
for p, t in rows:
    print(f"{name}\x1f{p}\x1f{t}")
PY
}

for config in "$DUMPDIR"/corpus/*.json; do
    name="$(basename "$config")"
    "$DESCRIPTOR" "$config" > "$WORK/desc.json"
    "$WORK/dump_from_descriptor" "$WORK/desc.json" > "$WORK/dump.json"
    flatten "$name" "$WORK/dump.json" >> "$OUT"
done

# The refusal cases, built from the first corpus descriptor.
first="$(ls "$DUMPDIR"/corpus/*.json | head -1)"
"$DESCRIPTOR" "$first" > "$WORK/base.json"
python3 - "$WORK" <<'PY'
import json, sys
work = sys.argv[1]
base = json.load(open(f"{work}/base.json"))
v2 = dict(base); v2["version"] = "pie.model/2"
json.dump(v2, open(f"{work}/refuse-version.json", "w"))
mk = dict(base); del mk["hidden_size"]
json.dump(mk, open(f"{work}/refuse-missing.json", "w"))
rs = dict(base); rs["rope_scaling_kind"] = "yarn_v3"
json.dump(rs, open(f"{work}/refuse-rope.json", "w"))
PY
for case in refuse-version refuse-missing refuse-rope; do
    "$WORK/dump_from_descriptor" "$WORK/$case.json" > "$WORK/refusal.json" || true
    python3 - "$case" "$WORK/refusal.json" <<'PY' >> "$OUT"
import json, sys
doc = json.load(open(sys.argv[2]))
print(f"{sys.argv[1]}\x1ferror\x1f{doc['error']}")
PY
done

if [[ -n "${HFD_ORACLE_OUT:-}" ]]; then
    cp "$OUT" "$HFD_ORACLE_OUT"
    echo "transcript written to $HFD_ORACLE_OUT" >&2
fi

python3 - "$OUT" <<'PY'
import sys
data = open(sys.argv[1], "rb").read()
h = 0xcbf29ce484222325
for b in data:
    h = ((h ^ b) * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF
print(f"GOLDEN_FNV1A64 = 0x{h:016x}")
print(f"GOLDEN_ROWS    = {sum(1 for _ in open(sys.argv[1]))}")
PY
