#!/bin/bash
# Regenerate the golden for tests/kv_cache_live_parity.rs — gate-kvcache-live.
#
# Builds exactly the layout oracle's tree (the stubs are SHARED from
# ../kv_cache/stub — one recorder, one elastic stub, one seed stub) and
# drives the live object instead of the allocation sweep: layer views,
# accessors, page buffers, envelope seeding, elastic forwarding.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../../../.." && pwd)"
SRC="$ROOT/crates/driver-cuda/csrc/src"
KSRC="$ROOT/crates/kernels-cuda/csrc/src"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

mkdir -p "$WORK/store" "$WORK/attn"
cp "$SRC/store/kv_cache.cpp" "$SRC/store/kv_cache.hpp" \
   "$SRC/store/kv_cache_format.cpp" "$SRC/store/kv_cache_format.hpp" "$WORK/store/"
cp "$KSRC/tensor.hpp" "$WORK/"
cp "$KSRC/attn/kv_cache_view.hpp" "$WORK/attn/"
cp -r "$HERE/../kv_cache/stub/." "$WORK/"

g++ -std=c++20 -O1 -Wall -Wextra -Wno-unused-parameter \
    -I"$WORK" -I"$WORK/store" \
    -o "$WORK/oracle" \
    "$HERE/oracle.cpp" "$WORK/store/kv_cache.cpp" \
    "$WORK/store/kv_cache_format.cpp" "$WORK/tensor_recorder.cpp"

OUT="$WORK/out.txt"
"$WORK/oracle" > "$OUT"

if [[ -n "${KVL_ORACLE_OUT:-}" ]]; then
    cp "$OUT" "$KVL_ORACLE_OUT"
    echo "transcript written to $KVL_ORACLE_OUT" >&2
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
