#!/bin/bash
# Regenerate the golden for tests/qwen35_la_ws_parity.rs —
# gate-linear-attn-ws.
#
# Same construction as the llama_like oracles: the whole real
# qwen3_5_forward.cpp over the SHARED llama_like_prepare stub tree,
# -ffunction-sections + --gc-sections, only the allocator driven. The one
# extra replaced implementation is `allocate_device_memory` (in
# oracle.cpp), which records ordinal and bytes.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../../../.." && pwd)"
SRC="$ROOT/crates/driver-cuda/csrc/src"
KSRC="$ROOT/crates/kernels-cuda/csrc/src"
DRV="$ROOT/crates/driver/include"
ABI="$ROOT/crates/driver-abi/include"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

mkdir -p "$WORK/model/qwen3_5"
cp "$SRC/model/qwen3_5/qwen3_5_forward.cpp" \
   "$SRC/model/qwen3_5/qwen3_5_forward.hpp" "$WORK/model/qwen3_5/"
cp -r "$HERE/../llama_like_prepare/stub/." "$WORK/"

g++ -std=c++20 -O1 -Wall -Wno-unused-parameter \
    -ffunction-sections -fdata-sections \
    -I"$WORK" -I"$SRC" -I"$KSRC" -I"$DRV" -I"$ABI" \
    -c -o "$WORK/fwd.o" "$WORK/model/qwen3_5/qwen3_5_forward.cpp"
g++ -std=c++20 -O1 -Wall -Wextra -Wno-unused-parameter \
    -ffunction-sections -fdata-sections \
    -I"$WORK" -I"$SRC" -I"$KSRC" -I"$DRV" -I"$ABI" \
    -c -o "$WORK/oracle.o" "$HERE/oracle.cpp"
# DeviceBuffer's inlined pinned-mirror teardown reaches the silent CUDA
# recorders even on the allocation-only path.
g++ -std=c++20 -O1 -Wall -Wno-unused-parameter \
    -ffunction-sections -fdata-sections \
    -I"$WORK" \
    -c -o "$WORK/cuda_recorder.o" "$WORK/cuda_recorder.cpp"
g++ -Wl,--gc-sections -o "$WORK/oracle" "$WORK/oracle.o" "$WORK/fwd.o" \
    "$WORK/cuda_recorder.o"

OUT="$WORK/out.txt"
"$WORK/oracle" > "$OUT"

if [[ -n "${Q35_ORACLE_OUT:-}" ]]; then
    cp "$OUT" "$Q35_ORACLE_OUT"
    echo "transcript written to $Q35_ORACLE_OUT" >&2
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
