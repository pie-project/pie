#!/bin/bash
# Regenerate the golden for tests/lora_stage_parity.rs — gate-lora slice A.
#
# The prepare oracle's construction: the whole real llama_like.cpp over the
# SHARED llama_like_prepare stub tree, --gc-sections, only the staging path
# driven. Replaced: the DeviceBuffer allocator (deterministic bases — the
# fingerprint mixes the arena address, so the VALUE must be computable on
# both sides), the bf16 cast, and the slab upload. `tensor_recorder.cpp`
# supplies `DeviceTensor::view` for the fixture's workspace buffers.
#
# `PIE_LORA_GROUPED` is cached in a function-local static, so its axis is
# swept across PROCESSES: the second run re-drives the grouping cases with
# the lowering off.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../../../.." && pwd)"
SRC="$ROOT/crates/driver-cuda/csrc/src"
KSRC="$ROOT/crates/kernels-cuda/csrc/src"
DRV="$ROOT/crates/driver/include"
ABI="$ROOT/crates/driver-abi/include"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

mkdir -p "$WORK/model/llama_like"
cp "$SRC/model/llama_like/llama_like.cpp" \
   "$SRC/model/llama_like/llama_like.hpp" "$WORK/model/llama_like/"
cp "$SRC/model/config.hpp" "$WORK/model/"
cp -r "$HERE/../llama_like_prepare/stub/." "$WORK/"
cp "$HERE/../kv_cache/stub/tensor_recorder.cpp" "$WORK/"

g++ -std=c++20 -O1 -Wall -Wno-unused-parameter \
    -ffunction-sections -fdata-sections \
    -I"$WORK" -I"$SRC" -I"$KSRC" -I"$DRV" -I"$ABI" \
    -c -o "$WORK/llama_like.o" "$WORK/model/llama_like/llama_like.cpp"
for tu in tensor_recorder cuda_recorder; do
    g++ -std=c++20 -O1 -Wall -Wno-unused-parameter \
        -ffunction-sections -fdata-sections \
        -I"$WORK" -I"$SRC" -I"$KSRC" -I"$DRV" -I"$ABI" \
        -c -o "$WORK/$tu.o" "$WORK/$tu.cpp"
done
g++ -std=c++20 -O1 -Wall -Wextra -Wno-unused-parameter \
    -ffunction-sections -fdata-sections \
    -I"$WORK" -I"$SRC" -I"$KSRC" -I"$DRV" -I"$ABI" \
    -c -o "$WORK/oracle.o" "$HERE/oracle.cpp"
g++ -Wl,--gc-sections -o "$WORK/oracle" "$WORK"/*.o

OUT="$WORK/out.txt"
env -u PIE_LORA_GROUPED -u PIE_LORA_FIRE_TRACE "$WORK/oracle" > "$OUT"
PIE_LORA_GROUPED=0 "$WORK/oracle" 2>/dev/null \
    | sed 's/^/off:/' >> "$OUT"

if [[ -n "${LS_ORACLE_OUT:-}" ]]; then
    cp "$OUT" "$LS_ORACLE_OUT"
    echo "transcript written to $LS_ORACLE_OUT" >&2
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
