#!/bin/bash
# Regenerate the golden for tests/llama_like_prepare_parity.rs — slice B of
# gate-plan-state.
#
# Same construction as ../llama_like_cfg: the whole real llama_like.cpp,
# -ffunction-sections + --gc-sections. Slice B additionally LINKS AND RUNS
# the real attention_workspace.cpp (the prepare hook plans into real
# workspaces) and the real kv_cache.cpp/kv_cache_format.cpp (it reads the
# cache's format/page geometry), over silent CUDA and the shared tensor
# recorder. The flashinfer planner boundary is the recorder surface.
#
# Three env gates are cached in function-local statics, so their axes are
# swept across PROCESSES: default, PIE_SPATIAL_MASK=0, PIE_MIXED_MID=0,
# PIE_PREFILL_GRAPH_PLAN=1.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../../../.." && pwd)"
SRC="$ROOT/crates/driver-cuda/csrc/src"
KSRC="$ROOT/crates/kernels-cuda/csrc/src"
DRV="$ROOT/crates/driver/include"
ABI="$ROOT/crates/driver-abi/include"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

mkdir -p "$WORK/model/llama_like" "$WORK/store"
cp "$SRC/model/llama_like/llama_like.cpp" \
   "$SRC/model/llama_like/llama_like.hpp" "$WORK/model/llama_like/"
cp "$SRC/attention_workspace.cpp" "$SRC/attention_workspace.hpp" "$WORK/"
cp "$SRC/store/kv_cache.cpp" "$SRC/store/kv_cache.hpp" \
   "$SRC/store/kv_cache_format.cpp" "$SRC/store/kv_cache_format.hpp" \
   "$WORK/store/"
cp "$SRC/model/config.hpp" "$WORK/model/"
cp -r "$HERE/stub/." "$WORK/"
cp "$HERE/../kv_cache/stub/tensor_recorder.cpp" "$WORK/"

g++ -std=c++20 -O1 -Wall -Wno-unused-parameter \
    -ffunction-sections -fdata-sections \
    -I"$WORK" -I"$WORK/store" -I"$SRC" -I"$KSRC" -I"$DRV" -I"$ABI" \
    -c -o "$WORK/llama_like.o" "$WORK/model/llama_like/llama_like.cpp"
for tu in attention_workspace store/kv_cache store/kv_cache_format \
          tensor_recorder cuda_recorder; do
    g++ -std=c++20 -O1 -Wall -Wno-unused-parameter \
        -ffunction-sections -fdata-sections \
        -I"$WORK" -I"$WORK/store" -I"$SRC" -I"$KSRC" -I"$DRV" -I"$ABI" \
        -c -o "$WORK/$(basename "$tu").o" "$WORK/$tu.cpp"
done
g++ -std=c++20 -O1 -Wall -Wextra -Wno-unused-parameter \
    -ffunction-sections -fdata-sections \
    -I"$WORK" -I"$WORK/store" -I"$SRC" -I"$KSRC" -I"$DRV" -I"$ABI" \
    -c -o "$WORK/oracle.o" "$HERE/oracle.cpp"
g++ -Wl,--gc-sections -o "$WORK/oracle" "$WORK"/*.o

OUT="$WORK/out.txt"
env -u PIE_SPATIAL_MASK -u PIE_MIXED_MID -u PIE_PREFILL_GRAPH_PLAN \
    -u PIE_REGION_TRACE -u PIE_CUDA_KV_ENVELOPES \
    "$WORK/oracle" main > "$OUT"
env -u PIE_MIXED_MID -u PIE_PREFILL_GRAPH_PLAN -u PIE_REGION_TRACE \
    -u PIE_CUDA_KV_ENVELOPES \
    PIE_SPATIAL_MASK=0 "$WORK/oracle" spatial-off >> "$OUT"
env -u PIE_SPATIAL_MASK -u PIE_PREFILL_GRAPH_PLAN -u PIE_REGION_TRACE \
    -u PIE_CUDA_KV_ENVELOPES \
    PIE_MIXED_MID=0 "$WORK/oracle" mid-off >> "$OUT"
env -u PIE_SPATIAL_MASK -u PIE_MIXED_MID -u PIE_REGION_TRACE \
    -u PIE_CUDA_KV_ENVELOPES \
    PIE_PREFILL_GRAPH_PLAN=1 "$WORK/oracle" graph-plan-on >> "$OUT"

if [[ -n "${LLP_ORACLE_OUT:-}" ]]; then
    cp "$OUT" "$LLP_ORACLE_OUT"
    echo "transcript written to $LLP_ORACLE_OUT" >&2
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
