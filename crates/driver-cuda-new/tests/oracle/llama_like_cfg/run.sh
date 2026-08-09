#!/bin/bash
# Regenerate the golden for tests/llama_like_cfg_parity.rs — slice A of
# gate-plan-state.
#
# Compiles the REAL model/llama_like/llama_like.cpp — the whole 3.2k-line TU,
# forward body and all — and drives only its host-pure surface: the config
# and plan-state defaults, the rope mapping, the fused-post env gate, and the
# three graph-layout functions. The build uses -ffunction-sections and the
# link uses --gc-sections, so the undriven 95% of the TU is discarded AFTER
# the compiler has type-checked it and BEFORE the linker demands definitions
# for what it calls. Any function a driven path actually reaches must be
# defined — the linker is the stub inventory's auditor.
#
# The only replaced implementations are the flashinfer plan-cache entry
# points (opaque types in the real header) and the CUDA/NCCL surface, which
# is declaration-only: nothing that would touch a device is ever linked.
#
# `decode_fused_post_enabled` caches its env read in a function-local static,
# so the env axis is swept across PROCESSES: one oracle run per value, rows
# concatenated into one transcript.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../../../.." && pwd)"
SRC="$ROOT/crates/driver-cuda/csrc/src"
KSRC="$ROOT/crates/kernels-cuda/csrc/src"
DRV="$ROOT/crates/driver/include"
ABI="$ROOT/crates/driver-abi/include"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

# The files under test are COPIED beside the stubs: quote-includes resolve
# against the including file's own directory first, so this is the only way
# `llama_like.hpp`'s `#include "distributed.hpp"` can land on the stub while
# every other header stays the shipping one.
mkdir -p "$WORK/model/llama_like"
cp "$SRC/model/llama_like/llama_like.cpp" \
   "$SRC/model/llama_like/llama_like.hpp" "$WORK/model/llama_like/"
cp -r "$HERE/stub/." "$WORK/"
cp "$HERE/../launch_abi/stub/cublas_v2.h" "$WORK/"

g++ -std=c++20 -O1 -Wall -Wno-unused-parameter \
    -ffunction-sections -fdata-sections \
    -I"$WORK" -I"$SRC" -I"$KSRC" -I"$DRV" -I"$ABI" \
    -c -o "$WORK/llama_like.o" "$WORK/model/llama_like/llama_like.cpp"
g++ -std=c++20 -O1 -Wall -Wextra -Wno-unused-parameter \
    -ffunction-sections -fdata-sections \
    -I"$WORK" -I"$SRC" -I"$KSRC" -I"$DRV" -I"$ABI" \
    -c -o "$WORK/oracle.o" "$HERE/oracle.cpp"
g++ -Wl,--gc-sections -o "$WORK/oracle" "$WORK/oracle.o" "$WORK/llama_like.o"

OUT="$WORK/out.txt"
"$WORK/oracle" sweep > "$OUT"

# The env axis, one process per value. `unset` really is unset.
env -u PIE_CUDA_DECODE_FUSED_POST "$WORK/oracle" fused_post unset  >> "$OUT"
PIE_CUDA_DECODE_FUSED_POST=""  "$WORK/oracle" fused_post empty >> "$OUT"
PIE_CUDA_DECODE_FUSED_POST="0" "$WORK/oracle" fused_post zero  >> "$OUT"
PIE_CUDA_DECODE_FUSED_POST="1" "$WORK/oracle" fused_post one   >> "$OUT"
PIE_CUDA_DECODE_FUSED_POST="x" "$WORK/oracle" fused_post other >> "$OUT"

if [[ -n "${LLC_ORACLE_OUT:-}" ]]; then
    cp "$OUT" "$LLC_ORACLE_OUT"
    echo "transcript written to $LLC_ORACLE_OUT" >&2
fi

python3 - "$OUT" <<'PY'
import sys
data = open(sys.argv[1], "rb").read()
h = 0xcbf29ce484222325
for b in data:
    h ^= b
    h = (h * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF
rows = data.count(b"\n")
print(f"GOLDEN_FNV1A64: 0x{h:016x}")
print(f"GOLDEN_ROWS: {rows}")
PY
