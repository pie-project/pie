#!/bin/bash
# Regenerate the golden for tests/workspace_parity.rs.
#
# Compiles the REAL model/workspace.cpp and drives `Workspace::allocate_full`
# over a grid of model shapes, reporting the exact sequence of tensor
# allocations each one makes — and, beside it, what `workspace_bytes` tells the
# memory planner the same shape costs.
#
# The only replaced implementation is `DeviceTensor::allocate`, whose real body
# ends in a `cudaMalloc`. Its DECLARATION comes from the verbatim
# `kernels-cuda/csrc/src/tensor.hpp`, so the shipping code still computes every
# shape; the recorder writes down what it was handed. Every decision about
# WHICH tensors exist — the always-on fused buffers, the padded q/k/v branch,
# the MTP row arithmetic — is shipping code.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../../../.." && pwd)"
SRC="$ROOT/crates/driver-cuda/csrc/src"
KSRC="$ROOT/crates/kernels-cuda/csrc/src"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

# Relative includes resolve against the including file's own directory, so -I
# cannot shadow them: the files under test are COPIED into a tree whose
# neighbours are the stubs. The copy happens at build time from the shipping
# source, so it cannot drift the way a checked-in duplicate would.
mkdir -p "$WORK/model"
cp "$SRC/model/workspace.cpp" "$SRC/model/workspace.hpp" "$WORK/model/"
cp "$KSRC/tensor.hpp" "$WORK/"
cp "$HERE/../kv_cache/stub/tensor_recorder.cpp" "$WORK/"
cp -r "$HERE/stub/." "$WORK/"

# `kArgmaxAccumSlots` is taken from the real header rather than retyped into
# the stub, so the oracle cannot keep using 32 after the kernel changes it.
SLOTS="$(sed -n 's/^constexpr int kArgmaxAccumSlots = \([0-9]\+\);.*/\1/p' \
    "$KSRC/sample/argmax.hpp")"
if [[ -z "$SLOTS" ]]; then
    echo "could not read kArgmaxAccumSlots from $KSRC/sample/argmax.hpp" >&2
    exit 1
fi

g++ -std=c++20 -O1 -Wall -Wextra -Wno-unused-parameter \
    -DPIE_ARGMAX_ACCUM_SLOTS="$SLOTS" \
    -I"$WORK" \
    -o "$WORK/oracle" \
    "$HERE/oracle.cpp" "$WORK/model/workspace.cpp" "$WORK/tensor_recorder.cpp"

OUT="$WORK/out.txt"
"$WORK/oracle" > "$OUT"

if [[ -n "${WS_ORACLE_OUT:-}" ]]; then
    cp "$OUT" "$WS_ORACLE_OUT"
    echo "transcript written to $WS_ORACLE_OUT" >&2
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
