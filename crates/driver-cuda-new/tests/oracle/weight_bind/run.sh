#!/bin/bash
# Regenerate the golden for tests/weight_bind_parity.rs.
#
# Compiles the REAL `model/llama_like/qwen3.cpp` — the shipping binders,
# unmodified — against a stub LoadedModel. The stub replaces the engine
# because a real one needs a checkpoint on disk and a GPU; it replaces nothing
# about the code under test.
#
# `qwen3.cpp` is copied into a temp tree beside the stubs rather than compiled
# in place, because its `#include "model/llama_like/qwen3.hpp"` and that
# header's `#include "model/loaded_model.hpp"` resolve against the including
# file's own directory first, where the real headers sit. `-I` cannot shadow
# that; a copy can.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../../../.." && pwd)"
DSRC="$ROOT/crates/driver-cuda/csrc/src"
KSRC="$ROOT/crates/kernels-cuda/csrc/src"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

mkdir -p "$WORK/src/model/llama_like"
cp "$DSRC/model/llama_like/qwen3.hpp" "$WORK/src/model/llama_like/"
cp "$DSRC/model/llama_like/qwen3.cpp" "$WORK/src/model/llama_like/"

g++ -std=c++20 -O1 -Wall -Wextra \
    -I"$HERE/stub" -I"$WORK/src" -I"$KSRC" \
    -o "$WORK/oracle" \
    "$HERE/oracle.cpp" "$WORK/src/model/llama_like/qwen3.cpp" "$KSRC/tensor.cpp"

OUT="$WORK/out.txt"
"$WORK/oracle" > "$OUT"

if [[ -n "${WB_ORACLE_OUT:-}" ]]; then
    cp "$OUT" "$WB_ORACLE_OUT"
    echo "transcript written to $WB_ORACLE_OUT" >&2
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
