#!/bin/bash
# Regenerate the golden for tests/attn_ws_parity.rs — gate-attn-ws.
#
# Compiles the REAL attention_workspace.cpp against its real header and the
# real kernels-cuda view/tensor headers. Replaced implementations:
# `DeviceTensor::allocate` (the shared tensor_recorder) and the six CUDA
# entry points the TU calls (stub/cuda_recorder.cpp), which record pins and
# events symbolically — the call sequence IS the behaviour under test.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../../../.." && pwd)"
SRC="$ROOT/crates/driver-cuda/csrc/src"
KSRC="$ROOT/crates/kernels-cuda/csrc/src"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

# The file under test is COPIED beside the stubs so its quote-includes
# (`cuda_check.hpp`, `<cuda_runtime.h>` via -I order) resolve to them.
cp "$SRC/attention_workspace.cpp" "$SRC/attention_workspace.hpp" "$WORK/"
cp "$KSRC/attention_workspace_view.hpp" "$KSRC/tensor.hpp" \
   "$KSRC/quant_meta.hpp" "$WORK/" 2>/dev/null || \
cp "$KSRC/attention_workspace_view.hpp" "$KSRC/tensor.hpp" "$WORK/"
cp "$HERE/../kv_cache/stub/tensor_recorder.cpp" "$WORK/"
cp -r "$HERE/stub/." "$WORK/"

g++ -std=c++20 -O1 -Wall -Wextra -Wno-unused-parameter \
    -I"$WORK" \
    -o "$WORK/oracle" \
    "$HERE/oracle.cpp" "$WORK/attention_workspace.cpp" \
    "$WORK/tensor_recorder.cpp" "$WORK/cuda_recorder.cpp"

OUT="$WORK/out.txt"
"$WORK/oracle" > "$OUT"

if [[ -n "${AW_ORACLE_OUT:-}" ]]; then
    cp "$OUT" "$AW_ORACLE_OUT"
    echo "transcript written to $AW_ORACLE_OUT" >&2
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
