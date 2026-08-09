#!/bin/bash
# Regenerate the golden for tests/attn_score_parity.rs — gate-score-capture.
#
# Compiles the REAL model/attn_score.cu (with -x c++ — it carries no device
# code) against the REAL hook_sideband_arena.cpp; the replaced surfaces are
# the CUDA entry points (recorders), the fold launch (a recorder), a
# three-field StageHooks and a one-integer KvCache. stderr is dropped: the
# refusal messages are prose, and the refusals themselves are in the rows.
#
# `default_attn_score_window` caches its env read in a static, so that axis
# is swept across PROCESSES.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../../../.." && pwd)"
SRC="$ROOT/crates/driver-cuda/csrc/src"
KSRC="$ROOT/crates/kernels-cuda/csrc/src"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

mkdir -p "$WORK/model"
cp "$SRC/model/attn_score.cu" "$SRC/model/attn_score.hpp" \
   "$SRC/model/attn_observation.hpp" \
   "$SRC/model/hook_sideband_arena.cpp" "$SRC/model/hook_sideband_arena.hpp" \
   "$WORK/model/"
cp -r "$HERE/stub/." "$WORK/"

g++ -std=c++20 -O1 -Wall -Wextra \
    -I"$WORK" -I"$KSRC" \
    -o "$WORK/oracle" \
    "$HERE/oracle.cpp" \
    -x c++ "$WORK/model/attn_score.cu" -x none \
    "$WORK/model/hook_sideband_arena.cpp"

OUT="$WORK/out.txt"
"$WORK/oracle" 2>/dev/null > "$OUT"

for w in unset empty 0 -1 33 4096 4097 abc 1e3; do
    if [[ "$w" == "unset" ]]; then
        env -u PIE_ATTN_SCORE_WINDOW "$WORK/oracle" window "w-unset" \
            2>/dev/null >> "$OUT"
    elif [[ "$w" == "empty" ]]; then
        PIE_ATTN_SCORE_WINDOW= "$WORK/oracle" window "w-empty" \
            2>/dev/null >> "$OUT"
    else
        PIE_ATTN_SCORE_WINDOW="$w" "$WORK/oracle" window "w-$w" \
            2>/dev/null >> "$OUT"
    fi
done

if [[ -n "${AS_ORACLE_OUT:-}" ]]; then
    cp "$OUT" "$AS_ORACLE_OUT"
    echo "transcript written to $AS_ORACLE_OUT" >&2
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
