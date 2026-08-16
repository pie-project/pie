#!/bin/bash
# Regenerate the golden for tests/store_parity.rs.
#
# Builds the real C++ as an oracle and prints the hash, byte count, and row
# count that store_parity.rs pins. Run it after any change to the C++ sources
# it extracts from: if the numbers move, either the C++ changed behaviour or
# the Rust port has drifted, and the parity test will say which.
#
# Requires g++ with C++20 and nlohmann/json.hpp on the include path. This
# machine has the latter only via the vendored cudnn-frontend copy, hence the
# search below.
#
# The `crates/kernels-cuda/csrc/src` on the g++ line is the ARCHIVE crate's —
# CMake+nvcc, a `csrc/` of host `.hpp` and `.cpp` — deleted whole at
# `85c6c674b`, along with the `$SRC` tree beside it (`4569b9e4b`). The line is
# left unchanged because it records the command that took the golden, not a
# path anyone can follow; see tests/oracle_census.rs.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../../../.." && pwd)"
SRC="$ROOT/crates/driver-cuda/csrc/src"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

# shellcheck source=../nlohmann.sh
source "$HERE/../nlohmann.sh"
NLOHMANN="$(find_nlohmann "$ROOT")"

cp -r "$HERE/stub" "$HERE/oracle.cpp" "$WORK/"
( cd "$WORK" && bash "$HERE/extract.sh" "$SRC" >/dev/null )

g++ -std=c++20 -I "$WORK/stub" -I "$SRC" -I "$ROOT/crates/kernels-cuda/csrc/src" \
    -I "$NLOHMANN" -I "$WORK" \
    "$WORK/oracle.cpp" "$SRC/store/kv_cache_format.cpp" -o "$WORK/oracle"

"$WORK/oracle" > "$WORK/cpp.txt"

python3 - "$WORK/cpp.txt" <<'PY'
import sys
data = open(sys.argv[1], 'rb').read()
h = 0xcbf29ce484222325
for b in data:
    h = ((h ^ b) * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF
print(f"GOLDEN_FNV1A64 = 0x{h:016x}")
print(f"GOLDEN_BYTES   = {len(data)}")
print(f"GOLDEN_ROWS    = {data.count(10)}")
PY
