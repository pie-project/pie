#!/bin/bash
# Regenerate the golden for tests/dtoa_parity.rs.
#
# The oracle here is `nlohmann::json` itself rather than anything extracted
# from the driver: what is being pinned is that this crate's Grisu2 port emits
# the same bytes the C++ writes into the planner profile cache.
#
# Requires g++ with C++20 and nlohmann/json.hpp on the include path.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../../../.." && pwd)"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

# shellcheck source=../nlohmann.sh
source "$HERE/../nlohmann.sh"
NLOHMANN="$(find_nlohmann "$ROOT")"

g++ -O2 -std=c++20 -I "$NLOHMANN" "$HERE/oracle.cpp" -o "$WORK/oracle"
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

# Also leave the corpus where a failing test run can diff against it.
if [[ -n "${DTOA_ORACLE_OUT:-}" ]]; then
  cp "$WORK/cpp.txt" "$DTOA_ORACLE_OUT"
  echo "corpus written to $DTOA_ORACLE_OUT" >&2
fi
