#!/bin/bash
# Regenerate the golden for tests/profile_cache_parity.rs.
#
# Compiles the REAL store/planner_profile_cache.cpp and drives it. Only its two
# external inputs are stubbed -- the engine's `cache_dir()` and the two structs
# `make_planner_profile_key` copies fields out of -- so the logic under test is
# the shipping code, not a copy of it.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../../../.." && pwd)"
SRC="$ROOT/crates/driver-cuda/csrc/src"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

# shellcheck source=../nlohmann.sh
source "$HERE/../nlohmann.sh"
NLOHMANN="$(find_nlohmann "$ROOT")"

# `planner_profile_cache.cpp` includes its neighbours RELATIVELY --
# `"../config.hpp"`, `"kv_cache_format.hpp"` -- and a relative include always
# resolves against the including file's own directory, so -I cannot shadow it.
# The two files under test are therefore COPIED verbatim into a tree whose
# neighbours are the stubs. `cp` is what keeps this honest: the copy is made at
# build time from the shipping source, so it cannot drift the way a
# checked-in duplicate would.
mkdir -p "$WORK/store" "$WORK/model"
cp "$SRC/store/planner_profile_cache.cpp" "$SRC/store/planner_profile_cache.hpp" "$WORK/store/"
cp "$HERE/stub/config.hpp" "$WORK/config.hpp"
cp "$HERE/stub/model/config.hpp" "$WORK/model/config.hpp"
cp "$HERE/stub/store/kv_cache_format.hpp" "$WORK/store/kv_cache_format.hpp"

g++ -std=c++20 -O1 \
    -I "$WORK" -I "$ROOT/crates/driver-cuda-new/tests/oracle/store/stub" \
    -I "$NLOHMANN" \
    "$HERE/oracle.cpp" "$WORK/store/planner_profile_cache.cpp" \
    -o "$WORK/oracle"

"$WORK/oracle" > "$WORK/cpp.txt"

python3 - "$WORK/cpp.txt" "$WORK/norm.txt" <<'NORM'
import sys

# The one normalisation, applied identically by tests/profile_cache_parity.rs.
#
# `planner_profile_cache_lookup` documents itself as never throwing, but does:
# on the inputs where nlohmann's `value()` raises type_error.302. The Rust
# returns `Lookup::Unusable` for exactly those inputs. WHICH inputs are refused
# is what parity means here, and that is compared byte for byte; only the
# wording -- nlohmann's, which the Rust cannot and should not reproduce -- is
# dropped.
def normalise(line):
    for marker in ("|THROWS|", "|unusable|"):
        if marker in line:
            return line.split(marker)[0] + "|REFUSED"
    if "|miss|err=" in line:
        head, rest = line.split("|miss|err=", 1)
        if rest and not rest.startswith("schema version"):
            return head + "|miss|err=PARSE"
    return line

raw = open(sys.argv[1], encoding="utf-8").read()
out = "".join(normalise(l) + "\n" for l in raw.split("\n")[:-1])
open(sys.argv[2], "w", encoding="utf-8").write(out)

data = out.encode()
h = 0xcbf29ce484222325
for b in data:
    h = ((h ^ b) * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF
print(f"GOLDEN_FNV1A64 = 0x{h:016x}")
print(f"GOLDEN_ROWS    = {data.count(10)}")
print(f"(raw, un-normalised: {len(raw.encode())} bytes)")
NORM

if [[ -n "${PC_ORACLE_OUT:-}" ]]; then
  cp "$WORK/norm.txt" "$PC_ORACLE_OUT"
  cp "$WORK/cpp.txt" "$PC_ORACLE_OUT.raw"
  echo "transcript written to $PC_ORACLE_OUT" >&2
fi
