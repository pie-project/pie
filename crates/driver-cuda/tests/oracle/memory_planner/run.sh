#!/bin/bash
# Regenerate the golden for tests/memory_planner_parity.rs.
#
# Compiles the REAL store/memory_planner.cpp -- all 1,221 lines, restored from
# git now that the C++ tree is deleted -- and sweeps
# `plan_cuda_memory` over a grid of device shapes, model shapes and configs.
# Only the function's external inputs are stubbed: the three CUDA queries, the
# ~14 model workspace formulas, and the profile-cache read. Everything the
# planner DECIDES is the shipping code.
#
# Stubbing the CUDA queries is the point, not a compromise: the planner's
# answer is a function of the device shape, and no single machine presents more
# than one. The C++ has never been exercised on a device it was not running on.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../../../.." && pwd)"
SRC="$ROOT/crates/driver-cuda/csrc/src"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

# `memory_planner.cpp` includes its neighbours RELATIVELY -- "../config.hpp",
# "kv_cache.hpp" -- and a relative include always resolves against the
# including file's own directory, so -I cannot shadow it. The file under test
# is therefore COPIED verbatim into a tree whose neighbours are the stubs. `cp`
# is what keeps this honest: the copy is made at build time from the shipping
# source, so it cannot drift the way a checked-in duplicate would.
#
# The shipping source is GONE -- `4569b9e4b` deleted `crates/driver-cuda/csrc`
# when the planner was ported to Rust -- so it is restored from git instead of
# copied out of the tree. `oracle_census.rs` records this oracle as dead for
# that reason and it is not: the file is two `git show`s away, the stub tree
# survived, and the whole thing builds with plain `g++` and no CUDA.
#
# `7559e4cea` is the last revision whose includes match the stub tree, which
# is what makes it the right one rather than the newest one: `bb7c2231a` adds
# `#include "attention_workspace.hpp"`, a header the stubs do not have, and
# the build stops there. It is also the 1,221 lines this file's header claims.
# Confirmed by reproducing `GOLDEN_FNV1A64` exactly.
CPP_REV="${MP_ORACLE_REV:-7559e4cea}"
CPP_DIR="crates/driver-cuda/csrc/src/store"

mkdir -p "$WORK/store"
if [[ -f "$SRC/store/memory_planner.cpp" ]]; then
  cp "$SRC/store/memory_planner.cpp" "$SRC/store/memory_planner.hpp" "$WORK/store/"
else
  git -C "$ROOT" show "$CPP_REV:$CPP_DIR/memory_planner.cpp" > "$WORK/store/memory_planner.cpp"
  git -C "$ROOT" show "$CPP_REV:$CPP_DIR/memory_planner.hpp" > "$WORK/store/memory_planner.hpp"
fi
cp -r "$HERE/stub/." "$WORK/"

# The ONE edit made to the source, and it flips a switch the author left in.
#
# `plan_cuda_memory` ends with an `if constexpr (false)` block whose comment
# says: "the selected plan alone cannot tell you WHY it won, nor what the
# score-ranked runner-up was". That is exactly this harness's problem. Without
# it the transcript sees the winning SHAPE but not the score, the candidate
# count, or the arena -- so a changed score weight that does not happen to flip
# an argmax is invisible, and a mutation test then reports a false pass.
#
# Asserted to apply exactly once: if the block is renamed or removed, the
# oracle fails loudly rather than quietly losing the introspection.
python3 - "$WORK/store/memory_planner.cpp" <<'FLIP'
import sys
path = sys.argv[1]
s = open(path, encoding="utf-8").read()
needle = "    if constexpr (false) {"
if s.count(needle) != 1:
    sys.exit(f"expected exactly one introspection block, found {s.count(needle)}")
open(path, "w", encoding="utf-8").write(s.replace(needle, "    if constexpr (true) {"))
FLIP

g++ -std=c++20 -O1 -Wall \
    -I "$WORK" -I "$WORK/store" \
    "$HERE/oracle.cpp" "$WORK/store/memory_planner.cpp" \
    -o "$WORK/oracle"

"$WORK/oracle" > "$WORK/cpp.txt"

python3 - "$WORK/cpp.txt" "$WORK/norm.txt" <<'NORM'
import sys

# The one normalisation, applied identically by tests/memory_planner_parity.rs.
#
# `plan_cuda_memory` throws `std::runtime_error`; the Rust returns
# `PlanError`. The messages are compared -- they are the operator-facing half
# of a failed boot -- but the C++ prefixes nothing and the Rust's Display is
# the same string, so only the harness's own "THROWS " marker is folded to a
# stable token.
def normalise(line):
    if "|THROWS " in line:
        head, msg = line.split("|THROWS ", 1)
        return head + "|FAILED " + msg
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
NORM

if [[ -n "${MP_ORACLE_OUT:-}" ]]; then
  cp "$WORK/norm.txt" "$MP_ORACLE_OUT"
  echo "transcript written to $MP_ORACLE_OUT" >&2
fi
