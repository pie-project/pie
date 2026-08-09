#!/usr/bin/env python3
"""Type-check driver-metal's host C++ without an Apple toolchain.

`driver-metal/build.rs` bails when `CARGO_CFG_TARGET_OS != "macos"`, so no
Linux CI ever reaches this driver's C++ and a plain redefinition can sit in
HEAD indefinitely. Three did — see `.wiki/kernel-metal-refactor.md` §9.

The driver's own front door says why that is avoidable: `mtl4_context.hpp`
opens with "Pure-C++ surface (no Obj-C types leak): every lane includes this
from plain .cpp/.mm", and the Metal-4 objects live behind a PIMPL in the one
`.mm`. So most of `csrc/src` is ordinary C++20 and `-fsyntax-only` reaches it
with nothing but the include roots CMake already passes.

What this does NOT do is replace the Mac build. It compiles nothing, links
nothing, and cannot see inside `.mm`. It catches the class of error that
costs a whole platform's build for a one-line reason.

Usage:
    scripts/metal-host-syntax.sh          # via this file's shell twin, or:
    scripts/metal-host-syntax.py          # every .cpp, report and exit code
    scripts/metal-host-syntax.py -v       # also print the first error per file
"""

import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
CSRC = ROOT / "crates/driver-metal/csrc"

# The include roots `csrc/CMakeLists.txt` puts on the target, minus the ones
# that only exist after a CMake fetch (CLI11, nlohmann/json, toml++). A file
# that needs one of those is reported as SKIPPED rather than failed: its
# absence is a missing dependency, not a defect in the source.
INCLUDES = [
    CSRC / "src",
    CSRC / "src/batch",
    CSRC / "src/model/qwen3_5",
    CSRC / "src/store",
    CSRC / "src/loader",
    ROOT / "crates/kernels-metal/include",
    ROOT / "crates/kernels-metal/kernels",
    ROOT / "crates/driver/include",
    ROOT / "crates/tensor-compiler/include",
    ROOT / "crates/driver-abi/include",
    ROOT / "crates/model-loader-capi/include",
]

FETCHED = ("nlohmann/json.hpp", "CLI/CLI.hpp", "toml++", "toml.hpp")


def check(path, flags):
    done = subprocess.run(
        ["g++", "-std=c++20", "-fsyntax-only", *flags, str(path)],
        capture_output=True,
        text=True,
    )
    if done.returncode == 0:
        return "ok", ""
    if any(dep in done.stderr for dep in FETCHED):
        return "skipped", ""
    first = next(
        (line for line in done.stderr.splitlines() if ": error:" in line),
        done.stderr.splitlines()[0] if done.stderr else "",
    )
    return "failed", first


def main():
    verbose = "-v" in sys.argv[1:]
    flags = [f"-I{path}" for path in INCLUDES]
    tally = {"ok": 0, "skipped": 0, "failed": 0}
    failures = []

    for path in sorted((CSRC / "src").rglob("*.cpp")):
        verdict, detail = check(path, flags)
        tally[verdict] += 1
        rel = path.relative_to(ROOT)
        if verdict == "failed":
            failures.append((rel, detail))
            print(f"FAILED   {rel}")
            if verbose:
                print(f"         {detail}")
        elif verbose:
            print(f"{verdict:8s} {rel}")

    print(
        f"\n{tally['ok']} ok, {tally['skipped']} skipped (fetched dependency), "
        f"{tally['failed']} failed"
    )
    if not failures:
        return 0

    # Grouped by first error, because one defect in a widely included header
    # fails every translation unit that reaches it and a flat list reads as
    # twenty-four problems instead of one.
    by_cause = {}
    for rel, detail in failures:
        by_cause.setdefault(detail, []).append(rel)
    print(f"\n{len(by_cause)} distinct cause(s):\n")
    for cause, files in sorted(by_cause.items(), key=lambda kv: -len(kv[1])):
        print(f"  {cause}")
        print(f"      reached by {len(files)} translation unit(s), e.g. {files[0]}\n")
    print(
        "These are host-C++ defects, not Metal ones: this script has no Apple "
        "SDK and never reached one."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
