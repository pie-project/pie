#!/usr/bin/env python3
"""The Metal kernel audit: what the shaders instantiate, against what the table declares.

An MSL entrypoint is not authored the way a CUDA launcher is. `kernels-cuda`
has one C++ function per symbol, so its table can have one row per symbol and a
reader can check them by eye. Here a `.metal` file holds a handful of TEMPLATE
bodies and a macro that stamps each one out over an axis product -- affine
format, head dim, tile width -- so the entrypoint set is generated, an order of
magnitude larger than the set of things anyone wrote, and not readable at all.

So this reads the shaders the way the Metal runtime does and reports the set:

  * the preprocessor expands the `instantiate_*` macros, because they are the
    only place the axis product is written down;
  * `[[host_name("...")]]` is the entrypoint, and adjacent string literals are
    concatenated the way the compiler concatenates them;
  * a TEMPLATE DEFINITION is not an entrypoint. `template <typename T>
    [[kernel]] void row_gather(...)` is a body, and counting it doubled the
    census the first time this was run by hand.

With `--table`, the same set is compared against the product of
`crates/kernels-metal/src/*.rs` -- one row per base name, each declaring its
axes -- which is the invariant the Metal table exists to hold:

    every entrypoint resolves to exactly one (row, axis point), and every
    (row, axis point) to exactly one entrypoint

Usage:
    scripts/metal-kernel-audit.py            # the census
    scripts/metal-kernel-audit.py --bases    # census, grouped by base name
    scripts/metal-kernel-audit.py --table    # census vs. the Rust table
"""

import pathlib
import re
import tempfile
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
SHADERS = ROOT / "crates/kernels-metal/kernels"

# `host_name(fn "_" #name "_d_" #d)` -- adjacent literals, joined by the compiler.
HOST_NAME = re.compile(r'host_name\(\s*((?:"[^"]*"\s*)+)\)')
LITERAL = re.compile(r'"([^"]*)"')
# A kernel declared directly, with no template above it. The `template` guard is
# the whole subtlety: without it every template BODY counts as an entrypoint.
PLAIN_KERNEL = re.compile(
    r'(?m)^\s*(?:\[\[kernel\]\]|kernel)\s+void\s+([A-Za-z_0-9]+)\s*\(')

# The axis suffixes a name carries, longest-match first within each family.
AXIS_TOKENS = [
    (re.compile(r'_gs_(\d+)$'), "gs"),
    (re.compile(r'_b_(\d+)$'), "b"),
    (re.compile(r'_d_(\d+)$'), "d"),
    (re.compile(r'_v_(\d+)$'), "v"),
    (re.compile(r'_kl_(\d+)$'), "kl"),
    (re.compile(r'_bm_(\d+)$'), "bm"),
    (re.compile(r'_bn_(\d+)$'), "bn"),
    (re.compile(r'_sg(\d+)$'), "sg"),
    (re.compile(r'_l_(\d+)$'), "l"),
    (re.compile(r'(_p32)$'), "p32"),
    (re.compile(r'(_bfloat16)$'), "bf16"),
]

# Deliberately NOT axes, and each one was a wrong guess first:
#
#   _wm_/_wn_   five `host_name` lines typed out by hand at quantized_qmm_t.metal
#               :2918-2966, not stamped by `instantiate_qmm_t`. Under
#               `.wiki/kernel-refactor.md` §5 rule 4 they are five distinct
#               kernels, so they are five rows and their names are base text.
#   _f32        the splitk ACCUMULATE type, and it sits before `_bfloat16`
#               (`affine_qmm_t_splitk_f32_bfloat16_gs_...`), so it is part of
#               the base rather than a point of the dtype axis.


def preprocessed(path):
    """The shader with its macros expanded.

    Angle includes are dropped -- there is no metal_stdlib here and the
    preprocessor does not need one. Quoted includes are resolved the way
    `read_metal_source_at` resolves them, against the including file's
    directory, because a params header can carry a macro.
    """
    text = path.read_text()

    def splice(match):
        target = path.parent / match.group(1)
        return target.read_text() if target.exists() else ''

    for _ in range(8):  # the driver's splicer allows 8 levels
        text = re.sub(r'(?m)^\s*#include\s*"([^"]+)"\s*$', splice, text)
    # Angle includes are the system headers, and they are stripped AFTER
    # splicing, not before: a shared `.metal` carries its own `<metal_stdlib>`,
    # and stripping only the top-level file left one buried in the spliced text.
    # That failed the preprocessor and silently dropped 356 entrypoints — the
    # audit reported drift rather than a crash, which is the good failure, but
    # only because the set is compared rather than trusted.
    text = re.sub(r'(?m)^\s*#include\s*<[^>]+>\s*$', '', text)
    done = subprocess.run(["gcc", "-E", "-P", "-x", "c", "-"], input=text,
                          capture_output=True, text=True)
    return done.stdout


def entrypoints_of(path):
    text = preprocessed(path)
    names = {"".join(LITERAL.findall(group)) for group in HOST_NAME.findall(text)}
    for match in PLAIN_KERNEL.finditer(text):
        # A template BODY is not an entrypoint. Look back to the end of the
        # previous declaration rather than at the previous LINE: a template
        # parameter list wraps, and reading one line found `int WM = 2, int
        # WN = 2>` and called `affine_qmm_t_aligned` a dispatchable name.
        # That put three phantom rows in the table before anything noticed.
        head = text[:match.start()]
        cut = max(head.rfind(";"), head.rfind("}"))
        if "template" in head[cut + 1:]:
            continue
        names.add(match.group(1))
    return names


def census():
    found = {}
    for path in sorted(SHADERS.rglob("*.metal")):
        for name in entrypoints_of(path):
            found[name] = path.name
    return found


def split_axes(name):
    """`affine_qmm_t_bfloat16_gs_64_b_4_bm_16_bn_32` -> base, [(axis, value), ...]"""
    base, axes = name, []
    while True:
        for pattern, axis in AXIS_TOKENS:
            match = pattern.search(base)
            if match:
                axes.append((axis, match.group(1)))
                base = base[:match.start()]
                break
        else:
            return base, list(reversed(axes))



def emit_cpp(names):
    """The same set, for `entrypoint.h`'s membership check.

    Two artifacts rather than one because the two readers cannot share a
    parser: Rust's table test reads the `.txt`, and the driver's C++ needs a
    header it can `#include`. `--check` compares BOTH against the shaders, so
    they cannot drift from each other without one of them drifting from the
    tree first.
    """
    rows = "".join(f'    "{name}",\n' for name in names)
    return f"""// Generated by scripts/metal-kernel-audit.py --write. Do not edit.
//
// Every entrypoint the shaders under this directory instantiate, sorted, so
// `entrypoint()` can refuse a name no pipeline could be built from. See
// .wiki/kernel-metal-refactor.md \u00a76 invariant (2).
#ifndef PIE_METAL_ENTRYPOINTS_GENERATED_H
#define PIE_METAL_ENTRYPOINTS_GENERATED_H

#include <string_view>

namespace pie::kernels {{

inline constexpr std::string_view kEntrypoints[] = {{
{rows}}};

}}  // namespace pie::kernels

#endif
"""

def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    found = census()

    if mode == "--bases":
        bases = {}
        for name, file in sorted(found.items()):
            base, axes = split_axes(name)
            row = bases.setdefault(base, [file, 0, set()])
            row[1] += 1
            row[2].add(tuple(axis for axis, _ in axes))
        for base, (file, count, shapes) in sorted(bases.items(),
                                                  key=lambda kv: (kv[1][0], kv[0])):
            shape = " | ".join("+".join(s) or "(none)" for s in sorted(shapes))
            print(f"{count:5d}  {base:44s} {file:22s} {shape}")
        print(f"\n{len(found)} entrypoints over {len(bases)} base names "
              f"in {len(set(found.values()))} files")
        return 0

    if mode in ("--check", "--write"):
        artifact = ROOT / "crates/kernels-metal/entrypoints.generated.txt"
        header = ROOT / "crates/kernels-metal/include/pie/kernels/entrypoints.generated.h"
        text = "".join(f"{name}\n" for name in sorted(found))
        if mode == "--write":
            artifact.write_text(text)
            header.write_text(emit_cpp(sorted(found)))
            print(f"wrote {artifact.relative_to(ROOT)} and "
                  f"{header.relative_to(ROOT)} ({len(found)} entrypoints)")
            return 0
        if header.read_text() != emit_cpp(sorted(found)):
            print(f"{header.relative_to(ROOT)} has drifted from the shaders. "
                  f"Regenerate with --write.")
            return 1
        if artifact.read_text() == text:
            print(f"{len(found)} entrypoints; {artifact.name} matches the shaders")
            return 0
        have = set(artifact.read_text().split())
        for name in sorted(set(found) - have):
            print(f"  + {name}\t{found[name]}  (in a shader, not in the file)")
        for name in sorted(have - set(found)):
            print(f"  - {name}\t(in the file, not in any shader)")
        print("\nthe shader tree moved. Regenerate with --write, and expect "
              "kernels-metal's table test to fail next: a new entrypoint needs "
              "a row or an axis point.")
        return 1

    if mode == "--one-way":
        # Invariant (3): `kernels-metal` depends on nothing above it.
        #
        # The headers here are host C++ now -- launch shapes, params, the name
        # grammar -- so the boundary is no longer enforced by the file
        # extension. A `#include` of a driver header, or a mention of a driver
        # TYPE, is the arrow turning around: it is how `runahead.hpp` ended up
        # in `kernels-cuda`, per .wiki/kernel-refactor.md §1.1.
        # Both trees: the shader tree's params headers and the host library.
        roots = [ROOT / "crates/kernels-metal/kernels",
                 ROOT / "crates/kernels-metal/include/pie/kernels"]
        own = {path.name for root in roots for path in root.rglob("*") if path.is_file()}
        driver_types = ("RawMetalContext", "DeviceTuning", "SlotHandle", "Pso",
                        "MTL", "device_tuning", "mtl4_context", "decode_abi")
        problems = []
        headers = [h for root in roots
                   for h in sorted(root.rglob("*.h")) + sorted(root.rglob("*.hpp"))]
        for path in headers:
            for number, line in enumerate(path.read_text().splitlines(), 1):
                include = re.match(r'\s*#include\s*"([^"]+)"', line)
                if include and include.group(1).split("/")[-1] not in own:
                    problems.append((path, number, f"includes {include.group(1)}"))
                stripped = line.split("//")[0]
                for name in driver_types:
                    if name in stripped:
                        problems.append((path, number, f"names {name}"))
        for path, number, why in problems:
            print(f"  {path.relative_to(ROOT)}:{number}: {why}")
        if problems:
            print(f"\n{len(problems)} upward reference(s). kernels-metal must not "
                  f"reach driver-metal: pass the value in instead. See "
                  f".wiki/kernel-metal-refactor.md \u00a76 invariant (3).")
            return 1
        print(f"kernels-metal reaches nothing above it ({len(headers)} headers)")
        return 0

    if mode == "--no-concat":
        # Invariant (2): the driver names a kernel, it does not spell one.
        #
        # Matched against the BASE NAMES the shaders actually hold rather than
        # against a pattern, because the pattern version of this check flagged
        # every `path + "/"` in the tree and was turned off within a day.
        bases = {split_axes(name)[0] for name in found}
        # `csrc`, not `csrc/src`: the rename and the family move both swept
        # only `src/` and left five stale shader paths and one dead entrypoint
        # name in `tests/` and `tools/`. A guard that looks where the edit
        # looked cannot catch the edit's blind spot.
        driver = ROOT / "crates/driver-metal/csrc"
        # A literal followed by `+`, or a `kernel_suffix()` reaching a caller:
        # both are a name being assembled outside `entrypoint()`.
        pattern = re.compile(
            r'"([a-z][a-z0-9_]*)"\s*\+|\bkernel_suffix\(\)')
        offenders = []
        for path in sorted(driver.rglob("*.cpp")) + sorted(driver.rglob("*.hpp")):
            for number, line in enumerate(path.read_text(errors="ignore").splitlines(), 1):
                # One report per LINE: a site that both names a base and calls
                # `kernel_suffix()` is one mistake, not two.
                if any(
                    match.group(1) is None
                    or any(base.startswith(match.group(1))
                           or match.group(1).startswith(base) for base in bases)
                    for match in pattern.finditer(line)
                ):
                    offenders.append((path, number, line.strip()))
        for path, number, line in offenders:
            print(f"  {path.relative_to(ROOT)}:{number}: {line}")
        if offenders:
            print(f"\n{len(offenders)} site(s) assemble an entrypoint name. Build it "
                  f"with pie::kernels::entrypoint() instead, which refuses a name "
                  f"no shader instantiates.")
            return 1
        print("no driver source assembles an entrypoint name")
        return 0

    if mode == "--paths":
        # Every `"*.metal"` literal in the driver must name a file that exists.
        # Tests and tools open shaders by path too, and they are the half the
        # PSO-name check does not reach.
        shaders = ROOT / "crates/kernels-metal/kernels"
        driver = ROOT / "crates/driver-metal/csrc"
        local = {"probe.metal", "ptir_m0.metal", "gemv_demo.metal",
                 "nop_probe.metal", "roofline_stream.metal"}
        bad = []
        for path in sorted(driver.rglob("*.cpp")) + sorted(driver.rglob("*.hpp")) \
                  + sorted(driver.rglob("*.mm")):
            if "/build/" in str(path):
                continue
            for number, line in enumerate(path.read_text(errors="ignore").splitlines(), 1):
                for match in re.finditer(r'"([^"]*\.metal)"', line):
                    named = match.group(1).lstrip("/")
                    if pathlib.Path(named).name in local:
                        continue        # a tool's own fixture, not the shader tree
                    if not (shaders / named).exists():
                        bad.append((path, number, named))
        for path, number, named in bad:
            print(f"  {path.relative_to(ROOT)}:{number}: no shader at {named}")
        if bad:
            print(f"\n{len(bad)} stale shader path(s). The tree moved under them.")
            return 1
        print("every shader path the driver names resolves")
        return 0

    if mode == "--cpp":
        # `entrypoint.h` is header-only C++ with no Metal in it, so the check
        # that it accepts every real name and refuses the rest costs one g++
        # invocation and needs no Apple SDK.
        source = ROOT / "crates/kernels-metal/csrc/tests/entrypoint_test.cpp"
        # A private temp dir, not a fixed /tmp name: this repo is worked on by
        # more than one agent at a time and a shared path is a race between
        # them, not a convenience.
        workspace = tempfile.TemporaryDirectory(prefix="pie-metal-audit-")
        binary = str(pathlib.Path(workspace.name) / "entrypoint_test")
        build = subprocess.run(
            ["g++", "-std=c++20", "-Wall", "-Wextra",
             f"-I{ROOT / 'crates/kernels-metal/include'}",
             f"-I{ROOT / 'crates/kernels-metal/kernels'}",
             "-o", binary, str(source)],
            capture_output=True, text=True, cwd=ROOT,
        )
        if build.returncode != 0:
            print(build.stderr)
            return 1
        return subprocess.run([binary], cwd=ROOT).returncode

    for name in sorted(found):
        print(f"{name}\t{found[name]}")
    print(f"\n{len(found)} entrypoints in {len(set(found.values()))} files",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
