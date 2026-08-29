#!/usr/bin/env python3
"""The Metal kernel audit: what the shaders instantiate, against what the table declares.

An MSL entrypoint is not authored the way a CUDA launcher is. `kernels-cuda`
has one C++ function per symbol, so its table can have one row per symbol and a
reader can check them by eye. Here a `.metal` file holds a handful of TEMPLATE
bodies and a macro that stamps each one out over an axis product -- affine
format, head dim, tile width -- so the entrypoint set is generated, an order of
magnitude larger than the set of things anyone wrote, and not readable at all.

That `kernels-cuda` is the ARCHIVE crate, which did have one host C++ launcher
per symbol and was deleted at `85c6c674b`. The crate holding the name now is a
JIT, and the contrast is weaker against it than the paragraph above claims:
its `csrc/src` is 498 `__global__` across 120 `.cuh`, most of them templates,
and the Rust side names INSTANTIATIONS rather than functions -- `src/norm.rs`
introduces one root as "twelve `__global__` templates, fifteen" routines,
which is an axis product in miniature. What survives the correction is the
reason this script exists: fifteen against twelve can still be read by eye,
and a preprocessor-expanded MSL entrypoint set cannot be read at all.

So this reads the shaders the way the Metal runtime does and reports the set:

  * the preprocessor expands the `instantiate_*` macros, because they are the
    only place the axis product is written down;
  * `[[host_name("...")]]` is the entrypoint, and adjacent string literals are
    concatenated the way the compiler concatenates them;
  * a TEMPLATE DEFINITION is not an entrypoint. `template <typename T>
    [[kernel]] void row_gather(...)` is a body, and counting it doubled the
    census the first time this was run by hand.

The set is REPORTED and no longer compared against anything. `--table` held
invariant (1) --

    every entrypoint resolves to exactly one (row, axis point), and every
    (row, axis point) to exactly one entrypoint

-- by running `crates/kernels-metal/examples/entrypoints.rs` for the table's
half and diffing it against this census. That example is deleted with the rest
of `examples/` and the mode went with it, so nothing holds the Metal table
against its shaders now: an entrypoint a `.metal` instantiates and no row
declares is a nil pipeline at first launch, and a row whose axes over-generate
is a name that never resolves. Neither is red anywhere.

The Vulkan and WGSL siblings still hold their copy of the invariant inside
`cargo test`, because a variant is DECLARED there on a `// pie:instantiate`
line and reading the set is a parse rather than a preprocessor run. That
difference is why theirs survived the deletion and this one did not.

Usage:
    scripts/metal-kernel-audit.py            # the census
    scripts/metal-kernel-audit.py --bases    # census, grouped by base name
"""

import pathlib
import re
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

    if mode in ("--table", "--check", "--write"):
        # RETIRED, all three, and the branch exists rather than being deleted
        # so that a caller asking for a comparison gets an error instead of a
        # census. Without it `--table` falls through to the default below,
        # which prints the shader set and exits 0 -- a check that passes
        # without checking, which is the failure the `--no-concat` comment
        # describes further down.
        #
        # `--check` and `--write` maintained a committed
        # `crates/kernels-metal/entrypoints.generated.txt`: the shader half of
        # invariant (1), written to a file so `tests/entrypoints.rs` could diff
        # the table against it without running a C preprocessor. That file went
        # first, and `--table` replaced the pair by doing both hops in one
        # process -- census here, table from `cargo run -p kernels-metal
        # --example entrypoints`.
        #
        # That example is deleted with the rest of `examples/`, so `--table` is
        # retired rather than repointed, and no route back is cheap: the shader
        # half needs `gcc -E` because the axis product is written nowhere but
        # the `instantiate_*` macros, the table half needs a Rust process, and
        # a `cargo test` in this crate can be neither. `tests/entrypoints.rs`
        # says what it still covers -- the table against itself and against the
        # shader FILE names -- and the set comparison is not part of it.
        print(f"{mode} is retired: it compared the shader census to the Rust "
              f"table's axis product, and the `cargo run -p kernels-metal "
              f"--example entrypoints` it read the table with is deleted. "
              f"Nothing holds that comparison now.", file=sys.stderr)
        return 2

    if mode == "--one-way":
        # Invariant (3): `kernels-metal` depends on nothing above it.
        #
        # ONE tree now. This walked two -- the shader tree and a host C++
        # library under `include/pie/kernels/` -- and the second is deleted:
        # its launch shapes are Rust in `engine-metal/src/lowering/grid.rs`
        # and the C++ driver that compiled against them is gone. What is left
        # is the `*_params.h` a shader and its host caller must agree on,
        # which cannot be Rust because a `.metal` `#include`s it.
        #
        # A `#include` of a driver header, or a mention of a driver TYPE, is
        # the arrow turning around: it is how `runahead.hpp` ended up in
        # `kernels-cuda`, per .wiki/kernel-refactor.md §1.1. That
        # `kernels-cuda` is the ARCHIVE crate, deleted at `85c6c674b`, and
        # `runahead.hpp` is in no tree today -- the incident is history, not a
        # place to go look. Cite it as the shape of the mistake and nothing
        # more.
        shaders = ROOT / "crates/kernels-metal/kernels"
        own = {path.name for path in shaders.rglob("*") if path.is_file()}
        driver_types = ("RawMetalContext", "DeviceTuning", "SlotHandle", "Pso",
                        "MTL", "device_tuning", "mtl4_context", "decode_abi")
        problems = []
        headers = sorted(shaders.rglob("*.h")) + sorted(shaders.rglob("*.hpp"))
        for path in headers:
            for number, line in enumerate(path.read_text().splitlines(), 1):
                include = re.match(r'\s*#include\s*"([^"]+)"', line)
                if include and include.group(1).split("/")[-1] not in own:
                    problems.append((path, number, f"includes {include.group(1)}"))
                stripped = line.split("//")[0]
                for name in driver_types:
                    if name in stripped:
                        problems.append((path, number, f"names {name}"))

        # And the other direction: a header in the SHADER tree that no shader
        # reaches is host C++ in the runtime search path, which is the defect
        # this tree was split to fix and then grew back anyway.
        # `quant/affine_format.hpp` was a host-only struct sitting among the
        # `.metal` files, included by the deleted C++ library and by nothing
        # the compiler ever saw. An orphan header is how that starts, so the
        # orphan is what this refuses -- by reachability rather than by
        # extension, because the extension was never the tell.
        included = set()
        for path in list(shaders.rglob("*.metal")) + headers:
            for line in path.read_text(errors="ignore").splitlines():
                match = re.match(r'\s*#include\s*"([^"]+)"', line)
                if match:
                    included.add(match.group(1).split("/")[-1])
        for path in headers:
            if path.name not in included:
                problems.append((path, 0, "no shader includes it -- host C++ "
                                          "does not belong in the shader tree"))

        for path, number, why in problems:
            where = f"{path.relative_to(ROOT)}" + (f":{number}" if number else "")
            print(f"  {where}: {why}")
        if problems:
            print(f"\n{len(problems)} boundary violation(s). kernels-metal must not "
                  f"reach engine-metal: pass the value in instead. See "
                  f".wiki/kernel-metal-refactor.md \u00a76 invariant (3).")
            return 1
        print(f"kernels-metal reaches nothing above it, and every one of its "
              f"{len(headers)} headers is reached by a shader")
        return 0

    if mode in ("--no-concat", "--paths", "--dead-paths-unused", "--cpp"):
        # RETIRED, and failing rather than passing is the point.
        #
        # All four served the C++ Metal driver, and that driver was deleted
        # whole. The first three walked `crates/engine-metal/csrc`: `rglob`
        # over a missing directory yields nothing, so each printed its success
        # line over an empty set and had been guaranteeing nothing for as long
        # as the port has been finished. `--cpp` is the opposite failure and
        # the more misleading one -- it kept PASSING honestly, compiling a
        # real test against real headers, which is why it outlived the other
        # three: nothing about a green check says the thing it checks has no
        # callers left.
        #
        # None of the invariants went away with the C++. Each moved somewhere
        # stronger, which is why these are retired rather than repointed:
        #
        # * `--no-concat` forbade ASSEMBLING an entrypoint name, on the
        #   syntax. `model-compiler`'s `kernels::check_plan` runs from
        #   `trace::finish` on every plan and refuses any launched symbol
        #   no row declares, which checks the RESULT and so catches a name
        #   built by any means. `engine-metal`'s coverage ledger asserts
        #   the same thing over all eight texts at once.
        # * `--paths` and `--dead-paths-unused` required every `.metal`
        #   literal to resolve. The literals are now the table's own
        #   `KernelSig::file` fields, and `engine-metal`'s
        #   `every_file_a_kernel_row_states_is_a_file_that_exists` holds
        #   every one of them against the tree.
        # * `--cpp` compiled `csrc/tests/entrypoint_test.cpp` against
        #   `include/pie/kernels/entrypoint.h` to check that the name grammar
        #   accepts every real entrypoint and refuses the rest. Both files are
        #   deleted. `kernels-metal/tests/entrypoints.rs` holds that same
        #   invariant in the language the driver is actually written in, and
        #   the grammar it checks is the table's own -- not a second copy that
        #   had to be kept in step with it.
        #
        # A repointed regex would have been worse than any of them:
        # `AffineFormat::kernel_suffix` has a legitimate Rust caller that
        # MATCHES a suffix against the table rather than launching it, so the
        # C++ pattern read against Rust reports a defect that is not there.
        print(f"{mode} is retired: it served the C++ Metal driver, which the "
              f"Rust port removed. See the comment at this branch for where "
              f"its invariant is held now.")
        return 1

    for name in sorted(found):
        print(f"{name}\t{found[name]}")
    print(f"\n{len(found)} entrypoints in {len(set(found.values()))} files",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
