"""The kernel-reachability audit: a C++ launcher no root reaches is DEAD.

`kernels-cuda`'s `kernels/` is 27,000 lines of CUDA that Rust can enter in
exactly two ways: through the generated shim's `pie_k_*` entry points, and
through the handful of `extern "C"` functions the driver names directly
(the plan lifecycle, the three vision towers, the allocator hooks). A
function no root reaches is not called by anything at all -- not by a fire,
not by a test, not by another `.cu`.

## Why the one-hop version was not enough

Every audit before this one asked *"does anything call THIS function"* and
answered it correctly. `.wiki/driver/new-horizon.md` §34 found that a wall
in front of a door nobody opens is not a wall; §38 found the same shape one
level down in `NoRow::Orphaned`, where two of three entries were held by a
prose reason that measurement did not support.

The shape survives one more level, and §38.9 named it:
`attention_mtp_history_bf16` was kept because `attention_naive.cu:80` calls
it -- a true statement, about a caller that is itself unreached. **Orphaned
at one remove.** No amount of care in a one-hop check finds that; it needs a
transitive closure, which is this.

## What it over-approximates, deliberately

Reachability, in every direction it is unsure about:

* an overload set resolves to EVERY definition of that name;
* a template instantiation resolves to its primary;
* a call through a macro resolves as if written out;
* a call through a function pointer or a `std::function` is missed, and this
  is the one under-approximation -- see below.

So a function this reports as UNREACHABLE is unreachable under the most
generous reading available without a compiler, which is the direction a
deletion candidate has to err in.

The exception is worth stating plainly because it is the way this tool can
be wrong in the dangerous direction: **a call made only through a function
pointer is invisible here.** `new-horizon.md` §37 inspected all 22
name-mentions that were not plain calls and found 21 to be
`device::<leaf><<<>>>` launches and one a parameter name -- so the tree has
no such indirection today. If one appears, this audit will call its target
dead. That is why its output is a CANDIDATE LIST and not a delete script:
§10.10's rule still stands, and a deletion is a separate claim with its own
evidence.

## Comments come out first, and this is not a detail

The first version of this ran its definition regex over raw text. `name(`
appears in prose constantly, and a match there brace-matches forward into
whatever real code follows -- which invented three functions named `once`,
`fit` and `both`, credited with 7, 5 and 2 kernel launches, out of the words
"run over all Ntot rows at once", "while the K winners fit" and "with bias
on both". Every one of them looked like a plausible finding.

## The tree this was written against is gone, and its name was reused

Everything above describes the ARCHIVE crate. `kernels-cuda` built
`libpie_kernels_cuda.a` with CMake and nvcc, and its `csrc/src` was host and
device C++ together: the `.cu` launchers, the generated `pie_k_*` shim they
were reached through, and the `extern "C"` entry points listed under
`DIRECT_ROOTS`. `85c6c674b` deleted that crate whole, `links` key and all.
The JIT crate that has since taken the name is not a smaller version of it --
its `kernels/` is 120 `.cuh` of DEVICE text with no host layer at all, and a
kernel is entered by `cuLaunchKernel` from Rust after NVRTC compiles it. Both
of the two ways in that this audit is built on are gone, not narrowed.

I ran it rather than reasoning about it. The path in `main` resolves again,
to that different tree, and the run exits 2 at the root check with

    only 1 of 12 roots were found among 862 definitions.
    The walk is not reading the tree, so nothing below means anything.

**and that message is now false in the way that matters.** The walk IS
reading a tree -- 120 files, 862 definitions, a live crate's device text.
What it is not reading is the tree its roots came from. The single root that
resolved, `attn_score_fold_heads`, resolved by NAME: it is the one
`__global__` in `attn/attention_flashinfer.cuh` and it happens to spell what
used to be a host entry point. Against the deleted path the same check
printed `0 of 12 roots ... 0 definitions`, which at least announced itself as
a dead path. The failure did not get better, it got quieter.

Nothing has been repointed, and not out of caution: there is nothing to point
it at. A reachability walk needs roots that live outside the graph it walks,
and the live crate's device text has no host callers to serve as them. What
this audit would have to become there -- reachability measured from the row
table, or from `Source`, or not at all -- is a design question, and not one a
path edit can answer quietly.

How it fails is not stable either, because `shim_roots` is still finding a
shim. The note there says which one and why that matters.
"""

import collections
import os
import re
import subprocess
import sys

EXTS = ("cu", "cpp", "cuh", "hpp")

# A DEFINITION closes its parameter list on a `{`. A declaration ends in `;`
# and is not a node -- which is the distinction §38.2 turned on, where
# `gemm.hpp:401` looked like a declaration and was an inline forwarder.
DEF = re.compile(
    r"(?<![A-Za-z0-9_])(?P<name>[A-Za-z_]\w*)\s*\((?P<args>[^;{}]*)\)\s*"
    r"(?:const\s*)?(?:noexcept\s*)?\{",
    re.S,
)
CALL = re.compile(r"(?<![A-Za-z0-9_.])(?P<name>[A-Za-z_]\w*)\s*(?:<[^;{}()]*>)?\s*[(<]")
LAUNCH = re.compile(r"(?<![A-Za-z0-9_])(?P<name>[A-Za-z_]\w*)\s*(?:<[^;{}]*?>)?\s*<<<")
GLOBAL = re.compile(r"__global__[^;{]*?(?<![A-Za-z0-9_])(?P<name>[A-Za-z_]\w*)\s*\(", re.S)

KEYWORDS = {
    "if", "for", "while", "switch", "return", "sizeof", "catch", "do", "else",
    "new", "delete", "throw", "static_cast", "reinterpret_cast", "const_cast",
    "dynamic_cast", "template", "typename", "struct", "class", "namespace",
    "operator", "case", "assert", "decltype", "alignof", "noexcept",
    "constexpr", "using", "typedef", "and", "or", "not", "explicit", "friend",
}

# The `extern "C"` roots that are not shim entries. Each is named by Rust
# somewhere under `crates/driver-cuda/src` or `crates/*/tests`, which is what
# makes it an entry point rather than a leftover.
# THE SEVEN `pie_x_*` PLAN-LIFECYCLE ROOTS ARE GONE. They were
# `pie_x_{make,destroy}_{decode,prefill}_plan`,
# `pie_x_set_decode_plan_int_base` and the two
# `pie_x_plan_attention_flashinfer_{decode,prefill}_bf16` planners, defined by
# `driver-cuda/csrc/attn/plan_lifecycle.cpp` and
# `csrc/attn/attention_flashinfer.cu`. North star §5 step 7 deleted both files
# and the whole of `driver-cuda/csrc/`; their Rust replacements are
# `fire::flashinfer_fa2::{DecodePlanCache, PrefillPlanCache}` and
# `bind::{DecodePlan, PrefillPlan}`, which cross no ABI at all.
DIRECT_ROOTS = (
    "gemma4_audio_encode", "gemma4_vision_encode", "qwen3vl_scatter",
    "set_device_memory_allocator", "set_device_tensor_memory_callback",
    "allocate_device_memory", "free_device_memory", "alloc_logging_enabled",
    "sample_memory_callback",
)


def strip_comments(text):
    """Comments and string literals out, everything else byte-for-byte."""
    out, i, n = [], 0, len(text)
    while i < n:
        if text.startswith("//", i):
            j = text.find("\n", i)
            i = n if j < 0 else j
        elif text.startswith("/*", i):
            j = text.find("*/", i)
            i = n if j < 0 else j + 2
        elif text[i] in '"\'':
            q, j = text[i], i + 1
            while j < n and text[j] != q:
                j += 2 if text[j] == "\\" else 1
            i = j + 1
        else:
            out.append(text[i])
            i += 1
    return "".join(out)


def body_end(text, open_brace):
    depth, i, n = 0, open_brace, len(text)
    while i < n:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return n - 1


def shim_roots(repo):
    """The launcher each `pie_k_*` forwards to, from the generated shim.

    Read from the build directory rather than derived from the tables,
    because the shim IS the routing decision: a symbol in `JIT_DISPATCHED`
    has no entry here, which is exactly what makes its launcher a deletion
    candidate. Deriving the set would reimplement `emit_c_shim`'s filter and
    could disagree with it.

    THE NEWEST ONE, and the first version of this did not say so. Cargo keeps
    a build directory per feature set and per fingerprint, so
    `target/debug/build/` holds many `kernels-cuda-*/out/shim.cpp` at once and
    they are not the same file. Taking the first in `listdir` order read a
    shim from before four rows were routed and three deleted: **225 roots
    instead of 194**, and the audit reported 69 unreachable launches where the
    truth was 97. A stale shim does not fail -- it makes the archive look
    healthier than it is, which is the one direction this tool must not err
    in. `driver-cuda/build.rs` names the same hazard as "the archive being
    older than the rows".

    THE SHIM IT PICKS TODAY IS THE DELETED CRATE'S, and "newest" cannot save
    it, because every candidate is a leftover now. `emit_c_shim` ran in the
    ARCHIVE's build script, so `out/shim.cpp` only ever appeared under that
    crate's build directories -- and those directories survive the crate: I
    found the walk landing on
    `target/debug/build/kernels-cuda-78cb325caadc8e09/out/shim.cpp`, which
    declares three entry points where the paragraph above is arguing about
    194 against 225. The mtime tiebreak was built to keep two live shims apart
    and has nothing to say about a shim whose producer is gone.

    The prefix over-matches as well, and did so before the rename rather than
    because of it. Under the suffix it carried before the rename, the JIT
    crate's package name also began `kernels-cuda-`, so its 68 build
    directories were already candidates here. They never won
    only because `os.path.isfile` rejects them -- that build script wrote
    `api.rs`, `carried.rs` and a `device-headers` tree and no `shim.cpp` at
    all. It writes nothing now: the JIT crate's build script was deleted along
    with `carried.rs` when the header set moved into `src/source.rs`, so that
    crate produces no build directories at all and the ones under `target/`
    matching its name are leftovers that a `cargo clean` ends for good.
    Post-rename its directories are spelled exactly like the archive's,
    so the two are no longer even distinguishable by name; the `isfile` guard
    is now the only thing separating them, and it separates them by accident.

    Which makes the failure mode depend on the state of `target/`. With the
    leftovers present, `main` reaches the root check and fails there. After a
    `cargo clean` this returns `(None, None)` and `main` prints its "build it
    first" line instead -- and that line's command has quietly inverted too:
    `--features native` was the ARCHIVE's feature, the one that ran CMake, and
    the crate answering to `-p kernels-cuda` now declares `default`,
    `cuda-12`, `cuda-13` and `_cuda`. Running it as written gets an error
    about an unknown feature, not a shim.
    """
    build = os.path.join(repo, "target", "debug", "build")
    if not os.path.isdir(build):
        return None, None
    best = None
    for entry in os.listdir(build):
        shim = os.path.join(build, entry, "out", "shim.cpp")
        if entry.startswith("kernels-cuda-") and os.path.isfile(shim):
            stamp = os.path.getmtime(shim)
            if best is None or stamp > best[0]:
                best = (stamp, shim)
    if best is None:
        return None, None
    text = open(best[1], errors="replace").read()
    found = re.findall(
        r"= &::pie_cuda_driver::kernels::(?:[A-Za-z_]\w*::)*(\w+)\s*;", text
    )
    # THE FILE STATES ITS OWN DENOMINATOR, so use it.
    #
    # The guard below this used to be "do the roots I asked for resolve",
    # which is a question about the tree and not about the read. Truncating
    # the root list to 30 passed it cleanly: 30 of 30 resolved, and the audit
    # went on to report 295 dead launches instead of 97 -- a fourfold
    # over-claim, printed with no warning.
    #
    # `emit_c_shim` writes exactly one `extern "C"` per row it keeps and each
    # forwards to exactly one launcher, so these two counts are equal or the
    # regex has stopped matching the emitter's output.
    entries = text.count('extern "C"')
    if len(found) != entries:
        raise SystemExit(
            f"{best[1]} defines {entries} entry points and the forwarding "
            f"pattern matched {len(found)}. The shim's shape has changed and "
            f"this audit is reading a fraction of it -- which UNDER-reports "
            f"reachability and over-reports dead code."
        )
    return (sorted(set(found)) or None), best[1]


def main():
    repo = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    # The archive crate's `csrc/src` -- see the last section of this file's
    # docstring. The path resolves today, to the JIT crate that took the name,
    # and the walk below reads its 120 `.cuh` of device text: a real tree, and
    # not the one the roots came from. It is left pointing where it points
    # because moving it somewhere it would "work" is the one change that could
    # turn this from a loud failure into a wrong answer.
    src = os.path.join(repo, "crates", "kernels-cuda", "kernels")

    roots, shim = shim_roots(repo)
    if roots is None:
        # `--features native` below is the ARCHIVE crate's feature and no
        # longer exists; `shim_roots`' docstring records what the command does
        # if a reader runs it. It is quoted rather than corrected because
        # there is no corrected form -- no crate in the tree generates a
        # `shim.cpp` for this audit to read.
        print(
            "no generated shim.cpp under target/debug/build/kernels-cuda-*/out.\n"
            "Build it first -- this audit's roots ARE the shim, and guessing them "
            "from the tables would reimplement the filter it exists to observe:\n"
            "  cargo build -p kernels-cuda --features native",
            file=sys.stderr,
        )
        return 2

    files = []
    for r, _, fs in os.walk(src):
        for f in fs:
            if f.rsplit(".", 1)[-1] in EXTS:
                p = os.path.join(r, f)
                files.append((p, strip_comments(open(p, errors="replace").read())))

    globals_ = {m.group("name") for _, t in files for m in GLOBAL.finditer(t)}

    defs = collections.defaultdict(list)
    for path, text in files:
        for m in DEF.finditer(text):
            name = m.group("name")
            if name in KEYWORDS:
                continue
            defs[name].append((path, text[m.end():body_end(text, m.end() - 1)]))

    edges = collections.defaultdict(set)
    launches = collections.Counter()
    for name, bodies in defs.items():
        for _, body in bodies:
            for m in CALL.finditer(body):
                callee = m.group("name")
                if callee not in KEYWORDS and callee != name:
                    if callee in defs or callee in globals_:
                        edges[name].add(callee)
            launches[name] += len(LAUNCH.findall(body))

    wanted = set(roots) | set(DIRECT_ROOTS)
    present = {r for r in wanted if r in defs}
    # A ROOT SET THAT SHRANK IS A BROKEN AUDIT, not a smaller archive. If the
    # shim names a launcher this walk never found, the walk is not reading
    # the tree -- and every "unreachable" below would be an artefact.
    absent = sorted(wanted - present)
    if len(present) < len(wanted) * 9 // 10:
        print(
            f"only {len(present)} of {len(wanted)} roots were found among "
            f"{len(defs)} definitions. The walk is not reading the tree, so "
            f"nothing below means anything. Missing: {absent[:10]}",
            file=sys.stderr,
        )
        return 2

    seen, queue = set(), list(present)
    while queue:
        cur = queue.pop()
        if cur in seen:
            continue
        seen.add(cur)
        queue.extend(edges.get(cur, ()))

    defined = set(defs)
    dead = {n: launches[n] for n in defined - seen if launches[n]}
    total = sum(launches.values())

    print(f"csrc reachability: {len(defined)} functions, {len(present)} roots")
    print(f"  shim read            {os.path.relpath(shim, repo)}")
    print(f"  reachable            {len(defined & seen)}")
    print(f"  unreachable          {len(defined - seen)}")
    print(f"  <<<>>> total         {total}")
    print(f"  <<<>>> unreachable   {sum(dead.values())}  in {len(dead)} functions")
    if absent:
        print(f"  roots not defined here  {len(absent)}: {absent}")
    print()
    print("Deletion CANDIDATES -- unreachable functions that launch a kernel.")
    print("Each is a claim about a whole consumer set; §10.10 still applies.")
    for name, count in sorted(dead.items(), key=lambda kv: (-kv[1], kv[0])):
        where = sorted({os.path.relpath(p, src) for p, _ in defs[name]})[0]
        print(f"  {count:3}  {name:52} {where}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
