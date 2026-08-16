"""The kernel-vocabulary audit, done by NAME rather than by prefix.

The old sweep matched `launch_*` / `dispatch_*` in the model C++ and compared
that against the table. Two things were wrong with it: it missed launchers
whose names start with neither (every `ops::` one that does not), and it
counted C++ identifiers and comment text that happen to match the pattern.

This one goes the other way round. It reads the HEADERS to learn what a
launcher IS -- a declaration whose parameter list takes a `cudaStream_t` --
and then looks for those exact names at call sites. A name that is not a
launcher cannot be a false positive, and a launcher with an unusual prefix
cannot be missed.
"""
import re, pathlib, collections

ROOT = pathlib.Path(".")
# Every header under the archive crate's `csrc/src`, at whatever depth. This used to name `ops/` and
# `kernels/` explicitly; the refactor deleted both directories in favour of one
# per family, and the glob went on matching nothing but `third_party`. The
# audit did not notice -- it found five launchers, compared them against a
# 192-row table, and reported ZERO UNDECLARED. A check that passes because it
# stopped looking is worse than no check, so `rglob` here and the launcher
# floor below exist together.
#
# BOTH PATHS BELOW ARE THE ARCHIVE CRATE'S, and the name has since been reused,
# so the two strings no longer denote what they denoted when they were written.
# `kernels-cuda` was the ahead-of-time crate: it built `libpie_kernels_cuda.a`
# with CMake and nvcc, declared `links = "pie_kernels_cuda"`, and its `csrc/`
# held the `.cu` translation units -- 138 of them until `345ea1ec6` turned
# these kernels into a JIT -- the host `.hpp` these globs were written for,
# and the `third_party/` tree the second one names. `efaad26b4` took the last
# 46 `.cu`, `third_party/` and the `CMakeLists.txt` together; the crate itself
# went at `85c6c674b`, seventeen surviving `.hpp` and all.
#
# What stands at `crates/kernels-cuda` now is the JIT crate that took the
# name, and it is a different tree: 120 `.cuh` of DEVICE text under `kernels/`
# that NVRTC compiles at run time, plus a `shim/` tree of device `.h`. I
# counted zero `.cu`, zero `.cpp` and zero `.hpp` under either; the one `.hpp`
# anywhere in that crate was a vendored `boost/math/ccmath/fabs.hpp` beneath
# `shim/`, outside both globs and not a declaration of anything.
#
# So this audit finds zero launchers and the floor below fires. **That failure
# is correct, and "Fix the glob" is not what it asks for.** A launcher is
# defined a few lines down as a HOST C++ declaration whose parameter list
# takes a `cudaStream_t` or a `cublasHandle_t`, and the live crate has no host
# C++ to declare one in -- a row there names device text and a launch is a
# `cuLaunchKernel` from Rust, so there is no such declaration to find and no
# path under `crates/kernels-cuda` that could supply one. Repointing these two
# globs at it would swap an honest failure for exactly the silent one the
# paragraph above describes. Making this audit answer again means deciding
# what "declared" MEANS for a crate whose device text is compiled at run time,
# which is a design question and not a glob.
HDRS = list((ROOT/"crates/kernels-cuda/csrc/src").rglob("*.hpp")) + \
       list((ROOT/"crates/kernels-cuda/csrc/third_party").rglob("*.hpp"))

def params(text, open_at):
    depth, i = 1, open_at
    while i < len(text) and depth:
        i += 1
        if i < len(text):
            if text[i] == '(': depth += 1
            elif text[i] == ')': depth -= 1
    return text[open_at:i]

launchers = {}
for h in HDRS:
    src = h.read_text(errors="ignore")
    # `inline` and `static` are not decoration here: `ops/gemm.hpp` defines
    # `gemm_batched_act_x_wt_bf16` as an inline forwarder, and requiring the
    # return type to start the line made every such launcher INVISIBLE to
    # this audit -- which then reported families as fully declared while
    # they fired symbols the table had never heard of.
    for m in re.finditer(
        r'^\s*(?:inline\s+|static\s+)*(?:void|bool|int)\s+(\w+)\s*\(', src, re.M
    ):
        p = params(src, m.end() - 1)
        # A launcher issues DEVICE work, and there are two ways to do that
        # here: a raw launch takes a `cudaStream_t`, a cuBLAS-backed one
        # takes a `cublasHandle_t`. `gemm_act_x_w` is the second kind and is
        # already a table entry, so the definition has to admit both.
        if "cudaStream_t" in p or "cublasHandle_t" in p:
            launchers[m.group(1)] = h.name

# The symmetric guard to the one below. `declared` got its floor after an empty
# table made this audit call every launcher undeclared; the mirror failure --
# an empty header scan calling every symbol declared -- then happened anyway,
# because only one side had been fenced. The table is ~192 rows and most rows
# have a launcher, so anything under 100 means the scan lost the tree, not that
# the tree lost its launchers.
#
# It is firing. I ran this and it exits 1 at zero launchers. The tree it was
# guarding is deleted rather than moved, and the paragraph above `HDRS` says
# why the message's "Fix the glob" is the one repair that must NOT be made --
# read it before touching either path.
if len(launchers) < 100:
    raise SystemExit(
        f"only {len(launchers)} launchers found under crates/kernels-cuda "
        f"-- the headers moved and HDRS above no longer reaches them. This "
        f"audit reports nonsense rather than nothing when that happens: too "
        f"few launchers makes every table symbol look declared, and it prints "
        f"a clean bill of health. Fix the glob."
    )

# What the table declares (strip the C++ namespace the symbol may carry).
#
# Every `.rs` in the crate, not just `lib.rs`: the table is one module per
# kernel family now (`attn.rs`, `moe.rs`, ...) and `lib.rs` holds only the
# concatenation. Reading the one file silently produced an EMPTY declared set,
# which made this audit report every launcher in the tree as undeclared.
#
# THIS `crates/kernels-cuda` IS THE LIVE JIT CRATE -- not the archive the two
# globs above name, and one path prefix denoting two different crates within
# one file is the whole trap the name reuse laid. Here the path is right: the
# rows MOVED into the JIT crate rather than being copied there, precisely so
# that one symbol would have one contract, and the archive then re-exported
# them back through an edge in the opposite direction. That inversion is why
# deleting it at `85c6c674b` moved no rows.
#
# What this glob actually reaches, measured rather than assumed: 16 top-level
# `.rs`, 171 symbols, so the floor below does not fire. It does not descend --
# `glob`, not `rglob` -- and five families are DIRECTORIES in that crate now
# (`attn/`, `cascade/`, `comm/`, `gemm/`, `jit/`), so their rows are outside
# `declared`. The parenthesis above still says `attn.rs`, which is the shape
# that file had before it grew a directory. A symbol declared only under one
# of those five would be reported here as undeclared; whether that is worth
# an `rglob` is a live question and not a typo, so it is recorded and not
# quietly changed.
tbl_src = "".join(
    p.read_text() for p in sorted((ROOT/"crates/kernels-cuda/src").glob("*.rs"))
)
declared = {s.split("::")[-1] for s in re.findall(r'"([a-z0-9_:]+)"', tbl_src)}
if not declared:
    raise SystemExit(
        "no symbols found in crates/kernels-cuda/src/*.rs -- the table moved "
        "again, and this audit reports nonsense rather than nothing when that "
        "happens. Fix the glob above."
    )

# what the emitters produce from SEMANTIC ops -- never a `Launch` in a trace
emitted = set()
for f in list((ROOT/"crates/model/src").rglob("emit.rs")) + [ROOT/"crates/model-compiler/src/lower.rs"]:
    if f.exists():
        emitted |= {s.split("::")[-1] for s in re.findall(r'"([a-z0-9_:]+)"', f.read_text())}

MODELS = ROOT/"crates/driver-cuda/csrc/src/model"
per_family = collections.defaultdict(set)
for cpp in MODELS.rglob("*"):
    if cpp.suffix not in (".cpp", ".hpp", ".cu", ".cuh", ".inc"): continue
    fam = cpp.relative_to(MODELS).parts[0]
    if fam.endswith((".cpp", ".hpp")): fam = "(root)"
    src = cpp.read_text(errors="ignore")
    # strip comments: a name in prose is not a call
    src = re.sub(r'//[^\n]*', '', src)
    src = re.sub(r'/\*.*?\*/', '', src, flags=re.S)
    for name in launchers:
        if re.search(r'\b' + re.escape(name) + r'\s*\(', src):
            per_family[fam].add(name)

print(f"launchers declared in headers: {len(launchers)}")
print(f"symbols in the kernel table:   {len(declared)}")
print()
total_gap = set()
for fam in sorted(per_family):
    gap = sorted(n for n in per_family[fam] if n not in declared and n not in emitted)
    if gap:
        print(f"{fam}: {len(gap)}")
        for n in gap: print(f"     {n:52s} {launchers[n]}")
        total_gap |= set(gap)
print()
print(f"DISTINCT UNDECLARED LAUNCHERS ACROSS ALL FAMILIES: {len(total_gap)}")
print()

# ── the expected residue, as data ───────────────────────────────────────────
#
# This used to be one `print("""...""")` of prose ending in "anything else in
# the list above is a launcher a model fires that no declaration can state" --
# an instruction to a human, in a script that then exited 0. It was wired into
# CI as a gate anyway (`.github/workflows/ci.yml`, workspace-verify), so CI
# went green while the audit printed a real gap: `argmax_bf16`, which
# `csm_backbone_forward.cu` had started firing with no row to declare it.
#
# A check whose finding only a reader can act on is not a check. So the prose
# is the data now: every excused name carries its reason here, the report is
# printed FROM this table, and anything outside it exits nonzero.
EXPECTED_RESIDUE = [
    ("plan_attention_", "prefix",
     "host PREPARES. A prepare is what `needs` names, not something a trace\n"
     "records; declaring one would make the table claim a statement exists\n"
     "that does not."),
    ("prepare_attention_", "prefix", None),   # same reason as the line above
    # `kernels-cuda/comm/` in the reason below is the ARCHIVE crate's
    # `csrc/src/comm/`, where `custom_all_reduce.cu` sat until `19a8db2ea`.
    # Read against the crate holding that name today the sentence inverts, so
    # it is worth stating which half survived: the ROWS came down into the JIT
    # crate's `src/comm/mod.rs`, the KERNEL did not. That module resolves the
    # cross product and then declines every point of it, because the flashinfer
    # tree it would need is CPM-fetched rather than vendored and its `comm/`
    # directory is not in `csrc/`. So the excuse below still holds -- no trace
    # names this, the model C++ reaches it as a method on the TP plane -- but
    # it holds for the reason it always did, not because a kernel is waiting
    # somewhere for a row.
    ("all_reduce_", "prefix",
     "a COLLECTIVE, and the shell's to schedule. The kernel moved to\n"
     "kernels-cuda/comm/ because it computes (a reduction fused with a\n"
     "residual add and an RMSNorm), but no trace names it: Rust emits nothing\n"
     "for it, the string dispatcher has no entry, and the model C++ reaches it\n"
     "as a METHOD on the TP plane (`tp->all_reduce_bf16(...)`), chosen by the\n"
     "custom-vs-NCCL fallback policy that stays driver-side. A table row would\n"
     "assert a statement that cannot be written. (These match here only\n"
     "because the scan is by name, and a method call spells the same name as\n"
     "a free one.)"),
    ("can_fuse_residual_rmsnorm", "exact",
     "the predicate of that same custom-vs-NCCL policy."),
    ("set_stream", "exact", "a cuBLAS handle setter."),
    ("maybe_bench_", "prefix", "a benchmark harness."),
    ("lm_head_argmax_chunked", "exact",
     "a host-side chunking helper over the real launcher."),
    ("causal_conv1d_prefill_noact_bf16", "exact",
     "same case as `argmax_bf16` below: fired only from a hand-written\n"
     "forward (`gemma4_audio_forward.cu`'s lconv1d), so no trace records it\n"
     "and a row would have no DSL statement behind it. The activated form\n"
     "`causal_conv1d_prefill_bf16` IS declared -- Qwen3.5 traces that one."),
    ("argmax_bf16", "exact",
     "fired only from a HAND-WRITTEN forward (`csm_backbone_forward.cu`), not\n"
     "a traced one. A row for it was written and `model`'s\n"
     "`the_table_is_exactly_the_dsl_surface` rejected it: that test holds the\n"
     "table and `dsl::cuda` to the same set, and a DSL statement is something\n"
     "a trace RECORDS. Nothing traces this argmax, so the statement would\n"
     "have no caller and the row would claim a surface that is not there.\n"
     "If CSM ever gets a declared forward, this entry is what should go."),
]


def excused(name: str) -> bool:
    return any(name.startswith(k) if kind == "prefix" else name == k
               for k, kind, _ in EXPECTED_RESIDUE)


print("EXPECTED RESIDUE -- these SHOULD be absent from the table:\n")
for key, kind, why in EXPECTED_RESIDUE:
    label = f"{key}*" if kind == "prefix" else key
    if why is None:
        print(f"  {label}")
        continue
    head, *rest = why.split("\n")
    print(f"  {label:26s}{head}")
    for line in rest:
        print(f"  {'':26s}{line}")

# An excused name that stopped appearing is worth saying -- it means the entry
# above outlived what it excused -- but it is not a failure, so it prints.
stale = [k for k, kind, _ in EXPECTED_RESIDUE
         if not any(n.startswith(k) if kind == "prefix" else n == k
                    for n in total_gap)]
if stale:
    print("\nNOTE: nothing matched these entries any more, so they now excuse")
    print("nothing and can go: " + ", ".join(stale))

# `crates/kernels-cuda/src/<family>.rs` in the message below is the LIVE JIT
# crate -- the same referent as the table read above, and NOT the archive the
# `HDRS` globs name. Adding a row there is still the right instruction; it is
# only worth saying which crate it lands in, because this file now spells two
# different crates the same way.
unexpected = sorted(n for n in total_gap if not excused(n))
if unexpected:
    raise SystemExit(
        "\n" + "=" * 72 + "\n"
        f"{len(unexpected)} launcher(s) a model fires with no declaration to\n"
        "state them, and no entry above excusing that:\n\n"
        + "".join(f"    {n:52s} {launchers[n]}\n" for n in unexpected)
        + "\nEither give each a row in crates/kernels-cuda/src/<family>.rs, or\n"
        "-- if it genuinely cannot be written as a traced statement -- add it\n"
        "to EXPECTED_RESIDUE with the reason why.\n" + "=" * 72
    )

print(f"\nAll {len(total_gap)} undeclared launchers are accounted for.")
