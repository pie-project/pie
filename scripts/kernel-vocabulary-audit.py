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
HDRS = list((ROOT/"crates/kernels-cuda/csrc/src/ops").glob("*.hpp")) + \
       list((ROOT/"crates/kernels-cuda/csrc/src/kernels").glob("*.hpp")) + \
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

# what the table declares (strip the C++ namespace the symbol may carry)
tbl_src = (ROOT/"crates/kernels-cuda/src/lib.rs").read_text()
declared = {s.split("::")[-1] for s in re.findall(r'"([a-z0-9_:]+)"', tbl_src)}

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
print("""EXPECTED RESIDUE -- these SHOULD be absent from the table:

  plan_attention_*        host PREPARES. A prepare is what `needs` names, not
  prepare_attention_*     something a trace records; declaring one would make
                          the table claim a statement exists that does not.
  set_stream              a cuBLAS handle setter.
  maybe_bench_*           a benchmark harness.
  lm_head_argmax_chunked  a host-side chunking helper over the real launcher.

Anything else in the list above is a launcher a model fires that no
declaration can state.""")
