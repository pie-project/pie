# North-star DSL — the declaration and its lowering are one text

Date: 2026-08-02. Direction set in review: **which kernel fires must be
stated in the DSL, and the driver must be dumb**. The forward pass as
written in the DSL must be statically convertible to the C++ the driver
runs — no semantic choice may live on the C++ side.

## The correction this encodes

plan.md line 141 ("Op→kernel selection belongs to the backend") was
implemented as "the driver C++ executor may choose" — and the smarts
accumulated there: decode/prefill arms, the fused-QKV peephole and its
eleven-term predicate, XQA eligibility, GDN's three-way lowering,
plan-cache thresholds (`declared_forward.cpp`, both families). That reading
is wrong. plan.md lines 146-148 already state the north star: *"the
declaration decides which existing kernels are called, in what order, with
what indirection."* The backend is not the driver's freedom; it is part of
the DSL text.

And there are not two blocks (declaration + separate lowering rules).
There is **one text**: the model file states the computation and the
kernel selection together, the way the hand-written pass always did — the
DSL is that pass, made declarative.

## The surface (v2, set in review 2026-08-02)

Two corrections from review shaped the surface, both now built (`dsl.rs`):

**One text, not two blocks.** The first design split declaration from a
`backend!` rules block; review rejected the split. The class arms live IN
the forward text, beside the fact arms, same `match`.

**Raw kernel signatures, not enum tags.** A lowered arm calls a function
named for the launcher symbol whose parameters are the launcher's
semantic operands (`cuda::attention_xqa_decode(&q, &w.kv, ..)`), not a
generic op carrying a kernel enum. The trace records ONE generic
`Launch { kernel, weights, state }` op for any stated kernel — so the
ABI stops growing per kernel: the driver resolves the symbol in a
name→launcher registry, and adding a kernel touches no enum anywhere.

The surface is plan.md's sketch made real: values carry the tape (free
functions, no builder threading), weights are typed handles from a
per-layer namespace (`w.qkv` — no strings, no widths in declarations),
state is an object (`w.kv.append(k, v)`), and `y += matmul(&a,
&w.o_proj)` IS the beta=1 cuBLAS fold (the tape rewrites the just-recorded
matmul, id-neutrally) while `y += anything_else` is the explicit
ResidualAdd landing. Op layer tags derive from what an op touches; the
semantic goldens pin that this reproduces the old bracketed tagging byte
for byte. Once-per-fire launches are VALUES, not latches — the rope
table is built once in the prologue and consumed by every fused-QKV
launch as an operand.

## The mechanism: fire class is just another trace-time match

The eDSL's one trick already covers this. Static facts resolve at trace
time because the declaration is Rust running at model load. A fire's
**class** (pure decode / prefill-shaped) is the one input that varies
after load — so the toolchain traces the declaration **once per class**,
and inside the declaration, class arms are ordinary Rust `match`es
alongside the fact arms:

```rust
// ONE text: computation and kernel choice together.
let packed = t.matmul(x, &w("qkv"), q_w + 2 * kv_w);
let q = if class == Decode && facts.fused_qkv && facts.qk_norm == PerHead
        && cuda.decode_fused_post {
    // the fused post kernel, stated — not pattern-matched back in C++
    t.qkv_decode_fused_post(packed, &w("q_norm"), &w("k_norm"), l, facts)
} else {
    let (q, k, v) = t.split_qkv(packed, q_w, kv_w);
    /* per-head norms, rope, kv_append — the general arm */
};
let attn = t.attention_with(l, q, q_w, match class {
    Decode if cuda.xqa_decode => AttnKernel::XqaDecode,
    Decode                    => AttnKernel::FlashinferDecode,
    Prefill                   => AttnKernel::FlashinferPrefill,
});
```

Consequences, in order of importance:

- **The traced form of a class IS the launch form.** Every op names its
  kernel (directly, or 1:1 via variant fields). Static conversion to C++
  is a transliteration of the op list; per-class branches in generated
  code come from the per-class traces, and every branch traces to a line
  of the declaration.
- **The general arm is the semantics.** The fused arm asserts equivalence
  to it; the parity harness tests that assertion. Nothing semantic is
  lost by fusing in the text.
- **Backend facts are facts.** XQA eligibility (env + head geometry +
  page size + window + all-full-attention), `decode_fused_post_enabled`,
  cache format — all load-time, all live in a `CudaFacts` struct beside
  the model facts, provenance-pinned like every other fact.
- **Runtime scalars get a `Guard` op** (emitted as `if (N > k)` in
  generated code) — the only branch that survives into a class trace.
  Not needed for llama_like; qwen3_5's prefill tiling thresholds will
  need it.
- **What stays runtime** (per plan.md Part 2, unchanged): the *values* of
  `dyn`, and the planner's per-fire divergence lowering choice
  (Uniform/Prefix/PerLane) — generated code takes `fast_rows` as a
  parameter, it does not choose it.

## The dumbness criterion

> The driver never chooses between two kernels for semantic reasons.
> Every choice is spelled in the program it received.

The driver keeps: kernel implementations, memory/arenas, streams/events,
graph-capture mechanics, and mechanical SSA-value→buffer binding (register
allocation is not a semantic choice). The driver loses: arm selection,
peepholes, eligibility predicates, thresholds.

## Migration (each rung parity-anchored, hand-written arms deleted last)

1. **forward/**: `FireClass`, kernel-granular vocabulary (`Attention`
   kernel variant, `QkvDecodeFusedPost`), `CudaFacts`, and the lowered
   llama_like — one function, class param, arms as above. Per-class
   goldens (`qwen3_0_6b.decode.json` / `.prefill.json`).
2. **driver interpreter goes dumb**: `declared_forward.cpp` consumes the
   class trace — the peephole matcher, `use_*_path` booleans, and the
   fused predicate are deleted; the switch dispatches on op kind + stated
   variant only. Parity: byte-identical to the current executor on the
   full battery (which is itself byte-identical to hand-written).
3. **Static C++ emission** — DONE (2026-08-02): `emit_cuda.rs` walks the
   class traces and writes `generated/qwen3_0_6b.inc` (committed,
   regeneration-clean-tested) — 4.5k lines of straight-line C++, one
   statement per op, the XQA-or-not question answered at emission, the
   layer loop unrolled to `w.layers[17]`. The driver runs it under
   `PIE_DECLARED_FORWARD_GENERATED=1` iff its live facts digest equals
   the constant the file embeds; mismatch falls back to the interpreter,
   loudly. Parity proven three ways on L40S: hand-written ≡ interpreter ≡
   generated, byte-identical. The digest mechanism paid for itself on its
   first run: the "measured" cuda-facts fixture had guessed xqa=true and
   tied=true; the live digest said xqa0/te0, and the mismatch print — not
   a human — caught it. The remaining runtime `if`s in the generated file
   (has_write_desc, compact logits, gate_up binding) are transliterated
   interpreter arms over runtime INPUTS, listed in declared_forward.hpp.
4. **qwen3_5**: same treatment; the 16 executor arms and the GDN
   three-way choice move into the declaration's class arms; thresholds
   become `Guard`.
5. **Delete** the hand-written arm code and the semantic-only trace path
   once every consumer reads class traces.

## What this does not reopen

Non-goals hold: no kernel is generated (the DSL *names* existing
kernels); PTIR untouched; planner's divergence lowerings still chosen at
fire time from candidates. The append-only ABI discipline holds — new op
kinds append (21+), variant fields ride param slots serde-defaulted so
every existing golden stays byte-identical.
