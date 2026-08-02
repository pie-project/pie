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
   become `Guard`. Slice 4a (DONE 2026-08-02) birthed the Guard on
   llama_like's KV-write mechanism, three-way-parity-proven. Slices
   4b/4c carry the qwen3_5 body.

   **The class geometry (decided, 4b).** The recon found the executor's
   matrix is not 2-dimensional: beyond decode/prefill, the MTP services
   change WHICH OPS RUN. The axes settle as follows:

   - `FireClass` grows to four: `Decode`, `Prefill`, and the two
     service shapes — `CommitAdvance` (the spec-decode repair: ONLY each
     linear layer's conv+prep+recurrence, fed from the verify stash; no
     embed/attention/MLP/epilogue — a genuinely different pass, so a
     genuinely different trace) and `StateOnly` (the whole backbone
     minus the logits epilogue). These are CLASSES, not guards, because
     they change the op list, and the toolchain's unit of specialization
     is the trace. A class per service combination stays bounded because
     the services do not compose (commit_advance excludes state_only by
     construction; the driver asserts it).
   - `verify_frozen` (write_state=false) is NOT a class and NOT a
     guard: the op list is identical and even the kernels are identical
     — it is a PARAMETER of the recurrence/conv launches. It crosses as
     an argument the stated kernel reads at fire time, like `fast_rows`.
   - The N-thresholds (`warp_tiled_max_tokens`,
     `cached_prefill_max_tokens`) become `Guard` predicates
     (`TokensLE(k)` / `TokensGT(k)`) around the stated recurrence
     kernels — the first VALUE-PRODUCING guards: both regions write the
     same output buffer, so the Guard op itself carries the output
     value and the region launches bind it (the design note 4a left
     open, resolved by "the value is the guard's, the regions are its
     lowerings").
   - The legacy slot-less/qo-less paths (`slot_ids_d == nullptr` — the
     parity entry point's host-loop forms) do NOT move into the
     declaration: they exist for a harness, not a deployment. The
     lowered traces state the batched kernels only, and the parity
     entry keeps the semantic trace + interpreter. If the harness path
     ever hits a lowered trace, the loud unknown-kernel throw names it.
5. **Delete** the choice-deriving code once every body() fire has a
   class. Scope, precisely — rung 5 deletes from the DECLARED executors:
   the semantic-walk cascades (the conv/recurrence/attention/KV-write
   choice arms), the hoisted `use_*` booleans, and the
   commit/state-only op filters. It does NOT delete the semantic TRACE
   (parity reference, site summaries, the Metal emitter's input) nor the
   hand-written paged bodies (they serve everything the declared gate
   excludes: hooks, lora, custom masks, TP, quantized projections — the
   fallback tier is a feature, not a leftover).

   **The mask classes (direction set in review, 2026-08-02).** Custom
   masks are the other load-bearing per-fire attachment, and they are
   CLASSES, not guards: a masked decode swaps the attention kernel to
   the custom-mask prefill dispatch AND breaks the fused decode-QKV
   arm's predicate — the op list changes. `MaskedDecode` (wire 5) and
   `MaskedPrefill` (wire 6): the general QKV arm + the fused
   qk-norm+rope + a DISTINCT stated symbol for the masked attention
   (`dispatch_attention_flashinfer_prefill_bf16_masked` — the stash
   pseudo-symbol precedent: same C++ dispatch, different operation,
   because "bind the mask if present" back in the driver would be the
   smarts we deleted). Mask data crosses as runtime args of the stated
   kernel, commit_lens's peer. This EXPANDS the declared gate — masked
   fires fall back to the hand-written path entirely today — and the
   item-A harnesses (naive-masked, attention-sink, sliding-window) are
   the ready-made parity gates. llama_like first, qwen3_5 after.

   **The hook axis (design set in review, 2026-08-02).** The PTIR stage
   programs (Prologue / OnAttnProj / OnAttn / Epilogue) also vary per
   forward pass — and they are NEITHER classes NOR guards. They are the
   third mechanism, the one plan.md's sketch carried from the start:
   `forward(tok, h: dyn Hooks)`. Three reasons, each disqualifying an
   axis: they are PER-LANE (a fire mixes hooked and hook-free lanes —
   done criterion #2 — and a class is per-fire); WHICH program attaches
   is `dyn` (user PTIR, runtime-compiled — unenumerable by a trace,
   the expert axis's peer); and the lowering is the PLANNER's
   (`Prefix{fast_rows}`, derived per fire — criterion #4's territory).
   What the declaration states is the SITES: `h.*` calls become
   `HookSite{stage, layer}` ops whose content is the site's contract
   (what it observes, what it may intervene on — the sideband types).
   What it does not state: the program (sideband data) and `fast_rows`
   (a runtime parameter of the generated code — plan.md Part 3
   verbatim: "its fast-path conditions take a row count where they used
   to take a boolean"; the fused decode-QKV arm's `fast_rows == R` gate
   term becomes that row count). A site with no program attached is a
   no-op by argument, not by branch — write_state's peer — which is
   exactly the condition under which the fused kernel survives on the
   hook-free prefix. Hooked fires stay on the hand-written path (where
   stages 1/6 built all of this by hand) until the HookSite slice, which
   follows rung 5 — it is the largest remaining expansion of the
   declared gate, and it is where the declared world and the
   polymorphic-batching machinery finally meet in one text.

   **The frozen-verify amendment (decided while 4c-iv landed).** The 4b
   geometry called `verify_frozen` a kernel parameter — true for
   `write_state`, but the frozen service ALSO stash-writes (the memcpy
   trio after the in-proj splits), which changes the op list, and by our
   own rule that makes it a CLASS: `FireClass::FrozenVerify` (wire 4) =
   the Prefill body + `verify_stash_store` per linear layer.
   `write_state` remains the runtime argument it already is — the class
   carries the op, not the flag. With it, every qwen3_5 body() fire has
   a class, which is exactly rung 5's precondition. The legacy
   slot-less parity-entry paths leave the declared executor entirely
   (fall back to hand-written) — they were a harness convenience, and
   the harness keeps the hand-written path anyway.

## What this does not reopen

Non-goals hold: no kernel is generated (the DSL *names* existing
kernels); PTIR untouched; planner's divergence lowerings still chosen at
fire time from candidates. The append-only ABI discipline holds — new op
kinds append (21+), variant fields ride param slots serde-defaulted so
every existing golden stays byte-identical.
