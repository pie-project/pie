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
3. **Static C++ emission** — DONE (2026-08-02); COMPLETE AT FULL
   WIDTH (2026-08-03, `31a28eed`): after the class collapse, the
   static form grew to cover every admitted fire — the mask arms, the
   Peel's row-windowed regions, the lora arms (the emitter constructs
   the fire staging and spells each correction's layer as a constant),
   and the hook machinery (sideband preamble, 280 constant-layer
   sites, page-mask brackets, capture publishes). The generated
   dispatch keeps NO per-attachment exclusions: digest match means
   static C++, full stop. Hook A/B through the generated path is
   byte-identical 12/12 under live page eviction. Original rung-3
   record: `emit_cuda.rs` walks the
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

   **HookSite recon findings (2026-08-02, slice begun).** Three facts
   sharpen the slice:
   - The MODEL body has exactly TWO sites, not four: `OnAttnProj`
     (observes q before attention, intervenes through the page-mask
     sink — `invoke_stage_hook` at llama_like.cpp:1260, preceded by
     `page_mask.begin_layer`) and `OnAttn` (post-attention, scores via
     the LayerScoreCapture sideband, :1502). PTIR's Prologue and
     Epilogue run DISPATCH-side around the logits — the post-logit
     divergence plan.md measured as nearly free — so they are not trace
     ops of the forward at all.
   - The incremental parity target is the ALL-HOOKED fire
     (fast_rows == 0): the hand-written path runs the general unfused
     sequence for every row plus the stage rings — exactly a
     `HookedDecode`/`HookedPrefill` class trace (general QKV arm + the
     fused qk-norm+rope, which is hook-independent + HookSite ops). The
     MIXED fire (0 < fast_rows < R) needs the `Peel` op — loop peeling
     as vocabulary: two regions that BOTH run, over complementary row
     ranges, `fast_rows` the runtime split — and is its own increment.
   - Open recon before the driver wiring: the hooked decode's attention
     kernel under a page-mask intervention (which dispatch consumes the
     narrowed page list) — the hooked classes must state it.

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

   **The class-collapse amendment (the A-ladder; direction set in
   review 2026-08-03).** The wiki review's sharpest finding: classes
   MULTIPLY (mask × hook × service × family), which re-derives the
   partition-and-batch enumeration this plan set out to beat.
   Masked×2/Hooked×2 are whole-trace-granularity treatments of deltas
   the vocabulary can express at op granularity. The mask-classes
   decision is REVERSED: "a masked decode is not a decode" stays true
   op-list-wise, but the delta is LOCAL (the attention site + the
   fused-QKV arm), and a local op-list delta over a runtime input is
   exactly Guard territory. The ladder:
   - **A1 — DONE (2026-08-03)**: `GuardPred::HasCustomMask` (wire 4);
     Decode/Prefill traces carry the mask arm as a guard chain at the
     attention site — in the fused_post deployment the mask arm holds
     the whole general QKV sequence, so its nested HasWriteDesc guard
     makes guard NESTING part of the vocabulary (the walk keeps a skip
     STACK, the emitter recurses; the aux wire encoding is unchanged —
     a nested guard is just an op inside a region). Masked classes
     deleted; wire values 5/6 answer InvalidArgument (append-only ABI
     keeps the numbers). The rope-table hoist stays unconditional in
     the fused deployment — a masked fire launches one table build it
     never reads (outputs unaffected; the ops-count line shows it).
   - **A2 — DONE (2026-08-03)**: same collapse for Hooked×2 via
     `GuardPred::HasStageHooks` (wire 5). Better than the sketch: the
     sites are not unconditional ops — they live INSIDE the hooked
     arm's region, so an unhooked fire's walk never reaches them (the
     per-fire launch-list truth is structural). The attention chain
     per layer: [HasCustomMask → custom | HasStageHooks → sites +
     side-effect WantsAttnScore guard | else → plain/fused]; the
     fused arm moved to the chain's else. The generated static form
     transliterates the hooked arm as an honest REFUSAL (throw at the
     first HookSite/capture) and its dispatch keeps hooked fires on
     the interpreter walk — extending the static form to the hook
     sideband machinery is its own later increment. Nine classes are
     five: Decode (788 ops), Prefill (619), CommitAdvance, StateOnly,
     FrozenVerify. Parity: 3-leg token parity, sink A4 OFF/ON/GEN,
     hook A/B 12/12 byte-identical (0 fallbacks — every hooked fire
     walks the collapsed traces), forward 54+16+regen-clean, engine
     394, metal 8/8.
   - **A3 — DONE (2026-08-03)**: `OpKind::Peel` (wire 26; region
     lengths in param0/param1, the split NEVER in the trace — it is
     the fire's `fast_rows`, a runtime input). Better than the
     sketch: the Peel does not live inside a hooks arm — it DISSOLVES
     the HasStageHooks arm entirely. The one else-arm body serves
     every hook composition: the packed GEMM full-N, the Peel
     splitting its postprocess (fused epilogue over rows
     `[0, fast_rows)`, general split/norm+rope/write over
     `[fast_rows, N)` at absolute offsets — the hand-written mixed
     fire launch for launch), then the sites (argument no-ops when
     unhooked, early-out on null hooks) and the WantsAttnScore-guarded
     attention, all full-N. `fast_rows == N` is the classic fused
     fire, `0` the all-hooked one; the gate now admits MIXED fires
     (only hooked+masked stays hand-written — the mask arm carries no
     sites). GuardPred::HasStageHooks (wire 5) is retired vocabulary
     after one rung of service — reserved, unstated. The interpreter
     binds a row window (start/len) set by Peel region events; the
     emitter derives `fast_rows` from the hooks argument and spells
     both regions as `if (fast_rows > 0)` / `if (fast_rows < N)`
     blocks with offset pointers. Decode 760 ops / Prefill 563.
     Parity: 3-leg token parity; sink A4 OFF/ON/GEN; hook A/B 12/12
     byte-identical; MIXED fires observed walking the declared Peel
     live (`N=4 R=4 fast_rows=2/3` decode, `N=148 R=4 fast_rows=3`
     prefill co-batches, ALL_OK liveness both gates — mixed
     compositions are not batch-deterministic, so the mixed gate is
     engagement + liveness + the solo byte-parities, the stage-2
     discipline); engine 394; metal 8/8.
   - **A4 — DONE (2026-08-03)**: qwen3_5's hooks are in scope —
     narrower than the sketch, because recon narrowed the target:
     qwen3_5's hand-written sites are OBSERVATION-only (all four
     `invoke_stage_hook` calls pass no mask sink and no score
     sideband: GDN layers observe the prep's fp32 q_pre, full-attn
     layers the roped bf16 q), so no guard is needed at all — the
     sites ride the lowered class bodies directly (argument no-ops,
     null-hooks early-out), including the commit-advance replay
     (which passes through both invokes before its early return).
     The hooks fallback term is deleted from the qwen3_5 gate. And
     custom masks turn out NOT to exist for qwen3_5: the hand-written
     body IGNORES `mask_d` entirely (commented-out params — a masked
     qwen3_5 fire runs unmasked today), so there is no semantics to
     declare; the mask fallback term stays as the honest record of
     that gap. Parity: qwen3.5-0.8B live A/B short+long
     byte-identical across the gate (51 declared fires, 0 fallbacks);
     llama 3-leg parity re-green; forward 54+16+regen; engine 394;
     metal 8/8.
   End state: `FireClass` = fire SHAPE × SERVICE only (Decode,
   Prefill, CommitAdvance, StateOnly, FrozenVerify) — the axes that
   change the pass wholesale; per-fire attachments (masks, hooks,
   lora) are guards, sites, and channel data. This is also the
   region vocabulary the future union pass (supergraph merge) emits
   into, which is why the collapse precedes it.

   **B — DONE (2026-08-03)**: the planner's first consumed lowering.
   `Prefix{fast_rows}` converts to wire rows and crosses the ABI
   (`planned_hook_free_prefix_rows`); the driver cross-checks it
   against its compiled-plan derivation (refusing on drift) and
   feeds the declared Peel's split. Live: planned=0 ×276
   (all-hooked), planned=2/3 ×48 (mixed), 0 refusals.

   **C — the fire census verdict (2026-08-03).** `PIE_FIRE_CENSUS=1`
   prints one line per sealed step group (size, the head's solo
   contract, join refusals by clause — `LaunchGrouping::refusal` and
   `solo_reason` are the reason-carrying twins of the old booleans).
   Measured over a realistic mix (mixed hooked fires, staggered
   dense-masked lanes, solo hook alternation, k=3 perf A/B):
   - ZERO solo-contract fires — the remaining `requires_solo` terms
     (rs-buffer: stage-2 contract verdict; prebuilt-untracked:
     harness; multirow-zero-tokens) never fire in real work.
   - The ONLY join refusals are `mask-compose` (303 events): a
     DENSE-masked lane vs device-resolved decode envelopes — the
     residual of stage 2's item A, which relaxed exactly the
     structured-mask subset. The relax path is a DRIVER capability
     (per-lane wire-mask packing on the composed path), with this
     census as its measured target; nothing scheduler-side remains
     to loosen.
   - Carrying a hooked lane in a mixed fire costs the plain lanes
     NOTHING (naive slowdown 1.00×/0.96×/0.90× across reps — the
     Peel + planner ordering did their job).
   The supergraph-ladder implication, recorded honestly: the class
   collapse has EATEN most of the union pass's original unblocked
   payload — per-fire attachment divergence now co-batches inside
   one fire, so D's remaining targets are the dense-mask compose
   capability above and the externally-blocked structural axes
   (multi-checkpoint serving, depth/MoD, vision). D stays
   measurement-gated on a workload that actually carries one of
   those.

## What this does not reopen

Non-goals hold: no kernel is generated (the DSL *names* existing
kernels); PTIR untouched; planner's divergence lowerings still chosen at
fire time from candidates. The append-only ABI discipline holds — new op
kinds append (21+), variant fields ride param slots serde-defaulted so
every existing golden stays byte-identical.
