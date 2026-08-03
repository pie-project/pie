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

## Dense-mask compose — the pinned design (2026-08-03, pre-implementation)

The census's one remaining scheduler refusal (`mask-compose`) and the
masked+hooked arm's missing producer share one capability: let a
wire-BRLE-masked lane ride a composed device-geometry batch. The seam
is `frame.cpp`'s composed-path throw ("non-causal wire masks cannot
ride..."), and the constraint that forces the shape of the fix is that
a device-resolved lane's TRUE kv length under run-ahead lives in
`kv_len_device` — device-resident — so host-side causal synthesis for
those lanes is not generally exact. Therefore a HYBRID DEVICE-SIDE
pack, extending item A's machinery (`pack_dense_mask.cu`):

1. **Wire lanes**: host-decode their BRLE rows against their (exact,
   wire) geometry — `brle::decode` restricted to the wire prefix —
   and stage the bit-packed bytes per lane.
2. **Device lanes**: structured `Causal` params (item A's vocabulary),
   packed device-side against the resolved `klen`.
3. **One `mask_indptr` over all lanes, computed DEVICE-side**: a small
   prefix-sum kernel over per-lane byte sizes (wire lanes' sizes are
   host constants passed in; device lanes' derive from klen), with the
   packed buffer allocated to the page-capacity upper bound.
4. **A scatter-copy kernel** placing the wire lanes' staged bytes at
   their device-computed offsets; `launch_pack_structured_mask` (or a
   variant taking the device indptr) fills the device lanes.
5. The staged pair feeds the SAME custom dispatch the HasCustomMask
   arm already states — no trace change, no new dispatch.
6. **Scheduler relax, LAST** (after the driver path is live-proven on
   a forced composition): drop `wire_mask_on_device_geometry` and the
   `has_user_mask × device_geometry` crosses for WIRE-mask lanes;
   dense DEVICE masks (no BRLE rows, device content) remain solo by
   nature.
Verification: the census workload (maskmix + h2o) should then show the
mask-compose refusals gone and mask=1 hooked=1 fires walking the
declared/generated paths; stage-2's 1.8–2.3× two-wave penalty is the
measured stake.

## Release-build A/B measurement — the walk overhead note (2026-08-03)

Question: does the declared-forward host walk (interpreter) or the
generated static form cost or save measurable wall-clock vs the
hand-written path, in an OPTIMIZED build? All prior parity work ran
debug builds, where host overhead is exaggerated and no perf claim is
honest.

Protocol: release build (`cargo build --release -p pie-bin --features
driver-cuda`), Qwen3-0.6B on the L40S, `mixed_fire_perf.py` (4
concurrent naive decode lanes × 48 tokens = pure; + hooked lane =
mixed), 3 reps per leg, three legs: OFF (hand-written), ON
(interpreter), GEN (generated .inc, digest-gated).

Result: **indistinguishable within rep noise.** Pure walls span
0.22–0.35 s across ALL legs with no leg-correlated ordering (each leg
contains both the fastest and the slowest reps); the mixed-lane
"naive slowdown" ratios scatter 0.74–1.31× with no leg trend.

Reading, honestly bounded:
- At this scale (0.6B, N≤5 rows, 48-token decode legs) kernel time
  dominates; the per-fire host walk — even the interpreter's
  guard-skip/window bookkeeping — is not the bottleneck, and the
  generated form's compile-time fact resolution buys no measurable
  wall-clock here.
- Therefore the generated form's PROVEN value is structural (the
  digest-gated static form, four caught fact-lies, the emitted code as
  reviewable artifact), not throughput. Any "supergraph saves X%"
  claim stays unearned until a workload where host walk shows up
  (many small fires, large N with short legs) is measured.
- What this DOES retire: the standing worry that the interpreter walk
  taxes the hot path. In release, on this hardware, it doesn't.

Follow-up (same day): the "workload where host walk shows up" was then
built and measured — `walk_stress_perf.py`, single-lane 256-step decode
(kernel-minimal fires, host-walk fraction maximal) and 16-lane 64-step
co-batch, 3 reps per leg, same release build. Single-lane steps/s:
OFF 210–218, ON 209–217, GEN 214–230 — rep ranges overlap; 16-lane
scatter 137–182 across all legs, no leg trend. So even at the walk-
heaviest point this engine reaches, the interpreter tax is unmeasurable
and the generated form buys no wall-clock. The perf claim ledger
closes: supergraph value on this hardware/scale is structural, and the
stage-2 wave-count argument (fewer FIRES via co-batching, not cheaper
walks) remains the only measured throughput stake.

## Dense-mask compose — first live producer, and what it taught (2026-08-03)

The compose relaxation finally met a producer. `naive-masked` grew
`mask_mode="dense-prefill"`: the semantically-causal host mask moves to
the PREFILL chunks (wire-geometry fires → wire BRLE rows — the exact
lane shape the C-relax admits), decode runs unmasked. Live on the L40S:

- **Semantics**: solo `none` vs `dense-prefill`, same seed → BYTE-
  IDENTICAL text. The prefill mask is proven causal-equivalent.
- **Compose**: 6 masked lanes raced against 2 decoding baselines —
  census shows ZERO mask-compose refusals and repeated members=3
  steps; the composed fire prints `[declared-forward-generated]
  N=52 R=3 decode=0` (50 wire prefill rows + 2 envelope rows walking
  the GENERATED path).
- **BUT the assembly path is still unexercised**: `brle::is_pure_causal`
  recognizes the causal wire rows and elides the mask before the
  composed-assembly branch (`frame.cpp`'s else-if) is reached. The
  composed batch runs the plain path — correct, and exactly what the
  elision is for. A producer for the ASSEMBLY needs a genuinely
  non-causal wire mask (e.g. a causal-minus-one-column "holed" mode):
  the next rung of this thread.

### The three-day-old trap this run sprang: cross-leg text A/B is
### invalid under composition races

The mixed workload's baseline texts differed OFF vs ON vs GEN —
deterministic per leg, reproducible across boots. It looked like a
parity breach in the composed prefill+envelope fire. It is not. The
census member fingerprint (added this session: per-member
`logical_fire_id × rows`) shows the legs compose DIFFERENT fires:
ON's sequential-lane prefill lands at baseline fire ~91, GEN's at ~33
and ~103 — the interpreter's slower host walk shifts WHEN the racing
prefill joins the decode stream. Different composition → different
reduction shapes → legitimate near-tie sampling flips. Solo masked
lanes: ON == GEN byte-identical. All-lockstep compositions (the prior
batteries' shape): ON == GEN byte-identical. The rule, recorded for
every future battery: **a cross-leg byte-parity claim requires equal
census fingerprints; when a workload races a sequential lane against
a decode stream, compare fingerprints first and texts second.**

Follow-up, same day — the ASSEMBLY got its producer too.
`mask_mode="dense-prefill-hole"` knocks column 1 out of the causal
envelope (rows p >= 2), so `is_pure_causal` cannot elide it and a
composed batch is FORCED through frame.cpp's wire-mask assembly
branch. Live, both legs:
- Interpreter: `N=52 R=3 mask=1` (holed prefill + 2 envelopes) and
  `N=150 R=3 mask=1` (THREE wire-masked prefills co-batched with each
  other) — the custom dispatch engaged on composed batches.
- Generated: 2× `N=52 R=3 decode=0` through the generated custom-mask
  arm; solo holed tokens ON == GEN byte-identical.
- Correctness oracle, composition-invariant by construction: every
  raced lane's max_tokens=1 output depends only on its own prefill
  logits, so SOLO token == MIXED token is the assembly-correctness
  signal. All 6 lanes match, on both legs.
The dense-mask compose design (the pinned section above) is now fully
live-proven end to end: scheduler relax → composed assembly → custom
dispatch, interpreter and generated. What remains honest: dense DEVICE
masks stay solo by nature, and no REAL policy inferlet ships a
non-causal wire mask yet — naive-masked is the measurement instrument
standing in for one.

Second follow-up — the holed producer swept the full deployment
matrix (2026-08-03, same session). The composed wire-mask assembly +
custom-dispatch proof is not a qwen3-0.6b artifact:
- **OLMo-2-1B** (post-norm, unfused, GENERATED inc): solo == mixed
  6/6 on interpreter AND generated legs, solo ON == GEN; composed
  `N=52 R=3 mask=1` ×2 and the three-prefill `N=150 R=3 mask=1`. The
  post-norm generated custom-mask arm had never run with a real
  non-causal mask before this.
- **Phi-3-mini** (interpreter): 6/6, `N=58 R=3 mask=1` ×4 + `N=168
  R=3 mask=1`. **Mistral-7B** (interpreter): 6/6, `N=56 R=3 mask=1`
  ×5 + `N=162 R=3 mask=1`. Zero errors anywhere.
Also probed and closed a question: qwen3_5's recorded custom-mask gap
(hand-written ignores `mask_d`) is UNREACHABLE today — naive-masked's
attention shape fails recurrent-state binding on the hybrid model
before any fire is built, so no current surface can deliver a custom
mask to qwen3_5. The recorded-gap status (fallback reason + note, the
same pattern as lora) is the honest state; a fail-loud would be dead
code. Separately confirmed the two write-descriptor conventions in
naive-baseline (`p / page_size`) and naive-masked (`pool_ids[...]`)
are the SAME convention: `reserve()` returns LOGICAL per-working-set
page indices (0..k for a fresh set), so `pool_ids[i] == i` — no trust
gap in the parity workhorses.

## HasStageHooks (wire 5) — the retirement disposition (2026-08-03)

Question recurring on the hygiene list: the pred is retired vocabulary
(A3 moved hooks to Peel row windows + sites), yet the emitter and the
interpreter both still carry an arm for it. Keep or remove?

**Keep, deliberately.** Both arms evaluate `hooks != nullptr` — the
semantics a trace stating the pred would want are still exactly what
the arms answer, so a resurrected or replayed old trace gets a CORRECT
walk, not drift. Removal breaks wire compatibility for zero payload;
converting the arms to throws would turn a correct answer into an
error. The discriminant stays reserved, the arms stay correct, the
comments already say "retired since A3". This closes the last item on
the residual-hygiene list; the vocabulary is stable as documented.

## Consolidated sweep at tip `5a17d4fac` (2026-08-03)

Unit tier: forward lib 54, goldens 18, regen pins, engine 395,
tokenizer, ABI layout — all green. Live tier (release build,
Qwen3-0.6B): fresh OFF vs GEN short+long BYTE-IDENTICAL;
trackb-snapkv + trackb-h2o hook workloads on GEN clean (28 layers
observed per fire, zero NaN, page masses sane). Everything since the
last consolidated stamp (census fingerprints, the two naive-masked
producer modes, doc notes) holds together.

## The qwen2_5 rung — attention biases enter the vocabulary (2026-08-03)

The blocked-board watch found an unused checkpoint in the cache
(Qwen2.5-1.5B-Instruct) carrying a fact axis no deployment had
exercised: attention biases. The rung, landed at `5c08a3e5e`:
- `OpKind::AddBias` (wire 27): broadcast bias add on the raw
  projections, stated after the lora guard and before norms/rope (the
  hand-written `maybe_add_bias` position; the lora-vs-bias ORDER
  matters — the adapter delta lands on base, not base + bias).
- `LlamaLikeFacts.qkv_bias` (serde-defaulted, append-only); digest
  grows the `qb` term in BOTH printers; existing incs regenerated.
- The build gate's bias refusal became a bound-tensor check;
  `decode_fused_post` carries `!use_qkv_bias` explicitly in both the
  family predicate and the driver derivation.
- Surprise second axis: qwen2_5 is the first FORCE-PREFILL deployment
  through the walk (GQA 6 outside the flashinfer decode set, XQA off
  live). The hand-written body's final else runs a PLAN-LESS prefill
  launcher; the executor's stated-prefill case now mirrors that
  fallback instead of throwing ("prepare built no plan" was the
  first live failure — caught at model load by the stated-kernel
  validation, exactly where the design wants drift caught).
Live: OFF vs ON byte-identical short+long, 53 declared fires. Goldens
pin semantic + lowered forms (xqa0/dfp0/rt1/fpp1). NEXT: the generated
inc (emitter AddBias arm + plan-less prefill fallback emission +
qwen2_5_1_5b write_inc + digest dispatch + GEN parity).

Generated leg, same day (`a4fffd177`): the emitter's AddBias arm
resolves buffer/width at emission (168 constant-layer launches), and
the force-prefill decode class emits the PLAN-LESS prefill launcher
directly — the static form makes at emission the choice the
interpreter defers to a runtime null-check, which is rung 3's whole
argument in one arm. qwen2_5_1_5b.inc (11.2k lines) joined the digest
table and the facts guess matched live on first boot (no fifth catch).
Live: OFF == GEN byte-identical short+long; the holed battery passes
through the generated custom-mask arm WITH biases (solo == mixed 6/6,
composed and three-prefill masked fires); qwen3 GEN sanity unchanged;
existing incs regenerate byte-identical. THREE llama deployments now
run generated at full width (qwen3_0_6b, olmo2_1b, qwen2_5_1_5b) plus
the qwen3_5 hybrid — four static forms, one digest mechanism.

## Mistral + Phi-3 generated legs — the matrix closes (2026-08-03, `6a1822a6f`)

Mistral-7B-v0.3: zero emitter work (fused/no-qk-norm arms existed);
digest matched first boot; OFF == GEN byte-identical at 7B.

Phi-3-mini: the last emission axis — the PADDED head dim.
`LlamaLikeCudaFacts.head_dim_padded` (appended, digest `pad` term,
driver derivation `cfg.head_dim != cfg.head_dim_kernel`); the emitter
resolves at emission what the interpreter resolves per fire: dk
staging aliases, the `1/sqrt(d)` softmax override, pad staging around
both KV-write forms, the post-attention strip after every attention
arm. XQA×padding and fused-post×padding are emission REFUSALS (no
deployment, no reference — refuse rather than guess). Live: OFF == GEN
byte-identical; the holed battery passes the generated PADDED
custom-mask arm 6/6.

**Every checkpoint in the cache now runs a digest-gated static form**
(qwen3_0_6b, olmo2_1b, qwen2_5_1_5b, mistral_7b_v03, phi3_mini +
qwen3_5_0_8b hybrid). Six emitted texts, one digest mechanism, five
axes it verified live this arc (xqa, dfp, wt/cm, te, and today's
first-boot matches for qb/fpp/pad). Rung 3's claim — "the declared
form is statically convertible to C++ at full width" — now has no
untested deployment in this environment left to test it on.

## The supergraph directive (2026-08-03) — unionized, no compromise

Directive (user, this date): go to the FULL supergraph; the unionized
supergraph is a non-negotiable condition; fix fallout after. This
supersedes the measurement-gated stance for this thread — the
measurement happens after the thing exists.

**What "union" means here.** Today's graph reality fragments on THREE
axes: the `ForwardGraphKey.variant` bits (mask/layout/spec), the hook
exec partitions (per program-set, fingerprint-guarded, churn-banned),
and outright eager fallbacks (lora fires, most attachment combos —
`forward_graph_replay_eligible`'s exclusion list). The supergraph
folds ALL of that into ONE conditional graph per (R, N) bucket: the
guard vocabulary (HasWriteDesc / WantsAttnScore / HasCustomMask /
HasLora, nested) becomes CUDA conditional IF/ELSE nodes whose
predicates a graph-embedded kernel reads from a DEVICE-resident aux
word, and every attachment combination replays through the same exec.
The (R, N) bucketing stays — kernel grid dims and scalar args are
baked at capture, which is CUDA-graph physics, and the request lattice
already handles it; the union axis is ATTACHMENTS, exactly the axis
the class-collapse amendment (A) reduced to five predicates.

**Proven premises** (tools/supergraph-poc, run on this box):
conditional IF/ELSE insertion during stream capture; device-read
predicates via `cudaGraphSetConditional` inside the graph; one exec
serving both arms across replays. CUDA 13.0 / driver 580 — all green.

**The ladder:**
- S1 ✓ — PoC + this design.
- S2 — the emitter's third mode: `..._supergraph_build(...)` per
  deployment × Decode class — the generated text emitted AS a capture
  builder (straight-line segments captured; each Guard emitted as
  conditional-node insertion + body-graph capture of its arms; the
  set_cond kernels at graph head consuming the device aux word).
- S3 — driver integration: the fire's aux/pred word moves to a device
  buffer the replay path updates; `capture_forward_graph_exec` gains
  the supergraph variant; the graph cache key drops the folded axes
  (variant mask bit, hook partitions) for supergraph-eligible fires.
- S4 — batteries (every attachment combo × every deployment, replay
  vs eager byte-parity) and the fallout-repair pass the directive
  orders LAST, including the axes the union cannot yet eat:
  * Peel row windows (`fast_rows` bakes into kernel args — device-read
    row windows are a kernel-surgery campaign; until then a mixed
    hooked fire replays the all-hooked arm or falls out),
  * flashinfer plan stability inside conditional bodies
    (page-count-independent plans — the page-mask compact precedent),
  * lora staging (host-driven apply inside a captured body needs the
    grouped-GEMM form to be capture-safe).

S2 landed (`1acb4ca03`): the emitter's third mode emits
`..._supergraph_build` for all five llama deployments — guards as
conditional nodes (mask slot 4, write-desc slot 0, chains nested into
else bodies), Peel as its endpoint conditionals (7/8), `stream` a
mutable local rebound per body boundary so launch text stays identical
to the plain fn's. WantsAttnScore and HasLora arms are explicitly
OUTSIDE the union (host-driven, capture-hostile — S4 names them);
their fires stay eager via eligibility. Existing plain emissions
byte-stable; +31k lines of committed builds; driver compiles; regen
pins green; live GEN byte-stable. NEXT — S3, driver integration:
1. a `supergraph_preds` device buffer on the persistent inputs
   (9 slots, uploaded per fire from the launch's aux/attachment bits +
   the two Peel endpoint bits);
2. an IModel/forward-fn hook exposing the build fn to
   `capture_forward_graph_exec` (the capture wraps
   cudaStreamBeginCapture around a SupergraphBuilder + the build call);
3. replay eligibility: supergraph-eligible = decode fire whose
   attachments ⊆ {mask, write-desc} × Peel endpoints (score/lora/hook
   fires eager as today);
4. cache-key collapse: the mask variant bit leaves the key for
   supergraph-eligible fires (one exec serves masked and unmasked).

S3 landed (`cb9ad3c58`) — the union is LIVE behind PIE_SUPERGRAPH=1:
one exec per (R, N) serves masked and unmasked decode fires (77
shared-key replays in the first flight, byte-identical to the
supergraph-off leg). The integration: the 9-slot device predicate
word on persistent inputs; IModel::supergraph_body dispatched by
digest; the capture's dual prepare materializing both arms' plans;
the union key (kGvSupergraph bit, mask bit folded, layout = a mix
spanning BOTH plans, post-capture re-key for the first-fire null-plan
case).

And the first flight earned its keep: the union key EXPOSED a real
pre-existing defect — masked pure-decode fires shared the custom
prefill plan slot with genuine prefill fires, whose per-request
re-planning oscillated the layout (one orphan capture per request;
today's masked-variant graphs quietly churned the same way). The
repair is now an axiom in code: **an arm may not share a mutable plan
slot with a foreign fire class** (`mask_decode_plan`, routed through
prepare, the hand-written body, the interpreter case and the
emitter's custom arm alike).

S4 board (fallout repair, ordered): hook fires into the union
(capture-safe score/page-mask machinery), lora staging capture
safety, Peel mixed fires (device-read row windows), multi-R sweep +
batteries at width (all deployments × attachment combos ×
buckets), then default-on.

S4 batteries, first tier (2026-08-03): the union holds at WIDTH.
- Multi-R: 4 plain + 2 masked-dense lanes concurrent — lane outputs
  identical ON vs OFF; the R=4 group and the R=1 solos each share one
  supergraph exec per bucket (2 captures total, the bucketing physics).
- All five deployments (qwen3-0.6b, olmo2, qwen2.5, phi3, mistral):
  baseline + masked-dense + masked-none A/B byte-identical with the
  supergraph on — the union survives bias, padded-head, post-norm, 7B
  scale and the force-prefill deployment's captured PLAN-LESS decode.
Remaining S4: hooks into the union, lora capture-safety, Peel mixed
fires, then default-on.

Default ON (same day): the interference battery (hook workloads
byte-identical to the off leg across 12 runs, lora exit criterion
holding, holed masked-prefill unchanged, small+wide A/Bs
byte-identical) proved the gate only reroutes union-eligible decode
fires. PIE_SUPERGRAPH=0 disarms. The supergraph is now the DEFAULT
serving configuration for eligible fires on every deployment in this
environment. Remaining S4: hooks and lora INTO the union (their
machinery must become capture-safe first), Peel mixed fires
(device-read row windows), and the perf measurement of what the union
bought (capture-count deltas; the masked-fire graph coverage that
eager fallbacks previously cost).

## The two-path lesson (2026-08-03, `4dcd7b112`)

The union's first perf guardrail run caught the per-guard form taxing
decode 5-15%: guards are stated PER LAYER, so the conditional graph
carried ~112 nodes and as many single-thread arm kernels per replay.
Two facts resolved it:
1. PoC-2 (tools/supergraph-poc/poc2.cu): sibling conditional nodes
   share one root handle; a nested body graph CANNOT reference an
   ancestor's handle. Per-slot shared handles only work per
   nesting scope.
2. Within today's union, every predicate but the mask is a CONSTANT
   of the graph context (eligibility: write-desc required, score/lora
   excluded, hooked rows out → Peel at all-fast).
The build therefore emits the RESOLVED form: one top-level IF/ELSE on
the mask, each body a fully guard-resolved straight-line walk
(SgValuation — emission-time evaluation of what the interpreter
branches on per fire). One handle, one arm kernel, zero nesting;
release throughput regression gone (ranges overlap, n16 often faster
ON). The k>1 form (per-guard conditionals + per-slot shared handles
per scope) returns when hooks/lora join the union and the predicate
space becomes genuinely multi-dimensional — enumerating 2^k resolved
paths stops scaling right around k=3 at 28 layers.

## The lora-graph campaign — design (2026-08-03)

Today every lora fire runs eager (`!has_lora` in eligibility) — for a
lora-serving workload that is EVERY decode step. The blockers, read
from `LoraFireState` itself, and their resolutions:
1. **Body-time allocation**: the A/B cast buffers and the grouped-GEMM
   pointer slab are cudaMallocAsync'd per fire ("a lora fire never
   enters capture, so body-time allocation is legal" — the comment
   that stops being true). Resolution: the buffers move to a
   fingerprint-owned persistent pool.
2. **Per-fire staging**: the handle's constructor does host loops
   (lane validation, cast uploads, slab fill). Resolution: split
   stage/launch — staging becomes a prepare-style host pass OUTSIDE
   the capture, the captured body reads only staged device buffers.
3. **Baked GEMM shapes**: ranks, lane counts, grouping and per-lane M
   bake into captured launches. The saving grace: in PURE DECODE every
   lane's token span is exactly 1, so the shape set is (R, lane
   structure, ranks) — stable across a workload's steps. Resolution:
   a lora fingerprint (the hook-graph pattern verbatim: per-key entry
   store, fingerprint check per fire, churn ban) keys captures;
   replay holds while the lane structure holds.
Steps: (1) stage/launch split + persistent buffers, eager path
byte-stable under the lora battery; (2) fingerprint + graph
eligibility for pure-decode lora fires (own exec store, hook
pattern); (3) union entry — lora as a third resolved path (mask x
lora paths at k=2 still enumerable; the per-guard shared-handle form
waits for k=3).

Consolidated sweep at tip `4dcd7b112` (default-on era): forward 78 +
engine 395 + ABI green; qwen3 GEN short byte-stable vs the prior
sweep; holed mixed tokens historical; hook battery 12/12 runs — the
supergraph default holds under the whole net.

Campaign step 1 landed (`c25a45026`): LoraStageArena on the Workspace
— 256-aligned bump allocation, grow-on-demand with retired blocks
held for in-flight readers, stream-safe per-fire reset. The cast
buffers and the pointer slab no longer cudaMallocAsync at body time;
the comment that justified it ("a lora fire never enters capture") is
retired. Gate: solo lora byte-stable across a cross-build stash diff;
zero-adapter equivalence holds. Step 2: slab uploads to a
prepare-style stage pass, then the fingerprint + eligibility.
