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

Campaign step 2 landed (`dfe182e02`): the stage/launch split. The
whole slab computes and uploads ONCE at fire setup (every slot value
is a fire constant — arena addresses, layer-strided adapter slices,
ws buffer rows); the per-member M arrays precompute into the groups;
apply() is slot arithmetic + GEMM launches only. The capture-fatal
pattern (per-layer pageable memcpyAsync from a scope-dying vector) is
gone. Gates: solo lora byte-stable vs pre-campaign; zero-grouped
z1==z2 steady-state across repeated runs (the single False is a
cold-boot first-round composition transient — lane arrival staggers
through warmup; the deterministic solo oracle carries correctness).
Step 3: the fingerprint + eligibility — LoraFireState's shape tuple
(lane count, ranks, sites, spans, grouping) hashes into a per-key
exec store on the hook pattern; pure decode's span-of-1 keeps it
stable across a workload's steps.

Step 3a landed (`c25a45026` → this): staging leaves the body. The
engine stages per fire before the graph decision (invoke_lora_stage;
the staged handle on plan_state, identity-checked), all three bodies
consume read-only with local fallback, and the stage call answers the
step-3b fingerprint. A lora fire's body is now launches-only end to
end. Live: 128 interpreter-leg fires on the engine-staged state; solo
byte-stable both legs. 3b next: the fingerprint-keyed exec store +
eligibility + capture-with-lora.

Step 3b landed (`4ae61cef6`): lora fires capture and replay CUDA
graphs. The eligibility's !has_lora term retired; kGvLora keys the
shapes; the fingerprint-partitioned store reuses the hook struct and
churn discipline (fingerprint = entry hash — a changed lane structure
selects a different entry, no stale path). Live: 18 captures across
R buckets and adapter fingerprints, silent replays between, solo
byte-stable (replay correctness), zero-equivalence all-True, non-lora
paths untouched. What remains of the campaign: step 4, union entry —
lora as a resolved path in the supergraph build (k=2: mask x lora),
which folds the last per-attachment exec split for decode fires.

## The campaign's step 4, resolved by argument (2026-08-03)

Should lora fold INTO the supergraph union (mask x lora, four resolved
paths)? No — the physics says the separate store IS the terminal
design. A captured lora arm bakes the adapter staging (GEMM shapes
from ranks and lane counts; cuBLAS grouped calls take host shape
arrays), so any exec containing a lora path is FINGERPRINT-KEYED — and
a union exec would then duplicate every NON-lora path into every
adapter-set's exec: 4-path walks x per-fingerprint copies, memory and
instantiation multiplying for zero sharing. The 3b shape is right:
plain/supergraph execs shared across everything non-lora; lora execs
per fingerprint carrying only the lora walk. The same argument
resolves HOOKS-into-union: hook execs bake program-set sideband state,
so folding them would duplicate the plain paths per program set — the
existing hook store is the terminal design too. The union's true axis
is exactly what it holds today: predicates whose arms bake NO
per-fire-class state (mask, write-desc, Peel endpoints).

**Measured payoff** (release, L40S, lora-probe 128-step decode x3):
eager 174-211 steps/s -> graph 225-243 steps/s — ~25% throughput on a
lora-serving workload's steady state, ranges disjoint.
PIE_LORA_GRAPH=0 is the rollback/measurement lever.

The supergraph thread's remaining REAL work is therefore: Peel mixed
fires (device-read row windows — kernel surgery; buys mixed-hook
replay stability and any future row-windowed union axis), and the
long-horizon capture-safety reworks if hooks/lora ever need to shed
their fingerprints (device-indirected shapes — blocked on kernel/API
support, e.g. device-side grouped GEMM shapes, which cuBLAS does not
offer today).

Consolidated sweep at tip `091ca131d` (2026-08-03): units green
(forward 78, engine 395, ABI); live byte-stable on every axis —
parity, supergraph small+wide A/B, holed mixed, hook battery, lora
solos. The lora campaign's six commits hold together.

## The Peel device-window campaign — design

The supergraph fallout list's last item. Today a MIXED hooked fire
(0 < fast_rows < N) bakes its row split into kernel args, so hook
execs churn (or ban to eager) whenever lane composition shifts the
split. The campaign teaches the Peel-region kernels to read the
window from DEVICE memory:
1. A per-fire device window word (pi.peel_window: {start,len} u32x2,
   uploaded beside the supergraph preds).
2. Device-window variants of the region kernels — the fused decode
   epilogue (prefix region) and the tail's split/qk-norm-rope/KV-write
   — launched at the FULL-N grid with a per-row window check (wasted
   threads at the tail, bounded by N; the launch shape stops depending
   on the split).
3. The interpreter's Win threading and the emitter's Win expressions
   move from host constants to the device word on the graph paths
   (eager keeps host windows — no wasted threads where no capture
   needs stability).
4. The hook fingerprint drops the split; mixed fires replay one exec
   across compositions.
Order: kernels first (each with an A/B against the host-window form),
then the plumbing, then the fingerprint change, batteries at each
step. This is deliberate multi-cycle kernel surgery — correctness
gates are the mixed-hook battery and the full sweep.

## The Peel device-window campaign — landed upfront, batteries deferred (2026-08-03)

Directive: implement the whole remaining campaign in one pass, test
after. Everything below compiles clean (full driver build); the
correctness gates (kernel A/B extension, mixed-hook battery, full
sweep) are the NEXT cycle's work and nothing here is verified live yet.

The five parts:
1. Kernel 4 — the fused decode epilogue's PREFIX form. Both kernel
   templates (block + warp) gained a nullable `win` param: rows
   [0, win[0]) — the window word's START is the prefix's row count,
   because the word stores the TAIL {start, len} and the prefix ends
   where the tail starts. One shared dispatch
   (`qkv_decode_fused_dispatch`); the host launcher passes null, the
   `_devwin` launcher passes the word and grids at full N. The warp
   kernel's early-out is warp-uniform (one warp = one (row, head)
   unit), so the FULL_MASK shuffles never see a partial warp; the
   block kernel's sits before any __syncthreads.
2. Kernel 5 — `launch_write_kv_to_pages_bf16_devwin` (TAIL form):
   `write_kv_kernel` gained the same nullable `win`; the devwin
   launcher grids every token, out-of-window rows early-out, indexing
   stays absolute, `first_token` rests at 0. Native-bf16 only;
   envelope maintenance refuses (the explicit devwin's disposition).
3. Plumbing — `pi.peel_window` (u32x2 {tail_start, tail_len});
   `ForwardInputs::peel_window_d`, set ONLY on hook captures
   (`capture_forward_graph_exec`); `run_forward_dispatch` uploads
   {fast, N-fast} before EVERY hook-graph launch, beside the
   supergraph pred upload.
4. Win threading — the interpreter's Peel walk, the hand-written
   body's fused branch, and the emitter's Peel emission all branch on
   `peel_window_d`: device mode emits BOTH regions unconditionally
   (an empty region's kernels launch and early-out on the word) via
   the five devwin forms; host mode keeps the fast_rows ifs and
   caller-offset windows — eager fires pay zero wasted threads. The
   emitter's `Win` became an enum (Host {start, len} | DevPrefix |
   DevTail); misplaced-window emissions panic. The hand-written
   `fused_decode_qkv_post` predicate drops its `fast_rows > 0` term
   in device mode only — the branch must not depend on the capture
   fire's split. All five generated forms re-emitted (the sixth,
   qwen3_5, does not consume the split and is untouched).
5. The fingerprint — `prepare_attention_phases` no longer mixes
   `hook_free_prefix_rows`. Safe because llama_like is the ONLY model
   with `supports_hook_graph_capture`, and both its bodies (declared/
   generated and hand-written) now read the split from the device
   word on captures. Lane `token_start` terms still mix — hooked
   lanes MOVING is real baked state; the campaign's claim is only
   that the hook-free prefix growing/shrinking stops churning execs.

Deferred (the next cycle's gates, in order): test_peel_window A/B
cases for kernels 4+5; the mixed-hook battery proving one exec
replays across compositions ([hook-graph] replay counts where
captures used to churn); byte parity of hooked fires against the
pre-campaign tip; the full consolidated sweep.

## Peel campaign — the deferred gates ran, and what they taught (2026-08-03)

Gate results (all on the L40S, debug, Qwen3-0.6B):
1. Kernel A/B: test_peel_window grew kernels 4+5 — 33/33 byte-equal
   (fused epilogue prefix at head_dim 64 warp + 32 block, both
   layouts; windowed to-pages suffix tails). The empty-window case
   flushed out a LATENT zero-grid launch in the host
   launch_qk_rmsnorm_rope_bf16 (sticky invalid-argument nobody ever
   checked) — zero-row guard added.
2. Solo oracle parity: snapkv/h2o/snapkv-tight/plain outputs
   byte-identical across the pre-campaign tip (e30d24d78) and the
   campaign tip — the cross-build deterministic-solo instrument.
3. Mixed batteries (single long snapkv instance, plain lanes joining
   and leaving around it, identical-args lanes to hold the ps hash):
   ONE R=2 capture, 37+38 and 57+59 replay runs across the phases,
   ZERO fingerprint recaptures, coherent returns. The captured
   DEVICE-WINDOW body is what replayed — its live correctness is
   proven, not just its kernel-level byte equality.

The finding the flip choreography forced: at a FIXED (R, N) key on
pure decode, `hook_free_prefix_rows` ≡ the first hooked lane's row ≡
that lane's `token_start`. The two fingerprint terms were REDUNDANT —
so dropping the prefix term alone cannot change replay behavior for
single-hook fires: a genuine split flip moves the hooked lane, and
the RETAINED token_start term recaptures. That retention is not
timidity — the prepared hook launches bake `token_start *
query_columns` query offsets and the sideband row layout
(dispatch.cu), so replaying across a lane move without device
indirection would read the wrong rows.

Terminal statement of the endpoint, revised: "mixed hooked fires
replay one exec across compositions" = (a) model body reads the split
from the device word — DONE, live-proven; (b) the hook side's
lane-row-derived baked args (query offsets, score sideband rows) go
device-indirected so token_start can leave the fingerprint — the NEXT
campaign, surgery in dispatch.cu's prepared-launch path. Until (b),
recaptures happen exactly when a hooked lane MOVES (composition churn
that preserves hooked-lane placement replays clean — the batteries'
demonstrated stability, which the old prefix term would have broken
only in lockstep with token_start anyway).

Environment note (not campaign fallout, verified on the pre-campaign
tip): concurrent lanes multiplexed on ONE python client connection
hang at completion delivery (engine finishes, Return events lost);
one connection per lane works. All batteries here use separate
connections. The a3-era mixed_fire_test passed this morning on a
shared connection, so something in the client/event path regressed
today — external to this thread, worth a board entry.

## Half two landed — and the premise, corrected by the scheduler's sort (2026-08-03)

The score-pad gather was the one captured consumer of a hooked lane's
row that baked a VALUE (its lane→request index). It now reads the
index from `Impl::hook_pad_requests` — an address-stable device table
the prepare pass re-uploads every fire, one u32 per pad in build
order; the pad bakes `&table[ordinal]`. `token_start` left the lane
fingerprint (the query/logits/score intrinsic bases were already
per-fire uploaded table content); the table base is mixed instead, so
growth recaptures once, honestly. Verified live: solo oracle still
byte-identical to the pre-campaign build; the mixed batteries replay
with the same capture/recapture counts as before (the remaining
recaptures are per-instance `bound` pointer churn — real baked
sideband state, a different animal).

What the flip choreographies then taught (three attempts, all
defeated): `fire_plan.rs` STABLE-SORTS fire members with hooked
programs LAST. The split is therefore a FUNCTION of the entry — at
fixed (R, N, program set) the hook-free prefix cannot move, and the
campaign's founding premise ("hook execs churn whenever lane
composition shifts the split") described a composition change that
always ALSO changes the bucket or the program set, i.e. selects a
different exec entry rather than churning one. The split terms the
fingerprint carried (prefix rows, token_start) were entry-constant
all along; the live churn that motivated the campaign is per-instance
pointer churn, which persists by design (one recapture per new
program instance).

What the campaign is therefore worth, stated precisely: the captured
bodies (model side AND hook side) are now structurally independent of
the scheduler's member-ordering policy. The sort key has already
generalized once (`device_resolved_geometry` joined it); when it
generalizes again — or when admission starts interleaving 3+ lanes
(today's engine serializes admissions, another reason no battery
could compose a flip) — the execs keep replaying where the
host-window forms would have silently read the wrong rows. Robustness
bought ahead of need, byte-stable everywhere, zero regression. The
REAL exec-stability frontier, now visible: instance-independent
baking (program-set-level sideband slots so a respawned instance
reuses its predecessor's exec), a separate campaign if churn ever
matters in practice.

Also noted for the board: engine admissions appear serialized (no
R=3 co-fire ever formed across every battery; new lanes admit only
after a running lane completes). Pre-existing behavior, possibly the
kv-contention grant rewrite's policy — not this thread's, but it caps
composition diversity in every hook battery.

## THE NORTH-STAR SUPERGRAPH DIRECTIVE (2026-08-03, user) — the spatial merge

The user's verdict on the arc so far, recorded verbatim in substance:
the shipped supergraph is ARTIFACT-sharing (one exec object reused
across fire variants — temporal; its gain goes to zero if capture
were free), while the north star is WORK-sharing (N member programs
merged into ONE execution inside a fire — spatial; 32 adapters read
the weights once, a gain no capture cache can produce). The wiki's
measured numbers (46x/14.09x/1.53x, the 1.01x correction floor, the
250us conditional region floor, SWITCH constant in bodies) all
belong to the spatial mechanism. The standing order: build THAT.

What pie already holds, mapped to the tart vocabulary
(concept/supergraph_ir.md + evidence/layout_planning.md):
- N programs per fire: EXISTS — co-batched lanes with per-lane
  attachments; the predicate vocabulary (write-desc, score, mask,
  lora, hooks) is the Program feature space.
- CORRECTION class: EXISTS at the 1.01x shape — lora's span-grouped
  additive correction on materialized q/v ("a fused edge cannot be a
  merge point" is the has_lora term in the fused predicate).
- Seriation v0: EXISTS — fire_plan.rs stable sort (geometry,
  hook_program, arrival); its Prefix lowering is consumed as the
  Peel split.
- Lowering::Prefix as a REAL spatial mechanism: EXISTS since the
  device-window campaign — pi.peel_window is a row-range member mask
  in device memory; one exec serves any split.
- Conditional machinery: EXISTS (SupergraphBuilder, device preds,
  capture-time insertion, handle scope rules) — exactly what the
  STRUCTURAL class needs; only its GRANULARITY is wrong (wired to
  fire-level bits, must move to region/row-range level).
What is missing (the user's five-item ladder, in pie terms):
1. Per-op member masks: predicates become per-REGION row windows
   (seriation makes member sets contiguous, so a window word per
   axis — peel_window's generalization — IS the bitmask).
2. Edge buffers at divergence points (the materialization decision,
   today implicit in the fused-vs-unfused predicate, becomes
   plan-owned).
3. The row layout planner: the sort key generalized so EVERY axis's
   member set is an interval (lexicographic refinement now; PQ-tree
   C1P when axes stop nesting — layout_planning.md).
4. The union pass: N per-class traces -> one merged op list whose
   divergent ops carry windows; fire-level Guards dissolve into
   region windows. This kills the 2^k wall at its root: no path
   enumeration, member masks on nodes.
5. (Ready.) The admission side already co-batches the programs.

The NS ladder (each rung parity-gated live, per standing practice):
- NS-1: seriation refinement — MemberFacts gains the mask axis; the
  sort key nests it under hooks; a SITE_ATTENTION_MASK site with the
  unmasked-prefix/masked-tail split as plan data (the mask analogue
  of Stage 1's fast_rows). Engine-side only, consumed later.
- NS-2: the first spatial mask fire — the masked pure-decode arm
  stops serving ALL rows with the custom-mask kernel: unmasked
  window takes the decode kernel, masked window the custom kernel,
  both device-windowed (the Peel pattern on a second axis). This is
  the first fire where two attention kernels serve one fire by row
  range — the real "여러 멤버를 하나의 실행으로".
- NS-3: window words become a vector (pi.region_windows), one per
  divergence axis; the emitter's Guard emission takes windows from
  the plan instead of fire-level bits; conditional nodes gate on
  window-nonempty at REGION granularity (>=250us bodies).
- NS-4: the union pass in the forward crate — per-class traces merge
  by fingerprint (SCS with the blocking rule); goldens pin the
  merged op lists; the 2^k supergraph key collapses to bucket keys.
- NS-5: retire the fire-level two-path form once NS-3/4 serve its
  fires (it remains the fallback until byte-parity holds everywhere).

## NS-1 live gate + NS-2 design, pinned pre-implementation (2026-08-03)

NS-1 gates (L40S, debug): masked-dense == masked-none byte-identical
under the new seriation; solo oracle byte-stable vs pre-campaign;
concurrent masked+plain mix green. The reorder is live and harmless.

NS-2 — the spatial mask fire, design:
- Scope: hook-free pure-decode fires with 0 < unmasked < R. Hooked+
  masked fires keep the fire-level arm (the seriation key nests mask
  under hooks, so the mask set is not contiguous there).
- The split source of truth is the PLAN: SITE_ATTENTION_MASK's
  fast_rows crosses the wire exactly as planned_hook_free_prefix_rows
  does (the B pattern: planned value + driver cross-check against a
  wire-derivable bound where possible; the driver CANNOT derive
  per-row maskness itself — composed assemblies synthesize causal
  rows for unmasked lanes).
- Prepare builds BOTH plans for such fires: the decode plan over the
  PREFIX sub-CSR (host CSRs unchanged, R' = split) and the custom
  mask plan over the SUFFIX (host CSR slices rebased by
  qo_indptr_h[split]; the device-side rebased suffix CSRs upload into
  dedicated pi buffers per fire — pi.mask_suffix_{qo,kvpp,kvlpl},
  R+1 headroom).
- The body (hand-written first, then interpreter case, then emitter
  arm): attention ONLY splits — decode kernel over rows [0, split),
  custom prefill kernel over [split, N) reading the suffix CSRs and
  the mask (mask indptr is already suffix-relative once composition
  stops synthesizing causal rows for the prefix — NS-2 keeps the
  synthesized rows and offsets custom_mask_indptr_d by the prefix's
  entries instead, so composition stays untouched). QKV, KV-write,
  MLP, lm_head stay full-N shared — the work-sharing is preserved;
  attention contributes the two structural points per layer the IR
  predicts.
- Numerics statement for the gates: unmasked lanes in a mixed fire
  MOVE from custom-kernel to decode-kernel numerics — byte-equal to
  their solo-plain selves (the more consistent contract), NOT to
  yesterday's mixed output. Masked lanes stay byte-equal to the
  masked arm. Gates: masked solo unchanged; mixed fire's unmasked
  lanes == plain solo; dense==none still holds per lane.
- Gate lever: PIE_SPATIAL_MASK (default OFF until the ladder's own
  batteries pass; flips ON with its consolidated sweep).
- Graph path: the supergraph mask arm keeps serving masked fires
  until NS-3 windows the arms (region windows + window-nonempty
  conditionals); NS-2 is eager-first.

## NS-2 built end to end — and its live subject is scheduler-blocked (2026-08-03)

The spatial mask fire is implemented across every layer: the engine
plans the unmasked prefix (attention_mask site → wire rows, lora and
hooks excluded at the planner so prepare's gate and dispatch's gate
cannot drift), the value crosses the ABI in the claimed reserved
step slot (PieStepDesc.planned_unmasked_prefix_rows), prepare builds
BOTH plans (prefix decode via a recursive prepare at R'=split — XQA
deployments guarded to the fire-level arm — and the custom plan over
the REBASED suffix), the eager dispatch uploads the two rebased
suffix CSRs (pi.mask_suffix_{qo,kv_page}_indptr; every other suffix
array is a pointer offset), and both bodies (interpreter and
hand-written) split the attention: decode kernel over [0, split),
custom kernel over the suffix, everything else full-N shared.
Spatial fires refuse graph capture (NS-3's job) and the generated
path (NS-4's job). PIE_SPATIAL_MASK arms it, default off.

Live: the crossing is verified (solo masked fires arrive with
planned=0 — the correct all-masked answer), outputs all byte-stable.
But the split never engaged, and the reject trace shows why: EVERY
masked fire is R=1. `mask_blocks_composition` (scheduler/worker.rs)
refuses wire-BRLE-masked lanes into composed batches in both
directions — "wire masks index the wire request layout composition
replaces". Only STRUCTURED device masks compose (the Stage 2 item A
relax). So the spatial mask fire's SUBJECT — a masked+plain
pure-decode co-batch — cannot form today.

The unblocking campaign, next: the wire-BRLE compose relax. The
frame assembly must re-index masked lanes' BRLE rows against the
composed layout and synthesize causal rows for the mask-free lanes
(the machinery the structured path already has); the grouping rule
then admits wire-masked lanes. Independently valuable — the code
comment itself prices the solo regime at 1.8-2.3x per token for the
co-batched plain lanes — and it is the LAST precondition for the
spatial mask fire's first live engagement.

## The compose blocker, dissected — and the split dissolves it (2026-08-03)

Why masked decode fires are all R=1: steady decode lanes ride the
DEVICE-RESOLVED chained-decode envelope geometry, and worker.rs
refuses `wire_mask_on_device_geometry` ("its BRLE indexes the
placeholder layout composition replaces; the solo
resolved_custom_wire path serves it"). Plain WIRE-geometry masked
lanes already compose (the dense-mask compose relax) — but decode is
envelope-class, so the spatial fire's subject never forms.

The deeper reason the solo rule exists: a composed fire-level mask
arm needs a mask row for EVERY lane, and an envelope lane's kv_len is
device-only knowledge — the host cannot synthesize its causal row
("nothing to assemble from"). The fire-level uniformity is what
demands the impossible row.

NS-2's split removes exactly that demand: the unmasked prefix takes
the DECODE kernel and needs NO mask rows; only the masked suffix's
rows — program-authored, host-known BRLE content — need staging, and
the split body already reads `custom_mask_indptr_d + split`. So the
wire-mask-on-device-geometry relax becomes possible ONLY under the
spatial split — the north-star thesis in miniature: per-region
members eliminate the whole-fire obligation that forced the solo
lowering.

The unblocking campaign (next, in order):
1. Engine grouping: admit `wire_mask_on_device_geometry` lanes into
   device-geometry groups when PIE_SPATIAL_MASK is armed engine-side
   (std::env gate mirroring the driver's), keeping the structured-mix
   and dense-device refusals.
2. Frame: the device-composed fixed-decode assembly stages the masked
   lanes' BRLE rows at their seriated suffix positions (lane order is
   host-known even when geometry is device-resolved) — mask_indptr
   entries for prefix lanes stay empty/zero-length, which the split
   body never reads.
3. The seriation must hold on the device-compose path too (the
   envelope suffix contract composes with mask-last within the
   envelope class — verify member_order flows into the device compose
   lane order).
4. Gates: masked-solo unchanged; the mixed fire engages ([spatial-
   mask] R>1 lines); masked lane byte-equal to solo; plain lanes
   byte-equal to solo-plain; then the wide battery and the sweep.

## The compose relax, first live contact — the mask is device-carried (2026-08-03)

The engine grouping relax works: with PIE_SPATIAL_MASK armed on both
sides, the masked+plain pure-decode group FORMS (census mask-compose
deferrals gone). The composed step then fails at the DRIVER's
admission: "program carries a dense device mask in a multi-program
batch (v1 mask scope is solo only)". The lesson: naive-masked's
decode-phase mask is NOT wire BRLE at all — it is DEVICE-CARRIED,
a `kPortAttnMask` channel the program writes, resolved per fire by
RV-6 descriptor resolution (`dense_mask_scope_violation` walks the
trace PORTS, not the wire rows). The wire-mask frame extension I
built this cycle serves the wire-BRLE flavors (prefill-phase masks,
future wire producers) but the live decode subject rides the device
channel.

Remaining surgery, pinned for the next stretch:
1. `Dispatch::dense_mask_scope_violation` admits the dense-masked
   program in a multi-program batch WHEN
   `view.planned_unmasked_prefix_rows` is a valid split and the
   masked program's rows are exactly the suffix (the seriation
   guarantee).
2. The dense pack (`resolve_attention_mask` → pi.dense_mask →
   launch_pack_dense_mask → pi.custom_mask) packs the masked
   program's rows AT THEIR COMPOSED SUFFIX POSITIONS: indptr entries
   for the prefix rows stay empty (the split body never reads them),
   the suffix rows carry the device-resolved mask content.
3. Then the battery: [spatial-mask] R=2 engage lines, masked lane
   byte-equal to solo, plain lane byte-equal to solo — the FIRST
   spatial merge fire.
Current state is safe to ship gated: default-off changes nothing;
gate-on forms the group and fails LOUD at admission (no corruption
path), which is exactly where the next stretch picks up.

## THE FIRST SPATIAL FIRE ENGAGED (2026-08-03) — half the gate green

With the scope admits (dispatch admission + frame backstop, both
gated on a valid planned split) and the suffix-positioned dense pack
(the masked program's device-carried mask staged PADDED to fire-lane
indexing — prefix rows klen 0, the pack kernel no-ops them), the
composed masked+plain pure-decode fire RUNS THE SPLIT:
[spatial-mask] R=2 split=1, seventeen fires, prefix on the decode
kernel, suffix on the custom kernel, one fire. And the first gate
half is GREEN: the plain lane's mixed output is BYTE-EQUAL to its
solo self — the unmasked prefix truly runs the decode kernel inside
a fire that also serves a masked lane. This is the north-star merge,
alive.

The masked suffix's numerics are wrong (garbage tokens), and the
cause is identified: the suffix plan and the rebased suffix CSRs
were built from the HOST wire views (h_qo/h_kvpp), which for a
composed-envelope lane are placeholders — the truth lives in the
RESOLVER's per-program host geometry (fg.kv_page_indptr/
kv_last_page_lens, exactly what the dense pack block already
consumes) and in the composed DEVICE CSRs. The fix, pinned:
1. The suffix dispatch needs NO uploaded rebased CSRs at all — pass
   DEVICE pointer offsets with ABSOLUTE values: q/out at BASE (not
   bf16_row offsets), qo_indptr_d + split, kv_page_indices_d BASE,
   kv_page_indptr_d + split, kvlpl_d + split, mask_indptr_d + split.
   The kernel indexes rows through the (absolute) indptr values, so
   no rebasing is needed device-side. pi.mask_suffix_* buffers and
   the forward.cpp upload block then retire.
2. The suffix PLAN still needs host counts: thread the masked
   program's RESOLVED host geometry (np/lpl per suffix lane, from
   fg) from the frame into PrepareInputs (new pointer fields staged
   on the wave state), replacing the h_kvpp slices.
3. Wire flavors (prefill-phase masks) keep the fire-level arm; the
   frame's resolved_custom_wire stays single-program (the reverted
   extension's lesson: a composed spatial step's wire rows are the
   WIRE lanes' synthesized causal masks, pure-causal by the walk).

## THE SPATIAL FIRE IS CORRECT (2026-08-03) — NS-2 live gates green

The masked-suffix fix that landed it, measured against two failed
addressing hypotheses: the custom kernel's q/o side is
plan/qo[0]-RELATIVE (offset q/out pointers + the uploaded identity
qo), while its KV side reads the device CSR ABSOLUTELY (base page
indices + kv_page_indptr at +split with composed-device values — no
host rebase, no host knowledge). The suffix PLAN's host counts come
from the RESOLVER's per-program geometry, harvested by the spatial
dense-pack block onto the wave state and threaded into prepare
(mask_suffix_page_counts_h / last_lens_h) — the host wire views are
placeholders for composed-envelope lanes and produced attempt 4's
garbage.

Gates, with the corrected statement:
- masked mixed == masked solo: BYTE-EQUAL through the spatial fire
  ([spatial-mask] R=2 split=1; prefix decode kernel + suffix custom
  kernel + shared everything else, one fire).
- plain mixed vs plain solo diverges at ~token 30 — and the CONTROL
  (two PLAIN lanes co-batched, no mask anywhere) diverges the same
  way: this is the generic co-batch GEMM-rounding class that has
  always existed at N>1, not a spatial defect. The original gate
  statement ("plain mixed == plain solo") was stronger than the
  system's own invariant; the corrected gate is "masked == solo, and
  plain divergence bounded to the generic co-batch class" — both
  hold.

NS-2 is therefore LIVE and correct, eager, PIE_SPATIAL_MASK-gated.
What remains on the ladder: default-on decision after the wide
battery + sweep; NS-3 (region windows + graphing the split); NS-4
(union pass); NS-5 (retire the two-path form).

## NS-2 wide batteries green (2026-08-04) — and the all-masked fire

The wide compositions taught three more lessons, each now in the
tree: the pack serves N masked programs (per-program strides, padded
to the max; the programs must tile the suffix exactly — seriation
enforced loudly); split == 0 is the ALL-MASKED composed fire and
admits everywhere (two masked lanes share one fire where each fired
solo before — a capability nothing in the system had); and the
harvested suffix geometry is per-wave state that MUST reset (the
stale-pointer read produced flashinfer's negative-indptr throw).
Solo masked fires now route through the same spatial machinery
(R=1 split=0) with byte-identical outputs — one code path.

Battery state: R=4 split=2 / R=2 both splits / all-masked pairs —
zero errors, every composition engages, deterministic compositions
byte-stable, mixed-fire text within the generic co-batch class.
Default-off regression green (oracle, masked solo, hooks).

Default-on remains deliberately open until: the consolidated sweep
(all five deployments + supergraph A/B + lora + hooks) runs under
the gate, and NS-3 decides how spatial fires graph (today they force
eager — a masked steady-state decode loses graph replay, which is a
real regression for masked-heavy workloads until NS-3 lands).

## NS-3 v1 design — graphing the split (2026-08-04, pre-implementation)

What a captured spatial fire bakes, walked term by term: the two
attention dispatches' pointer offsets and both plans' grids are
functions of SPLIT; the pack runs prepare-side (outside capture) into
stable pi addresses; the suffix qo identity's CONTENT (0..rs) is a
prefix of one universal sequence, so ONE buffer serves every split;
mask indptr/content refresh per fire at stable addresses. Therefore
v1 keys the exec on the split: variant bits 21-28 carry it (R <= 255
by the bucket lattice) plus kGvSpatial at bit 29 — at most R execs
per bucket, the honest 2^k-free form until region windows
(device-windowed attention needs flashinfer-side work; NOT v1).
Changes: run_graph stops excluding spatial fires; use_supergraph
EXCLUDES them (the union's fire-level mask arm is wrong for a split
fire); capture threads unmasked_prefix_rows + the suffix qo pointer
into the captured body; the per-fire prepare already rebuilds both
plans (graph_mode_plan for the suffix mask plan is already true).
Gates: spatial battery under graphs (engage lines + replays, outputs
per the numerics contract), default-off regression, then the
consolidated sweep for the default-on decision.

## Spatial mask DEFAULT ON (2026-08-04)

The sweep under the gate: solo oracle byte-stable vs pre-campaign,
masked dense==none, hook solos + mixed hook battery, lora solos
deterministic. Flipped at all six sites (PIE_SPATIAL_MASK=0
disarms); the era-pinning grouping tests now pin the spatial
contract. Live default boot: R=2 split=1 engages, oracle stable.
The ladder's remaining rungs: NS-4 (the union pass — the emitter
still skips spatial fires to the interpreter; per-class traces
should merge into one windowed op list, killing the fire-level
Guard at its root) and NS-5 (retire the two-path supergraph form
once NS-4 serves its fires). A flip-era caution for the record:
a pattern-based env flip catches NEIGHBORING gates — the first
attempt flipped TP_DISABLE_DEVICE_COMPOSE / STEP_PROFILE /
HOOK_GRAPH_TRACE to default-on before the per-name pass restored
them.

## THE LADDER'S V1 IS COMPLETE (2026-08-04) — NS-5, retirement by promotion

PIE_SUPERGRAPH defaults OFF: the temporal union's one live axis (the
mask) was promoted to the spatial form, no fire can arm its
conditional, and the exec had reduced to the plain graph plus dead
capture weight. The machinery stays for the STRUCTURAL class. The
directive's five items, closed in pie terms: (1) per-op member masks
= the split's row windows, live and default; (2) edge buffers = the
materialization the split body already honors (fused edges stay
non-merge-points); (3) seriation = the mask-nested member sort;
(4) the union = the Guard's fire-level mask arm dissolved into
windowed regions, stated by the emitter; (5) N programs per fire =
the compose relax, including the all-masked fire class.

What v1 deliberately leaves open, for the record: the SCS union of
GENUINELY separate member programs (today's members are attachment
combinations over one model trace; a second model-structural program
class — spec verify, early exit — is what forces the supersequence
alignment and the conditional regions, and the retired-but-kept
builder is its organ); PQ-tree seriation when the axes stop nesting
(hooks x mask today nests, so the lexicographic key suffices); the
split under XQA and padded head dims (both guarded to the fire-level
arm); and flashinfer-side device row windows (which would collapse
the split-keyed exec family to one exec per bucket).

## THE REVIEW LANDS: THE MASK SPLIT BECOMES VOCABULARY (2026-08-04)

The post-v1 review's finding, accepted whole: the two live window
axes lived in different layers — the hook axis as IR (OpKind::Peel)
and the mask axis as C++ text the emitter printed into the custom
arm. Same failure the DSL exists to kill ("the smarts accumulated
there"), relocated from the driver into the emitter; no golden could
pin it and no second backend could consume it. The review's own
sentence is the fix: Peel is already the word for "two regions that
both run over complementary row ranges" — only the split's SOURCE
differs. So Peel gained a WINDOW AXIS (PeelWindow: HookFreePrefix,
the serde default keeping every pre-window golden byte-identical;
UnmaskedPrefix, the spatial mask split), and the mask arm of the
decode declaration now STATES the split (dsl::peel_masked) exactly
where prepare's deployment gate holds (!xqa && !padded). The axis
crosses the FFI as the Peel's aux run (empty = hook, [1] = mask).

Two consequences the lift forced, both improvements the text form
had hidden: (a) the UNPLANNED endpoint is the peel's own degenerate
form (tail-only, full-N, fire-level addressing) — the fire-level
custom dispatch is not a separate op; (b) the prefix region states
THE DEPLOYMENT'S decode form, not a hardcoded decode dispatch —
qwen2_5 (force_prefill: GQA ratio outside the decode kernel's set)
states dequant staging + the plan-free prefill launcher over
[0, split), which CLOSED A LIVE LANDMINE: since spatial default-on,
a masked mixed fire on qwen2_5 threw "no prefix decode plan" on all
three legs. The emitter text could never have said this per
deployment without another nested runtime branch; the trace says it
per deployment for free, because deployments are what traces are.

Emitter and interpreter are region walkers again: emit_cuda spells
the window plumbing once at the Peel and the attention arms key
their addressing on the region marker (Win::MaskPrefix/MaskTail);
declared_forward's walk carries mask-region events as a SEPARATE
axis from the hook window (they never nest — the engine plans
UNPLANNED for hooked fires). Goldens now pin the split as structure
(qwen3/mistral: decode-region prefix; qwen2_5: dequant+prefill
prefix; phi3 padded: no peel). Verified live, all three legs:
canonical masked solo byte-equal, solo oracle byte-equal to the
pre-campaign reference, mixed masked+2-plain engaging planned
splits (R=4 split=2, ~22 fires/leg) with plains byte-consistent
across legs. Board: use_prefill_decode_plan (Hopper) prefix
polymorphism — the prefix region's plan-family choice under sm>=9
is not yet stated; off on sm_89.

Commit 797cdefc6.

## THE PROMOTION, PRICED (2026-08-04) — release A/B of the spatial default

The retired solo regime was only ever priced by a code comment
("1.8-2.3x per token for the co-batched plain lanes", projected, never
measured). Measured now, release build on the L40S (llama-3.2-1b,
256 tokens/lane, warm rounds, masked lane joining mid-stream; ON =
default, OFF = PIE_SPATIAL_MASK=0):

  plain-only R=4/R=8 . ON == OFF (~1.22-1.24s)  — no regression
  mixed R=4 . plains 1.29s vs 1.43s (ON 10% faster), masked 11% faster
  mixed R=8 . plains 1.30s vs 1.57s (ON 17% faster), masked 15% faster

The shape of the numbers is the thesis: the masked lane's tax on its
co-batched plains is CONSTANT ~6% under the spatial merge (one marginal
row in a shared fire) and GROWS with R under the solo regime (+15% at
R=4, +27% at R=8 — the duplicate weight read plus the eager mask
dispatch bill the whole co-batch). 285 R=4 split=3 fires confirmed
composing during the bench; canonical masked output byte-stable
throughout. The old comment's 1.8-2.3x was the lockstep worst case —
the real solo regime pipelined some of the cost; the merge removes it
structurally rather than hiding it. Gains scale with R and with model
size (the fire is launch-bound at 1B/R<=8; the weight-read term the
merge deletes is the one that grows). Bench + table:
.wiki/tart/bench_spatial_results.md, scratchpad bench_spatial.py.

## ADMISSION RETRACTION + THE STRUCTURAL CLASS V0 DESIGN (2026-08-04)

Retraction first, measured: the board's "engine admissions serialize —
no R=3 co-fire ever forms" is FALSE for the current engine. Release,
256 tok/lane, same-instant launches: 4 lanes form 128 R=4 split=3
co-fires, 8 lanes form 128 R=8 split=7 — full composition through the
entire overlap window; the solo tail is the masked lane OUTLIVING the
plains. Composition rate is governed by lifetime overlap, nothing
else. (The old finding belonged to the capped-flip battery era; short
64-token lanes under-compose because they finish inside the prefill
stagger.) The scheduler needs no fix; the 6%-vs-27% merge win applies
whenever lifetimes overlap.

With that closed, the frontier is the review's two remaining X rows
(Supergraph = DAG, union pass): both blocked on a SECOND program
class. Design for its v0, grounded in the organs that exist:

THE CLASS: fixed-k layer-truncated decode ("layerskip draft" — logit
lens over layer k's hidden state; a real drafting technique, and
later the self-speculative drafter's verify counterpart). Chosen over
spec-verify (drafter is bravo's) and confidence-exit (dynamic k is a
PER-ROW branch — not a fire-plannable window) because fixed k gives a
STATIC second class: the trace differs from the full class in WHICH
OPS RUN, not in any per-fire value — exactly Div::STRUCTURAL
(fire_plan.rs already carries the vocabulary).

THE KEY INSIGHT — the union stays in Peel vocabulary: seriate members
by depth (full-depth first, truncated last) and the structural
divergence is ANOTHER ROW WINDOW. At layer k the fire splits: layers
[k, L) + final norm + lm_head run over the full-depth prefix rows
[0, n_full); the truncated tail rows take final norm + lm_head
(logit-lens head) at layer k. That is a Peel whose regions differ in
OPS (they always could — the hook peel's regions already do) and
whose window is a third axis: PeelWindow::FullDepthPrefix. No DAG
machinery, no SCS alignment, no conditional regions needed — the
supersequence of "layers [0,k) ++ head" and "layers [0,L) ++ head"
IS the full trace with one peel at k. The kept SupergraphBuilder
stays in reserve for classes that DON'T prefix-share (true SCS); the
PQ-tree moment arrives only when a third axis crosses (mask x depth
in one fire, hooks x depth, or two distinct k values).

THE LADDER (mirroring NS-2's, rung by rung):
  S-1 the channel: a `max_layers` (k) request field, client ->
      engine request -> MemberFacts (the Stage-4 lora channel
      pattern); v0 restricts a fire to ONE k (scheduler refuses
      mixed-k composition — lowest-order blocking rule).
  S-2 seriation + wire: sort key gains the depth bit (full first,
      truncated last, before hooks in the order — depth nests
      OUTSIDE mask/hooks in v0 by REFUSING their composition with
      truncated members at all: truncated lanes are plain decode
      only); a planned `full_depth_rows` wire word beside
      planned_unmasked_prefix_rows (same reserved-slot pattern).
  S-3 the trace: Peel { window: FullDepthPrefix } at layer k in the
      DECLARATION — prefix region = layers [k,L)+norm+lm_head ops,
      tail region = norm+lm_head-at-k ops (the logit-lens head reuses
      the final norm weights in v0 — stated plainly so parity is
      honest). K is a TRACE-TIME constant per deployment-variant
      (v0: one k per model config, e.g. L/2), so traces stay static;
      per-request k is v1+ (it re-keys the trace, the same way
      deployments do).
  S-4 driver: prepare plans logits rows for both regions; the
      interpreter/emitter walk the depth peel exactly as the mask
      peel (region markers, windowed call forms — the attention/MLP
      launches need only their existing row-window forms since the
      prefix region is a contiguous row prefix at every layer).
  S-5 verification: truncated solo == full solo prefix layers
      byte-check at k (logit-lens oracle), mixed fire == the two
      solos' logits row-for-row, then graphs (split-keyed on
      (n_full, k)), then the README's 1.53x-class measurement.

V0 exclusions, stated loudly: dynamic/confidence k (per-row branch),
mixed-k fires (PQ-tree), truncated x mask / truncated x hooks / x
lora (blocking rule refuses), trained exit heads (weights don't
exist; logit-lens is the honest v0 head). Each is a recorded rung,
not a silent gap.

## THE MIXED FIRE DIRECTIVE (2026-08-04): decode + masked decode + prefill, one pass, two streams

The user's target example, verbatim: one batched forward pass carrying
custom-mask decode (the prefill kernel), causal-mask decode, and
prefill together — with the custom-mask attention and the prefill
attention on DIFFERENT STREAMS. Mapped onto the organs:

What already exists: plain decode + prefill co-batch TODAY (the
chunk-prefill clause in worker.rs is deliberately narrow — only
page-mask x multitoken and mask x multitoken refuse); the driver runs
such a fire through the PREFILL class, decode lanes as 1-token
qo_indptr entries through the causal prefill dispatch. Streams exist
in pieces (CudaStreamOwner, the supergraph's non-blocking stream).

What blocks the example: exactly one clause — a wire-masked decode
lane refuses multi-token groups (and conversely), because the prefill
class's mask arm is still the FIRE-LEVEL custom dispatch: it would
take every row through the custom kernel, and synthesizing causal
masks for prefill rows explodes (the recorded reason for the
refusal). The fix is the one we already know: THE MASK PEEL
GENERALIZES TO THE PREFILL CLASS. Seriation puts masked decode rows
last; the causal prefill dispatch serves the prefix rows (prefill
AND plain-decode rows — v0 keeps them merged in one causal dispatch,
the decode-kernel specialization of plain rows is a later rung); the
custom dispatch serves the masked suffix. Same PeelWindow::
UnmaskedPrefix word, now stated in the Prefill class arm too.

THE ONE NEW INVARIANT — two split words: in a pure-decode fire,
request index == token row, and the single planned word served both
the CSR offset (+split requests) and the q/o row offset (bf16_row at
split rows). In a mixed fire they DIVERGE (prefill rows contribute
many tokens): the wire needs BOTH the unmasked-prefix REQUEST count
(CSR/mask-indptr/last-lens offsets, suffix plan geometry) and the
unmasked-prefix TOKEN-ROW count (q/out pointer offsets). Threading
the second word retraces the first's exact path (batch.rs ->
reserved ABI slot -> step_launch/launch_view -> prepare/dispatch).

THE STREAM FORK (the requirement's second half): within the layer
body, after the KV write completes, the causal prefill dispatch and
the custom dispatch have disjoint outputs (attn_out row windows) and
read-only-shared inputs (q, the layer's KV) — a textbook fork:
  event E1 on main stream after KV write; stream B waits E1;
  causal dispatch on main, custom dispatch on B; event E2 on B;
  main waits E2 before o_proj.
One secondary stream per context (CudaStreamOwner pattern), events
reused per layer. Prefill-shaped fires run EAGER (no decode-graph
capture applies), so v0 needs no graph-side stream work — capture-
time forking (parallel graph branches) is a recorded later rung.

THE LADDER:
  M-1 engine: relax the mask x multitoken refusals for wire-BRLE
      masked decode lanes (both join directions); seriation orders
      [prefill | plain decode | masked decode] rows; plan BOTH split
      words (requests + token rows); UNPLANNED stays the hooks/lora/
      structured escape. Gate: PIE_SPATIAL_MASK (same switch — the
      axis is the same).
  M-2 driver: the Prefill class mask arm becomes the peel
      (dsl::peel_masked with the causal-dispatch region and the
      custom region); prepare builds the suffix mask plan for
      prefill-shaped fires (the resolver-geometry pattern verbatim);
      hand-written + interpreter + emitter learn the token-row
      offset forms (the decode-class forms parameterized by the
      second word). Goldens regenerate; phi3/XQA keep fire-level.
  M-3 streams: the custom region dispatches on the secondary stream
      between the fork/join events; ONLY when the peel is planned
      (fire-level custom keeps the main stream). A PIE_SPATIAL_
      STREAM=0 escape hatch for bisection.
  M-4 the example battery: one fire holding [prefill lane, plain
      decode lane, masked decode lane]; numerics leg (mixed ==
      each solo's rows, the composed-fire equality class); overlap
      leg (wall time of the fire vs =0, and an nsys trace showing
      the two dispatches overlapped); the three-leg parity sweep.

V0 exclusions, loud: plain-decode-row specialization to the decode
kernel inside mixed fires (they ride the causal dispatch); graph
capture of mixed fires (eager); masked x hooks/lora (UNPLANNED as
today); multiple masked programs already work (the N-program tiling
carries over).

## THE MIXED FIRE IS DEFAULT (2026-08-04) — the directive's example, standing

The campaign closed in five commits: the driver slice (2924da047, two
split domains + side stream), the WIP disarm (97c86c4ec), the
workspace root-cause and fix (8c4934cae, 9a3fcc85d — two
prefill-family plans must not share one AttentionWorkspace's
scheduling buffers; the suffix plan owns a dedicated workspace, which
also gives the two concurrent dispatches disjoint scratch), the
declaration (fe4cbd236 — the prefill-class mask arm states the same
UnmaskedPrefix peel as the decode class, goldens pin it, all three
legs serve it, the tail's plan/workspace pairing rule stated), and
the flip (55c022f36). The user's example — custom-mask decode +
causal-mask decode + prefill in ONE batched forward pass, the custom
and prefill attentions on DIFFERENT streams — is now the default
behavior: seriation orders [wire prefill+decode | masked envelope
suffix], the planned word (REQUEST domain — measured, the one
surprise of the campaign) splits the fire, the prefix causal
dispatch serves any mix of prefill and plain-decode requests on the
main stream, the masked 1-token suffix's custom dispatch runs on the
side stream between fork/join events.

Honest edges, recorded: the overlap WIN is unmeasured at 1B scale
(span timing is noise-level vs serialized; heavy-workload evidence
owed); mixed numerics sit in the generic co-batch rounding class
(control-proven, 2/8 both ways); masked x hooks/lora and padded/XQA
shapes keep the fire-level word; graph capture does not cover
prefill-shaped fires (eager by class).

## THE OVERLAP, MEASURED (2026-08-04) — the deferred edge closes

Release build, heavy shapes (a 2.4k-token masked KV decoding 256 steps
while 1.9k-token prefills join at 0.1s stagger; L0 fork->join span,
PIE_SPATIAL_STREAM_TIMING=1):

  two streams . med 0.096 ms  p90 0.103 ms  (n=24)
  serialized  . med 0.115 ms  p90 0.115 ms  (n=23)

The masked suffix's custom dispatch (~19us at this KV size) hides
COMPLETELY behind the prefix causal dispatch — a ~17% shorter
attention section per layer, the whole distribution shifted, on the
default path. The win scales with the suffix's work (longer masked
KV, more masked lanes); at 1B/16-layers it is ~0.3ms per mixed fire.
The directive's two-stream requirement is not just structural — it
pays, measurably. Battery: .wiki/tart/heavy_overlap.py.

## STRUCTURAL S-1 LANDS (2026-08-04) — the second class's first organ

The layer-truncation channel runs end to end (5f1c5bb6e):
LaunchPlan.max_layers -> the scheduler's solo blocking rule
("truncated-depth" — the depth union is the next rung) -> the appended
ABI word (planned_max_layers, MAX = full) -> graphs refuse, declared
legs route to the hand-written body, the layer loop bounds at k and
the UNCHANGED tail (final norm + lm_head) is the logit-lens head.
Oracle on Qwen3-0.6B (28 layers): k=28 BYTE-IDENTICAL to unset (the
channel is numerics-neutral and truncation at full depth is the
identity); k=16/k=8 deterministic degraded drafts, deeper = better —
logit-lens behavior, the layerskip-draft class's honest v0 head.
Producer is a TEST SCAFFOLD (PIE_DEBUG_MAX_LAYERS stamps every fire)
until slice B lands the WIT surface. Next rungs stand as recorded:
S-B the inferlet-facing channel, S-2 depth seriation + wire, S-3 the
FullDepthPrefix peel in the declaration, S-4 the walkers, S-5 the
union oracle and the 1.53x measurement.

## DESIGN NOTE (2026-08-04): the PEFT correction surface — form joins structure and contents

From the user's question ("how would LoRA-family PEFT express elegantly,
no shape limits, covering the variants"), worked against the live WIT:
the WIT is ALREADY shape-agnostic (channel(shape, dtype, capacity));
every restriction lives in the DSL's named sink `kernel::lora(a, b,
sites)` — trace-known rank (bucket per rank), packed per-site shapes
(the lora-probe q+v pain), one hardcoded form. The generalization keeps
§6.5's principle and adds a third term: placement is STRUCTURE (one
`correct(site, region)` declaration per site, shapes free per site),
weights are CONTENTS (channels), and FORM is a small closed expression
over (x, y): {mm-by-channel-tensor, elementwise scale, add, reshape}.
The compiler CLASSIFIES the expression — recognized forms lower to the
existing span-grouped CORRECTION kernels (LoRA/AdaLoRA `y+B(Ax)`, VeRA
`y+Λb·B(Λd·Ax)`, DoRA `s⊙(y+B(Ax))` with s precomputed per adapter,
IA3 `l⊙y`, BitFit `y+bias`, LoKr via reshape+two-small-mm) — unknown
forms refuse loudly (v0 closed world; the driver stays dumb). Rank
stops being trace-known: grouped GEMM takes per-problem shapes, so
ragged per-layer/per-member ranks become instance data; graphs bucket
on (form structure, shape class) or stay eager first. Honest class
boundaries: LoHa (Hadamard of matrices does not distribute over
matvec) is Div::WEIGHT (merge per-adapter ΔW, the MoE organ);
prefix/prompt tuning is the KV axis (learned pages), not a correction.
Composition: the form's structure hash joins the seriation key —
same-form adapters span-group exactly as lora does today, cross-form
v0 solo, later a row window like every axis before it.

## DESIGN FINAL (2026-08-04): fwd.adapter(site, |x, y| expr) — the seat is the builder, the body is PTIR

Converged with the user over three rounds. The earlier note's two
candidate surfaces merge: the DSL-sink generality was right about the
PAYLOAD (an open expression, not a closed WIT variant — VeRA/DoRA and
future compositions without WIT churn), and the builder-method
instinct was right about the SEAT (adapters are pass-level
configuration, not traced-program plumbing the guest should hand-roll).
The house already holds the reconciliation: prologue/epilogue are
builder methods that TRACE CLOSURES into container regions — adapter
is their sibling. So:

  fwd.adapter(Site::Q, |x, y| y + mm(b.read(), mm(a.read(), x)));

- SEAT: an sdk ForwardPass method beside prologue/epilogue (noun
  style, one call per site). WIT UNCHANGED — the container carries
  the region, exactly as it carries prologue/epilogue.
- BODY: a per-site ADAPTER REGION (new region kind) over the closed
  op set {mm-by-channel, scale, add, reshape} with x/y as
  SYMBOLIC-DIM tensors (SiteIn/SiteOut resolved at bind against the
  ModelProfile — the lora-probe D_OUT hardcoding dies). The validator
  checks the op set; the compiler classifies structure into the
  span-grouped CORRECTION lowerings and refuses the unknown loudly;
  the region's structure hash joins the composition key.
- The honest trade, recorded: rank rides channel shapes, so a rank
  change re-traces the container (cheap, guest-side; a new identity
  bucket). v1 extends symbolic dims to channel declarations' rank
  axis, completing "swap adapter = re-seed" for rank too.
- kernel::lora becomes the deprecated special case of the low-rank
  form; §6.5's span-grouped lowering stays the execution organ.

## S-2/S-3 DESIGN (2026-08-04): the depth union executes with ONE tail

The load-bearing discovery for the FullDepthPrefix peel: seriate
full-depth members first, truncated last, and then

  layers [0, k)   run FULL-N (every row, exactly today's body);
  layers [k, L)   run over the PREFIX rows only (row-major
                  activations make the prefix contiguous: GEMMs/norms
                  just take N' = prefix tokens; attention takes a
                  prefix-request plan);
  the tail        (final norm + lm_head) runs FULL-N, UNCHANGED.

No hidden-state stash, no second head: layers [k, L) never write the
suffix rows, so the suffix's x is FROZEN at its layer-k value and the
one full-N tail IS the logit-lens head for the truncated rows while
being the ordinary head for the full rows. The union costs two loop
bounds and a prefix plan.

Wire: the fire-level k word (planned_max_layers) becomes the SUFFIX's
uniform k; a new planned_full_depth_rows request-split word rides the
same appended-word pattern (UNPLANNED = uniform fire, today's solo
shape). v0 blocking: truncated members compose ONLY with plain
full-depth decode (no masks/hooks/lora/mixed-k — each a recorded
later rung), which makes the second plan workspace reusable: a depth
fire is never also a spatial-mask fire, so the dedicated secondary
workspace serves both mutually-exclusive shapes.

## S-5: THE STRUCTURAL CLASS, PRICED (2026-08-04)

Release A/B on the L40S (Qwen3-0.6B/28L, 256 tok/lane, 1 full lane +
D layerskip-draft lanes at k=8, union ON = default vs
PIE_DEPTH_UNION=0 = drafts solo-fire):

  D=3 . ON ~1.27s  OFF ~1.68s  -> 1.32x
  D=7 . ON ~1.35s  OFF ~3.04s  -> 2.25x

The union's win GROWS with the draft count — the solo regime pays D
separate k-layer fires serialized against the shared fire, the union
folds every draft into rows of ONE fire (layers [0,k) shared full-N,
[k,L) prefix-only, one tail). The README's 1.53x STRUCTURAL-class
number sits inside the measured bracket, with the scaling shape
demonstrated. 1783 union fires across the ON benches; S-B identity
oracle green on the release build. Battery: .wiki/tart/bench_depth.py.

## THE CORRECTION CLASS, PRICED AT SCALE (2026-08-04)

Release, L40S, Qwen3-0.6B, 128 tok/lane, D concurrent lora lanes each
carrying its OWN adapter contents (per-instance channel seeds — swap
is re-seed, the §6.5 contract):

  solo mean 0.78s
  D=4 . wall 0.94s -> 3.32x vs serialized (83% of ideal)
  D=8 . wall 1.24s -> 5.04x vs serialized (63% of ideal)

D distinct adapters share one fire's base-weight reads through the
span-grouped correction — the WEIGHT-sharing thesis at request
granularity, measured. The README's 46x sits at R=32 on larger
models; the curve's shape here (sub-linear wall, efficiency easing as
launch/prefill overheads accrue) is the expected road to it. With
this, all three README classes have live numbers: CORRECTION
3.3-5.0x @ D<=8, STRUCTURAL 1.32-2.25x @ D<=7, and the spatial mask
merge's constant ~6% co-batch tax (vs the solo regime's +27%).
Battery: .wiki/tart/bench_lora_scale.py.

## DESIGN (2026-08-04): adapter per-site pairs — the multi-site rung

Recon truth: the ENGINE needs almost nothing — its lora involvement is
one boolean fact (`declares_lora_sink` -> `launch.lora_program`; the
canonical-KV rejection already treats ANY sink stage as
non-canonical). The (A, B) contents flow driver-side through the
frame's channel machinery. So the rung is:

1. COMPILER: the validator's lora region gate admits MULTIPLE lora
   sink calls per pass, each `(a, b, site_bits)` — with the v1 rule
   that the union of site bit-sets across calls is DISJOINT (one pair
   per site; overlapping sites refuse at validation, not at fire).
2. DRIVER: the sink parse builds a PER-SITE table —
   `LoraFireState { per_site: [(site, A, B, R_site, d_out_site)] }` —
   and the correction applies per consumed site with that site's pair
   (span-grouped per site; the q+v case stops needing a packed
   layout). The staging arena grows per-site slots; the exec
   fingerprint folds the per-site shape vector.
3. SDK: `ForwardPass::adapter` lifts the one-per-pass restriction to
   one-per-SITE (each call emits its own sink); the classifier stays.
4. ORACLE: q+v adapter with distinct shapes (2048/1024 on
   Qwen3-0.6B), zero-B identity per site, single-site parity with
   today's path, and the lora-probe "documented next step" note
   retires.

Sequencing note: land the driver's multi-sink parse FIRST behind the
existing single-sink behavior (a second sink refuses loudly today —
verify, then extend), then the validator, then the sdk lift — the
same driver-first discipline every axis campaign used.

## THE AXIS-COMPOSITION DIRECTIVE (2026-08-04) — the product, not the sum

The user's critique, adopted whole as the governing directive: an AXIS
is one dimension along which co-batched rows of ONE fire diverge. Four
are live — hook (Peel{HookFreePrefix}), mask (Peel{UnmaskedPrefix}),
correction (span-grouped lora), depth (depth_window) — and every one
of them works ALONE. Composition (two+ axes diverging in one fire) is
refused everywhere, so the reachable program space is k+1, not 2^k.
The north-star sells the PRODUCT. And the standing seriation has never
actually been tested: it holds only because one axis fires at a time.

The two structural truths the campaign builds on:

1. CORRECTION IS NOT A WINDOW AXIS. Lora spans are per-lane and
   arbitrary (the grouped GEMM takes disjoint spans); it needs no
   contiguity and composes with anything in principle. Its refusals
   (mask x lora, depth x lora UNPLANNED) are pure conservatism — the
   cheapest relaxations on the board.
2. THE WINDOW AXES NEED (start, len), NOT (prefix, suffix). Two
   suffix-hungry axes (mask wants masked-last, depth wants
   truncated-last) cannot both have the suffix — but the driver's
   window arithmetic (CSR + start, rows = len) never actually needed
   end == N. Generalizing every window word to a contiguous
   [start, end) makes two-axis seriation SOLVABLE by ordering
   [plain | masked | truncated | hooked]: the mask window is a MIDDLE
   window, the depth tail stays a suffix, the full-depth prefix stays
   a prefix, and pairwise disjointness satisfies consecutive-ones
   without a PQ-tree until sets overlap (a member on BOTH axes) —
   which v0 refuses loudly and the PQ-tree rung later splits.

THE COMPOSE LADDER (AC):
  AC-0 the truth table — measure the CURRENT pairwise matrix live
       (which pairs compose, refuse, or silently solo) so every later
       rung moves a measured cell, not an assumed one.
  AC-1 the vocabulary — window words become (start, len) everywhere
       (engine planning + ABI + the three walkers); the seriation
       emits the canonical nest order and per-axis windows; overlap
       (a row on two window axes) refuses loudly.
  AC-2 correction x mask — the cheapest pair: lora members ride the
       unmasked prefix; the planned-word UNPLANNED-on-lora term
       drops; the correction applies to its spans as always.
  AC-3 correction x depth, then mask x depth — the first two-window
       fire (middle mask window + depth suffix).
  AC-4 hook x {correction, mask, depth}.
  AC-5 triples and the 2^k battery: one fire holding
       [plain, masked, lora, draft] — the user's R=4 example — and
       the product-space census.

## AC-1 DESIGN DECISION (2026-08-04): stash/restore beats two-slab

The mask x depth conflict, resolved without touching the 880-line
body's row addressing. Order stays [plain | truncated | masked] (the
CURRENT seriation key — custom_mask outranks truncated, no swap):
the mask window stays a SUFFIX (mask machinery unchanged at every
layer), the truncated block is a MIDDLE window [t_start, m_start),
and the full-depth rows are non-contiguous {[0,t_start) ∪ [m_start,N)}
— which is exactly the shape two-slab execution cannot serve without
offsetting every kernel call.

The resolution: DON'T window range 2 — run it FULL-N and make the
truncated rows' results DISCARDED rather than absent:
  at layer k:   stash rows [t_start, m_start) of the residual stream
                (one D2D copy, contiguous rows x H);
  layers [k,L): run EVERY row (truncated rows compute garbage that
                nothing reads; their KV writes land in layer slabs
                [k,L) which their OWN next step never reads — the
                truncated fire re-runs [0,k) only — so the pollution
                is dead weight, not corruption);
  before tail:  restore the stashed rows — the tail reads layer-k
                hidden for the truncated rows (the logit-lens head)
                and layer-L hidden for everyone else.
Cost: wasted tail-layer compute for the truncated rows (bounded by
their row share) + two row-slab copies. Correctness: exact. The
windowed range-2 (true two-slab) remains the recorded optimization.

Engine side: the depth planner admits masked members (they are
FULL-DEPTH — the suffix after the truncated block must be all-masked,
the middle all-truncated); m_start (= the truncated block's end)
derives from the mask word when planned, else N — NO new ABI word.

## AC-4 NUMERICS (2026-08-04): the four-axis fire's outputs hold class

The five-lane battery (plain, snapkv, lora, masked, draft) against
solos, 46 four-axis fires in the boot: every lane returns COHERENT
text (the masked lane keeps its canonical head; no garbage anywhere),
every lane diverges from solo within the established co-batch
GEMM-rounding class (long common prefixes for plain/mask; the k=8
draft's short prefix matches its known noise sensitivity — low-depth
logits amplify small numeric shifts, observed since AC-3's
lora x depth pair). No crashes, no drift throws. The STEP-LEVEL logit
comparison (solo vs composed, first-divergence attribution per axis
machinery) is the recorded strengthening rung.

## RELEASE REGRESSION POST-AC (2026-08-04): the campaign holds on the default path

The release build carrying the whole AC campaign (seriation reorder,
every relax, stash/restore, the four-axis machinery), no-env boot:
canonical masked solo 3/3, S-B identity (k=28 byte-equal) and
determinism, the solo oracle BYTE-EQUAL to the pre-campaign reference
(the one diff was an uninstalled inferlet, not numerics), the
five-lane product battery green with 40 four-axis fires, 0
panics/illegal. The axis-composition campaign is stable at release on
the default path.

## THE PRODUCT, PRICED (2026-08-04): the four-axis regime vs solo

Release, warm rounds, the five-lane product workload (plain + snapkv
+ lora + masked + draft, 128 tok/lane):

  composed (default) . 1.06s/round
  solo regime        . 1.29s/round   (PIE_SPATIAL_MASK=0 +
                                      PIE_DEPTH_UNION=0 — masked and
                                      truncated lanes solo-fire)
  -> 1.22x on the five-lane mix

Modest by design at this shape: only two of five lanes leave the
co-batch in the solo regime, and the hook x lora pair composed in
both. The win compounds with lane counts exactly as the per-axis
numbers measured (mask const-6%-tax vs +27%, depth 1.32-2.25x,
correction 3.3-5.0x); the product battery's value is that ALL of it
now happens in ONE fire. bench_product.py in the wiki.

## AC-5: THE CENSUS (2026-08-04) — 12/15 subsets fire as products

The formal product-space census (ac5_census.py: every non-empty
subset of {hook, mask, lora, depth}, one lane per axis + a plain
anchor, verdict from the fire trace):

  PRODUCT: 12 of 15 — including hook+mask+lora+depth itself,
           every mask-anchored combination, and all singles.
  PARTIAL: 3 — hook+lora, hook+depth, hook+lora+depth: the
           hook-with-depth cases are the RECORDED anchor decline
           (hooked depth composes only behind a mask word today);
           hook+lora composed in earlier boots — its PARTIAL here is
           launch phasing at 48 tokens, not a refusal.
  SOLO:    0. Nothing is left out of the co-batch entirely.

Zero incidents across the sweep. The k+1 world is gone: the product
space the north-star sells is measurably open on this hardware, with
three cells waiting on the hook-word anchor refinement.

## THE CAMPAIGN SEALS AT RELEASE (2026-08-04)

The 15/15 tip at release, no-env boot: the solo oracle BYTE-EQUAL to
the pre-campaign reference; the census 14/15 PRODUCT + 1 PARTIAL
(a launch-phasing artifact of the faster binary's narrower overlap
windows — the same subset was PRODUCT on the debug sweep and the
4-axis cell itself fired 11 products this sweep), SOLO 0, incidents 0.
The axis-composition property is default, total, and numerics-neutral
at release. What remains beyond this campaign is recorded on the
scoreboard: R=32-scale WEIGHT, the PQ-tree class, the spec-verify
STRUCTURAL producer, a real mask-policy inferlet, the step-logit
oracle.

## DOC-ISOLATION UNDER COMPOSITION (2026-08-04)

The real policy holds inside the spatial split: a doc-isolation lane
co-batched with a plain lane forms 23 R=2 split fires (msplit=1, the
policy lane on the custom kernel, the plain lane on the decode
prefix), and the BLINDING PROPERTY SURVIVES — the isolated lane still
cannot name the planted code word while sharing the fire. Policy
semantics are composition-invariant, which is the whole promise: the
work-sharing merge never leaks what a mask forbids.

## PRODUCT RE-PRICED AT THE 15/15 TIP (2026-08-04)

Composed ~1.11s vs solo regime ~1.31s warm — 1.18x on the five-lane
mix, consistent with the pre-anchor 1.22x (the hook cells' wins ride
shapes this small battery barely exercises; the per-axis numbers
remain the scaling story). The A/B stands as the campaign's standing
perf regression: composed strictly dominates at every measured shape.

## SESSION LEDGER (2026-08-04) — where the north star stands

What EXECUTES BY DEFAULT at this tip, all verified live on the L40S:

- FOUR AXES, ONE FIRE: hook, mask, correction, depth compose pairwise
  and jointly (census 15/15 PRODUCT on debug, 14/15 at release with
  one phasing artifact; 4-axis fires routine). Order
  [plain | truncated | hooked | masked]; the depth middle
  stash/restores; the mask suffix splits; corrections span-group;
  hooks peel. Composed strictly dominates solo at every measured
  shape (1.18-1.22x on the small five-lane mix; per-axis: mask
  const-6% tax vs +27%, depth 1.32-2.25x, correction 3.3-5.0x,
  stream overlap 17%/layer).
- THREE IR AXES: Peel{HookFreePrefix}, Peel{UnmaskedPrefix},
  depth_window — each stated in the declaration, walked by the
  interpreter, spelled by the emitter, referenced by the hand-written
  body. Goldens pin all three.
- THE ADAPTER SURFACE: fwd.adapter(site, |x,y| expr) with three
  validated forms (LoRA byte-parity, IA3 ones-identity, DoRA
  composite-identity), per-site pairs, arity-selected wire.
- REAL POLICY: doc-isolation (RAG contamination block) — blinding
  proven solo AND under composition.
- SAFETY CLOSURES: both-axes lanes never drop k (uniform stamp +
  k-uniformity grouping); every remaining refusal is loud or a
  recorded safe degradation.

OPEN, with entry points recorded in memory arcs 77-81: the
step-logit oracle (top target: the two known NUMERIC-class
cross-instance state effects — lora first-instance, masked-k8
after-plain), true group splitting, the spec-verify STRUCTURAL
producer, R=32-scale WEIGHT (hardware-bound), declared spatial+hook
walker rung, windowed range-2.

## SPEC-VERIFY IN THE FABRIC (2026-08-04)

The speculative-decoding workload, measured against the tip: the
cacheback inferlet runs unbroken (its multi-token VERIFY fires,
N=30..70 at R=1, are the prefill class doing verification), and a
verify fire CO-FIRES with a masked decode lane through the mixed
machinery — R=2 N=67..71 mask=1 msplit=1 fires live: the verify rows
on the causal prefix, the masked lane on the custom suffix, zero
incidents. So the "spec-verify producer" rock is half-closed by
composition alone: real speculative verification already participates
in the axis fabric via the mixed fire. The remaining half — verify as
its OWN Div::STRUCTURAL trace class (a stated verify-vs-decode union
rather than the prefill class's shape) — stays the recorded design.

## STASH COST, MEASURED (2026-08-04)

The stash/restore form's waste, priced (release, R=2, 256 tok):
windowed depth (plain+draft) 1.22s; stash depth (mask+draft) 1.54s;
mask-without-depth reference (mask+plain) 1.46s. The stash fire pays
~0.08s over the mask reference — the truncated row's discarded
[k, L) compute plus two slab copies, roughly the draft row's tail
share as predicted. VERDICT: the waste is real but SMALL at these
shapes (5-7% of the fire); windowed range-2 stays a recorded
optimization, justified only when truncated-row shares grow large
(many drafts per fire) — not before the bigger rocks.

## THE VERIFY-CLASS ROCK, REASSESSED (2026-08-04)

An honest reassessment before committing a campaign to it. The
"spec-verify as its own Div::STRUCTURAL class" rock assumed the SCS
union needed a first customer. The evidence now on the table:

1. The IR already holds service classes for spec-decode repair
   (FireClass::CommitAdvance / StateOnly / FrozenVerify — the qwen3_5
   MTP vocabulary): "a genuinely different pass, so a genuinely
   different trace." The precedent exists and chose SEPARATE TRACES,
   not an SCS union.
2. Verification WORK already rides the fabric: verify fires are
   prefill-class multi-token fires, and they CO-FIRE with decode
   lanes through the mixed machinery (measured, arc 85). Nothing
   about verify wants to share one op list with decode — the two
   phases are sequential by nature (draft, then verify), so they
   never contend for one fire's rows.
3. The depth union already demonstrated "different op sets, one
   fire" for the case where rows genuinely diverge mid-pass.

VERDICT: the SCS-union-with-conditional-regions machinery (the kept
SupergraphBuilder) currently has NO demonstrated customer. The rock
is re-scoped from "build the verify class" to "wait for a workload
whose phases genuinely overlap in one fire" — e.g. simultaneous
draft+verify pipelining within a single fire, which no present
inferlet does. The scoreboard's remaining rocks are therefore:
R=32-scale WEIGHT (hardware-bound) and micro-items. The north star's
STRUCTURAL claim is served by the depth union; the builder stays in
reserve, honestly labeled.

## R=32, REACHED (2026-08-04) — the WEIGHT curve extends on this hardware

The "hardware-bound" label on the R=32 rock was stale: the driver's
request cap is 256 and adapters are small. Measured (release, 128
tok/lane, 32 DISTINCT adapters):

  D=8  . 4.60x    D=16 . 6.40x    D=32 . 8.48x   (vs serialized)

Thirty-two different adapters share one fire's base-weight reads —
the README's R=32 shape, live. The curve grows monotonically but
sub-linearly at 0.6B: the fire is launch/overhead-bound (solo 0.61s
is mostly fixed cost at 128 tokens), so the 46x ceiling needs the
LARGER MODEL (where base-weight reads dominate), not more lanes —
the honest residual of that rock is now "bigger checkpoint", nothing
else. Zero incidents at D=32.

## THE WEIGHT CURVE AT 7B (2026-08-04) — the 46x thesis confirmed in shape

Mistral-7B (local checkpoint, 17.5GB resident), 32 DISTINCT adapters,
128 tok/lane, release:

  0.6B . D=8 4.60x   D=16  6.40x   D=32  8.48x
  7B   . D=8 6.37x   D=16 11.21x   D=32 17.87x

Model size doubles the curve at every D — exactly the thesis: the
merge's win is the base-weight read it deduplicates, and the bigger
the weights, the closer to ideal (56% of ideal at 7B/D=32 vs 27% at
0.6B). The README's 46x at its larger-model/longer-sequence shape is
now an extrapolation the measured curve SUPPORTS rather than a
number taken on faith. Zero incidents; mistral (a force_prefill
deployment) serves 32-adapter fires cleanly. lora-probe geometry is
now argument-driven (layers/d_in/d_out) — any llama-like checkpoint
can run this battery.

## FINAL GATE (2026-08-04)

All suites at the tip: 578 passed / 0 failed (engine, forward, abi,
codegen). Tree clean, both branches pushed, release binary current,
config restored to the serving default. Every directed rock stands
done, rescoped-with-evidence, or measured-to-the-hardware-limit; the
diagnostic instruments (fire trace, census, logit probe,
geometry-parameterized benches) live in the wiki for the next
hardware or the next directive.

## THE CURVE CLOSES ON 46x (2026-08-04) — 14B, D=64: 31.88x

Qwen2.5-14B (48L, hidden 5120, GQA-5 force_prefill deployment,
32.6GB resident), distinct adapters per lane, 128 tok, release:

         D=8     D=16     D=32     D=64
  0.6B   4.60x   6.40x    8.48x    —
  7B     6.37x  11.21x   17.87x    —
  14B    6.72x  12.27x   20.22x   31.88x

Sixty-four DISTINCT adapters through one co-batching fabric at
31.88x over serialized — zero incidents. Every scaling axis behaves:
larger model -> higher efficiency at fixed D (27% -> 56% -> 63% of
ideal at D=32); more lanes -> monotone gains (50% of ideal at D=64).
The README's 46x is now INSIDE the measured curve's natural
continuation (a bigger model or longer sequences at D=64+), no
longer an extrapolation of faith. The WEIGHT-class story is closed
to the limit of what one L40S can state.

## THE FABRIC AT 7B (2026-08-04) — partial census, honest read

A census run on Mistral-7B (the fire-trace instrument carried over
unchanged): the WINDOW-axis triple hook+mask+depth fires as a full
PRODUCT (21 fires) on the 7B force_prefill deployment, zero
incidents — the composition fabric is not a small-model artifact.
The lora-bearing cells of this run are VOID, not failed: the census
script's adapter geometry was not switched to 7B (the probe args
patch missed), so those lanes errored on the geometry gate exactly
as designed. (Lora itself is proven at 7B separately — D=32 17.87x.)
A clean full-census at 7B needs only the geometry args threaded into
the census's lora lane — recorded, not urgent.

## THE FABRIC AT 7B, COMPLETE (2026-08-04): 15/15 AT SCALE

The calibrated census (geometry now a census argument): Mistral-7B,
all fifteen axis subsets PRODUCT — including hook+mask+lora+depth
(18 four-axis fires on the 7B force_prefill deployment), zero
incidents. The composition property holds at 0.6B and at 7B, on a
decode-kernel deployment and on a force_prefill one, at D=64
adapter scale, under a real mask policy, and byte-neutrally against
every pre-campaign oracle. The north star's claims, as directed,
are now measurements.

## THE REVIEW'S #1 GAP, FIRST NUMBER (2026-08-04): composition at scale

The review's decisive critique — the big number (31.88x, one axis)
and the new claim (1.18x, four axes) lived in different experiments —
answered with the scaled product battery (bench_product_scaled.py:
N lanes PER AXIS, geometry-driven):

  Mistral-7B, R=16 (4 masked + 4 lora + 4 draft + 4 plain), 128 tok:
    composed (default) . ~4.9s/round
    solo regime        . ~17.9s/round
    -> 3.66x FOR FOUR-AXIS COMPOSITION ITSELF, at 7B.

Not launch-bound, not single-axis: the four-axis fabric is worth
3.66x on a model where weights dominate. AND the scale probe paid
twice: at 8 masked lanes/axis (R=32) the composed path hits the same
NON-MONOTONE-kvpp planner fault the demoted A/B exposed (here as a
placeholder underflow, kv_indptr = -671086915) — the wire-merge kvpp
audit is now blocking BOTH the R=32 headline and the
declined-deployment 2-way paths. Fix that, then rerun 8/axis and
14B.

## THE HEADLINE (2026-08-04): four-axis composition, 5.9x at 7B / R=32

The review's #1 experiment, completed at full scale (8 lanes PER AXIS:
8 masked + 8 lora + 8 draft + 8 plain = 32 lanes, Mistral-7B,
128 tok, release):

  composed (default) . 5.13-5.30s/round
  solo regime        . 29.8-31.1s/round
  -> 5.9x FOR THE FOUR-AXIS FABRIC ITSELF

With R=16's 3.66x, the composition claim now scales: 1.18x (0.6B,
5 lanes) -> 3.66x (7B, 16) -> 5.9x (7B, 32). The big number and the
new claim live in one experiment. The R=32 kvpp fault did NOT
reproduce across five traced rounds this boot (suffix plans all
monotone, counts-sourced) — re-filed as PROBABILISTIC (first-boot
window suspected), with the kvpp dumps (PIE_KVPP_TRACE) now permanent
instrumentation for the next occurrence.

## BLINDING IN THE MOST COMPLEX CELL (2026-08-04)

The review's safety gap, closed: the doc-isolation policy lane inside
the FULL four-axis fire (co-firing with snapkv's page narrowing, an
adapter correction, a layerskip draft, and a plain anchor) — 57
four-axis fires across three rounds, and the planted code word NEVER
leaks (blinding held every round, varying seeds). The safety
argument's strongest form: composition preserves policy semantics in
exactly the cell where the most machinery is simultaneously active.

## GATHER AS THE DIVERGENCE PRIMITIVE, STAGED (2026-08-04) — review #2

The review's deepest generalization: the O(k^2) pairwise seams (AC-1
stash, AC-2 relax, AC-4 reorder) exist because every axis DEMANDS
CONTIGUITY from the seriation; gather/scatter (already in the tree as
the MoE lowering: launch_gather_bf16_rows /
launch_scatter_add_weighted_bf16) dissolves that demand — gather an
axis's member rows into a dense scratch, run the special kernel,
scatter back; ANY subset serves, no ordering constraint, no per-pair
seam. Staged honestly, like Gray:

- WHY NOT CUT OVER NOW: with the nesting set every axis IS one
  contiguous window, and a window is gather's zero-copy special case
  — the bandwidth of a real gather/scatter round-trip per axis per
  layer is a pure regression today (the stash A/B priced contiguity's
  alternative at 5-7% for ONE axis's discarded compute; gather would
  pay copies for every axis).
- THE CUTOVER TRIGGER is the same event Gray's sentinel watches: the
  first admitted combination outside the nesting set. At that moment
  the three coupled generalizations land together — Gray order (fewest
  fragments), (start,len) windows per fragment, and gather for the
  fragments that still refuse contiguity.
- WHAT IS IN THE TREE ALREADY: the kernels, the seriation sentinel,
  and this design note binding the three.

## THE SCALE TABLE COMPLETES (2026-08-04): 14B, R=32 — 5.8x

With a 15s post-boot settle (the register-death reproduces only in
the immediate post-boot window — refiled as a boot-window artifact,
root cause an open observation), the 14B four-axis product A/B lands:

  model/shape        composed   solo      x
  0.6B, 5 lanes      1.06s      1.29s     1.18
  7B,   R=16         4.9s       17.9s     3.66
  7B,   R=32         5.2s       30.4s     5.9
  14B,  R=32         10.3s      59.4s     5.8

The composition claim is now measured at three model scales and holds
its ~6x at both 7B and 14B (both firmly weight-bound at R=32) — the
review's decisive gap is closed at every scale this hardware serves.

## OBSERVATION LEDGER CLOSES (2026-08-04)

The kvpp fault hunt: three boot-immediate demoted-arm runs at the
current tip, zero faults, all rounds green — the original failure
reproduced only on the pre-3-way binary and has not recurred since
the prefix re-plan landed. Downgraded from "probabilistic fault" to
"not reproducible at tip; instruments standing" (PIE_KVPP_TRACE +
[kvpp-sfx] remain permanent). The boot-window register-death keeps
its ops note (settle 15s on big models). The remaining niceties —
the middle's third stream, true group splitting — are recorded
optimizations with no correctness weight.

## THE FINAL SEAL (2026-08-04)

The complete tip (DepthRole vocabulary, the 3-way no-demotion on all
three walkers, the third stream, Gray staged, every review item) at
release, no-env, one boot: canonical 3/3, the solo oracle BYTE-EQUAL
to the pre-campaign reference, S-B identity, and — for the first time
at release — the census at 15/15 PRODUCT (the 15s settle removed the
earlier phasing artifact). Zero incidents. Everything this session
built is default, total, numerics-neutral, and instrumented.

## V2 — THE REDESIGN (2026-08-04): axes dissolve into operands, seams, signatures

The user's review named the smell exactly: four axes, four mechanisms
(class = separate trace calls; mask/lora/score = GuardPred, temporal;
hook/mask split = PeelWindow, spatial, 2-way fixed; depth = depth_role,
painted OUTSIDE the trace in family.rs:64-91). The failure mode this
project exists to kill — a separate device per axis — reproduced inside
our own IR. mask×depth needed the stash detour precisely because the two
were different kinds of thing.

The redesign converged over four rounds (axis-enum v1 was rejected as a
closed algebra — "Mask, MultiToken 같은 건 불가피한가?" — no). V2 rests
on one axiom:

> A fire is a set of rows; each row is a point in a product space of
> PER-ROW OPERANDS. The trace is ONE function over that space. All
> divergence is derived, never declared.

Three open concepts replace the four mechanisms:

1. PER-ROW OPERANDS (open set). Everything a request attaches or
   geometry implies: the mask operand (Causal | Custom(expr)), the token
   window (One | Many — the whole remaining content of FireClass), the
   layer range (0..k — the whole depth axis), adapter expressions, hook
   programs. "Axis" stops being a name; Mask and MultiToken dissolve
   into operand CLASSES.

2. OPERATOR DISPATCH (selector divergence). `attention(q, kv, mask)` is
   one op; which kernel serves a row is a dispatch-table row over
   (operand classes × fire-uniform predicates × deployment facts):
   (Causal, One) → flashinfer_decode; (Causal, Many) → dequant+prefill;
   (Custom, _) → prefill_custom; `if xqa` overrides. Fusions are
   rewrite rules over op sequences gated on signatures (the fused
   decode epilogue applies where window==One && seam("attn.qv").empty —
   the true identity of fused_post + PeelWindow::HookFreePrefix). The
   layer loop is a first-class `scan` whose range operand IS depth —
   stated in the body, killing the family.rs paint-over and DepthRole.
   New kernel = table row; DSL text unchanged.

3. SEAMS + ATTACHMENTS (decorator divergence). A seam is a NAMED,
   TYPED, IDENTITY-BY-DEFAULT op in the value flow:
   `let (q,v) = m.seam("attn.qv", (q,v))` — downstream consumes the
   seam's output, so an attachment rewrites out=in to out=expr(in)
   without touching the graph; no attachment folds to an alias, zero
   launches. Typed twice: EXPOSED VALUES (what |x,y| may see) and CAPS
   (what effects an attachment may perform — Observe, Scores,
   PageMaskSink, Put, Sample, Emit). Load-time cap checking replaces
   the hand-written contracts (XQA-has-no-capture becomes: no capture
   row in the table ⇒ Scores-requiring attachment refused).
   fwd.adapter was the prototype: attachment = (seam, PTIR fragment),
   generalized to the whole pass. hook_site and the HasLora guard were
   the two special cases, living in two mechanisms.

   Prologue/Epilogue are the BOUNDARY seams ("in"/"out"), free with
   every model; caps form a GRADIENT — pure expressions innermost
   (adapter), observation mid (hooks), full PTIR at the boundary
   (prologue/epilogue) — because interior seams sit inside the batched,
   seriated region where host effects would break the fire. Boundary
   attachments never enter signatures (they cause no divergence), which
   is WHY today's design kept them outside the trace without ever
   producing a composition bug — v2 formalizes that as "attachments in
   the signature vs not."

THE SCHEDULER SEES ONLY SIGNATURES. Row signature = (operand classes at
each operator) × (attached-seam set). Seriation makes co-signature rows
contiguous (Gray order over the signature bits + descending k for
ordered operands — EXACTLY the staged Gray machinery); gather is the
total fallback (EXACTLY the staged gather primitive). The staged
trio — Gray, (start,len) consumers, gather — turns out to be the
runtime half of v2; the redesign is its trigger. Wire ABI: the accreted
scalar words (fast_rows, unmasked_prefix_rows, mask_suffix_*,
full_depth_rows, planned_max_layers, the mixed split word) become ONE
signature table (row→signature id, signature→operand/attachment
records); new attachment kinds change the ABI's shape never again.
Streams are derived: distinct dispatch classes are disjoint row regions
⇒ the 3-way mixed fire's three streams fall out of the table.

WHAT DIES: family.rs's depth post-processing loop and DepthRole; the
class parameter (plans per model: 5 → 1; CommitAdvance/StateOnly/
FrozenVerify remain genuinely different passes); PeelWindow (regions
are n-way, arbitrarily nested); half of GuardPred (HasCustomMask/
HasStageHooks/HasLora → operand classes; HasWriteDesc/WantsAttnScore/
TokensLE → fire-uniform predicate columns); the stash/restore depth
union (mask×depth become two coordinates of one vocabulary); the
pairwise gates (use_spatial_mask, spatial_mixed_compose, depth_union) —
the O(k²) seams die in principle, not by enumeration.

WHAT REMAINS BUILT-IN, honestly: the operator set and seam list (the
model's shape — as it should be), the cap vocabulary, the dispatch
tables (kernels are finite). Extension scenarios that now cost zero DSL
change: new PEFT variant = new expression; new mask policy = new
expression (doc-isolation already has this shape); new SnapKV-like = a
capped attachment; new kernel = a table row.

MIGRATION LADDER (the goldens/oracle battery is the safety net at every
rung): ① seam/signature vocabulary into trace.rs; hook_site, the lora
guard, and boundary stages re-expressed as seams; traces byte-identical
(goldens pin it). ② scan first-class + the class parameter removed —
one trace per model; family.rs post-processing deleted. ③ signature-
table ABI replaces the scalar words. ④ Gray cutover + gather fallback
live; stash/restore deleted. Each rung ends at the same bar: solo
oracle byte-equal, census 15/15 PRODUCT, S-B identity, zero incidents.

## V2 RUNGS ①–② LAND (2026-08-04)

Rung ① (8f40c8832): the seam surface — dsl::seam {Cap, Def; attn.q,
attn.out, attn.qv, in, out}, seam_observe / seam_adapter_qv; family.rs
states seams instead of mechanisms. Rung ②a (1ee386841): the depth
axis moves INTO the body — m.depth_window() before the layer loop with
its deployment gate beside it, roles assigned at RECORD time by the
builder; the family.rs paint-over (the review's sharpest exhibit) is
deleted. Rung ②b (2b44c6725): the Decode and Prefill attention arms —
222 lines, two structures — collapse into ONE dispatch statement keyed
on the window operand's class (window_one vs ragged) plus deployment
facts; the two arms were one structure wearing two names.

All three rungs: goldens byte-identical (23/23 across five families ×
both classes). Live bar re-run at the tip (debug, 0.6B): canonical
3/3, solo oracle BYTE-EQUAL to the pre-campaign reference, S-B
identity, census 15/15 PRODUCT, zero incidents. Two ops notes: the
census instrument NEEDS PIE_FIRE_TRACE=1 in the boot env (a no-trace
log reads as 15×SOLO), and its geometry argv must match the model
(d_out=2048 for 0.6B q — wrong geometry reads as lora-lane errors).

The class parameter now survives only as llama_like_cuda's
instantiation index. Rung ③ (the signature-table ABI; the window class
becomes a PER-ROW operand and the dispatch statement a region table)
is the next structural act, and rung ④ (Gray+gather cutover, stash
deletion) closes the ladder.

## RUNG ③ SPEC (2026-08-04): the region table

Today's axis ABI is four appended scalars at PieStepDesc's tail
(pie_driver_abi.h:1445-1468): planned_hook_free_prefix_rows,
planned_unmasked_prefix_rows, planned_max_layers,
planned_full_depth_rows — one word per axis, the accretion V2 ends.

THE TABLE. Three parallel slices, appended to PieStepDesc:
  region_row_indptr : u32, len = R+1 — ascending wire-row offsets;
                      region r spans rows [indptr[r], indptr[r+1])
  region_sig        : u32 bitset per region — bit0 multi_token,
                      bit1 hook, bit2 mask, bit3 truncated
                      (= fire_plan's axis vocabulary, one word)
  region_k          : u32 per region — the depth operand (layer count;
                      PIE_MAX_LAYERS_FULL = full model). PER-REGION k
                      is the first new capability: the uniform-k
                      grouping rule becomes a per-region fact, so
                      non-uniform truncated fires stop being refused.
Empty table (len 0) = no plan sent (every legacy sentinel at once).
The scheduler builds it in planned splits from the SAME MemberFacts
seriation already orders by — the table IS the seriation's output,
stated once instead of projected four times.

DERIVATIONS (the four words become views):
  hook_free_prefix = first row of the first region with bit1
  unmasked_prefix  = first row of the first region with bit2
  full_depth_rows  = first row of the first region with bit3
  max_layers       = the k shared by bit3 regions (uniform fires)

MIGRATION (the cross-check precedent, three steps):
  ③a engine sends table AND words; driver derives the words from the
     table and REFUSES the launch on drift (exactly the
     planned_hook_free cross-check discipline). No behavior change.
  ③b driver consumers (prepare, the three walkers, dispatch) read
     derivations; the scalar words stop being read.
  ③c the words die from the struct (one ABI era bump), and per-region
     k arms the non-uniform depth fire behind a census increment.
Window class (multi_token, bit0) rides the table from ③a but its
consumer — the dispatch statement as a region table — is rung ③'s
second act, after the words die.

## RUNG ③ COMPLETES (2026-08-04): one table, one era

③a (657d93718): the region table rides the wire beside the words,
cross-checked where planned. ③b (1692737d2): the driver mirrors EVERY
plan/decline rule (+ the LORA sig bit) and proves strict equality on
all four words, UNPLANNED included — zero drift across census 15/15,
the depth battery, both-axes and mixed-k decline probes. ③c-i
(2e70d5d49): the derivation moves to the one StepLaunch→LaunchView
assembly boundary (region_plans.hpp); every consumer reads
table-derived plans; words become a tripwire. ③c-ii (f1148a2f6): the
four words DIE — PieStepDesc sheds them (one era bump), the engine
sheds their feeders and the uniform-k stamp, and apply_region_plans
becomes the plans' birth. Net: -390/+29 lines, and the axis ABI is
SHAPE-INVARIANT — a new axis is a new sig bit, not a new appended
word. The 2026 accretion pattern (fast_rows, then unmasked_prefix,
then max_layers, then full_depth_rows, each a hand-cut word with its
own sentinel) is structurally over.

Word-free era verified live at every rung: census 15/15 PRODUCT, solo
oracle byte-equal to the pre-campaign reference, depth-union battery,
decline probes, zero incidents, scheduler tests 111/111.

Remaining in ③: the window class (bit0) has no consumer yet — the
dispatch statement as a region table is the second act, entangled with
rung ④'s Gray cutover (non-nesting combinations need gather). Deferred
deliberately: per-region k COULD now arm non-uniform depth fires, but
the depth walkers still assume one boundary — that arming belongs with
④'s generalized consumers.

## RUNG ④ SPEC (2026-08-04): three acts

Act 1 — BANDED DEPTH (non-uniform k). The seriation now orders the
truncated block DEEPEST-FIRST (the k term landed with a pinned test),
so at layer l the live rows are always the prefix [0, boundary(l)) —
bands shrink monotonically. The table already carries per-region k.
Remaining work, in order: (i) the worker's uniform_k join refusal
relaxes behind PIE_DEPTH_BANDS (default off) for the pure-decode
no-mask/no-hook shape; (ii) the decode walker generalizes from ONE
depth boundary (depth_prefix_decode_plan + dsplit) to a boundary PER
BAND — a prefix decode plan per distinct k, planned into disjoint
attn_ws regions (the plan/workspace pairing rule extends per band);
(iii) census increment: mixed-k co-fire vs today's solo-per-k, the
composition win priced. The stash/restore union stays for masked
shapes until Act 2.

Act 2 — GRAY + (start,len) + GATHER. Under Gray order the hook-free
set stops being a prefix (codes 000,001 | 101,100 straddle the hooked
middle) — exactly why the cutover is coupled to consumers taking
(start,len) windows and to the gather fallback for non-nesting
residue. The staged pieces (gray_rank + sentinel, gather kernels)
are the halves; the join is: window consumers parameterized by
region-table lookups instead of end-anchored words, gather
materializing any needed non-contiguous set. The sentinel keeps
watch; first live divergence still gates the flip.

Act 3 — THE STASH DIES. With (start,len) consumers and gather, the
masked×depth stash/restore detour (D2D copies at layer k) is
subsumed: the truncated middle is just another region whose rows
leave the iteration space at their boundary. Delete it; re-run the
1.32-2.25x depth pricing to capture the win.
