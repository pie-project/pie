# Inferlet / attention refactor execution plan

## Goal

Land the changes with two primary parallel workstreams while keeping ownership
of `sdk/rust/inferlet/src/ptir.rs`, inferlet examples, runtime mask lowering,
and CUDA mask execution unambiguous.

## Working rules

- Use two long-lived implementation branches/worktrees:
  - Track A owns author-facing SDK/WIT/runtime lifecycle changes and all inferlet
    source migrations.
  - Track B owns runtime mask lowering, CUDA mask execution, and CUDA tests.
- Do not split inferlet example migrations among tasks. Track A updates each
  example once after the final author API is available.
- Do not let Track B change the public `ForwardPass` API or inferlet examples.
- Merge small contract commits before broad migrations so the final integration
  does not need to reconcile two competing API shapes.
- Preserve WIT compatibility for the PTIR `AttnMask` port itself (the container
  registry): the SDK's optional-mask lowering is SDK-only policy and introduces
  no PTIR port change. The inferlet WIT interfaces (channel, pipeline, forward)
  do change — `channel.set` is added and `pipeline.finish` is removed.

## Phase 0: Contract freeze

These contracts are defined at the WIT resource-method level
(`interface/inferlet/forward.wit`, `pipeline.wit`) and mirrored into the
generated WIT inputs; the inferlet Rust SDK is a THIN wrapper over them, not the
source of truth. Owners from both tracks agree on these before implementation:

1. `forward-pass.new()` is an EMPTY constructor (`new() -> forward-pass`). It no
   longer takes container bytes, channels, working sets, or page spans. Binding
   is builder-style through resource methods on the pass: `attention(...)`,
   `embed(...)`, the traced program / container-bytes attachment, and
   `submit(on)`. `forward-pass.attention(...)` is the sole attention binding
   method. Its signature is FLAT (individual `borrow<channel>` arguments, no
   wrapper record): `kv-working-set`, readable/writable `page-span`s, every
   geometry channel (`kv-len`, `pages`, `page-indptr`, `w-slot`, `w-off`,
   `positions`), and the optional mask. `ForwardPass::attention` in Rust is a
   thin wrapper over this WIT method.
2. The mask argument is `Option<&Channel>` (WIT `option<borrow<channel>>`).
   There is NO `AttentionMask` enum. `None` means causal and emits no PTIR
   `AttnMask` port. `Some(channel)` binds the supplied channel to the existing
   PTIR `AttnMask` port. No WIT PTIR port change is introduced.
3. Every `attention()` descriptor-port input is a `Channel`, including the
   inputs that are trace constants today (`Readout`, `EmbedIndptr`). `Tensor`
   remains valid only inside traced stage computation (prologue / attn / attn
   proj / epilogue closures).
4. The engine NEVER infers geometry. There is no dense-geometry derivation and
   no engine-owned geometry materialization. The author supplies every geometry
   channel to `attention()` and updates it per fire in the epilogue (the
   chat-completion / beam-search style). Rationale: geometry is not always a
   function of `KvLen` — beam-search computes page layout via a fork/prune tree
   over a shared pool, which no engine derivation can produce. One explicit path
   serves both the common contiguous-pool case and the exotic cases.
5. `Channel::put` remains lossless/backpressured queue insertion. `Channel::set`
   replaces the current committed (front) cell in place — a fused take+put — and
   does NOT enqueue. It leaves any staged (next-fire) put untouched.
6. `Pipeline::finish` is REMOVED. `close` is the single end-of-stream + teardown
   verb: it releases the scheduler wait-set immediately, all already-submitted
   fires (queued / preparing / dispatched) drain to settlement and stay
   take-able, and only never-submitted future work is dropped. `drop == close`.
   No reusable `drain` primitive is introduced.
7. Guest model configuration is removed. `num_layers` and the gate flags
   (`has_mtp_logits` / `has_value_head`) are bind-only metadata the engine
   already owns through its bound `ModelProfile`; production binding ignores the
   guest's values today, so they are deleted from the author surface (a
   `#[cfg(test)]` profile hook stays in `ptir-dsl` for standalone golden tests).
   `vocab` and `page_size` are trace-specializing (they shape SSA tensor shapes
   / geometry) but the SDK sources them from the engine at trace time via the
   existing host calls (`output_vocab_size`, `kv_page_size`), so `model::configure`
   leaves the author surface.
8. Because item 3 makes every custom mask a `Channel` (the const-mask path is
   gone), the runtime must classify each masked fire per-fire: a HOST-DERIVABLE
   mask (its value is known to the host shadow) lowers to the existing wire BRLE
   encoding and co-batches; a DEVICE-DERIVED mask (value produced by an epilogue
   put, host-unknown) keeps the device channel path. This makes Track B's B1 a
   REQUIRED consequence of item 3, not an independent optimization: without it,
   channelizing masks would push every currently-co-batchable mask onto the solo
   device path.
9. CUDA graph cache keys distinguish mask-aware and mask-free captures, and all
   captured mask pointers refer to persistent buffers.

Exit criterion: API signatures, mask classification, and close/drain semantics
are written in the implementing commits/tests with no open semantic questions.

## Track A: Author API and lifecycle

Owns the author-API and lifecycle tasks.

### A1. Introduce channel state replacement

Scope:

- Add `channel.set` to canonical inferlet WIT and mirrored/generated WIT inputs.
- Implement host/runtime front-cell replacement (fused take+put) WITHOUT changing
  `put` queue semantics and WITHOUT touching a staged next-fire put.
- Add `Channel::set` to the Rust inferlet wrapper.
- Add focused tests for empty/full front cell, repeated replacement, a queued
  `put` present, in-flight use, poison/error handling, and capacity greater than
  one (set targets the committed front, not the whole queue).

Likely files:

- `interface/inferlet/forward.wit`
- `sdk/rust/inferlet/wit/forward.wit`
- `sdk/tools/bakery/src/bakery/wit/forward.wit`
- `sdk/rust/inferlet/src/ptir.rs`
- `runtime/engine/src/inferlet/host/forward.rs`
- `runtime/engine/src/pipeline/channel.rs`

### A2. Remove finish; fold end-of-stream into close

Scope:

- Delete `finish` from WIT and the SDK.
- Move the scheduler wait-set release into `close`: releasing the wait-set the
  moment `close` runs, while already-submitted queued/preparing/dispatched fires
  drain to settlement and remain take-able. Only never-submitted work is dropped.
  This is a semantic change from today's `close` (which cancels queued/preparing
  fires); no example needs to kill an already-submitted-but-preparing fire, so
  nothing is lost.
- Keep `drop == close`.
- Add lifecycle tests: submit -> close -> drain-takes settle; close on an empty
  pipeline; submit-after-close errors; run-ahead tail is fully take-able after
  close.

Likely files:

- `interface/inferlet/pipeline.wit`
- WIT mirrors
- `sdk/rust/inferlet/src/ptir.rs`
- `runtime/engine/src/inferlet/host/pipeline.rs`
- `runtime/engine/src/pipeline/fire.rs`
- scheduler worker/quorum tests and pipeline tests

### A3. Land the unified attention API

Scope:

- Reshape the `forward-pass` WIT resource first: make `new()` an empty
  constructor (`new() -> forward-pass`) and move container bytes / channels /
  working sets / page spans off `new` onto builder-style resource methods
  (`attention`, `embed`, program attachment, `submit`). The old all-at-once
  `new(container-bytes, channels, kv-working-set, readable-pages, writable-pages,
  rs-working-sets)` signature is deleted.
- Add one FLAT `forward-pass.attention(...)` (WIT resource method) taking the
  working set, readable / writable page spans, every geometry channel
  individually, and the optional mask. The Rust `ForwardPass::attention` wraps
  it thinly.
- Lower `None` by omitting `Port::AttnMask` (causal); lower `Some(channel)` by
  binding the channel to the existing `AttnMask` port.
- Remove `attn_working_set`, `attn_mask`, `port_channel`, `port_const`, and every
  public geometry setter — `attention` is the only binding surface.
- Remove `PortSpec::Const`, `GeometryInput::Const`, the `Indptr` Tensor/Channel
  polymorphism, and all other descriptor-port `Tensor` inputs. `Readout` and
  `EmbedIndptr` become channels.
- Remove ALL geometry inference: delete `derive_dense_geometry`,
  `derive_dense_geometry_with_page_capacity`, the `dense_page_capacity` field,
  `materialize_geometry`, and `AutoGeometry`. `page_capacity` disappears with
  them — the author declares a pool-sized geometry channel and refreshes its
  value each fire.
- Retain `Tensor` for SSA values inside stage closures only.
- Decide and document whether `embed` and `readout` stay as separate binders
  (with their `Tensor` args converted to channels) or fold into `attention`.
  Default: keep `embed` / `readout` separate; only channelize their inputs.

Primary owner file:

- `sdk/rust/inferlet/src/ptir.rs`

Supporting files:

- `sdk/rust/ptir-dsl` only where builder constraints must enforce channel-only
  descriptor inputs
- inferlet SDK unit tests
- forward WIT documentation where semantics changed

### A4. Remove inferlet-owned model configuration

Scope:

- Remove `num_layers` and the gate flags from the author surface entirely: they
  are consumed only through the bound `ModelProfile` (`bound.profile.num_layers`)
  at bind/fire time, and production binding already uses the engine profile, not
  the guest's. Keep a `#[cfg(test)]` profile hook in `ptir-dsl` for standalone
  golden tests that must bind explicitly.
- Source `vocab` and `page_size` from the engine at trace time: `intrinsics::vocab()`
  reads the host `output_vocab_size()`; `intrinsics::page_size()` reads the
  engine `kv_page_size`. Remove `model::configure` / `configure_gates` from the
  author surface.
- Remove all inferlet-side configure calls and hard-coded model metadata.
- Add a regression test where guest assumptions differ from the bound model and
  confirm the engine-owned profile wins.

Likely files:

- `sdk/rust/inferlet/src/ptir.rs`
- `sdk/rust/ptir-dsl/src/model.rs`
- `sdk/rust/ptir-dsl/src/intrinsics.rs`
- `runtime/engine/src/inferlet/host/model.rs`
- model capability plumbing/tests

### A5. Migrate inferlets once

After A1-A4 APIs compile, perform one repository-wide migration:

- Remove `model::configure` / `configure_gates` calls.
- Replace all old attention/port/geometry calls with the one `attention(...)`.
- Pass `None` for causal or `Some(&channel)` for a custom mask at every pass.
- Replace host `take` + `put` front-cell-replacement idioms with `set`.
- Remove all `derive_dense_geometry` / `derive_dense_geometry_with_page_capacity`
  calls. For the two fully-derived inferlets (`text-completion-bench`, `generate`),
  add the explicit geometry recurrence to their decode epilogues (they currently
  rely on derivation; every other inferlet already declares geometry explicitly).
- Remove `finish` calls; rely on `close` (drain-then-teardown) after required
  output takes/settlement.
- Update directly related guide/examples.

Do not start this step before the API contract is final. Migrate 2-3 canary
inferlets (causal prefill, device-mask sliding-window decode, fork/prune
beam-search) DURING A3 to prove the flat `attention` signature covers every real
pattern before the contract is frozen.

Track A exit criteria:

- No inferlet author callsites remain for removed APIs.
- No descriptor port accepts a `Tensor`.
- Every attention pass passes `None` or `Some(&channel)` explicitly.
- close-based end-of-stream passes runtime scheduler tests.
- All Rust inferlet builds/tests pass.

## Track B: Mask lowering and CUDA graphs

Owns the mask-lowering and CUDA-graph tasks. B1 is a REQUIRED consequence of A3
(Phase 0 item 8), so it must be designed against A3's channel-only mask lowering,
not in isolation.

### B1. Classify and lower host-derivable masks

Scope:

- During fire preparation, distinguish a host-derivable `AttnMask` channel (its
  value is known to the host shadow) from a genuinely device-derived mask (fed by
  an epilogue put, host-unknown). Classification is per fire — the same pass can
  be host-known on its seed fire and device-derived on later fires.
- Evaluate/read the host-derivable mask through the runtime host shadow.
- Encode it into the existing wire `flattened_masks` / `mask_indptr` BRLE form.
- Stop setting the device-dense-mask / solo-batch classification for this case.
- Preserve explicit user-mask semantics while allowing compatible wire-mask fires
  to co-batch.
- Keep device-derived `AttnMask` on the current dense device path.
- Add tests for causal omission (`None`), host-derived custom masks, device-
  derived custom masks, mixed co-batches, and parity with the old dense path.

Likely files:

- `runtime/engine/src/pipeline/fire/geometry.rs`
- `runtime/engine/src/pipeline/fire.rs`
- `runtime/engine/src/pipeline/fire/kv.rs`
- `runtime/engine/src/pipeline/instance.rs`
- scheduler batch classification
- launch-plan serialization tests

### B2. Make custom-mask execution graph-safe

Scope:

- Remove `!have_custom_mask` from graph eligibility once mask inputs are safe.
- Ensure custom mask and mask indptr data always live at persistent addresses
  (the destination `pi.custom_mask` buffers are already persistent; the dense-mask
  SOURCE staging buffers, allocated per fire today, must be made persistent or
  packed in-graph).
- Pass persistent mask pointers through graph capture and replay.
- Add a mask-presence bit to `graph_variant` so mask-aware and mask-free graphs
  cannot alias.
- Support the custom-mask prefill fallback used by decode-shaped fires.
- Ensure graph padding and graph eligibility use the same mask-aware predicate.
- Verify TP broadcast paths preserve mask sizes/data before replay.

Likely files:

- `driver/cuda/src/batch/compose.cpp`
- `driver/cuda/src/batch/forward.cpp`
- `driver/cuda/src/batch/forward.hpp`
- `driver/cuda/src/batch/forward_graph.hpp`
- `driver/cuda/src/batch/graph_variant.hpp`
- `driver/cuda/src/batch/persistent_inputs.{hpp,cpp}`
- CUDA graph/masked-attention tests

Track B exit criteria:

- Host-derived mask fires can co-batch.
- Device-derived mask behavior remains correct.
- Causal (`None`), wire-custom, and device-custom paths all pass parity tests.
- Repeated custom-mask executions replay a captured graph using persistent
  buffers.
- Graph keys cannot alias between mask-aware and mask-free variants.

## Integration order

1. Merge A1 and A2 contract/runtime commits without broad example edits.
2. Merge A3 and A4 author API commits. A3's channel-only mask lowering is the
   input B1 depends on — treat A3 -> B1 as a hard pipeline, not parallel work.
3. Rebase Track B onto the integrated A3 mask lowering.
4. Merge B1, then B2.
5. Run A5 once against the integrated API/lowering behavior.
6. Resolve documentation/generated binding drift.
7. Run the full validation matrix.

If three workers are required, split Track A only by strict file ownership:

- A-core: A3/A4 and sole ownership of `sdk/rust/inferlet/src/ptir.rs`.
- A-runtime: A1/A2 runtime and WIT implementation, but no SDK wrapper/example
  edits.
- B: B1/B2.

A-core performs all SDK wrapper wiring and A5 migration after A-runtime lands.

## Validation matrix

Run existing repository commands only, in increasing cost:

1. Format/check changed Rust crates and generated WIT bindings.
2. `ptir-dsl` unit/golden tests.
3. inferlet SDK and engine unit tests.
4. compile every inferlet/example crate.
5. pipeline scheduler/run-ahead tests, especially close end-of-stream + take
   drain.
6. runtime launch-plan/mask serialization tests.
7. CUDA masked-attention parity and graph-key tests.
8. CUDA end-to-end causal, custom prefill, custom decode fallback, and
   co-batching tests.
9. Full repository test/build command used by CI.

## Final acceptance checklist

- `model::configure` / `configure_gates` are absent from inferlet author code;
  `num_layers` / gates are engine-owned; `vocab` / `page_size` are SDK-sourced
  from the engine.
- `Channel::set` replaces the front cell; `put` remains lossless/backpressured.
- No geometry inference remains: `derive_dense_geometry`,
  `derive_dense_geometry_with_page_capacity`, and `page_capacity` are absent from
  the API and callsites; the engine infers no geometry.
- `forward-pass.new()` takes no arguments; all binding is builder-style resource
  methods (`attention`, `embed`, program attachment, `submit`).
- `forward-pass.attention` (WIT resource method) is the only attention binding
  surface, with a flat signature; the Rust wrapper is thin.
- The mask argument is `Option<&Channel>`; `None` is causal, `Some` binds the
  `AttnMask` port. No `AttentionMask` enum.
- No public forward descriptor port accepts a `Tensor` (including `Readout` /
  `EmbedIndptr`).
- Host-derivable custom masks use wire encoding and co-batch; device-derived
  masks keep the device path.
- Every valid mask path supports CUDA graph capture/replay.
- `Pipeline::finish` is gone; `close` is the single end-of-stream + teardown verb
  and releases the wait-set while submitted fires drain.
- Inferlet examples call only `close`.
