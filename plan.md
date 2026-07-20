# Inferlet / attention refactor execution plan

## Goal

Land the nine changes with two primary parallel workstreams while keeping
ownership of `sdk/rust/inferlet/src/ptir.rs`, inferlet examples, runtime mask
lowering, and CUDA mask execution unambiguous.

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
- Preserve WIT compatibility for the `AttnMask` port itself; the author-facing
  `AttentionMask` enum is SDK-only lowering policy.

## Phase 0: Contract freeze

Owners from both tracks agree on these contracts before implementation:

1. `ForwardPass::attention(...)` is the sole author-facing attention binding
   operation. It accepts the working set, readable/writable page spans, all
   required geometry channels, and an explicit `AttentionMask`.
2. `AttentionMask` exposes only:
   - `Causal`
   - `Custom(&Channel)`
3. `Causal` emits no PTIR `AttnMask` port. `Custom` binds the supplied channel to
   the existing PTIR `AttnMask` port; no WIT port change is introduced.
4. Every `ForwardPass` descriptor-port input is a `Channel`. `Tensor` remains
   valid only inside traced stage computation.
5. Dense geometry is automatic/engine-owned. There is no guest opt-in method.
6. `Channel::put` remains lossless/backpressured queue insertion.
   `Channel::set` replaces current host-visible state rather than enqueueing.
7. `Pipeline::drain` is reusable: it releases the scheduler wait-set after all
   prior fires dispatch, but does not reject later submissions. `close` remains
   terminal/cancelling.
8. Host-derivable custom masks lower to the existing wire encoding and are
   eligible for ordinary co-batching. Device-derived masks keep the device
   channel path.
9. CUDA graph cache keys distinguish mask-aware and mask-free captures, and all
   captured mask pointers refer to persistent buffers.

Exit criterion: API signatures, mask classification, and drain semantics are
written in the implementing commits/tests with no open semantic questions.

## Track A: Author API and lifecycle

Owns tasks 1-6 and 9.

### A1. Introduce channel state replacement

Scope:

- Add `channel.set` to canonical inferlet WIT and mirrored/generated WIT inputs.
- Implement host/runtime state replacement without changing `put` queue
  semantics.
- Add `Channel::set` to the Rust inferlet wrapper.
- Add focused tests for empty/full state, repeated replacement, queued `put`,
  in-flight use, poison/error handling, and capacity greater than one.

Likely files:

- `interface/inferlet/forward.wit`
- `sdk/rust/inferlet/wit/forward.wit`
- `sdk/tools/bakery/src/bakery/wit/forward.wit`
- `sdk/rust/inferlet/src/ptir.rs`
- `runtime/engine/src/inferlet/host/forward.rs`
- `runtime/engine/src/pipeline/channel.rs`

### A2. Replace terminal finish with reusable drain

Scope:

- Rename WIT/SDK `finish` to `drain`.
- Replace the terminal `finished` bit/notification behavior with an ordered,
  reusable drain marker or generation that leaves the scheduler wait-set only
  after preceding fires dispatch.
- Ensure later submissions re-enter scheduling normally.
- Keep `close` terminal and cancelling.
- Add lifecycle tests covering submit -> drain -> submit -> drain, drain on an
  empty pipeline, close after drain, and submit after close.

Likely files:

- `interface/inferlet/pipeline.wit`
- WIT mirrors
- `sdk/rust/inferlet/src/ptir.rs`
- `runtime/engine/src/inferlet/host/pipeline.rs`
- `runtime/engine/src/pipeline/fire.rs`
- scheduler worker/quorum tests and pipeline tests

### A3. Land the unified attention API

Implement tasks 3-6 together:

- Add `AttentionMask::{Causal, Custom(&Channel)}`.
- Add one `ForwardPass::attention(...)`.
- Remove `attn_working_set`, `attn_mask`, raw `port_channel`/`port_const`, and
  public geometry setters superseded by `attention`.
- Remove `PortSpec::Const`, `GeometryInput::Const`, `Indptr` Tensor/Channel
  polymorphism, and other descriptor-port Tensor inputs.
- Retain `Tensor` for SSA values inside stage closures only.
- Make geometry materialization automatic when the pass is bound; remove
  `derive_dense_geometry()` and its page-capacity variant.
- Lower `Causal` by omitting `Port::AttnMask`; lower `Custom` by binding the
  channel to the existing port.
- Keep working-set replacement behavior available through the unified API or a
  clearly scoped post-bind update path without restoring mixed setters.

Primary owner file:

- `sdk/rust/inferlet/src/ptir.rs`

Supporting files:

- `sdk/rust/ptir-dsl` only where builder constraints must enforce channel-only
  descriptor inputs
- inferlet SDK unit tests
- forward WIT documentation where semantics changed

### A4. Remove inferlet-owned model configuration

Scope:

- Stop re-exporting or exposing `ptir_dsl::model::configure` and
  `configure_gates` to inferlet authors.
- Source trace/bind metadata from engine-owned model information.
- Keep any low-level `ptir-dsl` configuration hook test-only if standalone DSL
  tests still require an explicit profile.
- Remove all inferlet-side configure calls and hard-coded model metadata.
- Add a regression test where guest assumptions differ from the bound model and
  confirm the engine-owned profile wins.

Likely files:

- `sdk/rust/inferlet/src/ptir.rs`
- `sdk/rust/ptir-dsl/src/model.rs`
- `runtime/engine/src/inferlet/host/model.rs`
- model capability plumbing/tests

### A5. Migrate inferlets once

After A1-A4 APIs compile, perform one repository-wide migration:

- Remove `model::configure` calls.
- Replace old attention/port/geometry calls with `ForwardPass::attention`.
- Choose `AttentionMask::Causal` or `Custom` explicitly at every attention pass.
- Replace host `take` + `put` state-replacement idioms with `set`.
- Remove all `derive_dense_geometry` calls.
- Remove `finish` calls.
- Inferlet examples use `close()` only, after required output takes/settlement.
- Update directly related guide/examples.

Do not start this step before the API contract is final.

Track A exit criteria:

- No inferlet author callsites remain for removed APIs.
- No descriptor port accepts a `Tensor`.
- Every attention pass makes an explicit mask choice.
- Reusable drain behavior passes runtime scheduler tests.
- All Rust inferlet builds/tests pass.

## Track B: Mask lowering and CUDA graphs

Owns tasks 7 and 8.

### B1. Classify and lower host-derivable masks

Scope:

- Distinguish a host-derivable `AttnMask` channel from a genuinely
  device-derived mask during fire preparation.
- Evaluate/read the host-derivable mask through the runtime host shadow.
- Encode it into the existing wire `flattened_masks`/`mask_indptr` form.
- Stop setting the device-dense-mask/solo-batch classification for this case.
- Preserve explicit user-mask semantics while allowing compatible wire-mask
  fires to co-batch.
- Keep device-derived `AttnMask` on the current dense device path.
- Add tests for causal omission, host-derived custom masks, device-derived
  custom masks, mixed co-batches, and parity with the old dense path.

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
- Ensure custom mask and mask indptr data always live at persistent addresses.
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
- Causal, wire-custom, and device-custom paths all pass parity tests.
- Repeated custom-mask executions replay a captured graph using persistent
  buffers.
- Graph keys cannot alias between mask-aware and mask-free variants.

## Integration order

1. Merge A1 and A2 contract/runtime commits without broad example edits.
2. Merge A3 and A4 author API commits.
3. Rebase Track B if shared runtime mask-classification code changed.
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
5. pipeline scheduler/run-ahead tests, especially reusable drain and close.
6. runtime launch-plan/mask serialization tests.
7. CUDA masked-attention parity and graph-key tests.
8. CUDA end-to-end causal, custom prefill, custom decode fallback, and
   co-batching tests.
9. Full repository test/build command used by CI.

## Final acceptance checklist

- `model::configure` is absent from inferlet author code.
- `Channel::set` replaces state; `put` remains lossless/backpressured.
- `derive_dense_geometry` is absent from the inferlet API and callsites.
- `ForwardPass::attention` is the only attention binding surface.
- Public mask choices are exactly `Causal` and `Custom(&Channel)`.
- No public forward descriptor port accepts a `Tensor`.
- Host-derivable custom masks use wire encoding and co-batch.
- Every valid mask path supports CUDA graph capture/replay.
- `Pipeline::drain` is reusable; `close` is terminal.
- Inferlet examples call only `close`, not `drain` or removed `finish`.
