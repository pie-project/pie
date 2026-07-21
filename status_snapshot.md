# Inferlet / attention refactor status snapshot

Last updated: 2026-07-20

## Resume instructions

- Continue directly in this worktree. The user explicitly requested no
  subagents for the remaining work.
- Do not discard or overwrite the dirty working tree.
- Do not reintroduce any removed author API (`model::configure`,
  `derive_dense_geometry`, `attn_working_set`, `port_channel`,
  `Pipeline::finish`, etc.).
- Preserve the contention redesign documented in
  `KV_CONTENTION_ORCHESTRATOR_GAP.md`: demand/acquire must remain before fire
  leases and transactions, allocation waiters outrank restores, and host-swap
  exhaustion kills a quiesced victim.

## Git state

- Repository: `/home/ingim/Workspace/pie-wt/ptir-fusion-echo`
- Branch: `tasks/ptir-fusion/agents/echo`
- Implementation commit: `fc692e5f` (`Complete inferlet lifecycle refactor`)
- The implementation commit is pushed to both GitHub and lifecycle origin.
- This snapshot is a docs-only descendant of that implementation commit.
- The worktree is clean at handoff.

Already committed and pushed to both remotes:

| Commit | Purpose |
| --- | --- |
| `6a05cb27` | A1: atomic `Channel::set` |
| `5ccfcbcf` | A2: close-only graceful pipeline lifecycle |
| `d09dc3d7` | A3/A4: builder-style forward API and engine-owned model metadata |
| `07b4ec78` | B1: per-fire host/device attention-mask lowering |
| `144646db` | B2: persistent masked CUDA graph replay |
| `62f1fb4d` | Integrated audit fixes: set visibility, close behavior, mask-only CUDA resolution, explicit geometry enforcement, authoritative layer metadata |
| `fc692e5f` | Complete A5 migration, contention lifecycle, pipeline-scope identity, tests, and CUDA profiling harness |

All implementation changes are committed and pushed.

## Completed work

### A1: `Channel::set`

- Added canonical and mirrored WIT method.
- Added Rust SDK wrapper and runtime implementation.
- `set` replaces the committed front cell without enqueueing and leaves a
  staged/queued `put` untouched.
- Seeded descriptor-only channels derive a device-visible Writer endpoint so
  host replacement reaches the driver ring.
- Empty, repeated, capacity, in-flight, poison, close, and queued-put behavior
  is covered by unit tests.

### A2: close-only pipeline lifecycle

- Removed `Pipeline::finish` from WIT, mirrors, SDK, host glue, and author code.
- `close` marks the scope closed, releases scheduler waiting, rejects later
  submissions, opportunistically finalizes settled work, and returns without
  waiting on an output-ring-blocked fire.
- Already-submitted work is not cancelled.
- `drop == close`.
- Host `take` now settles the corresponding front fire before consuming its
  output, preventing a close/rebind race and ensuring drain-takes finalize KV/RS
  transactions.

### A3: unified forward API

- `forward-pass.new()` is empty.
- Binding is builder-style through `embed`, flat `attention`, `readout`,
  program attachment, RS working-set attachment, and `submit`.
- `ForwardPass::attention` is the only public attention-binding method.
- Every geometry descriptor is a `Channel`.
- Mask is `Option<&Channel>`:
  - `None` omits `AttnMask` and is causal.
  - `Some(channel)` binds the existing PTIR `AttnMask` port.
- No public `AttentionMask` enum exists.
- Removed author-facing tensor/const descriptor polymorphism and old geometry
  setters.
- Removed runtime dense geometry derivation/default materialization.
- Pool-sized rank-1 `Pages` channels are compacted using the explicit
  `PageIndptr` live prefix.
- Decode-envelope classification accepts explicit flat or rank-2 page pools and
  channel-backed `Readout`.

### A4: engine-owned model metadata

- Removed guest `model::configure` and gate configuration.
- `vocab` and page size come from engine host calls at trace time.
- `num_layers` and model gates come from the bound engine profile.
- Missing layer metadata now fails model registration rather than silently
  becoming one layer.
- Added `num_hidden_layers` to mock/smoke fixture metadata.

### A5: repository-wide inferlet migration

Completed across both inferlet workspaces:

- `runtime/engine/tests/inferlets`
- `tests/inferlets`

All migrated inferlets now:

- use channel-backed `EmbedIndptr`, `Readout`, and attention geometry;
- call the flat `attention(...)` API;
- pass `None` or `Some(&mask)` explicitly;
- materialize page pools and page-indptr recurrence in author code;
- update positions, lengths, write slots/offsets, and dynamic page CSR in
  epilogues;
- use only `Pipeline::close`;
- no longer call model configuration APIs.

Both fully-derived paths, including `generate` and
`text-completion-bench`, now have explicit decode geometry recurrence.

Legacy author-call search is clean. The only `.finish()` match under inferlet
sources is `Hasher::finish()` in greenlist watermarking.

### B1: mask classification/lowering

- Classification is per fire.
- Host-shadow-known masks lower to BRLE wire masks and remain co-batchable.
- Device-derived masks stay on the dense device path.
- Mask-only descriptor resolution is independent of geometry class.
- Causal omission remains mask-free.

### B2: CUDA graph-safe masks

- Mask source/destination data uses persistent storage.
- Mask-aware and mask-free graph variants cannot alias.
- Masked decode-shaped execution uses the custom-prefill fallback.
- Graph eligibility and graph padding use the same mask-aware behavior.
- TP payload handling preserves mask sizes/data.

### KV contention redesign

- Ordinary and explicit/device-geometry fires compute physical KV demand before
  acquiring fire leases, RS/KV transactions, completion cells, or channel
  tickets.
- Contended demand enters `ContentionOrchestrator` with no preparation pins.
- Grant-aware backing/COW APIs consume or return every reserved page exactly
  once.
- Allocation waiters outrank restores.
- Accepted scheduler work drains before victim suspend/freeze.
- Victims are youngest-first among processes at the pin-free allocation
  boundary; FCFS-oldest remains protected.
- Restore is retried on page-free events, with utilization pause/aging and a
  50 ms fallback poll.
- Zero host swap is valid: an unswappable quiesced victim is terminated, then
  its device pages return through normal native teardown.
- Stale park requests are cancelled when allocation pressure clears.
- Scheduler identity is split: process IDs own suspend/terminate, while each
  `PipelineScope` has an independent quorum ID for close and allocation wait.
  A process-wide suspend removes all owned scopes; one scope leaving does not
  perturb sibling pipelines.
- Full design and remaining limitations are documented in
  `KV_CONTENTION_ORCHESTRATOR_GAP.md`.

## Important integrated runtime/test fixes

In addition to inferlet migrations, `fc692e5f` contains:

- `runtime/engine/src/inferlet/host/forward.rs`
  - take-driven FIFO settlement before output consumption.
- `runtime/engine/src/pipeline/fire/geometry.rs`
  - no dense/default geometry inference;
  - explicit flat-page-pool compaction;
  - channel-backed Readout decode-envelope support.
- `driver/dummy/src/lib.rs`
  - low-level test containers now satisfy complete geometry validation;
  - endpoint tests reflect allowed private device-channel sharing.
- `bin/pie/tests/boot_smoke.rs` and
  `bin/pie/tests/fixtures/smoke-model-ascii/config.json`
  - boot smoke constructs a minimal valid snapshot with authoritative model
    metadata and a minimal safetensors header.
- `KV_CONTENTION_ORCHESTRATOR_GAP.md`
  - detailed design/problem statement for the separate KV overcommit issue.

## Validation completed

Passed:

| Validation | Result |
| --- | --- |
| WIT canonical/Rust/Bakery mirrors | Byte-identical |
| `git diff --check` | Passed |
| `ptir-dsl` unit/golden/doctests | Passed |
| Inferlet Rust SDK check | Passed |
| Model unit tests | 101 passed |
| Engine library tests | 354 passed |
| Engine test-inferlet workspace check | Passed |
| Repository inferlet workspace check | Passed |
| Inferlet canary | 2 passed, 3 intentionally ignored |
| Engine e2e | 10 passed |
| Direct-channel close/take/rebind regression | Passed |
| Standalone boot smoke | Passed |
| Dummy-driver unit tests | 13 passed |
| Worker library tests | 60 passed, 1 ignored |
| CUDA driver build | Passed with `PIE_COMPILER_LAUNCHER=env`, `-j4` |
| CUDA mask/graph tests | 5/5 passed: pack, BRLE, geometry, parity, graph variant |
| Workspace all-target build (excluding `pie-server-py`) | Passed |
| Active-preemption fleet | Passed repeatedly, 3/3 |
| Zero-host-swap victim-kill fleet | Passed repeatedly, 3/3 |
| Full serialized workspace test suite | Passed |

Useful validated commands:

```bash
cargo test --manifest-path sdk/rust/ptir-dsl/Cargo.toml --quiet -j4
cargo check --manifest-path sdk/rust/inferlet/Cargo.toml --quiet -j4
cargo test -p pie-model --lib --quiet -j4
cargo test -p pie-engine --lib --quiet -j4
cargo check --manifest-path runtime/engine/tests/inferlets/Cargo.toml --workspace --quiet -j4
cargo check --manifest-path tests/inferlets/Cargo.toml --workspace --quiet -j4
cargo test -p pie-engine --test inferlet_canary --quiet -j4
cargo test -p pie-engine --test e2e --quiet -j4
cargo test -p pie-bin --test boot_smoke --quiet -j4
cargo test -p pie-driver-dummy --lib --quiet -j4
cargo test -p pie-worker --lib --quiet -j4
PIE_COMPILER_LAUNCHER=env \
  cargo build -p pie-worker --no-default-features --features driver-cuda -j4 --quiet
```

CUDA targeted tests:

```bash
CUDA_BUILD_DIR="$(find target/debug/build -type d \
  -path '*/out/cuda/build' -print -quit)"
cmake --build "$CUDA_BUILD_DIR" \
  --target test_pack_dense_mask test_brle \
  test_masked_attention_geometry masked_attention_parity \
  test_graph_variant -j4
ctest --test-dir "$CUDA_BUILD_DIR" --output-on-failure \
  -R '^(pack_dense_mask|brle|masked_attention_geometry|masked_attention_parity|graph_variant)$'
```

## Contention validation

The prior full-suite blocker is resolved. These now pass:

```bash
RUST_TEST_THREADS=1 cargo test --workspace \
  --exclude pie-server-py --exclude pie-worker -j4 --quiet

cargo test -p pie-worker --lib -j4 --quiet
```

The active fleet and zero-host-swap kill tests each passed three consecutive
process-isolated runs. KV store unit tests cover demand computation, grant
installation, surplus return, and rollback.

### CUDA contention performance profile

The ignored real-GPU harness now supports three process-isolated modes with a
shared JSON record:

- `legacy`: roomy pool, contention mode disabled;
- `baseline`: roomy pool, preempt mode enabled;
- `contended`: preempt mode with an explicit page cap.

On the local RTX 4090 with Qwen3-0.6B, fleet 8, 48 output tokens per lane:

- Four legacy/baseline pairs gave median preempt hot-path throughput `0.982x`
  and p95 latency `1.018x` versus legacy. One baseline run was a noisy outlier;
  the other three throughput ratios were `0.979x`, `0.995x`, and `0.986x`.
- Four 8-page constrained runs gave median throughput `0.429x` and p95 latency
  `2.334x` versus roomy preempt mode. Individual throughput ratios were
  `0.415x` to `0.511x`.
- A 12-page constrained run gave throughput `0.525x` and p95 latency `1.906x`.
- The constrained runs completed without `WorkingSetError` and exercised
  balanced suspend/restore. Copy time was only part of the penalty; quorum
  narrowing and time outside the runnable set dominate.

Records and per-run logs are in the session `files/` directory, not the repo.

## Resolved scheduler identity risk

The earlier process-scoped quorum identity issue is fixed:

- `PendingRequest` carries both owner process ID and pipeline-scope quorum ID.
- close/allocation-wait removes only that scope;
- suspend/terminate removes every scope owned by the process;
- sibling close, sibling allocation-wait, process-wide suspend, independent
  waiter slots, bind-placeholder lifetime, and worker-level scoped leave have
  focused tests.

## Next steps

No implementation or validation work remains in this refactor. Explicit
follow-up workstreams, not acceptance blockers:

1. Integrate RS `OutOfSlots` contention with the KV grant lifecycle.
2. Support arbitrary running-process suspension after a bounded grace period;
   the current policy intentionally selects pin-free allocation-boundary
   victims.
3. Tighten conservative containment-bound demand if profiling shows material
   over-reservation.
4. Optimize severe-pressure throughput if the measured 8/12-page regimes are
   representative of production rather than synthetic stress.

## Acceptance checklist snapshot

| Requirement | Status |
| --- | --- |
| Guest model configuration absent | Done |
| `Channel::set` front replacement; `put` unchanged | Done |
| No author/runtime dense geometry inference | Done |
| Empty `forward-pass.new()` and builder methods | Done |
| Flat channel-only `attention(...)` | Done |
| `Option<&Channel>` mask; no `AttentionMask` enum | Done |
| No descriptor `Tensor` inputs in author code | Done |
| Host-known masks BRLE/co-batch | Done |
| Device-derived masks remain device-resolved | Done |
| Masked CUDA graph capture/replay | Done |
| `Pipeline::finish` removed | Done |
| Submitted work drains and remains take-able after close | Targeted tests pass |
| Independent concurrent pipeline close/allocation-wait identity | Done |
| All inferlets compile on final API | Done |
| Full workspace suite | Done |
| Final changes committed/pushed | Done |
