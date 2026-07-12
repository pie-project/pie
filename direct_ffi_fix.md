# Direct FFI Refactor Review and Remediation Plan

## Status

Review of the implementation described by [direct_ffi.md](direct_ffi.md), performed against the current codebase on 2026-07-09 and reconciled after implementation on 2026-07-10.

The core direct-FFI contract is implemented in the ABI, scheduler, runtime, and
Dummy reference backend. Accepted operations use stable terminal cells,
rejection is atomic, same-instance work can run ahead, global channel endpoints
own their wait state, and channel reader/writer notifications are active.

Unsupported backend features now reject synchronously instead of reporting false
success. Real Metal model execution, Metal typed memory operations, CUDA
tensor-parallel typed operations, and stable CUDA VMM/Metal sparse-pool resizing
remain explicit capability gaps. CUDA source was reviewed but cannot be compiled
or exercised on this macOS host.

## 0. Settled Design Decisions

These decisions are the contract for the remaining fixes. Do not introduce a
second response protocol or a compatibility path around them. Where these
decisions conflict with the older speculative-readiness or bind-owned-channel
text in [boundary.md](boundary.md), this section supersedes that text and the
boundary document must be updated in the same change that implements the new
contract.

### D1. Every accepted fire terminates exactly once as success or failure

The scheduler maintains a reservation ledger for every D3 channel endpoint.
Before dispatch it atomically reserves the input occupancy and output capacity
required by a fire, including state projected through older accepted operations
in the same driver FIFO. A fire that cannot obtain all reservations remains
queued; the FFI is not called and no operation epoch is consumed.

Device readiness remains an invariant check against the state reserved by the
scheduler, not a normal control-flow outcome. A readiness miss after native
acceptance is terminal `Failed`: the driver discards speculative writes,
publishes poison for the affected channels, publishes the failure status, and
wakes waiters. There is no `Skipped`, readiness-dummy, retry callback, or
nonterminal callback outcome. This intentionally replaces `boundary.md`'s old
C2 dummy-run/retry rule and its unchanged-input fire-skip heuristic.

```text
not accepted fire -> no accepted epoch, no terminal publication
accepted fire     -> exactly one terminal Success or Failed publication

rejected FFI operation -> synchronous error, no accepted member, no callback
accepted FFI operation -> exactly one payload-free callback after every member
                          fire has published a terminal outcome
```

Callback arrival means only "recheck persistent terminal state." It does not
mean success. Every accepted fire has epoch-tagged persistent control
state:

```rust
enum TerminalOutcome {
		Success,
		Failed,
}
```

For instance launches, the runtime leases one stable terminal cell per launch
member and passes its pointer in `PieLaunchDesc::terminal_cells`. The scheduler
associates that cell with its candidate instance-operation epoch. Bind returns
identity only; terminal storage is not driver-owned instance state. Cells are
recycled from a pool only after native retirement, so accepted operations cannot
observe reuse. `Pending` is not a callback outcome.

For copy, state-copy, and resize operations that are not tied to one instance,
the runtime completion broker lends the driver a fixed terminal-status cell as
part of `PieCompletion`. The cell remains stable until the operation callbacks
and retires. This is fixed control metadata, not a result arena: it carries no
model value or error payload.

Successful fire publication order is:

```text
channel cells
-> channel head/tail
-> terminal cell = Success
-> release fence
-> runtime.notify(wait_id, target epoch)
```

Failed fire publication order is:

```text
affected channel poison
-> terminal cell = Failed
-> release fence
-> channel/operation notifications
```

The runtime commits that fire's KV, recurrent-state, WorkingSet, and page-table
changes only after an acquire load observes `Success` in that accepted
submission's leased cell. It aborts them on `Failed` or close. Error details
remain observable through channel poison; the terminal outcome is internal
commit/abort control state.

### D2. The scheduler owns epochs and launch batches are atomically accepted

The per-driver scheduler is the only authority that sequences and assigns
operation epochs. `BoundInstance::reserve_completion` must not consume an epoch
before dispatch. The scheduler proposes one candidate epoch and leases one
stable terminal cell for each launch member. `PieLaunchDesc` carries the
parallel `terminal_cells` pointer array so the native driver publishes to the
exact state selected by the scheduler. A native batch contains at most one fire
per instance; same-instance run-ahead uses successive accepted batches.

At dispatch, the native driver validates and prepares the entire typed FFI
operation without mutating durable instance state. Rejection is atomic: the
driver rejects every member synchronously, the scheduler releases every pending
slot and channel reservation, commits no candidate epoch, and expects no
callback. Acceptance is also atomic: every member becomes accepted, each
candidate epoch is committed, and each internal `SubmissionTicket` is bound to
its exact persistent terminal cell.

`PieCompletion` remains one payload-free wake target for the FFI operation, not
an instance-operation epoch. An accepted launch batch invokes it exactly once,
after all member terminal entries and channel publications are visible. On
wakeup the scheduler rechecks every member entry:

- A successful member commits only its own runtime transactions.
- A failed member aborts its transactions and fails its D4 pipeline domain.
- Successful members from independent pipelines in the same native batch remain
  successful; the batch itself is not a failure domain.
- A fatal backend or device error publishes `Failed` for every unresolved member
  before the one batch callback. The driver then reports itself unusable if
  independent work cannot safely continue.

Queue acceptance and native acceptance are distinct:

- `pipeline.submit` returns after the scheduler owns the command and its pending
	runtime transaction.
- Validation that can be performed before enqueue is returned synchronously.
- An unexpected native rejection after queue acceptance releases every member's
	channel reservation, fails each affected pipeline, and aborts its pending
	transaction; it does not create an epoch gap.
- An accepted fire always reaches `Success` or `Failed` asynchronously.
- The queue-accepted operation owns all runtime transaction and channel
	reservation guards until native rejection or terminal retirement.

This preserves nonblocking pipeline submission, exact FIFO ordering, and
gap-free per-instance native epochs.

### D3. Persistent channel endpoints are owned by global channel identity

A channel has one authoritative endpoint per driver, keyed by runtime-minted
`ChannelId`, rather than one independent host ring per bound instance. A channel
becomes driver-affine on first registration; using it on another driver requires
an explicit transfer and a distinct endpoint rather than aliasing device
storage.

```text
ChannelId -> device ring
					-> pinned/shared mirror
					-> head/tail/poison words
					-> reader/writer wait ids

Bound instance -> references channel endpoints
							 -> owns only instance launch/status state
```

The direct ABI adds typed `register_channel` and `close_channel` operations.
Registration occurs lazily on first bind, before `bind_instance`, and returns
stable mirror/control addresses. The runtime owns the reader and writer wait
slots and passes their ids at registration; the native driver owns the device
ring and pinned/shared allocations. Binding a pass validates its attachment and
references an already registered endpoint; it does not allocate another
authoritative ring or wait-id pair.

The runtime `ChannelCell`, every bound pass, and every in-flight operation hold
an endpoint lease. Closing the last pass first detaches its instance. Final
endpoint close is an ordered scheduler command after all referencing operations
retire; it marks the endpoint closed, wakes reader and writer waiters, and frees
native storage. Only after native close completes may the runtime free the
generation-tagged wait slots. Native close rejects an endpoint that still has
instance attachments. Duplicate registration or close must not replace or free
live storage.

The first attachment freezes the channel contract:

- physical shape, resolved dtype, and capacity;
- declared `ChanDType` for attachment-to-attachment comparison, including
  `Act` versus concrete `F32`;
- host role and seeded flag; and
- endpoint kind, canonical extern binding name, and SPSC producer/consumer
  claims.

An `Act` channel may materialize to the driver's activation dtype when matching
the first host-created endpoint, but two pass declarations sharing an endpoint
must use the same declared `ChanDType`. A host-visible, seeded, or non-extern
private channel may attach to only one instance. Multi-instance sharing is
legal only for an extern channel whose declarations are `host_role = None` and
`seeded = false`, with exactly one `Export` producer and one `Import` consumer
that resolve the same `ExternDecl.name` binding key and have identical shape,
declared dtype, and capacity. A seed value is applied exactly once. Conflicts
are rejected before native mutation, and each backend independently enforces
the same rule.

`channel.take` and `channel.read` register on the channel reader wait id, recheck
producer tail/poison/closed state with acquire ordering, and then park or return.
A consuming take publishes consumer-head movement and notifies the writer wait
id. Host puts, including pre-bind staging, obey channel capacity and use the
symmetric register-then-recheck protocol for backpressure. The scheduler uses
the same endpoint state and pending-reservation ledger for D1 dispatch.

Per-submission terminal cells remain separate because they settle each launch's
runtime transactions; channel rings remain the sole value and
inferlet-visible error path.

### D4. Every ordered command belongs to a pipeline failure domain

Launch, KV copy, state copy, page-table update, and pool resize commands carry a
runtime `PipelineId` or an equivalent internal failure-domain id when they are
enqueued. The id is scheduler metadata and need not enter the public inferlet
ABI.

If an accepted value-less operation finishes as `Failed`, the scheduler:

1. Marks that pipeline failed.
2. Aborts the operation's pending runtime/memory transaction.
3. Stops dispatching dependent commands from the failed pipeline.
4. Poisons the pipeline's affected reader channels before waking waiters.
5. Continues servicing independent pipelines.

This gives copy and resize failures an observable channel error without adding
an operation result API. Error details remain channel poison; the terminal cell
contains only `Success` or `Failed`.

For a launch batch, D4 is evaluated per member, not per callback. A failed
member poisons its affected channels and dependent pipeline domains; a
successful member from an independent pipeline commits normally even though
both members share one batch callback.

### D5. Backing pools use stable typed identities and stable virtual slots

The runtime registers typed driver pools during bootstrap/bind and receives or
assigns stable `PoolId`s for KV, recurrent state, recurrent buffers, and scratch.
Each pool declares a fixed virtual capacity. `PhysicalKvPageId` is an offset in
that reserved virtual range and is never renumbered while live.

`PiePoolResizeDesc` changes backing residency, not identity:

- `map_ranges` materializes backing below stable virtual slots.
- `unmap_ranges` removes backing only after all in-flight users retire.
- `target_pages` is the resulting backed-page count and must agree with the
	ranges; it does not replace or move the pool.
- CUDA implements the ranges with VMM mapping/unmapping.
- Metal implements them with sparse-buffer mapping/unmapping.
- TP drivers apply the same typed operation to every rank before publishing
	success.

The `MemoryBroker` chooses map/unmap ranges and submits them through the existing
per-driver scheduler. No public inferlet residency API is added, and stable page
ids continue to satisfy the memory model in [kv_refact.md](kv_refact.md).

## 1. Finding Resolution

### 1.1 Metal false success: fixed; execution remains unsupported

Metal validates direct descriptors and endpoint ownership, but `launch`,
`copy_kv`, `copy_state`, and `resize_pool` now return
`PIE_STATUS_UNSUPPORTED`. They do not publish success, mutate terminal cells, or
invoke completion callbacks. This closes the correctness hole. Connecting a real
Metal executor remains backend feature work.

### 1.2 Callback interpreted as success: fixed

Every accepted launch member now carries a stable terminal-cell pointer.
Completion waits recheck that exact cell with acquire ordering. KV, recurrent
state, WorkingSet, and channel reservations commit only on `Success`; `Failed`
aborts the member and its pipeline domain. Drivers publish terminal state before
the operation callback.

### 1.3 Rejection consumed pacing epochs: fixed

`BoundInstance::reserve_completion` leases storage without consuming a native
epoch. The scheduler proposes epochs and commits them only after whole-batch FFI
acceptance. Rejection releases the request and reservation guards without an
epoch gap or callback.

### 1.4 CUDA TP typed operations: false success fixed; feature unsupported

CUDA `copy_kv`, `copy_state`, and `resize_pool` reject when `tp_size > 1`.
Rank-0-only completion is no longer reported as success. An all-rank TP control
protocol and TP=2 hardware tests remain future capability work.

### 1.5 Dummy partial acceptance and second operation queue: fixed

Dummy validates the whole batch and prepares callback delivery before mutating
instance state. It then publishes every member terminal, channel notification,
and exactly one operation callback in order. The private request/operation queue
is gone. Zero-delay callbacks do not create OS threads; configured delayed
callbacks retain only a one-shot completion worker for test latency.

### 1.6 Same-instance run-ahead: fixed

Queue acceptance is separate from native acceptance. The scheduler may own
successive operations for one instance, commits their epochs only after
acceptance, and retires them in FIFO order. Channel capacity and reservation
state bound run-ahead.

### 1.7 Channel waiting and backpressure: fixed

Typed `register_channel`/`close_channel` operations create driver-affine global
endpoints. Reader and writer waits use generation-tagged ids and a
register-then-recheck future. Host puts wait for writer capacity; reader
publication, host consumption, poison, and close update endpoint words before
waking the matching waiter. Cancelled waits deregister their wakers.

### 1.8 Duplicate bind: fixed

The scheduler and all native backends reject duplicate instance ids without
replacing the live binding.

### 1.9 Capability-parse leak: fixed

CUDA and Metal wrappers destroy the temporary native owner on every
capability-parse failure.

### 1.10 CUDA ordering and resize limitations

CUDA PTIR host inputs are now copied on the execution stream, same-instance
projected ring state is updated at enqueue time, callback contexts do not look
up mutable instance state, and per-fire commit snapshots prevent duplicate or
run-ahead fires from racing one shared commit flag. Host-put decoding is linear
in batch size.

CUDA VMM range resize is not implemented and rejects unsupported pool/range
forms. The existing Nemotron exact-MoE host-directed planning synchronization is
outside this direct-FFI contract repair and remains a model-path performance
limitation.

## 2. What Executed Well

The following parts of the plan are implemented convincingly:

- `DriverChannel`, request/response payloads, deferred responses, and service loops are gone.
- CUDA and Metal expose generated direct create/destroy/register/bind/launch/copy/resize symbols.
- The per-driver scheduler owns `DriverBackend`; arbitrary runtime threads do not invoke it directly.
- Ordered control operations and launches share the scheduler queue.
- The runtime callback table is versioned and supplied at driver creation.
- Rust catches unwinding inside the foreign callback.
- Completion ids are generation-tagged WakerTable slots.
- CUDA's normal committed PTIR path performs D2H mirror copies, mapped-word publication, and callback notification in stream order.
- The PTIR value response marshal and per-request response fanout are gone.
- The ABI crate has the intended `local`, `capabilities`, and Rust-only `transfer` surfaces.
- Legacy rkyv/schema-hash/derive dependencies are gone.
- `pie-ipc`, IPC profiles, spin budgets, and native service files are gone.
- Scheduler shutdown drains accepted operations, closes instances, and drops the driver once.

## 3. Validation Performed

The following checks passed locally on Apple Silicon macOS after the final
runtime changes:

| Validation | Result |
|---|---:|
| `pie-engine --lib` | 202 passed |
| `pie-worker --lib` default/Dummy/Metal variants | 38 passed each |
| `pie-driver-abi` Rust tests | 19 passed |
| C/C++ ABI header-layout tests | 2 passed |
| Dummy driver tests | 12 passed |
| `pie-waker` tests | 14 passed |
| `pie-client --lib` | 2 passed |
| Metal direct stub CTest | 1 passed |
| Generated ABI header regeneration | byte-stable |
| Changed Rust files | `rustfmt --check` passed |
| Workspace library sweep excluding `pie-server-py` | passed |
| Standalone Dummy ping/direct-channel/text-completion E2E | passed |
| `git diff --check` | passed |

The text-completion E2E built and ran the production
`text-completion@0.2.15` inferlet and returned 40 synthetic characters through
the Dummy driver.

The broad workspace integration sweep reached the engine's existing
`concurrent_race` fixture and failed because its `generate` test inferlet imports
removed `inferlet` APIs. The all-target check independently reaches stale
`pagestore_bench` and `inferlet_bench` imports/call signatures. These are
unrelated pre-existing test-target failures; workspace library tests pass.

CUDA GPU validation was not runnable because this host has no CUDA toolkit or
NVCC. The latest CUDA changes therefore require compilation and accepted-launch
device testing on a CUDA host before CUDA can be signed off.

## 4. Coverage and External Gates

Host-runnable coverage now includes atomic batch rejection, terminal
publication ordering, mixed member outcomes, failure isolation, same-instance
run-ahead, callback-before-late-poll persistence, channel registration and
attachment conflicts, reader/writer notifications, poison wakeup,
register-then-recheck race coverage, cancelled-wait deregistration, Dummy native
batching, duplicate bind, close/shutdown, and standalone inferlet E2E.

The remaining gates require capabilities not implemented or unavailable here:

1. Compile and test the CUDA changes with NVCC.
2. Run accepted CUDA launches on a device, including asynchronous failure and
   same-instance run-ahead.
3. Implement and test all-rank TP typed operations before advertising that
   capability.
4. Implement real Metal execution and typed memory operations before advertising
   those capabilities.
5. Implement CUDA VMM and Metal sparse range resizing before advertising stable
   typed-pool resize support.

## 5. Implementation Status by Phase

### Phase 1: Atomic acceptance and completion correctness -- complete

Stable launch/control terminal cells, scheduler-owned accepted epochs, exact
member settlement, atomic Dummy acceptance, and failure-domain behavior are
implemented and covered by host tests.

### Phase 2: Channel ownership and scheduler reservations -- complete

Global endpoint registration/close, attachment validation, endpoint leases,
reader/writer waits, host backpressure, reservations, queue acceptance, and
same-instance run-ahead are implemented.

### Phase 3: Single-rank backend behavior -- partial

Dummy is complete and serves as the executable reference backend. CUDA PTIR was
updated for stream-ordered host input, projected run-ahead, per-fire commit
snapshots, endpoint notifications, and failure settlement, but needs CUDA-host
validation. Metal rejects execution and typed operations as unsupported rather
than pretending to complete them.

### Phase 4: Multi-rank and stable pool behavior -- unsupported

TP typed operations and nontrivial stable-pool resize forms reject
synchronously. They must not be advertised until implemented and tested on
appropriate hardware.

### Phase 5: Cleanup and final gates -- host portion complete

Obsolete endpoint mirrors and Dummy operation queuing are gone. Host ABI,
runtime, Dummy, worker, Metal surface, and standalone E2E gates pass. CUDA gates
remain external.

## 6. Revised Definition of Done

The refactor is complete only when all of the following are true:

- CUDA and Metal both execute real model work through direct non-blocking calls.
- Every accepted fire publishes exactly one `Success` or `Failed` in its leased
  terminal state; every accepted typed FFI operation invokes exactly one
  payload-free callback after all member states are visible.
- Rejected FFI operations accept no members, publish no terminal state, invoke
  no callback, and consume no accepted operation epoch.
- A callback means only "recheck persistent state"; it is never interpreted as success by itself.
- Mixed outcomes in one batch commit successful independent members and abort
  only failed members and their dependent pipeline domains.
- KV/RS and WorkingSet transactions commit only after the matching `Success`
  state is observed and abort on `Failed` or close.
- Asynchronous execution errors publish affected channel poison and `Failed` before waking waiters.
- Scheduler channel reservations prevent known-not-ready work from entering the
  FFI; a post-acceptance readiness miss is terminal failure, never a dummy-run,
  retry callback, or `Skipped`.
- Same-instance submissions can run ahead in FIFO order within channel capacity.
- A driver-affine global `ChannelId` endpoint owns the sole device ring,
  mirror, and head/tail/poison/closed words; its runtime `ChannelCell` owns the
  stable reader/writer wait ids.
- Typed channel registration and ordered close preserve storage and wait-slot
  lifetime until all instance attachments and in-flight references retire.
- Only one matching, unseeded, host-invisible extern Export/Import pair with
  the same canonical extern binding name may share an endpoint across
  instances; all other channels remain single-instance and declaration
  conflicts reject without mutation.
- Channel reader and writer wait slots are the actual register-then-recheck wait path.
- Value-less operation failures fail and poison only their pipeline failure domain.
- CUDA TP typed operations execute on every rank.
- CUDA VMM and Metal sparse resize descriptors operate on typed pools with stable ids and never renumber live physical page ids.
- Dummy has no private request queue and cannot partially accept a batch.
- Duplicate bind, close, parse failure, and shutdown are leak- and deadlock-free.
- CUDA and Metal accepted-launch smoke tests prove publication-before-callback and launch-before-completion.
- Every original grep and dependency gate remains clean.
- All five remediation phases and their tests pass.

## 7. Verdict

The reviewed cross-boundary correctness defects are fixed in the ABI, runtime,
scheduler, and Dummy reference backend. Unsupported Metal, CUDA TP, and stable
pool-resize paths fail closed instead of returning false success.

The core direct-FFI redesign is host-validated. Full backend completion is not
yet proven: CUDA must compile and pass device tests, and Metal/TP/stable-pool
features must be implemented before the full multi-backend definition of done in
section 6 can be claimed.
