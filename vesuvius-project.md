# Project Vesuvius: Frame-Based Submission

Date: 2026-07-22 (revised same day: sealing corrected to wait-all at
frame granularity, Section 7; Phase 2 gate re-baselined, Section 12)
Status: Phases 0–1 landed (`vesuvius-phase1.md`); Phase 2 pending
Baseline: branch `tasks/ptir-fusion/agents/charlie`

This document is self-contained: it is the complete specification an
implementer needs, with no external design documents required. File
references point at code on the baseline branch.

## 1. Motivation

Pie's programming model assumes the host is much faster than a GPU forward
pass: every wave, the host (scheduler plus guest inferlets) composes and
submits the next batch. The current scheduler is a strict wait-all barrier
(`runtime/engine/src/scheduler/quorum.rs`, `WaitAllPolicy`; folded into
`scheduler/frame.rs`'s `FramePolicy` by operator directive 2026-07-22 —
every deployment including the default `PIE_FRAME_SIZE=1` now runs the
frame policy, where a 1-slot frame is exactly this per-wave rule): a wave
fires
only when every active pipeline's next pass is submitted and ready, so when
any CPU-side actor slows down — thousands of short inferlets, WASM
instantiate pressure, a convoyed control lane — the whole fleet stalls on
the slowest lane and the GPU synchronously waits on it.

Measured on the motivating workload (Qwen3-0.6B, CUDA TP1, RTX 4090; 2,048
requests of ~37 prompt tokens and exactly 32 output tokens, 512 active):

- Pie 19.08k output tok/s versus vLLM 30.81k (−38%), despite Pie executing
  ~14% *less* GPU kernel work — the gap is feeding and retiring work, not
  kernel speed.
- ~3 ms per wave of submit-side host work (571 ms across a 174-wave run:
  settlement enqueue 321 ms, epilogue assembly 164 ms, settle preparation
  156 ms as the dominant lines).
- ~4.3 ms of completion-detection latency per wave in the native completion
  path.
- Depth-2 run-ahead inflates waves ~37%: a successor reaches composition
  before its predecessor's channel publication, the composer observes
  incomplete readiness, and the launch retries or fragments.
- 27,657 `cudaLaunchKernel` calls versus vLLM's 3,318 for the same work.

These numbers were profiled a few commits before this document; Phase 0 of
the landing plan re-measures them at the implementation baseline.

Vesuvius removes the assumption instead of optimizing around it. The host
stops interacting with the device once per wave and starts interacting once
per **frame**: a fixed-length block of waves that executes uninterrupted on
the device.

## 2. Core Concept: the Frame

**Terminology.** A *wave* remains what it is today: one batched forward step
across lanes — the (1, n) unit. A *frame* is the new (k, n) unit: k
consecutive waves submitted as one unit, executed device-side without host
interaction, settled once at the end. Each lane contributes exactly k ordered
slots; an empty slot is a no-op for that lane in that wave. Every frame executes
within exactly one sealed membership epoch (defined in Section 7); *epoch* keeps its
existing meanings and is not reused for the work unit.

Rejected names: *cycle* (clock/scheduling ambiguity), *burst* (no fixed-length
connotation), *quantum* (considered; frame preferred), *epoch* (already has
three meanings in the codebase: channel publication epochs, pool reclamation
epochs, membership epochs).

**Frame size k is a static deployment constant**, exactly like the KV page
size:

- derived deterministically from the model profile or set by the operator;
- fixed at instantiation; never renegotiated per frame;
- never adapted from runtime timing measurements (heuristics ban);
- queried by the guest via `model.frame-size()` and adapted to, the same way
  guests adapt to `kv-page-size()`.

Keep k MINIMAL (amended by operator directive, 2026-07-22): with frames
overlapping on-stream (Section 7) there is no boundary bubble to amortize,
so k buys only host-cadence slack — enlarge it only when the host cannot
keep up with the per-frame submit deadline. Typical deployments are k = 2
or 3; slow models (large, TP) deploy k = 1. Guest programs must be
*output-correct* for any k: k determines submission granularity and
resource sizing, never token-level behavior.

**Why this fixes the laggard problem.** The guest's response deadline
stretches from "before the next wave" to "before the current frame drains".
After taking the first token of frame f, the guest has roughly
(k − 1) x step_time to submit frame f + 1 (~75 ms at k = 16 and ~5 ms steps,
versus the sub-step deadline of the per-wave model). The design is symmetric at both
extremes: small fast models get large k; large slow models get k = 1 but long
step times. The product k x step_time — the actual deadline — stays large
everywhere.

## 3. Execution Model

### 3.1 Device side: the composed wave program becomes multi-step

A frame is not "k blind wave submissions". It is one fused program: the
composed wave program loops k times over

    batched forward -> per-lane sample -> per-lane advance -> KV append

with inter-step dataflow resolved on device. The only per-step unknown in
decode is the previous step's sampled token, and channels already express
that dependency symbolically: the same channel handle is bound into
successive fires, and a fire's epilogue publishes what the next fire
consumes. The frame makes the driver *trust* this dataflow (stream ordering)
instead of the host observing readiness per successor.

### 3.2 Per-wave publication, per-frame settlement

Today's settle bundles four roles: making channel data host-visible and
waking waiters; retiring resources; surfacing failures (poison); and
scheduler bookkeeping. Frames split it in two:

- **Publication (per wave, data plane).** Each wave writes its outputs to
  the channels' pinned host mirrors, sets their commit words, and bumps one
  pinned doorbell counter. The host waker wakes on the doorbell and checks
  only channels with registered waiters — O(pending takes), no scheduler
  involvement, no per-wave completion event. Interior tokens are therefore
  host-visible one step after they are produced; this publication latency
  is the hidden constant of the interleaved schedule (Section 7).
- **Settlement (per frame, control plane).** Resource retirement, poison,
  and dead-lane release happen once per frame, as a batch over all k waves.

Because execution inside a frame is uninterrupted, a wave can never block
on a full channel: overflow is prevented statically at submit (Section 5),
never dynamically by back-pressure.

### 3.3 Graph capture is per wave, not per frame

Pie's geometry is already data, not topology: kv-len, page tables, indptr,
and positions enter kernels as channel/device data in persistent buffers, so
per-wave variation in sequence lengths and pages does not change captured
graph topology (the decode forward is already captured and replayed from a
cache keyed only on `{num_requests, num_tokens, variant}` —
`driver/cuda/src/batch/forward_graph.hpp`).

**The capture unit is the wave.** Each wave of a frame independently selects
its execution class by its own composition:

- **Decode wave** (every lane contributes one token): the existing bucketed
  decode graph class — buffer refresh only, no re-capture.
- **Chunk-carrying wave** (any lane contributes a prefill chunk): the wave
  runs the varlen kernels for *all* its lanes — decode lanes ride as
  length-1 rows, a negligible cost. These waves are captured too, as a
  varlen class on the same bucketed cache: geometry is data for the varlen
  path exactly as for decode, so nothing structural distinguishes the
  classes. The varlen class differs only in a coarser token lattice
  (padding to bucket boundaries, bounded by the chunk budget C) and
  correspondingly more capture memory.

Every wave launch is therefore a captured-graph replay. What Phase 2 adds
to capture *scope* is the per-step epilogue — sample, advance, publication
— which today stays outside the captured forward body; bringing it inside
(or enqueuing it alongside each replay) requires the deterministic device
RNG of Section 14.

Wave-level classing matters because frame-level mixing is the *common* case
on the motivating workload, not a rarity: with 32-token outputs and k = 16 a
lane lives ~2 frames, so at any boundary roughly half the fleet is newly
admitted and carries a chunk — essentially every frame contains chunks, and
a frame-level "pure decode" class would never trigger there. Under the slot
model (Section 4) a lane's chunks are its earliest passes and therefore
occupy its earliest slots, so chunk-carrying waves cluster at the front of
the frame and the tail waves are pure decode waves on the captured class.

### 3.4 Layering: no new driver primitive

The frame is a runtime/scheduling concept only. The driver's interface
semantics stay "one submitted program, one settle, one completion" — the
program is simply larger. Frame-boundary-only settlement therefore falls out
of existing per-fire semantics rather than being a new mechanism.

What the driver genuinely needs to add:

- multi-step execution of the composed program (loop or unrolled), with the
  sample-to-next-input wiring resolved on device;
- per-step device advance of geometry channels (kv-len increment, position
  increment, token append) and per-step consumption of staged host-writer
  cells;
- per-wave publication: mirror writes, commit words, one doorbell bump;
- per-step reads of asynchronous per-lane control words (abort, sampling
  parameters — Section 9);
- KV writes confined to the pages bound at prepare (Section 8);
- per-wave graph-class selection (the decode class exists today; the
  varlen class extends the same bucketed cache — Section 3.3);
- settlement and completion reporting once per frame.

What the runtime owns:

- collecting per-lane frame submissions;
- sealing membership epochs at frame boundaries (Section 7);
- composing the fused program and its per-wave batches;
- token-weighted admission at frame boundaries (Section 8).

The two sides share only the constant k (the `runahead.hpp` pattern), not a
new protocol noun.

## 4. Interface Changes (WIT)

The delta is small and mostly additive. Final naming:

```wit
// model.wit — static deployment constants, next to kv-page-size
frame-size: func() -> u32;        // k: waves per frame, fixed at instantiation
max-embed-length: func() -> u32;  // C: max embed tokens in a single pass
                                  //    (the prefill chunk budget)

// forward.wit — frame submission replaces per-pass submit
submit: func(
    on: borrow<pipeline>,
    slots: list<option<borrow<forward-pass>>>,
) -> result<_, error>;
```

Notes:

- `submit` moves from a `forward-pass` method to an interface-level function
  taking exactly k ordered slots. A single-pass frame is
  `submit(on, [some(pass), none, ...])`; the SDK migrates
  `pass.submit(on)` by filling the other slots with no-ops.
- The slot list may repeat the same handle: a plain decode frame is the same
  pass in all k slots, and steady-state refires carry only identity hash +
  instance, so repetition is cheap.
- Slot i executes in wave i; the composer never moves it to another wave.
  Slots may be heterogeneous:
  `[some(chunk), some(chunk), some(chunk-with-readout), some(decode), ...]`,
  with `none` wherever the lane is idle.
- There is no frame resource. A frame is one call, not a stateful object;
  a builder would add only lifetime and drop semantics.
- No device stop-spec, no contract-result type (Section 6).
- `put` and `set` keep their existing meanings and split the host-write
  roles cleanly: `put` enqueues staged cells consumed one per fire
  (per-fire inputs — prompt chunks, staged parameters), while `set`
  replaces the committed front cell in place (latest-value words —
  Section 9). `set` on a cell claimed by an in-flight fire is an error;
  that existing guard is what makes `set` boundary-only on consumed
  channels, while a channel the program binds read-only has no claimed
  front and accepts `set` mid-frame. No new channel operation is needed.

## 5. Submission Semantics and Validation

These rules carry the design's weight:

1. **Fusability is a deterministic program property.** At `submit`, every
   host-writer channel bound by the frame's non-no-op slots must satisfy one
   of:
   - *staged*: it holds at least as many staged cells as fires in this frame
     will consume (the chunked-prefill path — prompt tokens, positions, page
     tables are all known at submit time and staged via existing `put`
     semantics); or
   - *device-advanced*: the PTIR program contains an advance rule for it
     (the decode path — kv-len/position increment, sampled-token append); or
   - *latest-value*: the program only reads the channel, never takes it — a
     control word (Section 9). It needs one committed cell at submit and no
     staged count: the device re-reads it each step, and host `set` may
     replace it at any time.

   Anything else is a submit-time error. All three classes are structural
   properties of the program and the staged occupancy — whether a frame can
   fuse is decided by validation, never by timing.

2. **`slots.len() == frame-size()`; any other length is an error.** The guest
   queried k; adapting to it is the contract (the page-size pattern). A lane
   that has no work in wave i supplies `none` in slot i; there is no implicit
   padding or composer-side relocation.

3. **Frames do not change channel semantics.** `put`, `set`, `take`, and
   `read` retain their existing queue, visibility, claim, and error rules;
   no channel operation is delayed to a frame boundary. Submitted slots use
   the existing channel ordering and sequence-ticket rules. In particular,
   a cell already claimed by an in-flight fire is immutable — so `set` on a
   consumed channel is boundary-only by the existing guard — while later
   `put`s enqueue normally and are consumed by later fires. Latest-value
   channels (Section 9) are the read-only case of the same rules, not an
   exception to them.

4. **Channel overflow is prevented at submit, never by back-pressure.** A
   wave inside a frame can never block on a full channel — uninterrupted
   execution is the contract. `submit` therefore enforces, per host-reader
   channel:

       occupancy + cells reserved by all accepted, unsettled frames
                 + cells this frame will write  <=  capacity

   and errors on violation. All terms are host-known at submit time,
   so the check is deterministic — never a function of guest drain timing.
   Under the reactive decode SDK's one-successor schedule (Section 7), the
   worst case is (k − 1) undrained cells plus k new ones, so token channels
   need capacity >= 2k − 1. Each accepted frame reserves and binds its whole
   working-set delta; consecutively queued frames reserve their deltas
   cumulatively. The logical reservation remains
   `reserve(ceil(k / page-size))` per full decode frame.

## 6. Stop, Failure, and Lifecycle

### 6.1 Stop: guest-observed, no device stop-spec

There is **no device-side stop-spec**, by explicit simplicity decision. The
guest observes termination the same way it does today: it takes a token,
sees a stop token, and closes the pipeline. Existing `pipeline.close`
semantics already define everything needed — fires already submitted run to
settlement in order; later submissions fail; outputs stay take-able. A lane
can be logically dead while its frame (and possibly the next, already
submitted frame) is still computing. Their geometry is already baked, and
`close` does not remove their slots or recompose either frame.

The waste depends on why generation ends:

- For a known output bound, the SDK fills slots beyond the bound with no-ops
  and does not submit a successor beyond it. Waste is ~zero when the bound
  aligns with k (for example, fixed 32-token output with k dividing 32).
- For an unpredictable content stop, the reactive decode loop may already
  have accepted one successor. If the stop position is uniform within a
  frame, the mean excess is `3(k - 1) / 2` lane-steps: the remainder of the
  current frame plus, except when the first token stops, one full successor.
  That is ~9% at L = 256, k = 16. Without a queued successor, the residual is
  only `(k - 1) / 2`, or ~3% for the same shape.

Decode waves are memory-bound on weights, so dead rows are cheap at the
margin. Two requirements make this safe:

- each accepted frame's reservation must cover all k steps, so garbage
  appends can never overflow reserved pages;
- KV pages of a dead lane are released at most one frame late — bounded, and
  freed at frame settlement.

### 6.2 Failure: fail-stop execution, frame-scoped settlement

A frame settles once. On failure, waves already completed may have published
irreversibly, the remaining steps are absent, and there is no per-step retry
or recovery. This fail-stop model is only acceptable under two hard
requirements:

1. **Validation completeness at prepare.** Every per-lane recoverable error
   class — geometry, staged-cell counts, channel capacity, page binding —
   must be caught at submit/prepare, where it excludes only that lane from
   the frame. Nothing per-lane may be discoverable only mid-frame; fail-stop
   safety rests on this property. Eager page binding (Section 8) is what
   discharges the page class.
2. **A defined failure state.** The failure contract is observational —
   there is no rollback machinery. Publication cannot be undone: tokens
   already published — and possibly consumed by guests — stand. On
   mid-frame failure, the remaining steps are simply absent, publication
   stops (commit words are invalidated, which is what the existing abort
   path already does), and every member pass is poisoned with the frame's
   error, surfacing through the existing take/poison discipline. Nothing
   reverts and nothing needs to: the frame's KV writes were confined to
   pages bound at prepare, beyond each lane's frame-start frontier
   (Section 8), and a poisoned lane never fires again, so no consumer of
   the failed state exists. Its pages are released through the normal
   dead-lane path at settlement.

With per-lane errors excluded at prepare, the only remaining mid-frame
failures are device faults that are fleet-fatal today as well — the
effective blast radius does not grow.

## 7. Scheduling: Boundaries, Sealing, Pipelining

**The frame boundary is the only execution-state transition point** — future
submissions are sealed for launch, the prior frame settles, admission is
decided, and membership changes. Guest `submit` calls may arrive while another
frame executes or may queue work earlier; they do not mutate the executing
frame.

**Reactive decode pipelining without unbounded run-ahead.** The SDK uses this
queue-bounding pattern:

    submit frame 1
    take frame 1's first token                 (arrives one step in)
    if generation continues, submit frame 2    (deadline: (k-1) steps away)
    take frame 1's remaining tokens with take()
    take frame 2's first token
    if generation continues, submit frame 3
    ...

This is a decode queue policy, not a submission capability. It keeps at most
two decode frames accepted for a lane: the one executing plus the successor
submitted after its first non-stop token. Frames with no host-value dependency
between them — notably output-free pure-prefill frames whose inputs are already
known — may be submitted back-to-back; each still passes the cumulative
capacity and resource checks of Section 5. A reactive lane that runs late
delays the boundary rather than losing a frame (sealed membership below);
a finished lane must close promptly so the seal stops waiting on it. The
SDK encodes the decode pattern (Section 10) so authors do not hand-roll it.

Three consequences are load-bearing:

- **The runtime must compose incrementally.** Frame f + 1 is assembled as
  submissions arrive during frame f's execution, or earlier if already
  queued, and sealed the moment the wait-all gate holds — during f's
  execution, not at its settlement — so composition cost overlaps GPU
  execution. Waiting for the boundary to start composing would put today's
  compose cost on every boundary's critical path as a serial GPU gap.
- **Stream-order trust replaces readiness observation — and the launch-time
  barrier.** (Amended by operator directive, 2026-07-22; supersedes the
  original "frame f + 1 launches only after f completes.") Frame f + 1
  launches BEHIND f on-stream: the device readiness gate orders dependent
  fires by stream order and does not know about frame boundaries — the
  f-last-wave → f+1-wave-0 dependency is structurally identical to an
  intra-frame one, so frames overlap with no observation, no barrier, and
  no boundary bubble. f + 1 seals the moment the wait-all gate holds
  (normally during f's execution) and posts at the run-ahead depth;
  admission stays conservative (it never counts resources an unsettled f
  will free — the Section 5 cumulative-reservation pattern), and f's
  settlement gates only resource reclamation, never a launch. The
  successor-readiness observation problem (the measured retry waves,
  missing pipelines, and readiness inflation of Section 1) remains deleted
  by construction.
- **Wave-1 publication latency is the schedule's hidden constant.** The
  interleave slack starts when frame f's first token becomes host-visible
  (Section 3.2); the doorbell/waker path must be fast relative to one step.

**Sealed membership: wait-all at frame granularity.** The scheduler's
fundamental rule is unchanged from the per-wave quorum — infinite wait-all
with a 1 s report-only watchdog; the only thing that changes at k > 1 is
the awaited unit: a frame per lane instead of a wave. The seal holds until
every awaited lane's oldest queued frame is arrival-complete; pending
binds hold the seal; membership changes only through explicit events
(bind, graceful close, terminate), never through arrival timing — which
is what keeps admission's "no timing input" rule (Section 8) honest end
to end. The sealed frame is immutable, and the seal happens EARLY — the moment
the gate holds, normally while the prior frame executes (amended
2026-07-22; the boundary seals atomically: every ready lane's front
frame, partitioned only by the per-wave budgets, one frame per lane per
boundary). The laggard problem is solved by the k-times-looser
deadline of Section 2, not by refusing to wait: with a (k − 1)-step
budget, healthy guests never make the seal wait (measured: zero watchdog
reports across full 2,048-request fleets). A wedged — not merely slow —
lane still holds the fleet, exactly as at k = 1; the mitigation is
unchanged too: watchdog observability plus explicit close/terminate.

**Thundering herd.** Boundaries are fleet-aligned, so n guests wake on their
first tokens nearly simultaneously. Per-guest work is tiny (take + submit),
and the budget is (k − 1) steps rather than a sub-step window, but the submit
path must be lane-parallel from the start — a single control lane would
reproduce the measured bind convoy.

## 8. Chunked Prefill, Packing, and Admission

**Chunking is guest-side**, against the static constant
`max-embed-length()` (C). The guest splits a prompt of L tokens into
ceil(L / C) chunk passes. Engine-side transparent chunking is rejected: it
would be a hidden rewriting layer, and it is impossible anyway — guests own
the attention geometry (readable/writable pages, positions) per pass.

Mechanically, a chunk is an ordinary pass whose per-fire data (tokens,
positions, kv-len, page tables) is fully known at submit time and staged as
host-writer cells — it passes frame validation by the *staged* rule with no
new WIT. Interior chunks bind no readout; the final chunk samples the first
token. Frames mix freely:
`submit(on, [some(c1), some(c2), some(c3), some(d), ...])`, padded with
`none` to exactly k slots. Prompts longer than k chunks span multiple
pure-prefill frames. Because their inputs are known and interior frames have
no readout dependency, those frames may be submitted consecutively; TTFT
quantizes to frame granularity (accepted trade).

**Physical page binding: eager within the frame, lazy across frames.** The
working-set rule "physical pages are allocated only when a forward writes
them" cannot hold inside a frame — there is no host mid-frame to allocate.
Instead, prepare binds every page the frame's k waves will write, upfront,
at composition; across frames, binding stays lazy (only the frame's delta
is bound). Two consequences: the boundary admission knapsack counts the
*physical* frame-delta against the driver page ceiling, not just logical
reservation; and frame-scoped binding is what bounds the blast radius of a
frame failure (Section 6.2): garbage writes can only land in pages the
frame bound.

**Composition (the n axis).** Boundaries are fleet-aligned, so many lanes
may present chunks simultaneously. Two deterministic layers handle it:

- *Within a frame*: slot i is wave i; the composer never relocates a lane's
  pass. For each wave it batches the admitted lanes' non-no-op slots under a
  static per-wave token budget.
- *At the boundary*: admission chooses a deterministic subset whose demand
  satisfies every wave's token budget, as well as the physical KV limit.
  Overflow lanes are left out of this epoch's sealed membership and admitted
  at a later boundary. The rule is pure arithmetic over each lane's declared
  k-element demand vector (e.g. lane-id-ordered first-fit) — no timing input.

This *is* KV/token-weighted admission landing as a corollary of the frame
design: chunk declarations make token demand explicit, admission consumes
declarations instead of measurements, and the hand-tuned active-process
caps (today a workload-tuned count semaphore) go away.

## 9. Per-Step Control and Constrained Decoding

**Per-step host reactivity is an explicit non-goal at k > 1.** This is the
programming-model shift Vesuvius makes: the host programs each *frame*; the
steps inside are programmed declaratively — staged data, device advance
rules, device tables. A lane that insists on a host round trip per token
submits single-pass frames
(`submit(on, [some(pass), none, ..., none])`) and produces 1 token per frame
— k times slower *regardless of host speed*, because membership is sealed per
frame and interior waves cannot be joined. Workloads that are interactive by
construction belong on small-k deployments.

Host influence inside a frame still exists, split by a hard dichotomy:

- **Monotone/idempotent signals — asynchronous control words.** The host
  `set`s per-lane control words — host-writer channels the program binds
  read-only, which is exactly the *latest-value* class of Section 5 (the
  classification is program structure, so it enters container identity for
  free); the device re-reads them at each step start rather than
  snapshotting them at prepare. This explicit mode is orthogonal to ordinary
  queued-channel semantics. It is valid only for signals whose landing step
  may vary without affecting correctness: abort, temperature and
  sampling-parameter changes, and logit-bias updates. Abort suppresses
  later publication/commit after it is observed; it does not remove the
  lane from already baked geometry. Effect is best-effort per step,
  guaranteed within at most one frame — the landing step is
  timing-dependent, so identical-seed runs that change control words
  mid-stream are not bit-reproducible (accepted for this signal class).
- **Exactness-bound logic — device tables only.** Logic whose output must
  be exact at a specific step (grammar masks) may not ride the async path.

Constrained decoding under this dichotomy:

- **Tier 1 (future, additive)**: compile grammars to device tables
  (DFA transition + mask rows); the sampler's `mask-apply` reads device
  automaton state advanced per step inside the frame. Added only when
  justified; `grammar.wit` is untouched now.
- **Tier 2 (now)**: the existing host-side `matcher` loop
  (accept-tokens -> mask -> put) over single-pass frames, at the k-times
  cost stated above. No WIT change.
- **Possible future pattern**: optimistic in-frame decode + host-side
  validation + KV truncate/rollback at the violation point (shares machinery
  with speculative decoding; deterministic validation, so consistent with
  the heuristics ban). This is the recommended long-term direction for
  constrained decoding on large-k deployments.

## 10. SDK Surface

Most inferlet authors never see k or C:

- `ctx.prefill(tokens)` — chunks against `max-embed-length`, stages puts,
  fills prefill frames, and may queue consecutive output-free frames.
- `ctx.generate(n, stop)` — queries `frame-size`, sizes channels
  (capacity >= 2k − 1, Section 5) and reservations, runs the reactive
  frame loop (first-token `take` -> next-frame submit -> ordinary `take` per
  remaining token), stops on stop tokens, closes the pipeline.
- The per-step loop remains available for tier-2 constrained decoding and
  research inferlets.

WIT stays explicit; the SDK supplies ignorance. Advanced inferlets
(speculative trees, heterogeneous per-step programs) use the slot-list form
directly — the static multi-pass DAG expressiveness already exists via
symbolic channels.

## 11. Expected Impact on Measured Bottlenecks

| Measured problem (Section 1) | Effect of frames |
|---|---|
| Per-wave host cost: settle enqueue, epilogue, settle prep (~3 ms/wave) | Paid once per frame: divided by k |
| ~4.3 ms completion-detection latency per wave | Once per frame: divided by k |
| Readiness retry waves (~37% wave inflation at depth 2) | Eliminated by construction (no host readiness observation) |
| Wait-all barrier: one slow lane stalls the fleet | Rule kept at frame granularity; the (k − 1)-step budget makes seal waits not occur in practice (hang blast-radius unchanged from k = 1) |
| Hand-tuned active-process caps | Replaced by declarative token-weighted frame admission |
| Kernel-launch fan-out (27.7k launches) | Decode waves replay captured graphs; chunk waves cluster at frame front |
| Guest deadline < 1 step | (k − 1) x step_time |

At k = 16, the 2,048x32 target workload becomes ~2–3 frames per inferlet
(one chunk-carrying frame, then decode frames), with host interaction
amortized 16x and wave count approaching the useful-work floor. Nothing in
this table is claimed until Phase 0 re-measures the left column at the
implementation baseline.

## 12. Landing Plan

0. **Phase 0 — re-baseline.** The Section 1 profile predates recent
   scheduler changes. Re-run 2,048x32 (n = 5) plus the standard shapes at
   the implementation baseline; record waves, per-wave host cost lines,
   completion latency, and the vLLM reference. Gate for every later claim.
1. **Phase 1 — runtime-only (driver untouched).** The frame WIT surface,
   submission validation (Section 5), incremental composition, wait-all
   frame sealing (Section 7), and boundary admission (Section 8), executing
   over the existing per-wave driver path. Pacing is explicit: the runtime
   holds each sealed frame and feeds its waves to the driver at the
   existing in-flight depth — the driver's pinned staging pools are sized
   for that depth (`runahead.hpp` + 1), and flooding k waves through the
   per-wave path would recreate the readiness-retry pathology. Partial
   wins only (scheduler amortization, no host stall); per-wave driver
   costs remain. Gates: exact token parity on the serial and
   output-boundary oracles; c0/256 >= 99% of baseline; no regression on
   long-decode shapes; three consecutive 2,048-request fleets complete.
2. **Phase 2 — driver multi-step execution.** The composed wave program
   runs k steps enqueued ahead with device-resolved advance; per-wave
   publication with frame-boundary settlement and completion; per-wave
   graph-class selection (Section 3.3). Captures the dominant wins
   (Section 11 rows 1–2). Gates: settle/completion host lines divided by
   ~k against the Phase 0 profile; 2,048x32 >= 30k tok/s (re-baselined:
   Phase 0 measured k = 1 at ~24.7k and vLLM at 32.2k, so the original
   >= 25k gate was already met at baseline — the target is parity); Phase 1
   correctness gates unchanged.

## 13. Rejected Alternatives

- **Runtime auto-slicing of declarative contracts** ("guest declares n
  tokens, runtime cuts into frames"): a hidden rewriting layer between the
  guest's program and what executes, contrary to Pie's identity — the
  inferlet explicitly programs the engine. Ignorance of k belongs in the
  SDK, not the runtime.
- **Engine-side transparent prefill chunking** (vLLM-style): same objection,
  and guests own attention geometry.
- **Take-who's-there sealing / auto-Idling boundary stragglers** (this
  document's original Section 7): sealing membership from whoever arrived
  by the boundary makes candidacy timing-dependent — contradicting the
  Section 8 "no timing input" rule — and epoch duration becomes the
  arrival-gather window, collapsing into a fast-narrow-epoch attractor
  (measured on 2,048x32 at k = 16: median wave width 36 vs 227, and
  wait-all frames recovered +23–57% across every shape). Superseded by
  the wait-all frame quorum; evidence in `vesuvius-phase1-report.md`.
- **Device stop-spec / lane masking on stop**: dropped for simplicity;
  `pipeline.close` semantics already cover dead lanes at bounded, measured
  cost (Section 6).
- **Per-step host reactivity at k > 1**: not degraded — unsupported
  (Section 9). Keeping it would require mid-frame membership mutation or
  late-binding lanes, both of which reintroduce the host into the frame's
  interior — the exact dependency Vesuvius exists to remove.
- **Per-step fault recovery inside a frame**: execution is fail-stop and the
  frame settles once (Section 6.2); recoverable errors are excluded per-lane
  at prepare instead.
- **Mid-frame KV revert on failure**: rejected as unnecessary machinery — a
  poisoned lane's KV is never observed, so the failure contract is
  observational (Section 6.2), not transactional.
- **The frame as the graph-capture unit**: capture classes are per wave
  (Section 3.3); a frame-level pure-decode class would never trigger on
  short-output churn, where essentially every frame carries chunks.
- **A dedicated control-word operation in WIT**: rejected — existing `set`
  already has the right semantics (in-place replace of the committed front
  cell), and the in-flight claim guard already scopes it: boundary-only on
  consumed channels, any-time on read-only latest-value channels
  (Sections 4, 5, 9).
- **Timing-adaptive k**: forbidden (heuristics ban). k is static per
  deployment; behavior may vary across deployments only the way page size
  already does.
- **A frame resource in WIT**: a frame is a call, not an object.
- **k as a per-frame negotiation**: channel capacities enter PTIR container
  identity; changing k mid-stream forces re-trace. Static k, sized once.

## 14. Open Items

- Deterministic derivation rule for k from the model profile (and the
  per-wave token budget constant) — operator-set vs profile-derived.
- Doorbell/waker publication path: wakeup latency budget relative to one
  step (it bounds the interleave slack, Section 7); polling vs host-callback
  mechanism, and its CPU cost at fleet scale.
- Validation-completeness audit: enumerate every per-lane error class and
  prove each is caught at submit/prepare (the load-bearing premise of
  Section 6.2).
- Wave-graph maintenance: the varlen-class bucket lattice (padding waste
  vs bucket count and capture memory); exec-update vs re-instantiate when
  the sealed lane width changes across frames (the existing
  request-lattice bucketing is the precedent).
- Latest-value re-read mechanics in the driver: control words must be read
  from live memory each step (today host-writer rings are pulled and
  snapshotted at prepare); pick the visibility path (pinned read vs async
  H2D) and confirm the in-fire `read` binding is non-claiming.
- Slot-collision at admission: new lanes all place chunks in their earliest
  slots, so wave 0's token budget gates admission with no composer
  relocation to spread them — assess the TTFT/occupancy cost against the
  determinism win of fixed slots.
- Per-token `take` overhead at fleet scale (batch drain was considered and
  dropped) — confirm the guest-side call cost is acceptable at n = 512,
  k = 16.
- Lane-parallel frame-submit path design (herd at boundaries; avoid a
  second bind-convoy).
- Variable last chunk vs fixed channel cell shape: padding + indptr
  convention.
- Telemetry: frames, frame occupancy (lanes present per wave), missed-frame
  counts per lane, boundary admission decisions.
- Device RNG for stochastic sampling inside frames: counter-based,
  deterministically seeded per (lane, step).
