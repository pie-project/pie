<!-- Source of truth: wiki page `tensor-ir-plan.md` (slug tensor-ir-log). This folder is the split, on-disk copy for implementation teams. -->

# Thrust 2 — Pipelined Execution & the Fire Rule

**Status:** Ready for execution. Independent of thrusts 1 and 3 (see §6).
**Audience:** Runtime scheduler + driver-ABI engineers; SDK for phase S6.
**Realizes:** [`overview.md`](overview.md) §3 (host-side pipelining) and §7.2
(the fire rule). Contract provided: **C2 — readiness is a word**
([`masterplan.md`](masterplan.md) §3).
**Relationship to prior work:** this thrust incorporates
[`run-ahead-submission-scheduler-handover.md`](../docs/run-ahead-submission-scheduler-handover.md)
(referred to below as **RA**). RA's mechanism inventory (§2.1, §2.5, §2.6, §3,
phases 1–4, 6) carries over; RA's **parity barrier (§2.3) and firing formula
(§2.4) are superseded** by the quorum rule below. RA's locked decisions R1–R2,
R6, R8–R14 stand; R3–R5 and R7 are amended by F1–F6.

---

## 1. Goal

Replace the response-synchronous per-step forward path with a pipelined
lifecycle:

- `execute()` is non-blocking (launch only); results are async handles; a
  result can feed a later forward on-device (producer links) with no host
  round trip.
- Late-bound logit-side inputs are typed tensor-handle writes over one
  primitive (`tensor.write`) with per-binding readiness words.
- The scheduler fires **dense** batches by one rule with three clauses
  (quorum, idle escape, cold hold) and a **depth-1 driver queue** — zero
  steady-state bubble, no completion estimation anywhere.
- KV/RS write transactions are owned per forward, so multiple prepared
  forwards can be safely outstanding.

Standalone value: run-ahead greedy/constrained decode and zero-bubble batching
on the **existing** forward contract and sampler — no dependency on PTIR
programs or the new working-set semantics.

## 2. Current state

RA §3 is the detailed inventory; deltas verified against the tree today:

| area | today | anchor |
|---|---|---|
| fire loop | response-synchronous: `BatchScheduler::run` → `execute_batch` blocks until the driver returns; policy trait exposes `on_arrival`/`on_complete`/`on_fired`/`decide` → `Decision::{Fire, Wait}`; an accumulation-hold test lever exists | `runtime/src/inference/scheduler.rs` (:63–:99, :109, :37) |
| capacity | `BatchAccumulator` counts requests/tokens/pages plus sampler rows, mask bytes, logprob labels, spec tokens; `would_depend_on_batch` already models "consumer whose producer is in this batch" | `runtime/src/inference/scheduler.rs` (:381+, :567) |
| policies | a family of heuristics (adaptive, chunked prefill) | `runtime/src/inference/adaptive_policy.rs`, `scheduler/chunked.rs` |
| ABI carriers | `sampling_program_*`, `sampling_late_{keys,indptr,blob,offsets,lens}`, `SamplingBinding::Tensor{key}`, `pipeline_source_link`, `next_input_{producer_links,src_rows,dest_slots,free_links}` all exist, append-only | `interface/driver/src/schema.rs` (:229–:277, :306–:362) |
| batch merge | `append_request_with_options` does **not** yet merge the fields above | `runtime/src/inference/request.rs` |
| execute path | `execute()` is inline prepare→submit→await→finalize; per-forward txn guard with abort-on-drop exists | `runtime/src/api/inference.rs` (:813–:827, :1076–:1101) |
| direct channel | host→driver bypass precedent (`copy_d2d` via `with_channel(idx, ch.notify(...))`); deferred fire hook exists | `runtime/src/driver/ops.rs`, `runtime/src/driver/channel.rs` (:192) |
| working set | one set-level `pending` revert log (insufficient for >1 outstanding prepared write) | `runtime/src/working_set/kv.rs` (:202, :544, :550) |

## 3. Locked decisions

**Carried over from RA unchanged:** R1 (run-ahead is inferlet-declared
prefetch), R2 (runtime never invents t+1), R6 (late binding is logit-side
only), R8 (KV/RS writes stay explicit and transactional), R9 (repair by
rewriting the named location, no rollback chains), R10 (depth-1 first), R11
(late binding = `tensor.write`, sugar in SDK), R12 (dirty-flag readiness,
clear-on-bind → write → set-ready, release/acquire), R13 (direct channel for
late writes + readback only; forward submit stays on the scheduler path), R14
(non-blocking `execute` + async tensor outputs, no separate launched-forward
resource).

**The fire rule (supersedes RA §2.3, §2.4, R3–R5, R7)** — overview §7.2 is
normative:

| # | decision |
|---|---|
| F1 | **Quorum.** The moment every counted pipeline's next pass is structurally ready, enqueue the dense batch to the driver — one deep, behind the batch in flight. Steady state: quorum completes mid-flight, the device continues in stream order, bubble zero. |
| F2 | **Idle escape.** Device idle and queue empty → fire the ready subset immediately. Missing instances are absent (no holes, no padding rows) and rejoin a later fire; resubmission is just later membership. |
| F3 | **Cold hold.** Nothing in flight at all → hold sub-millisecond for arrivals, then fire partial. A step-scale timeout survives only as a hang backstop. |
| F4 | **Denominator.** Quorum counts pipelines that *can* be ready this round: a pipeline with a submitted next pass, or one whose current pass is in flight and which has been submitting run-ahead. A pipeline blocked on host work (tool call, drained compact) is absent, not awaited. |
| F5 | **Structural readiness.** Submitted, and every input dependency either already satisfied or produced by a pass ahead of it in flight (generalizing `would_depend_on_batch`). Genuinely late host edges (grammar masks) never gate the batch — they park their consuming stage at the device cut point (C2), not the fire. |
| F6 | **No estimation.** No lead-time EWMA, no completion prediction in the decision path. Timing measurements remain as probes only. |

Dense rebatching (RA R3's substance) is kept: batches are reconstructed dense
at every fire; what is dropped is parity phasing and the `fire_at = max(...)`
formula.

## 4. Boundaries

- **Working-set allocation semantics** (slot ids, non-compacting free) are
  thrust 1. Phase S4 below rekeys the *transaction lifecycle* in the same
  files — C4's ownership split; sequence PR trains with thrust 1.
- **Channels** are thrust 3. This thrust builds the C2 wait mechanism against
  dirty flags; the executor-side wait/predicate must be written against an
  opaque `(word, expected)` binding so thrust 3 can swap the producer without
  touching the executor (C2 rule).
- Batching identity: key membership and merge on an opaque `identity: u64`
  (today: derived from the legacy sampler/shape identity). Thrust 3 later
  supplies the stage-tuple hash through the same field (C3).

## 5. Phases

### Phase S0 — Terminology, probes, and scheduler tests

- Freeze names: `launch` (non-blocking execute), `late channel`, `output()` /
  `outputs()`, producer-link lifetime owner. Drop parity vocabulary.
- Add the probe set the rule needs: inter-batch bubble (device idle between
  retire and next launch), quorum latency (last-ready → enqueue), escape rate,
  cold-hold occupancy, dummy-run/readiness-miss rate. Fire-domain probe
  scaffolding exists (`profile-fire`, `runtime/src/probe/fire`).
- Scheduler-only tests (no driver): dense rebatch drops laggards; cold start
  fires immediately; capacity limits split before policy; convoy
  anti-bifurcation (see S5 tests).

Exit: probes land behind existing feature gates; test harness models the
intended semantics.

### Phase S1 — ABI merge preservation (RA phase 1, verbatim scope)

Extend `append_request_with_options` to merge `sampling_program_*`,
`sampling_input_*`, `sampling_late_*`, `sampling_binding_*`,
`pipeline_source_link`, `next_input_*` with correct CSR offsetting and
token-row rebasing; unit tests; legacy sampler SoA unchanged. Decide
`pipeline_source_link` batch-level vs per-request CSR (RA open Q2) here.

Exit: a `ForwardRequest` carrying those fields survives batching
byte-for-byte modulo documented offsets; all existing sampler/spec/multimodal
tests green.

### Phase S2 — Device-resident producer links (RA phase 2)

Runtime link allocation; producer sampled-token buffers retained until every
consumer drains (`next_input_free_links` accounting); executor resolves
`next_input_producer_links`, injects sampled rows into consumer input slots,
skips `u32::MAX` lanes; failure propagation (producer failed → consumers fail
cleanly before sampling).

Exit: synthetic two-forward chain with `t+1` bound to `sample(t)` equals
host-injected replay; re-formed batches may contain consumers whose producers
came from different prior batches.

### Phase S3 — Non-blocking execute + async tensor outputs (RA phase 3)

Split `execute()` into non-blocking launch + async `output()`/`outputs()`
returning `tensor` handles; `tensor.write` for late inputs with the R12
dirty-flag discipline; map typed slot-outputs onto typed tensors (RA open
Q10); `spec-tokens`/`spec-positions` surfacing (RA open Q11); compatibility
wrapper so `generate`-style inferlets keep working.

**S3b — direct late channel:** start with `RequestPayload::LateWrite` over
`ch.notify` (RA option 1); upgrade to the pinned-region poke only if the probe
shows the hop matters. The executor-side wait at the sampling cut point is the
**C2 mechanism** — implement it as `(word address, expected value)` bindings
from day one.

Exit: RA phase-3 exit criteria, plus: late value reaches the driver with no
`InferenceService`/scheduler hop; wait observed before the consuming kernel.

### Phase S4 — Per-forward write-transaction ownership (RA phase 4)

Replace set-level `pending` with txn-keyed pending ownership (the API-layer
guard at `runtime/src/api/inference.rs:1076` already models the lifecycle);
commit/abort touch only that forward's slots; reject overlapping live write
targets unless modeled as overwrite-after dependency; extend to RS
buffers/folds.

Exit: two prepared forwards against disjoint future slots of one working set
in flight safely; abort of one cannot revert the other. (Coordinate the
`kv.rs` diff with thrust 1's M1 — C4.)

### Phase S5 — The quorum core

The scheduler owns: per-pipeline next-pass state, one in-flight batch, one
queued batch (depth 1), the ready accumulator, and the three-clause rule.

1. **Async fire.** Decouple submission from completion: fire on a submission
   path (the deferred-fire hook at `runtime/src/driver/channel.rs:192` is the
   precedent), completion arrives as a callback/notification that retires the
   in-flight slot and promotes the queued batch bookkeeping.
2. **Per-pipeline state machine.** `absent → submitted → structurally-ready →
   enqueued → in-flight → retired`. Structural readiness per F5, computed
   from producer links and submit-time dependencies (generalize
   `would_depend_on_batch`); late-key presence never blocks (F5).
3. **Quorum bookkeeping (F4).** counted = pipelines in state
   `submitted`/`structurally-ready` ∪ pipelines `in-flight` that ran ahead
   last round. A pipeline that misses a fire is simply absent (stateless,
   RA R5) and re-enters on its next submit.
4. **Clauses.** F1: on quorum, enqueue dense batch (capacity split first —
   `BatchAccumulator` and chunked-prefill logic are preserved as the splitter,
   overview §7.2). F2: on device-idle + empty queue, fire ready subset. F3:
   cold hold via the existing accumulation-hold lever, retargeted sub-ms;
   step-scale hang backstop.
5. **Policy plug.** Implement the rule as a `SchedulingPolicy`; keep the trait
   so the legacy policies remain selectable during rollout.
6. Shutdown: drain launched work, fail unsampled late-missing work cleanly,
   exit (RA phase-5 scope).

Tests (beyond RA's): bubble p50 < 100 µs on an 8-pipeline homogeneous decode
fleet (mock timing harness); **convoy anti-bifurcation** — perturb one
pipeline by a one-step stall and assert the fleet re-converges to full-batch
fires within one round instead of settling into alternating half-batches;
agentic-fleet sim (random host-blocked intervals) — escape path dominates,
per-token latency does not regress vs. today's scheduler; laggard rejoin at
next fire; denominator excludes host-blocked pipelines (F4 test).

Exit: collection overlaps flight (inter-fire timing shows it); all S0/S5
tests green; single-stream and batch-throughput no-regression gates hold.

### Phase S6 — SDK run-ahead ergonomics (RA phase 6)

Rust generator first: opt-in run-ahead decode for straight-line generation;
generate `t+1` only when input structure is token-value independent; never run
ahead across dynamic adapter / attention-mask / splice / tool-call boundaries
unless the inferlet supplies fixed structure; keep existing speculation paths
as compatibility until subsumed; measure stop-token redundant final work.

Exit: greedy decode submits `(t, t+1)` pairs; constrained decode late-binds
masks; measured overlap on CUDA.

## 6. Interfaces

**Provides:** C2 (the wait-word mechanism + direct late channel), async fire
and completion notifications, the quorum scheduler, per-forward txns.

**Consumes:** C3's `identity: u64` when thrust 3 lands (until then, legacy
identity); nothing from thrust 1 (S4 rekeys the txn log over whichever
allocation semantics are present — dense array today, slot table after M1).

## 7. Risks

RA §8 items 8.1–8.7 all stand; additions:

| risk | mitigation |
|---|---|
| Convoy bifurcation under fire-on-idle | Quorum (not gpu-free) is the primary trigger; the anti-bifurcation regression test is an S5 gate |
| Denominator thrash (pipelines oscillating counted/absent) | F4 counts only submitted-or-running-ahead pipelines; membership is recomputed per fire, no hysteresis state to corrupt (R5) |
| Depth-1 queue hides a driver-side failure of the queued batch | Completion callback carries per-batch status; queued batch failure surfaces before its consumers' outputs resolve (S2 failure propagation) |
| Two batches resident on device (in-flight + queued) raise peak workspace | Queue admission respects capacity limits against *both* batches; probe peak workspace in S5 |
| `kv.rs` collision with thrust 1 | C4 ownership split; S4 and M1 land in separate PR trains with cross-review |

## 8. Open questions

1. RA §11 Q2 (`pipeline_source_link` granularity) — resolve in S1.
2. RA §11 Q3 (t/t+1 one txn or two with dependency edges) — resolve in S4.
3. RA §11 Q7 (notify vs pinned-region poke) — resolve by probe after S3b.
4. RA §11 Q8 (ring depth for late handles) — deferred; depth-1 is locked (R10).
5. Queue depth > 1: is there a workload where depth-2 enqueue beats depth-1 +
   quorum? (Suspected no — revisit only with bubble-probe evidence.)
6. Multi-driver: quorum per driver or per cohort? (Out of scope until a
   multi-GPU scheduler exists.)

## 9. Code anchors

| area | files |
|---|---|
| Fire loop & accumulator | `runtime/src/inference/scheduler.rs`, `runtime/src/inference/scheduler/chunked.rs`, `runtime/src/inference/adaptive_policy.rs` |
| Batch merge | `runtime/src/inference/request.rs` |
| Execute / txn guard | `runtime/src/api/inference.rs`, `runtime/src/inference/forward_prepare.rs` |
| Driver channel & ops | `runtime/src/driver/channel.rs`, `runtime/src/driver/ops.rs`, `runtime/src/driver/inproc.rs`, `runtime/src/driver/inproc_polling.rs` |
| ABI carriers | `interface/driver/src/schema.rs` (`sampling_late_*`, `next_input_*`, `pipeline_source_link`) |
| CUDA consumers | `driver/cuda/src/executor/executor.cpp`, `driver/cuda/src/spec_expansion.cpp` |
| Working-set txns | `runtime/src/working_set/kv.rs`, `runtime/src/working_set/rs.rs` |
| Prior design | [`../docs/run-ahead-submission-scheduler-handover.md`](../docs/run-ahead-submission-scheduler-handover.md) |
