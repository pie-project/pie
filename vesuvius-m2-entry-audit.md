# M2 Entry-Gate Audit — Per-Lane Error/Rejection/Abnormal-Completion Classes

Audited at commit `8184a5f1` (post scheduler-path unification), 2026-07-22,
as the entry gate for M2 (frame-boundary settlement) required by spec §6.2
requirement 1 ("validation completeness at prepare"). Read-only audit; all
file:line cites re-verified at that commit.

## 0. Executive verdict

**§6.2 requirement 1 holds today, with three flagged constraints and no
strict GATE-BLOCKER.** The load-bearing structural fact: **the driver's
asynchronous stream-settlement path (`notify_runtime_callback`) can produce
exactly two per-lane outcomes — SUCCESS and RETRY. Per-lane FAILED is
unreachable at stream settlement** (the `entry.poison` flag that would
produce it is declared, reset, and read, but never set — verified by
exhaustive grep). Every FAILED terminal in the system is **batch-scoped and
written synchronously on the driver lane thread at launch time**
(`launch_impl`'s catch blocks), before any wave of the batch has settled
on-stream. Every guest-recoverable per-lane error class (geometry, staged
cells, channel capacity, page binding, RS binding, masks) is caught at guest
submit or engine admission. What remains discoverable mid-frame: (a) the
RETRY/makeup correctness mechanism (out of scope as an error class, but its
*handling* is per-wave-retirement-driven — §5.1), (b) per-wave driver
prepare admission (an M3 concern, batch-scoped, near-unreachable mid-frame —
§5.2), and (c) one letter-of-the-spec gap: device-only ring capacity is not
checked by §5 frame validation and surfaces only as a device-side RETRY
loop (§5.3, ambiguous).

## 1. How outcomes flow (architecture snapshot, verified)

- One wave = one `PieLaunchDesc` batch posted by the engine's driver lane
  thread (`worker.rs:868-895`) via `pie_cuda_launch`/`pie_cuda_launch_prepared`
  (frozen ABI, `interface/driver/include/pie_driver_abi.h:851-862`).
- **Synchronous rejection channel**: the ABI call's `int32` return
  (`PIE_STATUS_*`, `pie_driver_abi.h:68-93`). Engine maps it to
  `LaunchState::Failed` (`worker.rs:3960-3970`) and rejects every batch
  member unsubmitted at retirement (`worker.rs:3542-3591`).
- **Asynchronous per-lane channel**: one `PieTerminalCell` per lane
  (`pie_driver_abi.h:442-460`), a bare u32: PENDING/SUCCESS/FAILED/RETRY
  (`pie_driver_abi.h:811-826`). No error payload crosses the ABI at
  completion; FAILED detail exists only on driver stderr.
- Driver execution is synchronous on the lane thread: `launch_impl`
  (`context.cpp:1861`) → `handle_fire_batch` (`compose.cpp:343`) = begin →
  resolve_descriptors → compose → forward → `Dispatch::finish`
  (`dispatch.cu:3889`), which enqueues on-stream: commit-bump kernel,
  per-lane publish D2H copies, settle kernel (`channels.hpp:312-351`), then
  `cudaLaunchHostFunc(notify_runtime_callback)` (`dispatch.cu:4242-4243`).
- `notify_runtime_callback` (`dispatch.cu:1514-1633`) computes per lane:
  `committed = *commit_host != 0`; `failed = entry.poison` (**always false
  today**); `retry = !failed && !committed`; writes the terminal cell
  (`dispatch.cu:1590-1597`), wakes channel waiters on commit, fires the
  batch completion notify.
- Engine retirement (`retire_ready_launches`, `worker.rs:3528-3736`) pops
  FIFO, reads terminal cells via `resolve_from_terminal`
  (`driver/completion.rs:635-664`), converts RETRY into
  requeue-or-per-lane-failure.
- Guest surfacing: a failed completion resolves with a message; on drain
  (`fire.rs:1475-1544`) `FinalizeAction::Fail` → `poison_readers` +
  `fail_pass` + pipeline failure (`fire.rs:584-595, 1650-1656`). Driver-side
  channel poison words additionally fail `take`/`read`/`set` directly
  (`channel.rs:553-567, 767-777`).

## 2. Exhaustive class table

Legend (f) M2 verdict: **NONE** = caught at submit/prepare, before frame
execution; **SAFE** = surfaced later but batch/fleet-fatal anyway, or pure
bookkeeping whose k-wave delay is benign, or carried by per-wave publication
which M2 keeps; **GATE-BLOCKER** = per-lane, recoverable, discoverable only
at per-wave settlement.

### 2.1 Guest submit validation (per-lane; recoverable; guest sees `Err(String)` synchronously)

| # | Class | Evidence | (f) |
|---|---|---|---|
| S1 | Frame shape: `slots.len() != k`; all-`none` frame | `pipeline/fire.rs:1106-1122` | NONE |
| S2 | Unattached pass in a frame slot | `fire.rs:1123-1130` | NONE |
| S3 | §5 *staged*-class insufficiency (host-writer cells staged < frame consumes) | `fire.rs:1206-1222`; `channel.rs:735-744` | NONE |
| S4 | §5 *latest-value* control word with no committed cell | `fire.rs:1223-1234`; `channel.rs:757` | NONE |
| S5 | §5 host-reader ring capacity overflow (static `2k-1` rule) | `fire.rs:1238-1258`; `channel.rs:746` | NONE |
| S6 | Mid-frame host submit failure → **FrameTruncate** (already-submitted fires stand; frame seals truncated) | `fire.rs:1146-1168`; `worker.rs:2520-2526`; `frame.rs:307-314` | NONE (arrival bookkeeping) |
| S7 | Pipeline closed / already failed / pass failed by earlier fire | `fire.rs:713-728, 757-762` | NONE |
| S8 | Geometry evaluation errors (host wire + envelope template), incl. in-band `-1` skip token (RV-12) | `fire.rs:763-820` (skip guard 796-803) | NONE |
| S9 | Attention-mask evaluation/lowering errors | `fire.rs:773-779, 804-813` | NONE |
| S10 | KV working-set declaration errors (`BadWriteSet`, invalid resolve, empty writable, read page escapes readable) | `fire.rs:850-872, 915-932`; `store/kv.rs:50-58` | NONE |
| S11 | **KV page binding at submit** (eager, §8): realize + backing frontier `OutOfPages` (cache-root reclaim retry), contention-orchestrator grant/suspend loop | `fire.rs:158-250, 938-972`; device-geometry analogs `fire.rs:284-335` | NONE — discharges the page class |
| S12 | KV translation snapshot failure | `fire.rs:973-986` | NONE |
| S13 | RS binding errors: count/scope/prepare, cross-model WS | `fire.rs:479-529` | NONE |
| S14 | Scheduler submit error (rejects, fails pass + pipeline) | `fire.rs:1026-1057` | NONE |
| S15 | Channel op errors at guest calls: `Full` (put back-pressure), `WrongRole`, `BadLength`, `SeedAlreadyStaged`, `Poisoned`, `Closed`, `Empty`, **`InFlight` (set on a fire-claimed cell — the "late set" guard)** | `channel.rs:143-184, 488-535, 541-599 (InFlight 585-587)` | NONE |
| S16 | Device-resolved pass with a bounded readable span | `inferlet/host/forward.rs:747-752` | NONE |
| S17 | Channel attach/registration errors at bind; geometry-class ACK at bind, driver-verified against the registered trace | `forward.rs:753-781`; `pie_driver_abi.h:397-429` | NONE |

### 2.2 Engine admission (per-lane; recoverable; rejection via completion)

| # | Class | Evidence | (f) |
|---|---|---|---|
| A1 | Cancelled before admission | `worker.rs:2444-2445` | NONE |
| A2 | Process terminated before admission (+ post-terminate ghost-lane guard) | `worker.rs:2446-2450, 2463-2478` | NONE |
| A3 | Unknown/stale instance | `worker.rs:2451-2455` | NONE |
| A4 | Single-request structural limit violation | `worker.rs:2456-2457` | NONE |
| A5 | Scheduler shutting down (new submissions) | `worker.rs:2458-2459` | SAFE (shutdown) |
| A6 | Terminate purge of queued fires + `FramePolicy::on_lane_leave(purge)` | `worker.rs:2411-2435, 2617-2662`; `frame.rs:341-359` | NONE |
| A7 | Rejected-at-admission fire still counts toward frame arrival completeness (k>1) | `worker.rs:2463-2478`; `frame.rs:270-281` | NONE |
| A8 | k=1 FrameStamp synthesis on the accept path only | `worker.rs:2480-2499` | NONE |

### 2.3 Engine dispatch (`dispatch_frame_batch` — runs per wave, i.e. mid-frame for waves 1..k−1)

| # | Class | Evidence | (f) |
|---|---|---|---|
| D1 | Cancelled/settled while queued in a sealed wave → reject + `on_fire_dropped` | `worker.rs:3350-3360`; `frame.rs:374-383` | SAFE — guest-initiated; dispatch stays per-wave under M2 |
| D2 | Instance stale at dispatch | `worker.rs:3361-3368` | SAFE |
| D3 | Driver prepare **EXHAUSTED** → 1 ms backoff requeue, wave books preserved | `worker.rs:3418-3425`; `context.cpp:1785-1788` | SAFE-with-caveat (§5.2) |
| D4 | Driver prepare **IMPOSSIBLE** (demand > hard ceiling) → reject all selected | `worker.rs:3426-3440`; `context.cpp:1789-1792` | AMBIGUOUS (§5.2) |
| D5 | Driver prepare transport error → reject all selected | `worker.rs:3442-3450` | AMBIGUOUS (§5.2) |
| D6 | `AdmissionWatermark` covered-demand skip (no violation class; reset on pool resize) | `worker.rs:655-707, 734-744, 760-785` | NONE — routes covered launches to launch-time arena commit (L4) |

### 2.4 Driver synchronous submit validation (batch-fatal, ABI status → all members rejected unsubmitted)

| # | Class | Evidence | (f) |
|---|---|---|---|
| V1 | ABI resource validation: physical page index bounds; page/last-len mismatch; query rows sans pages; RS slot ranges; multimodal limits | `entry_validation.hpp:135-260`; `context.cpp:1681-1692` | NONE (pre-execution) |
| V2 | `required_kv_pages` > device pages; empty PTIR hash set; instance resolve; registry `validate_launch` | `context.cpp:1693-1707` | NONE |
| V3 | Prepared-lease mismatch (unknown lease id, target bytes exceed lease) | `context.cpp:1865-1892` | NONE |

These are the engine's own bugs by construction; blast radius one batch,
surfaced before anything executed.

### 2.5 Driver launch composition (host; terminal cells written synchronously in `launch_impl` catch blocks — not at stream settlement)

| # | Class | Evidence | (f) |
|---|---|---|---|
| L1 | **Whole-batch RETRY** (`RetryableLaunchError` → all cells RETRY + notify): writer-input visibility ("late put", `dispatch.cu:4925-4931, 5332-5342`); readiness snapshot `ready==0` (`dispatch.cu:5545-5551`); stateful-RS `settle_readiness` miss (`dispatch.cu:3733-3769`); dense-mask solo contract (`compose.cpp:651-660`); mixed structured-mask coverage (`compose.cpp:718-722`) | catch: `context.cpp:1951-1978` | SAFE — synchronous with posting |
| L2 | **Whole-batch FAILED** (other `std::exception` → `Dispatch::abort` + `settle_failed_launch` + all cells FAILED): compose.cpp throws — fallback KV-write containment (`dispatch.cu:5581-5595`), workspace overflow (`compose.cpp:972-976`), RS plan/shape (`compose.cpp:890-941, 1409-1437`), sampling CSR (`compose.cpp:1376-1396`), mask capacity (`compose.cpp:1094-1121, 1215-1218`), MTP preflight (`compose.cpp:65-215`), envelope enqueue (`compose.cpp:1446-1516`); begin throws (`dispatch.cu:3581-3617, 1851-1863`); **CUDA API errors via `CUDA_CHECK` incl. alloc OOM** (`cuda_check.hpp:14-22`); non-prepared elastic commit exhausted (`context.cpp:1914-1923`); lease-floor trim (`context.cpp:1896-1908`); KV-proportionality assert (`context.cpp:1925-1944`) | catch: `context.cpp:1979-2011`; abort: `compose.cpp:1870-1878`, `dispatch.cu:4274-4318` | SAFE — batch-scoped, synchronous at post |
| L3 | `settle_failed_launch` — poisons every host-visible channel of every batch member (poison word + waiter wake) after draining both streams | `dispatch.cu:4382-4427`; engine reads: `channel.rs:355-374, 553-567, 767-777` | SAFE — launch path, unchanged by M2 |
| L4 | Watermark-skip path: covered launch skips prepare; launch-time atomic arena commit failure ⇒ L2 FAILED | `worker.rs:760-785` + `context.cpp:1914-1923` | SAFE (synchronous); noted §5.2 |

### 2.6 Device-side self-invalidate (per-lane; surfaces as terminal-RETRY at that wave's settlement)

| # | Class | Evidence | (f) |
|---|---|---|---|
| X1 | **Ticket gate**: expected head/tail vs live rings, require-input, publish-capacity (`tail-head < cap1-1`) → `atomicAnd(pass_commit, 0)` — the §3.1 ordering gate | `channels.hpp:221-299` (capacity 249-256) | SAFE as mechanism (§5.1 handling; §5.3 capacity edge) |
| X2 | **Fixed-decode containment kill**: commit==0, port not ready, null ports, indptr, page count, translation bounds, write position outside `[write_lower_bound, write_upper_bound)`, physical page bounds → `*commit=0` + `chain_kills++`, dummy-run | `dispatch.cu:441-627` (kill 546-560) | §5.1 |
| X3 | **Decode-envelope containment kill**: device position escapes containment window or template page span | `dispatch.cu:891-980` (kill 918-927) | §5.1 |
| X4 | Chain-kill loud host report — diagnostics only, folded at next enqueue / `stats()` | `dispatch.cu:4627-4642, 5023-5035, 2102-2116` | SAFE |
| X5 | Settle kernel publishes only if committed; host-commit mirror is the settlement input | `channels.hpp:312-351`; commit-bump `channels.hpp:118-165` | SAFE — M2 keeps per-wave publication |

**Terminal-RETRY vs error:** the ABI cannot distinguish X1 (transient,
replays clean) from X2/X3 (deterministic geometry violation). The only
discriminators are engine-side at retirement: retry budget (1024 attempts
≈ 1 s, `worker.rs:103-108, 2609-2615`) and `retry_classifier`
(poisoned/closed endpoint, consumerless device ring after close —
`fire.rs:1003-1025`, `channel.rs:355-383`).

### 2.7 Driver stream settlement (`Dispatch::finish` enqueue + callback)

| # | Class | Evidence | (f) |
|---|---|---|---|
| F1 | `finish` host-side validation throws (staged-state, phase counts, vocab/MTP layout, logits bounds, arena capacity, host-output count) ⇒ escalates to L2 batch-FAILED, synchronous | `dispatch.cu:3912-4025, 4103-4107, 1749-1787` | SAFE |
| F2 | Epilogue/sampling kernel-enqueue errors ⇒ L2 | `dispatch.cu:4010-4025` | SAFE |
| F3 | Async callback outcomes: **SUCCESS / RETRY only**; `entry.poison` never set (grep: only `dispatch.cu:1411, 1447, 1526`; `entry.poisoned` finalize list 4152-4165/1584-1589 dead in practice) | `dispatch.cu:1514-1633` | SAFE — M2 splits this callback |
| F4 | Callback wakes suppressed during driver shutdown (cells still written) | `dispatch.cu:1517-1520, 2167` | SAFE |

### 2.8 Engine retirement (`retire_ready_launches`; per-wave today — the path M2 moves to the frame tail)

| # | Class | Evidence | (f) |
|---|---|---|---|
| R1 | Committed → success; token accounting | `worker.rs:3604-3610` | SAFE (publication per-wave; completion resolve ≤ k−1 waves late — SDK reads tokens via channels) |
| R2 | FAILED terminal → per-lane failure, generic message (driver detail stderr-only) | `worker.rs:3611-3614`; `completion.rs:647-650` | SAFE — FAILED only from synchronous batch path (L2) |
| R3 | RETRY → requeue as makeup (backoff, `QueueEnd::Front`) | `worker.rs:3648-3661, 3675-3677`; `frame.rs:402-417, 655-694` | §5.1 |
| R4 | RETRY + cancel-requested → reject "cancelled after native attempt" | `worker.rs:3623-3626` | SAFE |
| R5 | RETRY + stopping → reject "scheduler shutdown interrupted a retrying fire" | `worker.rs:3627-3630` | SAFE (shutdown) |
| R6 | RETRY + RS-carrying (retry-ineligible) → immediate per-lane failure | `worker.rs:3631-3634, 226-230` | SAFE |
| R7 | RETRY + permanent-retry classifier → per-lane failure without burning budget | `worker.rs:3635-3642`; `fire.rs:1003-1025` | SAFE — cause already published per-wave/synchronously |
| R8 | RETRY budget exhaustion (>1024) → per-lane failure | `worker.rs:3643-3647, 103-108` | §5.1 (k× slower under frame-tail retirement) |
| R9 | Terminal still PENDING / invalid → `settlement_error` (bug guard) | `worker.rs:3664-3672`; `completion.rs:636-642, 655-662` | SAFE |
| R10 | Batch completion `Err` (driver callback guard closed — teardown/panic) → reject all + close instance wait slots | `worker.rs:3703-3731`; `completion.rs:615-622` | SAFE (fleet-fatal) |
| R11 | `LaunchState::Failed` (sync ABI reject) → all members rejected | `worker.rs:3542-3591` | SAFE (synchronous at post) |
| R12 | Guest finalization: failed completion → `poison_readers` + `fail_pass` + pipeline failure; KV/RS txn finalize folds in | `fire.rs:1579-1669, 1522-1544, 584-595` | SAFE |

### 2.9 Fleet-fatal classes

| # | Class | Evidence | (f) |
|---|---|---|---|
| G1 | Sticky CUDA device fault: every later CUDA call throws → every later batch L2-FAILED; settlement-cleanup sync failures logged, arena leaked | `cuda_check.hpp`; `dispatch.cu:1800-1819, 4244-4263` | SAFE (the §6.2 "remaining mid-frame failures" class) |
| G2 | TP follower loop exception → `tp_comm->abort()` | `context.cpp:301-310` | SAFE |
| G3 | Driver lane thread panic / send failure → completions closed → R10 | `worker.rs:745-746, 942-945` | SAFE |

## 3. Specific questions

**Q1 — settlement-time outcomes beyond {published-ok, RETRY, fleet-fatal}?
None exist at driver stream settlement.** `notify_runtime_callback` maps
each lane to SUCCESS (`committed`) or RETRY (`!committed`); the FAILED
branch requires `entry.poison == true`, which no code path sets. Two
engine-side retirement degenerates are bug/teardown guards (R9, R10), not
error classes. Nuance: **terminal-RETRY is overloaded** — it carries both
makeup-replay (X1) and deterministic containment kills (X2/X3), separable
only by engine-side budget/classifier at retirement.

**Q2 — detected at compose/prepare, reported at settlement?** Strictly none
ride the deferred stream-settlement callback: all compose/prepare failures
(L1, L2, V*) write terminal cells and fire notifies **synchronously inside
`launch_impl`** (`context.cpp:1951-2011`). But the engine *consumes* those
cells at `retire_ready_launches` — per-wave today, frame-tail under M2.
M2-SAFE with reporting-at-frame-tail (delay ≤ k−1 waves, blast radius
already whole-batch): L1, L2 (+ its immediate channel-poison side effect
L3, which wakes waiters independently of retirement), V1-V3.

**Q3 — shutdown drain (k=1 unification):** engine dispatch + retirement;
per-lane; recoverable; intentional. `stopping` set (`worker.rs:2390-2392`);
new submissions reject at admission (A5); `dispatch_frame_work` bypasses
the wait-all gate and posts accepted fires in queue order = per-lane ticket
order (`worker.rs:3216-3227`); a bouncing fire rejects at retirement (R5)
instead of requeuing — correct, its predecessor may never replay. M2: SAFE
(shutdown is fleet-terminal). Note: the drain scan stops at the first
queued async control (`worker.rs:3191-3214`) — consistent with FIFO
barrier semantics.

## 4. Verdict detail

**§6.2 requirement 1 holds for every designed error class.** Geometry (all
three classes ACK'd at bind; host-wire geometry fully evaluated at submit;
envelope templates built at submit; DeviceGeometry leases physical pages at
submit), staged-cell counts (S3), channel capacity (S5 host-reader;
put-side back-pressure never blocks a frame), page binding (S11 eager
binding + per-wave driver lease D3/D4), RS (S13), masks (S9) — all excluded
at submit/admission, per-lane. Driver validation layers (V*, L2) are
backstops for engine bugs, batch-scoped, pre-execution or
synchronous-at-post; device-side kills (X2/X3) are the fail-stop backstop
the spec's abort path anticipates. Per-lane FAILED cannot appear from
stream settlement, so once-per-frame settlement cannot *lose* an error
class that per-wave settlement would have caught.

## 5. Flagged constraints (ranked by risk)

### 5.1 RETRY handling is per-wave-retirement-driven (highest; not a spec violation)

The makeup machinery's *input* is per-wave settlement: `plan_dispatch`
gates makeup replay on `wave_outstanding`/`outstanding` draining
(`frame.rs:655-694`), decremented only by `on_fire_retired`
(`frame.rs:404-417`), called only by `retire_ready_launches` per retired
wave batch. Under frame-tail retirement, RETRY discovery moves to the frame
tail: a wave-0 kill's chained successors all bounce too (benign,
self-invalidating), and replay + the retry-budget clock (R8) stretch up to
k×. Nothing is lost, but **M2 must decide**: keep per-wave commit-word D2H
mirrors host-visible so retirement decisions could stay per-wave
(`commit_host` already rides the publication copy list,
`dispatch.cu:4144-4151` — retry inference `dispatch.cu:1524-1527`), or
accept frame-granular makeups and rescale retry budget/backoff.

### 5.2 Per-wave driver prepare (pre-M3) is a mid-frame batch admission point (medium-low)

`dispatch_frame_batch` runs `driver_lane.prepare` per wave
(`worker.rs:3403-3451`): for waves 1..k−1, EXHAUSTED (D3) delays mid-frame
(liveness, not error) and IMPOSSIBLE/Err (D4/D5) reject fires of a
partially-executed frame — batch-scoped, recoverable, mid-frame-only.
Mitigants: demand derives from data fixed at submit
(`AdmissionWatermark::demand`, `worker.rs:663-694`), so wave-j IMPOSSIBLE
would essentially always be IMPOSSIBLE at wave 0; pool resizes are
FIFO-barriered and reset the watermark. This is M3's exact target — the
letter-of-spec gap closes when M3 leases once per sealed frame. To close
by proof instead: show wave-j demand ≤ frame max-wave demand at submit for
all frame shapes (chunked prefill + decode).

### 5.3 Device-only ring capacity is not §5-validated (candidate GATE-BLOCKER, likely vacuous)

`validate_frame`'s `HostRole::None`/seeded arm is a no-op whose comment
claims "remaining per-lane error classes surface at prepare"
(`fire.rs:1259-1265`) — but no prepare-side occupancy check for device-only
rings exists; the only enforcement is the device publish-capacity gate
(`channels.hpp:249-256`), firing mid-frame as per-lane RETRY →
deterministic re-kill → budget exhaustion (~1 s) → per-lane failure. The
consumerless-ring classifier (`channel.rs:376-383`) only decides after
pipeline close. If a legal program can structurally overfill a device-only
ring within one accepted frame, this is a true GATE-BLOCKER; if ticket
pre-reservation at submit (`channel.rs:385-401` — monotonic expected
head/tail per fire, all reserved before dispatch) plus program advance
rules bound occupancy structurally, it is vacuous. **Close before M2
lands**: a proof (or counterexample program) over `reserve_device_ticket`
sequencing that for any §5-valid frame, every device-only ring's
`expected_tail − expected_head < capacity` at each wave — decidable from
the same structural data §5 already reads; if it can fail, the fix is a
fourth arm in `validate_frame`, not an M2 redesign.

### 5.4 Minor observations

- **FAILED detail is stderr-only** (`completion.rs:647-650` generic string;
  `context.cpp:1980` text never crosses the ABI). Pre-existing; M2-neutral;
  frame-tail reporting makes stderr↔wave correlation harder.
- `entry.poisoned` finalize machinery (`dispatch.cu:4152-4165, 1584-1589`)
  is dead code; if M2 rewrites the callback split, wire it (a semantic
  change — per-lane FAILED at settlement would then exist, revisiting Q1)
  or delete it.
- Intra-batch occurrence chaining (`initial_commit = occurrence==0`,
  `dispatch.cu:3626-3627`; commit-bump `channels.hpp:118-165`) is a
  settlement-ordering mechanism M2's per-group settle must preserve for
  multi-occurrence instances.
- `PIE_FIRE_FORCE_RETRY` diagnostic hook (`dispatch.cu:3663-3683`) injects
  X1-class RETRYs — use it in M2's retry-path test battery.

## 6. Risk-ranked summary

1. **§5.1 RETRY/makeup timing under frame-tail retirement** — design
   decision required in M2; correctness preserved either way; rescale
   budget/backoff if makeups go frame-granular.
2. **§5.3 device-only ring capacity** — the only candidate true
   GATE-BLOCKER; decidable statically; close with a proof or a fourth
   `validate_frame` arm before M2 lands.
3. **§5.2 per-wave prepare mid-frame rejection** — pre-existing,
   batch-scoped, near-unreachable; M3 closes it.
4. Everything else: per-lane recoverable classes all at submit/admission
   (NONE), batch classes all synchronous-at-post (SAFE), stream settlement
   provably {SUCCESS, RETRY} only, fleet-fatal classes unchanged by M2.
