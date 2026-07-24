# Project Venus — frame-native driver execution

**Thesis.** Make the *frame* (k consecutive forward steps, sealed as one unit)
the conceptual GPU forward-pass unit end to end. The frame crosses the driver
ABI. Inside the driver it is an **amortization unit for host cost** — prepare
once, settle once — **not an execution unit**: execution stays per-step graph
replay, enqueued ahead of the GPU. Goals: maximize throughput in the regime
where the GPU is faster than the host, and *simplify* — the design must delete
more machinery than it adds.

## Why (measured)

Today every step pays the full host pipeline (begin/pull_validate → resolve →
compose → plan → h2d → enqueue → settle). At k=1, 2048×32:
`settlement_enqueue` is 65% of driver-lane host time; the GPU shadow of these
per-wave host lines is the 100µs–1ms compute-gap class ≈ **24% of wall**. M2
(settlement deferral) proved this cannot be fixed by *moving* bookkeeping while
keeping the per-wave structure: per-wave publication forces per-wave sync, and
the next wave's `begin` waits on the prior wave's `publications_done`. The
structure itself must change.

## The model: two decoupled tracks

**Enqueue track (host, sole producer).** Processes sealed frames in order:
frame prepare (validate once, admit once against frame-union demand, plan once,
write all k per-step parameter blocks into the pinned staging ring), then
enqueue k× (H2D → graph replay → publication ops). This track **never waits on
the GPU** — no completion waits, no event waits. Its only backpressure is
staging capacity (ring sized in frames, ≥2).

**Completion track (one callback per frame).** Latches k step outcomes,
terminal cells, completion notify, `cuda_settled`, fences, arena/lease return.
Feeds back into resource accounting only. It **never gates the enqueue track**.

The stream serializes everything: `H2D_i → replay_i → publish_i → H2D_{i+1} → …`
across step *and frame* boundaries alike. On the stream, a frame boundary is
nothing — consecutive ops. The boundary exists only on the host tracks.

## Principles

- **P1 — The frame is a closed system.** Step i+1 observes step i only through
  device state (DecodeEnvelope token feedback, on-stream channel writes). The
  host observes execution only at frame boundaries. This dissolves the two
  couplings that killed M2: the `publications_done` → `begin` wait (k× → 1×,
  overlapped with the previous frame) and per-wave commit-word volatility.

- **P2 — No whole-frame capture.** Reuse today's per-step graph exec cache and
  the staging-ring → persistent-slab H2D machinery as-is. Queuing step n+1's
  replay while step n runs is sufficient and already stream-legal; what blocks
  it today is only the host loop. This kills the {R,k} capture lattice, the
  mixed-frame (prefill+decodes) exception, and any new capture surface — a
  mixed frame is just heterogeneous step bodies enqueued in order.

- **P3 — Publication streams, settlement batches.** Per-step data publication
  (epilogue, commit-bump, frame-carrier D2H mirror, doorbell) stays on-stream,
  fire-and-forget — token streaming latency is unchanged. Only control
  settlement (outcomes, notify, fences) moves to the frame tail. This keeps
  §6.2's guarantee without its sync.

- **P4 — Stream work is SUCCESS-only.** Every per-lane recoverable failure is
  caught at guest submit or engine admission (audit Q1). Mid-frame RETRY is
  made vacuous by admission: `validate_frame`'s device-ring arm bounds the
  whole frame's publication backlog, and frame-union admission reserves all k
  steps' pages up front. Batch-scoped FAILED is synchronous at post time,
  before enqueue. Hence frames are **atomic** (all-or-nothing), one completion
  each, and enqueue-ahead across frames can never need un-doing.

- **P5 — Host waits are severed, not tuned.** Three specific breaks:
  (1) `begin` no longer waits `publications_done` — stream order replaces host
  observation; (2) the depth gate counts *staging capacity*, not in-flight
  completions; (3) settle occupies the lane once per frame, not per wave.

- **P6 — k=1 is wire-identical.** A 1-slot frame degenerates to today's wave.
  Existing oracle dumps remain the regression floor at every k.

## The bubble-freedom inequality

The GPU never idles iff, for every frame k+1:

```
T_prepare(frame) + k·T_enqueue(step)  <  k·T_gpu(step)
```

Current numbers: T_prepare ≈ 2–3 ms (today's per-wave host cost, paid once per
frame), T_enqueue ≈ tens of µs, T_gpu ≈ 5–7 ms at 256-lane decode. k=1
satisfies it with one step of slack (today's 24% gap class is that slack's
jitter, measured); **k is the knob that buys slack** as GPUs get faster
relative to hosts. This inequality is the design's load-bearing invariant.

## The arrival contract

The standing assumption — *frame n+1 has always arrived while frame n runs* —
is a three-party contract, not something the runtime can prove:

- **Guest**: keeps a run-ahead window ≥ 2 FRAMES (= 2k fires) in flight per
  lane (`submit_ahead`, window = 2k). The unit is frames, not fires: frames
  settle atomically, so results arrive only at frame boundaries — a 2-fire
  window at k ≥ 2 is a single frame and drains the pipe every boundary
  (measured: k=2 28k → 34.4k when the window became two frames).
- **Engine**: seals early (while the current frame executes) and posts the
  sealed frame immediately — posting is decoupled from retirement.
- **Driver**: prepare + enqueue fits inside k·T_gpu (the inequality).

A breach (slow lane delaying the quorum, generation-boundary churn) is a
**bubble, not a bug** — correctness is held by stream order regardless. Make
breaches observable: one counter for "enqueue horizon caught by execution".
Boundary churn remains the orthogonal lifecycle track.

## ABI (v14 — one jump, one path)

`PieFrameDesc`: shared invariants (instance set, page translation, base
geometry, channel accesses) + k per-step sections (position/kv_len advance,
write-target step; step 0 may carry prefill payloads — multimodal side-channels
ride the step that owns them). Per-step indptr/last-page arrays are derived
arithmetically at frame prepare — sealed frames make step geometry pure
arithmetic. The ABI changes **once**: v14 is the delta form from day one; the
driver expands internally where today's structures are still consumed.

**Single path.** `LaunchFrame` is the *only* submission unit engine-side. A
1-slot frame is today's wave (P6), so there is no legacy launch path to keep:
remote and TP>1 are served by a thin **decomposition adapter** at the driver
interface edge (frame → per-step posts), never by a second scheduler path.
Admission folds into the frame post itself (reply carries EXHAUSTED /
IMPOSSIBLE) — the `prepare`/`launch_prepared` entry points and the whole lease
registry are deleted. Elastic-pool trim rule: never below the enqueue
horizon's committed targets (trim at horizon-empty points only).

## Closed decisions

1. **Single path** — one scheduler path, adapter at the edge (above).
2. **One ABI jump** — v14 delta form directly; no interim array-of-descs.
3. **Step = ordered homogeneous sub-batches**, computed at frame prepare.
   Mixed geometry classes within a step (fleet lanes at different phases)
   become sequential enqueues, never a fallback path and never a host wait.
4. **The snapshot/commit-word ring is the one new mechanism.** The
   `publications_done` → `begin` wait is a buffer-reuse hazard (pull_validate
   rewrites the per-instance snapshot slot the prior step's publication still
   reads); sever it by ringing those slots to the enqueue-horizon depth
   (frames-in-flight × k). Design this one carefully; everything else deletes.
5. **RETRY vacuity is a build gate.** Enumerate every path that leaves a
   lane's commit word unbumped and prove each is admission-bounded before
   relying on P4. Provable → makeup replay, retry-auto-flush, and the
   force-retry battery are deleted *in the same change*. Any unprovable edge
   degrades to whole-frame atomic RETRY (re-post) — still one mechanism.
6. **Both dormant levers die**: ABI v13 `settle_defer` (structurally
   superseded) and the prefill graph-capture lever (`PIE_PREFILL_GRAPH_*`,
   negative economics). Git remembers.
7. **Cancel resolves at frame boundaries** (posted frames are atomic; today's
   semantics already never cancel posted fires). Watchdog budgets re-scope ×k.
   Fire-timing schema becomes frame-scoped events + the bubble counter.
8. **k stays config-constant** (default 2); dynamic k waits for production
   instrumentation of the inequality.

## What gets deleted

The entire M2 apparatus (ABI v13 `settle_defer`, park/drain/arm accumulator,
retry-auto-flush, `FlushSettle`, group-aware depth gate), the
`prepare`/`launch_prepared` lease surface and its double batch build, the
engine's wave-level makeup ordering and `wave_outstanding` chaining gate, the
prefill graph-capture lever, and the whole-frame-capture idea. The
frame-native path must end net-negative in mechanism count.

## RETRY audit — gate #5 result (closed)

Exhaustive: `pass_commit` has exactly four writers — the seed (:= 1 at stage
setup), `k_stage_readiness` (ANDs two conditions), and the two compose
fail-stop kills. `entry.poison` is never set at stream settle (M2 entry
audit Q1), so every uncommitted lane classifies as RETRY today. Root sources
and Venus disposition:

- **S1 — ring full** (`k_stage_readiness` need-empty): statically vacuous —
  `validate_frame`'s §5 device-ring arm bounds the whole frame's publication
  backlog at admission. Proven, landed, tested.
- **S2 — consume not ready** (need-full): only reachable when a producer→
  consumer pair lands in one batch (or cascades from an upstream kill).
  Frame prepare orders sub-batches so visible dependencies precede their
  consumers; anything residual takes decision #5's backstop — **whole-frame
  atomic RETRY** (re-post), no per-lane makeup. Zero organic retries across
  every cert to date says the backstop is practically dead code.
- **S3 — compose chain-kill** (`compose_fixed_decode` /
  `compose_decode_envelopes` device-side geometry validation): deterministic
  — a retry re-derives the same state and kills again; today's RETRY
  classification for kills is a misclassification absorbed by the retry
  budget. Venus reclassifies kills as per-lane **FAILED** at frame settle,
  loud. Requires a per-lane kill bit alongside the commit word — folded into
  the decision-#4 snapshot-ring entry (still the one new mechanism).

Consequence: the per-lane RETRY class, makeup replay, retry budgets, and the
force-retry battery all delete; the settle callback classifies
committed → SUCCESS, killed → FAILED, else → whole-frame RETRY (backstop).

## Build strategy — north star, no interim milestones

Build the end state directly; certify once; then tune. No incrementally
landed, individually certified half-forms — dead code is removed in the same
change that obsoletes it, and no compatibility levers are left behind.

Construction order (one landing, bottom-up): ABI v14 → driver frame pipeline
(`frame prepare / step enqueue / frame settle` as modules, replacing the
`handle_fire_batch` monolith zones) → engine frame posting (worker posts
`LaunchFrame`; `frame.rs` reduces to the seal policy: quorum, truncation,
arrival) → edge adapter (remote/TP) → deletions → certification.

Certification campaign (single, at the end): engine units green; **token
streams EXACT vs today's oracle dumps at k∈{1,2,3}** (semantics are unchanged
— tokens are the non-circular regression floor; the byte-exact *wire* floor is
deliberately re-baselined once, since v14 changes the wire by design);
2048-fleet certs at every k; zero watchdog; bubble counter reported. Then the
refine phase: profile, close bubbles, tune horizon depth.
