# Vesuvius Progress Summary — Phase 0 + Phase 1

Date: 2026-07-22
Branch: `tasks/ptir-fusion/agents/golf` (agent: golf; not yet pushed)
Baseline: `73e48f05`
Commits: `2253892a`, `f164dac1`, `1afc992d`, `a6913892` + wait-all rework
Detailed measurements and attribution: `vesuvius-phase1-report.md`

## Status at a glance

| Milestone | Status |
|---|---|
| Phase 0 — re-baseline | **Done** |
| Phase 1 — frame WIT surface (`frame-size`, `max-embed-length`, `submit(on, slots)`) | **Done** (mirrors byte-identical) |
| Phase 1 — SDK (`submit_frame`, sugar for `pass.submit`, constants) | **Done** (all existing inferlets compile unchanged) |
| Phase 1 — §5 submission validation (staged / device-advanced / latest-value, static overflow) | **Done** (enforced at k > 1) |
| Phase 1 — frame scheduling = WAIT-ALL quorum at frame granularity (`scheduler/frame.rs`) | **Done** (operator-corrected principle; see below) |
| Phase 1 — bench inferlet frame-aware reactive loop | **Done** |
| Correctness gates (parity oracles, fleets, k=1 no-regression) | **All green** (re-run after the rework) |
| Phase 1 throughput at k > 1 | **−33% on 2048×32, −2…−19% elsewhere** — wait-all recovered +23–57% over the first cut; the rest is the per-wave driver path (Phase 2's target) |
| Phase 2 — driver multi-step execution | Not started; entry points scoped |

## The scheduling principle (corrected)

The first cut sealed epochs take-who's-there (arrival-complete lanes
only). The operator corrected it: the fundamental rule stays **quorum
infinite wait-all, 1 s report-only watchdog** — the only thing that
changes at k > 1 is the awaited unit: a FRAME per lane instead of a wave.
Landed semantics: the seal waits for every awaited lane's frame (infinite;
watchdog only reports), pending binds hold the seal unconditionally,
graceful close releases a lane immediately (accepted frames drain),
capacity overflow partitions the epoch under the quorum's round rule.
Prepare (driver KV admission) and commit/retire remain per-wave in
Phase 1 — frame-level prepare and frame-boundary settlement are Phase 2
driver work.

## Phase 0 results (the numbers every later claim gates on)

RTX 4090 / Qwen3-0.6B / TP1, greedy, prefix caching off. The spec's
Section 1 profile was stale:

- 2048×32 c512: **24.7k tok/s** warm median (21.8k cold) — not 19.08k.
- vLLM reference re-pinned: **32.2k** (vLLM 0.16.0, n=3) — not 30.81k.
- Current gap to vLLM ≈ **−23%**. The spec's Phase 2 "≥ 25k" gate is
  effectively already met at baseline; the meaningful target is parity.
- Secondary shapes: 512×512 c256 ≈ 19.3k; 64×1536 ≈ 7.3k; 16×1900 ≈ 3.57k;
  512×256 c256 ≈ 27.2k.

## Phase 1 correctness (all gates green, re-run after the rework)

- 369/369 engine unit tests; dummy e2e 10/10 at both k=1 and k=4;
  worker/canary/boot-smoke/dummy-driver suites green; both inferlet
  workspaces compile unchanged.
- CUDA serial oracles: **exact token parity** vs the baseline engine for
  k=1 AND k=16 at t ∈ {1, 2, 15, 16, 17, 31, 32, 33, 256}.
- 2048-request fleets at k=16: 3 consecutive completions, 0 failures,
  0 `frame_wait_watchdog` reports (the gather never stalls a second).
- k=1 default path: measured 24.5k / 27.3k — in the Phase 0 band,
  **zero regression**.

## Phase 1 performance (after the wait-all rework)

k=16 wait-all frames vs the k=1 quorum: 2048×32 **16.1–17.0k vs 24.5k**
(−33%); c0/256 **22.1k vs 27.3k** (−19%); 512×512 −13%; 64×1536 −6%;
16×1900 −2%. Restoring the wait-all principle recovered **+23–57%** over
the superseded take-who's-there sealing on every shape; instrumented
width on 2048×32 went from 562 waves / mean 117 / median 36 to **292
waves / mean 225 / median 227** (uniform, half the waves).

The remaining gap is the per-wave driver path: (1) a frame's
data-dependent waves cannot overlap (the driver polls
predecessor-publication cross-stream and can only RETRY), so each wave
costs a ~0.4–1 ms host gap × k per epoch; (2) a freshly bound lane joins
at the next FRAME boundary (~16 waves) instead of the next wave, which
caps epoch width at ~225/512 on the churny shape (stable fleets are
already at −2…−6%). Both are exactly what Phase 2's device-side
multi-step execution removes.

Consequence: **k > 1 stays deploy-gated behind `PIE_FRAME_SIZE` (default
k = 1) until Phase 2.**

## What Phase 2 needs (scoped, not started)

- Multi-step execution of the composed program with device-resolved
  sample→next-input advance: `driver/cuda/src/pipeline/dispatch.cu`
  (replace the staging readiness poll with per-step device advance).
- Per-wave publication (mirror writes + commit words + one pinned
  doorbell) with frame-boundary settlement/completion:
  `driver/cuda/src/context.cpp`.
- Per-wave graph-class selection (decode class exists; varlen class
  extends the bucketed cache): `driver/cuda/src/batch/forward_graph.hpp`.
- Staging-depth constants shared with the scheduler:
  `driver/cuda/src/runahead.hpp`.
- Declarative frame-delta KV admission (falls out of eager frame binding).

## Operational notes

- `sdk/python-server` uv cache-keys were missing runtime/driver/WIT trees
  → stale engine wheels; fixed on this branch. When in doubt:
  `uv pip install --no-cache --reinstall-package pie-server sdk/python-server`.
- The CLI `--driver dummy` bench path fails config parsing pre-existing
  (reproduced at the pristine baseline) — unrelated to this work.
- A pristine baseline worktree for A/Bs remains at `../golf-phase0`.
- Raw run logs/JSON live in the session scratchpad (`phase0/`, `oracle/`,
  `final/`), not the repo.
