# Vesuvius Progress Summary — Phase 0 + Phase 1

Date: 2026-07-22
Branch: `tasks/ptir-fusion/agents/golf` (agent: golf; not yet pushed)
Baseline: `73e48f05`
Commits: `2253892a`, `f164dac1`, `1afc992d`, `a6913892`
Detailed measurements and attribution: `vesuvius-phase1-report.md`

## Status at a glance

| Milestone | Status |
|---|---|
| Phase 0 — re-baseline | **Done** |
| Phase 1 — frame WIT surface (`frame-size`, `max-embed-length`, `submit(on, slots)`) | **Done** (mirrors byte-identical) |
| Phase 1 — SDK (`submit_frame`, sugar for `pass.submit`, constants) | **Done** (all existing inferlets compile unchanged) |
| Phase 1 — §5 submission validation (staged / device-advanced / latest-value, static overflow) | **Done** (enforced at k > 1) |
| Phase 1 — sealed frame scheduling (`scheduler/frame.rs`, auto-Idle, admission, makeups) | **Done** |
| Phase 1 — bench inferlet frame-aware reactive loop | **Done** |
| Correctness gates (parity oracles, fleets, k=1 no-regression) | **All green** |
| Phase 1 throughput win at k > 1 | **Not achieved on the per-wave driver path — by structure, not by bug** (see below) |
| Phase 2 — driver multi-step execution | Not started; entry points scoped |

## Phase 0 results (the numbers every later claim gates on)

RTX 4090 / Qwen3-0.6B / TP1, greedy, prefix caching off. The spec's
Section 1 profile was stale:

- 2048×32 c512: **24.7k tok/s** warm median (21.8k cold) — not 19.08k.
- vLLM reference re-pinned: **32.2k** (vLLM 0.16.0, n=3) — not 30.81k.
- Current gap to vLLM ≈ **−23%**. The spec's Phase 2 "≥ 25k" gate is
  effectively already met at baseline; the meaningful target is parity.
- Secondary shapes: 512×512 c256 ≈ 19.3k; 64×1536 ≈ 7.3k; 16×1900 ≈ 3.57k;
  512×256 c256 ≈ 27.2k.

## Phase 1 correctness (all gates green)

- 367/367 engine unit tests; dummy e2e 10/10 at both k=1 and k=4;
  worker/canary/boot-smoke/dummy-driver suites green; both inferlet
  workspaces compile unchanged.
- CUDA serial oracles: **exact token parity** vs the baseline engine for
  k=1 AND k=16 at t ∈ {1, 2, 15, 16, 17, 31, 32, 33, 256}.
- 2048-request fleets at k=16: 3+ consecutive completions, 0 failures.
- k=1 default path: byte-identical code path, measured 24.6–25.2k —
  **zero regression**.

## Phase 1 performance (honest finding)

k=16 frames are correct but **slower** than the k=1 wait-all quorum on
every shape (2048×32: ~13.5k vs ~24.7k; c0/256: ~16.6k vs ~27.4k; long
shapes −25…−37%). Instrumented attribution, in one paragraph:

The per-wave CUDA driver polls each lane's predecessor-publication
snapshot cross-stream at launch staging, so a data-dependent wave posted
early can only RETRY, never wait — feeding frame waves at depth 2 was a
fleet-wide retry storm (6k tok/s), and the landed fix (per-frame
retirement gating) serializes a frame's waves with ~0.4–1 ms of host gap
each. Independently, joins quantize to k-wave boundaries, so epoch width
settles at the arrival equilibrium (~120 lanes vs the barrier's ~450 —
wait-all gets its density by blocking the fleet on binding lanes, exactly
the behavior Vesuvius deletes). Eager and half-phase seal pacing were both
implemented and measured worse (stable fast-narrow-epoch attractor, width
12, 5.7k tok/s): epoch duration is itself the arrival-gather window, so
boundary-paced sealing is the right cadence and is what's landed.

Consequence: **k > 1 stays deploy-gated behind `PIE_FRAME_SIZE` (default
k = 1) until Phase 2.** Both structural costs are exactly what Phase 2's
device-side multi-step execution removes (no per-wave completion round
trip; O(1) host interaction per epoch makes narrow epochs cheap).

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
