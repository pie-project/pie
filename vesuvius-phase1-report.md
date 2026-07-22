# Vesuvius Phase 0/1 Report — Frame-Based Submission

Date: 2026-07-22
Branch: `tasks/ptir-fusion/agents/golf`
Baseline: `73e48f05` (Phase 0 measured there)
Spec: the Project Vesuvius frame-based submission design (2026-07-22)

## Phase 0 — re-baseline at the implementation baseline

Measured on RTX 4090, Qwen3-0.6B, CUDA TP1, chat-templated ~34-token
prompts, greedy, prefix caching off. The spec's Section 1 numbers predate
recent scheduler work and moved substantially:

| Shape | Spec §1 figure | Phase 0 re-measure | vLLM reference |
|---|---:|---:|---:|
| 2048×32, active 512 | 19.08k tok/s | 21.8k cold, 24.6–25.2k warm (median ~24.7k) | 32.2k (vLLM 0.16.0; spec said 30.81k) |
| 512×512, c256 | 18.51k | 19.13–19.48k | — |
| 64×1536 | 7.34k | 7.20–7.33k | — |
| 16×1900 | 3.72k | 3.57–3.58k | — |
| 512 req × 256 tok, c256 | — | 27.17–27.30k | — |

Consequences for the plan's gates:
- The Phase 2 "≥ 25k tok/s" gate is effectively met at baseline already;
  the meaningful target is vLLM parity at ~32.2k (current gap ≈ −23%).
- Every Phase 1 comparison below is against these re-measured numbers,
  not the spec's Section 1 profile.

Raw logs and JSON for every run are in the session scratchpad
(`phase0/`), not the repo.

## Phase 1 — runtime-only frames (implemented)

Landed in commits `2253892a` and `f164dac1`:

- **WIT**: `model.frame-size()` (k; `PIE_FRAME_SIZE`, default 1, clamp
  1..=64) and `model.max-embed-length()` (C; the driver's structural token
  cap). `forward-pass.submit` replaced by the interface-level
  `submit(on, slots)` frame call. Mirrors synced byte-identically.
- **SDK**: `pass.submit(on)` remains as single-slot-frame sugar (all
  existing inferlets compile unchanged); `submit_frame(on, slots)`,
  `frame_size()`, `max_embed_length()` are the frame-aware surface.
- **Host validation (§5, k > 1 only)**: staged / device-advanced /
  latest-value classification per host-writer channel from
  `channel_accesses` + declared roles; seeded-writer supply counted by
  ring sequence; static reader-capacity overflow prevention (2k−1
  guidance in the error). k = 1 is byte-identical to the legacy path.
- **Scheduler**: `FramePolicy` (sealed membership epochs) replaces the
  wait-all quorum at k > 1. Seal-at-last-wave-posted keeps the pipe full
  across boundaries; RETRY fires become makeups that gate wave advance;
  deterministic first-fit admission against per-wave row/token budgets;
  auto-Idle for lanes that miss a boundary. Controls, untracked riders,
  retries, and the k = 1 path reuse the existing machinery.
- **Bench inferlet**: frame-aware reactive loop (first frame = prefill +
  k−1 decode slots; then all-decode frames submitted after each
  predecessor's first token; out-ring capacity 2k−1).

### Correctness gates

| Gate | Result |
|---|---|
| Engine unit tests | 366/366 |
| Dummy e2e, k=1 | 10/10 |
| Dummy e2e, k=4 | 10/10 |
| Worker / canary / boot-smoke / dummy-driver suites | green |
| CUDA serial oracle k=1 vs baseline engine (t = 1..33, 256) | exact token parity |
| CUDA serial oracle k=16 vs baseline (t = 1..33, 256) | (see below) |
| Three consecutive 2048-request fleets at k=16 | (see below) |

### Performance (Phase 1 A/B)

(filled below)

## Notes and deviations

- The staged-class rule rejects the `Channel::writer` late-put run-ahead
  pattern at k > 1 by design (§9: per-step host inputs are staged before
  submit). `direct-channel-e2e` now pre-stages on frame deployments and
  keeps late-put coverage at k = 1.
- Physical-KV admission remains at the existing layers (contention
  orchestrator at preparation + per-launch driver prepare lease); seal
  admission is row/token arithmetic. Declarative frame-delta admission is
  Phase 2 work, alongside per-wave publication and frame-boundary
  settlement in the driver.
- The CLI `--driver dummy` bench path fails config parsing on this branch
  independently of these changes (pre-existing; reproduced at the pristine
  baseline).
- `sdk/python-server` uv cache-keys were missing the runtime/driver/WIT
  trees, so engine edits could serve a stale cached wheel; fixed here.
