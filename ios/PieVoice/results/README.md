# Benchmark results

`bench-ladder.sh sim|dev <udid>` writes one `.jsonl` per run here, and
`bench-report.py` formats it.

Every number is produced by the app measuring itself (`BenchmarkRunner`),
not read off a screenshot. Per turn it records time-to-first-token, decode
rate over the generation window (excluding TTFT, so prompt length does not
distort it), prefill and reused token counts from the engine's own
accounting, and physical memory footprint — the figure jetsam judges.

## Reading these honestly

- **Simulator runs execute on the Mac's CPU.** Decode rates are not
  predictive of iPhone hardware in either direction. Memory footprint
  translates far better than speed does.
- **Decode rate varies run to run** with host load; we have seen the same
  build and model report anywhere from 7 to 55 tok/s across sessions.
  Within a single ladder run the models are measured back to back, so the
  *ordering* is much more trustworthy than any absolute figure.
- **Prefill and reuse counts are exact** — they come from the engine, not
  from timing, and they are the point: follow-up prefill does not grow
  with the conversation. With Pie 0.4's snapshots it stayed flat at the
  new question alone (83 → 19 → 21 → 22); with Pie 0.5's prefix cache the
  reusable prefix ends at the previous user message, so each turn
  prefills the previous reply plus the new question (54-101 tokens in
  `dev-20261008-*`) while `reused` climbs by a turn's worth each time
  (64 → 128 → 192 → 256).

## Runs

- `dev-20261008-iphone16pro-qwen3.5-0.8b-run{1,2}.jsonl` — Pie 0.5, Metal
  engine, Qwen3.5-0.8B (`qwen35-d0.8b-u4g64-kv-bf16`), iPhone 16 Pro, iOS
  26.1. Run 1 is the first launch after install (turn 1 pays for kernel
  warm-up), run 2 a warm relaunch.
- `dev-20260921-iphone16pro-qwen3-0.6b.jsonl` — Pie 0.4, ggml CPU driver,
  Qwen3-0.6B Q4_K_M, same phone.
- `sim-20260825-205655.jsonl` — Pie 0.4 Simulator ladder (Mac CPU).
