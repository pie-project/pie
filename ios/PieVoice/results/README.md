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
  from timing, and they are the point: follow-up prefill stays flat
  (83 → 19 → 21 → 22) no matter how long the conversation grows.
