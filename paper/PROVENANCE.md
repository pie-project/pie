# Provenance

Every number in `engrain.tex` comes from a harness in this repository, run on
one NVIDIA A100 80GB PCIe. This file maps each table to what produced it, so a
number can be re-derived rather than trusted.

Numbers that were measured against code paths since deleted are **not** in the
paper. `GOAL.md` still carries some of those in its narrative sections; they
are historical record, not results.

| table | what it reports | produced by | date |
|---|---|---|---|
| 1 | full step vs XGrammar and llguidance | `files/exp_baselines_latency.py` | 2026-07-31 |
| 2 | coverage and compile time, three engines | `files/exp_baselines_coverage.py` | 2026-07-31 |
| 3 | heterogeneous batches, and the cost of mixing | `files/probe_mixed_cost.py` | 2026-07-31 |
| 4 | ablation of the memo and of graph capture | `files/exp_ablation.py` | 2026-07-31 |
| 5 | walk completion and over-acceptance, before/after | `files/exp_soundness_split.py` | 2026-07-31 |
| 6 | resident table memory | `engrain_lab.rigor.cost` | 2026-07-30 |

Reported in prose rather than a table, for space: allowed-set width
(`results/rigorous-summary.json`), speculative decoding
(`files/probe_speculative.py`), overlap-adjusted cost
(`engrain_lab.rigor.overlap`), fill under CPU contention
(`engrain_lab.rigor.serving`), fused sampling across regimes (session
measurements, 2026-07-30).

Supporting runs not given their own table:

- `files/exp_precision.py` — precision level distribution over 528 schemas
  (99.0% `Shadowed`, 0.8% `Merged`, 0.2% `Branches`) and validity per level.
- `files/exp_exactness.py` — the exactness/coverage tradeoff: forcing the most
  faithful lowering moves 181/194 schemas to `Exact`, costs 4x compile time and
  2x tables, and leaves validity unchanged. This is what ruled out the
  precision level as the cause of over-acceptance.
- `python -m engrain_lab.verify` — the four differential verifications,
  7,198 rows, zero failures.

## Baselines

Three engines, each given its strongest interface: XGrammar with
`BatchGrammarMatcher`, the thread count swept over 1/2/4/8/16 and
**`any_order=True`**; llguidance with `LLExecutor` and its `_par` fill and
consume; outlines_core in a Python loop, because it has no batch interface.

`any_order=True` is a fairness correction we had to make. XGrammar's default
fixes object property order at the schema's declaration, so it rejects
`{"b":2,"a":1}` where engrain accepts it. Since engrain is order-free by
construction, benchmarking against the default was benchmarking a strictly
weaker configuration. The flag also compiles *faster* - 0.81 s against 4.89 s
over 60 schemas - so it is XGrammar's better setting on both axes. Its per-step
cost is within noise of the default, so the correction did not change any
conclusion; it was made anyway.

outlines_core cannot be configured out of the same restriction: its lowering
emits a literal regex with the properties in declared order, so order is a
property of the language it accepts. It could not be seeded into a live state on
any of the three model-generated documents the other three engines accept, and
building its vocabulary-indexed DFA cost 33.8 s over 20 schemas against
engrain's 1.6 s, XGrammar's 0.5 s and llguidance's 0.0 s.

## Two measurements that were wrong before they were right

Recorded because the paper's methodology claims depend on them.

**The ablation.** Setting `memo_slots` on a batch that has already been
captured changes a Python attribute and nothing the device does: a CUDA graph
bakes in its kernels' scalar arguments at record time. The first run therefore
reported that the memo was worth 1.00x while hitting 90% of steps. An ablation
of a graph-resident system has to be re-recorded, not re-configured.

**The soundness aggregate.** `rigor.soundness` counts a random walk that runs
out of byte budget as an invalid document, which mixes walk truncation into a
correctness metric. Decomposing it into completion and over-acceptance is what
made the failure attributable to a single keyword, and from there to a bug.
