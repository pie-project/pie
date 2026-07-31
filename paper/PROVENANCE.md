# Provenance

Every number in `engrain.tex` comes from a harness in this repository, run on
one NVIDIA A100 80GB PCIe. This file maps each table to what produced it, so a
number can be re-derived rather than trusted.

Numbers that were measured against code paths since deleted are **not** in the
paper. `GOAL.md` still carries some of those in its narrative sections; they
are historical record, not results.

| table | what it reports | produced by | date |
|---|---|---|---|
| 1 | allowed-set width over real steps | `results/rigorous-summary.json` | 2026-07-29 |
| 2 | claims and their falsifiers | design, not measurement | — |
| 3 | full step vs XGrammar and vs our Triton reference | `engrain_lab.rigor.latency` | 2026-07-30 |
| 4 | heterogeneous batches, and the cost of mixing | `files/probe_mixed_cost.py` | 2026-07-31 |
| 5 | speculative decoding vs `traverse_draft_tree` | `files/probe_speculative.py` | 2026-07-31 |
| 6 | cost added to an overlapped decode step | `engrain_lab.rigor.overlap` | 2026-07-31 |
| 7 | fill under CPU contention, p50 and p99 | `engrain_lab.rigor.serving` | 2026-07-31 |
| 8 | ablation of the memo and of graph capture | `files/exp_ablation.py` | 2026-07-31 |
| 9 | fused sampling across regimes | session measurements | 2026-07-30 |
| 10 | walk completion and over-acceptance, before/after | `files/exp_soundness_split.py` | 2026-07-31 |
| 11 | compile time, table memory, coverage | `engrain_lab.rigor.cost` | 2026-07-30 |

Supporting runs not given their own table:

- `files/exp_precision.py` — precision level distribution over 528 schemas
  (99.0% `Shadowed`, 0.8% `Merged`, 0.2% `Branches`) and validity per level.
- `files/exp_exactness.py` — the exactness/coverage tradeoff: forcing the most
  faithful lowering moves 181/194 schemas to `Exact`, costs 4x compile time and
  2x tables, and leaves validity unchanged. This is what ruled out the
  precision level as the cause of over-acceptance.
- `python -m engrain_lab.verify` — the four differential verifications,
  7,198 rows, zero failures.

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
