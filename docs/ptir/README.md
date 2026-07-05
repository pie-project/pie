<!-- PTIR implementation doc set -->
# PTIR — implementation doc set

The PTIR (pie tensor IR) execution package. Read [`overview.md`](overview.md) for the
frozen model, then [`masterplan.md`](masterplan.md) for the three-thrust decomposition
and the C1–C4 convergence contracts, then your thrust doc.

| doc | role |
|---|---|
| [`overview.md`](overview.md) | The model (§0–§7 + op-set appendix). Authoritative for semantics. |
| [`masterplan.md`](masterplan.md) | North star, 3 thrusts, contracts C1–C4, milestones M0–M3, risks. |
| [`thrust-1-memory.md`](thrust-1-memory.md) | Working set, length columns, reclamation, attention-adjacent kernels. |
| [`thrust-2-scheduler.md`](thrust-2-scheduler.md) | Non-blocking execute, run-ahead, quorum fire rule. |
| [`thrust-3-programs.md`](thrust-3-programs.md) | Trace format, channels/epoch rings, compilation tiers 0/1/2. |

Master copy: wiki page `tensor-ir-plan.md` (slug `tensor-ir-log`).
