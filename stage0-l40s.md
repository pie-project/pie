# Stage 0 — re-measuring §3 on L40S (sm_89, CUDA 13.0)

Prototype: `ingim/tart` @ ea66db98, patched for CUDA 13
(`cudaGraphAddNode` gained an edge-data arg; legacy overload removed).
Env: torch 2.13.0+cu130, FlashInfer 0.6.15.post1, Qwen2.5-1.5B shapes.
Reference numbers: RTX 3090 (sm_86), `bench/RESULTS-ALL.md` @ df9e7bf.

## Status

- [x] prototype unit tests: 30/30 pass (ir 6, plan 6, adopt 6, layout 4, cond 3, lora_batch 5)
- [x] bench_axes.py (split baseline), K=2..32 — `l40s_axes.txt`
- [x] bench_baselines.py (§3.1 padding decision) — `l40s_baselines.txt`
- [x] cond_l40s (§3.4 conditional node costs) — `l40s_cond.txt`
- [x] bench_compose.py (§3.2) — `l40s_compose.txt`
- [x] bench_fusion.py (§3.3) — `l40s_fusion.txt`
- [ ] pie-side floor measurements (deferred into Stages 1+)

Raw outputs: `benches/stage0-l40s/`.

## §3.2 — the case for a planner HOLDS on L40S

`bench_compose.py`, 8 programs, depth × rank × vision:

| axes live | rows | padded | merged | PLANNED | vs padded |
|---|---|---|---|---|---|
| depth | 8 | 6.87 | 6.77 | **6.78** | 1.01× |
| × rank | 8 | 6.85 | 13.19 | **6.78** | 1.01× |
| × rank × vision | 8 | 6.81 | 13.37 | **6.88** | 0.99× |
| × rank × vision | 512 | 11.62 | 14.13 | **10.26** | 1.13× |

Same shape as the 3090 table: pure merging inherits the axis it loses (rank
doubles it), pad waste (2.68×) does not show up at thin rows, and PLANNED —
merged on depth, padded on rank — is the only column never worse than both
pure strategies. At 512 rows planned beats padded 1.13× (3090: 1.25×).

## §3.3 — fusion crossover HOLDS, split cost is larger here

`bench_fusion.py`, one MLP block, 2048 rows:

- floor (merged+fused) 0.979 ms; split keeps fusion, break materialises.
- crossover sits at **K≈4** (split wins K≤2 0.77–0.98×; break wins K≥4:
  1.27× → 5.03× at K=32). 3090 said K=4–8. Same conclusion: below the
  crossover keep fusion and split; above it break and merge.
- STRUCTURAL split cost: ~1.13–1.15× the floor on L40S (3090: ~1.06×).
- CORRECTION additive fix: 1.12× the floor (3090: 1.01×) — costlier here,
  still far below any branch; revisit before promising "~1× the floor".

## §3.1 — the padding decision HOLDS on L40S

`bench_baselines.py`, Qwen2.5-1.5B-Instruct end-to-end decode:

- uniform rank 16: planner chose **batched**; matches hand-written padded
  kernel at 1.00–1.03×, beats separate by up to 24.75× (K=32).
- mixed ranks 8–128: planner chose **padded**; matches padded point solution
  at 0.99–1.00×, beats separate by up to 22.59×.

→ "store adapters at max rank, do not build exact bucketing" survives on
sm_89. DSL stays lowering-free on the WEIGHT axis.

## §3.4 — conditional graph nodes HOLD on L40S (CUDA 13.0)

- SWITCH over N bodies, 1 taken: 8.1/7.9/8.1 µs for N=8/16/32 — **constant
  in body count** (3090: 9.6–9.8 µs).
- idle overhead per enumerated-but-absent IF subgraph: 1.2–1.4 µs
  (3090: 0.66–1.3) — slightly higher, same order.
- cost tracks TOTAL enumerated count, not live count (K=32: ~42 µs flat).
- 32 separate graph launches: 78 µs vs 43 µs conditional — conditional still
  wins, but margin narrowed (3090: 99.5 vs 28).
- overhead <10% needs a fatter body than on 3090 (~30% at 32k iters) —
  reinforces "conditionals at layer granularity only, coalesce regions".

CUDA 13 note: `cudaGraphAddNode` lost its legacy overload (edge-data param
now mandatory) — patched in tart-ref `tart/cond.py` + `research/poc/cond.cu`.

## Axis sweep vs split baseline (K=8, 512 rows)

| axis | 3090 speedup | L40S speedup | reads |
|---|---|---|---|
| depth | 6.8× | 3.92× | direction holds, magnitude ~halves |
| qlen | 7.5× | 4.13× | same |
| rank | 3.6× | 2.76× | same |
| vision (mid frac) | 2.0× | ~1.3× | same |
| sampler (8 cfg) | 1.2–5.5× | 3.99× | in range |
| MoD 12.5% kept | — | 1.68× | routed wins thin, parity at 100% |
| MoE gathered (32 exp) | — | 0.90× | **loses at high expert count on L40S** |

L40S has ~2× the compute of the 3090, so the split baseline is relatively
cheaper and every speedup compresses. The *shape* of every curve matches:
- merged cost barely moves in K (depth: 7.7ms → 8.1ms from K=2→32)
- rows/program sweep: 5.98× at 1 row/prog → 1.14× at 256 (merging matters
  exactly when spread thin — same knee as 3090)
- shared-fraction: monotone 1.00× at 0% shared → 2.13× at 100%
- homogeneous edges cost ~nothing (vision/CFG at 0%/100% ≈ 1.0×)

Flags for the plan:
1. MoE gathered-vs-dense **crosses under 1.0 at 32 experts** on L40S
   (0.90×). The plan's "merging wins on MoE" needs the vs-padded framing
   re-checked on this card before Stage 5 encodes the rule.
2. All numbers still prototype-side. The pie-side floor (Stage 1+) is what
   ultimately prices the decisions.
