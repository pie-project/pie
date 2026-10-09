# Results

Machine: Apple M5, 24 GB. Engine: branch `perf/metal-auto-kv-pool` (Pie 0.5.4, Metal).
Models: Qwen3.5-0.8B and Qwen3.5-4B (4-bit weights). Task: exact-match answers on MATH-500,
the same concise prompt for every strategy, temperature 0.7. "Tokens" are decode tokens the
engine actually ran (rows x tokens), the compute each strategy spent. Full run files are
produced by `bench.py`; `results-summary.json` has every run's summary.

**Read this first.** A single 80-problem run has a standard error of about +-5 points, and a
120-problem run about +-4.5. Differences smaller than that are noise. Where I could, I report a
resampling estimate over recorded lanes as well (`simulate.py`), because it has far smaller error
bars; it is labelled "sim" below and is an estimate, not a measurement.

## 1. The systems layer: what co-design bought (measured)

| | result |
|---|---|
| Lanes in one pass vs. on separate pipelines | separate pipelines timeslice: ~170 tok/s at 1 lane, 82 at 8. One batched pass: 417 at 8, 586 at 16, 697 at 32 (0.8B). A real run holds 700-1,000 tok/s with 32 agents (0.8B) and 160-180 tok/s with 16 agents (4B) |
| Fork cost | 0.1-0.6 ms for any width; a child forked mid-generation continues exactly like its parent (`--mode kernel_test`) |
| Shared prompt prefill | 120 problems x 4 agents read 15,104 tokens instead of 75,520 (5x); 60 problems x 16 agents read 7,776 instead of 132,192 (17x). Prompts here are ~128 tokens; long shared contexts would show far more |
| Batched row efficiency | 99% of decoded row-tokens are useful |
| Memory discipline | peak system memory 80% over 37 guarded runs, zero guard pauses; the engine's own load check refused a 4B deployment that would not fit and named the knobs to lower |
| Footprint finding | engine scratch is sized up front: `max_forward_tokens`/`max_forward_requests` at 10240/512 cost ~11 GB RSS before a token is decoded; 2048/64 cost ~2 GB. And declaring a pass's writable page span exactly is what keeps KV backing proportional to use |

## 2. Strategy on Qwen3.5-0.8B (MATH-500 levels 1-3): no advantage, found and explained

The model is right ~20% of the time per sample, wrong answers are scattered, and 45% of lanes
hit the length cap without answering (about half of all tokens).

| strategy | problems | accuracy | tokens / problem |
|---|---|---|---|
| fixed 1 | 60 | 20.0% | 601 |
| fixed 4 | 60 / 120 | 48.3% / 45.0% | 2,247 / 2,307 |
| fixed 8 | 60 | 45.0% | 4,565 |
| fixed 16 | 60 | 51.7% | 9,197 |
| fixed 6 / 7 | 120 | 41.7% / 40.8% | 3,486 / 4,025 |
| council 4+2 / 4+3 / 3+3 (judges read the blackboard when solvers split) | 120 | 35.0% / 41.7% / 35.8% | 2,593 / 2,667 / 1,936 |

- Resampling 31 recorded lanes per problem puts fixed 4 at 36.9% +-4.7; the engine's 48% was a
  lucky draw. This is why single runs must not be used to rank strategies.
- Adaptive consensus lies on the same curve as fixed voting (sim: 44.9% at 3,719 tokens against
  fixed 6 at 43.1% for 3,387 and fixed 8 at 46.0% for 4,516).
- **Two ideas that sounded good and measured badly.** A learned straggler deadline cuts accuracy
  sharply (fixed 4 with a 448-token deadline: 30.0% at 1,589 tokens, where plain fixed 3 gets 33.5% at 1,689), because many
  correct answers arrive late. Referee judges did not beat plain voting at equal cost.
- Why: with low per-sample accuracy, two agreeing answers rarely happen early, so there is little
  for early stopping to save.

## 3. Strategy on Qwen3.5-4B (MATH-500 levels 4-5, 80 problems): allocation is roughly neutral

Here a single sample is right 49% of the time, so agreement carries real information.

| strategy | accuracy | tokens / problem |
|---|---|---|
| fixed 1 | 48.8% | 685 |
| fixed 2 | 55.0% | 1,354 |
| fixed 4 | 65.0% | 2,830 |
| fixed 8 | 61.3% | 5,603 |
| adaptive, judge stage, width cap 12 | 66.2% | 4,475 |
| adaptive, no judge | 63.7% | 4,218 |
| adaptive, widen one at a time, cap 8 | 60.0% | 3,411 |
| adaptive, widen one at a time, cap 6 | 61.3% | 2,938 |
| council 4+2 | 63.7% | 3,122 |

(standard error about +-5.3 points for each)

- **On the engine, no adaptive run is distinguishable from the fixed-width curve.** All points sit
  within one standard error of it (`report/report-qwen3.5-4b.html`).
- The sim (15 recorded lanes per problem, 300 resamples) estimates a small real lead for
  consensus-driven widening: fixed 1/2/3/4/5/6/8 give 49.4/56.1/59.3/61.7/62.4/63.3/64.1%, and
  "adaptive, cap 8" gives 63.9% at 3,246 tokens, matching fixed 8's accuracy at 41% fewer
  tokens (about +2 points over the fixed curve at equal cost). That lead is below what a single
  80-problem engine run can detect, so it remains an estimate.
- Where the adaptive run's tokens went (same problems, `q4b-adaptive` vs `q4b-fixed4`):
  - 52 problems ended on agreement. They were right **90.4%** of the time, at 2,284 tokens against
    fixed 4's 2,303. Agreement is a precise signal, but reaching it cost about as much as four agents.
  - 15 "cut losses" problems (no agent produced an answer) cost 6,417 tokens each for 0 correct,
    more than fixed 4's 4,092. Giving up after 6 tries was too slow; the sim prefers 3.
  - 11 problems that hit the width cap cost 11,875 tokens each for 4 correct.

## 4. What this supports, and what it does not

Supported by measurement:
- One inferlet can run dozens of agents as batched lanes over shared, refcounted KV, with exact
  copy-on-write forks, a memory-aware scheduler, exact accounting and a decision trace that says why.
- Batching lanes in one pass is the difference between flat and scaling throughput on this engine.
- Dynamic widening and early stopping do not beat fixed-width voting by a margin this hardware can
  resolve with 60-120 problems, on either model; judges and deadlines hurt or do nothing.

Not supported, and not claimed: that the strategist "understands" a problem; that adaptive
allocation wins on this benchmark; any result on open problems. **Nothing here solves, or comes
close to solving, a Millennium Prize problem.** The task here has checkable answers; open research
questions do not, and no reward signal exists for a controller to learn from.

## 5. Findings for PR #830 (`total_pages = 0`)

- With the corrected harness, `total_pages = 0` ran the 24-problem, 32-agent workload with 79%
  peak system memory against 77% for an explicit 4,096-page pool, identical accuracy, no guard
  pauses. An earlier "Insufficient Memory" crash was caused by my harness declaring the whole
  arena writable, not by auto-sizing.
- The engine's load check ("the device does not hold this deployment ... `max_forward_tokens`
  ... lower `[model] max_context`") worked as a safety net for the 4B model.
- The large baseline footprint comes from `max_forward_tokens`/`max_forward_requests`, not from KV.

## 6. Next, if you want this to pay off

1. **A bigger model on the 128 GB machine** (see README, "Scaling up"). Allocation pays when
   agreement is both likely and informative; the 0.8B model is too weak and the 4B model is near a
   plateau at ~64%.
2. **Long shared contexts.** These prompts are 128 tokens, so prefix sharing is nearly free to
   skip here. With an 8k-token shared notebook (the setting `pie-code` studies) the 5-17x prefill
   saving becomes the dominant cost.
3. **Fork inside a reasoning trace**, not only at the root: copy-on-write forks at wave boundaries
   are exact and free here, so a controller could branch where the model is uncertain. That needs a
   per-wave entropy or value signal from the epilogue, which this version does not compute.
4. **More problems per strategy** (hundreds) to resolve the 2-point effects the simulator suggests.

## 7. Profile: what stops a clear win (measured)

- **Not the engine's decode loop.** A wave is one device loop, so host overhead is small. Per-step cost on the
  0.8B model is ~7 ms at 1 row, ~13 at 8 and ~27 at 32 (short context); on 4B it is 38 ms at 1 row and 98 at 16.
  Batching already buys ~6x over running the rows one by one. What is left is weight-read bandwidth on a base M5.
  Context length costs little (32 rows: 1,390 tok/s at 48 tokens, 990 at 2,000; `probe_batching.py --pad`).
- **Statistics, not speed, is the limit.** 80-problem runs have +-5 points. A 16-problem test of "2 agents with a
  2,560-token cap" looked like a big win (11/16, the same as 4 agents at cap 1,024, with 35% fewer tokens). On 48
  problems it did not hold: 2 agents at cap 2,560 get 60.4% at 2,309 tokens, while 4 agents at cap 1,024 get 66.7%
  at 2,788. A small sample fooled me; the larger one corrected it.
- **Where tokens go on 4B.** 39% of lanes hit the 1,024 cap without an answer and use 58% of all tokens. Raising the
  cap to 2,560 leaves 23-26% unfinished (they are long explorations, not loops) and helps only ~2-4 points.
- **Paired bootstrap over problems on the recorded 4B lanes:** consensus-driven widening (cap 8, give up after 3)
  beats fixed 4 by +1.9 points (90% CI +0.6 to +3.3) for +175 tokens, and ties fixed 5 (+0.5, CI -0.4 to +1.4) with
  511 fewer tokens. A real but small edge, resampled from 15 lanes per problem, not yet confirmed on the engine.
