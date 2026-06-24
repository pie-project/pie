# M>1 multi-batch parity gate

The throughput bench runs **M sequences concurrently**. Correctness for batch=M
reduces to one invariant:

> **Every sequence in the batch produces output bit-identical-to-gate to what it
> would produce ALONE (M=1).**

i.e. batching introduces (a) no cross-sequence contamination, (b) correct per-seq
paged-KV indexing, (c) ragged (mixed-length) prompts that don't perturb each other.

`batch_parity.py` enforces it: for each slot `i`, it walks the batched run's seq-`i`
taps through the **same per-kernel `cosine_bisect` comparison** against that prompt's
sealed **M=1 golden**, AND checks the seq-`i` logits argmax matches the golden argmax.

## Tap-emission contract (M>1 executor — pick ONE layout)

The gate supports both; tell charlie which the executor emits.

- **`rowslice`** (recommended for decode): each tap is ONE file `<layer>.<kernel>.npy`
  shaped `[M, ...]` (the natural decode-step buffer — M seqs x 1 token). seq `i` = row `i`
  of dim 0. Singleton dim-0 taps (shared consts tapped once) are broadcast.
- **`subdir`**: `<dump>/seq<i>/<layer>.<kernel>.npy` — one standard single-seq dump dir
  per sequence (each is a vanilla `cosine_bisect` dir).

Tap names + per-kernel execution order are IDENTICAL to the M=1 golden (see
`cosine_bisect.py` `QWEN36_INTRA_ORDER` / `GEMMA4_INTRA_ORDER`). Same `--skip q_norm,k_norm`
for gemma4 in-place-rope tap artifacts.

## Prompt set (`mbatch_prompts.json`, varied lengths = ragged stress)

| slot | qwen3.6 tok | gemma4 tok | intent |
|---|---|---|---|
| 0 | 8  | 9  | short (golden-style continuation) |
| 1 | 16 | 17 | medium |
| 2 | 30 | 24 | long |
| 3 | 54 | 53 | longest (near-canonical 64-prompt) |

The length spread forces ragged batching: per-seq positions, `qo_indptr` spans, and
`kv_page_indptr`/`kv_last_page_lens` must all be correct or seqs cross-contaminate.

## Goldens (NOT committed — live at `~/parity-golden/mbatch/`)

- `reference_argmax.json` — cross-engine (mlx-lm) first-decode argmax per (model, slot):
  the secondary cross-engine anchor each batched seq must hit.
- `m1/<model>/seq<i>/` — the sealed **M=1 raw-Metal per-kernel goldens** (produced by the
  decode harness with `PIE_METAL_GOLDEN_DIR` per prompt; primary gate target). [pending box+M>1 build]

## Usage

```
python batch_parity.py --batched <M>1-dump> \
  --golden 0=~/parity-golden/mbatch/m1/qwen3.6/seq0 \
           1=~/parity-golden/mbatch/m1/qwen3.6/seq1 ... \
  --layout rowslice --threshold 0.999 [--skip q_norm,k_norm]
```
Exit 0 = every seq matches its M=1 golden (cosine >= 0.999 + argmax exact); 1 = a seq diverged.
