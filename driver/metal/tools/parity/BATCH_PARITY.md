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

## Tap-emission contract (M>1 executor → LOCKED with beta)

Each tap is ONE file `<layer>.<kernel>.npy` shaped **`[N, ...]`** (N = total tokens in
the batch, Pie's token-major activation buffer — zero-copy row emission), plus
**`qo_indptr.npy`** `[R+1]` emitted alongside. Sequence `i` owns activation rows
**`[qo_indptr[i], qo_indptr[i+1])`**:
- **Pure decode** (N==R, 1 tok/seq): seq `i` = row `i`.
- **Ragged prefill** (the mbatch set, lengths 8/16/30/54): seq `i` owns its qo-span; the
  gate compares the seq's **decision row** (last of the span — the token that feeds the next
  decode) to its single-position M=1 golden, or the **full span** with `--full-span` + a
  full-span golden (catches mid-prompt contamination). No padding — qo_indptr-sliced,
  matching delta's paged-KV `qo_indptr` walk 1:1.

Alternative `subdir` layout (`<dump>/seq<i>/<layer>.<kernel>.npy`, one vanilla dump dir per
seq) is also supported via `--layout subdir`.

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
- `token_ids.json` + `ids_<model>_seq<i>.csv` — pre-tokenized prompt ids (per model) for the
  golden producer below (BOS-prepended for gemma).
- `m1/<model>/seq<i>/` — the sealed **M=1 raw-Metal per-kernel goldens** (the primary gate
  target). Produced per prompt by the raw-Metal decode harness — same kernels the M>1 path
  slices, so the gate isolates *batching only*:
  ```
  PIE_DUMP_TAPS=~/parity-golden/mbatch/m1/qwen3.6/seq0 \
    build-gpu/bin/decode_run <ckpt_dir> <kernels_dir> "$(cat ~/parity-golden/mbatch/ids_qwen3.6_seq0.csv)"
  ```
  decode_run taps the final (decision) decode step — matching the gate's per-seq decision-row
  comparison. [pending box + the M>1 build, run alongside the first M>1 batched dump]

## Usage

```
python batch_parity.py --batched <M>1-dump> \
  --golden 0=~/parity-golden/mbatch/m1/qwen3.6/seq0 \
           1=~/parity-golden/mbatch/m1/qwen3.6/seq1 ... \
  --layout rowslice --threshold 0.999 [--qo-indptr <dump>/qo_indptr.npy] [--full-span] [--skip q_norm,k_norm]
```
Exit 0 = every seq matches its M=1 golden (cosine >= 0.999 + argmax exact); 1 = a seq diverged.
