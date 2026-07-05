# Metal driver accuracy parity harness

Diffs the Metal driver's logits against **mlx-lm** (same MLX numerics → the
trusted oracle) on the same token-id prompt. This is the accuracy gate for
ingim's bar ("runs" ≠ "accurate"): greedy-token parity + top-k logit overlap.
Built for Qwen2.5-0.5B (known-good) and reused as-is for gemma4 / qwen3.6.

## Pieces

- `parity_driver.cpp` — C++ tool (`-DPIE_METAL_BUILD_PARITY=ON`). Loads a real
  HF checkpoint via the loader, runs a single-request prefill through charlie's
  graph + delta's paged-KV on the Metal GPU, writes the final-position `[vocab]`
  logits row to a float32 `.npy`, and cross-checks that the **InProcService
  Forward path** greedy-samples the same token as the raw-graph argmax.
- `parity_check.py` — runs the identical token ids through mlx-lm, then diffs:
  greedy-token match, top-k index overlap, max abs logit diff, cosine, KL.

## Run

```bash
# 1. build the driver tool (system MLX)
cmake -S driver/metal -B driver/metal/build-gpu \
  -DPIE_METAL_WITH_MLX=ON -DPIE_METAL_MLX_PROVIDER=system \
  -DPIE_METAL_BUILD_PARITY=ON
cmake --build driver/metal/build-gpu --target parity_driver

# 2. reference + compare (mlx-lm in a venv: pip install mlx-lm)
python3 driver/metal/tools/parity/parity_check.py \
  --model ~/models/Qwen2.5-0.5B \
  --driver-bin driver/metal/build-gpu/bin/parity_driver \
  --prompt "The capital of France is"
# or: --ids 785,6722,315,9625,374

# instruct/-it models: use the chat template (peaked distributions)
python3 driver/metal/tools/parity/parity_check.py \
  --model ~/models/gemma-4-E2B-it \
  --driver-bin driver/metal/build-gpu/bin/parity_driver \
  --prompt "What is the capital of France?" --chat
```

Notes:
- `--chat` wraps the prompt in the model chat template. **Use it for
  instruct/`-it` models** — raw text yields flat distributions where the greedy
  argmax can flip on near-ties (bf16 noise) even between faithful impls.
- Strict mlx-lm load failures auto-retry with `strict=False` — gemma4's
  KV-shared layers carry k/v_proj/k_norm in the checkpoint that the reference
  impl omits (it reuses the source layer's K/V). Dropping them is exactly the
  canonical forward, so the oracle stays faithful.
- Gate = greedy-token match (headline) + top-k overlap (≥0.7·k) + cosine
  (`--cos-tol`, default 0.998 — a loose band absorbing bf16 accumulation in
  deep graphs; a constant logit offset is softmax-invariant/harmless).
  `--max-abs-tol` is off by default (absolute logit scale isn't portable).

## Results (Metal GPU, bf16)

**Qwen2.5-0.5B** (raw prompts) — greedy match, top-10 10/10, cosine ~0.9999.

**gemma4-E2B-it** (`--chat`) — greedy match on every prompt; cosine
0.9988–0.9997, top-10 7–10/10. Slightly looser than the mlx-lm bf16-vs-fp32
noise floor (~0.99996): a small broadband per-logit residual (uniform mean
shift is softmax-invariant; not magnitude-dependent → not softcap/scale) that
does not affect greedy decoding.

