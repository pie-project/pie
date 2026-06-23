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
```

## Result (Qwen2.5-0.5B, bf16, Metal GPU)

```
reference (mlx-lm) argmax = 12095 (' Paris')
driver service     greedy = 12095 (' Paris')
GREEDY TOKEN MATCH        = True
top-10 index overlap      = 10/10
max abs logit diff        = 0.32
cosine similarity         = 0.99987
```

Greedy token matches across prompts; small per-logit diffs (≲1.0) are expected
bf16 graph slack (`--max-abs-tol`, default 2.0). The headline gate is the
greedy-token match + top-k overlap.
