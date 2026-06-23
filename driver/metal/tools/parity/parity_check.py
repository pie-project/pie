#!/usr/bin/env python3
"""parity_check — accuracy harness for the Metal driver vs mlx-lm.

Runs the SAME token-id prompt through:
  * the reference: mlx-lm (identical MLX numerics, the trusted oracle), and
  * the driver: the `parity_driver` C++ tool (loader -> graph -> paged-KV ->
    InProcService Forward on the Metal GPU),

then diffs the final-position logits: greedy-token parity (the headline
accuracy gate), top-k index overlap, and logit-vector closeness (max abs diff,
cosine, KL). Reusable for any arch the driver supports (Qwen2.5/3, gemma) —
just point --model + --driver-bin at the target.

Examples:
  python3 parity_check.py \
      --model ~/models/Qwen2.5-0.5B \
      --driver-bin ../../build-gpu/parity_driver \
      --prompt "The capital of France is"

  # explicit ids (skip the tokenizer):
  python3 parity_check.py --model ~/models/Qwen2.5-0.5B \
      --driver-bin .../parity_driver --ids 9707,11,1879,0
"""
import argparse
import os
import subprocess
import sys
import tempfile

import mlx.core as mx
from mlx_lm import load as mlx_load


def reference_logits(model_path, ids):
    """Last-position logits [vocab] (float32) from mlx-lm — the oracle."""
    model, tokenizer = mlx_load(model_path)
    inputs = mx.array([ids])  # [1, L]
    out = model(inputs)        # [1, L, vocab]
    row = out[0, -1, :].astype(mx.float32)
    mx.eval(row)
    return row, tokenizer


def driver_logits(driver_bin, model_path, ids, out_npy):
    """Run the C++ parity tool; returns (logits_row[vocab] f32, greedy_tok)."""
    csv = ",".join(str(i) for i in ids)
    proc = subprocess.run(
        [driver_bin, model_path, csv, out_npy],
        capture_output=True, text=True,
    )
    sys.stderr.write(proc.stderr)
    if proc.returncode != 0:
        raise RuntimeError(f"parity_driver failed (rc={proc.returncode})")
    greedy_tok = int(proc.stdout.strip().splitlines()[-1])
    row = mx.load(out_npy).astype(mx.float32)
    mx.eval(row)
    return row, greedy_tok


def topk_ids(row, k):
    idx = mx.argsort(-row)[:k]
    mx.eval(idx)
    return [int(x) for x in idx.tolist()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--driver-bin", required=True)
    ap.add_argument("--prompt", default=None,
                    help="text prompt (tokenized via the model tokenizer)")
    ap.add_argument("--ids", default=None,
                    help="comma-separated token ids (overrides --prompt)")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--max-abs-tol", type=float, default=2.0,
                    help="max |logit| abs diff allowed (bf16 graph slack)")
    args = ap.parse_args()

    model_path = os.path.expanduser(args.model)
    driver_bin = os.path.expanduser(args.driver_bin)

    # Resolve the prompt -> token ids (shared by both sides for exact parity).
    if args.ids:
        ids = [int(x) for x in args.ids.split(",") if x.strip()]
        tokenizer = None
    else:
        prompt = args.prompt or "The capital of France is"
        _, tokenizer = mlx_load(model_path)
        ids = tokenizer.encode(prompt)
        print(f"prompt    = {prompt!r}")
    print(f"token ids = {ids}")

    ref, tokenizer = reference_logits(model_path, ids)
    with tempfile.TemporaryDirectory() as td:
        drv, drv_greedy = driver_logits(
            driver_bin, model_path, ids, os.path.join(td, "driver_logits.npy"))

    ref_argmax = int(mx.argmax(ref).item())
    drv_argmax = int(mx.argmax(drv).item())

    # ── Greedy-token parity (the headline accuracy gate) ──
    token_match = (ref_argmax == drv_argmax == drv_greedy)

    # ── Top-k index overlap ──
    k = min(args.topk, ref.shape[0])
    ref_topk, drv_topk = topk_ids(ref, k), topk_ids(drv, k)
    overlap = len(set(ref_topk) & set(drv_topk))

    # ── Logit-vector closeness ──
    diff = drv - ref
    max_abs = float(mx.max(mx.abs(diff)).item())
    cos = float((mx.sum(ref * drv) /
                 (mx.sqrt(mx.sum(ref * ref)) *
                  mx.sqrt(mx.sum(drv * drv)) + 1e-9)).item())
    # KL(ref || drv) over softmax distributions.
    lref = ref - mx.logsumexp(ref)
    ldrv = drv - mx.logsumexp(drv)
    kl = float(mx.sum(mx.exp(lref) * (lref - ldrv)).item())

    def tok_str(t):
        if tokenizer is None:
            return ""
        try:
            return f" ({tokenizer.decode([t])!r})"
        except Exception:
            return ""

    print("\n── parity ───────────────────────────────────────────")
    print(f"reference (mlx-lm) argmax = {ref_argmax}{tok_str(ref_argmax)}")
    print(f"driver raw-graph   argmax = {drv_argmax}{tok_str(drv_argmax)}")
    print(f"driver service     greedy = {drv_greedy}{tok_str(drv_greedy)}")
    print(f"GREEDY TOKEN MATCH        = {token_match}")
    print(f"top-{k} index overlap      = {overlap}/{k}")
    print(f"max abs logit diff        = {max_abs:.4f}")
    print(f"cosine similarity         = {cos:.6f}")
    print(f"KL(ref||drv)              = {kl:.6e}")
    print("─────────────────────────────────────────────────────")

    ok = token_match and overlap >= max(1, k - 1) and max_abs <= args.max_abs_tol
    print("PASS" if ok else "FAIL", "parity gate")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
