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
from mlx_lm.utils import _download, load_model, load_tokenizer


def mlx_load(model_path, strict=True):
    """Mirror mlx_lm.load but expose `strict` so we can drop canonically-unused
    weights. gemma4 KV-shared layers (the last `num_kv_shared_layers`) carry
    k_proj/v_proj/k_norm in the HF checkpoint that the reference impl omits
    (it reuses the source layer's K/V) — strict=False drops exactly those, the
    same thing the canonical forward does, so the oracle stays faithful."""
    mp = _download(model_path)
    model, config = load_model(mp, strict=strict)
    model.eval()
    tok = load_tokenizer(mp, eos_token_ids=config.get("eos_token_id", None))
    return model, tok


def load_oracle(model_path):
    """Load the reference model, falling back to strict=False (with a warning)
    when the only mismatch is canonically-dropped weights (e.g. gemma4 KV
    share). Re-raises if a non-strict load still fails."""
    try:
        return mlx_load(model_path, strict=True)
    except ValueError as e:
        sys.stderr.write(
            f"[parity] strict mlx-lm load failed ({str(e).splitlines()[0]}); "
            "retrying strict=False (canonically-dropped weights)\n")
        return mlx_load(model_path, strict=False)


def reference_logits(model_path, ids):
    """Last-position logits [vocab] (float32) from mlx-lm — the oracle."""
    model, tokenizer = load_oracle(model_path)
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
    ap.add_argument("--chat", action="store_true",
                    help="wrap --prompt in the model chat template "
                         "(use for instruct/-it models: peaked distributions)")
    ap.add_argument("--cos-tol", type=float, default=0.998,
                    help="min cosine similarity of the logit vectors "
                         "(numerical-equivalence band; bf16 deep-model slack)")
    ap.add_argument("--max-abs-tol", type=float, default=None,
                    help="optional max |logit| abs diff gate (off by default; "
                         "absolute logit scale is not portable across models)")
    args = ap.parse_args()

    model_path = os.path.expanduser(args.model)
    driver_bin = os.path.expanduser(args.driver_bin)

    # Resolve the prompt -> token ids (shared by both sides for exact parity).
    if args.ids:
        ids = [int(x) for x in args.ids.split(",") if x.strip()]
        tokenizer = None
    else:
        prompt = args.prompt or "The capital of France is"
        _, tokenizer = load_oracle(model_path)
        if args.chat:
            ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True)
            print(f"prompt    = {prompt!r} (chat template)")
        else:
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
    print(f"cosine similarity         = {cos:.6f}  (gate >= {args.cos_tol})")
    print(f"KL(ref||drv)              = {kl:.6e}")
    print(f"max abs logit diff        = {max_abs:.4f}"
          f"{'' if args.max_abs_tol is None else f'  (gate <= {args.max_abs_tol})'}")
    print("─────────────────────────────────────────────────────")

    # Numerical-equivalence gate. The headline accuracy signal is the greedy
    # token match (what "accurate" means for decoding). top-k overlap + cosine
    # bound the per-logit closeness; a constant logit offset is softmax-
    # invariant (harmless), so cosine is a deliberately loose band that absorbs
    # bf16 accumulation in deep graphs. Drive instruct models with --chat for
    # peaked (non-degenerate) distributions, else greedy can flip on near-ties.
    ok = (token_match and overlap >= max(1, int(0.7 * k + 0.999))
          and cos >= args.cos_tol)
    if args.max_abs_tol is not None:
        ok = ok and max_abs <= args.max_abs_tol
    print("PASS" if ok else "FAIL", "parity gate")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
