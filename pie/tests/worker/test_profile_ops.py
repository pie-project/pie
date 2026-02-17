"""
Profile per-operation wall-clock time in the GPT-OSS forward pass on MPS.

Usage:
    cd /Users/ingim/Workspace/pie-mac/pie
    PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 uv run python tests/worker/test_profile_ops.py
"""

from __future__ import annotations

import sys
import time
from collections import defaultdict

import torch

sys.path.insert(0, "src")

from pie_backend.engine import Engine
from pie_backend.config import RuntimeConfig


class OpProfiler:
    """Records cumulative wall-clock time per operation name."""

    def __init__(self):
        self.timings: dict[str, float] = defaultdict(float)
        self._last: float = time.perf_counter()

    def record(self, name: str) -> None:
        now = time.perf_counter()
        self.timings[name] += now - self._last
        self._last = now

    def reset_clock(self) -> None:
        self._last = time.perf_counter()

    def reset(self) -> None:
        self.timings.clear()
        self._last = time.perf_counter()


def print_table(title: str, timings: dict[str, float], num_layers: int, divisor: int = 1):
    """Print a formatted breakdown table."""
    # Order: attention ops first, then MoE ops
    attn_order = [
        "attn_rms_norm", "attn_qkv_proj", "attn_rope",
        "attn_kv_append", "attn_compute", "attn_o_proj", "attn_residual",
    ]
    moe_order = [
        "moe_rms_norm", "moe_router", "moe_routing",
        "moe_gemm1", "moe_gemm2", "moe_scatter", "moe_residual",
    ]
    ordered_keys = attn_order + moe_order
    # Append any keys not in the predefined order
    extra = [k for k in timings if k not in ordered_keys]
    ordered_keys += extra

    total_ms = sum(timings.values()) / divisor * 1000

    print(f"\n{'=' * 70}")
    print(title)
    print(f"{'=' * 70}")
    print(f"{'Operation':<20s} {'Total ms':>10s} {'Per-layer ms':>14s} {'% of total':>12s}")
    print(f"{'─' * 20}  {'─' * 10} {'─' * 14} {'─' * 12}")

    for key in ordered_keys:
        if key not in timings:
            continue
        t = timings[key] / divisor * 1000  # to ms
        per_layer = t / num_layers
        pct = t / total_ms * 100 if total_ms > 0 else 0
        print(f"{key:<20s} {t:>10.2f} {per_layer:>14.3f} {pct:>11.1f}%")

    print(f"{'─' * 20}  {'─' * 10} {'─' * 14} {'─' * 12}")
    print(f"{'TOTAL':<20s} {total_ms:>10.2f}")


def run_step(engine, token_ids, kv_page_indices, kv_page_indptr, seq_len, is_prefill):
    """Run a single prefill or decode step through the forward pass."""
    device = engine.config.device

    embeddings = engine.forward_pass.embed_tokens(token_ids)
    position_ids = torch.arange(seq_len, dtype=torch.long, device=device)

    if is_prefill:
        qo_indptr = torch.tensor([0, seq_len], dtype=torch.int32, device=device)
        input_embeds = embeddings
    else:
        qo_indptr = torch.tensor([0, 1], dtype=torch.int32, device=device)
        input_embeds = embeddings[-1:]
        position_ids = position_ids[-1:]

    page_size = engine.config.kv_page_size
    kv_last_page_lens = torch.tensor(
        [seq_len % page_size or page_size], dtype=torch.int32, device=device
    )

    hidden = engine.forward_pass.transform(
        input_embeds=input_embeds,
        position_ids=position_ids,
        qo_indptr=qo_indptr,
        kv_cache_at_layer=engine.kv_cache_at_layer,
        kv_page_indices=kv_page_indices,
        kv_page_indptr=kv_page_indptr,
        kv_last_page_lens=kv_last_page_lens,
        custom_mask=None,
        single_token_inference_mode=not is_prefill,
        adapter_subpass=None,
    )

    logits = engine.forward_pass.lm_head(hidden)
    next_token = torch.argmax(logits[-1, :]).item()
    return next_token


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Profile GPT-OSS forward pass ops")
    parser.add_argument("--model", default="openai/gpt-oss-20b", help="Model name")
    parser.add_argument("--prefill-tokens", type=int, default=5, help="Number of prefill tokens")
    parser.add_argument("--decode-steps", type=int, default=10, help="Number of decode steps to average")
    args = parser.parse_args()

    print("Loading model...")
    config = RuntimeConfig.from_args(hf_repo=args.model)
    engine = Engine.load(config)
    device = config.device
    num_layers = engine.model_config.num_layers
    page_size = config.kv_page_size

    print(f"Model: {args.model} ({num_layers} layers)")

    profiler = OpProfiler()

    # Allocate pages for the full sequence
    max_seq = args.prefill_tokens + args.decode_steps + 10
    num_pages = (max_seq + page_size - 1) // page_size
    kv_page_indices = torch.arange(num_pages, dtype=torch.int32, device=device)
    kv_page_indptr = torch.tensor([0, num_pages], dtype=torch.int32, device=device)

    # Use token IDs 1..N for prefill
    prefill_ids = torch.arange(1, args.prefill_tokens + 1, dtype=torch.long, device=device)

    # ── Warm-up (not measured) ──
    print("Running warm-up cycle...")
    generated = prefill_ids.tolist()
    next_tok = run_step(engine, prefill_ids, kv_page_indices, kv_page_indptr,
                        len(generated), is_prefill=True)
    generated.append(next_tok)
    next_tok = run_step(engine, torch.tensor(generated, dtype=torch.long, device=device),
                        kv_page_indices, kv_page_indptr, len(generated), is_prefill=False)
    generated.append(next_tok)
    torch.mps.synchronize()

    # Reset KV cache for profiled runs
    for layer_cache in engine.kv_cache_at_layer:
        layer_cache.zero_()

    # ── Profiled prefill ──
    print(f"\nProfiling prefill ({args.prefill_tokens} tokens)...")
    engine.forward_pass.profiler = profiler
    profiler.reset()

    torch.mps.synchronize()
    profiler.reset_clock()
    t0 = time.perf_counter()

    generated = prefill_ids.tolist()
    next_tok = run_step(engine, prefill_ids, kv_page_indices, kv_page_indptr,
                        len(generated), is_prefill=True)
    generated.append(next_tok)

    torch.mps.synchronize()
    prefill_wall = time.perf_counter() - t0

    print_table(
        f"PREFILL ({args.prefill_tokens} tokens, {num_layers} layers, {prefill_wall*1000:.1f} ms wall)",
        dict(profiler.timings),
        num_layers,
    )

    # ── Profiled decode ──
    print(f"\nProfiling decode ({args.decode_steps} steps)...")
    profiler.reset()

    torch.mps.synchronize()
    profiler.reset_clock()
    t0 = time.perf_counter()

    for _ in range(args.decode_steps):
        all_ids = torch.tensor(generated, dtype=torch.long, device=device)
        next_tok = run_step(engine, all_ids, kv_page_indices, kv_page_indptr,
                            len(generated), is_prefill=False)
        generated.append(next_tok)

    torch.mps.synchronize()
    decode_wall = time.perf_counter() - t0

    print_table(
        f"DECODE (1 token, avg of {args.decode_steps} steps, {num_layers} layers each, "
        f"{decode_wall/args.decode_steps*1000:.1f} ms/step wall)",
        dict(profiler.timings),
        num_layers,
        divisor=args.decode_steps,
    )

    engine.forward_pass.profiler = None
    print(f"\nTotal generated tokens: {len(generated)}")


if __name__ == "__main__":
    main()
