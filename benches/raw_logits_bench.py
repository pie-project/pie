"""Microbenchmark: cost of `Sampler::raw_logits` inside the worker.

Calls `engine.forward_pass.sample(hidden_states, sampling_metadata)` directly
with synthetic hidden states. This bypasses the runtime, ZMQ, WIT, and WASM
layers — what's left is the cost the worker pays when packaging a batch.

What we want to know:

  1. Steady-state per-batch cost for three sampling configurations:
       (a) all greedy            — sampler_type=1 (multinomial T=0)
       (b) all raw-logits        — sampler_type=7 (skips softmax for the batch)
       (c) mixed (1 raw + N-1 greedy)
              — pessimistic case: at least one raw-logits request keeps the
                softmax kernel alive AND adds the D2H+tobytes step. This is
                the case that affects siblings in the same batch.

  2. How (c) scales with batch size: does adding one raw-logits request to a
     128-request decode batch hurt the other 127?

Run::

    cd ~/Workspace/pie
    FLASHINFER_CUDA_ARCH_LIST="12.0f" \\
      uv --project pie run python benches/raw_logits_bench.py \\
         --model Qwen/Qwen3-1.7B --runs 200
"""
from __future__ import annotations

import argparse
import statistics
import time
from typing import Any

import torch

from pie_driver_dev.config import NativeRuntimeConfig
from pie_driver_dev.engine import Engine


def _make_metadata(
    batch_size: int,
    sampler_types: list[int],
    *,
    device: torch.device,
    dtype: torch.dtype,
    label_per_slot: int = 1,
) -> dict[str, Any]:
    """Hand-build the sampling_metadata dict that sample_common expects.

    Mirrors what `Batch.get_sampling_metadata` produces, but skips all the
    Rust-bytes decoding and per-request bookkeeping — we just want the
    sampler dispatch to fire.

    `label_per_slot` controls the ragged label list used by Logprob (1) and
    Logprobs (K). For other sampler types it is ignored.
    """
    import numpy as np

    assert len(sampler_types) == batch_size
    sampler_groups: dict[int, list[int]] = {}
    for i, st in enumerate(sampler_types):
        sampler_groups.setdefault(st, []).append(i)

    # Build per-sampler ragged label arrays for Logprob (8) / Logprobs (9).
    label_indptr = [0]
    label_ids: list[int] = []
    for st in sampler_types:
        if st == 8:
            label_ids.append(0)  # any token id will do for the bench
            label_indptr.append(label_indptr[-1] + 1)
        elif st == 9:
            label_ids.extend(list(range(label_per_slot)))
            label_indptr.append(label_indptr[-1] + label_per_slot)
        else:
            label_indptr.append(label_indptr[-1])

    return {
        "indices_for_logits": list(range(batch_size)),
        "temperatures": torch.full(
            (batch_size, 1), 1.0, device=device, dtype=dtype
        ),
        "sampler_groups": sampler_groups,
        # Distribution mode (sampler_type=0) reads top_k; we never set type 0
        # here, so the contents below are irrelevant — just well-shaped.
        "top_k": torch.zeros(batch_size, device=device, dtype=torch.long),
        "top_p": torch.full((batch_size,), 1.0, device=device, dtype=dtype),
        "min_p": torch.zeros(batch_size, device=device, dtype=dtype),
        "seeds": torch.zeros(batch_size, device=device, dtype=torch.long),
        "sampler_label_ids": np.asarray(label_ids, dtype=np.int32),
        "sampler_label_indptr": np.asarray(label_indptr, dtype=np.int32),
        "sampling_masks": None,
    }


def _time_one(
    fn,
    runs: int,
    *,
    use_cuda_event: bool,
) -> tuple[float, float, float]:
    """Return (avg_ms, min_ms, p95_ms) over `runs` invocations.

    Uses `torch.cuda.Event` when available (more accurate than wall clock
    when work is queued on the device); falls back to `perf_counter` on CPU.
    The returned numbers always include the implicit GPU sync that
    `sample_common` performs at the end (`.tolist()` / `.cpu()`).
    """
    samples_ms: list[float] = []
    if use_cuda_event:
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(runs)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(runs)]
        for i in range(runs):
            starts[i].record()
            fn()
            ends[i].record()
        torch.cuda.synchronize()
        samples_ms = [starts[i].elapsed_time(ends[i]) for i in range(runs)]
    else:
        for _ in range(runs):
            t0 = time.perf_counter()
            fn()
            samples_ms.append((time.perf_counter() - t0) * 1000.0)
    samples_ms.sort()
    avg = statistics.fmean(samples_ms)
    return avg, samples_ms[0], samples_ms[int(0.95 * (runs - 1))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B",
                        help="HuggingFace repo id")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--runs", type=int, default=200,
                        help="Timed iterations per cell")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--batches", default="1,4,16,64,128",
                        help="Comma-separated batch sizes")
    args = parser.parse_args()

    batch_sizes = [int(b) for b in args.batches.split(",")]

    cfg = NativeRuntimeConfig.from_args(hf_repo=args.model, device=args.device)
    engine = Engine.load(cfg)

    device = cfg.device
    dtype = cfg.activation_dtype
    use_cuda_event = device.type == "cuda"

    # Probe vocab + hidden size from the loaded model.
    fp = engine.forward_pass
    # All of pie's ForwardPass impls expose `lm_head` and an `embed_tokens`
    # path; the embedding weight has shape [vocab, hidden].
    embed_w = fp.weights.get("embed_token")
    vocab_size, hidden_size = int(embed_w.shape[0]), int(embed_w.shape[1])

    print(f"model        : {args.model}")
    print(f"device       : {device}  dtype={dtype}")
    print(f"vocab × hidden : {vocab_size} × {hidden_size}")
    print(f"runs/cell    : {args.runs}  (warmup {args.warmup})")
    print()
    payload_per_req_bytes = vocab_size * 4
    print(f"raw-logits payload per request: {payload_per_req_bytes / 1024:.1f} KiB")
    print()

    # Header
    print(f"{'batch':>6}  {'config':<22}  {'avg_ms':>9}  {'min_ms':>9}  {'p95_ms':>9}  {'vs greedy avg':>14}")
    print("-" * 84)

    for bs in batch_sizes:
        # Synthetic hidden states — random, but fixed seed for reproducibility.
        torch.manual_seed(0)
        hs = torch.randn(bs, hidden_size, device=device, dtype=dtype)

        configs = [
            ("all greedy (type 1)",       [1] * bs),
            ("all raw-logits (7)",        [7] * bs),
            ("all logprob (8)",           [8] * bs),
            ("all logprobs k=8 (9)",      [9] * bs),
            ("all entropy (10)",          [10] * bs),
        ]
        if bs > 1:
            configs.append((f"mixed (1×7 + {bs-1}×1)",     [7] + [1] * (bs - 1)))
            configs.append((f"mixed (1×8 + {bs-1}×1)",     [8] + [1] * (bs - 1)))
            configs.append((f"mixed (1×10 + {bs-1}×1)",    [10] + [1] * (bs - 1)))
            configs.append((f"mixed (1×7+1×8+1×10 + {bs-3}×1)",
                            [7, 8, 10] + [1] * (bs - 3)))

        greedy_avg = None
        for name, stypes in configs:
            meta = _make_metadata(
                bs, stypes, device=device, dtype=dtype, label_per_slot=8
            )

            def call():
                # `sample` returns a dict with .tolist() data, which forces
                # an implicit GPU sync — so the timing reflects end-to-end
                # cost (kernel + D2H + Python packaging).
                _ = fp.sample(hs, meta)

            # Warmup
            for _ in range(args.warmup):
                call()
            if use_cuda_event:
                torch.cuda.synchronize()

            avg, mn, p95 = _time_one(call, args.runs, use_cuda_event=use_cuda_event)
            if greedy_avg is None:
                greedy_avg = avg
                delta = "—"
            else:
                d = avg - greedy_avg
                delta = f"{d:+.3f} ms ({d / greedy_avg * 100:+.1f}%)"
            print(
                f"{bs:>6}  {name:<22}  {avg:>9.3f}  {mn:>9.3f}  {p95:>9.3f}  {delta:>14}"
            )
        print()


if __name__ == "__main__":
    main()
