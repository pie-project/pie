"""Direct A/B harness for the custom-mask paths in pie_driver_vllm.

Runs the same synthetic forward pass — identical token IDs, positions, KV
pages, and BRLE attention mask — through both drivers:

  * pie_driver       (FlashInfer reference, gold standard)
  * pie_driver_vllm + _FlashInferStrategy (the production fast path)

For each, we print:
  * the sampled (argmax) token at the last position
  * end-to-end fire_batch wall time (warmup + average)

The test mask is "attention sink (4) + sliding window (16)" — a non-trivial
masking pattern that materially changes the logit at the last position once
seq_len > 20.

Usage::

    cd pie/
    uv run --extra vllm python ../benches/mask_compare.py
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Synthetic batch builder
# ---------------------------------------------------------------------------


def _build_brle(seq_len: int, sink: int, window: int) -> np.ndarray:
    """Build a per-token BRLE mask: [sink attend][middle masked][window attend]."""
    if seq_len <= sink + window:
        # Whole sequence fits — attend to all.
        return np.array([0, seq_len], dtype=np.uint32)
    middle = seq_len - sink - window
    return np.array([0, sink, middle, window], dtype=np.uint32)


def make_inputs(
    *,
    prompt_len: int,
    sink: int,
    window: int,
    page_size: int,
    device: torch.device,
    seed: int = 1234,
) -> dict[str, Any]:
    """Construct a single-request batch dict matching `pie_driver.batching.Batch.get_model_inputs()`."""
    rng = np.random.default_rng(seed)
    token_ids = rng.integers(low=10, high=10000, size=prompt_len, dtype=np.int64)

    # Single request, prefill: positions 0..prompt_len-1.
    positions = np.arange(prompt_len, dtype=np.int32)
    qo_indptr = np.array([0, prompt_len], dtype=np.int32)

    # KV pages: ceil(prompt_len / page_size)
    n_pages = (prompt_len + page_size - 1) // page_size
    kv_page_indices = np.arange(n_pages, dtype=np.int32)
    kv_page_indptr = np.array([0, n_pages], dtype=np.int32)
    last_page_len = prompt_len - (n_pages - 1) * page_size
    kv_last_page_lens = np.array([last_page_len], dtype=np.int32)

    # Per-token BRLE: token k can attend to positions [0, k+1) under sink+window.
    flattened: list[int] = []
    indptr = [0]
    for k in range(prompt_len):
        valid_len = k + 1
        runs = _build_brle(valid_len, sink, window)
        flattened.extend(int(x) for x in runs)
        indptr.append(len(flattened))
    flattened_u32 = np.array(flattened, dtype=np.uint32)
    mask_indptr_u32 = np.array(indptr, dtype=np.uint32)

    # Decode BRLE to flat bool mask using pie's decoder for parity.
    from pie_driver.batching import decode_brle_batch

    valid_lens = positions + 1
    token_acc = np.zeros(prompt_len + 1, dtype=np.int32)
    token_acc[1:] = np.cumsum(valid_lens, dtype=np.int32)
    flat_bool = decode_brle_batch(
        flattened_u32.astype(np.int32), mask_indptr_u32.astype(np.int32),
        positions.astype(np.int32), token_acc,
    )

    # Sample only the last token; argmax sampler.
    sampling_indices = np.array([prompt_len - 1], dtype=np.int32)
    sampling_indptr = np.array([0, 1], dtype=np.int32)
    request_num_samplers = np.array([1], dtype=np.uint32)
    sampler_types = np.array([1], dtype=np.uint32)  # ARGMAX (uniform with temp=0)
    sampler_temperatures = np.array([0.0], dtype=np.float32)
    sampler_top_k = np.array([0], dtype=np.uint32)
    sampler_top_p = np.array([1.0], dtype=np.float32)
    sampler_min_p = np.array([0.0], dtype=np.float32)
    sampler_seeds = np.array([0], dtype=np.uint32)

    return {
        # Mirrors what RPC delivers; pie_driver.batching.Batch.__init__ consumes this.
        "_rpc_args": {
            "token_ids": token_ids.astype(np.uint32).tobytes(),
            "position_ids": positions.astype(np.uint32).tobytes(),
            "kv_page_indices": kv_page_indices.astype(np.uint32).tobytes(),
            "kv_page_indptr": kv_page_indptr.astype(np.uint32).tobytes(),
            "kv_last_page_lens": kv_last_page_lens.astype(np.uint32).tobytes(),
            "qo_indptr": qo_indptr.astype(np.uint32).tobytes(),
            "single_token_mode": False,
            "flattened_masks": flattened_u32.astype(np.uint32).tobytes(),
            "mask_indptr": mask_indptr_u32.astype(np.uint32).tobytes(),
            "logit_masks": np.array([], dtype=np.uint32).tobytes(),
            "logit_mask_indptr": np.array([0], dtype=np.uint32).tobytes(),
            "sampling_indices": sampling_indices.astype(np.uint32).tobytes(),
            "sampling_indptr": sampling_indptr.astype(np.uint32).tobytes(),
            "request_num_samplers": request_num_samplers.tobytes(),
            "sampler_types": sampler_types.tobytes(),
            "sampler_temperatures": sampler_temperatures.tobytes(),
            "sampler_top_k": sampler_top_k.tobytes(),
            "sampler_top_p": sampler_top_p.tobytes(),
            "sampler_min_p": sampler_min_p.tobytes(),
            "sampler_seeds": sampler_seeds.tobytes(),
            "adapter_indices": [None],
            "adapter_seeds": [None],
            "spec_token_ids": np.array([], dtype=np.uint32).tobytes(),
            "spec_position_ids": np.array([], dtype=np.uint32).tobytes(),
            "spec_indptr": np.array([0], dtype=np.uint32).tobytes(),
            "output_spec_flags": [False],
        },
        "expected_total_bits": int(token_acc[-1]),
        "flat_bool": flat_bool,
        "prompt_len": prompt_len,
    }


# ---------------------------------------------------------------------------
# Engine drivers
# ---------------------------------------------------------------------------


def load_native_engine(model_repo: str, device: str = "cuda:0"):
    from pie_driver.config import NativeRuntimeConfig
    from pie_driver.engine import Engine

    cfg = NativeRuntimeConfig(
        hf_repo=model_repo,
        cache_dir=os.path.expanduser("~/.cache/huggingface"),
        adapter_path="",
        devices=[torch.device(device)],
        rank=0,
        tensor_parallel_size=1,
        activation_dtype=torch.bfloat16,
        random_seed=0,
        telemetry_enabled=False,
        telemetry_endpoint="",
        telemetry_service_name="bench",
        swap_budget_bytes=0,
        max_num_kv_pages=2048,
        gpu_mem_utilization=0.5,
        kv_page_size=16,
        max_batch_tokens=4096,
        max_batch_size=8,
        max_dist_size=32,
        max_num_embeds=8,
        max_num_adapters=0,
        max_adapter_rank=0,
        weight_dtype="auto",
        dummy_mode=False,
    )
    return Engine.load(cfg)


def load_vllm_engine(model_repo: str, device: str = "cuda:0", attention_backend: str = "FLASHINFER"):
    from pie_driver.config import RuntimeConfig
    from pie_driver_vllm.config import VllmDriverConfig
    from pie_driver_vllm.engine import VllmEngine

    cfg = RuntimeConfig(
        hf_repo=model_repo,
        cache_dir=os.path.expanduser("~/.cache/huggingface"),
        adapter_path="",
        devices=[torch.device(device)],
        rank=0,
        tensor_parallel_size=1,
        activation_dtype=torch.bfloat16,
        random_seed=0,
        telemetry_enabled=False,
        telemetry_endpoint="",
        telemetry_service_name="bench",
        swap_budget_bytes=0,
        max_num_kv_pages=2048,
    )
    drv = VllmDriverConfig(
        attention_backend=attention_backend,
        gpu_memory_utilization=0.5,
        max_num_seqs=8,
        max_num_batched_tokens=4096,
        block_size=16,
        enforce_eager=True,
    )
    return VllmEngine.load(cfg, drv)


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


def run_engine(engine, inputs_pkg: dict, *, with_mask: bool, kv_page_size: int) -> dict:
    """Run one fire_batch on the given engine. Returns the sampled token + last-pos hidden."""
    from pie_driver.batching import Batch

    rpc = dict(inputs_pkg["_rpc_args"])
    if not with_mask:
        # Strip the mask data so we exercise the no-mask path on the same batch.
        rpc["flattened_masks"] = np.array([], dtype=np.uint32).tobytes()
        rpc["mask_indptr"] = np.array(
            [0] * (inputs_pkg["prompt_len"] + 1), dtype=np.uint32
        ).tobytes()

    vocab_size = (
        getattr(engine.model_config, "vocab_size", None)
        or getattr(engine.model_config, "num_vocabs", None)
    )
    batch = Batch(
        rpc, kv_page_size=kv_page_size, max_dist_size=32, adapters={},
        vocab_size=vocab_size,
    )
    device = torch.device(engine.config.devices[0])
    inputs = batch.get_model_inputs(device)
    sampling_metadata = batch.get_sampling_metadata(device, engine.config.activation_dtype)

    # Time
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = engine.fire_batch(inputs, sampling_metadata)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return {"out": out, "elapsed": elapsed}


def benchmark(engine, inputs_pkg, kv_page_size, *, with_mask: bool, n_warmup: int = 2, n_runs: int = 5):
    # Warmup
    for _ in range(n_warmup):
        run_engine(engine, inputs_pkg, with_mask=with_mask, kv_page_size=kv_page_size)
    # Measure
    times = []
    last_out = None
    for _ in range(n_runs):
        r = run_engine(engine, inputs_pkg, with_mask=with_mask, kv_page_size=kv_page_size)
        times.append(r["elapsed"])
        last_out = r["out"]
    return {"out": last_out, "mean_ms": 1000 * float(np.mean(times)), "std_ms": 1000 * float(np.std(times))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--prompt-len", type=int, default=128)
    parser.add_argument("--sink", type=int, default=4)
    parser.add_argument("--window", type=int, default=16)
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--n-runs", type=int, default=5)
    parser.add_argument("--n-warmup", type=int, default=2)
    parser.add_argument("--vllm-only", action="store_true")
    parser.add_argument("--native-only", action="store_true")
    parser.add_argument(
        "--paths",
        default="native,vllm-flashinfer",
        help="Comma-separated subset of {native,vllm-flashinfer}",
    )
    args = parser.parse_args()

    paths = [p.strip() for p in args.paths.split(",")]
    if args.vllm_only:
        paths = [p for p in paths if p.startswith("vllm")]
    if args.native_only:
        paths = [p for p in paths if p == "native"]

    inputs_pkg = make_inputs(
        prompt_len=args.prompt_len, sink=args.sink, window=args.window,
        page_size=args.page_size, device=torch.device(args.device),
    )
    print(f"Test: prompt_len={args.prompt_len} sink={args.sink} window={args.window}")
    print(f"      total_mask_bits={inputs_pkg['expected_total_bits']}\n")

    results = {}
    for path in paths:
        print(f"=== {path} ===")
        # Important: run with-mask BEFORE no-mask. The KV cache persists
        # across calls; running no-mask first would seed it with K/V from
        # the same prompt, masking any latent bug where the masked path
        # forgets to write KV.
        if path == "native":
            engine = load_native_engine(args.model, device=args.device)
            r_yes = benchmark(engine, inputs_pkg, args.page_size, with_mask=True,
                              n_warmup=args.n_warmup, n_runs=args.n_runs)
            r_no = benchmark(engine, inputs_pkg, args.page_size, with_mask=False,
                             n_warmup=args.n_warmup, n_runs=args.n_runs)
        elif path == "vllm-flashinfer":
            engine = load_vllm_engine(args.model, device=args.device)
            r_yes = benchmark(engine, inputs_pkg, args.page_size, with_mask=True,
                              n_warmup=args.n_warmup, n_runs=args.n_runs)
            r_no = benchmark(engine, inputs_pkg, args.page_size, with_mask=False,
                             n_warmup=args.n_warmup, n_runs=args.n_runs)
        else:
            print(f"  unknown path {path!r}, skipping")
            continue

        token_no = _extract_argmax_token(r_no["out"])
        token_yes = _extract_argmax_token(r_yes["out"])
        print(f"  no-mask  : token={token_no:>6}  mean={r_no['mean_ms']:.2f} ± {r_no['std_ms']:.2f} ms")
        print(f"  with-mask: token={token_yes:>6}  mean={r_yes['mean_ms']:.2f} ± {r_yes['std_ms']:.2f} ms")
        results[path] = {"no": (token_no, r_no["mean_ms"]),
                         "yes": (token_yes, r_yes["mean_ms"])}
        print()

        # Free the engine before loading the next one (model + KV are large).
        del engine
        torch.cuda.empty_cache()

    # Cross-backend correctness summary: the no-mask token should agree across
    # all backends (modulo float nondeterminism); the with-mask token agreement
    # is the actual mask plumbing test.
    print("=== summary ===")
    if "native" in results:
        ref_no = results["native"]["no"][0]
        ref_yes = results["native"]["yes"][0]
        for path, r in results.items():
            no_match = "✓" if r["no"][0] == ref_no else "✗"
            yes_match = "✓" if r["yes"][0] == ref_yes else "✗"
            print(f"  {path:18s} no-mask {no_match} ({r['no'][0]} vs {ref_no})  "
                  f"with-mask {yes_match} ({r['yes'][0]} vs {ref_yes})")


def _extract_argmax_token(out) -> int:
    """Pull the first sampled token from sample_common's output structure.

    `sample_common` returns a dict shaped like
    ``{"tokens": [int, ...], "dists": [...], "nan_indices": [...]}``.
    """
    if isinstance(out, dict):
        toks = out.get("tokens") or out.get("sampled_tokens")
        if toks:
            return int(toks[0])
    if isinstance(out, (list, tuple)) and out:
        first = out[0]
        if isinstance(first, dict):
            toks = first.get("tokens")
            if toks:
                return int(toks[0])
        if isinstance(first, (list, tuple)) and first:
            return int(first[0])
    return -1


if __name__ == "__main__":
    main()
