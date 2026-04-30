"""Kernel-level profile of pie's decode graph.

Loads Qwen3-0.6B into the same ForwardPass used by the engine, captures
CUDA graphs, then replays a representative decode batch many times under
``torch.profiler``. Prints the top kernels by self-CUDA-time. Intended to
identify which kernels inside the 10-ms decode replay dominate.

Run with::

    PATH=/root/Workspace/pie/pie/.venv/bin:$PATH \\
        /root/Workspace/pie/pie/.venv/bin/python benches/decode_kernel_profile.py
"""
import argparse
import time

import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--batch", type=int, default=200, help="Decode batch size")
    p.add_argument("--ctx-len", type=int, default=120, help="Per-seq context length (tokens already generated)")
    p.add_argument("--reps", type=int, default=50, help="Decode iterations to profile")
    p.add_argument("--warmup", type=int, default=20)
    args = p.parse_args()

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    from pie_driver.config import NativeRuntimeConfig
    from pie_driver.engine import Engine

    cfg = NativeRuntimeConfig.from_args(
        hf_repo=args.model,
        activation_dtype="bfloat16",
        random_seed=42,
        telemetry_enabled=False,
        telemetry_endpoint=None,
        telemetry_service_name=None,
        gpu_mem_utilization=0.8,
        max_batch_size=512,
        cpu_mem_budget_in_gb=0,
        devices=["cuda:0"],
        rank=0,
        tensor_parallel_size=1,
    )
    print("Loading engine...")
    engine = Engine.load(cfg, compute_process_group=None)
    fp = engine.forward_pass
    kv = engine.kv_cache_at_layer

    page_size = int(kv[0].shape[2])
    pages_per_seq = (args.ctx_len + page_size - 1) // page_size
    last_page_len = args.ctx_len - (pages_per_seq - 1) * page_size

    bsz = args.batch
    # Build representative decode inputs.
    hidden_size = fp.model_config.dim_hidden
    hidden = torch.randn(bsz, hidden_size, device=device, dtype=torch.bfloat16)
    position_ids = torch.full((bsz,), args.ctx_len - 1, device=device, dtype=torch.int32)
    qo_indptr = torch.arange(bsz + 1, device=device, dtype=torch.int32)

    # Each seq gets `pages_per_seq` distinct pages (clamped to allocator size).
    total_pages_needed = bsz * pages_per_seq
    max_pages = kv[0].shape[0]
    if total_pages_needed >= max_pages:
        raise RuntimeError(f"Need {total_pages_needed} pages, KV has {max_pages}")
    page_ids = torch.arange(1, 1 + bsz * pages_per_seq, device=device, dtype=torch.int32)
    kv_page_indices = page_ids
    kv_page_indptr = torch.arange(0, (bsz + 1) * pages_per_seq, pages_per_seq,
                                  device=device, dtype=torch.int32)
    kv_last_page_lens = torch.full((bsz,), last_page_len, device=device, dtype=torch.int32)

    print(f"Inputs ready (batch={bsz}, ctx_len={args.ctx_len}, pages/seq={pages_per_seq})")

    def step():
        return fp.transform(
            input_embeds=hidden,
            position_ids=position_ids,
            qo_indptr=qo_indptr,
            kv_cache_at_layer=kv,
            kv_page_indices=kv_page_indices,
            kv_page_indptr=kv_page_indptr,
            kv_last_page_lens=kv_last_page_lens,
            custom_mask=None,
            single_token_inference_mode=True,
            adapter_subpass=None,
        )

    print(f"Warmup ({args.warmup} reps)...")
    for _ in range(args.warmup):
        step()
    torch.cuda.synchronize()

    # Wall-time
    print(f"Timing ({args.reps} reps)...")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(args.reps):
        step()
    torch.cuda.synchronize()
    wall = (time.perf_counter() - t0) / args.reps * 1000
    print(f"Wall time per step: {wall:.3f} ms")

    # Kernel-level profile
    print(f"Profiling ({args.reps} reps)...")
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=False,
    ) as prof:
        for _ in range(args.reps):
            step()
        torch.cuda.synchronize()

    print()
    print("=== Top kernels by self CUDA time ===")
    print(prof.key_averages().table(
        sort_by="self_cuda_time_total",
        row_limit=30,
    ))


if __name__ == "__main__":
    main()
