"""Kernel-level profile of pie's sample stage on a representative batch.

Times sample_common with a single Sampler-3 group (top_p=0.95, temp=0.6),
which is what the text-completion inferlet uses. Reports the top kernels
inside lm_head + softmax + sampler.
"""
import argparse
import time

import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--batch", type=int, default=200)
    p.add_argument("--reps", type=int, default=100)
    p.add_argument("--warmup", type=int, default=20)
    args = p.parse_args()

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    from pie_driver_dev.config import NativeRuntimeConfig
    from pie_driver_dev.engine import Engine
    from pie_driver_dev.model.common import sample_common, NEEDS_PROBS_TYPES

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

    # ---- Build inputs ------------------------------------------------------
    bsz = args.batch
    hidden_size = fp.model_config.dim_hidden
    hidden_states = torch.randn(bsz, hidden_size, device=device, dtype=torch.bfloat16)

    # Sampling metadata that mimics a homogeneous Sampler-3 (top_p) batch
    indices = list(range(bsz))
    sampling_metadata = {
        "indices_for_logits": indices,
        "sampling_masks": None,
        "sampler_groups": {3: indices},  # Sampler 3 = top-p
        "temperatures": torch.full((bsz,), 0.6, device=device, dtype=torch.float32),
        "top_k": torch.full((bsz,), -1, device=device, dtype=torch.int32),
        "top_p": torch.full((bsz,), 0.95, device=device, dtype=torch.float32),
        "min_p": torch.full((bsz,), 0.0, device=device, dtype=torch.float32),
        "seeds": None,
        "sampler_label_ids": None,
        "sampler_label_indptr": None,
    }

    def step():
        return sample_common(
            hidden_states=hidden_states,
            sampling_metadata=sampling_metadata,
            lm_head_fn=lambda x: fp.lm_head(x),
            device=device,
            dtype=torch.bfloat16,
        )

    print(f"Warmup ({args.warmup} reps)...")
    for _ in range(args.warmup):
        step()
    torch.cuda.synchronize()

    print(f"Timing ({args.reps} reps)...")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(args.reps):
        step()
    torch.cuda.synchronize()
    wall = (time.perf_counter() - t0) / args.reps * 1000
    print(f"Wall time per sample: {wall:.3f} ms")

    print(f"Profiling ({args.reps} reps)...")
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
    ) as prof:
        for _ in range(args.reps):
            step()
        torch.cuda.synchronize()

    print()
    print("=== Top kernels by self CUDA time ===")
    print(prof.key_averages().table(
        sort_by="self_cuda_time_total",
        row_limit=20,
    ))


if __name__ == "__main__":
    main()
