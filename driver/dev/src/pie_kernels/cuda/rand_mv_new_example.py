"""End-to-end benchmark of AdapterSubpass-style LoRA inference.

Times a full 32-layer forward pass for one adapter group. Mirrors the hot path
of workload_example.py's AdapterSubpass.execute() faithfully — the only thing
that changes between variants is the rand_mv implementation and/or the specific
optimizations being applied.

Run variants via --variant:
    baseline_triton    - Triton baseline (slow reference)
    cuda_base          - rand_mv_cuda, no other changes
    cuda_no_sync       - + remove the rand_seeds.any().item() sync
    cuda_out_kwarg     - + use out= kwarg for in-place accumulation
    cuda_sectioned     - + fused Q/K/V via batched_randn_matmul_sectioned
    cuda_graph         - + CUDA Graphs capture across layers

Each variant is incremental on top of the previous. Example:

    python adapter_bench.py --variants baseline_triton cuda_base cuda_no_sync

Use --shape {decode1, decode8, prefill128, prefill512} to pick batch size.
"""
from __future__ import annotations
import argparse
import math
import sys
import time
import torch
import torch.nn as nn

# --- Import rand_mv backends ---------------------------------------------------

import rand_mv_cuda
import baseline as triton_rand_mv


# --- Adapter config mirroring CmaesAdapter ------------------------------------

class AdapterConfig:
    def __init__(self, hidden, rank, alpha, num_layers, out_features,
                 dtype=torch.float16, device="cuda", initial_sigma=0.01):
        self.hidden = hidden
        self.rank = rank
        self.alpha = alpha
        self.num_layers = num_layers
        self.out_features = out_features  # [d_q, d_k, d_v]
        self.out_features_indptr = [0]
        for d in out_features:
            self.out_features_indptr.append(self.out_features_indptr[-1] + d)
        self.dtype = dtype
        self.device = device
        self.initial_sigma = initial_sigma


def build_state(cfg: AdapterConfig):
    """Allocate realistic LoRA weight + sigma tensors (random init, correct shapes)."""
    I = cfg.hidden
    r = cfg.rank
    D = sum(cfg.out_features)
    state = {
        "Wd":    [],  # list of (I, 3r)
        "Wu":    [],  # list of (r, D)
        "Sd":    [],  # list of (I, 3r)
        "Su":    [],  # list of (r, D)
    }
    for _ in range(cfg.num_layers):
        Wd = torch.empty(I, 3 * r, device=cfg.device, dtype=cfg.dtype)
        nn.init.kaiming_uniform_(Wd, a=math.sqrt(5))
        Wu = torch.zeros(r, D, device=cfg.device, dtype=cfg.dtype)
        Sd = torch.full((I, 3 * r), cfg.initial_sigma, device=cfg.device, dtype=cfg.dtype)
        Su = torch.full((r, D),     cfg.initial_sigma, device=cfg.device, dtype=cfg.dtype)
        state["Wd"].append(Wd)
        state["Wu"].append(Wu)
        state["Sd"].append(Sd)
        state["Su"].append(Su)
    return state


def build_inputs(B: int, cfg: AdapterConfig):
    torch.manual_seed(0)
    x  = torch.randn(B, cfg.hidden, device=cfg.device, dtype=cfg.dtype)
    seeds = torch.randint(1, 1 << 30, (B,), device=cfg.device, dtype=torch.int64)
    D = sum(cfg.out_features)
    q_state = torch.zeros(B, cfg.out_features[0], device=cfg.device, dtype=cfg.dtype)
    k_state = torch.zeros(B, cfg.out_features[1], device=cfg.device, dtype=cfg.dtype)
    v_state = torch.zeros(B, cfg.out_features[2], device=cfg.device, dtype=cfg.dtype)
    return x, seeds, q_state, k_state, v_state


# --- Variants ------------------------------------------------------------------
#
# Each variant is a function  execute(cfg, state, x, seeds, q/k/v_state)  that
# runs one full 32-layer subpass.

def _make_execute_cuda(rand_mv, *, with_sync: bool, with_out_kwarg: bool,
                       with_sectioned: bool, with_fused_mean: bool = False):
    """Factory so we don't duplicate the body for each minor variation."""

    rank_lora = None  # filled per-cfg via closure capture below

    def execute(cfg: AdapterConfig, state, x, seeds, q_state, k_state, v_state):
        rank = cfg.rank
        D_q, D_k, D_v = cfg.out_features
        out_indptr = cfg.out_features_indptr
        scaling = cfg.alpha / float(rank)

        for layer_idx in range(cfg.num_layers):
            Wd = state["Wd"][layer_idx]
            Wu = state["Wu"][layer_idx]
            Sd = state["Sd"][layer_idx]
            Su = state["Su"][layer_idx]

            if with_sync:
                # Reproduces the faithful line-82 .item() sync from the original.
                _ = seeds.any().item()

            # --- DOWN projection ---
            acc_dtype = torch.float32 if (with_out_kwarg or with_sectioned) else x.dtype
            if with_fused_mean and with_sectioned:
                # Single kernel: x @ Wd + noisy_sectioned(x, Sd). No cublas DOWN call.
                qkv_down = torch.empty(x.size(0), 3 * rank, device=x.device, dtype=acc_dtype)
                rand_mv.batched_randn_matmul_sectioned(
                    x, seeds, Sd,
                    section_widths=(rank, rank, rank),
                    section_offsets=(layer_idx, layer_idx + 100, layer_idx + 200),
                    W_mean=Wd, out=qkv_down, beta=0.0,
                )
                d_q, d_k, d_v = torch.split(qkv_down, [rank, rank, rank], dim=-1)
            elif with_sectioned:
                qkv_down = (x @ Wd).to(acc_dtype) if acc_dtype != x.dtype else (x @ Wd)
                d_q, d_k, d_v = torch.split(qkv_down, [rank, rank, rank], dim=-1)
                rand_mv.batched_randn_matmul_sectioned(
                    x, seeds, Sd,
                    section_widths=(rank, rank, rank),
                    section_offsets=(layer_idx, layer_idx + 100, layer_idx + 200),
                    out=qkv_down, beta=1.0,
                )
            elif with_out_kwarg:
                qkv_down = (x @ Wd).to(acc_dtype) if acc_dtype != x.dtype else (x @ Wd)
                d_q, d_k, d_v = torch.split(qkv_down, [rank, rank, rank], dim=-1)
                rand_mv.batched_randn_matmul(x, seeds, Sd[:, 0:rank],
                    out=d_q, beta=1.0, seed_offset=layer_idx)
                rand_mv.batched_randn_matmul(x, seeds, Sd[:, rank:2*rank],
                    out=d_k, beta=1.0, seed_offset=layer_idx + 100)
                rand_mv.batched_randn_matmul(x, seeds, Sd[:, 2*rank:3*rank],
                    out=d_v, beta=1.0, seed_offset=layer_idx + 200)
            else:
                qkv_down = x @ Wd
                d_q, d_k, d_v = torch.split(qkv_down, [rank, rank, rank], dim=-1)
                # 3 kernels, 3 new tensors, 3 adds.
                q_noise = rand_mv.batched_randn_matmul(
                    x, seeds=seeds + layer_idx, S=Sd[:, 0:rank], out_dtype=x.dtype)
                k_noise = rand_mv.batched_randn_matmul(
                    x, seeds=seeds + (layer_idx + 100), S=Sd[:, rank:2*rank], out_dtype=x.dtype)
                v_noise = rand_mv.batched_randn_matmul(
                    x, seeds=seeds + (layer_idx + 200), S=Sd[:, 2*rank:3*rank], out_dtype=x.dtype)
                d_q = d_q + q_noise
                d_k = d_k + k_noise
                d_v = d_v + v_noise

            # --- UP projection ---
            Wu_q = Wu[:, out_indptr[0]:out_indptr[1]]
            Wu_k = Wu[:, out_indptr[1]:out_indptr[2]]
            Wu_v = Wu[:, out_indptr[2]:out_indptr[3]]
            Su_q = Su[:, out_indptr[0]:out_indptr[1]]
            Su_k = Su[:, out_indptr[1]:out_indptr[2]]
            Su_v = Su[:, out_indptr[2]:out_indptr[3]]

            if with_fused_mean and (with_out_kwarg or with_sectioned):
                # Fuse the cublas (d_x @ Wu_x) into the noise kernel for each Q/K/V.
                # No separate cublas call; one fused kernel per section.
                d_q_n = d_q.to(x.dtype) if d_q.dtype != x.dtype else d_q
                d_k_n = d_k.to(x.dtype) if d_k.dtype != x.dtype else d_k
                d_v_n = d_v.to(x.dtype) if d_v.dtype != x.dtype else d_v
                u_q = torch.empty(x.size(0), D_q, device=x.device, dtype=torch.float32)
                u_k = torch.empty(x.size(0), D_k, device=x.device, dtype=torch.float32)
                u_v = torch.empty(x.size(0), D_v, device=x.device, dtype=torch.float32)
                rand_mv.batched_randn_matmul(d_q_n, seeds, Su_q,
                    W_mean=Wu_q, out=u_q, beta=0.0, seed_offset=-layer_idx)
                rand_mv.batched_randn_matmul(d_k_n, seeds, Su_k,
                    W_mean=Wu_k, out=u_k, beta=0.0, seed_offset=-(layer_idx + 100))
                rand_mv.batched_randn_matmul(d_v_n, seeds, Su_v,
                    W_mean=Wu_v, out=u_v, beta=0.0, seed_offset=-(layer_idx + 200))
            elif with_out_kwarg or with_sectioned:
                d_q_n = d_q.to(x.dtype) if d_q.dtype != x.dtype else d_q
                d_k_n = d_k.to(x.dtype) if d_k.dtype != x.dtype else d_k
                d_v_n = d_v.to(x.dtype) if d_v.dtype != x.dtype else d_v
                u_q = (d_q_n @ Wu_q).to(torch.float32)
                u_k = (d_k_n @ Wu_k).to(torch.float32)
                u_v = (d_v_n @ Wu_v).to(torch.float32)
                rand_mv.batched_randn_matmul(d_q_n, seeds, Su_q,
                    out=u_q, beta=1.0, seed_offset=-layer_idx)
                rand_mv.batched_randn_matmul(d_k_n, seeds, Su_k,
                    out=u_k, beta=1.0, seed_offset=-(layer_idx + 100))
                rand_mv.batched_randn_matmul(d_v_n, seeds, Su_v,
                    out=u_v, beta=1.0, seed_offset=-(layer_idx + 200))
            else:
                u_q = d_q @ Wu_q
                u_k = d_k @ Wu_k
                u_v = d_v @ Wu_v
                q_noise_up = rand_mv.batched_randn_matmul(
                    d_q, seeds=seeds - layer_idx, S=Su_q, out_dtype=x.dtype)
                k_noise_up = rand_mv.batched_randn_matmul(
                    d_k, seeds=seeds - (layer_idx + 100), S=Su_k, out_dtype=x.dtype)
                v_noise_up = rand_mv.batched_randn_matmul(
                    d_v, seeds=seeds - (layer_idx + 200), S=Su_v, out_dtype=x.dtype)
                u_q = u_q + q_noise_up
                u_k = u_k + k_noise_up
                u_v = u_v + v_noise_up

            q_state.add_(scaling * u_q)
            k_state.add_(scaling * u_k)
            v_state.add_(scaling * u_v)
    return execute


# Variant registry
VARIANTS = {
    "baseline_triton":
        lambda: _make_execute_cuda(triton_rand_mv, with_sync=True, with_out_kwarg=False, with_sectioned=False),
    "cuda_base":
        lambda: _make_execute_cuda(rand_mv_cuda, with_sync=True, with_out_kwarg=False, with_sectioned=False),
    "cuda_no_sync":
        lambda: _make_execute_cuda(rand_mv_cuda, with_sync=False, with_out_kwarg=False, with_sectioned=False),
    "cuda_out_kwarg":
        lambda: _make_execute_cuda(rand_mv_cuda, with_sync=False, with_out_kwarg=True, with_sectioned=False),
    "cuda_sectioned":
        lambda: _make_execute_cuda(rand_mv_cuda, with_sync=False, with_out_kwarg=False, with_sectioned=True),
    "cuda_fused_mean":
        lambda: _make_execute_cuda(rand_mv_cuda, with_sync=False, with_out_kwarg=False,
                                   with_sectioned=True, with_fused_mean=True),
    # cuda_graph wraps an inner variant and captures via CUDA Graphs
}


SHAPES = {
    "decode1":    dict(B=1,   hidden=4096, rank=16, alpha=32, num_layers=32,
                       out_features=(4096, 4096, 4096)),
    "decode8":    dict(B=8,   hidden=4096, rank=16, alpha=32, num_layers=32,
                       out_features=(4096, 4096, 4096)),
    "prefill128": dict(B=128, hidden=4096, rank=16, alpha=32, num_layers=32,
                       out_features=(4096, 4096, 4096)),
    "prefill512": dict(B=512, hidden=4096, rank=16, alpha=32, num_layers=32,
                       out_features=(4096, 4096, 4096)),
}


def bench_variant(name: str, cfg: AdapterConfig, B: int, *, iters=10, warmup=3, graph=False):
    state = build_state(cfg)
    x, seeds, q_state, k_state, v_state = build_inputs(B, cfg)

    execute = VARIANTS[name]()

    if graph:
        return _bench_with_graph(execute, cfg, state, x, seeds, q_state, k_state, v_state,
                                 iters=iters, warmup=warmup)

    # Warmup
    for _ in range(warmup):
        q_state.zero_(); k_state.zero_(); v_state.zero_()
        execute(cfg, state, x, seeds, q_state, k_state, v_state)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        q_state.zero_(); k_state.zero_(); v_state.zero_()
        execute(cfg, state, x, seeds, q_state, k_state, v_state)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters  # ms per forward pass


def _bench_with_graph(execute, cfg, state, x, seeds, q_state, k_state, v_state, *, iters, warmup):
    # Dedicated capture stream
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        # Warmup on stream
        for _ in range(3):
            q_state.zero_(); k_state.zero_(); v_state.zero_()
            execute(cfg, state, x, seeds, q_state, k_state, v_state)
    torch.cuda.current_stream().wait_stream(s)

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        q_state.zero_(); k_state.zero_(); v_state.zero_()
        execute(cfg, state, x, seeds, q_state, k_state, v_state)

    # Warmup graph replay
    for _ in range(warmup):
        g.replay()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        g.replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variants", nargs="+", default=["baseline_triton", "cuda_base"])
    p.add_argument("--shape", choices=list(SHAPES.keys()), default="decode1")
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--warmup", type=int, default=3)
    args = p.parse_args()

    shape_args = SHAPES[args.shape]
    B = shape_args["B"]
    cfg = AdapterConfig(
        hidden=shape_args["hidden"], rank=shape_args["rank"],
        alpha=shape_args["alpha"], num_layers=shape_args["num_layers"],
        out_features=list(shape_args["out_features"]),
    )

    print(f"Shape = {args.shape}:  B={B}  hidden={cfg.hidden}  rank={cfg.rank}  "
          f"layers={cfg.num_layers}  out={cfg.out_features}")
    print(f"{'variant':<22} {'ms/forward':>12} {'vs baseline':>14}")
    print("-" * 50)

    base_ms = None
    for v in args.variants:
        if v == "cuda_graph":
            inner = "cuda_fused_mean" if "cuda_fused_mean" in VARIANTS else "cuda_sectioned"
            ms = bench_variant(inner, cfg, B, iters=args.iters, warmup=args.warmup, graph=True)
        else:
            ms = bench_variant(v, cfg, B, iters=args.iters, warmup=args.warmup)
        ratio = base_ms / ms if base_ms else 1.0
        if base_ms is None:
            base_ms = ms
            print(f"{v:<22} {ms:>9.3f} ms {'(baseline)':>14}")
        else:
            print(f"{v:<22} {ms:>9.3f} ms {ratio:>12.2f}x")


if __name__ == "__main__":
    main()
