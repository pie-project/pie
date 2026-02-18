"""
Microbenchmark: MoE GEMM kernel time vs achievable bandwidth.

Measures SIMD K-split decode kernels (production decode path):
1. Isolated GEMM1/GEMM2 dispatch timing
2. Correctness check vs PyTorch reference
3. Back-to-back 24-layer pipelined timing
4. bf16 linear comparison (achievable bandwidth baseline)

Usage:
    cd /Users/ingim/Workspace/pie-mac/pie
    PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 uv run python \
        tests/flashinfer_metal/bench_gemm_micro.py
"""

import sys
import time

sys.path.insert(0, "src")

import torch
import torch.nn.functional as fun

# Model dimensions (GPT-OSS-20B)
H = 2880
H_PAD = 3072  # padded hidden for MoE
I = 2880       # intermediate_size
E = 32         # num_experts
K = 4          # top_k
NUM_LAYERS = 24
DEVICE = "mps"
DTYPE = torch.bfloat16


def bench(fn, warmup=10, iters=50, label=""):
    for _ in range(warmup):
        fn()
    torch.mps.synchronize()

    times = []
    for _ in range(iters):
        torch.mps.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.mps.synchronize()
        times.append((time.perf_counter() - t0) * 1000)

    times.sort()
    n = len(times)
    med = times[n // 2]
    lo = times[n // 10]
    hi = times[9 * n // 10]
    if label:
        print(f"  {label:40s}  med={med:.3f}  p10={lo:.3f}  p90={hi:.3f} ms")
    return (med, lo, hi)


def main():
    from flashinfer_metal._compiler import MetalCompiler
    compiler = MetalCompiler()

    torch.manual_seed(42)

    print("=" * 72)
    print("MoE GEMM Microbenchmark (decode kernels)")
    print(f"  H={H}, H_PAD={H_PAD}, I={I}, E={E}, K={K}")
    print("=" * 72)

    # Allocate weight tensors matching real model shapes (single layer)
    gemm1_blocks = torch.randint(0, 256, (E, 2 * I, H_PAD // 2),
                                  dtype=torch.uint8, device=DEVICE)
    gemm1_scales = torch.full((E, 2 * I, H_PAD // 32), 127,
                               dtype=torch.uint8, device=DEVICE)
    gemm2_blocks = torch.randint(0, 256, (E, H_PAD, I // 2),
                                  dtype=torch.uint8, device=DEVICE)
    gemm2_scales = torch.full((E, H_PAD, I // 32), 127,
                               dtype=torch.uint8, device=DEVICE)

    # Per-layer weights (24 layers, each unique)
    gemm1_blocks_list = [
        torch.randint(0, 256, (E, 2 * I, H_PAD // 2), dtype=torch.uint8, device=DEVICE)
        for _ in range(NUM_LAYERS)
    ]
    gemm1_scales_list = [
        torch.full((E, 2 * I, H_PAD // 32), 127, dtype=torch.uint8, device=DEVICE)
        for _ in range(NUM_LAYERS)
    ]
    gemm2_blocks_list = [
        torch.randint(0, 256, (E, H_PAD, I // 2), dtype=torch.uint8, device=DEVICE)
        for _ in range(NUM_LAYERS)
    ]
    gemm2_scales_list = [
        torch.full((E, H_PAD, I // 32), 127, dtype=torch.uint8, device=DEVICE)
        for _ in range(NUM_LAYERS)
    ]

    hidden = torch.randn(1, H_PAD, dtype=DTYPE, device=DEVICE)
    expert_ids = torch.tensor([3, 7, 15, 28], dtype=torch.int32, device=DEVICE)
    fused_scales = torch.tensor([0.3, 0.25, 0.25, 0.2],
                                 dtype=torch.float32, device=DEVICE)

    # bf16 attention weights for bandwidth comparison
    qkv_w = torch.randn(5120, H, dtype=DTYPE, device=DEVICE)
    qkv_b = torch.randn(5120, dtype=DTYPE, device=DEVICE)

    # Weight data sizes
    g1_bytes = K * 2 * I * (H_PAD // 2 + H_PAD // 32)
    g2_bytes = K * H_PAD * (I // 2 + I // 32)
    total_bytes = g1_bytes + g2_bytes

    # --- FP4 dequant helpers (used by correctness checks) ---
    def dequant_expert_weights(blocks, scales, expert_id):
        """Dequant one expert's FP4 weights to bf16."""
        e_blocks = blocks[expert_id]
        e_scales = scales[expert_id]
        rows, half_cols = e_blocks.shape
        cols = half_cols * 2

        lo = (e_blocks & 0x0F).to(torch.int32)
        hi = (e_blocks >> 4).to(torch.int32)
        unpacked = torch.stack([lo, hi], dim=-1).reshape(rows, cols)

        fp4_lut = torch.tensor([
            0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
            -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0
        ], dtype=torch.float32, device=DEVICE)
        dequant = fp4_lut[unpacked]

        scale_vals = torch.pow(2.0, e_scales.to(torch.float32) - 127.0)
        scale_vals = scale_vals.unsqueeze(-1).expand(-1, -1, 32).reshape(rows, cols)
        return (dequant * scale_vals).to(torch.bfloat16)

    # ===================================================================
    # Part 1: Isolated GEMM timing
    # ===================================================================
    print("\n--- Part 1: Isolated GEMM timing ---\n")

    # Decode raw GEMM1 (no SwiGLU)
    def do_decode_gemm1():
        return compiler.run_moe_decode_gemm1(
            input=hidden, all_w_blocks=gemm1_blocks,
            all_w_scales=gemm1_scales,
            intermediate_size=I, expert_ids=expert_ids,
        )

    activated_decode = do_decode_gemm1()

    # Decode fused GEMM2
    def do_decode_gemm2():
        return compiler.run_moe_decode_gemm2_fused(
            input=activated_decode[:, :I].contiguous(),
            all_w_blocks=gemm2_blocks, all_w_scales=gemm2_scales,
            all_bias=None, out_dim=H_PAD, expert_ids=expert_ids,
            fused_scales=fused_scales,
        )

    # Decode SwiGLU GEMM1 (production kernel)
    def do_decode_gemm1_swiglu():
        return compiler.run_moe_decode_gemm1_swiglu(
            input=hidden, all_w_blocks=gemm1_blocks,
            all_w_scales=gemm1_scales, all_bias=None,
            intermediate_size=I, expert_ids=expert_ids,
            alpha=1.702, clamp_limit=100.0,
        )

    activated_swiglu = do_decode_gemm1_swiglu()

    # Decode SwiGLU GEMM1 + fused GEMM2 (production pipeline)
    def do_decode_gemm1_gemm2():
        act = compiler.run_moe_decode_gemm1_swiglu(
            input=hidden, all_w_blocks=gemm1_blocks,
            all_w_scales=gemm1_scales, all_bias=None,
            intermediate_size=I, expert_ids=expert_ids,
            alpha=1.702, clamp_limit=100.0,
        )
        return compiler.run_moe_decode_gemm2_fused(
            input=act, all_w_blocks=gemm2_blocks,
            all_w_scales=gemm2_scales, all_bias=None,
            out_dim=H_PAD, expert_ids=expert_ids,
            fused_scales=fused_scales,
        )

    # bf16 linear baseline
    def do_bf16_qkv():
        return fun.linear(hidden[:, :H], qkv_w, qkv_b)

    gemm1_decode_t = bench(do_decode_gemm1, label="GEMM1 decode (raw, no SwiGLU)")
    gemm1_swiglu_t = bench(do_decode_gemm1_swiglu, label="GEMM1 decode (fused SwiGLU)")
    gemm2_decode_t = bench(do_decode_gemm2, label="GEMM2 decode (fused scatter)")
    both_decode_t = bench(do_decode_gemm1_gemm2, label="GEMM1+GEMM2 decode (production)")
    bf16_t = bench(do_bf16_qkv, label="bf16 linear [5120,2880] (BW baseline)")

    # ===================================================================
    # Part 2: Correctness check
    # ===================================================================
    print("\n--- Part 2: Correctness check ---\n")

    # Decode GEMM1: [K, 2*I] raw matmul
    decode_g1_out = compiler.run_moe_decode_gemm1(
        input=hidden, all_w_blocks=gemm1_blocks,
        all_w_scales=gemm1_scales,
        intermediate_size=I, expert_ids=expert_ids,
    )
    torch.mps.synchronize()

    eid_list = expert_ids.tolist()
    g1_max_err = 0.0
    for k_idx, eid in enumerate(eid_list):
        ref_w = dequant_expert_weights(gemm1_blocks, gemm1_scales, eid)
        ref_out = (hidden.float() @ ref_w.float().T).squeeze(0)
        # Compare vs bf16-rounded ref (kernel writes bf16 output)
        ref_bf16 = ref_out.bfloat16().float()
        kernel_out = decode_g1_out[k_idx].float()
        err = (kernel_out - ref_bf16).abs().max().item()
        g1_max_err = max(g1_max_err, err)
    g1_ok = g1_max_err < 0.01
    print(f"  GEMM1 decode max abs error (vs bf16 ref): {g1_max_err:.6f}  {'PASS' if g1_ok else 'FAIL'}")

    # Decode GEMM2 fused: [1, H_PAD] accumulated across K experts
    g2_input = decode_g1_out[:, :I].contiguous()
    decode_g2_out = compiler.run_moe_decode_gemm2_fused(
        input=g2_input, all_w_blocks=gemm2_blocks,
        all_w_scales=gemm2_scales, all_bias=None,
        out_dim=H_PAD, expert_ids=expert_ids, fused_scales=fused_scales,
    )
    torch.mps.synchronize()

    ref_g2_acc = torch.zeros(H_PAD, dtype=torch.float32, device=DEVICE)
    for k_idx, eid in enumerate(eid_list):
        ref_w2 = dequant_expert_weights(gemm2_blocks, gemm2_scales, eid)
        ref_out2 = (g2_input[k_idx].float().unsqueeze(0) @ ref_w2[:H_PAD].float().T).squeeze(0)
        ref_g2_acc += ref_out2 * fused_scales[k_idx].item()
    # GEMM2 outputs float32, compare directly
    g2_max_err = (decode_g2_out.squeeze(0).float() - ref_g2_acc).abs().max().item()
    g2_ok = g2_max_err < 0.5  # float32 accumulation, small rounding differences
    print(f"  GEMM2 decode max abs error (vs f32 ref):  {g2_max_err:.6f}  {'PASS' if g2_ok else 'FAIL'}")

    if not g1_ok or not g2_ok:
        print("  WARNING: Decode kernel correctness check FAILED!")

    # ===================================================================
    # Part 3: Bandwidth analysis
    # ===================================================================
    print("\n--- Part 3: Bandwidth analysis ---\n")

    bf16_bytes = qkv_w.nelement() * 2 + qkv_b.nelement() * 2  # weight + bias
    print(f"  GEMM1 weight data:   {g1_bytes / 1e6:.1f} MB")
    print(f"  GEMM2 weight data:   {g2_bytes / 1e6:.1f} MB")
    print(f"  Total MoE/layer:     {total_bytes / 1e6:.1f} MB")
    print(f"  bf16 QKV weight:     {bf16_bytes / 1e6:.1f} MB")
    print()
    print(f"  GEMM1 decode BW:     {g1_bytes / (gemm1_decode_t[0] / 1000) / 1e9:.1f} GB/s")
    print(f"  GEMM1 SwiGLU BW:     {g1_bytes / (gemm1_swiglu_t[0] / 1000) / 1e9:.1f} GB/s")
    print(f"  GEMM2 decode BW:     {g2_bytes / (gemm2_decode_t[0] / 1000) / 1e9:.1f} GB/s")
    print(f"  Combined BW:         {total_bytes / (both_decode_t[0] / 1000) / 1e9:.1f} GB/s")
    print(f"  bf16 linear BW:      {bf16_bytes / (bf16_t[0] / 1000) / 1e9:.1f} GB/s")
    print(f"  Pipeline overlap:    {(1.0 - both_decode_t[0]/(gemm1_swiglu_t[0]+gemm2_decode_t[0]))*100:.1f}%")

    # ===================================================================
    # Part 4: Back-to-back 24-layer GEMM
    # ===================================================================
    print("\n--- Part 4: Back-to-back 24-layer GEMM ---\n")

    def do_24_decode_gemm1():
        for i in range(NUM_LAYERS):
            compiler.run_moe_decode_gemm1(
                input=hidden, all_w_blocks=gemm1_blocks_list[i],
                all_w_scales=gemm1_scales_list[i],
                intermediate_size=I, expert_ids=expert_ids,
            )

    def do_24_decode_both():
        for i in range(NUM_LAYERS):
            act = compiler.run_moe_decode_gemm1_swiglu(
                input=hidden, all_w_blocks=gemm1_blocks_list[i],
                all_w_scales=gemm1_scales_list[i], all_bias=None,
                intermediate_size=I, expert_ids=expert_ids,
                alpha=1.702, clamp_limit=100.0,
            )
            compiler.run_moe_decode_gemm2_fused(
                input=act, all_w_blocks=gemm2_blocks_list[i],
                all_w_scales=gemm2_scales_list[i], all_bias=None,
                out_dim=H_PAD, expert_ids=expert_ids, fused_scales=fused_scales,
            )

    g1_24_decode = bench(do_24_decode_gemm1, warmup=3, iters=15, label="24x GEMM1 decode")
    both_24_decode = bench(do_24_decode_both, warmup=3, iters=15, label="24x GEMM1+GEMM2 decode (prod)")

    print(f"\n  Decode GEMM1/layer:    {g1_24_decode[0]/NUM_LAYERS:.3f} ms")
    print(f"  Decode both/layer:     {both_24_decode[0]/NUM_LAYERS:.3f} ms")
    print(f"  Decode 24-layer total: {both_24_decode[0]:.1f} ms")
    print(f"  24-layer BW:         {total_bytes*NUM_LAYERS/(both_24_decode[0]/1000)/1e9:.0f} GB/s")

    # ===================================================================
    # Part 5: CPU submission overhead
    # ===================================================================
    print("\n--- Part 5: CPU submission overhead ---\n")

    # Measure CPU-side time without final sync (GPU still running)
    for _ in range(5):
        do_24_decode_both()
    torch.mps.synchronize()

    cpu_times = []
    for _ in range(15):
        torch.mps.synchronize()
        t0 = time.perf_counter()
        do_24_decode_both()
        cpu_times.append((time.perf_counter() - t0) * 1000)
        torch.mps.synchronize()

    cpu_times.sort()
    cpu_med = cpu_times[len(cpu_times) // 2]

    # Also measure with sync (total wall time)
    sync_times = []
    for _ in range(15):
        torch.mps.synchronize()
        t0 = time.perf_counter()
        do_24_decode_both()
        torch.mps.synchronize()
        sync_times.append((time.perf_counter() - t0) * 1000)

    sync_times.sort()
    sync_med = sync_times[len(sync_times) // 2]
    gpu_only = sync_med - cpu_med

    print(f"  CPU submission (no sync): {cpu_med:.1f} ms")
    print(f"  Total (with sync):        {sync_med:.1f} ms")
    print(f"  GPU execution (approx):   {gpu_only:.1f} ms")

    if cpu_med > gpu_only:
        print(f"  >>> CPU-bound: submission ({cpu_med:.1f}ms) > GPU ({gpu_only:.1f}ms)")
    else:
        print(f"  >>> GPU-bound: GPU ({gpu_only:.1f}ms) > submission ({cpu_med:.1f}ms)")

    print("\nDone.")


if __name__ == "__main__":
    main()
