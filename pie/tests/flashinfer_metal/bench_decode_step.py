"""
Decode step benchmarks for GPT-OSS on MPS.

Measures per-operation GPU time with correct methodology:
  1. Isolated: sync→op→sync measures raw GPU time for one operation.
  2. Pipeline: sync→(all ops)→sync measures actual end-to-end time.
  3. The gap between sum-of-isolated and pipeline shows GPU overlap.
  4. Differential: full step with one op removed shows critical-path impact.

WARNING about the in-model profiler (_sync_record):
  It calls torch.mps.synchronize() between EVERY operation. This drains the
  GPU pipeline, so time attributed to an op includes all preceding GPU work
  still in flight. E.g., "387ms for KV append" was really "time for QKV proj
  + RoPE + KV append to finish on GPU."

Usage:
    cd /Users/ingim/Workspace/pie-mac/pie
    PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 uv run python \
        tests/flashinfer_metal/bench_decode_step.py
"""

import sys
import time

import torch
import torch.nn.functional as fun

sys.path.insert(0, "src")


# ---------------------------------------------------------------------------
# Bench helper
# ---------------------------------------------------------------------------

def bench(fn, warmup=10, iters=50):
    """Benchmark fn with sync before and after. Returns (median, p10, p90) ms."""
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
    return times[n // 2], times[n // 10], times[n * 9 // 10]


def fmt(ms_tuple):
    return f"{ms_tuple[0]:8.3f} ms  (p10={ms_tuple[1]:.3f}, p90={ms_tuple[2]:.3f})"


# ---------------------------------------------------------------------------
# GPT-OSS-20B exact dims (tp_size=1)
# ---------------------------------------------------------------------------

H = 2880                # dim_hidden
H_PAD = 2944            # padded for MoE (align to 64)
DIM_HEAD = 64           # head_dim
NUM_Q_HEADS = 64        # num_attention_heads
NUM_KV_HEADS = 8        # num_key_value_heads
Q_SIZE = NUM_Q_HEADS * DIM_HEAD      # 4096
KV_SIZE = NUM_KV_HEADS * DIM_HEAD    # 512
QKV_SIZE = Q_SIZE + 2 * KV_SIZE      # 5120
I = 2944                # intermediate_size (padded from 2880)
E = 32                  # num_experts
K = 4                   # top_k
PAGE_SIZE = 16
NUM_PAGES = 512         # realistic KV cache
NUM_LAYERS = 24
VOCAB = 201088          # vocab_size
RMS_EPS = 1e-5

DEVICE = "mps"
DTYPE = torch.bfloat16


def main():
    from flashinfer_metal._compiler import MetalCompiler
    from flashinfer_metal._wrappers import (
        append_paged_kv_cache,
        apply_rope_with_cos_sin_cache_inplace,
        BatchAttentionWithAttentionSinkWrapper,
    )

    torch.manual_seed(42)
    compiler = MetalCompiler()

    print("=" * 72)
    print("GPT-OSS-20B Decode Step Benchmark (1 token)")
    print(f"  H={H}, head={DIM_HEAD}, Q_heads={NUM_Q_HEADS}, KV_heads={NUM_KV_HEADS}")
    print(f"  MoE: E={E}, K={K}, I={I} (padded)")
    print(f"  Vocab: {VOCAB}, Layers: {NUM_LAYERS}")
    print("=" * 72)

    # ----- Allocate weight tensors matching real model shapes -----
    # NOTE: Per-layer weights are allocated as separate tensors so each layer
    # reads from different DRAM locations (realistic cache behavior).
    # The old code reused one set of weights for all 24 layers, which kept
    # them hot in GPU cache and produced unrealistically fast results.

    # Embedding + LM head
    embed_w = torch.randn(VOCAB, H, dtype=DTYPE, device=DEVICE)
    lm_head_w = torch.randn(VOCAB, H, dtype=DTYPE, device=DEVICE)
    norm_last_w = torch.randn(H, dtype=DTYPE, device=DEVICE)

    # Per-layer attention weights (unique per layer)
    norm_attn_ws = [torch.randn(H, dtype=DTYPE, device=DEVICE) for _ in range(NUM_LAYERS)]
    qkv_ws = [torch.randn(QKV_SIZE, H, dtype=DTYPE, device=DEVICE) for _ in range(NUM_LAYERS)]
    qkv_bs = [torch.randn(QKV_SIZE, dtype=DTYPE, device=DEVICE) for _ in range(NUM_LAYERS)]
    o_ws = [torch.randn(H, Q_SIZE, dtype=DTYPE, device=DEVICE) for _ in range(NUM_LAYERS)]

    # Per-layer MoE weights (unique per layer)
    norm_mlp_ws = [torch.randn(H, dtype=DTYPE, device=DEVICE) for _ in range(NUM_LAYERS)]
    router_ws = [torch.randn(E, H, dtype=DTYPE, device=DEVICE) for _ in range(NUM_LAYERS)]
    router_bs = [torch.randn(E, dtype=DTYPE, device=DEVICE) for _ in range(NUM_LAYERS)]

    # Per-layer MoE FP4 packed weights (uint8, unique per layer)
    gemm1_blocks_list = [torch.randint(0, 256, (E, 2 * I, H_PAD // 2),
                                        dtype=torch.uint8, device=DEVICE) for _ in range(NUM_LAYERS)]
    gemm1_scales_list = [torch.full((E, 2 * I, H_PAD // 32), 127,
                                     dtype=torch.uint8, device=DEVICE) for _ in range(NUM_LAYERS)]
    gemm2_blocks_list = [torch.randint(0, 256, (E, H_PAD, I // 2),
                                        dtype=torch.uint8, device=DEVICE) for _ in range(NUM_LAYERS)]
    gemm2_scales_list = [torch.full((E, H_PAD, I // 32), 127,
                                     dtype=torch.uint8, device=DEVICE) for _ in range(NUM_LAYERS)]

    # Single-layer references for isolated benchmarks (Part 1)
    norm_attn_w = norm_attn_ws[0]
    qkv_w = qkv_ws[0]
    qkv_b = qkv_bs[0]
    o_w = o_ws[0]
    norm_mlp_w = norm_mlp_ws[0]
    router_w = router_ws[0]
    router_b = router_bs[0]
    gemm1_blocks = gemm1_blocks_list[0]
    gemm1_scales = gemm1_scales_list[0]
    gemm2_blocks = gemm2_blocks_list[0]
    gemm2_scales = gemm2_scales_list[0]

    # RoPE cache
    rope_cache = torch.randn(8192, DIM_HEAD, dtype=torch.float32, device=DEVICE)

    # KV cache + metadata
    paged_kv = torch.randn(NUM_PAGES, 2, PAGE_SIZE, NUM_KV_HEADS, DIM_HEAD,
                            dtype=DTYPE, device=DEVICE)
    kv_page_indices = torch.arange(NUM_PAGES, dtype=torch.int32, device=DEVICE)
    kv_page_indptr = torch.tensor([0, NUM_PAGES], dtype=torch.int32, device=DEVICE)
    kv_last_page_len = torch.tensor([11], dtype=torch.int32, device=DEVICE)
    batch_indices = torch.tensor([0], dtype=torch.int32, device=DEVICE)
    batch_positions = torch.tensor([42], dtype=torch.int32, device=DEVICE)
    position_ids = torch.tensor([42], dtype=torch.long, device=DEVICE)
    sinks = torch.randn(NUM_Q_HEADS, dtype=torch.float32, device=DEVICE)
    scaling = DIM_HEAD ** -0.5
    qo_indptr = torch.tensor([0, 1], dtype=torch.int32, device=DEVICE)

    # Attention wrapper
    attn_wrapper = BatchAttentionWithAttentionSinkWrapper(
        float_workspace_buffer=torch.empty(1, device=DEVICE),
        window_left=-1, q_data_type=DTYPE, kv_data_type=DTYPE,
    )
    attn_wrapper.plan(
        qo_indptr, kv_page_indptr, kv_page_indices, kv_last_page_len,
        NUM_Q_HEADS, NUM_KV_HEADS, DIM_HEAD, PAGE_SIZE,
        causal=True, window_left=-1, q_data_type=DTYPE, kv_data_type=DTYPE,
    )

    hidden = torch.randn(1, H, dtype=DTYPE, device=DEVICE)
    expert_ids = torch.tensor([3, 7, 15, 28], dtype=torch.int32, device=DEVICE)
    routing_weights = torch.tensor([0.3, 0.25, 0.25, 0.2],
                                    dtype=torch.float32, device=DEVICE)

    # Pre-compute some intermediates for isolated benchmarks
    normed = fun.rms_norm(hidden, [H], norm_attn_w, RMS_EPS)
    qkv = fun.linear(normed, qkv_w, qkv_b)
    q_base = qkv[:, :Q_SIZE].view(1, NUM_Q_HEADS, DIM_HEAD).clone()
    k_base = qkv[:, Q_SIZE:Q_SIZE + KV_SIZE].view(1, NUM_KV_HEADS, DIM_HEAD).clone()
    v_base = qkv[:, Q_SIZE + KV_SIZE:].view(1, NUM_KV_HEADS, DIM_HEAD).clone()

    # ===================================================================
    # Part 1: Isolated per-operation GPU time
    # ===================================================================

    print("\n--- Part 1: Isolated GPU time (sync → op → sync) ---\n")

    def op_embed():
        return fun.embedding(torch.tensor([42], device=DEVICE), embed_w)

    def op_rms_norm():
        return fun.rms_norm(hidden, [H], norm_attn_w, RMS_EPS)

    def op_qkv_proj():
        return fun.linear(hidden, qkv_w, qkv_b)

    def op_rope():
        q = q_base.clone()
        k = k_base.clone()
        apply_rope_with_cos_sin_cache_inplace(
            positions=position_ids.to(torch.int32),
            query=q, key=k, head_size=DIM_HEAD,
            cos_sin_cache=rope_cache, is_neox=True,
        )
        return q, k

    def op_kv_append():
        append_paged_kv_cache(
            k_base, v_base, batch_indices, batch_positions,
            paged_kv, kv_page_indices, kv_page_indptr, kv_last_page_len,
        )

    def op_attention():
        return attn_wrapper.run(q_base, paged_kv, sinks, scaling)

    def op_o_proj():
        x = torch.randn(1, Q_SIZE, dtype=DTYPE, device=DEVICE)
        return fun.linear(x, o_w)

    def op_residual():
        a = torch.randn(1, H, dtype=DTYPE, device=DEVICE)
        return hidden + a

    def op_router():
        return fun.linear(hidden, router_w, router_b)

    # MoE routing (softmax + topk)
    def op_moe_routing():
        logits = fun.linear(hidden, router_w, router_b)
        scores = torch.softmax(logits.float(), dim=-1)
        return torch.topk(scores, K, dim=-1)

    # MoE GEMM1 (batched, 4 experts)
    def op_moe_gemm1():
        return compiler.run_moe_batched_gemm1_fp4(
            input=hidden[:, :H_PAD] if H_PAD > H else
                  torch.nn.functional.pad(hidden, (0, H_PAD - H)),
            all_w_blocks=gemm1_blocks, all_w_scales=gemm1_scales,
            all_bias=None, intermediate_size=I, expert_ids=expert_ids,
            expert_alphas=[1.702] * K, expert_betas=[1.0] * K,
            expert_clamp_limits=[100.0] * K,
            expert_scale_gates=[1.0] * K, expert_scale_ups=[1.0] * K,
        )

    # Pre-compute activated for GEMM2 benchmark
    hidden_padded = torch.nn.functional.pad(hidden, (0, H_PAD - H))
    activated = compiler.run_moe_batched_gemm1_fp4(
        input=hidden_padded,
        all_w_blocks=gemm1_blocks, all_w_scales=gemm1_scales,
        all_bias=None, intermediate_size=I, expert_ids=expert_ids,
        expert_alphas=[1.702] * K, expert_betas=[1.0] * K,
        expert_clamp_limits=[100.0] * K,
        expert_scale_gates=[1.0] * K, expert_scale_ups=[1.0] * K,
    )

    def op_moe_gemm2_fused():
        return compiler.run_moe_batched_gemm2_fused_fp4(
            input=activated, all_w_blocks=gemm2_blocks,
            all_w_scales=gemm2_scales, all_bias=None,
            out_dim=H_PAD, expert_ids=expert_ids, fused_scales=routing_weights,
        )

    def op_lm_head():
        normed = fun.rms_norm(hidden, [H], norm_last_w, RMS_EPS)
        return fun.linear(normed, lm_head_w)

    ops_layer = [
        ("rms_norm (attn)",  op_rms_norm),
        ("qkv_proj",         op_qkv_proj),
        ("rope",             op_rope),
        ("kv_append",        op_kv_append),
        ("attention",        op_attention),
        ("o_proj",           op_o_proj),
        ("residual (attn)",  op_residual),
        ("rms_norm (moe)",   op_rms_norm),
        ("moe_routing",      op_moe_routing),
        ("moe_gemm1",        op_moe_gemm1),
        ("moe_gemm2_fused",  op_moe_gemm2_fused),
        ("residual (moe)",   op_residual),
    ]

    layer_isolated_sum = 0.0
    for name, fn in ops_layer:
        t = bench(fn)
        layer_isolated_sum += t[0]
        print(f"  {name:20s}  {fmt(t)}")

    print(f"\n  {'--- per-layer sum':20s}  {layer_isolated_sum:8.3f} ms")
    print(f"  {'--- × 24 layers':20s}  {layer_isolated_sum * NUM_LAYERS:8.3f} ms")

    # Non-layer ops
    print()
    t_embed = bench(op_embed)
    t_lm_head = bench(op_lm_head)
    print(f"  {'embed_tokens':20s}  {fmt(t_embed)}")
    print(f"  {'lm_head':20s}  {fmt(t_lm_head)}")

    total_isolated = layer_isolated_sum * NUM_LAYERS + t_embed[0] + t_lm_head[0]
    print(f"\n  {'TOTAL ISOLATED':20s}  {total_isolated:8.3f} ms")

    # ===================================================================
    # Part 2: Full layer pipeline
    # ===================================================================

    print("\n--- Part 2: Full layer pipeline (sync → all ops → sync) ---\n")

    def one_layer(h, layer_idx):
        # Attention block
        normed = fun.rms_norm(h, [H], norm_attn_ws[layer_idx], RMS_EPS)
        qkv = fun.linear(normed, qkv_ws[layer_idx], qkv_bs[layer_idx])
        q = qkv[:, :Q_SIZE].view(1, NUM_Q_HEADS, DIM_HEAD)
        k = qkv[:, Q_SIZE:Q_SIZE + KV_SIZE].view(1, NUM_KV_HEADS, DIM_HEAD)
        v = qkv[:, Q_SIZE + KV_SIZE:].view(1, NUM_KV_HEADS, DIM_HEAD)
        apply_rope_with_cos_sin_cache_inplace(
            positions=position_ids.to(torch.int32),
            query=q, key=k, head_size=DIM_HEAD,
            cos_sin_cache=rope_cache, is_neox=True,
        )
        append_paged_kv_cache(
            k, v, batch_indices, batch_positions,
            paged_kv, kv_page_indices, kv_page_indptr, kv_last_page_len,
        )
        attn_out = attn_wrapper.run(q, paged_kv, sinks, scaling)
        attn_out = attn_out.reshape(1, -1)
        attn_proj = fun.linear(attn_out, o_ws[layer_idx])
        h = h + attn_proj

        # MoE block
        normed2 = fun.rms_norm(h, [H], norm_mlp_ws[layer_idx], RMS_EPS)
        router_logits = fun.linear(normed2, router_ws[layer_idx], router_bs[layer_idx])
        scores = torch.softmax(router_logits.float(), dim=-1)
        _, topk_ids = torch.topk(scores, K, dim=-1)
        local_ids = topk_ids[0].to(torch.int32)
        h_padded = torch.nn.functional.pad(normed2.to(DTYPE), (0, H_PAD - H))
        act = compiler.run_moe_batched_gemm1_fp4(
            input=h_padded, all_w_blocks=gemm1_blocks_list[layer_idx],
            all_w_scales=gemm1_scales_list[layer_idx], all_bias=None,
            intermediate_size=I, expert_ids=local_ids,
            expert_alphas=[1.702] * K, expert_betas=[1.0] * K,
            expert_clamp_limits=[100.0] * K,
            expert_scale_gates=[1.0] * K, expert_scale_ups=[1.0] * K,
        )
        moe_out = compiler.run_moe_batched_gemm2_fused_fp4(
            input=act, all_w_blocks=gemm2_blocks_list[layer_idx],
            all_w_scales=gemm2_scales_list[layer_idx], all_bias=None,
            out_dim=H_PAD, expert_ids=local_ids, fused_scales=routing_weights,
        )
        return h + moe_out[:, :H].to(DTYPE)

    def run_one_layer():
        one_layer(hidden, 0)

    layer_t = bench(run_one_layer, warmup=5, iters=30)
    print(f"  {'1 layer pipeline':20s}  {fmt(layer_t)}")
    print(f"  {'× 24 layers':20s}  {layer_t[0] * NUM_LAYERS:8.3f} ms")
    overlap_pct = (1.0 - layer_t[0] / layer_isolated_sum) * 100
    print(f"\n  GPU overlap: {overlap_pct:.1f}% "
          f"(isolated {layer_isolated_sum:.3f} → pipeline {layer_t[0]:.3f} ms)")

    # ===================================================================
    # Part 3: Full 24-layer decode step
    # ===================================================================

    print("\n--- Part 3: Full 24-layer decode step ---\n")

    def full_step():
        h = hidden
        for i in range(NUM_LAYERS):
            h = one_layer(h, i)
        # lm_head
        normed = fun.rms_norm(h, [H], norm_last_w, RMS_EPS)
        return fun.linear(normed, lm_head_w)

    step_t = bench(full_step, warmup=3, iters=20)
    print(f"  {'24-layer + lm_head':20s}  {fmt(step_t)}")
    print(f"  {'vs 24 × 1-layer':20s}  {layer_t[0] * NUM_LAYERS + t_lm_head[0]:8.3f} ms")
    print(f"  {'vs isolated sum':20s}  {total_isolated:8.3f} ms")

    # ===================================================================
    # Part 4: Differential profiling
    # ===================================================================

    print("\n--- Part 4: Differential profiling (critical-path impact) ---")
    print("  Full 24-layer step with one op class removed.\n")

    baseline = step_t[0]

    def make_step_fn(skip: str):
        """Build a full-step function with one op class skipped."""
        def step():
            h = hidden
            for i in range(NUM_LAYERS):
                # Attention block
                if skip != "rms_norm":
                    normed = fun.rms_norm(h, [H], norm_attn_ws[i], RMS_EPS)
                else:
                    normed = h
                qkv = fun.linear(normed, qkv_ws[i], qkv_bs[i])
                q = qkv[:, :Q_SIZE].view(1, NUM_Q_HEADS, DIM_HEAD)
                k = qkv[:, Q_SIZE:Q_SIZE + KV_SIZE].view(1, NUM_KV_HEADS, DIM_HEAD)
                v = qkv[:, Q_SIZE + KV_SIZE:].view(1, NUM_KV_HEADS, DIM_HEAD)
                if skip != "rope":
                    apply_rope_with_cos_sin_cache_inplace(
                        positions=position_ids.to(torch.int32),
                        query=q, key=k, head_size=DIM_HEAD,
                        cos_sin_cache=rope_cache, is_neox=True,
                    )
                if skip != "kv_append":
                    append_paged_kv_cache(
                        k, v, batch_indices, batch_positions,
                        paged_kv, kv_page_indices, kv_page_indptr,
                        kv_last_page_len,
                    )
                if skip != "attention":
                    attn_out = attn_wrapper.run(q, paged_kv, sinks, scaling)
                    attn_out = attn_out.reshape(1, -1)
                    attn_proj = fun.linear(attn_out, o_ws[i])
                else:
                    attn_proj = torch.zeros(1, H, dtype=DTYPE, device=DEVICE)
                h = h + attn_proj

                # MoE block
                if skip != "rms_norm":
                    normed2 = fun.rms_norm(h, [H], norm_mlp_ws[i], RMS_EPS)
                else:
                    normed2 = h
                if skip != "moe_gemms":
                    router_logits = fun.linear(normed2, router_ws[i], router_bs[i])
                    scores = torch.softmax(router_logits.float(), dim=-1)
                    _, topk_ids = torch.topk(scores, K, dim=-1)
                    local_ids = topk_ids[0].to(torch.int32)
                    h_padded = torch.nn.functional.pad(
                        normed2.to(DTYPE), (0, H_PAD - H))
                    act = compiler.run_moe_batched_gemm1_fp4(
                        input=h_padded,
                        all_w_blocks=gemm1_blocks_list[i],
                        all_w_scales=gemm1_scales_list[i], all_bias=None,
                        intermediate_size=I, expert_ids=local_ids,
                        expert_alphas=[1.702] * K, expert_betas=[1.0] * K,
                        expert_clamp_limits=[100.0] * K,
                        expert_scale_gates=[1.0] * K,
                        expert_scale_ups=[1.0] * K,
                    )
                    moe_out = compiler.run_moe_batched_gemm2_fused_fp4(
                        input=act,
                        all_w_blocks=gemm2_blocks_list[i],
                        all_w_scales=gemm2_scales_list[i], all_bias=None,
                        out_dim=H_PAD, expert_ids=local_ids,
                        fused_scales=routing_weights,
                    )
                    h = h + moe_out[:, :H].to(DTYPE)

            # lm_head
            if skip != "lm_head":
                normed = fun.rms_norm(h, [H], norm_last_w, RMS_EPS)
                return fun.linear(normed, lm_head_w)
            return h
        return step

    skip_targets = [
        "kv_append",
        "rope",
        "attention",
        "moe_gemms",
        "rms_norm",
        "lm_head",
    ]

    for target in skip_targets:
        fn = make_step_fn(target)
        t = bench(fn, warmup=3, iters=20)
        delta = baseline - t[0]
        pct = delta / baseline * 100
        print(f"  skip {target:15s}  {fmt(t)}  | delta = {delta:+.1f} ms ({pct:+.1f}%)")

    print(f"\n  {'baseline':20s}  {baseline:8.3f} ms")

    print("\n" + "=" * 72)
    print("DONE")
    print("=" * 72)


if __name__ == "__main__":
    main()
