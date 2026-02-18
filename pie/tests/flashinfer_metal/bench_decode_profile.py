"""
Real model decode profiling for GPT-OSS-20B.

Loads the actual model (with compact_weights) and measures per-component
timing with careful GPU synchronization.

Usage:
    cd /Users/ingim/Workspace/pie-mac/pie
    PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 uv run python \
        tests/flashinfer_metal/bench_decode_profile.py
"""

import sys
import time

sys.path.insert(0, "src")

import torch
import torch.nn.functional as fun

from pie_backend.engine import Engine
from pie_backend.config import RuntimeConfig


def bench(fn, warmup=5, iters=20):
    """Benchmark fn with sync before and after. Returns (median, min, max) ms."""
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
    return times[n // 2], times[0], times[-1]


def fmt(ms_tuple):
    return f"{ms_tuple[0]:8.3f} ms  (min={ms_tuple[1]:.3f}, max={ms_tuple[2]:.3f})"


def main():
    print("=" * 72)
    print("GPT-OSS-20B Real Model Decode Profiling")
    print("=" * 72)

    # Load model
    print("\nLoading model...")
    config = RuntimeConfig.from_args(hf_repo="openai/gpt-oss-20b")
    engine = Engine.load(config)
    engine.forward_pass.profiler = None  # Disable profiler syncs
    fp = engine.forward_pass
    cfg = fp.model_config
    device = config.device

    print(f"  H={cfg.dim_hidden}, head={cfg.dim_head}, Q={cfg.num_q_heads}, KV={cfg.num_kv_heads}")
    print(f"  MoE: E={cfg.num_experts}, K={cfg.experts_per_token}, I={cfg.dim_mlp}")
    print(f"  Layers: {cfg.num_layers}, Padded H={fp.padded_hidden_size}")

    # Set up minimal decode state
    from flashinfer_metal._wrappers import (
        get_seq_lens,
        get_batch_indices_positions,
        BatchAttentionWithAttentionSinkWrapper,
    )

    page_size = int(engine.kv_cache_at_layer[0].shape[2])
    num_pages = 32  # ~512 tokens of context
    kv_page_indices = torch.arange(num_pages, dtype=torch.int32, device=device)
    kv_page_indptr = torch.tensor([0, num_pages], dtype=torch.int32, device=device)
    kv_last_page_lens = torch.tensor([11], dtype=torch.int32, device=device)
    qo_indptr = torch.tensor([0, 1], dtype=torch.int32, device=device)

    seq_lens = get_seq_lens(kv_page_indptr, kv_last_page_lens, page_size)
    batch_indices, batch_positions = get_batch_indices_positions(
        append_indptr=qo_indptr, seq_lens=seq_lens, nnz=1,
    )
    position_ids_i32 = torch.tensor([42], dtype=torch.int32, device=device)

    # Plan attention wrappers
    local_num_q_heads = cfg.num_q_heads // fp.tp_size
    local_num_kv_heads = cfg.num_kv_heads // fp.tp_size

    fp.wrapper_window.plan(
        qo_indptr, kv_page_indptr, kv_page_indices, kv_last_page_lens,
        local_num_q_heads, local_num_kv_heads, cfg.dim_head, page_size,
        causal=True, window_left=cfg.sliding_window - 1,
        q_data_type=config.activation_dtype, kv_data_type=config.activation_dtype,
    )
    fp.wrapper_full.plan(
        qo_indptr, kv_page_indptr, kv_page_indices, kv_last_page_lens,
        local_num_q_heads, local_num_kv_heads, cfg.dim_head, page_size,
        causal=True, window_left=-1,
        q_data_type=config.activation_dtype, kv_data_type=config.activation_dtype,
    )

    # Dummy hidden state
    hidden = torch.randn(1, cfg.dim_hidden, dtype=config.activation_dtype, device=device)

    # ===================================================================
    # Part 1: Full decode step timing
    # ===================================================================
    print("\n--- Part 1: Full decode step ---\n")

    def full_step():
        h = fun.embedding(torch.tensor([42], device=device), fp._embed_token)
        for layer_idx in range(cfg.num_layers):
            wrapper = fp.wrapper_window if layer_idx % 2 == 0 else fp.wrapper_full
            h = fp.attention(
                h, layer_idx, position_ids_i32,
                engine.kv_cache_at_layer[layer_idx],
                kv_page_indices, kv_page_indptr, kv_last_page_lens,
                batch_indices, batch_positions, None, wrapper,
            )
            h = fp.moe(h, layer_idx)
        return fp.lm_head(h)

    step_t = bench(full_step, warmup=3, iters=15)
    print(f"  {'Full 24-layer step':25s}  {fmt(step_t)}")

    # ===================================================================
    # Part 2: Per-component timing (with sync between each)
    # ===================================================================
    print("\n--- Part 2: Per-component timing (synced) ---\n")

    # Single layer: attention only
    def one_attention(layer_idx=0):
        wrapper = fp.wrapper_window if layer_idx % 2 == 0 else fp.wrapper_full
        return fp.attention(
            hidden, layer_idx, position_ids_i32,
            engine.kv_cache_at_layer[layer_idx],
            kv_page_indices, kv_page_indptr, kv_last_page_lens,
            batch_indices, batch_positions, None, wrapper,
        )

    # Single layer: moe only
    def one_moe(layer_idx=0):
        return fp.moe(hidden, layer_idx)

    # LM head
    def do_lm_head():
        return fp.lm_head(hidden)

    # Embed
    def do_embed():
        return fun.embedding(torch.tensor([42], device=device), fp._embed_token)

    attn_t = bench(one_attention)
    moe_t = bench(one_moe)
    lm_head_t = bench(do_lm_head)
    embed_t = bench(do_embed)

    print(f"  {'1 attention layer':25s}  {fmt(attn_t)}")
    print(f"  {'1 MoE layer':25s}  {fmt(moe_t)}")
    print(f"  {'lm_head':25s}  {fmt(lm_head_t)}")
    print(f"  {'embed':25s}  {fmt(embed_t)}")
    print()
    per_layer = attn_t[0] + moe_t[0]
    total_est = per_layer * cfg.num_layers + lm_head_t[0] + embed_t[0]
    print(f"  {'per-layer (attn+moe)':25s}  {per_layer:8.3f} ms")
    print(f"  {'estimated 24-layer':25s}  {total_est:8.3f} ms  (vs actual {step_t[0]:.3f})")
    print(f"  {'pipeline overlap':25s}  {(1.0 - step_t[0]/total_est)*100:.1f}%")

    # ===================================================================
    # Part 3: Attention sub-operations
    # ===================================================================
    print("\n--- Part 3: Attention sub-ops (layer 0) ---\n")

    from flashinfer_metal._wrappers import (
        apply_rope_with_cos_sin_cache_inplace,
        append_paged_kv_cache,
    )

    lw = fp._layer_weights[0]
    local_q_size = local_num_q_heads * cfg.dim_head
    local_kv_size = local_num_kv_heads * cfg.dim_head

    def op_attn_rms_norm():
        return fun.rms_norm(hidden, [cfg.dim_hidden], lw["norm_attn"], cfg.rms_norm_eps)

    def op_qkv_proj():
        return fun.linear(hidden, lw["proj_qkv.weight"], lw["proj_qkv.bias"])

    normed = fun.rms_norm(hidden, [cfg.dim_hidden], lw["norm_attn"], cfg.rms_norm_eps)
    qkv = fun.linear(normed, lw["proj_qkv.weight"], lw["proj_qkv.bias"])
    q_base = qkv[:, :local_q_size].view(1, local_num_q_heads, cfg.dim_head).clone()
    k_base = qkv[:, local_q_size:local_q_size+local_kv_size].view(1, local_num_kv_heads, cfg.dim_head).clone()
    v_base = qkv[:, local_q_size+local_kv_size:].view(1, local_num_kv_heads, cfg.dim_head).clone()

    def op_rope():
        q = q_base.clone(); k = k_base.clone()
        apply_rope_with_cos_sin_cache_inplace(
            positions=position_ids_i32, query=q, key=k,
            head_size=cfg.dim_head, cos_sin_cache=fp._rope_cos_sin_cache, is_neox=True,
        )

    def op_kv_append():
        append_paged_kv_cache(
            k_base, v_base, batch_indices, batch_positions,
            engine.kv_cache_at_layer[0], kv_page_indices, kv_page_indptr, kv_last_page_lens,
        )

    sinks = lw["attn_sinks"]
    scaling = cfg.dim_head ** -0.5
    wrapper0 = fp.wrapper_window

    def op_attention():
        return wrapper0.run(q_base, engine.kv_cache_at_layer[0], sinks, scaling)

    attn_out = wrapper0.run(q_base, engine.kv_cache_at_layer[0], sinks, scaling)
    attn_out_flat = attn_out.reshape(1, -1)

    def op_o_proj():
        return fun.linear(attn_out_flat, lw["proj_o"])

    def op_residual():
        return hidden + attn_out_flat[:, :cfg.dim_hidden] if attn_out_flat.shape[1] > cfg.dim_hidden else hidden + hidden

    sub_ops = [
        ("rms_norm (attn)", op_attn_rms_norm),
        ("qkv_proj", op_qkv_proj),
        ("rope", op_rope),
        ("kv_append", op_kv_append),
        ("attention_decode", op_attention),
        ("o_proj", op_o_proj),
        ("residual_add", op_residual),
    ]

    attn_sub_sum = 0.0
    for name, fn in sub_ops:
        t = bench(fn)
        attn_sub_sum += t[0]
        print(f"  {name:25s}  {fmt(t)}")
    print(f"  {'--- sum':25s}  {attn_sub_sum:8.3f} ms  (vs pipeline {attn_t[0]:.3f})")

    # ===================================================================
    # Part 4: MoE sub-operations
    # ===================================================================
    print("\n--- Part 4: MoE sub-ops (layer 0) ---\n")

    from flashinfer_metal._compiler import MetalCompiler
    compiler = MetalCompiler()

    def op_moe_rms_norm():
        return fun.rms_norm(hidden, [cfg.dim_hidden], lw["norm_mlp"], cfg.rms_norm_eps)

    def op_router():
        normed2 = fun.rms_norm(hidden, [cfg.dim_hidden], lw["norm_mlp"], cfg.rms_norm_eps)
        return fun.linear(normed2.reshape(-1, cfg.dim_hidden), lw["router.weight"], lw["router.bias"])

    def op_routing_only():
        """Just the routing ops (softmax + topk + normalize)."""
        logits = torch.randn(1, cfg.num_experts, dtype=torch.float32, device=device)
        scores = torch.softmax(logits, dim=-1)
        topk_w, topk_i = torch.topk(scores, cfg.experts_per_token, dim=-1)
        topk_w = topk_w / topk_w.sum(dim=-1, keepdim=True)
        return topk_w, topk_i

    def op_full_moe():
        return fp.moe(hidden, 0)

    moe_sub_ops = [
        ("rms_norm (moe)", op_moe_rms_norm),
        ("router proj", op_router),
        ("routing (softmax+topk)", op_routing_only),
        ("full MoE layer", op_full_moe),
    ]

    for name, fn in moe_sub_ops:
        t = bench(fn)
        print(f"  {name:25s}  {fmt(t)}")

    # ===================================================================
    # Part 5: CPU submission time (no GPU sync)
    # ===================================================================
    print("\n--- Part 5: CPU submission time (no final sync) ---\n")

    # Measure how long Python takes to submit all ops (without waiting for GPU)
    torch.mps.synchronize()
    times_cpu = []
    for _ in range(10):
        torch.mps.synchronize()
        t0 = time.perf_counter()
        h = fun.embedding(torch.tensor([42], device=device), fp._embed_token)
        for layer_idx in range(cfg.num_layers):
            wrapper = fp.wrapper_window if layer_idx % 2 == 0 else fp.wrapper_full
            h = fp.attention(
                h, layer_idx, position_ids_i32,
                engine.kv_cache_at_layer[layer_idx],
                kv_page_indices, kv_page_indptr, kv_last_page_lens,
                batch_indices, batch_positions, None, wrapper,
            )
            h = fp.moe(h, layer_idx)
        _ = fp.lm_head(h)
        cpu_ms = (time.perf_counter() - t0) * 1000
        times_cpu.append(cpu_ms)
        torch.mps.synchronize()  # drain before next iteration

    times_cpu.sort()
    cpu_median = times_cpu[len(times_cpu) // 2]
    print(f"  {'CPU submission (no sync)':25s}  {cpu_median:8.3f} ms")
    print(f"  {'Full step (with sync)':25s}  {step_t[0]:8.3f} ms")
    print(f"  {'GPU-only (diff)':25s}  {step_t[0] - cpu_median:8.3f} ms")

    # ===================================================================
    # Part 6: Per-layer scaling
    # ===================================================================
    print("\n--- Part 6: Per-layer scaling ---\n")

    for n_layers in [1, 4, 8, 12, 24]:
        def step_n():
            h = hidden
            for i in range(n_layers):
                wrapper = fp.wrapper_window if i % 2 == 0 else fp.wrapper_full
                h = fp.attention(
                    h, i, position_ids_i32,
                    engine.kv_cache_at_layer[i],
                    kv_page_indices, kv_page_indptr, kv_last_page_lens,
                    batch_indices, batch_positions, None, wrapper,
                )
                h = fp.moe(h, i)
            return h

        t = bench(step_n, warmup=3, iters=10)
        per_l = t[0] / n_layers
        print(f"  {n_layers:2d} layers:  {fmt(t)}   ({per_l:.2f} ms/layer)")

    # ===================================================================
    # Summary
    # ===================================================================
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"  Full decode step:        {step_t[0]:.1f} ms")
    print(f"  CPU submission:          {cpu_median:.1f} ms")
    print(f"  GPU execution overhead:  {step_t[0] - cpu_median:.1f} ms")
    print(f"  Attention/layer (synced):{attn_t[0]:.2f} ms")
    print(f"  MoE/layer (synced):      {moe_t[0]:.2f} ms")
    print(f"  LM head:                 {lm_head_t[0]:.2f} ms")
    print(f"  Target:                  30.0 ms")
    print(f"  Gap:                     {step_t[0] - 30:.1f} ms ({step_t[0]/30:.1f}x)")
    print()

    # Weight bandwidth utilization
    # Total weight reads: ~3.75 GB, M1 Max bandwidth: 400 GB/s
    bw_util = 3.75 / (step_t[0] / 1000) / 400 * 100
    print(f"  Bandwidth utilization:   {bw_util:.1f}% of 400 GB/s")
    print(f"  (3.75 GB read in {step_t[0]:.1f} ms = {3.75/(step_t[0]/1000):.0f} GB/s)")


if __name__ == "__main__":
    main()
