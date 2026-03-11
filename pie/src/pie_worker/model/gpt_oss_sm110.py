"""SM110 (Jetson Thor) kernel implementations for GPT-OSS.

FlashInfer CUDA kernels (attention, RoPE, KV cache append) and TRT-LLM's
trtllm_fp4_block_scale_moe are broken on SM110. This module provides torch-based
replacements for attention/RoPE/KV and uses FlashInfer's cutlass_fused_moe in
BF16/BF16 mode with per-layer FP4→BF16 weight dequantization.

SM100 (datacenter Blackwell) code paths are unaffected — these functions are
only called when ``_is_sm110_fallback()`` returns True.
"""

from __future__ import annotations

import os
import torch
import flashinfer as ops  # type: ignore[import]

from .gpt_oss_utils import (
    TUNE_MAX_NUM_TOKENS,
    dequant_mxfp4_to_bf16,
)


def _drop_page_cache() -> None:
    """Drop OS page cache to reclaim unified memory on Jetson.

    On Jetson's unified memory architecture, mmap'd safetensors pages
    accumulate as page cache during weight loading and persist across
    inference. CUDA cannot reclaim this memory, so we must explicitly
    drop caches before large allocations (FP4→BF16 dequantization)
    to prevent OOM during sampling.
    """
    os.system(
        "sync; (echo 3 > /host_drop_caches "
        "|| echo 3 > /proc/sys/vm/drop_caches) 2>/dev/null; true"
    )


def init_dequant_buffers(
    num_experts: int,
    padded_hidden_size: int,
    padded_intermediate_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Allocate BF16 buffers for per-layer FP4→BF16 weight dequantization.

    Returns (w1_bf16, w2_bf16) — reused across layers to avoid repeated allocation.
    """
    w1_bf16 = torch.empty(
        num_experts, 2 * padded_intermediate_size, padded_hidden_size,
        dtype=torch.bfloat16, device=device,
    )
    w2_bf16 = torch.empty(
        num_experts, padded_hidden_size, padded_intermediate_size,
        dtype=torch.bfloat16, device=device,
    )
    return w1_bf16, w2_bf16


def apply_rope_and_append_kv(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    position_ids: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    dim_head: int,
    kv_cache_layer: torch.Tensor,
    kv_page_indices: torch.Tensor,
    kv_page_indptr: torch.Tensor,
    batch_indices: torch.Tensor,
    batch_positions: torch.Tensor,
) -> None:
    """Apply NeoX RoPE in-place and append K/V to paged KV cache.

    Replaces FlashInfer's ``apply_rope_with_cos_sin_cache_inplace`` and
    ``append_paged_kv_cache`` which segfault on SM110.
    """
    n = q.size(0)
    half = dim_head // 2

    # --- RoPE (NeoX style) ---
    pos_ids = position_ids.to(torch.int64)
    cos = cos_sin_cache[pos_ids, :half].unsqueeze(1)
    sin = cos_sin_cache[pos_ids, half:].unsqueeze(1)

    q1, q2 = q[..., :half].float(), q[..., half:].float()
    q[..., :half] = (q1 * cos - q2 * sin).to(q.dtype)
    q[..., half:] = (q2 * cos + q1 * sin).to(q.dtype)

    k1, k2 = k[..., :half].float(), k[..., half:].float()
    k[..., :half] = (k1 * cos - k2 * sin).to(k.dtype)
    k[..., half:] = (k2 * cos + k1 * sin).to(k.dtype)
    del q1, q2, k1, k2, cos, sin

    # --- Append to paged KV cache (vectorized, zero GPU-CPU syncs) ---
    page_size = kv_cache_layer.shape[2]
    pg_offsets = kv_page_indptr[batch_indices]
    pages = kv_page_indices[pg_offsets + batch_positions // page_size]
    slots = batch_positions % page_size
    kv_cache_layer[pages, 0, slots] = k
    kv_cache_layer[pages, 1, slots] = v


def attention_with_sinks(
    q: torch.Tensor,
    kv_cache_layer: torch.Tensor,
    sinks: torch.Tensor,
    layer_idx: int,
    dim_head: int,
    num_q_heads: int,
    num_kv_heads: int,
    tp_size: int,
    sliding_window: int,
    kv_page_indices: torch.Tensor,
    kv_page_indptr: torch.Tensor,
    kv_last_page_lens: torch.Tensor,
    qo_indptr: torch.Tensor,
) -> torch.Tensor:
    """Compute attention with sinks using torch ops.

    Replaces FlashInfer's ``BatchAttentionWithAttentionSinkWrapper.run``
    which crashes on SM110.
    """
    n = q.size(0)
    local_q_heads = num_q_heads // tp_size
    local_kv_heads = num_kv_heads // tp_size
    gqa = local_q_heads // local_kv_heads
    scale = dim_head ** -0.5
    page_size = kv_cache_layer.shape[2]

    is_swa = (layer_idx % 2 == 0)
    win = sliding_window - 1 if is_swa else -1
    batch_sz = kv_page_indptr.size(0) - 1

    # Pre-compute batch metadata on CPU (single bulk transfer, no per-iter syncs)
    token_counts_cpu = (qo_indptr[1:] - qo_indptr[:-1]).cpu().tolist()
    page_indptr_cpu = kv_page_indptr.cpu().tolist()
    last_page_lens_cpu = kv_last_page_lens.cpu().tolist()

    attn_parts = []
    t_off = 0
    for b in range(batch_sz):
        nt = token_counts_cpu[b]
        q_b = q[t_off:t_off + nt]

        # Gather KV from pages (using pre-computed CPU metadata)
        pg_s = page_indptr_cpu[b]
        pg_e = page_indptr_cpu[b + 1]
        pgs = kv_page_indices[pg_s:pg_e]
        sl = (pg_e - pg_s - 1) * page_size + last_page_lens_cpu[b]
        k_s = kv_cache_layer[pgs, 0].reshape(-1, local_kv_heads, dim_head)[:sl]
        v_s = kv_cache_layer[pgs, 1].reshape(-1, local_kv_heads, dim_head)[:sl]

        # GQA expand via view (zero-copy where possible)
        if gqa > 1:
            k_e = k_s.unsqueeze(2).expand(-1, -1, gqa, -1).reshape(sl, local_q_heads, dim_head)
            v_e = v_s.unsqueeze(2).expand(-1, -1, gqa, -1).reshape(sl, local_q_heads, dim_head)
        else:
            k_e, v_e = k_s, v_s

        # Attention scores (float32 for precision)
        Q = q_b.unsqueeze(0).transpose(1, 2).float()
        K = k_e.unsqueeze(0).transpose(1, 2).float()
        V = v_e.unsqueeze(0).transpose(1, 2)
        sc = torch.matmul(Q, K.transpose(-2, -1)) * scale

        # Causal mask (+ sliding window for even layers)
        qp = torch.arange(sl - nt, sl, device=q.device)
        kvp = torch.arange(sl, device=q.device)
        cm = kvp[None, :] > qp[:, None]
        if win >= 0:
            cm = cm | (kvp[None, :] < (qp[:, None] - win))
        sc.masked_fill_(cm[None, None], float('-inf'))

        # Add sink as virtual token (no value, just absorbs weight).
        # Sink acts as a "dummy KV position" that absorbs probability mass
        # in softmax, matching FlashInfer's BatchAttentionWithAttentionSinkWrapper.
        # sinks is already [local_q_heads] (column-sharded per Q head).
        sk = sinks[None, :, None, None].expand(1, -1, nt, 1).float()
        sc_s = torch.cat([sc, sk], dim=-1)
        pr = torch.softmax(sc_s, dim=-1)[:, :, :, :-1]

        o = torch.matmul(pr.to(V.dtype), V)
        attn_parts.append(o.squeeze(0).transpose(0, 1))
        t_off += nt

    attn_output = torch.cat(attn_parts, dim=0)
    return attn_output.reshape(n, -1)


def moe_forward(
    hidden_bf16: torch.Tensor,
    router_logits: torch.Tensor,
    layer_idx: int,
    weights_get,
    experts_per_token: int,
    dequant_pI: int,
    dequant_pH: int,
    w1_bf16: torch.Tensor,
    w2_bf16: torch.Tensor,
    gemm1_alpha: torch.Tensor,
    gemm1_beta: torch.Tensor,
    gemm1_clamp_limit: torch.Tensor,
) -> torch.Tensor:
    """Execute MoE with selective FP4→BF16 dequant + CUTLASS BF16/BF16 kernel.

    Replaces TRT-LLM's ``trtllm_fp4_block_scale_moe`` which produces garbage
    output on SM110.
    """
    topk_weights, topk_ids = torch.topk(
        router_logits.float(), k=experts_per_token, dim=-1
    )
    topk_weights = torch.softmax(topk_weights, dim=-1)

    # Drop page cache before dequant to reclaim unified memory (Jetson only).
    # Without this, mmap'd safetensors pages consume unified memory that CUDA
    # needs for the BF16 dequant buffers and downstream sampling allocations.
    _drop_page_cache()

    # Only dequant the experts actually selected by routing
    _selected = topk_ids.view(-1).unique()
    _w1_fp4 = weights_get(f"layers.{layer_idx}.moe.gemm1_weights")
    _s1_fp4 = weights_get(f"layers.{layer_idx}.moe.gemm1_scales")
    _w2_fp4 = weights_get(f"layers.{layer_idx}.moe.gemm2_weights")
    _s2_fp4 = weights_get(f"layers.{layer_idx}.moe.gemm2_scales")
    dequant_mxfp4_to_bf16(
        _w1_fp4, _s1_fp4,
        2 * dequant_pI, dequant_pH,
        w1_bf16, expert_ids=_selected,
    )
    dequant_mxfp4_to_bf16(
        _w2_fp4, _s2_fp4,
        dequant_pH, dequant_pI,
        w2_bf16, expert_ids=_selected,
    )

    # FlashInfer cutlass_fused_moe: BF16/BF16 fused MoE with SwiGLU activation.
    # Returns Tensor or tuple depending on FlashInfer version; guard handles both.
    output = ops.cutlass_fused_moe(
        input=hidden_bf16,
        token_selected_experts=topk_ids.to(torch.int32),
        token_final_scales=topk_weights.to(torch.float32),
        fc1_expert_weights=w1_bf16,
        fc2_expert_weights=w2_bf16,
        output_dtype=torch.bfloat16,
        quant_scales=[],
        fc1_expert_biases=weights_get(
            f"layers.{layer_idx}.moe.gemm1_bias"
        ).to(torch.bfloat16),
        fc2_expert_biases=weights_get(
            f"layers.{layer_idx}.moe.gemm2_bias"
        ).to(torch.bfloat16),
        swiglu_alpha=gemm1_alpha,
        swiglu_beta=gemm1_beta,
        swiglu_limit=gemm1_clamp_limit,
        tune_max_num_tokens=TUNE_MAX_NUM_TOKENS,
    )
    if isinstance(output, (list, tuple)):
        output = output[0]
    return output
