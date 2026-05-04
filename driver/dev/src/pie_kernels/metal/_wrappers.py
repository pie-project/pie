"""FlashInfer-compatible public API for Metal backend.

Provides drop-in replacements for FlashInfer wrapper classes and functions.
"""

from typing import Optional, Tuple

import torch

from ._compiler import MetalCompiler, _validate_mps_device


# ---------------------------------------------------------------------------
# Wrapper classes (plan/run pattern matching FlashInfer API)
# ---------------------------------------------------------------------------


class BatchPrefillWithPagedKVCacheWrapper:
    """Drop-in replacement for flashinfer.BatchPrefillWithPagedKVCacheWrapper."""

    def __init__(self, workspace_buffer: torch.Tensor, kv_layout: str = "NHD", **kwargs):
        self._planned: Optional[dict] = None

    def plan(
        self,
        qo_indptr: torch.Tensor,
        paged_kv_indptr: torch.Tensor,
        paged_kv_indices: torch.Tensor,
        paged_kv_last_page_len: torch.Tensor,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim_qk: int,
        page_size: int,
        pos_encoding_mode: str = "NONE",
        custom_mask: Optional[torch.Tensor] = None,
        q_data_type: torch.dtype = torch.float16,
    ) -> None:
        _validate_mps_device(qo_indptr, "qo_indptr")
        _validate_mps_device(paged_kv_indptr, "paged_kv_indptr")
        _validate_mps_device(paged_kv_indices, "paged_kv_indices")
        _validate_mps_device(paged_kv_last_page_len, "paged_kv_last_page_len")
        if custom_mask is not None:
            _validate_mps_device(custom_mask, "custom_mask")

        self._planned = {
            "qo_indptr": qo_indptr,
            "kv_page_indptr": paged_kv_indptr,
            "kv_page_indices": paged_kv_indices,
            "kv_last_page_lens": paged_kv_last_page_len,
        }

    def run(self, query: torch.Tensor, kv_cache: torch.Tensor) -> torch.Tensor:
        if self._planned is None:
            raise RuntimeError("Must call plan() before run()")
        _validate_mps_device(query, "query")
        _validate_mps_device(kv_cache, "kv_cache")
        return MetalCompiler().run_attention(
            query=query,
            kv_cache=kv_cache,
            kv_page_indices=self._planned["kv_page_indices"],
            kv_page_indptr=self._planned["kv_page_indptr"],
            kv_last_page_lens=self._planned["kv_last_page_lens"],
            qo_indptr=self._planned["qo_indptr"],
        )


class BatchDecodeWithPagedKVCacheWrapper:
    """Drop-in replacement for flashinfer.BatchDecodeWithPagedKVCacheWrapper."""

    def __init__(self, workspace_buffer: torch.Tensor, kv_layout: str = "NHD", **kwargs):
        self._planned: Optional[dict] = None

    def plan(
        self,
        indptr: torch.Tensor,
        indices: torch.Tensor,
        last_page_len: torch.Tensor,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        page_size: int,
        pos_encoding_mode: str = "NONE",
        q_data_type: torch.dtype = torch.float16,
    ) -> None:
        _validate_mps_device(indptr, "indptr")
        _validate_mps_device(indices, "indices")
        _validate_mps_device(last_page_len, "last_page_len")

        self._planned = {
            "kv_page_indptr": indptr,
            "kv_page_indices": indices,
            "kv_last_page_lens": last_page_len,
        }

    def run(self, query: torch.Tensor, kv_cache: torch.Tensor) -> torch.Tensor:
        if self._planned is None:
            raise RuntimeError("Must call plan() before run()")
        _validate_mps_device(query, "query")
        _validate_mps_device(kv_cache, "kv_cache")

        batch_size = self._planned["kv_page_indptr"].shape[0] - 1
        qo_indptr = torch.arange(batch_size + 1, dtype=torch.int32, device=query.device)

        return MetalCompiler().run_attention(
            query=query,
            kv_cache=kv_cache,
            kv_page_indices=self._planned["kv_page_indices"],
            kv_page_indptr=self._planned["kv_page_indptr"],
            kv_last_page_lens=self._planned["kv_last_page_lens"],
            qo_indptr=qo_indptr,
        )


class BatchAttentionWithAttentionSinkWrapper:
    """Drop-in replacement for FlashInfer's BatchAttentionWithAttentionSinkWrapper.

    Handles attention with sink tokens prepended to the KV context.
    Delegates to MetalCompiler().run_attention() for the main computation.
    """

    def __init__(
        self,
        float_workspace_buffer: torch.Tensor,
        kv_layout: str = "NHD",
        window_left: int = -1,
        q_data_type: torch.dtype = torch.float16,
        kv_data_type: torch.dtype = torch.float16,
        head_dim_qk: int = 128,
        head_dim_vo: int = 128,
    ):
        self._window_left = window_left
        self._q_data_type = q_data_type
        self._kv_data_type = kv_data_type
        self._head_dim_qk = head_dim_qk
        self._head_dim_vo = head_dim_vo
        self._planned: Optional[dict] = None

    def plan(
        self,
        qo_indptr: torch.Tensor,
        paged_kv_indptr: torch.Tensor,
        paged_kv_indices: torch.Tensor,
        paged_kv_last_page_len: torch.Tensor,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        page_size: int,
        causal: bool = True,
        window_left: int = -1,
        q_data_type: torch.dtype = torch.float16,
        kv_data_type: torch.dtype = torch.float16,
        non_blocking: bool = False,
    ) -> None:
        _validate_mps_device(qo_indptr, "qo_indptr")
        _validate_mps_device(paged_kv_indptr, "paged_kv_indptr")
        _validate_mps_device(paged_kv_indices, "paged_kv_indices")
        _validate_mps_device(paged_kv_last_page_len, "paged_kv_last_page_len")

        self._planned = {
            "qo_indptr": qo_indptr,
            "kv_page_indptr": paged_kv_indptr,
            "kv_page_indices": paged_kv_indices,
            "kv_last_page_lens": paged_kv_last_page_len,
            "num_qo_heads": num_qo_heads,
            "num_kv_heads": num_kv_heads,
            "head_dim": head_dim,
            "page_size": page_size,
            "causal": causal,
            "window_left": window_left,
        }

    def run(
        self,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        sinks: Optional[torch.Tensor] = None,
        scaling: Optional[float] = None,
    ) -> torch.Tensor:
        """Run attention with optional attention sink logits.

        Args:
            query: [num_tokens, num_heads, head_dim]
            kv_cache: paged KV cache tensor.
            sinks: optional [num_qo_heads] float32 tensor of per-head sink logits.
                   Injects a virtual sink token into the softmax denominator.
            scaling: attention scale factor. If None, uses 1/sqrt(head_dim).

        Returns:
            [num_tokens, num_heads * head_dim] attention output.
        """
        if self._planned is None:
            raise RuntimeError("Must call plan() before run()")
        _validate_mps_device(query, "query")
        _validate_mps_device(kv_cache, "kv_cache")

        return MetalCompiler().run_attention(
            query=query,
            kv_cache=kv_cache,
            kv_page_indices=self._planned["kv_page_indices"],
            kv_page_indptr=self._planned["kv_page_indptr"],
            kv_last_page_lens=self._planned["kv_last_page_lens"],
            qo_indptr=self._planned["qo_indptr"],
            sinks=sinks,
            scaling=scaling,
        )


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def apply_rope_pos_ids_inplace(
    q: torch.Tensor,
    k: torch.Tensor,
    pos_ids: torch.Tensor,
    rope_theta: float = 10000.0,
    interleave: bool = False,
) -> None:
    """Apply standard RoPE encoding in-place."""
    _validate_mps_device(q, "q")
    _validate_mps_device(k, "k")
    _validate_mps_device(pos_ids, "pos_ids")
    compiler = MetalCompiler()
    compiler.run_rope(q, pos_ids, rope_theta=rope_theta, interleaved=interleave)
    compiler.run_rope(k, pos_ids, rope_theta=rope_theta, interleaved=interleave)


def apply_llama31_rope_pos_ids_inplace(
    q: torch.Tensor,
    k: torch.Tensor,
    pos_ids: torch.Tensor,
    rotary_dim: Optional[int] = None,
    interleave: bool = False,
    rope_scale: float = 32.0,
    rope_theta: float = 500000.0,
    low_freq_factor: float = 1.0,
    high_freq_factor: float = 4.0,
    old_context_len: int = 8192,
) -> None:
    """Apply LLaMA 3.1-style RoPE encoding in-place."""
    if rotary_dim is not None:
        raise ValueError("rotary_dim not supported in Metal RoPE")
    if low_freq_factor != 1.0:
        raise ValueError("low_freq_factor not supported in Metal RoPE")
    if high_freq_factor != 4.0:
        raise ValueError("high_freq_factor not supported in Metal RoPE")
    if old_context_len != 8192:
        raise ValueError("old_context_len not supported in Metal RoPE")

    _validate_mps_device(q, "q")
    _validate_mps_device(k, "k")
    _validate_mps_device(pos_ids, "pos_ids")
    compiler = MetalCompiler()
    compiler.run_rope(
        q, pos_ids, rope_theta=rope_theta, rope_factor=rope_scale, interleaved=interleave
    )
    compiler.run_rope(
        k, pos_ids, rope_theta=rope_theta, rope_factor=rope_scale, interleaved=interleave
    )


def apply_rope_with_cos_sin_cache_inplace(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    head_size: int,
    cos_sin_cache: torch.Tensor,
    is_neox: bool = True,
) -> None:
    """Apply RoPE using a precomputed cos/sin cache, in-place (Metal kernel).

    Uses a fused kernel that processes both Q and K in a single dispatch,
    halving dispatch overhead (24 dispatches/step instead of 48).

    Args:
        positions: [num_tokens] int32/int64 position indices.
        query: [num_tokens, num_heads, head_dim] or [num_tokens, num_heads * head_dim].
        key: same shape as query.
        head_size: dimension of each attention head (= head_dim).
        cos_sin_cache: [max_pos, head_dim] — first half cols are cos, second half sin.
        is_neox: True for GPT-NeoX style (non-interleaved halves).
    """
    _validate_mps_device(query, "query")
    _validate_mps_device(key, "key")
    _validate_mps_device(positions, "positions")

    compiler = MetalCompiler()

    # Reshape to 3D if needed
    q_3d = query.view(query.shape[0], -1, head_size) if query.ndim == 2 else query
    k_3d = key.view(key.shape[0], -1, head_size) if key.ndim == 2 else key

    compiler.run_rope_cos_sin_fused(q_3d, k_3d, positions, cos_sin_cache, head_size, is_neox)

    if query.ndim == 2 and query.data_ptr() != q_3d.data_ptr():
        query.copy_(q_3d.view(query.shape))
    if key.ndim == 2 and key.data_ptr() != k_3d.data_ptr():
        key.copy_(k_3d.view(key.shape))


def mm_fp8(
    input: torch.Tensor,
    weight: torch.Tensor,
    alpha: float,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """FP8 matrix multiplication fallback: dequant to bf16 + matmul.

    Apple Silicon lacks FP8 hardware, so we dequantize to the output dtype
    and perform a standard matmul.

    Args:
        input: [m, k] float8_e4m3fn activations.
        weight: [n, k] float8 weights.
        alpha: scalar scale factor.
        out_dtype: output dtype (default bfloat16).

    Returns:
        [m, n] tensor in out_dtype.
    """
    return (input.to(out_dtype) @ weight.to(out_dtype).T) * alpha


def _to_scalar(v, idx=None):
    """Extract a Python float from a scalar, 0-d tensor, or per-expert 1-d tensor."""
    if isinstance(v, (int, float)):
        return float(v)
    if v.dim() == 0:
        return v.item()
    if idx is not None:
        return v[idx].item()
    return v[0].item()


def trtllm_fp4_block_scale_moe(
    *,
    routing_logits: torch.Tensor,
    routing_bias: Optional[torch.Tensor],
    hidden_states: torch.Tensor,
    hidden_states_scale: Optional[torch.Tensor],
    gemm1_weights: torch.Tensor,
    gemm1_weights_scale: torch.Tensor,
    gemm1_bias: Optional[torch.Tensor],
    gemm1_alpha: float,
    gemm1_beta: float,
    gemm1_clamp_limit: float,
    gemm2_weights: torch.Tensor,
    gemm2_weights_scale: torch.Tensor,
    gemm2_bias: Optional[torch.Tensor],
    output1_scale_scalar: float,
    output1_scale_gate_scalar: float,
    output2_scale_scalar: float,
    num_experts: int,
    top_k: int,
    n_group: Optional[int],
    topk_group: Optional[int],
    intermediate_size: int,
    local_expert_offset: int,
    local_num_experts: int,
    routed_scaling_factor: Optional[float],
    routing_method_type: int,
    gated_act_type: int,
    do_finalize: bool,
    tune_max_num_tokens: int,
    profiler=None,
) -> Tuple[torch.Tensor]:
    """MoE for Apple Silicon with FP4 packed weights via Metal kernels.

    Requires FP4 packed (uint8) weights on an MPS device.

    Returns:
        Tuple where [0] is the output tensor of shape [num_tokens, hidden_dim].
    """
    dtype = hidden_states.dtype
    device = hidden_states.device
    num_tokens, hidden_dim = hidden_states.shape

    if gemm1_weights.is_floating_point() or device.type != "mps":
        raise RuntimeError(
            "trtllm_fp4_block_scale_moe requires FP4 packed (uint8) weights on MPS"
        )

    def _sync_record(name):
        if profiler is not None:
            torch.mps.synchronize()
            profiler.record(name)

    # --- Metal kernel path ---
    compiler = MetalCompiler()

    if num_tokens == 1:
        # --- Batched decode fast-path: all K experts in 2 dispatches ---
        # Fused routing: softmax + topk + normalize + scale in ONE Metal dispatch
        # (replaces ~10 PyTorch dispatches)

        # Handle routing bias (rare; None for most models)
        if routing_bias is not None:
            route_logits = routing_logits.float() + routing_bias.float()
            route_logits = route_logits.to(routing_logits.dtype)
        else:
            route_logits = routing_logits

        # Lazily allocate decode routing buffers on MetalCompiler singleton
        if not hasattr(compiler, '_decode_expert_ids') or compiler._decode_expert_ids is None \
                or compiler._decode_expert_ids.shape[0] != top_k:
            compiler._decode_expert_ids = torch.empty(top_k, dtype=torch.int32, device=device)
            compiler._decode_fused_scales = torch.empty(top_k, dtype=torch.float32, device=device)

        # Compute output2_scale as a scalar float for the fused kernel
        if isinstance(output2_scale_scalar, (int, float)):
            o2_scale = float(output2_scale_scalar)
        else:
            o2_scale = float(output2_scale_scalar)

        compiler.run_moe_route_topk(
            logits=route_logits,
            expert_ids_out=compiler._decode_expert_ids,
            fused_scales_out=compiler._decode_fused_scales,
            num_experts=num_experts,
            top_k=top_k,
            output2_scale=o2_scale,
            local_expert_offset=local_expert_offset,
        )
        _sync_record("moe_routing")

        local_ids = compiler._decode_expert_ids
        fused_scales = compiler._decode_fused_scales

        # Decode GEMM1 with fused SwiGLU → [K, I]
        # Uses SIMD-parallel K-split for 2.7× speedup over previous kernels.
        # Scalar params only (no per-expert lists needed)
        alpha = float(gemm1_alpha) if isinstance(gemm1_alpha, (int, float)) else float(gemm1_alpha)
        clamp_l = float(gemm1_clamp_limit) if isinstance(gemm1_clamp_limit, (int, float)) else float(gemm1_clamp_limit)

        activated = compiler.run_moe_decode_gemm1_swiglu(
            input=hidden_states,  # [1, H]
            all_w_blocks=gemm1_weights,
            all_w_scales=gemm1_weights_scale,
            all_bias=gemm1_bias,
            intermediate_size=intermediate_size,
            expert_ids=local_ids,
            alpha=alpha,
            clamp_limit=clamp_l,
        )
        _sync_record("moe_gemm1")

        # Decode GEMM2 fused across experts → [1, H] float32
        output_f32 = compiler.run_moe_decode_gemm2_fused(
            input=activated,  # [K, I]
            all_w_blocks=gemm2_weights,
            all_w_scales=gemm2_weights_scale,
            all_bias=gemm2_bias,
            out_dim=hidden_dim,
            expert_ids=local_ids,
            fused_scales=fused_scales,
        )
        _sync_record("moe_gemm2")
    else:
        # --- Prefill path: per-expert FP4 GEMM with inline dequant ---
        # PyTorch routing for prefill (multi-token)
        logits = routing_logits.float()
        if routing_bias is not None:
            logits = logits + routing_bias.float()
        scores = torch.softmax(logits, dim=-1)
        topk_weights, topk_indices = torch.topk(scores, top_k, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        topk_weights = topk_weights.to(dtype)

        # Routing prep: sort/bincount/cumsum for batched per-expert processing
        flat_expert_ids = topk_indices.view(-1)  # [num_tokens * top_k]
        flat_weights = topk_weights.view(-1)     # [num_tokens * top_k]
        flat_token_ids = torch.arange(num_tokens, device=device).unsqueeze(1).expand(
            -1, top_k
        ).reshape(-1)  # [num_tokens * top_k]

        sorted_order = flat_expert_ids.argsort()
        sorted_expert_ids = flat_expert_ids[sorted_order]
        sorted_token_ids = flat_token_ids[sorted_order]
        sorted_weights = flat_weights[sorted_order]

        expert_counts = torch.bincount(
            sorted_expert_ids, minlength=local_expert_offset + local_num_experts
        )
        expert_offsets = torch.zeros(
            local_num_experts + 1, dtype=torch.int64, device=device
        )
        local_counts = expert_counts[local_expert_offset:local_expert_offset + local_num_experts]
        expert_offsets[1:] = local_counts.cumsum(0)
        base_offset = expert_counts[:local_expert_offset].sum().item()
        _sync_record("moe_routing")

        # Accumulate into float32 for accuracy, convert at end
        output_f32 = torch.zeros(num_tokens, hidden_dim, dtype=torch.float32, device=device)

        for local_idx in range(local_num_experts):
            start = base_offset + expert_offsets[local_idx].item()
            end = base_offset + expert_offsets[local_idx + 1].item()
            if start == end:
                continue

            token_ids = sorted_token_ids[start:end]
            expert_w = sorted_weights[start:end]
            x = hidden_states[token_ids]  # [count, hidden_dim]

            # Per-expert scalar params (may be float or per-expert tensor)
            expert_idx = local_expert_offset + local_idx

            # GEMM1 with fused SwiGLU via Metal kernel
            activated = compiler.run_moe_prefill_gemm1(
                input=x,
                w_blocks=gemm1_weights[local_idx],
                w_scales=gemm1_weights_scale[local_idx],
                bias=gemm1_bias[local_idx] if gemm1_bias is not None else None,
                intermediate_size=intermediate_size,
                alpha=_to_scalar(gemm1_alpha, expert_idx),
                beta=_to_scalar(gemm1_beta, expert_idx),
                clamp_limit=_to_scalar(gemm1_clamp_limit, expert_idx),
                scale_gate=_to_scalar(output1_scale_gate_scalar, expert_idx),
                scale_up=_to_scalar(output1_scale_scalar, expert_idx),
            )
            _sync_record("moe_gemm1")

            # GEMM2 via Metal kernel
            g2 = compiler.run_moe_prefill_gemm2(
                input=activated,
                w_blocks=gemm2_weights[local_idx],
                w_scales=gemm2_weights_scale[local_idx],
                bias=gemm2_bias[local_idx] if gemm2_bias is not None else None,
                out_dim=hidden_dim,
                scale=_to_scalar(output2_scale_scalar, expert_idx),
            )
            _sync_record("moe_gemm2")

            # Weighted scatter-add back (PyTorch — small relative to GEMM cost)
            output_f32.index_add_(
                0, token_ids, (g2 * expert_w.unsqueeze(-1)).float(),
            )
            _sync_record("moe_scatter")

    output = output_f32.to(dtype)

    if routed_scaling_factor is not None:
        output = output * routed_scaling_factor

    return (output,)


def append_paged_kv_cache(
    append_key: torch.Tensor,
    append_value: torch.Tensor,
    batch_indices: torch.Tensor,
    positions: torch.Tensor,
    paged_kv_cache: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_last_page_len: torch.Tensor,
    kv_layout: str = "NHD",
) -> None:
    """Append key-value states to paged KV cache."""
    _validate_mps_device(append_key, "append_key")
    _validate_mps_device(append_value, "append_value")
    _validate_mps_device(batch_indices, "batch_indices")
    _validate_mps_device(positions, "positions")
    _validate_mps_device(paged_kv_cache, "paged_kv_cache")
    _validate_mps_device(kv_indices, "kv_indices")
    _validate_mps_device(kv_indptr, "kv_indptr")
    _validate_mps_device(kv_last_page_len, "kv_last_page_len")

    num_tokens, num_kv_heads, head_dim = append_key.shape
    _num_pages, _, page_size, _, _ = paged_kv_cache.shape

    k_flat = append_key.contiguous().reshape(num_tokens, num_kv_heads * head_dim)
    v_flat = append_value.contiguous().reshape(num_tokens, num_kv_heads * head_dim)

    MetalCompiler().run_append_kv(
        k_flat,
        v_flat,
        paged_kv_cache.view(-1),
        batch_indices,
        positions,
        kv_indices,
        kv_indptr,
        kv_last_page_len,
        num_kv_heads,
        head_dim,
        page_size,
    )


# ---------------------------------------------------------------------------
# Utility functions (vectorized)
# ---------------------------------------------------------------------------


def get_seq_lens(
    kv_page_indptr: torch.Tensor,
    kv_last_page_lens: torch.Tensor,
    page_size: int,
) -> torch.Tensor:
    """Calculate sequence lengths from paging metadata (vectorized)."""
    _validate_mps_device(kv_page_indptr, "kv_page_indptr")
    _validate_mps_device(kv_last_page_lens, "kv_last_page_lens")

    num_pages = kv_page_indptr[1:] - kv_page_indptr[:-1]
    return torch.where(
        num_pages > 0,
        (num_pages - 1) * page_size + kv_last_page_lens,
        torch.zeros_like(num_pages),
    ).to(torch.int32)


def get_batch_indices_positions(
    append_indptr: torch.Tensor,
    seq_lens: torch.Tensor,
    nnz: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Get per-token batch indices and positions (vectorized)."""
    _validate_mps_device(append_indptr, "append_indptr")
    _validate_mps_device(seq_lens, "seq_lens")

    device = append_indptr.device
    counts = (append_indptr[1:] - append_indptr[:-1]).to(torch.int64)
    batch_size = counts.shape[0]

    # Batch indices: repeat each batch_idx by its token count
    batch_indices = torch.repeat_interleave(
        torch.arange(batch_size, dtype=torch.int32, device=device), counts
    )

    # Positions: for each token, compute its position within the sequence
    offsets = torch.arange(nnz, dtype=torch.int32, device=device)
    segment_starts = torch.repeat_interleave(
        append_indptr[:-1].to(torch.int32), counts
    )
    local_offsets = offsets - segment_starts

    num_new = counts.to(torch.int32)
    pos_starts = (seq_lens - num_new).to(torch.int32)
    pos_start_per_token = torch.repeat_interleave(pos_starts, counts)
    positions = pos_start_per_token + local_offsets

    return batch_indices, positions
