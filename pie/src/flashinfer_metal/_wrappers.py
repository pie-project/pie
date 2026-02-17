"""FlashInfer-compatible public API for Metal backend.

Provides drop-in replacements for FlashInfer wrapper classes and functions.
"""

from typing import Optional, Tuple

import torch
import torch.nn.functional as F

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
        """Run attention with optional attention sink tokens.

        Args:
            query: [num_tokens, num_heads, head_dim]
            kv_cache: paged KV cache tensor.
            sinks: optional sink token KV tensor. Ignored in current Metal
                   implementation (no native support).
            scaling: attention scale factor. Ignored (Metal kernel computes its own).

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
    """Apply RoPE using a precomputed cos/sin cache, in-place.

    Args:
        positions: [num_tokens] int32/int64 position indices.
        query: [num_tokens, num_heads, head_dim] or [num_tokens, num_heads * head_dim].
        key: same shape as query.
        head_size: dimension of each attention head (= head_dim).
        cos_sin_cache: [max_pos, head_dim] — first half cols are cos, second half sin.
        is_neox: True for GPT-NeoX style (non-interleaved halves).
    """
    half_dim = head_size // 2

    # Extract cos and sin for the given positions
    cos = cos_sin_cache[positions.long(), :half_dim]  # [num_tokens, half_dim]
    sin = cos_sin_cache[positions.long(), half_dim:]   # [num_tokens, half_dim]

    def _apply(x: torch.Tensor) -> None:
        orig_shape = x.shape
        if x.ndim == 2:
            # [num_tokens, num_heads * head_dim] -> [num_tokens, num_heads, head_dim]
            x_3d = x.view(x.shape[0], -1, head_size)
        else:
            x_3d = x

        # cos/sin: [num_tokens, 1, half_dim] for broadcasting over heads
        c = cos.unsqueeze(1).to(x_3d.dtype)
        s = sin.unsqueeze(1).to(x_3d.dtype)

        if is_neox:
            # GPT-NeoX: split into first half and second half
            x1 = x_3d[..., :half_dim]
            x2 = x_3d[..., half_dim:]
            r1 = x1 * c - x2 * s
            r2 = x2 * c + x1 * s
            x_3d[..., :half_dim] = r1
            x_3d[..., half_dim:] = r2
        else:
            # Interleaved: pairs (x[0], x[1]), (x[2], x[3]), ...
            x1 = x_3d[..., 0::2]
            x2 = x_3d[..., 1::2]
            r1 = x1 * c - x2 * s
            r2 = x2 * c + x1 * s
            x_3d[..., 0::2] = r1
            x_3d[..., 1::2] = r2

        if x.ndim == 2 and orig_shape != x_3d.shape:
            x.copy_(x_3d.view(orig_shape))

    _apply(query)
    _apply(key)


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
) -> Tuple[torch.Tensor]:
    """Decomposed PyTorch MoE fallback for TensorRT-LLM FP4 block-scale MoE.

    Apple Silicon lacks FP4 hardware. This provides a functional (not optimized)
    fallback that dequantizes weights and performs the MoE computation in bf16.

    Returns:
        Tuple where [0] is the output tensor of shape [num_tokens, hidden_dim].
    """
    dtype = hidden_states.dtype
    device = hidden_states.device
    num_tokens, hidden_dim = hidden_states.shape

    # --- Routing ---
    logits = routing_logits.float()
    if routing_bias is not None:
        logits = logits + routing_bias.float()

    # routing_method_type=1 => TopK then Softmax (renormalize)
    scores = torch.softmax(logits, dim=-1)
    topk_weights, topk_indices = torch.topk(scores, top_k, dim=-1)
    # Renormalize top-k weights
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    topk_weights = topk_weights.to(dtype)

    # --- Dequantize weights ---
    # Weights may be packed FP4 (int8 with block scales) or pre-dequantized.
    # We handle both: if integral dtype, multiply by scale; otherwise use as-is.
    def _dequant(w: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        if w.is_floating_point():
            return w.to(dtype)
        # Packed FP4 in int8: treat each byte as a scaled value
        return w.to(dtype) * scale.to(dtype)

    # gemm1_weights: [num_experts, 2 * intermediate_size, hidden_dim] (gate+up fused)
    # gemm2_weights: [num_experts, hidden_dim, intermediate_size]
    w1 = _dequant(gemm1_weights, gemm1_weights_scale)
    w2 = _dequant(gemm2_weights, gemm2_weights_scale)

    # --- Per-token expert computation ---
    output = torch.zeros(num_tokens, hidden_dim, dtype=dtype, device=device)

    for k_idx in range(top_k):
        expert_ids = topk_indices[:, k_idx]   # [num_tokens]
        weights = topk_weights[:, k_idx]      # [num_tokens]

        for expert_id in range(local_expert_offset, local_expert_offset + local_num_experts):
            mask = expert_ids == expert_id
            if not mask.any():
                continue

            x = hidden_states[mask]  # [sel, hidden_dim]
            local_idx = expert_id - local_expert_offset

            # GEMM1: gate+up projection
            g1 = F.linear(x, w1[local_idx])  # [sel, 2 * intermediate_size]
            if gemm1_bias is not None:
                g1 = g1 + gemm1_bias[local_idx]

            # Apply output1 scales
            gate_part = g1[:, :intermediate_size] * output1_scale_gate_scalar
            up_part = g1[:, intermediate_size:] * output1_scale_scalar

            # SwiGLU activation (gated_act_type=0)
            activated = (F.silu(gate_part * gemm1_alpha + gemm1_beta)).clamp(
                -gemm1_clamp_limit, gemm1_clamp_limit
            ) * up_part

            # GEMM2: down projection
            g2 = F.linear(activated, w2[local_idx])  # [sel, hidden_dim]
            if gemm2_bias is not None:
                g2 = g2 + gemm2_bias[local_idx]
            g2 = g2 * output2_scale_scalar

            # Weighted contribution
            output[mask] += g2 * weights[mask].unsqueeze(-1)

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
