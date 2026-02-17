"""GPT-OSS Utility Components

This module contains utility functions, constants, and helper layers
used by the GPT-OSS model architecture.

Ported from backend-python-legacy/model/gptoss_utils.py
"""

from __future__ import annotations

import math
from typing import Any

import torch

from ..utils import is_apple_silicon


# Mapping from fp4 (e2m1) to float values
FP4_VALUES = (
    +0.0,
    +0.5,
    +1.0,
    +1.5,
    +2.0,
    +3.0,
    +4.0,
    +6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)

# Alignment requirement for `trtllm_fp4_block_scale_moe`
ALIGNMENT = 256

# Max num tokens to tune for trtllm-gen fused moe
TUNE_MAX_NUM_TOKENS = 4096


def pad_to_multiple(size: int, multiple: int = ALIGNMENT) -> int:
    """Calculate padded size to be a multiple of the given number."""
    if size % multiple == 0:
        return size
    return ((size + multiple - 1) // multiple) * multiple


def deinterleave_gate_up_weights(weights: torch.Tensor) -> torch.Tensor:
    """
    De-interleave gate_up weights from GPT OSS format to FlashInfer format.

    GPT OSS stores gate_up as interleaved: [gate[0], linear[0], gate[1], linear[1], ...]
    FlashInfer expects non-interleaved: [linear[0], linear[1], ..., gate[0], gate[1], ...]

    Args:
        weights: Input tensor of shape [num_experts, 2*intermediate_size, hidden_size]
                 where even columns are gate, odd columns are linear (interleaved)

    Returns:
        De-interleaved tensor of shape [num_experts, 2*intermediate_size, hidden_size]
        where first half is linear, second half is gate
    """
    # Extract interleaved parts: gate at even indices, linear at odd indices
    gate_part = weights[:, 0::2, :]  # [num_experts, intermediate_size, hidden_size]
    linear_part = weights[:, 1::2, :]  # [num_experts, intermediate_size, hidden_size]

    # Concatenate: linear first, gate second (FlashInfer format)
    return torch.cat([linear_part, gate_part], dim=1)


def deinterleave_gate_up_bias(bias: torch.Tensor) -> torch.Tensor:
    """
    De-interleave gate_up bias from GPT OSS format to FlashInfer format.

    Args:
        bias: Input tensor of shape [num_experts, 2*intermediate_size]
              where even indices are gate, odd indices are linear (interleaved)

    Returns:
        De-interleaved tensor of shape [num_experts, 2*intermediate_size]
        where first half is linear, second half is gate
    """
    # Extract interleaved parts
    gate_part = bias[:, 0::2]  # [num_experts, intermediate_size]
    linear_part = bias[:, 1::2]  # [num_experts, intermediate_size]

    # Concatenate: linear first, gate second (FlashInfer format)
    return torch.cat([linear_part, gate_part], dim=1)


def pad_gate_up_weights(
    weights: torch.Tensor,
    padded_hidden_size: int,
    padded_intermediate_size: int,
) -> torch.Tensor:
    """
    Pad gate_up fused weights from [num_experts, 2*intermediate_size, hidden_size]
    to [num_experts, 2*padded_intermediate_size, padded_hidden_size].

    IMPORTANT: The first half is the linear part, the second half is the gate part.
    We need to pad each part separately, NOT pad all at the end of the fused matrix.
    """
    num_experts, fused_size, hidden_size = weights.shape
    intermediate_size = fused_size // 2

    if (
        hidden_size == padded_hidden_size
        and intermediate_size == padded_intermediate_size
    ):
        return weights

    # Split into linear and gate parts
    linear_part = weights[:, :intermediate_size, :]
    gate_part = weights[:, intermediate_size:, :]

    # Pad each part separately
    padded_linear = torch.zeros(
        (num_experts, padded_intermediate_size, padded_hidden_size),
        dtype=weights.dtype,
        device=weights.device,
    )
    padded_linear[:, :intermediate_size, :hidden_size] = linear_part

    padded_gate = torch.zeros(
        (num_experts, padded_intermediate_size, padded_hidden_size),
        dtype=weights.dtype,
        device=weights.device,
    )
    padded_gate[:, :intermediate_size, :hidden_size] = gate_part

    # Concatenate back: [linear_padded | gate_padded]
    return torch.cat([padded_linear, padded_gate], dim=1)


def pad_down_weights(
    weights: torch.Tensor,
    padded_hidden_size: int,
    padded_intermediate_size: int,
) -> torch.Tensor:
    """
    Pad down projection weights from [num_experts, hidden_size, intermediate_size]
    to [num_experts, padded_hidden_size, padded_intermediate_size].
    """
    num_experts, hidden_size, intermediate_size = weights.shape

    if (
        hidden_size == padded_hidden_size
        and intermediate_size == padded_intermediate_size
    ):
        return weights

    padded = torch.zeros(
        (num_experts, padded_hidden_size, padded_intermediate_size),
        dtype=weights.dtype,
        device=weights.device,
    )
    padded[:, :hidden_size, :intermediate_size] = weights
    return padded


def pad_gate_up_bias(
    bias: torch.Tensor,
    padded_intermediate_size: int,
) -> torch.Tensor:
    """
    Pad gate_up bias from [num_experts, 2*intermediate_size]
    to [num_experts, 2*padded_intermediate_size].

    IMPORTANT: Same as gate_up weights, pad each part separately.
    """
    num_experts, fused_size = bias.shape
    intermediate_size = fused_size // 2

    if intermediate_size == padded_intermediate_size:
        return bias

    # Split into linear and gate parts
    linear_part = bias[:, :intermediate_size]
    gate_part = bias[:, intermediate_size:]

    # Pad each part separately
    padded_linear = torch.zeros(
        (num_experts, padded_intermediate_size),
        dtype=bias.dtype,
        device=bias.device,
    )
    padded_linear[:, :intermediate_size] = linear_part

    padded_gate = torch.zeros(
        (num_experts, padded_intermediate_size),
        dtype=bias.dtype,
        device=bias.device,
    )
    padded_gate[:, :intermediate_size] = gate_part

    # Concatenate back: [linear_padded | gate_padded]
    return torch.cat([padded_linear, padded_gate], dim=1)


def pad_down_bias(
    bias: torch.Tensor,
    padded_hidden_size: int,
) -> torch.Tensor:
    """
    Pad down projection bias from [num_experts, hidden_size]
    to [num_experts, padded_hidden_size].
    """
    num_experts, hidden_size = bias.shape

    if hidden_size == padded_hidden_size:
        return bias

    padded = torch.zeros(
        (num_experts, padded_hidden_size),
        dtype=bias.dtype,
        device=bias.device,
    )
    padded[:, :hidden_size] = bias
    return padded


def deinterleave_gate_up_fp4(
    blocks: torch.Tensor,
    scales: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    De-interleave gate_up in FP4 packed domain (no dequant needed).

    GPT OSS stores gate_up as interleaved rows: [gate[0], linear[0], gate[1], linear[1], ...]
    FlashInfer expects: [linear[0], linear[1], ..., gate[0], gate[1], ...]

    This is a pure row permutation — works identically on uint8 packed data.

    Args:
        blocks: [E, 2*I, H/2] uint8 packed FP4
        scales: [E, 2*I, H/32] uint8 E8M0 exponents

    Returns:
        Reordered (blocks, scales) with same shapes
    """
    gate_blocks = blocks[:, 0::2, :]
    linear_blocks = blocks[:, 1::2, :]
    reordered_blocks = torch.cat([linear_blocks, gate_blocks], dim=1)

    gate_scales = scales[:, 0::2, :]
    linear_scales = scales[:, 1::2, :]
    reordered_scales = torch.cat([linear_scales, gate_scales], dim=1)

    return reordered_blocks, reordered_scales


def pad_gate_up_fp4(
    blocks: torch.Tensor,
    scales: torch.Tensor,
    padded_H: int,
    padded_I: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pad gate_up FP4 packed blocks and scales.

    Splits into linear/gate halves, pads each separately, re-concatenates.
    Blocks padded with 0x00 (FP4 +0.0), scales padded with 0x7F (E8M0 exp=127 → scale=1.0).

    Args:
        blocks: [E, 2*I, H/2] uint8
        scales: [E, 2*I, H/32] uint8
        padded_H: target hidden size (must be even for blocks, divisible by 32 for scales)
        padded_I: target intermediate size per half

    Returns:
        blocks [E, 2*padded_I, padded_H/2], scales [E, 2*padded_I, padded_H/32]
    """
    E = blocks.shape[0]
    I = blocks.shape[1] // 2
    H_half = blocks.shape[2]          # H/2
    H_scale = scales.shape[2]         # H/32

    padded_H_half = padded_H // 2
    padded_H_scale = padded_H // 32

    if I == padded_I and H_half == padded_H_half:
        return blocks, scales

    # Split into linear and gate halves
    lin_b, gate_b = blocks[:, :I, :], blocks[:, I:, :]
    lin_s, gate_s = scales[:, :I, :], scales[:, I:, :]

    # Pad blocks with 0x00
    def _pad_blocks(t: torch.Tensor) -> torch.Tensor:
        padded = torch.zeros(
            (E, padded_I, padded_H_half), dtype=torch.uint8, device=t.device,
        )
        padded[:, :I, :H_half] = t
        return padded

    # Pad scales with 0x7F (E8M0 exponent=127 → scale=1.0)
    def _pad_scales(t: torch.Tensor) -> torch.Tensor:
        padded = torch.full(
            (E, padded_I, padded_H_scale), 0x7F, dtype=torch.uint8, device=t.device,
        )
        padded[:, :I, :H_scale] = t
        return padded

    out_blocks = torch.cat([_pad_blocks(lin_b), _pad_blocks(gate_b)], dim=1)
    out_scales = torch.cat([_pad_scales(lin_s), _pad_scales(gate_s)], dim=1)
    return out_blocks, out_scales


def pad_down_fp4(
    blocks: torch.Tensor,
    scales: torch.Tensor,
    padded_H: int,
    padded_I: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pad down projection FP4 packed blocks and scales.

    Args:
        blocks: [E, H, I/2] uint8
        scales: [E, H, I/32] uint8
        padded_H: target hidden size (rows)
        padded_I: target intermediate size (cols before packing)

    Returns:
        blocks [E, padded_H, padded_I/2], scales [E, padded_H, padded_I/32]
    """
    E, H, I_half = blocks.shape
    I_scale = scales.shape[2]          # I/32

    padded_I_half = padded_I // 2
    padded_I_scale = padded_I // 32

    if H == padded_H and I_half == padded_I_half:
        return blocks, scales

    # Pad blocks with 0x00
    padded_blocks = torch.zeros(
        (E, padded_H, padded_I_half), dtype=torch.uint8, device=blocks.device,
    )
    padded_blocks[:, :H, :I_half] = blocks

    # Pad scales with 0x7F
    padded_scales = torch.full(
        (E, padded_H, padded_I_scale), 0x7F, dtype=torch.uint8, device=scales.device,
    )
    padded_scales[:, :H, :I_scale] = scales

    return padded_blocks, padded_scales


def dequantize_from_mxfp4(
    blocks: torch.Tensor,
    scales: torch.Tensor,
    device: str,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Convert MXFP4 format tensors (blocks and scales) to the target dtype.

    Args:
        blocks: The packed FP4 values tensor (uint8)
        scales: The block scales tensor
        device: Target device string
        dtype: Target dtype for conversion

    Returns:
        Converted tensor in the target dtype
    """
    # CRITICAL FIX: If scales are loaded as float8 (e.g. from safetensors), we must
    # view them as uint8 to interpret them as raw exponent bits. Converting float8
    # directly to int32 would convert the *value* (approx 1.0 -> 1), not the bit
    # representation (e.g., 127), destroying the exponent information.
    if scales.dtype == torch.float8_e4m3fn:
        scales = scales.view(torch.uint8)

    # Convert raw exponent bits to int32 and subtract bias
    # We must do this on the source device to avoid dtype mismatch issues during copy
    scales = scales.to(torch.int32) - 127

    assert (
        blocks.shape[:-1] == scales.shape
    ), f"{blocks.shape=} does not match {scales.shape=}"

    lut = torch.tensor(FP4_VALUES, dtype=dtype, device=device)

    *prefix_shape, g, b = blocks.shape
    rows_total = math.prod(prefix_shape) * g

    # Move to target device
    blocks = blocks.reshape(rows_total, b).to(device)
    scales = scales.reshape(rows_total, 1).to(device)

    # Extract low and high 4-bit indices
    idx_lo = (blocks & 0x0F).to(torch.long)
    idx_hi = (blocks >> 4).to(torch.long)

    # Create output tensor and populate
    out = torch.empty(rows_total, b * 2, dtype=dtype, device=device)
    out[:, 0::2] = lut[idx_lo]  # Low 4-bit values at even indices
    out[:, 1::2] = lut[idx_hi]  # High 4-bit values at odd indices

    torch.ldexp(out, scales, out=out)

    return out.reshape(*prefix_shape, g, b * 2).view(*prefix_shape, g * b * 2)


def quantize_into_mxfp4(
    a: torch.Tensor,
    is_sf_swizzled_layout: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """FP4 batch quantization function."""
    from flashinfer import fp4_quantize  # type: ignore[import]

    num_experts = a.shape[0]
    sf_vec_size = 32  # MXFP4 uses 32-element blocks
    device = a.device

    quant_a = []
    sfs = []
    # Ensure scale factor is on the same device as the input
    a_global_sf = torch.tensor(1.0, dtype=torch.float32, device=device)

    for i in range(num_experts):
        # Input 'a' is likely already on GPU from dequantize step
        a_fp4, a_sf = fp4_quantize(
            a[i], a_global_sf, sf_vec_size, True, is_sf_swizzled_layout
        )
        quant_a.append(a_fp4)
        sfs.append(a_sf)

    result_quant_a = torch.stack(quant_a)
    result_sfs = torch.stack(sfs)

    return result_quant_a, result_sfs


def quantize_shuffle_gate_up_weights(
    gate_up_weights: torch.Tensor,
    padded_hidden_size: int,
    padded_intermediate_size: int,
    num_experts: int,
    gate_up_bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """
    Prepare gate_up (gemm1) weights for FlashInfer fused MoE kernel.

    This includes:
    1. Quantizing weights to MXFP4
    2. Reshaping to proper formats
    3. Shuffling for transposed MMA output
    4. Shuffling bias vectors to match weight row reordering
    """
    from flashinfer.fp4_quantization import block_scale_interleave  # type: ignore[import]
    from flashinfer.fused_moe.core import _maybe_get_cached_w3_w1_permute_indices  # type: ignore[import]

    epilogue_tile_m = 128
    cache_permute_indices: dict[tuple, torch.Tensor] = {}

    # Quantize weights with swizzled layout
    weights_quant, _ = quantize_into_mxfp4(gate_up_weights, True)

    # Quantize weights with linear layout for scales
    _, scales_linear = quantize_into_mxfp4(gate_up_weights, False)

    # Convert quantized weights to proper shapes
    weights_fp4 = weights_quant.view(torch.uint8).reshape(
        num_experts, 2 * padded_intermediate_size, padded_hidden_size // 2
    )
    scales_linear_fp4 = scales_linear.view(torch.uint8).reshape(
        num_experts, 2 * padded_intermediate_size, padded_hidden_size // 32
    )

    # Shuffle weights and scales for each expert
    weights_fp4_shuffled = []
    scales_fp4_shuffled = []
    bias_shuffled_list = []

    for i in range(num_experts):
        # Shuffle weights
        permute_indices = _maybe_get_cached_w3_w1_permute_indices(
            cache_permute_indices,
            weights_fp4[i].view(torch.uint8),
            epilogue_tile_m,
        )
        weights_fp4_shuffled.append(
            weights_fp4[i][permute_indices.to(weights_fp4.device)].contiguous()
        )

        # Shuffle bias using row permutation derived from weight permutation
        if gate_up_bias is not None:
            bias_shuffled_list.append(
                gate_up_bias[i][permute_indices.to(gate_up_bias.device)].contiguous()
            )

        # Shuffle scales
        permute_sf_indices = _maybe_get_cached_w3_w1_permute_indices(
            cache_permute_indices,
            scales_linear_fp4[i].view(torch.uint8),
            epilogue_tile_m,
            num_elts_per_sf=16,
        )
        scales_fp4_shuffled.append(
            block_scale_interleave(
                scales_linear_fp4[i][
                    permute_sf_indices.to(scales_linear_fp4.device)
                ].contiguous()
            )
        )

    # Stack weights for all experts
    weights_shuffled = torch.stack(weights_fp4_shuffled)
    scales_shuffled = (
        torch.stack(scales_fp4_shuffled)
        .view(torch.float8_e4m3fn)
        .reshape(num_experts, 2 * padded_intermediate_size, padded_hidden_size // 32)
    )

    # Stack bias tensors if provided
    bias_shuffled = None
    if gate_up_bias is not None:
        bias_shuffled = torch.stack(bias_shuffled_list)

    return weights_shuffled, scales_shuffled, bias_shuffled


def quantize_shuffle_down_weights(
    down_weights: torch.Tensor,
    padded_hidden_size: int,
    padded_intermediate_size: int,
    num_experts: int,
    down_bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """
    Prepare down (gemm2) weights for FlashInfer fused MoE kernel.

    This includes:
    1. Quantizing weights to MXFP4
    2. Reshaping to proper formats
    3. Shuffling for transposed MMA output
    4. Shuffling bias vectors to match weight row reordering
    """
    from flashinfer.fp4_quantization import block_scale_interleave  # type: ignore[import]
    from flashinfer.fused_moe.core import get_w2_permute_indices_with_cache  # type: ignore[import]

    epilogue_tile_m = 128
    cache_permute_indices: dict[tuple, torch.Tensor] = {}

    # Quantize weights with swizzled layout
    weights_quant, _ = quantize_into_mxfp4(down_weights, True)

    # Quantize weights with linear layout for scales
    _, scales_linear = quantize_into_mxfp4(down_weights, False)

    # Convert quantized weights to proper shapes
    weights_fp4 = weights_quant.view(torch.uint8).reshape(
        num_experts, padded_hidden_size, padded_intermediate_size // 2
    )
    scales_linear_fp4 = scales_linear.view(torch.uint8).reshape(
        num_experts, padded_hidden_size, padded_intermediate_size // 32
    )

    # Shuffle weights and scales for each expert
    weights_fp4_shuffled = []
    scales_fp4_shuffled = []
    bias_shuffled_list = []

    for i in range(num_experts):
        # Shuffle weights
        permute_indices = get_w2_permute_indices_with_cache(
            cache_permute_indices,
            weights_fp4[i],
            epilogue_tile_m,
        )
        weights_fp4_shuffled.append(
            weights_fp4[i][permute_indices.to(weights_fp4.device)].contiguous()
        )

        # Shuffle bias using row permutation derived from weight permutation
        if down_bias is not None:
            bias_shuffled_list.append(
                down_bias[i][permute_indices.to(down_bias.device)].contiguous()
            )

        # Shuffle scales
        permute_sf_indices = get_w2_permute_indices_with_cache(
            cache_permute_indices,
            scales_linear_fp4[i],
            epilogue_tile_m,
            num_elts_per_sf=16,
        )
        scales_fp4_shuffled.append(
            block_scale_interleave(
                scales_linear_fp4[i][
                    permute_sf_indices.to(scales_linear_fp4.device)
                ].contiguous()
            )
        )

    # Stack weights for all experts
    weights_shuffled = torch.stack(weights_fp4_shuffled)
    scales_shuffled = (
        torch.stack(scales_fp4_shuffled)
        .view(torch.float8_e4m3fn)
        .reshape(num_experts, padded_hidden_size, padded_intermediate_size // 32)
    )

    # Stack bias tensors if provided
    bias_shuffled = None
    if down_bias is not None:
        bias_shuffled = torch.stack(bias_shuffled_list)

    return weights_shuffled, scales_shuffled, bias_shuffled


def prepare_gptoss_moe_gate_up(
    tensors: list[torch.Tensor],
    config: dict[str, Any],
) -> dict[str, torch.Tensor]:
    """
    Prepare gate_up MoE weights for FlashInfer kernel.

    This transform function is designed to be used with Source.gather().transform().

    Args:
        tensors: List of [blocks, scales, bias] tensors
        config: Configuration dict with hidden_size, intermediate_size,
                padded_hidden_size, padded_intermediate_size, num_experts, device
                Optional: rank, world_size for TP sharding

    Returns:
        Dict with 'weights', 'scales', 'bias' keys containing prepared tensors
    """
    blocks, scales, bias = tensors

    device = config["device"]
    hidden_size = config["hidden_size"]
    intermediate_size = config["intermediate_size"]
    padded_hidden_size = config["padded_hidden_size"]
    padded_intermediate_size = config["padded_intermediate_size"]
    num_experts = config["num_experts"]

    # Handle Tensor Parallelism (TP)
    rank = config.get("rank", 0)
    world_size = config.get("world_size", 1)

    if is_apple_silicon():
        # --- Apple Silicon: keep FP4 packed, skip dequantization ---
        # Handle E8M0 scale dtype
        if scales.dtype == torch.float8_e4m3fn:
            scales = scales.view(torch.uint8)

        # Reshape to [E, 2*I, H/2] and [E, 2*I, H/32]
        blocks = blocks.reshape(num_experts, intermediate_size * 2, hidden_size // 2)
        scales = scales.reshape(num_experts, intermediate_size * 2, hidden_size // 32)

        # Deinterleave in FP4 domain (pure row permutation)
        blocks, scales = deinterleave_gate_up_fp4(blocks, scales)

        # TP sharding: row slicing in FP4 domain (rows are independent)
        if world_size > 1:
            if intermediate_size % world_size != 0:
                raise ValueError(
                    f"intermediate_size {intermediate_size} not divisible by world_size {world_size}"
                )
            local_intermediate = intermediate_size // world_size
            local_padded_intermediate = padded_intermediate_size // world_size
            start_idx = rank * local_intermediate
            end_idx = (rank + 1) * local_intermediate

            # Slice both linear and gate halves
            lin_b = blocks[:, :intermediate_size, :]
            gate_b = blocks[:, intermediate_size:, :]
            blocks = torch.cat(
                [lin_b[:, start_idx:end_idx, :].contiguous(),
                 gate_b[:, start_idx:end_idx, :].contiguous()], dim=1,
            )

            lin_s = scales[:, :intermediate_size, :]
            gate_s = scales[:, intermediate_size:, :]
            scales = torch.cat(
                [lin_s[:, start_idx:end_idx, :].contiguous(),
                 gate_s[:, start_idx:end_idx, :].contiguous()], dim=1,
            )

            intermediate_size = local_intermediate
            padded_intermediate_size = local_padded_intermediate

        # Pad in FP4 domain
        blocks, scales = pad_gate_up_fp4(
            blocks, scales, padded_hidden_size, padded_intermediate_size,
        )

        # Prepare bias (de-interleave + TP shard + pad, same as before)
        bias_deinterleaved = deinterleave_gate_up_bias(bias.to(device))
        if world_size > 1:
            orig_I = config["intermediate_size"]
            local_I = orig_I // world_size
            lin_bias = bias_deinterleaved[:, :orig_I]
            gate_bias = bias_deinterleaved[:, orig_I:]
            bias_deinterleaved = torch.cat(
                [lin_bias[:, rank * local_I:(rank + 1) * local_I].contiguous(),
                 gate_bias[:, rank * local_I:(rank + 1) * local_I].contiguous()],
                dim=1,
            )
        bias_padded = pad_gate_up_bias(
            bias_deinterleaved, padded_intermediate_size,
        ).to(torch.float32)

        return {
            "weights": blocks.to(device),
            "scales": scales.to(device),
            "bias": bias_padded,
        }

    # --- CUDA path: dequantize to bf16, then quantize/shuffle ---

    # Step 1: Dequantize MXFP4 to bfloat16
    weights_bf16 = dequantize_from_mxfp4(blocks, scales, device, torch.bfloat16)
    # Reshape to [num_experts, intermediate_size * 2, hidden_size]
    weights_bf16 = weights_bf16.reshape(num_experts, intermediate_size * 2, hidden_size)

    # Free input tensors early to reduce memory pressure
    del blocks, scales

    # Step 2: De-interleave from GPT OSS format to FlashInfer format
    weights_deinterleaved = deinterleave_gate_up_weights(weights_bf16)
    del weights_bf16  # Free intermediate tensor
    bias_deinterleaved = deinterleave_gate_up_bias(bias.to(device))
    del bias  # Free input bias

    # Step 3: Apply TP sharding AFTER de-interleaving (on float tensor)
    if world_size > 1:
        if intermediate_size % world_size != 0:
            raise ValueError(
                f"intermediate_size {intermediate_size} not divisible by world_size {world_size}"
            )

        local_intermediate = intermediate_size // world_size
        local_padded_intermediate = padded_intermediate_size // world_size

        start_idx = rank * local_intermediate
        end_idx = (rank + 1) * local_intermediate

        linear_part = weights_deinterleaved[:, :intermediate_size, :]
        gate_part = weights_deinterleaved[:, intermediate_size:, :]

        local_linear = linear_part[:, start_idx:end_idx, :].contiguous()
        local_gate = gate_part[:, start_idx:end_idx, :].contiguous()

        weights_deinterleaved = torch.cat([local_linear, local_gate], dim=1)

        linear_bias = bias_deinterleaved[:, :intermediate_size]
        gate_bias = bias_deinterleaved[:, intermediate_size:]

        local_linear_bias = linear_bias[:, start_idx:end_idx].contiguous()
        local_gate_bias = gate_bias[:, start_idx:end_idx].contiguous()

        bias_deinterleaved = torch.cat([local_linear_bias, local_gate_bias], dim=1)

        intermediate_size = local_intermediate
        padded_intermediate_size = local_padded_intermediate

    # Step 4: Pad to alignment
    weights_padded = pad_gate_up_weights(
        weights_deinterleaved,
        padded_hidden_size,
        padded_intermediate_size,
    )
    bias_padded = pad_gate_up_bias(
        bias_deinterleaved,
        padded_intermediate_size,
    ).to(torch.float32)

    # Step 5: Quantize and shuffle
    del weights_deinterleaved  # Free before final transform

    weights_shuffled, scales_shuffled, bias_shuffled = quantize_shuffle_gate_up_weights(
        weights_padded,
        padded_hidden_size,
        padded_intermediate_size,
        num_experts,
        gate_up_bias=bias_padded,
    )
    del weights_padded, bias_padded  # Free after quantization

    # Force CUDA memory cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "weights": weights_shuffled,
        "scales": scales_shuffled,
        "bias": bias_shuffled,
    }


def prepare_gptoss_moe_down(
    tensors: list[torch.Tensor],
    config: dict[str, Any],
) -> dict[str, torch.Tensor]:
    """
    Prepare down MoE weights for FlashInfer kernel.

    This transform function is designed to be used with Source.gather().transform().

    Args:
        tensors: List of [blocks, scales, bias] tensors
        config: Configuration dict with hidden_size, intermediate_size,
                padded_hidden_size, padded_intermediate_size, num_experts, device
                Optional: rank, world_size for TP sharding

    Returns:
        Dict with 'weights', 'scales', 'bias' keys containing prepared tensors
    """
    blocks, scales, bias = tensors

    device = config["device"]
    hidden_size = config["hidden_size"]
    intermediate_size = config["intermediate_size"]
    padded_hidden_size = config["padded_hidden_size"]
    padded_intermediate_size = config["padded_intermediate_size"]
    num_experts = config["num_experts"]

    # Handle Tensor Parallelism (TP)
    rank = config.get("rank", 0)
    world_size = config.get("world_size", 1)

    if is_apple_silicon():
        # --- Apple Silicon: keep FP4 packed, skip dequantization ---
        if scales.dtype == torch.float8_e4m3fn:
            scales = scales.view(torch.uint8)

        # Reshape to [E, H, I/2] and [E, H, I/32]
        blocks = blocks.reshape(num_experts, hidden_size, intermediate_size // 2)
        scales = scales.reshape(num_experts, hidden_size, intermediate_size // 32)

        # TP sharding: slice columns in FP4 packed domain
        if world_size > 1:
            if intermediate_size % world_size != 0:
                raise ValueError(
                    f"intermediate_size {intermediate_size} not divisible by world_size {world_size}"
                )
            local_intermediate = intermediate_size // world_size
            local_padded_intermediate = padded_intermediate_size // world_size
            start_half = rank * (local_intermediate // 2)
            end_half = (rank + 1) * (local_intermediate // 2)
            start_scale = rank * (local_intermediate // 32)
            end_scale = (rank + 1) * (local_intermediate // 32)

            blocks = blocks[:, :, start_half:end_half].contiguous()
            scales = scales[:, :, start_scale:end_scale].contiguous()

            # Bias for Down proj: divide by world_size to avoid over-counting
            bias = bias / world_size

            intermediate_size = local_intermediate
            padded_intermediate_size = local_padded_intermediate

        # Pad in FP4 domain
        blocks, scales = pad_down_fp4(
            blocks, scales, padded_hidden_size, padded_intermediate_size,
        )

        bias_padded = pad_down_bias(
            bias.to(device), padded_hidden_size,
        ).to(torch.float32)

        return {
            "weights": blocks.to(device),
            "scales": scales.to(device),
            "bias": bias_padded,
        }

    # --- CUDA path: dequantize to bf16, then quantize/shuffle ---

    # Step 1: Dequantize MXFP4 to bfloat16
    weights_bf16 = dequantize_from_mxfp4(blocks, scales, device, torch.bfloat16)
    # Reshape to [num_experts, hidden_size, intermediate_size]
    weights_bf16 = weights_bf16.reshape(num_experts, hidden_size, intermediate_size)

    # Free input tensors early to reduce memory pressure
    del blocks, scales

    # Step 2: Apply TP sharding AFTER dequantization (on float tensor)
    if world_size > 1:
        if intermediate_size % world_size != 0:
            raise ValueError(
                f"intermediate_size {intermediate_size} not divisible by world_size {world_size}"
            )

        local_intermediate = intermediate_size // world_size
        local_padded_intermediate = padded_intermediate_size // world_size

        start_idx = rank * local_intermediate
        end_idx = (rank + 1) * local_intermediate

        weights_bf16 = weights_bf16[:, :, start_idx:end_idx].contiguous()

        bias = bias / world_size

        intermediate_size = local_intermediate
        padded_intermediate_size = local_padded_intermediate

    # Step 3: Pad to alignment
    weights_padded = pad_down_weights(
        weights_bf16,
        padded_hidden_size,
        padded_intermediate_size,
    )
    del weights_bf16  # Free intermediate tensor
    bias_padded = pad_down_bias(
        bias.to(device),
        padded_hidden_size,
    ).to(torch.float32)
    del bias  # Free input bias

    # Step 4: Quantize and shuffle
    weights_shuffled, scales_shuffled, bias_shuffled = quantize_shuffle_down_weights(
        weights_padded,
        padded_hidden_size,
        padded_intermediate_size,
        num_experts,
        down_bias=bias_padded,
    )
    del weights_padded, bias_padded  # Free after quantization

    # Force CUDA memory cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "weights": weights_shuffled,
        "scales": scales_shuffled,
        "bias": bias_shuffled,
    }
