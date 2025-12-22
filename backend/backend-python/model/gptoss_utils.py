"""GPT OSS Utility Components

This module contains utility functions, constants, and helper layers
used by the GPT OSS model architecture.
"""

from __future__ import annotations

import math
from typing import Dict

import torch
from flashinfer import fp4_quantize
from flashinfer.fp4_quantization import block_scale_interleave
from flashinfer.fused_moe.core import (
    get_w2_permute_indices_with_cache,
    _maybe_get_cached_w3_w1_permute_indices,
)


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

# Alignment requirement for `trtllm_fp4_block_scale_moe``
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


def quantize_into_mxfp4(
    a: torch.Tensor,
    is_sf_swizzled_layout: bool,
):
    """FP4 batch quantization function."""
    num_experts = a.shape[0]
    sf_vec_size = 32  # MXFP4 uses 32-element blocks

    quant_a = []
    sfs = []
    a_global_sf = torch.tensor(1.0, dtype=torch.float32).cuda()
    for i in range(num_experts):
        a_fp4, a_sf = fp4_quantize(
            a[i].cuda(), a_global_sf, sf_vec_size, True, is_sf_swizzled_layout
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

    The operations are learned from FlashInfer's unit tests.
    https://github.com/flashinfer-ai/flashinfer/blob/454e7b25d0c21da176f0ef775e3de4a12d6c7118/tests/moe/test_trtllm_gen_fused_moe.py

    This includes:
    1. Quantizing weights to MXFP4
    2. Reshaping to proper formats
    3. Shuffling for transposed MMA output
    4. Shuffling bias vectors to match weight row reordering

    Args:
        gate_up_weights: Padded gate_up weights
                         [num_experts, 2*padded_intermediate_size, padded_hidden_size]
        padded_hidden_size: Padded hidden size (multiple of 256)
        padded_intermediate_size: Padded intermediate size (multiple of 256)
        num_experts: Number of experts
        gate_up_bias: Optional padded gate_up bias
                      [num_experts, 2*padded_intermediate_size]

    Returns:
        Tuple of (weights_shuffled, scales_shuffled, bias_shuffled)
    """
    epilogue_tile_m = 128
    cache_permute_indices: Dict[tuple, torch.Tensor] = {}

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

    The operations are learned from FlashInfer's unit tests.
    https://github.com/flashinfer-ai/flashinfer/blob/454e7b25d0c21da176f0ef775e3de4a12d6c7118/tests/moe/test_trtllm_gen_fused_moe.py

    This includes:
    1. Quantizing weights to MXFP4
    2. Reshaping to proper formats
    3. Shuffling for transposed MMA output
    4. Shuffling bias vectors to match weight row reordering

    Args:
        down_weights: Padded down weights
                      [num_experts, padded_hidden_size, padded_intermediate_size]
        padded_hidden_size: Padded hidden size (multiple of 256)
        padded_intermediate_size: Padded intermediate size (multiple of 256)
        num_experts: Number of experts
        down_bias: Optional padded down bias [num_experts, padded_hidden_size]

    Returns:
        Tuple of (weights_shuffled, scales_shuffled, bias_shuffled)
    """
    epilogue_tile_m = 128
    cache_permute_indices: Dict[tuple, torch.Tensor] = {}

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
    scales = scales.to(torch.int32) - 127

    assert (
        blocks.shape[:-1] == scales.shape
    ), f"{blocks.shape=} does not match {scales.shape=}"

    lut = torch.tensor(FP4_VALUES, dtype=dtype, device=device)

    *prefix_shape, g, b = blocks.shape
    rows_total = math.prod(prefix_shape) * g

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


def prepare_gptoss_moe_gate_up(
    blocks: torch.Tensor,
    scales: torch.Tensor,
    bias: torch.Tensor,
    config: dict,
    device: str,
) -> dict:
    """
    Prepare gate_up MoE weights for FlashInfer kernel.

    This includes:
    1. Dequantizing MXFP4 to bfloat16
    2. De-interleaving from GPT OSS format to FlashInfer format
    3. Padding to alignment
    4. Re-quantizing and shuffling for the kernel

    Args:
        blocks: MXFP4 blocks tensor
        scales: MXFP4 scales tensor
        bias: Bias tensor (bfloat16)
        config: Configuration dict with hidden_size, intermediate_size, etc.
        device: Target device string

    Returns:
        Dict with 'weights', 'scales', 'bias' keys containing prepared tensors
    """
    hidden_size = config["hidden_size"]
    intermediate_size = config["intermediate_size"]
    padded_hidden_size = config["padded_hidden_size"]
    padded_intermediate_size = config["padded_intermediate_size"]
    num_experts = config["num_experts"]

    # Step 1: Dequantize MXFP4 to bfloat16
    weights_bf16 = dequantize_from_mxfp4(blocks, scales, device, torch.bfloat16)
    # Reshape to [num_experts, intermediate_size * 2, hidden_size]
    weights_bf16 = weights_bf16.reshape(num_experts, intermediate_size * 2, hidden_size)

    # Step 2: De-interleave from GPT OSS format to FlashInfer format
    # GPT OSS: interleaved [gate, linear, gate, linear, ...]
    # FlashInfer: non-interleaved [linear..., gate...]
    weights_deinterleaved = deinterleave_gate_up_weights(weights_bf16)
    bias_deinterleaved = deinterleave_gate_up_bias(bias.to(device))

    # Step 3: Pad to alignment
    weights_padded = pad_gate_up_weights(
        weights_deinterleaved,
        padded_hidden_size,
        padded_intermediate_size,
    )
    bias_padded = pad_gate_up_bias(
        bias_deinterleaved,
        padded_intermediate_size,
    ).to(torch.float32)

    # Step 4: Quantize and shuffle
    weights_shuffled, scales_shuffled, bias_shuffled = quantize_shuffle_gate_up_weights(
        weights_padded,
        padded_hidden_size,
        padded_intermediate_size,
        num_experts,
        gate_up_bias=bias_padded,
    )

    return {
        "weights": weights_shuffled,
        "scales": scales_shuffled,
        "bias": bias_shuffled,
    }


def prepare_gptoss_moe_down(
    blocks: torch.Tensor,
    scales: torch.Tensor,
    bias: torch.Tensor,
    config: dict,
    device: str,
) -> dict:
    """
    Prepare down MoE weights for FlashInfer kernel.

    This includes:
    1. Dequantizing MXFP4 to bfloat16
    2. Padding to alignment
    3. Re-quantizing and shuffling for the kernel

    Args:
        blocks: MXFP4 blocks tensor
        scales: MXFP4 scales tensor
        bias: Bias tensor (bfloat16)
        config: Configuration dict with hidden_size, intermediate_size, etc.
        device: Target device string

    Returns:
        Dict with 'weights', 'scales', 'bias' keys containing prepared tensors
    """
    hidden_size = config["hidden_size"]
    intermediate_size = config["intermediate_size"]
    padded_hidden_size = config["padded_hidden_size"]
    padded_intermediate_size = config["padded_intermediate_size"]
    num_experts = config["num_experts"]

    # Step 1: Dequantize MXFP4 to bfloat16
    weights_bf16 = dequantize_from_mxfp4(blocks, scales, device, torch.bfloat16)
    # Reshape to [num_experts, hidden_size, intermediate_size]
    weights_bf16 = weights_bf16.reshape(num_experts, hidden_size, intermediate_size)

    # Step 2: Pad to alignment
    weights_padded = pad_down_weights(
        weights_bf16,
        padded_hidden_size,
        padded_intermediate_size,
    )
    bias_padded = pad_down_bias(
        bias.to(device),
        padded_hidden_size,
    ).to(torch.float32)

    # Step 3: Quantize and shuffle
    weights_shuffled, scales_shuffled, bias_shuffled = quantize_shuffle_down_weights(
        weights_padded,
        padded_hidden_size,
        padded_intermediate_size,
        num_experts,
        down_bias=bias_padded,
    )

    return {
        "weights": weights_shuffled,
        "scales": scales_shuffled,
        "bias": bias_shuffled,
    }
