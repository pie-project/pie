"""
Common modeling components for PIE backend.
"""

from __future__ import annotations

import torch
import math

from typing import Callable, Any

# Import flashinfer or flashinfer_metal depending on platform
if torch.backends.mps.is_available():
    from flashinfer_metal.sampling import (  # type: ignore[import-not-found]
        sampling_from_probs,
        top_p_sampling_from_probs,
        top_k_sampling_from_probs,
        min_p_sampling_from_probs,
        top_k_top_p_sampling_from_probs,
    )
else:
    from flashinfer.sampling import (  # type: ignore[import-not-found]
        sampling_from_probs,
        top_p_sampling_from_probs,
        top_k_sampling_from_probs,
        min_p_sampling_from_probs,
        top_k_top_p_sampling_from_probs,
    )

if torch.cuda.is_available():
    NUM_SM = torch.cuda.get_device_properties(
        torch.device("cuda")
    ).multi_processor_count
else:
    NUM_SM = 108


def _safe_scaled_softmax_impl_torch(logits, temperatures, greedy_threshold=1e-5):
    """
    Optimized Approach: Branchless safe_scaled_softmax (PyTorch Fallback)
    """
    # Ensure temperatures broadcasts correctly for where
    if temperatures.ndim == 1:
        temperatures = temperatures.unsqueeze(1)

    greedy_mask = temperatures < greedy_threshold

    # Branchless logic
    safe_temps = torch.where(greedy_mask, 1.0, temperatures)
    scaled_logits = logits / safe_temps
    probs_sampling = torch.softmax(scaled_logits, dim=-1)

    greedy_indices = logits.argmax(dim=-1)
    probs_greedy = torch.nn.functional.one_hot(
        greedy_indices, num_classes=logits.shape[-1]
    )
    probs_greedy = probs_greedy.to(dtype=logits.dtype)

    return torch.where(greedy_mask, probs_greedy, probs_sampling)


if torch.backends.mps.is_available():
    safe_scaled_softmax = _safe_scaled_softmax_impl_torch
else:
    import triton
    import triton.language as tl

    @triton.jit
    def _safe_softmax_kernel(
        output_ptr,
        logits_ptr,
        temps_ptr,
        stride_logits_row,
        stride_logits_col,
        stride_temps_row,
        stride_out_row,
        stride_out_col,
        n_cols,
        greedy_threshold,
        BLOCK_SIZE: tl.constexpr,
    ):
        # Row index this program instance is processing
        row_idx = tl.program_id(0)

        # Calculate pointers for the specific row
        logits_row_ptr = logits_ptr + row_idx * stride_logits_row
        temps_ptr_loc = temps_ptr + row_idx * stride_temps_row
        out_row_ptr = output_ptr + row_idx * stride_out_row

        # Load temperature for this row
        # We assume temps is shape (Batch, 1) or (Batch,)
        T = tl.load(temps_ptr_loc)

        # Create offsets for column loading
        col_offsets = tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols

        # Load logits with -inf padding for safety in max/argmax operations
        row_logits = tl.load(
            logits_row_ptr + col_offsets * stride_logits_col,
            mask=mask,
            other=-float("inf"),
        )

        # -----------------------------------------------------------
        # Optimization: Uniform Control Flow
        # Since T is constant for the whole row, all threads in the
        # warp/block take the same branch.
        # -----------------------------------------------------------
        if T < greedy_threshold:
            # --- GREEDY PATH (Argmax) ---
            # 1. Find the index of the max value
            max_idx = tl.argmax(row_logits, axis=0)

            # 2. Create One-Hot encoding
            # Compare every offset to the max_idx
            result = tl.where(col_offsets == max_idx, 1.0, 0.0)

        else:
            # --- SAMPLING PATH (Softmax) ---
            # --- SAMPLING PATH (Softmax) ---
            # 1. Apply temperature scaling
            scaled_logits = row_logits / T

            # 2. Subtract max for numerical stability (standard softmax trick)
            max_val = tl.max(scaled_logits, axis=0)
            logits_minus_max = scaled_logits - max_val

            # 3. Exponentiate
            numerator = tl.exp(logits_minus_max)

            # 4. Sum (normalization factor)
            denominator = tl.sum(numerator, axis=0)

            # 5. Divide
            result = numerator / denominator

        # Store result
        tl.store(out_row_ptr + col_offsets * stride_out_col, result, mask=mask)

    def safe_scaled_softmax_triton(logits, temperatures, greedy_threshold=1e-5):
        """
        Triton wrapper for safe scaled softmax.
        """
        # Input handling
        n_rows, n_cols = logits.shape

        # Ensure temperatures broadcasts correctly.
        # If 1D (Batch,), reshape to (Batch, 1) for consistent striding logic
        if temperatures.ndim == 1:
            temperatures = temperatures.unsqueeze(1)

        # Output allocation
        output = torch.empty_like(logits)

        # Heuristics for Block Size and Warps
        # Next power of 2 to fit the row in SRAM
        BLOCK_SIZE = triton.next_power_of_2(n_cols)

        # Manage maximum block size (hardware limit is usually 128KB, so ~32k float32s)
        # If n_cols is massive, we would need a tiled implementation, but this covers most cases.
        num_warps = 4
        if BLOCK_SIZE >= 2048:
            num_warps = 8
        if BLOCK_SIZE >= 4096:
            num_warps = 16

        # Launch Kernel
        grid = (n_rows,)

        _safe_softmax_kernel[grid](
            output,
            logits,
            temperatures,
            logits.stride(0),
            logits.stride(1),
            temperatures.stride(0),
            output.stride(0),
            output.stride(1),
            n_cols,
            greedy_threshold,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )

        return output

    safe_scaled_softmax = safe_scaled_softmax_triton


def sample_common(
    hidden_states: torch.Tensor,
    sampling_metadata: dict,
    lm_head_fn: Callable[[torch.Tensor], torch.Tensor],
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    """
    Execute the sampling pass.

    Args:
        hidden_states: The output hidden states from the model.
        sampling_metadata: Dictionary containing prepared sampling metadata.
        lm_head_fn: Function to compute logits from hidden states.
        device: Torch device.
        dtype: Torch dtype for intermediate calculations.

    Returns:
        Dictionary containing 'tokens' (list[int]) and 'dists' (list[tuple]).
    """
    if not sampling_metadata.get("indices_for_logits"):
        return {"tokens": [], "dists": []}

    indices_for_logits = sampling_metadata["indices_for_logits"]

    # Stage 1: Compute logits via LM head
    logits_input = hidden_states[indices_for_logits]

    # TODO: NaN check.
    # TODO: NaN check.
    # TODO: NaN check.
    nan_indices = []
    if torch.isnan(logits_input).any():
        nan_mask = torch.isnan(logits_input).any(dim=-1)
        nan_local_indices = torch.nonzero(nan_mask, as_tuple=True)[0]
        nan_indices = [indices_for_logits[i] for i in nan_local_indices.tolist()]
        print(f"Warning: NaNs detected in logits_input for indices: {nan_indices}")

    logits_input = torch.nan_to_num(logits_input)

    # Apply lm_head_fn
    logits = lm_head_fn(logits_input)

    # Apply sampling mask
    if (sampling_masks := sampling_metadata.get("sampling_masks")) is not None:
        logits.masked_fill_(~sampling_masks, -float("inf"))

    # Apply safe_scaled_softmax
    temperatures = sampling_metadata["temperatures"]
    probs = safe_scaled_softmax(logits, temperatures)

    num_logit_requests = len(indices_for_logits)
    final_dists = [None] * num_logit_requests
    final_tokens_tensor = torch.empty(
        num_logit_requests, dtype=torch.long, device=device
    )

    sampler_groups = sampling_metadata["sampler_groups"]
    # Pre-built tensors for sampler params (no per-group dict extraction)
    top_k_all = sampling_metadata["top_k"]
    top_p_all = sampling_metadata["top_p"]
    min_p_all = sampling_metadata["min_p"]

    for sampler_idx, indices in sampler_groups.items():
        if not indices:
            continue

        indices_tensor = torch.tensor(indices, device=device, dtype=torch.long)
        group_probs = probs.index_select(0, indices_tensor)

        if sampler_idx == 0:
            # Distribution mode - need top_k for each index
            group_top_k = top_k_all[indices_tensor]
            _process_distributions(indices, group_probs, final_dists, group_top_k)
        else:
            # Sampling mode - index into pre-built tensors
            group_top_k = top_k_all[indices_tensor]
            group_top_p = top_p_all[indices_tensor]
            group_min_p = min_p_all[indices_tensor]

            sampled = _execute_sampler(
                sampler_idx, group_probs, group_top_k, group_top_p, group_min_p
            )
            if sampled.dtype != torch.long:
                sampled = sampled.to(torch.long)

            final_tokens_tensor.scatter_(0, indices_tensor, sampled)

    # Stage 5: Combine results
    final_tokens_list = final_tokens_tensor.tolist()

    return {
        "tokens": final_tokens_list,
        "dists": final_dists,
        "nan_indices": nan_indices,
    }


def _process_distributions(
    indices: list[int],
    group_probs: torch.Tensor,
    final_dists: list[tuple[list[int], list[float]] | None],
    group_top_k: torch.Tensor,
) -> None:
    """Process distribution requests."""
    # group_top_k is already indexed for this group
    top_k_list = group_top_k.tolist()
    max_k = max(top_k_list) if top_k_list else 0

    if max_k > 0:
        topk_vals, topk_inds = torch.topk(group_probs, k=max_k, sorted=True)

        topk_vals_list = topk_vals.tolist()
        topk_inds_list = topk_inds.tolist()

        for i, original_idx in enumerate(indices):
            k = top_k_list[i]
            ids = topk_inds_list[i][:k]
            vals = topk_vals_list[i][:k]
            final_dists[original_idx] = (ids, vals)


def _execute_sampler(
    sampler_idx: int,
    group_probs: torch.Tensor,
    top_k: torch.Tensor,
    top_p: torch.Tensor,
    min_p: torch.Tensor,
) -> torch.Tensor:
    """Execute the appropriate sampling operation.

    Args:
        sampler_idx: Sampler type (1=uniform, 2=top_p, 3=top_k, 4=min_p, 5=top_k_top_p)
        group_probs: Probability tensor for this group
        top_k: Pre-indexed top_k tensor for this group
        top_p: Pre-indexed top_p tensor for this group
        min_p: Pre-indexed min_p tensor for this group

    Returns:
        Sampled token indices
    """
    if sampler_idx == 1:
        return sampling_from_probs(group_probs)

    elif sampler_idx == 2:
        return top_p_sampling_from_probs(group_probs, top_p=top_p)

    elif sampler_idx == 3:
        return top_k_sampling_from_probs(group_probs, top_k=top_k)

    elif sampler_idx == 4:
        return min_p_sampling_from_probs(group_probs, min_p=min_p)

    elif sampler_idx == 5:
        return top_k_top_p_sampling_from_probs(group_probs, top_k=top_k, top_p=top_p)

    else:
        raise ValueError(f"Unknown sampler index: {sampler_idx}")


def estimate_flashinfer_workspace_size(
    # Inputs corresponding to transform() arguments
    element_size: int,
    total_qo_len: int,
    batch_size: int,
    single_token_inference_mode: bool,
    # Config objects available in self
    model_config,  # needs .num_q_heads, .num_kv_heads, .dim_head
    runtime_config,  # needs .world_size, .device
) -> int:
    """
    Estimates the required workspace buffer size in bytes for FlashInfer operations.
    Replicates the C++ logic from flashinfer/attention/scheduler.cuh.
    """

    # --- 1. Setup Constants & GPU Properties ---
    local_num_qo_heads = model_config.num_q_heads // runtime_config.world_size
    local_num_kv_heads = model_config.num_kv_heads // runtime_config.world_size
    head_dim = model_config.dim_head

    # Helper for 16-byte alignment (FlashInfer Requirement)
    def align16(n):
        return (n + 15) // 16 * 16

    # Get GPU Multi-Processor Count for Split-KV estimation
    # FlashInfer uses heuristics based on available SMs to decide splitting.
    num_sm = NUM_SM

    # FlashInfer typically limits parallelism to 2 blocks per SM for these kernels
    max_grid_size = num_sm * 2
    gqa_group_size = local_num_qo_heads // local_num_kv_heads

    size = 0
    id_size = 4  # int32 used for indices

    # --- 2. Decode Path (Single Token) ---
    if single_token_inference_mode:

        # Simulate Work Estimation Logic:
        # If batch is small, FlashInfer splits KV to fill the GPU (Split-KV).
        # We calculate the "padded" batch size used for allocation.
        if batch_size * gqa_group_size >= max_grid_size:
            split_kv = False
            padded_batch_size = batch_size
        else:
            split_kv = True
            # In worst case (or CUDA graph), it pads to max capacity
            padded_batch_size = max_grid_size // max(1, gqa_group_size)

        # -- Int Buffer Allocations --
        size += align16(padded_batch_size * id_size)  # request_indices
        size += align16(padded_batch_size * id_size)  # kv_tile_indices
        size += align16((padded_batch_size + 1) * id_size)  # o_indptr
        size += align16(id_size)  # kv_chunk_size_ptr

        if split_kv:
            size += align16(padded_batch_size * 1)  # block_valid_mask (bool)

            # -- Float Buffer Allocations (Temporary Accumulation) --
            # V Buffer: [num_heads, padded_batch, head_dim] (Output Type)
            v_size = local_num_qo_heads * padded_batch_size * head_dim * element_size
            size += align16(v_size)

            # S Buffer: [num_heads, padded_batch] (Float32)
            s_size = local_num_qo_heads * padded_batch_size * 4
            size += align16(s_size)

    # --- 3. Prefill Path (Append) ---
    else:
        # Determine Tile Size (cta_tile_q)
        # Standard FlashInfer logic: 128 for dim <= 128, else 64
        cta_tile_q = 128 if head_dim <= 128 else 64

        # Calculate Padded Batch Size (Total Tiles)
        # In prefill, "batch" often refers to total number of tiles across all requests
        packed_total_len = total_qo_len * gqa_group_size
        total_num_tiles_q = math.ceil(packed_total_len / cta_tile_q) + batch_size

        # FlashInfer bounds prefill splitting by available SMs to avoid OOM
        # So allocation size is max(sm_capacity, needed_tiles)
        padded_batch_size = max(max_grid_size, total_num_tiles_q)

        # -- Int Buffer Allocations --
        size += align16(padded_batch_size * id_size)  # request_indices
        size += align16(padded_batch_size * id_size)  # qo_tile_indices
        size += align16(padded_batch_size * id_size)  # kv_tile_indices
        size += align16((batch_size + 1) * id_size)  # o_indptr
        size += align16(id_size)  # kv_chunk_size_ptr

        # Merge Indptr (Conservative Estimate for Split-KV)
        size += align16((total_qo_len + 1) * id_size)
        size += align16(padded_batch_size * 1)  # block_valid_mask

        # -- Float Buffer Allocations --
        # FlashInfer allocates float buffers for Split-KV prefill.
        # Crucially, it bounds this by `max_grid_size` (execution parallelism),
        # NOT by `total_num_tiles_q` (data length), otherwise long contexts would OOM.

        alloc_units = max_grid_size

        # V Buffer: [num_heads, alloc_units, tile_size, head_dim] (Float32)
        # Note: FlashInfer uses float32 for prefill accumulation
        v_size = local_num_qo_heads * alloc_units * cta_tile_q * head_dim * 4
        size += align16(v_size)

        # S Buffer: [num_heads, alloc_units, tile_size] (Float32)
        s_size = local_num_qo_heads * alloc_units * cta_tile_q * 4
        size += align16(s_size)

    return size
