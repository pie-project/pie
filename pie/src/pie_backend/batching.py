"""
Batch management for PIE backend inference.

This module provides the Batch dataclass that holds inference batch state
and handles tensor creation and response packaging.
"""

from __future__ import annotations

from typing import Any

import time
import numpy as np
import torch
from numba import njit, prange

from . import message


class Batch:
    """
    Holds the accumulated state for a specific inference step and handles packaging.

    This consolidates the state storage (formerly BatchState) and packaging logic
    (formerly ResponsePackager) into a single unified class.
    """

    def __init__(
        self,
        args: dict[str, Any],
        kv_page_size: int,
        max_dist_size: int,
        adapters: dict[int, Any],
        vocab_size: int | None = None,
    ) -> None:
        """
        Initialize a Batch from BatchedForwardPassRequest dict.

        Args:
            args: Dictionary with batched request fields
            kv_page_size: KV cache page size from model config
            max_dist_size: Max distribution size from model config
            adapters: Dictionary of active adapters
            vocab_size: Model vocabulary size (optional, needed for sampling masks)
        """
        # Initialize timing dict as instance attribute
        self.timing: dict[str, float] = {
            "decode_u32": 0.0,
            "mask_loop": 0.0,
            "brle_decode": 0.0,
            "sampler_loop": 0.0,
        }

        # Initialize mutable containers
        self.adapter_indices: list[int] = []
        self.adapter_seeds: list[int] = []
        self.adapter_subpass_needed: bool = False
        self.indices_for_logits: list[int] = []
        self.indices_for_embed_storage: list[int] = []
        self.embed_storage_pointers: list[int] = []

        # Helper to decode bytes as u32 array or pass through lists
        def _decode_u32(data):
            if isinstance(data, bytes):
                return np.frombuffer(data, dtype=np.uint32)

            print("Decoding u32 from list")
            return np.array(data, dtype=np.uint32)

        # Helper to decode bytes as i64 array
        def _decode_i64(data):
            if isinstance(data, bytes):
                # Assume input is u32/i32 for now as input is from msgpack
                # but we need i64 for token_ids in torch
                return np.frombuffer(data, dtype=np.uint32).astype(np.int64)

            print("Decoding i64 from list")
            return np.array(data, dtype=np.int64)

        # Direct assignments - decode bytes as u32 arrays
        t0 = time.perf_counter()
        # token_ids are long (i64)
        self.token_ids = _decode_i64(args["token_ids"])

        self.position_ids = _decode_u32(args["position_ids"]).astype(np.int32)
        self.kv_page_indices = _decode_u32(args["kv_page_indices"]).astype(np.int32)
        self.kv_page_indptr = _decode_u32(args["kv_page_indptr"]).astype(np.int32)
        self.kv_last_page_lens = _decode_u32(args["kv_last_page_lens"]).astype(np.int32)
        self.qo_indptr = _decode_u32(args["qo_indptr"]).astype(np.int32)
        self.single_token_mode = args["single_token_mode"]
        self.total_tokens = int(len(self.token_ids))
        self.timing["decode_u32"] = time.perf_counter() - t0

        # ===== VECTORIZED BATCH METADATA GENERATION =====
        t0 = time.perf_counter()

        # Process per-request data
        flattened_masks_u32 = _decode_u32(args["flattened_masks"]).astype(np.int32)
        mask_indptr = _decode_u32(args["mask_indptr"]).astype(np.int32)

        num_requests = len(args["adapter_indices"])

        # [OPTIMIZATION] Vectorized computation of per-request token counts
        req_token_counts = np.diff(self.qo_indptr)

        # [OPTIMIZATION] Vectorized sequence length calculation
        # seq_len = (num_pages - 1) * kv_page_size + last_len for num_pages > 0
        num_pages = np.diff(self.kv_page_indptr)
        seq_lens = np.where(
            num_pages > 0,
            (num_pages - 1) * kv_page_size + self.kv_last_page_lens,
            self.kv_last_page_lens,
        )

        # [OPTIMIZATION] Vectorized repeat - No Python loops!
        all_seq_lens = np.repeat(seq_lens, req_token_counts)

        # [OPTIMIZATION] Vectorized position IDs calculation
        # Formula: global_index - request_start + context_len
        context_lens = seq_lens - req_token_counts
        global_indices = np.arange(self.total_tokens, dtype=np.int32)
        request_starts = np.repeat(self.qo_indptr[:-1], req_token_counts)
        request_contexts = np.repeat(context_lens, req_token_counts)
        position_ids_np = global_indices - request_starts + request_contexts

        # [OPTIMIZATION] Vectorized cumsum for bit offsets
        token_acc_seq_lens_np = np.zeros(self.total_tokens + 1, dtype=np.int32)
        np.cumsum(all_seq_lens, out=token_acc_seq_lens_np[1:])

        self.timing["mask_loop"] = time.perf_counter() - t0

        # Call batch decoder ONCE for all tokens
        t_brle = time.perf_counter()
        self.attention_masks = decode_brle_batch(
            flattened_masks_u32, mask_indptr, position_ids_np, token_acc_seq_lens_np
        )
        self.timing["brle_decode"] = time.perf_counter() - t_brle

        # Helper to decode bytes as f32 array
        def _decode_f32(data):
            if isinstance(data, bytes):
                return np.frombuffer(data, dtype=np.float32)
            print("Decoding f32 from list")
            return np.array(data, dtype=np.float32)

        # ===== ZERO-COPY SAMPLER PARAMETER LOADING (SoA from Rust) =====
        t0 = time.perf_counter()

        # Load sampler parameters zero-copy from Rust SoA arrays
        self.temperatures = _decode_f32(args["sampler_temperatures"])
        self.top_k_values = _decode_u32(args["sampler_top_k"]).astype(np.int32)
        self.top_p_values = _decode_f32(args["sampler_top_p"])
        self.min_p_values = _decode_f32(args["sampler_min_p"])
        self.sampler_types = _decode_u32(args["sampler_types"]).tolist()
        self.request_output_counts = _decode_u32(args["request_num_samplers"]).astype(
            np.int32
        )

        # Load flattened output token indices
        flat_output_indices = _decode_u32(args["flat_output_token_indices"]).astype(
            np.int32
        )
        output_token_indptr = _decode_u32(args["output_token_indptr"]).astype(np.int32)

        # ===== VECTORIZED OUTPUT INDICES OFFSET CALCULATION =====
        # Each request's indices are relative to its token range, we need global offsets
        num_requests = len(self.request_output_counts)

        # For each index in flat_output_indices, add the corresponding request's token offset
        # Create an array mapping each flat index to its request's token offset
        indices_per_request = np.diff(output_token_indptr)
        request_token_offsets = self.qo_indptr[:-1]  # Token offset for each request

        # Expand offsets to match each output index
        flat_offsets = np.repeat(request_token_offsets, indices_per_request)

        # Apply offsets to get global indices
        if len(flat_output_indices) > 0:
            self.indices_for_logits = (flat_output_indices + flat_offsets).tolist()
        else:
            self.indices_for_logits = []

        # ===== SAMPLING MASK HANDLING =====
        self.sampling_masks = None
        if vocab_size is not None and "sampling_masks" in args:
            sampling_masks_u32 = _decode_u32(args["sampling_masks"]).astype(np.int32)
            sampling_mask_indptr = _decode_u32(args["sampling_mask_indptr"]).astype(
                np.int32
            )

            # Check if we have any sampling masks at all
            if len(sampling_masks_u32) > 0:
                # 1. Decode per-request masks (num_requests, vocab_size)
                # Initialize to True (allow all) so empty masks work as expected
                per_request_masks = decode_sampling_masks(
                    sampling_masks_u32, sampling_mask_indptr, num_requests, vocab_size
                )

                # 2. Expand to match logits (tokens per request)
                # If doing single-token decoding, indices_per_request is all 1s
                # Logic matches indices_for_logits expansion
                if len(self.indices_for_logits) > 0:
                    # Use indices_per_request from earlier (diff of output_token_indptr)
                    # We repeat the per-request mask rows
                    self.sampling_masks = np.repeat(
                        per_request_masks, indices_per_request, axis=0
                    )

        # ===== ADAPTER HANDLING (still needs per-request loop for now) =====
        adapter_indices = args["adapter_indices"]
        adapter_seeds = args["adapter_seeds"]

        for i in range(num_requests):
            req_token_count = int(req_token_counts[i])
            adapter_idx = adapter_indices[i]
            if adapter_idx is not None:
                seed = adapter_seeds[i] if adapter_seeds[i] is not None else 0
                self.adapter_seeds.extend([seed] * req_token_count)
                self.adapter_indices.append(adapter_idx)
                self.adapter_subpass_needed = True

        self.timing["sampler_loop"] = time.perf_counter() - t0

        # ===== SPECULATIVE DECODING INPUTS =====
        self.spec_token_ids = _decode_i64(args["spec_token_ids"])
        self.spec_position_ids = _decode_u32(args["spec_position_ids"]).astype(np.int32)
        self.spec_indptr = _decode_u32(args["spec_indptr"]).astype(np.int32)
        self.output_spec_flags = args["output_spec_flags"]
        self.sampler_seeds_arr = _decode_u32(args["sampler_seeds"])

        # ===== LOGIT MASKS (BRLE per request → bool matrix) =====
        logit_masks_u32 = _decode_u32(args["logit_masks"]).astype(np.int32)
        logit_mask_indptr = _decode_u32(args["logit_mask_indptr"]).astype(np.int32)
        if len(logit_masks_u32) > 0 and vocab_size is not None:
            self.logit_masks = decode_sampling_masks(
                logit_masks_u32, logit_mask_indptr, num_requests, vocab_size
            )
        else:
            self.logit_masks = None

    def get_model_inputs(self, device: torch.device) -> dict[str, Any]:
        """
        Finalize batch preparation and create input tensors for the model.

        Args:
            device: The torch device to create tensors on.

        Returns:
            Dictionary containing input tensors for the model engine.
        """
        # self.adapter_subpass_needed = False  # disable ZO for testing
        return {
            "token_ids": torch.as_tensor(
                self.token_ids, device=device, dtype=torch.long
            ),
            "position_ids": torch.as_tensor(
                self.position_ids, device=device, dtype=torch.int32
            ),
            "qo_indptr": torch.as_tensor(
                self.qo_indptr, device=device, dtype=torch.int32
            ),
            "kv_page_indices": torch.as_tensor(
                self.kv_page_indices, device=device, dtype=torch.int32
            ),
            "kv_page_indptr": torch.as_tensor(
                self.kv_page_indptr, device=device, dtype=torch.int32
            ),
            "kv_last_page_lens": torch.as_tensor(
                self.kv_last_page_lens, device=device, dtype=torch.int32
            ),
            "custom_mask": torch.as_tensor(
                self.attention_masks, device=device, dtype=torch.bool
            ),
            "single_token_inference_mode": self.single_token_mode,
            "adapter_indices": (
                self.adapter_indices if self.adapter_subpass_needed else []
            ),
            "adapter_seeds": (
                torch.as_tensor(self.adapter_seeds, device=device, dtype=torch.long)
                if self.adapter_subpass_needed
                else None
            ),
            "total_pages_cpu": self.kv_page_indptr[-1],
            "spec_token_ids": (
                torch.as_tensor(self.spec_token_ids, device=device, dtype=torch.long)
                if len(self.spec_token_ids) > 0
                else None
            ),
            "spec_position_ids": (
                torch.as_tensor(self.spec_position_ids, device=device, dtype=torch.int32)
                if len(self.spec_position_ids) > 0
                else None
            ),
            "spec_indptr": (
                torch.as_tensor(self.spec_indptr, device=device, dtype=torch.int32)
                if len(self.spec_indptr) > 1
                else None
            ),
        }

    def get_sampling_metadata(
        self, device: torch.device, dtype: torch.dtype
    ) -> dict[str, Any]:
        """
        Prepare the metadata required for the SamplingPass.

        Args:
            device: Torch device.
            dtype: Torch dtype for temperatures.

        Returns:
            Dictionary containing sampling metadata.
        """
        # Return empty if no logits needed
        if not self.indices_for_logits:
            return {"indices_for_logits": None}

        indices_for_logits = self.indices_for_logits

        # Vectorized tensor creation from NumPy arrays (no list comprehension)
        temperatures = (
            torch.tensor(self.temperatures, device=device, dtype=dtype)
            .clamp(min=1e-6)
            .unsqueeze(1)
        )

        # Pre-build sampler param tensors (avoid per-group construction)
        top_k_tensor = torch.tensor(self.top_k_values, device=device, dtype=torch.long)
        top_p_tensor = torch.tensor(self.top_p_values, device=device, dtype=dtype)
        min_p_tensor = torch.tensor(self.min_p_values, device=device, dtype=dtype)

        # Group samplers
        sampler_groups: dict[int, list[int]] = {}
        for i, sampler_idx in enumerate(self.sampler_types):
            if sampler_idx not in sampler_groups:
                sampler_groups[sampler_idx] = []
            sampler_groups[sampler_idx].append(i)

        return {
            "indices_for_logits": indices_for_logits,
            "temperatures": temperatures,
            "sampler_groups": sampler_groups,
            "top_k": top_k_tensor,
            "top_p": top_p_tensor,
            "min_p": min_p_tensor,
            "sampling_masks": (
                torch.as_tensor(self.sampling_masks, device=device, dtype=torch.bool)
                if self.sampling_masks is not None
                else None
            ),
            "logit_masks": (
                torch.as_tensor(
                    np.repeat(self.logit_masks, indices_per_request, axis=0),
                    device=device,
                    dtype=torch.bool,
                )
                if self.logit_masks is not None
                else None
            ),
        }

    def create_responses(
        self, sampling_results: dict[str, Any]
    ) -> list[message.ForwardPassResponse]:
        """
        Package the sampling results into responses for each original request.

        Args:
            sampling_results: Dictionary containing 'tokens' and 'dists'.

        Returns:
            List of responses in the order of requests.
        """
        num_requests = len(self.request_output_counts)

        # Early return if no logits needed
        if not self.indices_for_logits:
            return [
                message.ForwardPassResponse(dists=[], tokens=[])
                for _ in range(num_requests)
            ]

        final_dists = sampling_results["dists"]
        final_tokens_list = sampling_results["tokens"]
        spec_tokens_all = sampling_results.get("spec_tokens", None)
        spec_positions_all = sampling_results.get("spec_positions", None)

        responses = []
        cursor = 0

        for req_idx in range(num_requests):
            num_outputs = int(self.request_output_counts[req_idx])
            request_dists = []
            request_tokens = []

            for i in range(cursor, cursor + num_outputs):
                if self.sampler_types[i] == 0:
                    # Distribution request
                    if final_dists[i] is not None:
                        request_dists.append(final_dists[i])
                else:
                    # Sampling request
                    request_tokens.append(final_tokens_list[i])

            # Build response with optional speculation
            resp = message.ForwardPassResponse(dists=request_dists, tokens=request_tokens)
            if (
                spec_tokens_all is not None
                and self.output_spec_flags[req_idx]
                and spec_tokens_all[req_idx] is not None
            ):
                resp.spec_tokens = spec_tokens_all[req_idx]
                resp.spec_positions = spec_positions_all[req_idx]

            responses.append(resp)
            cursor += num_outputs

        return responses


@njit(cache=True)
def decode_brle_batch(
    flattened_masks: np.ndarray,
    mask_indptr: np.ndarray,
    position_ids: np.ndarray,
    token_acc_seq_lens: np.ndarray,
) -> np.ndarray:
    """
    Decode BRLE masks for an entire batch using Numba JIT.

    Optimized with slice assignment which Numba compiles to SIMD/memset
    block writes - faster than parallel (prange) due to thread pool overhead.

    Args:
        flattened_masks: Concatenated BRLE run lengths (int32)
        mask_indptr: Pointers to BRLE ranges per token (int32)
        position_ids: Position of each token, defines valid_len (int32)
        token_acc_seq_lens: Cumulative bit offsets per token (int32)

    Returns:
        Flat boolean array with all mask values
    """
    num_tokens = len(position_ids)
    total_bits = token_acc_seq_lens[-1]
    result = np.zeros(total_bits, dtype=np.bool_)

    for k in range(num_tokens):
        rle_start = mask_indptr[k]
        rle_end = mask_indptr[k + 1]
        global_bit_start = token_acc_seq_lens[k]
        valid_len = position_ids[k] + 1

        curr_bit_pos = global_bit_start
        bits_consumed = 0
        is_true_run = True

        for run_idx in range(rle_start, rle_end):
            if bits_consumed >= valid_len:
                break

            run_len = flattened_masks[run_idx]
            remaining = valid_len - bits_consumed
            eff_len = min(run_len, remaining)

            if is_true_run and eff_len > 0:
                # Slice assignment compiles to SIMD/memset
                result[curr_bit_pos : curr_bit_pos + eff_len] = True

            bits_consumed += eff_len
            curr_bit_pos += eff_len
            is_true_run = not is_true_run

    return result


@njit(cache=True)
def decode_sampling_masks(
    flattened_masks: np.ndarray,
    mask_indptr: np.ndarray,
    num_requests: int,
    vocab_size: int,
) -> np.ndarray:
    """
    Decode BRLE sampling masks.

    Args:
        flattened_masks: Concatenated BRLE run lengths (int32)
        mask_indptr: Pointers to BRLE ranges per request (int32)
        num_requests: Number of requests
        vocab_size: Size of vocabulary (dim 1 of output)

    Returns:
        Boolean array of shape (num_requests, vocab_size).
        Initialized to True (allow all). BRLE runs are applied on top.
    """
    result = np.ones((num_requests, vocab_size), dtype=np.bool_)

    for i in range(num_requests):
        rle_start = mask_indptr[i]
        rle_end = mask_indptr[i + 1]

        # If empty RLE, we leave as True (allow all)
        if rle_end == rle_start:
            continue

        # Standard BRLE decoding
        curr_pos = 0
        is_true_run = True  # Starts with True runs

        for run_idx in range(rle_start, rle_end):
            if curr_pos >= vocab_size:
                break

            run_len = flattened_masks[run_idx]
            remaining = vocab_size - curr_pos
            eff_len = min(run_len, remaining)

            if eff_len > 0:
                if not is_true_run:
                    # Set to False
                    result[i, curr_pos : curr_pos + eff_len] = False
                # Else leave as True

            curr_pos += eff_len
            is_true_run = not is_true_run

        # If RLE ended before vocab_size, remaining are left as True (since we init to True)
        # This is safe? Usually mask covers full range.
        # If mask is short, we assume trailing are allowed? Or blocked?
        # Protocol should assert full coverage, but 'allow' is safer default.

    return result
