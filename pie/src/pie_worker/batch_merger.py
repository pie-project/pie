"""Merge multiple fire_batch kwargs dicts into one for Python-side batching.

When Python accumulates multiple fire_batch RPCs while the GPU is busy,
this module merges them into a single fire_batch kwargs dict that can be
passed to PieVllmRuntime.fire_batch() as if it came from one large Rust batch.

The merge operates at the raw array level: concatenate flat arrays, adjust
CSR indptr offsets, and combine list fields. The merged dict is structurally
identical to a single large BatchedForwardPassRequest from Rust.

After execute_model, split_fire_batch_results() divides the results list
back into per-original-batch slices for individual pycrust responses.
"""
from __future__ import annotations

import numpy as np

from .vllm_batch_translator import PieVllmBatchTranslator


def merge_fire_batch_kwargs(kwargs_list: list[dict]) -> dict:
    """Merge multiple fire_batch kwargs dicts into one.

    Each kwargs dict contains the same fields as a BatchedForwardPassRequest
    from Rust IPC (msgpack-decoded). This function concatenates flat arrays,
    adjusts CSR indptr offsets, and combines list fields to produce one
    merged kwargs dict that fire_batch() can process as a single batch.

    Args:
        kwargs_list: List of fire_batch kwargs dicts to merge.
            Must contain at least one element.

    Returns:
        Single merged kwargs dict.
    """
    if len(kwargs_list) == 1:
        return kwargs_list[0]

    decode = PieVllmBatchTranslator.decode_binary_array

    # Accumulators for flat arrays (will be concatenated)
    all_token_ids = []
    all_kv_page_indices = []
    all_kv_last_page_lens = []
    all_flattened_masks = []
    all_flat_output_token_indices = []

    # Accumulators for SoA sampler arrays
    all_sampler_temps = []
    all_sampler_top_k = []
    all_sampler_top_p = []
    all_sampler_min_p = []
    all_sampler_types = []
    all_request_num_samplers = []

    # Accumulators for CSR indptr arrays (need offset adjustment)
    qo_parts = []       # each: np.diff of qo_indptr (tokens per req)
    kv_parts = []       # each: np.diff of kv_page_indptr (pages per req)
    mask_parts = []     # each: np.diff of mask_indptr (mask elements per token)
    oti_parts = []      # each: np.diff of output_token_indptr

    # Accumulators for list fields
    all_adapter_indices = []
    all_adapter_seeds = []
    all_output_embed_ptrs = []
    all_output_embed_indices = []

    single_token_mode = True
    for kw in kwargs_list:
        # Decode flat arrays
        tids = decode(kw["token_ids"], np.uint32)
        qo = decode(kw["qo_indptr"], np.uint32)
        kv_idx = decode(kw["kv_page_indices"], np.uint32)
        kv_indptr = decode(kw["kv_page_indptr"], np.uint32)
        kv_last = decode(kw["kv_last_page_lens"], np.uint32)

        num_req = len(qo) - 1

        all_token_ids.append(tids)
        qo_parts.append(np.diff(qo))

        all_kv_page_indices.append(kv_idx)
        kv_parts.append(np.diff(kv_indptr))
        all_kv_last_page_lens.append(kv_last)

        # Masks
        fm = decode(kw.get("flattened_masks", b""), np.uint32)
        mi = decode(kw.get("mask_indptr", b""), np.uint32)
        all_flattened_masks.append(fm)
        if len(mi) > 1:
            mask_parts.append(np.diff(mi))
        else:
            # No mask data for this batch — but we need placeholder for
            # the tokens in this batch.  Each token has 0 mask elements.
            total_tokens = int(np.sum(np.diff(qo)))
            if total_tokens > 0:
                mask_parts.append(np.zeros(total_tokens, dtype=np.uint32))

        # SoA sampler arrays
        all_sampler_temps.append(decode(kw.get("sampler_temperatures", b""), np.float32))
        all_sampler_top_k.append(decode(kw.get("sampler_top_k", b""), np.uint32))
        all_sampler_top_p.append(decode(kw.get("sampler_top_p", b""), np.float32))
        all_sampler_min_p.append(decode(kw.get("sampler_min_p", b""), np.float32))
        all_sampler_types.append(decode(kw.get("sampler_types", b""), np.uint32))
        all_request_num_samplers.append(decode(kw.get("request_num_samplers", b""), np.uint32))

        # Output token indices
        foti = decode(kw.get("flat_output_token_indices", b""), np.uint32)
        oti = decode(kw.get("output_token_indptr", b""), np.uint32)
        all_flat_output_token_indices.append(foti)
        if len(oti) > 1:
            oti_parts.append(np.diff(oti))
        else:
            oti_parts.append(np.zeros(num_req, dtype=np.uint32))

        # List fields
        all_adapter_indices.extend(kw.get("adapter_indices", [None] * num_req))
        all_adapter_seeds.extend(kw.get("adapter_seeds", [None] * num_req))
        all_output_embed_ptrs.extend(kw.get("output_embed_ptrs", [[] for _ in range(num_req)]))
        all_output_embed_indices.extend(kw.get("output_embed_indices", [[] for _ in range(num_req)]))

        single_token_mode = single_token_mode and kw.get("single_token_mode", True)

    # Build merged arrays
    def _concat_and_encode(parts: list[np.ndarray]) -> bytes:
        """Concatenate arrays and convert back to raw bytes."""
        if not parts or all(len(p) == 0 for p in parts):
            return b""
        return np.concatenate(parts).tobytes()

    def _rebuild_indptr(diff_parts: list[np.ndarray], dtype=np.uint32) -> bytes:
        """Rebuild a CSR indptr array from per-batch diffs."""
        if not diff_parts or all(len(p) == 0 for p in diff_parts):
            return np.array([0], dtype=dtype).tobytes()
        all_diffs = np.concatenate(diff_parts)
        indptr = np.empty(len(all_diffs) + 1, dtype=dtype)
        indptr[0] = 0
        np.cumsum(all_diffs, out=indptr[1:])
        return indptr.tobytes()

    merged = {
        "token_ids": _concat_and_encode(all_token_ids),
        "qo_indptr": _rebuild_indptr(qo_parts),
        "kv_page_indices": _concat_and_encode(all_kv_page_indices),
        "kv_page_indptr": _rebuild_indptr(kv_parts),
        "kv_last_page_lens": _concat_and_encode(all_kv_last_page_lens),
        "flattened_masks": _concat_and_encode(all_flattened_masks),
        "mask_indptr": _rebuild_indptr(mask_parts),
        "sampler_temperatures": _concat_and_encode(all_sampler_temps),
        "sampler_top_k": _concat_and_encode(all_sampler_top_k),
        "sampler_top_p": _concat_and_encode(all_sampler_top_p),
        "sampler_min_p": _concat_and_encode(all_sampler_min_p),
        "sampler_types": _concat_and_encode(all_sampler_types),
        "request_num_samplers": _concat_and_encode(all_request_num_samplers),
        "flat_output_token_indices": _concat_and_encode(all_flat_output_token_indices),
        "output_token_indptr": _rebuild_indptr(oti_parts),
        "adapter_indices": all_adapter_indices,
        "adapter_seeds": all_adapter_seeds,
        "output_embed_ptrs": all_output_embed_ptrs,
        "output_embed_indices": all_output_embed_indices,
        "single_token_mode": single_token_mode,
    }

    # Copy optional scalar fields from first batch
    for key in ("trace_context", "group_id"):
        if key in kwargs_list[0]:
            merged[key] = kwargs_list[0][key]

    return merged


def compute_batch_generate_counts(kwargs_list: list[dict]) -> list[int]:
    """Count non-flush (generate) requests per original batch.

    Flush requests have request_num_samplers == 0 and are omitted from
    the results list. This function returns the number of generate requests
    in each original batch, needed for splitting results after merge.

    Args:
        kwargs_list: List of original fire_batch kwargs dicts.

    Returns:
        List of generate request counts, one per original batch.
    """
    decode = PieVllmBatchTranslator.decode_binary_array
    counts = []
    for kw in kwargs_list:
        rns = decode(kw.get("request_num_samplers", b""), np.uint32)
        counts.append(int(np.sum(rns > 0)))
    return counts


def split_fire_batch_results(
    merged_response: dict,
    generate_counts: list[int],
) -> list[dict]:
    """Split merged fire_batch response back into per-original-batch responses.

    After merging N fire_batch calls and executing once, the merged response
    has a combined results list. This splits it at generate_count boundaries
    so each original pycrust request gets its own response.

    Args:
        merged_response: Response dict from fire_batch with 'results' and 'metrics'.
        generate_counts: Number of generate (non-flush) requests per original batch.

    Returns:
        List of response dicts, one per original batch, each with its own
        'results' slice and shared 'metrics'.
    """
    results = merged_response["results"]
    metrics = merged_response["metrics"]

    responses = []
    offset = 0
    for count in generate_counts:
        responses.append({
            "results": results[offset:offset + count],
            "metrics": metrics,
        })
        offset += count

    return responses
