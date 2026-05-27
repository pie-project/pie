"""Tensor materialization for `._bridge.batching.Batch`.

Bridge owns the numpy/numba wire model; this module is the per-flavor
torch translation layer. Each flavor wheel ships its own copy so the
bridge stays torch-free.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from ._bridge.batching import Batch


def _shared_kv_inputs(batch: Batch, device: torch.device) -> dict[str, Any]:
    return {
        "kv_page_indices": torch.as_tensor(
            batch.kv_page_indices, device=device, dtype=torch.int32
        ).contiguous(),
        "kv_page_indptr": torch.as_tensor(
            batch.kv_page_indptr, device=device, dtype=torch.int32
        ).contiguous(),
    }


def _adapter_inputs(batch: Batch, device: torch.device) -> dict[str, Any]:
    return {
        "adapter_indices": (
            batch.adapter_indices if batch.adapter_subpass_needed else []
        ),
        "adapter_seeds": (
            torch.as_tensor(batch.adapter_seeds, device=device, dtype=torch.long)
            if batch.adapter_subpass_needed
            else None
        ),
    }


def build_model_inputs(batch: Batch, device: torch.device) -> dict[str, Any]:
    """Finalize batch preparation and create input tensors for the model."""
    if len(batch.kv_page_indices) > 0:
        _kv_idx_max = int(batch.kv_page_indices.max())
        _kv_idx_min = int(batch.kv_page_indices.min())
    else:
        _kv_idx_max = 0
        _kv_idx_min = 0
    _kv_indptr_last = (
        int(batch.kv_page_indptr[-1]) if len(batch.kv_page_indptr) > 0 else 0
    )
    if len(batch.kv_last_page_lens) > 0:
        _kv_last_max = int(batch.kv_last_page_lens.max())
        _kv_last_min = int(batch.kv_last_page_lens.min())
    else:
        _kv_last_max = 0
        _kv_last_min = 1

    return {
        "kv_page_indices_max": _kv_idx_max,
        "kv_page_indices_min": _kv_idx_min,
        "kv_page_indptr_last": _kv_indptr_last,
        "kv_last_page_lens_max": _kv_last_max,
        "kv_last_page_lens_min": _kv_last_min,
        "token_ids": torch.as_tensor(batch.token_ids, device=device, dtype=torch.long),
        "position_ids": torch.as_tensor(
            batch.position_ids, device=device, dtype=torch.int32
        ),
        "qo_indptr": torch.as_tensor(batch.qo_indptr, device=device, dtype=torch.int32),
        **_shared_kv_inputs(batch, device),
        "kv_last_page_lens": torch.as_tensor(
            batch.kv_last_page_lens, device=device, dtype=torch.int32
        ),
        "custom_mask": (
            torch.as_tensor(batch.attention_masks, device=device, dtype=torch.bool)
            if batch.attention_masks is not None
            else None
        ),
        "single_token_inference_mode": batch.single_token_mode,
        **_adapter_inputs(batch, device),
        "spec_token_ids": (
            torch.as_tensor(batch.spec_token_ids, device=device, dtype=torch.long)
            if len(batch.spec_token_ids) > 0
            else None
        ),
        "spec_position_ids": (
            torch.as_tensor(batch.spec_position_ids, device=device, dtype=torch.int32)
            if len(batch.spec_position_ids) > 0
            else None
        ),
        "spec_indptr": (
            torch.as_tensor(batch.spec_indptr, device=device, dtype=torch.int32)
            if len(batch.spec_indptr) > 1
            else None
        ),
    }


def build_spec_expanded_model_inputs(
    batch: Batch, device: torch.device
) -> dict[str, Any]:
    """Like `build_model_inputs`, but with drafts spliced into the forward.

    Only call when `batch.has_speculative_inputs` is True. Triggers
    `batch._build_spec_plan` (idempotent) so subsequent
    `build_spec_expanded_sampling_metadata` and `batch.verify_drafts`
    can read the cached `_spec_*` fields.
    """
    batch._build_spec_plan()
    plan = batch._spec_plan
    pending_counts = batch._spec_pending_counts
    n_drafts_per_req = batch._spec_n_drafts_per_req
    qo_indptr_ext = batch._spec_qo_indptr_ext
    seq_lens_old = batch._spec_seq_lens_old
    num_requests = len(batch.request_output_counts)
    new_token_counts = pending_counts + n_drafts_per_req

    new_total = int(new_token_counts.sum())
    token_ids_ext = np.empty(new_total, dtype=batch.token_ids.dtype)
    position_ids_ext = np.empty(new_total, dtype=batch.position_ids.dtype)

    for i in range(num_requests):
        dst = qo_indptr_ext[i]
        n_p = int(pending_counts[i])
        n_d = int(n_drafts_per_req[i])
        ps = int(batch.qo_indptr[i])
        token_ids_ext[dst : dst + n_p] = batch.token_ids[ps : ps + n_p]
        position_ids_ext[dst : dst + n_p] = batch.position_ids[ps : ps + n_p]
        if n_d > 0:
            ds = plan[i]["drafts_start"]
            token_ids_ext[dst + n_p : dst + n_p + n_d] = batch.spec_token_ids[
                ds : ds + n_d
            ]
            position_ids_ext[dst + n_p : dst + n_p + n_d] = batch.spec_position_ids[
                ds : ds + n_d
            ]

    page_size = batch._kv_page_size
    new_total_kv_last = batch.kv_last_page_lens.astype(np.int64) + n_drafts_per_req
    kv_last_page_lens_ext = ((new_total_kv_last - 1) % page_size + 1).astype(np.int32)

    seq_lens_per_req = seq_lens_old + n_drafts_per_req
    per_token_seq_lens = np.repeat(seq_lens_per_req, new_token_counts)
    total_bits = int(per_token_seq_lens.sum())
    attention_masks_ext = np.zeros(total_bits, dtype=np.bool_)
    offset = 0
    for k in range(new_total):
        row_len = int(per_token_seq_lens[k])
        valid = min(int(position_ids_ext[k]) + 1, row_len)
        attention_masks_ext[offset : offset + valid] = True
        offset += row_len

    single_token_mode_ext = (
        batch.single_token_mode and not batch.has_speculative_inputs
    )

    return {
        "token_ids": torch.as_tensor(token_ids_ext, device=device, dtype=torch.long),
        "position_ids": torch.as_tensor(
            position_ids_ext, device=device, dtype=torch.int32
        ),
        "qo_indptr": torch.as_tensor(qo_indptr_ext, device=device, dtype=torch.int32),
        **_shared_kv_inputs(batch, device),
        "kv_last_page_lens": torch.as_tensor(
            kv_last_page_lens_ext, device=device, dtype=torch.int32
        ),
        "custom_mask": torch.as_tensor(
            attention_masks_ext, device=device, dtype=torch.bool
        ),
        "single_token_inference_mode": single_token_mode_ext,
        **_adapter_inputs(batch, device),
    }


def build_sampling_metadata(
    batch: Batch, device: torch.device, dtype: torch.dtype
) -> dict[str, Any]:
    """Prepare the metadata required for the SamplingPass."""
    if not batch.indices_for_logits:
        return {"indices_for_logits": None}

    indices_for_logits = batch.indices_for_logits

    temperatures = (
        torch.tensor(batch.temperatures, device=device, dtype=dtype)
        .clamp(min=1e-6)
        .unsqueeze(1)
    )

    top_k_tensor = torch.tensor(batch.top_k_values, device=device, dtype=torch.long)
    top_p_tensor = torch.tensor(batch.top_p_values, device=device, dtype=dtype)
    min_p_tensor = torch.tensor(batch.min_p_values, device=device, dtype=dtype)

    sampler_groups: dict[int, list[int]] = {}
    for i, sampler_idx in enumerate(batch.sampler_types):
        sampler_groups.setdefault(sampler_idx, []).append(i)

    seeds_tensor = (
        torch.as_tensor(
            batch.sampler_seeds_arr.astype(np.int64),
            device=device,
            dtype=torch.long,
        )
        if batch.has_user_seeds
        else None
    )

    return {
        "indices_for_logits": indices_for_logits,
        "temperatures": temperatures,
        "sampler_groups": sampler_groups,
        "top_k": top_k_tensor,
        "top_p": top_p_tensor,
        "min_p": min_p_tensor,
        "seeds": seeds_tensor,
        "sampler_label_ids": batch.sampler_label_ids,
        "sampler_label_indptr": batch.sampler_label_indptr,
        "sampling_masks": (
            torch.as_tensor(batch.sampling_masks, device=device, dtype=torch.bool)
            if batch.sampling_masks is not None
            else (
                torch.as_tensor(
                    np.repeat(batch.logit_masks, batch.indices_per_request, axis=0),
                    device=device,
                    dtype=torch.bool,
                )
                if batch.logit_masks is not None
                else None
            )
        ),
    }


def build_spec_expanded_sampling_metadata(
    batch: Batch, device: torch.device, dtype: torch.dtype
) -> dict[str, Any]:
    """Sampling metadata for the spec-expanded forward.

    Must be called after `build_spec_expanded_model_inputs`, which populates
    `batch._spec_plan`.
    """
    if batch._spec_plan is None:
        raise RuntimeError(
            "build_spec_expanded_sampling_metadata called before "
            "build_spec_expanded_model_inputs (no _spec_plan cached)."
        )
    plan = batch._spec_plan
    qo_indptr_ext = batch._spec_qo_indptr_ext
    num_requests = len(batch.request_output_counts)

    if len(batch._sampling_indices_relative) > 0:
        flat_offsets_new = np.repeat(qo_indptr_ext[:-1], batch.indices_per_request)
        inferlet_indices = (
            batch._sampling_indices_relative + flat_offsets_new
        ).tolist()
    else:
        inferlet_indices = []

    verify_indices: list[int] = []
    verify_slot_starts: list[tuple[int, int] | None] = [None] * num_requests
    first_sampler_idx_per_req: list[int | None] = [None] * num_requests
    cursor = 0
    for i in range(num_requests):
        if int(batch.request_output_counts[i]) > 0:
            first_sampler_idx_per_req[i] = cursor
        cursor += int(batch.request_output_counts[i])

    running = 0
    for i in range(num_requests):
        if plan[i] is None:
            continue
        n_p = plan[i]["n_pending"]
        n_d = plan[i]["n_drafts"]
        base = int(qo_indptr_ext[i]) + n_p - 1
        verify_indices.extend(range(base, base + n_d + 1))
        verify_slot_starts[i] = (running, n_d)
        running += n_d + 1

        if first_sampler_idx_per_req[i] is None:
            raise ValueError(
                f"Request {i}: speculative input supplied but no inferlet "
                "sampler — verification needs a sampler config to clone."
            )

    indices_for_logits = inferlet_indices + verify_indices
    if not indices_for_logits:
        return {"indices_for_logits": None}

    def _extend_with_clones(src: np.ndarray) -> np.ndarray:
        extra = []
        for i in range(num_requests):
            if verify_slot_starts[i] is None:
                continue
            fs = first_sampler_idx_per_req[i]
            count = verify_slot_starts[i][1] + 1
            extra.append(np.repeat(src[fs : fs + 1], count))
        if not extra:
            return src
        return np.concatenate([src] + extra)

    temperatures_arr = _extend_with_clones(batch.temperatures)
    top_k_arr = _extend_with_clones(batch.top_k_values)
    top_p_arr = _extend_with_clones(batch.top_p_values)
    min_p_arr = _extend_with_clones(batch.min_p_values)
    seeds_arr = _extend_with_clones(batch.sampler_seeds_arr)

    sampler_types_ext = list(batch.sampler_types)
    for i in range(num_requests):
        if verify_slot_starts[i] is None:
            continue
        fs = first_sampler_idx_per_req[i]
        count = verify_slot_starts[i][1] + 1
        stype = batch.sampler_types[fs]
        if stype == 0:
            raise ValueError(
                f"Request {i}: first sampler is Distribution-mode, "
                "which can't be used for spec verification."
            )
        if stype in (7, 8, 9, 10):
            raise ValueError(
                f"Request {i}: first sampler is type {stype} "
                "(RawLogits/Logprob/Logprobs/Entropy), which can't be "
                "used for spec verification."
            )
        sampler_types_ext.extend([stype] * count)

    sampler_groups: dict[int, list[int]] = {}
    for i, sampler_idx in enumerate(sampler_types_ext):
        sampler_groups.setdefault(sampler_idx, []).append(i)

    temperatures_t = (
        torch.tensor(temperatures_arr, device=device, dtype=dtype)
        .clamp(min=1e-6)
        .unsqueeze(1)
    )
    top_k_t = torch.tensor(top_k_arr, device=device, dtype=torch.long)
    top_p_t = torch.tensor(top_p_arr, device=device, dtype=dtype)
    min_p_t = torch.tensor(min_p_arr, device=device, dtype=dtype)
    seeds_t = (
        torch.as_tensor(
            seeds_arr.astype(np.int64), device=device, dtype=torch.long
        )
        if batch.has_user_seeds
        else None
    )

    sampling_masks_t = None
    if batch.sampling_masks is not None:
        extras = []
        for i in range(num_requests):
            if verify_slot_starts[i] is None:
                continue
            fs = first_sampler_idx_per_req[i]
            count = verify_slot_starts[i][1] + 1
            extras.append(
                np.repeat(batch.sampling_masks[fs : fs + 1], count, axis=0)
            )
        mask_np = (
            np.concatenate([batch.sampling_masks] + extras, axis=0)
            if extras
            else batch.sampling_masks
        )
        sampling_masks_t = torch.as_tensor(mask_np, device=device, dtype=torch.bool)
    elif batch.logit_masks is not None:
        base = np.repeat(batch.logit_masks, batch.indices_per_request, axis=0)
        extras = []
        for i in range(num_requests):
            if verify_slot_starts[i] is None:
                continue
            count = verify_slot_starts[i][1] + 1
            extras.append(np.repeat(batch.logit_masks[i : i + 1], count, axis=0))
        mask_np = np.concatenate([base] + extras, axis=0) if extras else base
        sampling_masks_t = torch.as_tensor(mask_np, device=device, dtype=torch.bool)

    batch._verify_slot_starts = verify_slot_starts
    batch._verify_block_offset = len(inferlet_indices)

    return {
        "indices_for_logits": indices_for_logits,
        "temperatures": temperatures_t,
        "sampler_groups": sampler_groups,
        "top_k": top_k_t,
        "top_p": top_p_t,
        "min_p": min_p_t,
        "seeds": seeds_t,
        "sampling_masks": sampling_masks_t,
    }
