"""
Common modeling components for the Pie driver.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import numpy as np
import sys
import math

from typing import Callable, Any

from pie_kernels.sampling import (  # noqa: F401
    sampling_from_probs,
    top_p_sampling_from_probs,
    top_k_sampling_from_probs,
    min_p_sampling_from_probs,
    top_k_top_p_sampling_from_probs,
)
# NOTE: We intentionally route token sampling through the `*_from_probs`
# kernels rather than flashinfer's fused `*_from_logits` variants. Mapping
# pie's most common sampler (TopP, type 3) onto
# `top_k_top_p_sampling_from_logits` with `top_k=vocab_size` (no-op filter)
# was measured slower than the explicit `scaled_softmax` +
# `top_p_sampling_from_probs` pair on Qwen3-0.6B / A40 — the kernel still
# pays for the unused top-k step. Revisit if a future flashinfer release
# short-circuits or if sampler 1 (Multinomial) becomes the dominant type.

if torch.cuda.is_available():
    NUM_SM = torch.cuda.get_device_properties(
        torch.device("cuda")
    ).multi_processor_count
else:
    NUM_SM = 108


# Sampler type IDs (must agree with Rust `Sampler::type_id()`):
#   0  Distribution        — top-k probs (post-softmax, with temperature)
#   1  Multinomial         } token-producing
#   2  TopK                }
#   3  TopP                }   need temperature-scaled probs
#   4  MinP                }
#   5  TopKTopP            }
#   6  Embedding           — placeholder, not wired in worker
#   7  RawLogits           — full vocab f32 bytes, no softmax needed
#   8  Logprob(t)          } need log_softmax (no temperature)
#   9  Logprobs([t1..tk])  }
#  10  Entropy             }
TOKEN_SAMPLING_TYPES = frozenset({1, 2, 3, 4, 5})
DIST_TYPES = frozenset({0})
RAW_LOGITS_TYPE = 7
LOGPROB_TYPES = frozenset({8, 9, 10})
_TOKEN_COPY_BUFFERS: dict[int, torch.Tensor] = {}
_TOKEN_COPY_EVENTS: dict[int, torch.cuda.Event] = {}


def _cuda_device_index(device: torch.device) -> int:
    if device.index is not None:
        return int(device.index)
    return torch.cuda.current_device()


def _tokens_to_list(tokens: torch.Tensor) -> list[int]:
    """Copy sampled ids to CPU using vLLM's pinned-buffer/event pattern.

    Direct `Tensor.tolist()` on a CUDA tensor can force a device-wide sync.
    A non-blocking copy into pinned CPU memory followed by a CUDA event sync
    waits only for the token-copy stream dependency before converting the
    already-host-visible tensor to Python ints.
    """
    if not tokens.is_cuda:
        return tokens.tolist()

    flat = tokens.reshape(-1)
    n = int(flat.numel())
    device_idx = _cuda_device_index(flat.device)
    pinned = _TOKEN_COPY_BUFFERS.get(device_idx)
    if pinned is None or pinned.dtype != flat.dtype or pinned.numel() < n:
        pinned = torch.empty(
            max(n, 1024),
            dtype=flat.dtype,
            device="cpu",
            pin_memory=True,
        )
        _TOKEN_COPY_BUFFERS[device_idx] = pinned
    event = _TOKEN_COPY_EVENTS.get(device_idx)
    if event is None:
        event = torch.cuda.Event()
        _TOKEN_COPY_EVENTS[device_idx] = event

    view = pinned[:n]
    view.copy_(flat, non_blocking=True)
    event.record(torch.cuda.current_stream(flat.device))
    event.synchronize()
    return view.tolist()


def scaled_softmax(logits, temperatures, greedy_threshold=1e-5):
    """Temperature-scaled softmax with safe handling of T≈0 (greedy).

    Clamps temperature to `greedy_threshold` before dividing. PyTorch's
    softmax is numerically stable (subtracts max), so `softmax(logits / 1e-5)`
    yields a near-one-hot distribution at the argmax — functionally
    indistinguishable from a true one-hot for downstream sampling.

    This is much cheaper than the previous "safe" impl which allocated a
    `(batch × vocab)` one-hot tensor and ran a `torch.where` over the full
    probs tensor for every step.
    """
    if temperatures.ndim == 1:
        temperatures = temperatures.unsqueeze(1)
    return torch.softmax(
        logits / temperatures.clamp(min=greedy_threshold), dim=-1
    )


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
        Dictionary containing 'tokens' (list[int]), 'dists' (list[tuple]),
        'logits' (list[bytes|None]), 'logprobs' (list[list[float]|None]),
        and 'entropies' (list[float|None]).
    """
    if not sampling_metadata.get("indices_for_logits"):
        return {"tokens": [], "dists": [], "logits": [], "logprobs": [], "entropies": []}

    indices_for_logits = sampling_metadata["indices_for_logits"]

    # Stage 1: Compute logits via LM head.
    # We deliberately skip an explicit `torch.isnan(...).any()` check here:
    # the reduction forced a CPU↔GPU sync on every fire_batch, and pre-trained
    # weights don't produce NaNs in normal inference. `torch.nan_to_num` below
    # still scrubs any stray NaN as a defense-in-depth.
    logits_input = hidden_states[indices_for_logits]
    nan_indices: list[int] = []
    logits_input = torch.nan_to_num(logits_input)

    # Apply lm_head_fn
    logits = lm_head_fn(logits_input)

    # Apply sampling mask
    if (sampling_masks := sampling_metadata.get("sampling_masks")) is not None:
        logits.masked_fill_(~sampling_masks, -float("inf"))

    sampler_groups = sampling_metadata["sampler_groups"]
    num_logit_requests = len(indices_for_logits)

    greedy_mask = sampling_metadata.get("greedy_mask")
    if greedy_mask is None:
        greedy_mask = np.zeros(num_logit_requests, dtype=np.bool_)

    def _non_greedy_slots(indices: list[int]) -> list[int]:
        return [idx for idx in indices if not bool(greedy_mask[idx])]

    needs_probs = any(idx in DIST_TYPES for idx in sampler_groups) or any(
        _non_greedy_slots(indices)
        for idx, indices in sampler_groups.items()
        if idx in TOKEN_SAMPLING_TYPES
    )
    needs_log_probs = any(idx in LOGPROB_TYPES for idx in sampler_groups)

    if needs_probs:
        probs = scaled_softmax(logits, sampling_metadata["temperatures"])
    else:
        probs = None

    # log_softmax is computed on the un-temperatured logits — Logprob/Entropy
    # report model-native quantities. Cost is one extra kernel only when
    # those samplers are present.
    #
    # Sparse-batch optimization: when only a small fraction of the batch
    # actually needs log_probs (the typical "1-2 logprob slots in a mostly-
    # greedy batch" case), gather those rows first so log_softmax operates
    # on a (k, vocab) tensor instead of (batch, vocab). Cuts the per-batch
    # log_softmax cost from O(batch * vocab) to O(k * vocab); we hit it
    # whenever an inferlet asks for a logprob/entropy sample alongside
    # other requests in the same batch.
    log_probs = None
    lp_slot_to_row: dict[int, int] = {}
    if needs_log_probs:
        lp_slots = sorted({
            s for t in LOGPROB_TYPES if t in sampler_groups
            for s in sampler_groups[t]
        })
        num_lp = len(lp_slots)
        if num_lp > 0 and num_lp <= num_logit_requests // 2:
            lp_rows_t = torch.as_tensor(lp_slots, device=device, dtype=torch.long)
            log_probs = F.log_softmax(
                logits.index_select(0, lp_rows_t).to(torch.float32), dim=-1
            )
            lp_slot_to_row = {s: i for i, s in enumerate(lp_slots)}
        else:
            log_probs = F.log_softmax(logits.to(torch.float32), dim=-1)
            lp_slot_to_row = {s: s for s in lp_slots}

    final_dists = [None] * num_logit_requests
    final_logits = [None] * num_logit_requests
    final_logprobs: list[list[float] | None] = [None] * num_logit_requests
    final_entropies: list[float | None] = [None] * num_logit_requests
    final_tokens_tensor = torch.empty(
        num_logit_requests, dtype=torch.int32, device=device
    )

    # Pre-built tensors for sampler params (no per-group dict extraction)
    top_k_all = sampling_metadata["top_k"]
    top_p_all = sampling_metadata["top_p"]
    min_p_all = sampling_metadata["min_p"]
    seeds_all = sampling_metadata.get("seeds")
    # Per-sampler ragged label lists for Logprob/Logprobs.
    label_ids = sampling_metadata.get("sampler_label_ids")
    label_indptr = sampling_metadata.get("sampler_label_indptr")

    for sampler_idx, indices in sampler_groups.items():
        if not indices:
            continue

        indices_tensor = torch.tensor(indices, device=device, dtype=torch.long)

        if sampler_idx == RAW_LOGITS_TYPE:
            # Raw-logits mode — serialize pre-softmax logits as f32 bytes.
            group_logits = logits.index_select(0, indices_tensor)
            _process_raw_logits(indices, group_logits, final_logits)
            continue

        if sampler_idx in LOGPROB_TYPES:
            # Translate original slot positions → row positions inside
            # `log_probs` (which may be a sparse subset of the batch).
            subset_rows = torch.as_tensor(
                [lp_slot_to_row[s] for s in indices], device=device, dtype=torch.long
            )
            if sampler_idx == 8:
                _process_logprob(
                    indices, log_probs, subset_rows, label_ids, label_indptr, final_logprobs
                )
            elif sampler_idx == 9:
                _process_logprobs_many(
                    indices, log_probs, subset_rows, label_ids, label_indptr, final_logprobs
                )
            else:  # sampler_idx == 10
                _process_entropy(indices, log_probs, subset_rows, final_entropies)
            continue

        if sampler_idx in TOKEN_SAMPLING_TYPES:
            greedy_indices = [idx for idx in indices if bool(greedy_mask[idx])]
            if greedy_indices:
                greedy_rows = torch.as_tensor(
                    greedy_indices, device=device, dtype=torch.long
                )
                greedy_tokens = logits.index_select(0, greedy_rows).argmax(dim=-1)
                if greedy_tokens.dtype != torch.int32:
                    greedy_tokens = greedy_tokens.to(torch.int32)
                final_tokens_tensor.scatter_(0, greedy_rows, greedy_tokens)

            indices = _non_greedy_slots(indices)
            if not indices:
                continue
            indices_tensor = torch.as_tensor(indices, device=device, dtype=torch.long)

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
            # Seed: per-sampler u32; 0 means "user didn't supply one" — pass
            # None in that case so flashinfer falls back to its default RNG.
            group_seeds = (
                seeds_all[indices_tensor] if seeds_all is not None else None
            )

            sampled = _execute_sampler(
                sampler_idx,
                group_probs,
                group_top_k,
                group_top_p,
                group_min_p,
                group_seeds,
            )
            if sampled.dtype != torch.int32:
                sampled = sampled.to(torch.int32)

            final_tokens_tensor.scatter_(0, indices_tensor, sampled)

    # Stage 5: Combine results into plain Python ints for the wire path.
    # Use a pinned CPU staging buffer instead of CUDA `tolist()` so this
    # synchronization matches vLLM's hot path.
    final_tokens_list = _tokens_to_list(final_tokens_tensor)

    return {
        "tokens": final_tokens_list,
        "dists": final_dists,
        "logits": final_logits,
        "logprobs": final_logprobs,
        "entropies": final_entropies,
        "nan_indices": nan_indices,
    }


def _process_raw_logits(
    indices: list[int],
    group_logits: torch.Tensor,
    final_logits: list[bytes | None],
) -> None:
    """Serialize raw (pre-softmax) logits as native-endian f32 bytes.

    `group_logits` shape: (len(indices), vocab_size). One batched D2H transfer
    happens here; per-row `.tobytes()` is a host-side memcpy of vocab_size * 4
    bytes — the minimum payload size for full-vocab logits.
    """
    arr = group_logits.contiguous().to(torch.float32).cpu().numpy()
    for i, original_idx in enumerate(indices):
        final_logits[original_idx] = arr[i].tobytes()


def _process_logprob(
    indices: list[int],
    log_probs: torch.Tensor,
    subset_rows: torch.Tensor,
    label_ids: np.ndarray,
    label_indptr: np.ndarray,
    final_logprobs: list[list[float] | None],
) -> None:
    """Single-label logprob: one float per slot.

    `log_probs` may be the full-batch tensor or a sparse-subset gather; in
    either case `subset_rows[i]` is the row in `log_probs` for slot
    `indices[i]`. Result is wrapped in a length-1 list to keep the wire
    shape uniform with Logprobs (multi).
    """
    labels = [int(label_ids[label_indptr[s]]) for s in indices]
    lab_t = torch.as_tensor(labels, device=log_probs.device, dtype=torch.long)
    rows = log_probs.index_select(0, subset_rows)
    vals = rows.gather(1, lab_t.unsqueeze(1)).squeeze(1).tolist()
    for i, original_idx in enumerate(indices):
        final_logprobs[original_idx] = [vals[i]]


def _process_logprobs_many(
    indices: list[int],
    log_probs: torch.Tensor,
    subset_rows: torch.Tensor,
    label_ids: np.ndarray,
    label_indptr: np.ndarray,
    final_logprobs: list[list[float] | None],
) -> None:
    """Multi-label logprobs: K floats per slot (K can vary per slot).

    Builds a flat (row, col) gather over all slots × all labels in one
    .tolist() call to amortize the device sync. `subset_rows[i]` is the
    row in `log_probs` for slot `indices[i]`.
    """
    counts: list[int] = []
    flat_labels: list[np.ndarray] = []
    for s in indices:
        start = int(label_indptr[s])
        end = int(label_indptr[s + 1])
        flat_labels.append(label_ids[start:end])
        counts.append(end - start)

    if sum(counts) == 0:
        for s in indices:
            final_logprobs[s] = []
        return

    flat = np.concatenate(flat_labels) if flat_labels else np.zeros(0, dtype=np.int32)
    subset_rows_np = subset_rows.cpu().numpy()
    rows_np = np.repeat(subset_rows_np, counts)
    rows_t = torch.as_tensor(rows_np, device=log_probs.device, dtype=torch.long)
    cols_t = torch.as_tensor(flat, device=log_probs.device, dtype=torch.long)
    vals = log_probs[rows_t, cols_t].tolist()

    pos = 0
    for s, k in zip(indices, counts):
        final_logprobs[s] = vals[pos : pos + k]
        pos += k


def _process_entropy(
    indices: list[int],
    log_probs: torch.Tensor,
    subset_rows: torch.Tensor,
    final_entropies: list[float | None],
) -> None:
    """Shannon entropy of the unscaled next-token distribution.

    H(p) = -sum(p * log p). Computed via probs = exp(log_probs) to avoid a
    second softmax. Numerically stable because log_probs comes from
    log_softmax (max-subtracted internally). `subset_rows[i]` is the row
    in `log_probs` for slot `indices[i]`.
    """
    rows = log_probs.index_select(0, subset_rows)
    probs = rows.exp()
    H = -(probs * rows).sum(dim=-1)
    H_list = H.tolist()
    for i, original_idx in enumerate(indices):
        final_entropies[original_idx] = H_list[i]


def _process_distributions(
    indices: list[int],
    group_probs: torch.Tensor,
    final_dists: list[tuple[list[int], list[float]] | None],
    group_top_k: torch.Tensor,
) -> None:
    """Process distribution requests.

    top_k=0 means "return full distribution" (all vocab tokens).
    """
    vocab_size = group_probs.shape[-1]
    # Replace 0 with vocab_size (0 = return all)
    effective_k = group_top_k.clone()
    effective_k[effective_k == 0] = vocab_size
    top_k_list = effective_k.tolist()
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
    seeds: torch.Tensor | None,
) -> torch.Tensor:
    """Execute the appropriate sampling operation.

    Args:
        sampler_idx: Sampler type (1=uniform, 2=top_k, 3=top_p, 4=min_p, 5=top_k_top_p)
        group_probs: Probability tensor for this group
        top_k: Pre-indexed top_k tensor for this group
        top_p: Pre-indexed top_p tensor for this group
        min_p: Pre-indexed min_p tensor for this group
        seeds: Optional per-sampler u64 seeds (length = group_probs.size(0)).
            Zeros are treated as "no seed supplied" → fall back to the
            kernel's default RNG. If at least one slot has a non-zero seed
            we forward the whole tensor (flashinfer accepts a per-row seed
            tensor and seeds each row independently).

    Returns:
        Sampled token indices
    """
    # `seeds` is None when no inferlet in this batch asked for
    # determinism (the host-side `has_user_seeds` gate in batching.py
    # short-circuits the device transfer). When non-None, every entry is
    # honoured even if some are 0 — flashinfer treats 0 as a valid seed.
    seed_arg = seeds

    if sampler_idx == 1:
        result = sampling_from_probs(group_probs, seed=seed_arg)

    elif sampler_idx == 2:
        result = top_k_sampling_from_probs(group_probs, top_k=top_k, seed=seed_arg)

    elif sampler_idx == 3:
        result = top_p_sampling_from_probs(group_probs, top_p=top_p, seed=seed_arg)

    elif sampler_idx == 4:
        result = min_p_sampling_from_probs(group_probs, min_p=min_p, seed=seed_arg)

    elif sampler_idx == 5:
        result = top_k_top_p_sampling_from_probs(
            group_probs, top_k=top_k, top_p=top_p, seed=seed_arg
        )

    else:
        raise ValueError(f"Unknown sampler index: {sampler_idx}")

    return result


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
            # Worst-case: pad to max grid capacity.
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
