"""Build vllm `GDNAttentionMetadata` from pie's per-batch CSR layout.

vllm's GDN backend (`vllm/v1/attention/backends/gdn_attn.py`) consumes a
struct that splits a batch into prefill / decode (and spec-decode, which
pie does not exercise via vllm — pie's NGRAM drafter is engine-side and
never produces > 1 query/req on this path). The struct also carries:

* per-request **state-slot indices** — `non_spec_state_indices_tensor`,
  one int32 per request pointing at the mamba state slot for that request.
  Pie uses the request's first KV page id as the state slot (see
  `kv_cache.py` for the addressing rationale).
* per-prefill **conv1d metadata** — `nums_dict / batch_ptr /
  token_chunk_offset_ptr`, computed once per batch via
  `compute_causal_conv1d_metadata`.
* prefill **chunk indices/offsets** for FLA's chunked gated-delta-rule
  (qwen3-next / qwen3.5-moe specific) — computed via
  `prepare_chunk_indices` / `prepare_chunk_offsets` on the non-spec
  `query_start_loc_cpu`.
* `has_initial_state` — `True` for prefill rows that carry pre-existing
  recurrent state (i.e., `num_computed_tokens > 0`).

We don't populate any spec-decode fields (`spec_*`, `num_accepted_tokens`,
`spec_token_indx`, …) — pie's runtime won't issue spec-decode batches
through this driver until ticket #107's spec_decode follow-up.
"""

from __future__ import annotations

from typing import Any

import torch


def build_gdn_metadata(
    *,
    qo_indptr: torch.Tensor,            # int32 (batch+1,) on GPU or CPU
    kv_page_indices: torch.Tensor,      # int32 (total_pages,) GPU or CPU
    kv_page_indptr: torch.Tensor,       # int32 (batch+1,) GPU or CPU
    seq_lens: torch.Tensor,             # int32 (batch,) GPU
    num_computed_tokens: torch.Tensor,  # int32 (batch,) GPU
    num_actual_tokens: int,
    device: torch.device,
) -> Any:
    """Construct a `GDNAttentionMetadata` from pie batch fields.

    `seq_lens` and `num_computed_tokens` are computed by the caller (in
    `forward_pass`) from the FlashInfer/`CommonAttentionMetadata` build —
    avoiding redundant CSR sweeps.
    """
    from ._vllm_compat import get_gdn_backend, get_mamba_helpers

    _, GDNAttentionMetadata, _ = get_gdn_backend()
    _, compute_causal_conv1d_metadata = get_mamba_helpers()

    # Move CPU views once. `query_start_loc` is small (≤ batch+1), so the
    # round-trip is cheap and lets us classify decode-vs-prefill on CPU.
    if qo_indptr.device.type == "cpu":
        query_start_loc_cpu = qo_indptr.to(torch.int32)
    else:
        query_start_loc_cpu = qo_indptr.to(torch.int32).cpu()
    if qo_indptr.device.type == "cuda":
        query_start_loc_gpu = qo_indptr.to(torch.int32)
    else:
        query_start_loc_gpu = query_start_loc_cpu.to(device, non_blocking=True)

    if kv_page_indptr.device.type == "cpu":
        kv_page_indptr_cpu = kv_page_indptr.to(torch.int32)
    else:
        kv_page_indptr_cpu = kv_page_indptr.to(torch.int32).cpu()
    if kv_page_indices.device.type == "cpu":
        kv_page_indices_cpu = kv_page_indices.to(torch.int32)
    else:
        kv_page_indices_cpu = kv_page_indices.to(torch.int32).cpu()

    batch = int(query_start_loc_cpu.shape[0]) - 1
    query_lens_cpu = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]

    # Decode rows = qlen == 1; prefill rows = qlen > 1. Pie never produces
    # qlen == 0 on this path (an empty request would have been excluded
    # from the batch by the runtime).
    decode_mask = query_lens_cpu == 1
    num_decodes = int(decode_mask.sum().item())
    num_prefills = batch - num_decodes
    num_decode_tokens = num_decodes  # qlen=1 each
    num_prefill_tokens = int(query_lens_cpu.sum().item()) - num_decode_tokens

    # Per-request state-slot index. Pie addresses mamba state by the
    # request's first KV page id (see `kv_cache.py`); pull that off CPU
    # CSR and ship a contiguous int32 tensor to GPU.
    first_pages_cpu = kv_page_indices_cpu[kv_page_indptr_cpu[:-1]]
    non_spec_state_indices_tensor = first_pages_cpu.to(
        device, dtype=torch.int32, non_blocking=True
    )

    # `has_initial_state[req]` flags prefill rows whose recurrent state is
    # carried over from a prior fire_batch (i.e., `num_computed_tokens >
    # 0`). Decode rows always have non-zero state but the field is sliced
    # to non-spec prefill rows only.
    has_initial_state: torch.Tensor | None = None
    nums_dict = None
    batch_ptr = None
    token_chunk_offset_ptr = None
    chunk_indices: torch.Tensor | None = None
    chunk_offsets: torch.Tensor | None = None
    if num_prefills > 0:
        has_initial_state = num_computed_tokens > 0
        # vllm 0.20.0's compute_causal_conv1d_metadata signature:
        # (query_start_loc_cpu, *, device=...). Earlier versions exposed
        # a positional-only signature without `device`; both are tolerated
        # via try/except so the bridge survives a vllm minor bump.
        try:
            nums_dict, batch_ptr, token_chunk_offset_ptr = compute_causal_conv1d_metadata(
                query_start_loc_cpu, device=device
            )
        except TypeError:
            nums_dict, batch_ptr, token_chunk_offset_ptr = compute_causal_conv1d_metadata(
                query_start_loc_cpu
            )

        # FLA chunk indices/offsets — qwen3-next / qwen3.5-moe-specific.
        # vllm 0.20.0 imports these from `model_executor.layers.fla.ops.index`.
        try:
            from vllm.model_executor.layers.fla.ops.index import (
                prepare_chunk_indices,
                prepare_chunk_offsets,
            )
            from vllm.model_executor.layers.fla.ops.utils import FLA_CHUNK_SIZE

            chunk_indices = prepare_chunk_indices(
                query_start_loc_cpu, FLA_CHUNK_SIZE
            ).to(device, non_blocking=True)
            chunk_offsets = prepare_chunk_offsets(
                query_start_loc_cpu, FLA_CHUNK_SIZE
            ).to(device, non_blocking=True)
        except ImportError:
            # Older vllm without FLA chunk pre-computation; the GDN forward
            # falls back to its own on-the-fly path. Leave fields as None.
            pass

    return GDNAttentionMetadata(
        num_prefills=num_prefills,
        num_prefill_tokens=num_prefill_tokens,
        num_decodes=num_decodes,
        num_decode_tokens=num_decode_tokens,
        num_spec_decodes=0,
        num_spec_decode_tokens=0,
        num_actual_tokens=num_actual_tokens,
        has_initial_state=has_initial_state,
        spec_query_start_loc=None,
        non_spec_query_start_loc=query_start_loc_gpu,
        spec_state_indices_tensor=None,
        non_spec_state_indices_tensor=non_spec_state_indices_tensor,
        spec_sequence_masks=None,
        spec_token_indx=None,
        non_spec_token_indx=None,
        num_accepted_tokens=None,
        chunk_indices=chunk_indices,
        chunk_offsets=chunk_offsets,
        nums_dict=nums_dict,
        batch_ptr=batch_ptr,
        token_chunk_offset_ptr=token_chunk_offset_ptr,
    )
