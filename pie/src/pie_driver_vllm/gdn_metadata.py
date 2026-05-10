"""Build vllm `GDNAttentionMetadata` from pie's per-batch CSR layout.

vllm's GDN backend (`vllm/v1/attention/backends/gdn_attn.py`) consumes a
struct that splits a batch into prefill / decode (and spec-decode, which
pie does not exercise via vllm — pie's NGRAM drafter is engine-side and
never produces > 1 query/req on this path). The struct also carries:

* per-request **state-slot indices** — `non_spec_state_indices_tensor`,
  one int32 per non-spec request pointing at the mamba state slot for
  that request. The slot id is computed via `_state_slot_for_request`
  below, deliberately isolated behind a single helper so #108's
  per-request slot allocator can replace the current first-page-id
  derivation in one place. See `kv_cache.py` for the addressing
  rationale and the known fork-aliasing limitation.
* per-prefill **conv1d metadata** — `nums_dict / batch_ptr /
  token_chunk_offset_ptr`, computed once per batch via
  `compute_causal_conv1d_metadata`.
* prefill **chunk indices/offsets** for FLA's chunked gated-delta-rule
  (qwen3-next / qwen3.5-moe specific) — computed via
  `prepare_chunk_indices` / `prepare_chunk_offsets` on the non-spec
  `query_start_loc_cpu`.
* `has_initial_state` — `True` for non-spec rows whose context_lens > 0
  (i.e., recurrent state is carried over from a prior fire_batch). vllm's
  builder emits the same shape: full non-spec batch length (in our
  no-spec path that is the whole batch); the GDN forward then consumes
  the full vector when `causal_conv1d_fn` is dispatched on prefill rows
  (it walks `non_spec_query_start_loc` to know which token slice belongs
  to which row). See `vllm/v1/attention/backends/gdn_attn.py:build` for
  the canonical reference and `gdn_linear_attn.py:_forward` for the
  consumption pattern.

We don't populate any spec-decode fields (`spec_*`, `num_accepted_tokens`,
`spec_token_indx`, …) — pie's runtime won't issue spec-decode batches
through this driver until ticket #107's spec_decode follow-up.
"""

from __future__ import annotations

from typing import Any

import torch


def _state_slot_tensor(
    state_slot_ids: list[int],
    expected_batch: int,
) -> torch.Tensor:
    """Return one int32 mamba state-slot id per request, on CPU.

    Slots come from `mamba_state.MambaSlotAllocator` (#108 phase 5a),
    which is keyed on the stable per-context `ContextId` carried in
    `BatchedForwardPassRequest.context_ids` (shmem A_CONTEXT_IDS=27,
    Rust `runtime/src/inference/request.rs:431`). Pie mints a fresh
    ContextId on fork, so siblings reach the driver with distinct ids
    and the page-id-keyed alias hazard from #107 cannot occur — the
    duplicate-slot guard from the previous scheme is therefore dropped.

    Returning a CPU tensor here keeps the call site indexed-copy free;
    the caller stages the tensor onto the device via `non_blocking=True`.
    """
    if len(state_slot_ids) != expected_batch:
        raise RuntimeError(
            f"GDN state slot count {len(state_slot_ids)} does not match "
            f"batch size {expected_batch}; the runtime must publish a "
            "ContextId per non-spec request via "
            "BatchedForwardPassRequest.context_ids."
        )
    return torch.tensor(state_slot_ids, dtype=torch.int32)


def build_gdn_metadata(
    *,
    qo_indptr: torch.Tensor,            # int32 (batch+1,) on GPU or CPU
    seq_lens: torch.Tensor,             # int32 (batch,) GPU
    num_computed_tokens: torch.Tensor,  # int32 (batch,) GPU
    num_actual_tokens: int,
    device: torch.device,
    state_slot_ids: list[int],
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

    # Per-request state-slot index. The slot list is supplied by the
    # caller (typically `forward_pass.transform` via
    # `MambaSlotAllocator.slots_for(context_ids)`). See `_state_slot_tensor`
    # for the rationale (#108 phase 5a — ContextId-keyed allocation).
    state_slots_cpu = _state_slot_tensor(state_slot_ids, batch)
    non_spec_state_indices_tensor = state_slots_cpu.to(
        device, dtype=torch.int32, non_blocking=True
    )

    # `has_initial_state[req]` flags rows whose recurrent state is
    # carried over from a prior fire_batch (i.e., `num_computed_tokens >
    # 0`). Length = full non-spec batch (= `batch` in our no-spec path),
    # matching vllm 0.20.0's GDN builder: the field is computed over the
    # non-spec batch and passed unsliced into `GDNAttentionMetadata`.
    # `gdn_linear_attn._forward` then dispatches `causal_conv1d_fn` only
    # when `num_prefills > 0` and walks `non_spec_query_start_loc` to
    # pick the per-row token slices, so decode rows in a mixed batch are
    # tolerated despite carrying a `True` flag.
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
