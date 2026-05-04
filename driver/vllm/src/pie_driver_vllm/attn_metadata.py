"""Translate pie's batch metadata into vllm's `CommonAttentionMetadata`.

Pie emits metadata in CSR form (qo_indptr, kv_page_indices, kv_page_indptr,
kv_last_page_lens). vllm consumes a backend-agnostic struct
(`query_start_loc`, `seq_lens`, `block_table_tensor`, `slot_mapping`). Block
IDs come through unchanged — only the shape changes.

Conversions:
  - `qo_indptr`              → `query_start_loc`     (rename only)
  - per-request page count   = kv_page_indptr.diff()
  - `seq_lens`               = (page_count - 1) * page_size + last_page_len
  - CSR (`kv_page_indices`, `kv_page_indptr`)
                             → dense `block_table` 2D (right-padded per row)
  - `qo_indptr` + page state → `slot_mapping` (flat write index per query token)
"""

from __future__ import annotations

import numba
import numpy as np
import torch


@numba.njit(cache=True, parallel=False)
def _build_block_table(
    kv_page_indices: np.ndarray,    # int32 (total_pages,)
    kv_page_indptr: np.ndarray,     # int32 (batch+1,)
    max_blocks_per_req: int,
    out: np.ndarray,                # int32 (batch, max_blocks_per_req)
):
    """Right-pad CSR block lists into a dense 2D block table.

    Out-of-range slots are left as zeros — vllm's kernels guard via seq_lens.
    """
    batch = kv_page_indptr.shape[0] - 1
    for r in range(batch):
        start = kv_page_indptr[r]
        end = kv_page_indptr[r + 1]
        for j in range(end - start):
            out[r, j] = kv_page_indices[start + j]


@numba.njit(cache=True, parallel=False)
def _build_slot_mapping(
    qo_indptr: np.ndarray,           # int32 (batch+1,)
    kv_page_indices: np.ndarray,     # int32 (total_pages,)
    kv_page_indptr: np.ndarray,      # int32 (batch+1,)
    seq_lens: np.ndarray,            # int32 (batch,)
    page_size: int,
    out: np.ndarray,                 # int64 (num_tokens,)
):
    """Per-query-token: flat slot index into the page pool.

    For request r with query length q_r and total context s_r:
      - The query covers absolute positions [s_r - q_r, s_r) within the request.
      - For each absolute position p, page_idx = p // page_size,
        offset = p % page_size, slot = block_id * page_size + offset.
    """
    batch = qo_indptr.shape[0] - 1
    for r in range(batch):
        q_start = qo_indptr[r]
        q_end = qo_indptr[r + 1]
        q_len = q_end - q_start
        s_r = seq_lens[r]
        page_base = kv_page_indptr[r]
        first_pos = s_r - q_len
        for i in range(q_len):
            p = first_pos + i
            page_idx = p // page_size
            offset = p % page_size
            block_id = kv_page_indices[page_base + page_idx]
            out[q_start + i] = block_id * page_size + offset


def build_common_metadata(
    *,
    qo_indptr: torch.Tensor,
    kv_page_indices: torch.Tensor,
    kv_page_indptr: torch.Tensor,
    kv_last_page_lens: torch.Tensor,
    page_size: int,
    device: torch.device,
):
    """Produce a `CommonAttentionMetadata` from pie's batch dict fields.

    All inputs may live on CPU or GPU; outputs land on `device`. We do the
    Numba work on numpy arrays (CPU) and copy the result to GPU at the end.
    """
    from ._vllm_compat import CommonAttentionMetadata

    qo_np = qo_indptr.to(torch.int32).cpu().numpy()
    kv_idx_np = kv_page_indices.to(torch.int32).cpu().numpy()
    kv_indptr_np = kv_page_indptr.to(torch.int32).cpu().numpy()
    kv_last_np = kv_last_page_lens.to(torch.int32).cpu().numpy()

    batch = qo_np.shape[0] - 1
    num_tokens = int(qo_np[-1])

    # seq_lens = (num_pages - 1) * page_size + last_page_len
    pages_per_req = (kv_indptr_np[1:] - kv_indptr_np[:-1]).astype(np.int32)
    # Defensive: pages_per_req must be > 0 for any request that has KV state.
    # A request with 0 pages and a non-empty query is impossible in pie.
    seq_lens_np = ((pages_per_req - 1) * page_size + kv_last_np).astype(np.int32)

    # block_table: (batch, max_blocks_per_req) right-padded with 0
    max_blocks = int(pages_per_req.max()) if batch > 0 else 0
    block_table_np = np.zeros((batch, max_blocks), dtype=np.int32)
    if batch > 0 and max_blocks > 0:
        _build_block_table(kv_idx_np, kv_indptr_np, max_blocks, block_table_np)

    # slot_mapping: (num_tokens,) absolute slot per query token
    slot_mapping_np = np.zeros(num_tokens, dtype=np.int64)
    if num_tokens > 0:
        _build_slot_mapping(
            qo_np, kv_idx_np, kv_indptr_np, seq_lens_np, page_size, slot_mapping_np
        )

    # Move to GPU
    query_start_loc_cpu = torch.from_numpy(qo_np)
    query_start_loc = query_start_loc_cpu.to(device, non_blocking=True)
    seq_lens = torch.from_numpy(seq_lens_np).to(device, non_blocking=True)
    block_table_tensor = torch.from_numpy(block_table_np).to(device, non_blocking=True)
    slot_mapping = torch.from_numpy(slot_mapping_np).to(device, non_blocking=True)

    query_lens = qo_np[1:] - qo_np[:-1]
    max_query_len = int(query_lens.max()) if batch > 0 else 0
    max_seq_len = int(seq_lens_np.max()) if batch > 0 else 0

    return CommonAttentionMetadata(
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc_cpu,
        seq_lens=seq_lens,
        num_reqs=batch,
        num_actual_tokens=num_tokens,
        max_query_len=max_query_len,
        max_seq_len=max_seq_len,
        block_table_tensor=block_table_tensor,
        slot_mapping=slot_mapping,
        causal=True,
    )
