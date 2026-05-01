"""Translate pie's CSR-form batch metadata into an SGLang `ForwardBatch`.

Pie emits per-batch metadata in CSR form:
  - `qo_indptr[r..r+1]`     → query token range for request r
  - `kv_page_indices`       → flat list of block IDs across all requests
  - `kv_page_indptr[r..r+1]`→ block range for request r
  - `kv_last_page_lens[r]`  → valid token count in request r's last block

SGLang's `ForwardBatch` (and the attention backends downstream of it) consume:
  - `req_pool_indices: (batch,)`        — slot in `req_to_token_pool` per request
  - `seq_lens: (batch,)`                — total sequence length per request
  - `extend_seq_lens, extend_prefix_lens` — prefill split
  - `out_cache_loc: (num_query_tokens,)` — destination slot per query token
  - `req_to_token_pool.req_to_token[req_pool_idx, :seq_len]` — slot per token

We bypass `req_to_token_pool.alloc()` and write rows directly: for request r,
expand its CSR block list into per-token slots `block_id * page_size + offset`,
where `block_id` is owned by pie's Rust scheduler. `out_cache_loc` is the
per-query-token slice of those same rows (so we compute it from the rows
rather than running a second numba kernel).
"""

from __future__ import annotations

from typing import Any

import numba
import numpy as np
import torch


@numba.njit(cache=True, parallel=False)
def _build_req_to_token_rows(
    kv_page_indices: np.ndarray,    # int32 (total_pages,)
    kv_page_indptr: np.ndarray,     # int32 (batch+1,)
    seq_lens: np.ndarray,           # int32 (batch,)
    page_size: int,
    out: np.ndarray,                # int32 (batch, max_seq_len)
):
    """Expand CSR block IDs into a `(batch, max_seq_len)` per-token slot table.

    Each row r is filled with `seq_lens[r]` valid slot indices; the rest is
    left as zeros (SGLang's kernels guard via `seq_lens`).
    """
    batch = kv_page_indptr.shape[0] - 1
    for r in range(batch):
        s_r = seq_lens[r]
        page_base = kv_page_indptr[r]
        for i in range(s_r):
            block_id = kv_page_indices[page_base + i // page_size]
            out[r, i] = block_id * page_size + (i % page_size)


def build_sglang_forward_batch(
    *,
    runner: Any,                     # sglang ModelRunner
    inputs: dict,
    page_size: int,
    device: torch.device,
) -> Any:
    """Build a `ForwardBatch` from pie's `inputs` dict."""
    from sglang.srt.model_executor.forward_batch_info import (
        ForwardBatch,
        ForwardMode,
    )

    # ---- Pie's CSR metadata to numpy (cheap; already int32 on CPU) ----
    qo_indptr_np = inputs["qo_indptr"].cpu().to(torch.int32).numpy()
    kv_idx_np = inputs["kv_page_indices"].cpu().to(torch.int32).numpy()
    kv_indptr_np = inputs["kv_page_indptr"].cpu().to(torch.int32).numpy()
    kv_last_np = inputs["kv_last_page_lens"].cpu().to(torch.int32).numpy()

    batch_size = qo_indptr_np.shape[0] - 1
    num_query_tokens = int(qo_indptr_np[-1])

    # Per-request total seq_len = (num_pages - 1) * page_size + last_page_len
    pages_per_req = (kv_indptr_np[1:] - kv_indptr_np[:-1]).astype(np.int32)
    seq_lens_np = ((pages_per_req - 1) * page_size + kv_last_np).astype(np.int32)
    extend_seq_lens_np = (qo_indptr_np[1:] - qo_indptr_np[:-1]).astype(np.int32)
    extend_prefix_lens_np = (seq_lens_np - extend_seq_lens_np).astype(np.int32)

    # ---- req_to_token rows (per-request, per-token slot table) ----
    # We claim the first `batch_size` slots; pie's runtime owns block IDs and
    # the pool's free-list bookkeeping is unused.
    req_pool_indices_np = np.arange(batch_size, dtype=np.int32)

    if batch_size > 0:
        max_seq_len = int(seq_lens_np.max())
        rows_np = np.zeros((batch_size, max_seq_len), dtype=np.int32)
        _build_req_to_token_rows(
            kv_idx_np, kv_indptr_np, seq_lens_np, page_size, rows_np
        )
    else:
        max_seq_len = 0
        rows_np = np.zeros((0, 0), dtype=np.int32)

    rows = torch.from_numpy(rows_np).to(device, non_blocking=True)

    # Write rows into sglang's pool at our chosen slots.
    if batch_size > 0:
        req_pool_indices_t = torch.from_numpy(req_pool_indices_np).to(device)
        runner.req_to_token_pool.req_to_token[req_pool_indices_t, :max_seq_len] = rows

    # ---- out_cache_loc (per-query-token destination slot) ----
    # For request r the query tokens occupy positions [s_r - q_r, s_r) within
    # the request. Their slot indices are exactly that slice of `rows[r]`.
    if num_query_tokens > 0:
        out_cache_loc = torch.empty(num_query_tokens, dtype=torch.int64, device=device)
        for r in range(batch_size):
            q_start, q_end = int(qo_indptr_np[r]), int(qo_indptr_np[r + 1])
            q_len = q_end - q_start
            first_pos = int(seq_lens_np[r]) - q_len
            out_cache_loc[q_start:q_end] = rows[r, first_pos:first_pos + q_len]
    else:
        out_cache_loc = torch.empty(0, dtype=torch.int64, device=device)

    # ---- ForwardBatch tensors ----
    seq_lens_cpu_t = torch.from_numpy(seq_lens_np).to(torch.int64)
    seq_lens_t = seq_lens_cpu_t.to(device, non_blocking=True)
    req_pool_indices_t = torch.from_numpy(req_pool_indices_np).to(torch.int64).to(device)
    extend_seq_lens_t = torch.from_numpy(extend_seq_lens_np).to(device, non_blocking=True)
    extend_prefix_lens_t = torch.from_numpy(extend_prefix_lens_np).to(device, non_blocking=True)
    positions = inputs["position_ids"].to(device=device, dtype=torch.int64, non_blocking=True)

    # We always use EXTEND. SGLang's decode kernel doesn't accept custom_mask
    # nor pie's caller-supplied positions; the extend kernel handles
    # query_len=1 fine via per-request `extend_seq_lens=1, prefix_lens=N`.
    #
    # NOTE: this means sglang's CUDA graphs (captured for the DECODE kernel)
    # can't be used at high concurrency — pie throughput on sglang plateaus
    # around 100 req/s on c=256 workloads on a 4090 because every step runs
    # the prefill kernel. Proper fix requires routing single-token-no-mask
    # batches through ForwardMode.DECODE plus the missing graph-runner
    # plumbing (`capture_hidden_mode`, etc.) — out of scope for now.
    fb = ForwardBatch(
        forward_mode=ForwardMode.EXTEND,
        batch_size=batch_size,
        input_ids=inputs["token_ids"].to(device=device, dtype=torch.int32, non_blocking=True),
        req_pool_indices=req_pool_indices_t,
        seq_lens=seq_lens_t,
        out_cache_loc=out_cache_loc,
        seq_lens_sum=int(seq_lens_np.sum()),
        seq_lens_cpu=seq_lens_cpu_t,
        positions=positions,
        extend_num_tokens=int(extend_seq_lens_np.sum()),
        extend_seq_lens=extend_seq_lens_t,
        extend_prefix_lens=extend_prefix_lens_t,
        extend_prefix_lens_cpu=extend_prefix_lens_np.tolist(),
        extend_seq_lens_cpu=extend_seq_lens_np.tolist(),
        req_to_token_pool=runner.req_to_token_pool,
        token_to_kv_pool=runner.token_to_kv_pool,
        attn_backend=runner.attn_backend,
    )
    return fb
