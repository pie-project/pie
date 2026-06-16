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
def _build_pool_rows_and_out_loc(
    qo_indptr: np.ndarray,          # int32 (batch+1,)
    kv_page_indices: np.ndarray,    # int32 (total_pages,)
    kv_page_indptr: np.ndarray,     # int32 (batch+1,)
    seq_lens: np.ndarray,           # int32 (batch,)
    page_size: int,
    rows: np.ndarray,               # int32 (batch, max_seq_len) — output
    out_cache_loc: np.ndarray,      # int64 (num_query_tokens,) — output
):
    """One pass: fill (a) per-request per-token slot table for sglang's
    `req_to_token_pool`, and (b) per-query-token destination slot for
    `out_cache_loc`. Both share the same per-token slot computation, so
    doing them together avoids a second pass — and a Python-level
    `for r in range(batch_size)` GPU index-copy fan-out at the call site.

    `rows` is filled `seq_lens[r]` wide per row; the rest stays zero
    (SGLang's kernels guard via `seq_lens`). `out_cache_loc[q_start:q_end]`
    receives the slot for each query token in request r.
    """
    batch = qo_indptr.shape[0] - 1
    for r in range(batch):
        s_r = seq_lens[r]
        page_base = kv_page_indptr[r]
        q_start = qo_indptr[r]
        q_end = qo_indptr[r + 1]
        prefix = s_r - (q_end - q_start)
        for i in range(s_r):
            block_id = kv_page_indices[page_base + i // page_size]
            slot = block_id * page_size + (i % page_size)
            rows[r, i] = slot
            if i >= prefix:
                out_cache_loc[q_start + (i - prefix)] = slot


def build_sglang_forward_batch(
    *,
    runner: Any,                     # sglang ModelRunner
    inputs: dict,
    page_size: int,
    device: torch.device,
) -> Any:
    """Build a `ForwardBatch` from pie's `inputs` dict."""
    from sglang.srt.model_executor.forward_batch_info import (
        CaptureHiddenMode,
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
    use_decode_mode = (
        bool(inputs.get("single_token_inference_mode"))
        and inputs.get("custom_mask") is None
        and bool(np.all(extend_seq_lens_np == 1))
    )

    # ---- req_to_token rows + out_cache_loc, built in one numba pass ----
    # We claim the first `batch_size` slots in sglang's req_to_token_pool;
    # pie's runtime owns block IDs and the pool's free-list bookkeeping is
    # unused. `out_cache_loc[q]` is the destination slot for query token q.
    req_pool_indices_np = np.arange(batch_size, dtype=np.int32)

    if batch_size > 0:
        max_seq_len = int(seq_lens_np.max())
        rows_np = np.zeros((batch_size, max_seq_len), dtype=np.int32)
        out_cache_loc_np = np.zeros(num_query_tokens, dtype=np.int64)
        _build_pool_rows_and_out_loc(
            qo_indptr_np, kv_idx_np, kv_indptr_np, seq_lens_np,
            page_size, rows_np, out_cache_loc_np,
        )

        rows = torch.from_numpy(rows_np).to(device, non_blocking=True)
        out_cache_loc = torch.from_numpy(out_cache_loc_np).to(device, non_blocking=True)
        req_pool_indices_t = torch.from_numpy(req_pool_indices_np).to(device)
        runner.req_to_token_pool.req_to_token[req_pool_indices_t, :max_seq_len] = rows
    else:
        out_cache_loc = torch.empty(0, dtype=torch.int64, device=device)

    # ---- ForwardBatch tensors ----
    seq_lens_cpu_t = torch.from_numpy(seq_lens_np).to(torch.int64)
    seq_lens_t = seq_lens_cpu_t.to(device, non_blocking=True)
    req_pool_indices_t = torch.from_numpy(req_pool_indices_np).to(torch.int64).to(device)
    extend_seq_lens_t = torch.from_numpy(extend_seq_lens_np).to(device, non_blocking=True)
    extend_prefix_lens_t = torch.from_numpy(extend_prefix_lens_np).to(device, non_blocking=True)
    positions = inputs["position_ids"].to(device=device, dtype=torch.int64, non_blocking=True)

    forward_mode = ForwardMode.DECODE if use_decode_mode else ForwardMode.EXTEND

    fb = ForwardBatch(
        forward_mode=forward_mode,
        batch_size=batch_size,
        input_ids=inputs["token_ids"].to(device=device, dtype=torch.int32, non_blocking=True),
        req_pool_indices=req_pool_indices_t,
        seq_lens=seq_lens_t,
        out_cache_loc=out_cache_loc,
        seq_lens_sum=int(seq_lens_np.sum()),
        seq_lens_cpu=seq_lens_cpu_t,
        positions=positions,
        extend_num_tokens=None if use_decode_mode else int(extend_seq_lens_np.sum()),
        extend_seq_lens=None if use_decode_mode else extend_seq_lens_t,
        extend_prefix_lens=None if use_decode_mode else extend_prefix_lens_t,
        extend_prefix_lens_cpu=None if use_decode_mode else extend_prefix_lens_np.tolist(),
        extend_seq_lens_cpu=None if use_decode_mode else extend_seq_lens_np.tolist(),
        req_to_token_pool=runner.req_to_token_pool,
        token_to_kv_pool=runner.token_to_kv_pool,
        attn_backend=runner.attn_backend,
        capture_hidden_mode=CaptureHiddenMode.NULL,
    )
    return fb
