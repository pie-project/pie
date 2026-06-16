"""Translate pie's batch metadata into vllm's `CommonAttentionMetadata`.

Pie emits metadata in CSR form (qo_indptr, kv_page_indices, kv_page_indptr,
kv_last_page_lens). vllm consumes a backend-agnostic struct
(`query_start_loc`, `seq_lens`, `block_table_tensor`, `slot_mapping`). Block
IDs are shifted by `block_id_offset` before they reach vLLM. vLLM reserves
physical block id 0 as the NULL/padding block, while Pie's scheduler uses
zero-based page ids for real cache pages.

Conversions:
  - `qo_indptr`              → `query_start_loc`     (rename only)
  - per-request page count   = kv_page_indptr.diff()
  - `seq_lens`               = (page_count - 1) * page_size + last_page_len
  - CSR (`kv_page_indices`, `kv_page_indptr`)
                             → dense `block_table` 2D (right-padded per row)
  - `qo_indptr` + page state → `slot_mapping` (flat write index per query token)
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass

import numba
import numpy as np
import torch

_COMMON_ATTENTION_METADATA_PARAMS = None


@numba.njit(cache=True, parallel=False)
def _build_block_table(
    kv_page_indices: np.ndarray,    # int32 (total_pages,)
    kv_page_indptr: np.ndarray,     # int32 (batch+1,)
    max_blocks_per_req: int,
    block_id_offset: int,
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
            out[r, j] = kv_page_indices[start + j] + block_id_offset


@numba.njit(cache=True, parallel=False)
def _build_split_block_table(
    kv_page_indices: np.ndarray,    # int32 (total_pages,)
    kv_page_indptr: np.ndarray,     # int32 (batch+1,)
    max_blocks_per_req: int,
    blocks_per_page: int,
    block_id_offset: int,
    out: np.ndarray,                # int32 (batch, max_blocks_per_req * blocks_per_page)
):
    """Expand scheduler page IDs into attention-kernel block IDs."""
    batch = kv_page_indptr.shape[0] - 1
    for r in range(batch):
        start = kv_page_indptr[r]
        end = kv_page_indptr[r + 1]
        dst = 0
        for j in range(end - start):
            base = (kv_page_indices[start + j] + block_id_offset) * blocks_per_page
            for k in range(blocks_per_page):
                out[r, dst] = base + k
                dst += 1


@numba.njit(cache=True, parallel=False)
def _build_slot_mapping(
    qo_indptr: np.ndarray,           # int32 (batch+1,)
    kv_page_indices: np.ndarray,     # int32 (total_pages,)
    kv_page_indptr: np.ndarray,      # int32 (batch+1,)
    seq_lens: np.ndarray,            # int32 (batch,)
    page_size: int,
    block_id_offset: int,
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
            block_id = kv_page_indices[page_base + page_idx] + block_id_offset
            out[q_start + i] = block_id * page_size + offset


@numba.njit(cache=True, parallel=False)
def _build_split_slot_mapping(
    qo_indptr: np.ndarray,           # int32 (batch+1,)
    kv_page_indices: np.ndarray,     # int32 (total_pages,)
    kv_page_indptr: np.ndarray,      # int32 (batch+1,)
    seq_lens: np.ndarray,            # int32 (batch,)
    manager_page_size: int,
    kernel_page_size: int,
    blocks_per_page: int,
    block_id_offset: int,
    out: np.ndarray,                 # int64 (num_tokens,)
):
    """Flat slot index when one scheduler page maps to many kernel blocks."""
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
            manager_page_idx = p // manager_page_size
            offset_in_page = p - manager_page_idx * manager_page_size
            kernel_page_idx = offset_in_page // kernel_page_size
            offset = offset_in_page - kernel_page_idx * kernel_page_size
            manager_block_id = (
                kv_page_indices[page_base + manager_page_idx] + block_id_offset
            )
            kernel_block_id = manager_block_id * blocks_per_page + kernel_page_idx
            out[q_start + i] = kernel_block_id * kernel_page_size + offset


@dataclass
class CommonMetadataWorkspace:
    """Stable GPU buffers for full CUDA-graph attention metadata.

    vLLM's FULL graphs capture attention metadata tensor addresses. Runtime
    values therefore have to be copied into persistent buffers rather than
    rebuilt as fresh tensors on every step.
    """

    max_reqs: int
    max_tokens: int
    max_blocks_per_req: int
    device: torch.device

    def __post_init__(self) -> None:
        self.query_start_loc = torch.empty(
            self.max_reqs + 1, dtype=torch.int32, device=self.device
        )
        self.seq_lens = torch.empty(
            self.max_reqs, dtype=torch.int32, device=self.device
        )
        self.block_table = torch.empty(
            (self.max_reqs, self.max_blocks_per_req),
            dtype=torch.int32,
            device=self.device,
        )
        self.slot_mapping = torch.empty(
            self.max_tokens, dtype=torch.int64, device=self.device
        )

    def ensure(self, *, max_reqs: int, max_tokens: int, max_blocks_per_req: int):
        if (
            max_reqs <= self.max_reqs
            and max_tokens <= self.max_tokens
            and max_blocks_per_req <= self.max_blocks_per_req
        ):
            return self
        return CommonMetadataWorkspace(
            max_reqs=max(max_reqs, self.max_reqs),
            max_tokens=max(max_tokens, self.max_tokens),
            max_blocks_per_req=max(max_blocks_per_req, self.max_blocks_per_req),
            device=self.device,
        )


def build_common_metadata(
    *,
    qo_indptr: torch.Tensor,
    kv_page_indices: torch.Tensor,
    kv_page_indptr: torch.Tensor,
    kv_last_page_lens: torch.Tensor,
    qo_indptr_cpu: np.ndarray | None = None,
    kv_page_indices_cpu: np.ndarray | None = None,
    kv_page_indptr_cpu: np.ndarray | None = None,
    kv_last_page_lens_cpu: np.ndarray | None = None,
    page_size: int,
    kernel_page_size: int | None = None,
    device: torch.device,
    position_ids: torch.Tensor | None = None,
    pad_num_reqs: int | None = None,
    pad_num_tokens: int | None = None,
    workspace: CommonMetadataWorkspace | None = None,
    block_id_offset: int = 0,
    max_seq_len_override: int | None = None,
    force_decode: bool = False,
):
    """Produce a `CommonAttentionMetadata` from pie's batch dict fields.

    All inputs may live on CPU or GPU; outputs land on `device`. We do the
    Numba work on numpy arrays (CPU) and copy the result to GPU at the end.
    """
    from ._vllm_compat import CommonAttentionMetadata

    qo_np = _i32_np(qo_indptr, qo_indptr_cpu)
    kv_idx_np = _i32_np(kv_page_indices, kv_page_indices_cpu)
    kv_indptr_np = _i32_np(kv_page_indptr, kv_page_indptr_cpu)
    kv_last_np = _i32_np(kv_last_page_lens, kv_last_page_lens_cpu)

    batch = qo_np.shape[0] - 1
    num_tokens = int(qo_np[-1])
    out_batch = int(pad_num_reqs or batch)
    out_tokens = int(pad_num_tokens or num_tokens)
    if out_batch < batch:
        raise ValueError(f"pad_num_reqs ({out_batch}) cannot be < batch ({batch})")
    if out_tokens < num_tokens:
        raise ValueError(
            f"pad_num_tokens ({out_tokens}) cannot be < num_tokens ({num_tokens})"
        )

    kernel_page_size = int(kernel_page_size or page_size)
    if page_size % kernel_page_size != 0:
        raise ValueError(
            f"kernel_page_size ({kernel_page_size}) must divide page_size ({page_size})"
        )
    blocks_per_page = page_size // kernel_page_size

    # seq_lens = (num_pages - 1) * page_size + last_page_len
    pages_per_req = (kv_indptr_np[1:] - kv_indptr_np[:-1]).astype(np.int32)
    # Defensive: pages_per_req must be > 0 for any request that has KV state.
    # A request with 0 pages and a non-empty query is impossible in pie.
    seq_lens_np = ((pages_per_req - 1) * page_size + kv_last_np).astype(np.int32)

    # block_table: (batch, max_blocks_per_req) right-padded with 0. When the
    # attention kernel uses smaller virtual blocks, expand each scheduler page
    # ID into the contiguous kernel block IDs vLLM's block table would produce.
    max_blocks = int(pages_per_req.max()) if batch > 0 else 0
    block_table_cols = max_blocks * blocks_per_page
    block_table_np = np.zeros((out_batch, block_table_cols), dtype=np.int32)
    if batch > 0 and max_blocks > 0:
        if blocks_per_page == 1:
            _build_block_table(
                kv_idx_np,
                kv_indptr_np,
                max_blocks,
                block_id_offset,
                block_table_np[:batch],
            )
        else:
            _build_split_block_table(
                kv_idx_np,
                kv_indptr_np,
                max_blocks,
                blocks_per_page,
                block_id_offset,
                block_table_np[:batch],
            )

    # slot_mapping: (num_tokens,) absolute slot per query token.
    slot_mapping_np = np.full(out_tokens, -1, dtype=np.int64)
    if num_tokens > 0:
        if blocks_per_page == 1:
            _build_slot_mapping(
                qo_np,
                kv_idx_np,
                kv_indptr_np,
                seq_lens_np,
                page_size,
                block_id_offset,
                slot_mapping_np[:num_tokens],
            )
        else:
            _build_split_slot_mapping(
                qo_np,
                kv_idx_np,
                kv_indptr_np,
                seq_lens_np,
                page_size,
                kernel_page_size,
                blocks_per_page,
                block_id_offset,
                slot_mapping_np[:num_tokens],
            )

    query_lens_np = qo_np[1:] - qo_np[:-1]
    num_computed_tokens_np = seq_lens_np - query_lens_np
    # Mamba metadata needs the same prefill/decode hint vLLM's InputBatch
    # supplies. Pie does not expose the prompt boundary here, but for serving
    # batches the query span itself identifies the initial/extend prefill
    # work: multi-token queries are prefills, and a single-token query with no
    # prior computed context is also a prefill.
    is_prefilling_np = (
        (query_lens_np > 1) | (num_computed_tokens_np == 0)
    ).astype(np.bool_)
    if force_decode:
        is_prefilling_np[:] = False

    query_start_loc_np = np.empty(out_batch + 1, dtype=np.int32)
    query_start_loc_np[: batch + 1] = qo_np
    if out_batch > batch:
        query_start_loc_np[batch + 1 :] = num_tokens

    seq_lens_out_np = np.zeros(out_batch, dtype=np.int32)
    seq_lens_out_np[:batch] = seq_lens_np
    num_computed_tokens_out_np = np.zeros(out_batch, dtype=np.int32)
    num_computed_tokens_out_np[:batch] = num_computed_tokens_np
    is_prefilling_out_np = np.zeros(out_batch, dtype=np.bool_)
    is_prefilling_out_np[:batch] = is_prefilling_np

    query_start_loc_cpu = torch.from_numpy(query_start_loc_np)
    seq_lens_cpu = torch.from_numpy(seq_lens_out_np)
    num_computed_tokens_cpu = torch.from_numpy(num_computed_tokens_out_np)
    is_prefilling_cpu = torch.from_numpy(is_prefilling_out_np)

    if workspace is not None:
        workspace = workspace.ensure(
            max_reqs=out_batch,
            max_tokens=out_tokens,
            max_blocks_per_req=max(block_table_np.shape[1], 1),
        )
        query_start_loc_src = torch.from_numpy(query_start_loc_np)
        seq_lens_src = torch.from_numpy(seq_lens_out_np)
        block_table_src = torch.from_numpy(block_table_np)
        slot_mapping_src = torch.from_numpy(slot_mapping_np)
        workspace.query_start_loc[: out_batch + 1].copy_(
            query_start_loc_src, non_blocking=True
        )
        workspace.seq_lens[:out_batch].copy_(seq_lens_src, non_blocking=True)
        workspace.block_table[
            :out_batch, : block_table_np.shape[1]
        ].copy_(block_table_src, non_blocking=True)
        workspace.slot_mapping[:out_tokens].copy_(
            slot_mapping_src, non_blocking=True
        )
        query_start_loc = workspace.query_start_loc[: out_batch + 1]
        seq_lens = workspace.seq_lens[:out_batch]
        block_table_tensor = workspace.block_table[
            :out_batch, : block_table_np.shape[1]
        ]
        slot_mapping = workspace.slot_mapping[:out_tokens]
    else:
        query_start_loc = query_start_loc_cpu.to(device, non_blocking=True)
        seq_lens = seq_lens_cpu.to(device, non_blocking=True)
        block_table_tensor = torch.from_numpy(block_table_np).to(
            device, non_blocking=True
        )
        slot_mapping = torch.from_numpy(slot_mapping_np).to(
            device, non_blocking=True
        )

    positions = None
    if position_ids is not None:
        if position_ids.shape[0] >= out_tokens:
            positions = position_ids[:out_tokens].to(
                device, dtype=torch.int64, non_blocking=True
            )
        else:
            positions = position_ids.to(device, dtype=torch.int64, non_blocking=True)

    max_query_len = int(query_lens_np.max()) if batch > 0 else 0
    max_seq_len = (
        int(max_seq_len_override)
        if max_seq_len_override is not None
        else (int(seq_lens_np.max()) if batch > 0 else 0)
    )

    kwargs = {
        "query_start_loc": query_start_loc,
        "query_start_loc_cpu": query_start_loc_cpu,
        "seq_lens": seq_lens,
        "num_reqs": out_batch,
        "num_actual_tokens": out_tokens,
        "max_query_len": max_query_len,
        "max_seq_len": max_seq_len,
        "block_table_tensor": block_table_tensor,
        "slot_mapping": slot_mapping,
        "causal": True,
        "_seq_lens_cpu": seq_lens_cpu,
        "_num_computed_tokens_cpu": num_computed_tokens_cpu,
    }
    optional_kwargs = {
        # vLLM revisions differ on whether these fields are constructor args
        # or populated elsewhere in the attention path.
        "positions": positions,
        "seq_lens_cpu_upper_bound": seq_lens_cpu,
        "is_prefilling": is_prefilling_cpu,
    }
    params = _common_attention_metadata_params(CommonAttentionMetadata)
    for name, value in optional_kwargs.items():
        if name in params:
            kwargs[name] = value

    return CommonAttentionMetadata(**kwargs)


def _common_attention_metadata_params(cls) -> object:
    global _COMMON_ATTENTION_METADATA_PARAMS
    if _COMMON_ATTENTION_METADATA_PARAMS is None:
        _COMMON_ATTENTION_METADATA_PARAMS = inspect.signature(cls).parameters
    return _COMMON_ATTENTION_METADATA_PARAMS


def _i32_np(tensor: torch.Tensor, cpu: np.ndarray | None) -> np.ndarray:
    if cpu is not None:
        return np.asarray(cpu, dtype=np.int32)
    return tensor.to(torch.int32).cpu().numpy()
