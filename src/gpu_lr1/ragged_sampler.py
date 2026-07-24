"""Ragged fused constrained sampling.

The engine never materializes a full-vocabulary mask. Each sequence carries a
grammar state whose allowed tokens are one CSR row; the kernel reads only that
row, applies temperature/top-k/top-p, samples one token by inverse CDF, and
writes the next grammar state.

Two kernels share one algorithm:

* ``_ragged_sample_single_tile_kernel`` keeps the whole row in registers and is
  used when every selected row fits one block. This is the fast path and covers
  structural grammar positions, which are narrow.
* ``_ragged_sample_tiled_kernel`` streams the row in tiles and therefore has no
  width limit, at the cost of re-reading the row once per phase.

Threshold selection avoids sorting. For probabilities ``p`` over the allowed
set, top-k keeps ``{p >= tau_k}`` where ``tau_k`` is the k-th largest value, and
top-p keeps ``{p >= tau_p}`` where ``tau_p`` is the largest threshold whose
retained mass still reaches ``top_p``. Both are found by bisection on the value
range, and the surviving set is their intersection ``{p >= max(tau_k, tau_p)}``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import triton
import triton.language as tl


DEFAULT_BISECT_STEPS = 32
MAX_SINGLE_TILE = 8192


@triton.jit
def _bisect_thresholds(
    p,
    valid,
    max_prob,
    top_k,
    top_p,
    NUM_BISECT: tl.constexpr,
):
    """Return max(tau_k, tau_p) for an in-register probability block."""
    low_k = 0.0
    high_k = max_prob
    low_p = 0.0
    high_p = max_prob
    for _ in tl.static_range(NUM_BISECT):
        mid_k = (low_k + high_k) * 0.5
        count = tl.sum(tl.where(valid & (p >= mid_k), 1, 0), axis=0)
        keep_k = count <= top_k
        high_k = tl.where(keep_k, mid_k, high_k)
        low_k = tl.where(keep_k, low_k, mid_k)

        mid_p = (low_p + high_p) * 0.5
        mass = tl.sum(tl.where(valid & (p >= mid_p), p, 0.0), axis=0)
        keep_p = mass >= top_p
        low_p = tl.where(keep_p, mid_p, low_p)
        high_p = tl.where(keep_p, high_p, mid_p)
    return tl.maximum(high_k, low_p)


@triton.jit
def _ragged_sample_single_tile_kernel(
    logits_ptr,
    csr_indptr_ptr,
    csr_indices_ptr,
    csr_next_state_ptr,
    rows_ptr,
    seq_index_ptr,
    temperature_ptr,
    top_k_ptr,
    top_p_ptr,
    uniform_ptr,
    out_tokens_ptr,
    out_states_ptr,
    vocab_size: tl.constexpr,
    BLOCK: tl.constexpr,
    NUM_BISECT: tl.constexpr,
    WRITE_STATE: tl.constexpr,
    BUCKET: tl.constexpr,
    BUCKET_LIMIT: tl.constexpr,
):
    pid = tl.program_id(0)
    seq = tl.load(seq_index_ptr + pid)
    row = tl.load(rows_ptr + seq)
    start = tl.load(csr_indptr_ptr + row)
    end = tl.load(csr_indptr_ptr + row + 1)
    length = end - start
    if BUCKET == 1 and length > BUCKET_LIMIT:
        return
    if BUCKET == 2 and length <= BUCKET_LIMIT:
        return

    offsets = tl.arange(0, BLOCK)
    valid = offsets < length
    tokens = tl.load(csr_indices_ptr + start + offsets, mask=valid, other=0)
    logits = tl.load(
        logits_ptr + seq.to(tl.int64) * vocab_size + tokens,
        mask=valid,
        other=0.0,
    ).to(tl.float32)

    temperature = tl.load(temperature_ptr + seq)
    scaled = logits / temperature
    neg_inf = float("-inf")
    scaled = tl.where(valid, scaled, neg_inf)
    maximum = tl.max(scaled, axis=0)
    weights = tl.where(valid, tl.exp(scaled - maximum), 0.0)
    total = tl.sum(weights, axis=0)
    probs = weights / total

    top_k = tl.load(top_k_ptr + seq)
    top_p = tl.load(top_p_ptr + seq)
    threshold = _bisect_thresholds(
        probs,
        valid,
        tl.max(probs, axis=0),
        top_k,
        top_p,
        NUM_BISECT=NUM_BISECT,
    )

    kept = tl.where(valid & (probs >= threshold), probs, 0.0)
    kept_mass = tl.sum(kept, axis=0)
    cumulative = tl.cumsum(kept, axis=0)
    uniform = tl.load(uniform_ptr + seq)
    target = uniform * kept_mass
    reached = (cumulative >= target) & (kept > 0.0)
    chosen = tl.min(tl.where(reached, offsets, BLOCK), axis=0)
    chosen = tl.where(chosen < BLOCK, chosen, 0)

    picked = tl.sum(tl.where(offsets == chosen, tokens, 0), axis=0)
    token = tl.where(length > 0, picked, -1)
    tl.store(out_tokens_ptr + seq, token)

    if WRITE_STATE:
        next_state = tl.load(
            csr_next_state_ptr + start + chosen,
            mask=length > 0,
            other=0,
        )
        tl.store(out_states_ptr + seq, next_state)


@triton.jit
def _ragged_sample_tiled_kernel(
    logits_ptr,
    csr_indptr_ptr,
    csr_indices_ptr,
    csr_next_state_ptr,
    rows_ptr,
    seq_index_ptr,
    temperature_ptr,
    top_k_ptr,
    top_p_ptr,
    uniform_ptr,
    out_tokens_ptr,
    out_states_ptr,
    vocab_size: tl.constexpr,
    BLOCK: tl.constexpr,
    NUM_BISECT: tl.constexpr,
    WRITE_STATE: tl.constexpr,
    BUCKET: tl.constexpr,
    BUCKET_LIMIT: tl.constexpr,
):
    pid = tl.program_id(0)
    seq = tl.load(seq_index_ptr + pid)
    row = tl.load(rows_ptr + seq)
    start = tl.load(csr_indptr_ptr + row)
    end = tl.load(csr_indptr_ptr + row + 1)
    length = end - start
    if BUCKET == 1 and length > BUCKET_LIMIT:
        return
    if BUCKET == 2 and length <= BUCKET_LIMIT:
        return
    temperature = tl.load(temperature_ptr + seq)
    offsets = tl.arange(0, BLOCK)
    neg_inf = float("-inf")

    maximum = neg_inf
    for base in range(0, length, BLOCK):
        valid = base + offsets < length
        tokens = tl.load(
            csr_indices_ptr + start + base + offsets, mask=valid, other=0
        )
        logits = tl.load(
            logits_ptr + seq.to(tl.int64) * vocab_size + tokens,
            mask=valid,
            other=0.0,
        ).to(tl.float32)
        scaled = tl.where(valid, logits / temperature, neg_inf)
        maximum = tl.maximum(maximum, tl.max(scaled, axis=0))

    total = 0.0
    for base in range(0, length, BLOCK):
        valid = base + offsets < length
        tokens = tl.load(
            csr_indices_ptr + start + base + offsets, mask=valid, other=0
        )
        logits = tl.load(
            logits_ptr + seq.to(tl.int64) * vocab_size + tokens,
            mask=valid,
            other=0.0,
        ).to(tl.float32)
        scaled = tl.where(valid, logits / temperature, neg_inf)
        total += tl.sum(tl.where(valid, tl.exp(scaled - maximum), 0.0), axis=0)

    top_k = tl.load(top_k_ptr + seq)
    top_p = tl.load(top_p_ptr + seq)
    low_k = 0.0
    high_k = 1.0
    low_p = 0.0
    high_p = 1.0
    for _ in tl.static_range(NUM_BISECT):
        mid_k = (low_k + high_k) * 0.5
        mid_p = (low_p + high_p) * 0.5
        count = 0
        mass = 0.0
        for base in range(0, length, BLOCK):
            valid = base + offsets < length
            tokens = tl.load(
                csr_indices_ptr + start + base + offsets, mask=valid, other=0
            )
            logits = tl.load(
                logits_ptr + seq.to(tl.int64) * vocab_size + tokens,
                mask=valid,
                other=0.0,
            ).to(tl.float32)
            scaled = tl.where(valid, logits / temperature, neg_inf)
            probs = tl.where(valid, tl.exp(scaled - maximum) / total, 0.0)
            count += tl.sum(tl.where(valid & (probs >= mid_k), 1, 0), axis=0)
            mass += tl.sum(tl.where(valid & (probs >= mid_p), probs, 0.0), axis=0)
        keep_k = count <= top_k
        high_k = tl.where(keep_k, mid_k, high_k)
        low_k = tl.where(keep_k, low_k, mid_k)
        keep_p = mass >= top_p
        low_p = tl.where(keep_p, mid_p, low_p)
        high_p = tl.where(keep_p, high_p, mid_p)
    threshold = tl.maximum(high_k, low_p)

    kept_mass = 0.0
    for base in range(0, length, BLOCK):
        valid = base + offsets < length
        tokens = tl.load(
            csr_indices_ptr + start + base + offsets, mask=valid, other=0
        )
        logits = tl.load(
            logits_ptr + seq.to(tl.int64) * vocab_size + tokens,
            mask=valid,
            other=0.0,
        ).to(tl.float32)
        scaled = tl.where(valid, logits / temperature, neg_inf)
        probs = tl.where(valid, tl.exp(scaled - maximum) / total, 0.0)
        kept_mass += tl.sum(
            tl.where(valid & (probs >= threshold), probs, 0.0), axis=0
        )

    uniform = tl.load(uniform_ptr + seq)
    target = uniform * kept_mass
    running = 0.0
    chosen = -1
    for base in range(0, length, BLOCK):
        valid = base + offsets < length
        tokens = tl.load(
            csr_indices_ptr + start + base + offsets, mask=valid, other=0
        )
        logits = tl.load(
            logits_ptr + seq.to(tl.int64) * vocab_size + tokens,
            mask=valid,
            other=0.0,
        ).to(tl.float32)
        scaled = tl.where(valid, logits / temperature, neg_inf)
        probs = tl.where(valid, tl.exp(scaled - maximum) / total, 0.0)
        kept = tl.where(valid & (probs >= threshold), probs, 0.0)
        cumulative = running + tl.cumsum(kept, axis=0)
        reached = (cumulative >= target) & (kept > 0.0)
        local = tl.min(tl.where(reached, base + offsets, length), axis=0)
        chosen = tl.where((chosen < 0) & (local < length), local, chosen)
        running += tl.sum(kept, axis=0)

    chosen = tl.where(chosen < 0, 0, chosen)
    token = tl.load(
        csr_indices_ptr + start + chosen, mask=length > 0, other=-1
    )
    tl.store(out_tokens_ptr + seq, token)
    if WRITE_STATE:
        next_state = tl.load(
            csr_next_state_ptr + start + chosen, mask=length > 0, other=0
        )
        tl.store(out_states_ptr + seq, next_state)


@dataclass
class RaggedSamplerTables:
    """CSR allowed-token rows shared by every sequence in a batch."""

    csr_indptr: torch.Tensor
    csr_indices: torch.Tensor
    csr_next_state: torch.Tensor | None = None

    def __post_init__(self) -> None:
        for name in ("csr_indptr", "csr_indices", "csr_next_state"):
            tensor = getattr(self, name)
            if tensor is None:
                continue
            if tensor.dtype != torch.int32:
                raise TypeError(f"{name} must be int32")
            if not tensor.is_cuda:
                raise ValueError(f"{name} must be a CUDA tensor")
        if (
            self.csr_next_state is not None
            and self.csr_next_state.shape != self.csr_indices.shape
        ):
            raise ValueError("token and next-state arrays must match")

        widths = (self.csr_indptr[1:] - self.csr_indptr[:-1]).cpu().numpy()
        self._widths = widths
        self.max_row_nnz = int(widths.max()) if widths.size else 0
        narrow = widths[widths <= MAX_SINGLE_TILE]
        self.narrow_max_nnz = int(narrow.max()) if narrow.size else 0
        self.has_wide_rows = bool((widths > MAX_SINGLE_TILE).any())

    @property
    def num_rows(self) -> int:
        return int(self.csr_indptr.numel() - 1)

    def row_widths(self) -> np.ndarray:
        return self._widths


def ragged_sample(
    logits: torch.Tensor,
    tables: RaggedSamplerTables,
    rows: torch.Tensor,
    *,
    temperature: torch.Tensor,
    top_k: torch.Tensor,
    top_p: torch.Tensor,
    uniform: torch.Tensor,
    max_row_nnz: int | None = None,
    out_tokens: torch.Tensor | None = None,
    out_states: torch.Tensor | None = None,
    bisect_steps: int = DEFAULT_BISECT_STEPS,
    force_tiled: bool = False,
    bucket: bool = True,
    seq_index: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Sample one token per sequence from its grammar row only.

    With ``bucket`` the batch is split by row width so that narrow sequences
    keep the single-tile fast path even when a wide sequence shares the batch.
    """
    if not logits.is_cuda:
        raise ValueError("ragged sampling requires CUDA logits")
    if logits.ndim != 2:
        raise ValueError("logits must have shape [batch, vocab]")
    batch = logits.shape[0]
    for name, tensor in (
        ("rows", rows),
        ("temperature", temperature),
        ("top_k", top_k),
        ("top_p", top_p),
        ("uniform", uniform),
    ):
        if tensor.shape != (batch,):
            raise ValueError(f"{name} must have one entry per sequence")
        if not tensor.is_cuda:
            raise ValueError(f"{name} must be a CUDA tensor")
    if rows.dtype != torch.int32 or top_k.dtype != torch.int32:
        raise TypeError("rows and top_k must be int32")
    if bisect_steps < 1:
        raise ValueError("bisect_steps must be positive")

    if max_row_nnz is None:
        max_row_nnz = tables.max_row_nnz
    if max_row_nnz < 1:
        raise ValueError("every selected row needs at least one allowed token")

    out_tokens = (
        out_tokens
        if out_tokens is not None
        else torch.empty(batch, dtype=torch.int32, device=logits.device)
    )
    write_state = tables.csr_next_state is not None
    if write_state and out_states is None:
        out_states = torch.empty_like(rows)
    if seq_index is None:
        seq_index = _all_sequences(batch, logits.device)

    common = dict(
        logits=logits,
        tables=tables,
        rows=rows,
        seq_index=seq_index,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        uniform=uniform,
        out_tokens=out_tokens,
        out_states=out_states,
        bisect_steps=bisect_steps,
        write_state=write_state,
    )
    result = (out_tokens, out_states if write_state else None)

    if force_tiled:
        _launch(tiled=True, block=1024, bucket=0, **common)
        return result
    if max_row_nnz <= MAX_SINGLE_TILE:
        _launch(
            tiled=False,
            block=triton.next_power_of_2(max_row_nnz),
            bucket=0,
            **common,
        )
        return result
    if not bucket:
        _launch(tiled=True, block=1024, bucket=0, **common)
        return result

    # Two sync-free launches over the same grid. Each program inspects its own
    # row width and returns immediately if it belongs to the other bucket, so a
    # single wide sequence cannot drag narrow ones onto the streaming path.
    if tables.narrow_max_nnz:
        _launch(
            tiled=False,
            block=triton.next_power_of_2(tables.narrow_max_nnz),
            bucket=1,
            **common,
        )
    _launch(tiled=True, block=1024, bucket=2, **common)
    return result


def _all_sequences(batch: int, device: torch.device) -> torch.Tensor:
    return torch.arange(batch, dtype=torch.int32, device=device)


def _launch(
    *,
    logits,
    tables,
    rows,
    seq_index,
    temperature,
    top_k,
    top_p,
    uniform,
    out_tokens,
    out_states,
    bisect_steps,
    write_state,
    tiled,
    block,
    bucket,
):
    next_state_ptr = (
        tables.csr_next_state if write_state else tables.csr_indices
    )
    state_out_ptr = out_states if write_state else out_tokens
    kernel = (
        _ragged_sample_tiled_kernel if tiled else _ragged_sample_single_tile_kernel
    )
    kernel[(seq_index.numel(),)](
        logits,
        tables.csr_indptr,
        tables.csr_indices,
        next_state_ptr,
        rows,
        seq_index,
        temperature,
        top_k,
        top_p,
        uniform,
        out_tokens,
        state_out_ptr,
        vocab_size=logits.shape[1],
        BLOCK=block,
        NUM_BISECT=bisect_steps,
        WRITE_STATE=write_state,
        BUCKET=bucket,
        BUCKET_LIMIT=MAX_SINGLE_TILE,
        num_warps=8 if tiled else _num_warps(block),
    )


def ragged_sample_reference(
    logits: np.ndarray,
    csr_indptr: np.ndarray,
    csr_indices: np.ndarray,
    rows: np.ndarray,
    *,
    temperature: np.ndarray,
    top_k: np.ndarray,
    top_p: np.ndarray,
    uniform: np.ndarray,
) -> np.ndarray:
    """Sorted-reference implementation with the same semantics as the kernel."""
    batch = logits.shape[0]
    result = np.full(batch, -1, dtype=np.int32)
    for index in range(batch):
        row = int(rows[index])
        start = int(csr_indptr[row])
        end = int(csr_indptr[row + 1])
        if start == end:
            continue
        tokens = csr_indices[start:end].astype(np.int64)
        values = logits[index, tokens].astype(np.float64)
        values = values / float(temperature[index])
        values -= values.max()
        weights = np.exp(values)
        probs = weights / weights.sum()

        order = np.argsort(-probs, kind="stable")
        ordered = probs[order]
        k = min(int(top_k[index]), ordered.size)
        threshold_k = ordered[k - 1] if k >= 1 else ordered[0]

        cumulative = np.cumsum(ordered)
        within = np.searchsorted(cumulative, float(top_p[index]), side="left")
        within = min(int(within), ordered.size - 1)
        threshold_p = ordered[within]

        threshold = max(threshold_k, threshold_p)
        kept = np.where(probs >= threshold, probs, 0.0)
        mass = kept.sum()
        target = float(uniform[index]) * mass
        running = np.cumsum(kept)
        candidates = np.flatnonzero((running >= target) & (kept > 0.0))
        chosen = int(candidates[0]) if candidates.size else 0
        result[index] = int(tokens[chosen])
    return result


def _num_warps(block: int) -> int:
    if block <= 256:
        return 2
    if block <= 1024:
        return 4
    return 8
