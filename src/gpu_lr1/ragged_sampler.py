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

from gpu_lr1.wide_sampler import make_wide_workspace, wide_sample_split


DEFAULT_BISECT_STEPS = 32
MAX_SINGLE_TILE = 8192
WIDE_BLOCK = 4096
WIDE_PROBES = 8
WIDE_ROUNDS = 7
WIDE_NUM_WARPS = 8


def wide_splits_for(batch: int) -> int:
    """Chunks per wide row, chosen so the launch keeps the GPU occupied.

    A wide row sweeps the whole vocabulary several times, so at small batch a
    program-per-sequence layout leaves almost every SM idle. Measured optima on
    an A100 for a 151,669-token vocabulary.
    """
    if batch <= 16:
        return 64
    if batch <= 64:
        return 16
    if batch <= 256:
        return 8
    if batch <= 1024:
        return 4
    return 2


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
    # A tree reduction and a scan need not agree to the last bit, so a uniform
    # near 1 can overshoot the scan. The intended draw is then the final
    # surviving token, never the first one.
    last_kept = tl.max(tl.where(kept > 0.0, offsets, 0), axis=0)
    chosen = tl.where(chosen < BLOCK, chosen, last_kept)

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
    last_kept = 0
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
        last_kept = tl.max(
            tl.where(kept > 0.0, base + offsets, last_kept), axis=0
        )
        running += tl.sum(kept, axis=0)

    chosen = tl.where(chosen < 0, last_kept, chosen)
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

        self.bitset = None
        self.row_slot = None
        self._wide_workspaces = {}
        self._sequence_index = {}
        self.bitset_words = 0
        self.vocab_size = 0
        widths = (self.csr_indptr[1:] - self.csr_indptr[:-1]).cpu().numpy()
        self._widths = widths
        self.max_row_nnz = int(widths.max()) if widths.size else 0
        narrow = widths[widths <= MAX_SINGLE_TILE]
        self.narrow_max_nnz = int(narrow.max()) if narrow.size else 0
        self.has_wide_rows = bool((widths > MAX_SINGLE_TILE).any())

    def build_wide_bitsets(self, vocab_size: int) -> "RaggedSamplerTables":
        """Attach a complement bitset for every row wider than one tile."""
        if not self.has_wide_rows:
            return self
        indptr = self.csr_indptr.cpu().numpy()
        indices = self.csr_indices.cpu().numpy()
        words = (vocab_size + 31) // 32
        wide_rows = np.flatnonzero(self._widths > MAX_SINGLE_TILE)
        slots = np.full(self._widths.size, -1, dtype=np.int32)
        packed = np.zeros((wide_rows.size, words), dtype=np.uint32)
        for slot, row in enumerate(wide_rows):
            slots[row] = slot
            tokens = indices[indptr[row] : indptr[row + 1]].astype(np.int64)
            if tokens.size and np.any(np.diff(tokens) <= 0):
                raise ValueError("wide CSR rows must hold sorted token ids")
            np.bitwise_or.at(
                packed[slot],
                tokens >> 5,
                (np.uint32(1) << (tokens & 31).astype(np.uint32)),
            )
        device = self.csr_indices.device
        self.bitset = torch.from_numpy(packed.view(np.int32)).to(device)
        self.row_slot = torch.from_numpy(slots).to(device)
        self.bitset_words = words
        self.vocab_size = vocab_size
        return self

    def sequence_index(self, batch: int, device: torch.device) -> torch.Tensor:
        cached = self._sequence_index.get(batch)
        if cached is None:
            cached = torch.arange(batch, dtype=torch.int32, device=device)
            self._sequence_index[batch] = cached
        return cached

    def wide_workspace(self, batch: int, splits: int, probes: int):
        """Cache split-CTA scratch so the hot path never allocates."""
        key = (batch, splits, probes)
        workspace = self._wide_workspaces.get(key)
        if workspace is None:
            workspace = make_wide_workspace(
                batch,
                splits=splits,
                probes=probes,
                device=self.csr_indices.device,
            )
            self._wide_workspaces[key] = workspace
        return workspace

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
    wide_probes: int = WIDE_PROBES,
    wide_rounds: int = WIDE_ROUNDS,
    wide_splits: int | None = None,
    wide_present: bool = True,
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
        seq_index = tables.sequence_index(batch, logits.device)

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
    if tables.bitset is not None and wide_present:
        splits = wide_splits if wide_splits is not None else wide_splits_for(batch)
        if splits > 1:
            workspace = tables.wide_workspace(batch, splits, wide_probes)
            wide_sample_split(
                logits,
                tables,
                rows,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                uniform=uniform,
                out_tokens=out_tokens,
                out_states=out_states if write_state else None,
                workspace=workspace,
                rounds=wide_rounds,
                block=WIDE_BLOCK,
                num_warps=WIDE_NUM_WARPS,
            )
        else:
            _launch_wide(
                probes=wide_probes,
                rounds=wide_rounds,
                **common,
            )
    else:
        _launch(tiled=True, block=1024, bucket=2, **common)
    return result


def _launch_wide(
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
    probes,
    rounds,
):
    del bisect_steps
    _wide_sample_kernel[(seq_index.numel(),)](
        logits,
        tables.bitset,
        tables.row_slot,
        tables.csr_indptr,
        tables.csr_indices,
        tables.csr_next_state if write_state else tables.csr_indices,
        rows,
        seq_index,
        temperature,
        top_k,
        top_p,
        uniform,
        out_tokens,
        out_states if write_state else out_tokens,
        vocab_size=logits.shape[1],
        bitset_words=tables.bitset_words,
        BLOCK=WIDE_BLOCK,
        PROBES=probes,
        ROUNDS=rounds,
        WRITE_STATE=write_state,
        num_warps=WIDE_NUM_WARPS,
    )


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


# ---------------------------------------------------------------------------
# Wide rows: complement representation
#
# A JSON string state allows 147,144 of 151,669 tokens. Listing the allowed
# tokens is the wrong way round: the row is better described by the 4,525
# tokens it forbids. The kernel below therefore reads logits contiguously and
# tests membership against a per-state bitset (18.5 KiB, about 6% of the logit
# traffic it already pays), instead of gathering through an index list.
#
# The threshold search also changes shape. Bisecting one candidate per pass
# costs one full sweep per bit; here each sweep evaluates PROBES candidates at
# once, so a sweep yields log2(PROBES + 1) bits and the whole search fits in a
# handful of contiguous passes.
# ---------------------------------------------------------------------------


@triton.jit
def _wide_sample_kernel(
    logits_ptr,
    bitset_ptr,
    slot_ptr,
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
    bitset_words: tl.constexpr,
    BLOCK: tl.constexpr,
    PROBES: tl.constexpr,
    ROUNDS: tl.constexpr,
    WRITE_STATE: tl.constexpr,
):
    pid = tl.program_id(0)
    seq = tl.load(seq_index_ptr + pid)
    row = tl.load(rows_ptr + seq)
    slot = tl.load(slot_ptr + row)
    if slot < 0:
        return

    base = slot.to(tl.int64) * bitset_words
    logit_base = seq.to(tl.int64) * vocab_size
    temperature = tl.load(temperature_ptr + seq)
    lane = tl.arange(0, BLOCK)
    probe = tl.arange(0, PROBES)

    maximum = -1e30
    total = 0.0
    for start in range(0, vocab_size, BLOCK):
        offsets = start + lane
        inside = offsets < vocab_size
        word = tl.load(bitset_ptr + base + (offsets >> 5), mask=inside, other=0)
        allowed = inside & (((word >> (offsets & 31)) & 1) == 1)
        values = tl.load(
            logits_ptr + logit_base + offsets, mask=allowed, other=0.0
        ).to(tl.float32) / temperature
        values = tl.where(allowed, values, -1e30)
        updated = tl.maximum(maximum, tl.max(values, axis=0))
        total = total * tl.exp(maximum - updated) + tl.sum(
            tl.where(allowed, tl.exp(values - updated), 0.0), axis=0
        )
        maximum = updated

    top_k = tl.load(top_k_ptr + seq)
    top_p = tl.load(top_p_ptr + seq)
    peak = 1.0 / total
    low_k = 0.0
    high_k = peak
    low_p = 0.0
    high_p = peak

    for _ in range(ROUNDS):
        step_k = (high_k - low_k) / (PROBES + 1)
        step_p = (high_p - low_p) / (PROBES + 1)
        probes_k = low_k + (probe + 1).to(tl.float32) * step_k
        probes_p = low_p + (probe + 1).to(tl.float32) * step_p
        counts = tl.zeros([PROBES], dtype=tl.float32)
        masses = tl.zeros([PROBES], dtype=tl.float32)
        for start in range(0, vocab_size, BLOCK):
            offsets = start + lane
            inside = offsets < vocab_size
            word = tl.load(
                bitset_ptr + base + (offsets >> 5), mask=inside, other=0
            )
            allowed = inside & (((word >> (offsets & 31)) & 1) == 1)
            values = tl.load(
                logits_ptr + logit_base + offsets, mask=allowed, other=0.0
            ).to(tl.float32) / temperature
            probs = tl.where(
                allowed, tl.exp(values - maximum) / total, 0.0
            )
            wide = probs[:, None]
            counts += tl.sum(
                tl.where(wide >= probes_k[None, :], 1.0, 0.0), axis=0
            )
            masses += tl.sum(
                tl.where(wide >= probes_p[None, :], wide, 0.0), axis=0
            )
        fits_k = counts <= top_k.to(tl.float32)
        next_high_k = tl.min(tl.where(fits_k, probes_k, high_k), axis=0)
        low_k = tl.max(tl.where(fits_k, low_k, probes_k), axis=0)
        high_k = next_high_k

        fits_p = masses >= top_p
        next_low_p = tl.max(tl.where(fits_p, probes_p, low_p), axis=0)
        high_p = tl.min(tl.where(fits_p, high_p, probes_p), axis=0)
        low_p = next_low_p

    threshold = tl.maximum(high_k, low_p)

    # The retained mass must be accumulated in exactly the tile order the
    # sampling sweep uses, otherwise a uniform near 1 overshoots a mass that
    # was summed in a different order.
    kept_mass = 0.0
    for start in range(0, vocab_size, BLOCK):
        offsets = start + lane
        inside = offsets < vocab_size
        word = tl.load(bitset_ptr + base + (offsets >> 5), mask=inside, other=0)
        allowed = inside & (((word >> (offsets & 31)) & 1) == 1)
        values = tl.load(
            logits_ptr + logit_base + offsets, mask=allowed, other=0.0
        ).to(tl.float32) / temperature
        probs = tl.where(allowed, tl.exp(values - maximum) / total, 0.0)
        kept_mass += tl.sum(tl.where(probs >= threshold, probs, 0.0), axis=0)

    target = tl.load(uniform_ptr + seq) * kept_mass
    running = 0.0
    chosen = -1
    last_kept = 0
    for start in range(0, vocab_size, BLOCK):
        offsets = start + lane
        inside = offsets < vocab_size
        word = tl.load(bitset_ptr + base + (offsets >> 5), mask=inside, other=0)
        allowed = inside & (((word >> (offsets & 31)) & 1) == 1)
        values = tl.load(
            logits_ptr + logit_base + offsets, mask=allowed, other=0.0
        ).to(tl.float32) / temperature
        probs = tl.where(allowed, tl.exp(values - maximum) / total, 0.0)
        kept = tl.where(probs >= threshold, probs, 0.0)
        cumulative = running + tl.cumsum(kept, axis=0)
        reached = (cumulative >= target) & (kept > 0.0)
        local = tl.min(tl.where(reached, offsets, vocab_size), axis=0)
        chosen = tl.where((chosen < 0) & (local < vocab_size), local, chosen)
        last_kept = tl.max(tl.where(kept > 0.0, offsets, last_kept), axis=0)
        running += tl.sum(kept, axis=0)

    chosen = tl.where(chosen < 0, last_kept, chosen)
    tl.store(out_tokens_ptr + seq, chosen)

    if WRITE_STATE:
        # The row is sorted, so recover the CSR position of the sampled token
        # with a binary search rather than carrying an inverse table.
        left = tl.load(csr_indptr_ptr + row)
        right = tl.load(csr_indptr_ptr + row + 1)
        for _ in range(32):
            middle = (left + right) // 2
            probe_token = tl.load(
                csr_indices_ptr + middle, mask=middle < right, other=2147483647
            )
            go_right = (probe_token < chosen) & (left < right)
            left = tl.where(go_right, middle + 1, left)
            right = tl.where(go_right, right, middle)
        next_state = tl.load(csr_next_state_ptr + left, mask=left >= 0, other=0)
        tl.store(out_states_ptr + seq, next_state)


@dataclass(frozen=True)
class RaggedSampleGraph:
    """A captured constrained-sampling step.

    The split-CTA wide path issues about twenty launches, which dominates the
    step whenever the device work is small. Capturing the whole step collapses
    that into one replay and is also what a serving engine needs in order to
    fold the constraint into the model graph.
    """

    graph: torch.cuda.CUDAGraph
    tokens: torch.Tensor
    states: torch.Tensor | None

    def replay(self) -> tuple[torch.Tensor, torch.Tensor | None]:
        self.graph.replay()
        return self.tokens, self.states


def capture_ragged_sample(
    logits: torch.Tensor,
    tables: RaggedSamplerTables,
    rows: torch.Tensor,
    *,
    temperature: torch.Tensor,
    top_k: torch.Tensor,
    top_p: torch.Tensor,
    uniform: torch.Tensor,
    out_tokens: torch.Tensor | None = None,
    out_states: torch.Tensor | None = None,
    **kwargs,
) -> RaggedSampleGraph:
    """Warm up, then capture one constrained-sampling step as a CUDA graph."""
    batch = logits.shape[0]
    device = logits.device
    if out_tokens is None:
        out_tokens = torch.empty(batch, dtype=torch.int32, device=device)
    if out_states is None and tables.csr_next_state is not None:
        out_states = torch.empty(batch, dtype=torch.int32, device=device)

    def step() -> None:
        ragged_sample(
            logits,
            tables,
            rows,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            uniform=uniform,
            out_tokens=out_tokens,
            out_states=out_states,
            **kwargs,
        )

    for _ in range(3):
        step()
    torch.cuda.synchronize(device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step()
    torch.cuda.synchronize(device)
    return RaggedSampleGraph(graph=graph, tokens=out_tokens, states=out_states)
