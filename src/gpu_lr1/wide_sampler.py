"""Split-CTA sampling for wide grammar rows.

`ragged_sampler` gives one program to each sequence. That is right for narrow
rows, but a wide row makes a single program sweep the whole vocabulary several
times, so a batch of 128 sequences occupies 128 of the GPU's 108 SMs for a long
serial loop and nothing hides the latency. Measured, fp16 logits did not help
that kernel at all, which rules out bandwidth and points at occupancy.

Here each sequence's row is cut into `splits` chunks and every chunk gets its
own program, so the launch has `batch * splits` programs regardless of batch
size. Chunk results meet in small scratch buffers, and a per-sequence kernel
between phases folds them together.

The sampling algorithm is unchanged and still matches the sorted reference:
threshold by bisection with several candidates evaluated per sweep, then an
inverse-CDF draw over the survivors.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
import triton.language as tl


NEG_LARGE = tl.constexpr(-1e30)
UNSET = tl.constexpr(2147483647)


@triton.jit
def _chunk_bounds(pid_split, vocab_size, SPLITS: tl.constexpr):
    chunk = (vocab_size + SPLITS - 1) // SPLITS
    start = pid_split * chunk
    stop = tl.minimum(start + chunk, vocab_size)
    return start, stop


@triton.jit
def _wide_stats_kernel(
    logits_ptr,
    bitset_ptr,
    slot_ptr,
    rows_ptr,
    temperature_ptr,
    part_max_ptr,
    part_sum_ptr,
    vocab_size,
    bitset_words,
    SPLITS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    seq = tl.program_id(0)
    split = tl.program_id(1)
    row = tl.load(rows_ptr + seq)
    slot = tl.load(slot_ptr + row)
    if slot < 0:
        return
    base = slot.to(tl.int64) * bitset_words
    logit_base = seq.to(tl.int64) * vocab_size
    temperature = tl.load(temperature_ptr + seq)
    lane = tl.arange(0, BLOCK)
    start, stop = _chunk_bounds(split, vocab_size, SPLITS)

    maximum = NEG_LARGE
    total = 0.0
    for offset in range(start, stop, BLOCK):
        offsets = offset + lane
        inside = offsets < stop
        word = tl.load(bitset_ptr + base + (offsets >> 5), mask=inside, other=0)
        allowed = inside & (((word >> (offsets & 31)) & 1) == 1)
        values = tl.load(
            logits_ptr + logit_base + offsets, mask=allowed, other=0.0
        ).to(tl.float32) / temperature
        values = tl.where(allowed, values, NEG_LARGE)
        updated = tl.maximum(maximum, tl.max(values, axis=0))
        total = total * tl.exp(maximum - updated) + tl.sum(
            tl.where(allowed, tl.exp(values - updated), 0.0), axis=0
        )
        maximum = updated

    tl.store(part_max_ptr + seq * SPLITS + split, maximum)
    tl.store(part_sum_ptr + seq * SPLITS + split, total)


@triton.jit
def _wide_reduce_stats_kernel(
    part_max_ptr,
    part_sum_ptr,
    stats_ptr,
    bounds_ptr,
    probes_ptr,
    rows_ptr,
    slot_ptr,
    SPLITS: tl.constexpr,
    PROBES: tl.constexpr,
):
    seq = tl.program_id(0)
    if tl.load(slot_ptr + tl.load(rows_ptr + seq)) < 0:
        return
    lane = tl.arange(0, SPLITS)
    maxima = tl.load(part_max_ptr + seq * SPLITS + lane)
    sums = tl.load(part_sum_ptr + seq * SPLITS + lane)
    maximum = tl.max(maxima, axis=0)
    total = tl.sum(sums * tl.exp(maxima - maximum), axis=0)
    tl.store(stats_ptr + seq * 2, maximum)
    tl.store(stats_ptr + seq * 2 + 1, total)

    peak = 1.0 / total
    tl.store(bounds_ptr + seq * 4 + 0, 0.0)
    tl.store(bounds_ptr + seq * 4 + 1, peak)
    tl.store(bounds_ptr + seq * 4 + 2, 0.0)
    tl.store(bounds_ptr + seq * 4 + 3, peak)
    probe = tl.arange(0, PROBES)
    step = peak / (PROBES + 1)
    candidates = (probe + 1).to(tl.float32) * step
    tl.store(probes_ptr + seq * 2 * PROBES + probe, candidates)
    tl.store(probes_ptr + seq * 2 * PROBES + PROBES + probe, candidates)


@triton.jit
def _wide_probe_kernel(
    logits_ptr,
    bitset_ptr,
    slot_ptr,
    rows_ptr,
    temperature_ptr,
    stats_ptr,
    probes_ptr,
    part_count_ptr,
    part_mass_ptr,
    vocab_size,
    bitset_words,
    SPLITS: tl.constexpr,
    PROBES: tl.constexpr,
    BLOCK: tl.constexpr,
):
    seq = tl.program_id(0)
    split = tl.program_id(1)
    row = tl.load(rows_ptr + seq)
    slot = tl.load(slot_ptr + row)
    if slot < 0:
        return
    base = slot.to(tl.int64) * bitset_words
    logit_base = seq.to(tl.int64) * vocab_size
    temperature = tl.load(temperature_ptr + seq)
    maximum = tl.load(stats_ptr + seq * 2)
    total = tl.load(stats_ptr + seq * 2 + 1)
    probe = tl.arange(0, PROBES)
    probes_k = tl.load(probes_ptr + seq * 2 * PROBES + probe)
    probes_p = tl.load(probes_ptr + seq * 2 * PROBES + PROBES + probe)
    lane = tl.arange(0, BLOCK)
    start, stop = _chunk_bounds(split, vocab_size, SPLITS)

    counts = tl.zeros([PROBES], dtype=tl.float32)
    masses = tl.zeros([PROBES], dtype=tl.float32)
    for offset in range(start, stop, BLOCK):
        offsets = offset + lane
        inside = offsets < stop
        word = tl.load(bitset_ptr + base + (offsets >> 5), mask=inside, other=0)
        allowed = inside & (((word >> (offsets & 31)) & 1) == 1)
        values = tl.load(
            logits_ptr + logit_base + offsets, mask=allowed, other=0.0
        ).to(tl.float32) / temperature
        probs = tl.where(allowed, tl.exp(values - maximum) / total, 0.0)
        wide = probs[:, None]
        counts += tl.sum(tl.where(wide >= probes_k[None, :], 1.0, 0.0), axis=0)
        masses += tl.sum(tl.where(wide >= probes_p[None, :], wide, 0.0), axis=0)

    out = (seq * SPLITS + split) * PROBES + probe
    tl.store(part_count_ptr + out, counts)
    tl.store(part_mass_ptr + out, masses)


@triton.jit
def _wide_update_kernel(
    part_count_ptr,
    part_mass_ptr,
    bounds_ptr,
    probes_ptr,
    top_k_ptr,
    top_p_ptr,
    threshold_ptr,
    rows_ptr,
    slot_ptr,
    SPLITS: tl.constexpr,
    PROBES: tl.constexpr,
):
    seq = tl.program_id(0)
    if tl.load(slot_ptr + tl.load(rows_ptr + seq)) < 0:
        return
    probe = tl.arange(0, PROBES)
    split = tl.arange(0, SPLITS)
    index = (seq * SPLITS + split)[:, None] * PROBES + probe[None, :]
    counts = tl.sum(tl.load(part_count_ptr + index), axis=0)
    masses = tl.sum(tl.load(part_mass_ptr + index), axis=0)

    low_k = tl.load(bounds_ptr + seq * 4 + 0)
    high_k = tl.load(bounds_ptr + seq * 4 + 1)
    low_p = tl.load(bounds_ptr + seq * 4 + 2)
    high_p = tl.load(bounds_ptr + seq * 4 + 3)
    probes_k = tl.load(probes_ptr + seq * 2 * PROBES + probe)
    probes_p = tl.load(probes_ptr + seq * 2 * PROBES + PROBES + probe)

    fits_k = counts <= tl.load(top_k_ptr + seq).to(tl.float32)
    high_k = tl.min(tl.where(fits_k, probes_k, high_k), axis=0)
    low_k = tl.max(tl.where(fits_k, low_k, probes_k), axis=0)
    fits_p = masses >= tl.load(top_p_ptr + seq)
    low_p = tl.max(tl.where(fits_p, probes_p, low_p), axis=0)
    high_p = tl.min(tl.where(fits_p, high_p, probes_p), axis=0)

    tl.store(bounds_ptr + seq * 4 + 0, low_k)
    tl.store(bounds_ptr + seq * 4 + 1, high_k)
    tl.store(bounds_ptr + seq * 4 + 2, low_p)
    tl.store(bounds_ptr + seq * 4 + 3, high_p)
    step_k = (high_k - low_k) / (PROBES + 1)
    step_p = (high_p - low_p) / (PROBES + 1)
    tl.store(
        probes_ptr + seq * 2 * PROBES + probe,
        low_k + (probe + 1).to(tl.float32) * step_k,
    )
    tl.store(
        probes_ptr + seq * 2 * PROBES + PROBES + probe,
        low_p + (probe + 1).to(tl.float32) * step_p,
    )
    tl.store(threshold_ptr + seq, tl.maximum(high_k, low_p))


@triton.jit
def _wide_mass_kernel(
    logits_ptr,
    bitset_ptr,
    slot_ptr,
    rows_ptr,
    temperature_ptr,
    stats_ptr,
    threshold_ptr,
    part_mass_ptr,
    vocab_size,
    bitset_words,
    SPLITS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    seq = tl.program_id(0)
    split = tl.program_id(1)
    row = tl.load(rows_ptr + seq)
    slot = tl.load(slot_ptr + row)
    if slot < 0:
        return
    base = slot.to(tl.int64) * bitset_words
    logit_base = seq.to(tl.int64) * vocab_size
    temperature = tl.load(temperature_ptr + seq)
    maximum = tl.load(stats_ptr + seq * 2)
    total = tl.load(stats_ptr + seq * 2 + 1)
    threshold = tl.load(threshold_ptr + seq)
    lane = tl.arange(0, BLOCK)
    start, stop = _chunk_bounds(split, vocab_size, SPLITS)

    kept = 0.0
    for offset in range(start, stop, BLOCK):
        offsets = offset + lane
        inside = offsets < stop
        word = tl.load(bitset_ptr + base + (offsets >> 5), mask=inside, other=0)
        allowed = inside & (((word >> (offsets & 31)) & 1) == 1)
        values = tl.load(
            logits_ptr + logit_base + offsets, mask=allowed, other=0.0
        ).to(tl.float32) / temperature
        probs = tl.where(allowed, tl.exp(values - maximum) / total, 0.0)
        kept += tl.sum(tl.where(probs >= threshold, probs, 0.0), axis=0)
    tl.store(part_mass_ptr + seq * SPLITS + split, kept)


@triton.jit
def _wide_prefix_kernel(
    part_mass_ptr,
    prefix_ptr,
    target_ptr,
    uniform_ptr,
    rows_ptr,
    slot_ptr,
    SPLITS: tl.constexpr,
):
    seq = tl.program_id(0)
    if tl.load(slot_ptr + tl.load(rows_ptr + seq)) < 0:
        return
    lane = tl.arange(0, SPLITS)
    masses = tl.load(part_mass_ptr + seq * SPLITS + lane)
    exclusive = tl.cumsum(masses, axis=0) - masses
    tl.store(prefix_ptr + seq * SPLITS + lane, exclusive)
    total = tl.sum(masses, axis=0)
    tl.store(target_ptr + seq, tl.load(uniform_ptr + seq) * total)


@triton.jit
def _wide_draw_kernel(
    logits_ptr,
    bitset_ptr,
    slot_ptr,
    rows_ptr,
    temperature_ptr,
    stats_ptr,
    threshold_ptr,
    prefix_ptr,
    target_ptr,
    chosen_ptr,
    last_ptr,
    vocab_size,
    bitset_words,
    SPLITS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    seq = tl.program_id(0)
    split = tl.program_id(1)
    row = tl.load(rows_ptr + seq)
    slot = tl.load(slot_ptr + row)
    if slot < 0:
        return
    base = slot.to(tl.int64) * bitset_words
    logit_base = seq.to(tl.int64) * vocab_size
    temperature = tl.load(temperature_ptr + seq)
    maximum = tl.load(stats_ptr + seq * 2)
    total = tl.load(stats_ptr + seq * 2 + 1)
    threshold = tl.load(threshold_ptr + seq)
    target = tl.load(target_ptr + seq)
    running = tl.load(prefix_ptr + seq * SPLITS + split)
    lane = tl.arange(0, BLOCK)
    start, stop = _chunk_bounds(split, vocab_size, SPLITS)

    local = UNSET
    last = -1
    for offset in range(start, stop, BLOCK):
        offsets = offset + lane
        inside = offsets < stop
        word = tl.load(bitset_ptr + base + (offsets >> 5), mask=inside, other=0)
        allowed = inside & (((word >> (offsets & 31)) & 1) == 1)
        values = tl.load(
            logits_ptr + logit_base + offsets, mask=allowed, other=0.0
        ).to(tl.float32) / temperature
        probs = tl.where(allowed, tl.exp(values - maximum) / total, 0.0)
        kept = tl.where(probs >= threshold, probs, 0.0)
        cumulative = running + tl.cumsum(kept, axis=0)
        reached = (cumulative >= target) & (kept > 0.0)
        hit = tl.min(tl.where(reached, offsets, UNSET), axis=0)
        local = tl.where(local == UNSET, hit, local)
        last = tl.maximum(last, tl.max(tl.where(kept > 0.0, offsets, -1), axis=0))
        running += tl.sum(kept, axis=0)

    tl.atomic_min(chosen_ptr + seq, local)
    tl.atomic_max(last_ptr + seq, last)


@triton.jit
def _wide_finalize_kernel(
    chosen_ptr,
    last_ptr,
    override_indptr_ptr,
    override_tokens_ptr,
    override_states_ptr,
    default_state_ptr,
    rows_ptr,
    slot_ptr,
    out_tokens_ptr,
    out_states_ptr,
    WRITE_STATE: tl.constexpr,
):
    seq = tl.program_id(0)
    row = tl.load(rows_ptr + seq)
    slot = tl.load(slot_ptr + row)
    if slot < 0:
        return
    chosen = tl.load(chosen_ptr + seq)
    chosen = tl.where(chosen == UNSET, tl.load(last_ptr + seq), chosen)
    chosen = tl.where(chosen < 0, 0, chosen)
    tl.store(out_tokens_ptr + seq, chosen)
    if WRITE_STATE:
        # A wide row keeps no token list, so its successor is the row's
        # default state unless the token appears in a small override list.
        left = tl.load(override_indptr_ptr + slot)
        right = tl.load(override_indptr_ptr + slot + 1)
        state = tl.load(default_state_ptr + slot)
        for _ in range(24):
            middle = (left + right) // 2
            token = tl.load(
                override_tokens_ptr + middle, mask=middle < right, other=UNSET
            )
            go_right = (token < chosen) & (left < right)
            left = tl.where(go_right, middle + 1, left)
            right = tl.where(go_right, right, middle)
        found = tl.load(
            override_tokens_ptr + left,
            mask=left < tl.load(override_indptr_ptr + slot + 1),
            other=UNSET,
        )
        state = tl.where(
            found == chosen,
            tl.load(
                override_states_ptr + left,
                mask=left < tl.load(override_indptr_ptr + slot + 1),
                other=0,
            ),
            state,
        )
        tl.store(out_states_ptr + seq, state)


@dataclass
class WideSamplerWorkspace:
    """Scratch buffers reused across steps so the hot path allocates nothing."""

    splits: int
    probes: int
    part_max: torch.Tensor
    part_sum: torch.Tensor
    stats: torch.Tensor
    bounds: torch.Tensor
    probe_values: torch.Tensor
    part_count: torch.Tensor
    part_mass: torch.Tensor
    chunk_mass: torch.Tensor
    prefix: torch.Tensor
    threshold: torch.Tensor
    target: torch.Tensor
    chosen: torch.Tensor
    last: torch.Tensor


def make_wide_workspace(
    batch: int,
    *,
    splits: int,
    probes: int,
    device: torch.device,
) -> WideSamplerWorkspace:
    empty = lambda *shape: torch.empty(  # noqa: E731
        *shape, dtype=torch.float32, device=device
    )
    return WideSamplerWorkspace(
        splits=splits,
        probes=probes,
        part_max=empty(batch, splits),
        part_sum=empty(batch, splits),
        stats=empty(batch, 2),
        bounds=empty(batch, 4),
        probe_values=empty(batch, 2 * probes),
        part_count=empty(batch, splits, probes),
        part_mass=empty(batch, splits, probes),
        chunk_mass=empty(batch, splits),
        prefix=empty(batch, splits),
        threshold=empty(batch),
        target=empty(batch),
        chosen=torch.empty(batch, dtype=torch.int32, device=device),
        last=torch.empty(batch, dtype=torch.int32, device=device),
    )


def wide_sample_split(
    logits: torch.Tensor,
    tables,
    rows: torch.Tensor,
    *,
    temperature: torch.Tensor,
    top_k: torch.Tensor,
    top_p: torch.Tensor,
    uniform: torch.Tensor,
    out_tokens: torch.Tensor,
    out_states: torch.Tensor | None,
    workspace: WideSamplerWorkspace,
    rounds: int,
    block: int = 4096,
    num_warps: int = 8,
) -> None:
    batch = logits.shape[0]
    vocab_size = logits.shape[1]
    splits = workspace.splits
    probes = workspace.probes
    write_state = out_states is not None and tables.csr_next_state is not None
    grid = (batch, splits)
    shared = dict(
        vocab_size=vocab_size,
        bitset_words=tables.bitset_words,
        SPLITS=splits,
        BLOCK=block,
        num_warps=num_warps,
    )

    _wide_stats_kernel[grid](
        logits,
        tables.bitset,
        tables.row_slot,
        rows,
        temperature,
        workspace.part_max,
        workspace.part_sum,
        **shared,
    )
    _wide_reduce_stats_kernel[(batch,)](
        workspace.part_max,
        workspace.part_sum,
        workspace.stats,
        workspace.bounds,
        workspace.probe_values,
        rows,
        tables.row_slot,
        SPLITS=splits,
        PROBES=probes,
    )
    for _ in range(rounds):
        _wide_probe_kernel[grid](
            logits,
            tables.bitset,
            tables.row_slot,
            rows,
            temperature,
            workspace.stats,
            workspace.probe_values,
            workspace.part_count,
            workspace.part_mass,
            PROBES=probes,
            **shared,
        )
        _wide_update_kernel[(batch,)](
            workspace.part_count,
            workspace.part_mass,
            workspace.bounds,
            workspace.probe_values,
            top_k,
            top_p,
            workspace.threshold,
            rows,
            tables.row_slot,
            SPLITS=splits,
            PROBES=probes,
        )
    _wide_mass_kernel[grid](
        logits,
        tables.bitset,
        tables.row_slot,
        rows,
        temperature,
        workspace.stats,
        workspace.threshold,
        workspace.chunk_mass,
        **shared,
    )
    _wide_prefix_kernel[(batch,)](
        workspace.chunk_mass,
        workspace.prefix,
        workspace.target,
        uniform,
        rows,
        tables.row_slot,
        SPLITS=splits,
    )
    workspace.chosen.fill_(int(UNSET.value))
    workspace.last.fill_(-1)
    _wide_draw_kernel[grid](
        logits,
        tables.bitset,
        tables.row_slot,
        rows,
        temperature,
        workspace.stats,
        workspace.threshold,
        workspace.prefix,
        workspace.target,
        workspace.chosen,
        workspace.last,
        **shared,
    )
    _wide_finalize_kernel[(batch,)](
        workspace.chosen,
        workspace.last,
        tables.override_indptr,
        tables.override_tokens,
        tables.override_states,
        tables.default_state,
        rows,
        tables.row_slot,
        out_tokens,
        out_states if write_state else out_tokens,
        WRITE_STATE=write_state,
    )
