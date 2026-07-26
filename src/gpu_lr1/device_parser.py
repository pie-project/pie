"""Run the parser on the device, so a decode step never reaches the host.

This is the thesis in code. A constrained decode step needs, per sequence, the
set of tokens the grammar admits. Today's engines compute that on the CPU and
hand it to the GPU, which means the grammar cannot be inside the model's CUDA
graph, cannot verify speculative drafts without paying for each one on the
critical path, and cannot feed the sampler directly.

Here the parser state — a lexer state and an LR stack — lives in device memory
and is advanced by kernels. One step is:

```text
for each group of this sequence's lexer state          # 1 to 13 of them
    for each way the group's tokens can be read        # usually one
        replay its terminals against a copy of the stack
    if any reading survives, scatter the group's tokens into the mask
```

The replay is the interesting part: a reduction pops the stack and the goto that
follows depends on the state underneath, so this is a real pushdown step, not a
table lookup. It fits on device because LR gives a *viable prefix* property —
which terminals are admissible follows from the stack top alone — so only the
advance needs the stack, and the stack is per sequence and small.

Nothing here is per-request except the stacks and the mask. The tables are a
pure function of the grammar and the vocabulary.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import triton
import triton.language as tl

_GROUP_BLOCK = 64
# Configurations admitted per pass. Bounds the replay scratch, which is one
# stack copy per (sequence, configuration, group).
_CONFIG_CHUNK = 4
ACCEPT = -(2**31)
SPARSE, COMPLEMENT, DENSE = 0, 1, 2

_ACCEPT = tl.constexpr(ACCEPT)
_SPARSE = tl.constexpr(SPARSE)
_COMPLEMENT = tl.constexpr(COMPLEMENT)
_DENSE = tl.constexpr(DENSE)


@triton.jit
def _search(keys_ptr, low, high, needle):
    """Index of `needle` in the sorted run `[low, high)`, or -1."""
    found = -1
    while low < high:
        middle = (low + high) // 2
        value = tl.load(keys_ptr + middle)
        if value == needle:
            found = middle
            low = high
        elif value < needle:
            low = middle + 1
        else:
            high = middle
    return found


@triton.jit
def _mask_kernel(
    # tables
    group_offsets_ptr,
    group_set_kind_ptr,
    group_set_offset_ptr,
    group_set_length_ptr,
    set_payload_ptr,
    reading_offsets_ptr,
    reading_index_ptr,
    reading_next_state_ptr,
    reading_term_offsets_ptr,
    reading_terminals_ptr,
    action_offsets_ptr,
    action_terminals_ptr,
    action_values_ptr,
    goto_offsets_ptr,
    goto_nonterminals_ptr,
    goto_targets_ptr,
    production_lhs_ptr,
    production_arity_ptr,
    pending_offsets_ptr,
    pending_terminals_ptr,
    # per sequence
    lexer_state_ptr,
    stack_ptr,
    stack_depth_ptr,
    config_count_ptr,
    widest_ptr,
    representative_ptr,
    scratch_ptr,
    admitted_ptr,
    mask_ptr,
    overflow_ptr,
    mask_words,
    LIVE,
    config_base,
    BATCH: tl.constexpr,
    CONFIGS: tl.constexpr,
    MAX_GROUPS: tl.constexpr,
    STACK_STRIDE: tl.constexpr,
    MAX_REDUCTIONS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """One program per (sequence, configuration, group).

    Groups are independent - each asks whether the parser survives one reading
    of one set of tokens - so they are the axis to parallelise over. A sequence
    can have hundreds of them, and deciding them in series inside one program is
    what makes a large schema slow however few sequences there are.

    A sequence can hold more than one configuration, because scanning a
    generated lexicon is not deterministic: a declared property name is also a
    generic string, and `{` is both a complete terminal and the start of a
    longer one. A token is admissible when *some* live configuration admits it,
    so configurations union into the same mask row and need no coordination.
    """
    launched = tl.program_id(0)
    slot = tl.program_id(1)
    sequence = launched % BATCH
    config = config_base + launched // BATCH
    if config >= tl.load(widest_ptr):
        return
    if config >= tl.load(config_count_ptr + sequence):
        return
    # Sequence varies fastest. The grid is sized for the configuration ceiling
    # because the real width lives on the device and asking for it would be a
    # synchronisation, so most of the grid is programs that exit at once - and
    # laying them out this way puts those exits in whole blocks rather than
    # scattering one live program among fifteen dead ones in every block.
    # A sequence whose parse state an earlier one already holds does no work;
    # its mask is copied afterwards. In a serving batch that is 93% to 95% of
    # them, because there are only so many places to be in one grammar.
    if tl.load(representative_ptr + sequence) != sequence:
        return
    row_index = sequence * CONFIGS + config
    state = tl.load(lexer_state_ptr + row_index)
    depth = tl.load(stack_depth_ptr + row_index)
    first = tl.load(group_offsets_ptr + state)
    last = tl.load(group_offsets_ptr + state + 1)
    group = first + slot
    if group >= last:
        return

    # Replaying may push, so each program needs its own stack copy. The live
    # prefix is shared and read-only; the copy only has to hold what a replay
    # adds, which is bounded by the reduction limit.
    scratch = (launched * tl.num_programs(1) + slot) * 2 * STACK_STRIDE
    probe = scratch + STACK_STRIDE
    base = row_index * STACK_STRIDE

    admitted = 0
    use = tl.load(reading_offsets_ptr + group)
    use_end = tl.load(reading_offsets_ptr + group + 1)
    while use < use_end:
        reading = tl.load(reading_index_ptr + use)
        # Most groups die on their first terminal, so the stack is only copied
        # once that is known to be shiftable. Copying first, and a byte at a
        # time, cost more than the decision it was for.
        term = tl.load(reading_term_offsets_ptr + reading)
        term_end = tl.load(reading_term_offsets_ptr + reading + 1)
        top = tl.load(stack_ptr + base + depth - 1)
        alive = 1
        if term < term_end:
            terminal = tl.load(reading_terminals_ptr + term)
            row = tl.load(action_offsets_ptr + top)
            row_end = tl.load(action_offsets_ptr + top + 1)
            if _search(action_terminals_ptr, row, row_end, terminal) < 0:
                alive = 0

        if alive == 1:
            lane = tl.arange(0, STACK_STRIDE)
            tl.store(
                scratch_ptr + scratch + lane,
                tl.load(stack_ptr + base + lane, mask=lane < depth, other=0),
                mask=lane < depth,
            )
        copy_depth = depth
        while term < term_end and alive == 1:
            terminal = tl.load(reading_terminals_ptr + term)
            settled = 0
            # Bounded, but not fixed. A reduction chain ends at the first
            # shift, and on real documents that is two to four steps, while the
            # bound has to cover the deepest chain the grammar admits - the
            # stack depth, 256. Running the bound every time did sixty times
            # the work a typical token needs. The counter is a guard against a
            # grammar that never settles, not the schedule.
            spins = 0
            while settled == 0 and alive == 1 and spins < MAX_REDUCTIONS:
                spins = spins + 1
                if settled == 0 and alive == 1:
                    row = tl.load(action_offsets_ptr + top)
                    row_end = tl.load(action_offsets_ptr + top + 1)
                    entry = _search(action_terminals_ptr, row, row_end, terminal)
                    if entry < 0:
                        alive = 0
                    else:
                        value = tl.load(action_values_ptr + entry)
                        if value == _ACCEPT:
                            alive = 0
                        elif value > 0:
                            if copy_depth >= STACK_STRIDE:
                                alive = 0
                                tl.store(overflow_ptr + sequence, 1)
                            else:
                                tl.store(scratch_ptr + scratch + copy_depth, value - 1)
                                copy_depth = copy_depth + 1
                                top = value - 1
                                settled = 1
                        else:
                            production = -value - 1
                            arity = tl.load(production_arity_ptr + production)
                            if copy_depth <= arity:
                                alive = 0
                            else:
                                copy_depth = copy_depth - arity
                                exposed = tl.load(
                                    scratch_ptr + scratch + copy_depth - 1
                                )
                                lhs = tl.load(production_lhs_ptr + production)
                                grow = tl.load(goto_offsets_ptr + exposed)
                                grow_end = tl.load(goto_offsets_ptr + exposed + 1)
                                target = _search(
                                    goto_nonterminals_ptr, grow, grow_end, lhs
                                )
                                if target < 0:
                                    alive = 0
                                else:
                                    top = tl.load(goto_targets_ptr + target)
                                    tl.store(
                                        scratch_ptr + scratch + copy_depth, top
                                    )
                                    copy_depth = copy_depth + 1
            if settled == 0:
                alive = 0
            term = term + 1

        # A reading that leaves a lexeme in progress needs some continuation the
        # parser would accept, or a finished document could be followed by the
        # opening of another. Asking only whether an action exists is not the
        # same question: a reduce action may still fail once it has popped, and
        # taking the shortcut admitted tokens the reference matcher refuses.
        if alive == 1:
            next_state = tl.load(reading_next_state_ptr + reading)
            pend = tl.load(pending_offsets_ptr + next_state)
            pend_end = tl.load(pending_offsets_ptr + next_state + 1)
            if pend < pend_end:
                any_ok = 0
                while pend < pend_end and any_ok == 0:
                    terminal = tl.load(pending_terminals_ptr + pend)
                    # Probe on a copy, since a reduce rewrites the stack.
                    lane = tl.arange(0, STACK_STRIDE)
                    tl.store(
                        scratch_ptr + probe + lane,
                        tl.load(
                            scratch_ptr + scratch + lane,
                            mask=lane < copy_depth,
                            other=0,
                        ),
                        mask=lane < copy_depth,
                    )
                    probe_depth = copy_depth
                    probe_top = top
                    probe_alive = 1
                    probe_settled = 0
                    probe_spins = 0
                    while (
                        probe_settled == 0
                        and probe_alive == 1
                        and probe_spins < MAX_REDUCTIONS
                    ):
                        probe_spins = probe_spins + 1
                        if probe_settled == 0 and probe_alive == 1:
                            row = tl.load(action_offsets_ptr + probe_top)
                            row_end = tl.load(action_offsets_ptr + probe_top + 1)
                            entry = _search(
                                action_terminals_ptr, row, row_end, terminal
                            )
                            if entry < 0:
                                probe_alive = 0
                            else:
                                value = tl.load(action_values_ptr + entry)
                                if value == _ACCEPT:
                                    probe_settled = 1
                                elif value > 0:
                                    probe_settled = 1
                                else:
                                    production = -value - 1
                                    arity = tl.load(production_arity_ptr + production)
                                    if probe_depth <= arity:
                                        probe_alive = 0
                                    else:
                                        probe_depth = probe_depth - arity
                                        exposed = tl.load(
                                            scratch_ptr + probe + probe_depth - 1
                                        )
                                        lhs = tl.load(production_lhs_ptr + production)
                                        grow = tl.load(goto_offsets_ptr + exposed)
                                        grow_end = tl.load(
                                            goto_offsets_ptr + exposed + 1
                                        )
                                        target = _search(
                                            goto_nonterminals_ptr, grow, grow_end, lhs
                                        )
                                        if target < 0:
                                            probe_alive = 0
                                        else:
                                            probe_top = tl.load(
                                                goto_targets_ptr + target
                                            )
                                            tl.store(
                                                scratch_ptr + probe + probe_depth,
                                                probe_top,
                                            )
                                            probe_depth = probe_depth + 1
                    if probe_alive == 1 and probe_settled == 1:
                        any_ok = 1
                    pend = pend + 1
                alive = any_ok

        if alive == 1:
            admitted = 1
            use = use_end
        else:
            use = use + 1

    if admitted == 1:
        tl.store(admitted_ptr + row_index * MAX_GROUPS + slot, 1)
        # A complement group is written here, before anything else: it sets the
        # whole mask and then punches its exclusions out, and the groups of one
        # lexer state are disjoint, so every other admitted group's tokens are
        # among those exclusions. Punching after them would erase them. The
        # additive groups follow in a second pass.



@triton.jit
def _scatter_kernel(
    group_offsets_ptr,
    group_set_kind_ptr,
    group_set_offset_ptr,
    group_set_length_ptr,
    set_payload_ptr,
    lexer_state_ptr,
    admitted_ptr,
    mask_ptr,
    mask_words,
    LIVE,
    BATCH: tl.constexpr,
    CONFIGS: tl.constexpr,
    MAX_GROUPS: tl.constexpr,
    COMPLEMENTS_ONLY: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Write the sets of the admitted groups. Launched twice.

    A complement sets the whole mask row and then punches its exclusions out,
    and the groups of a lexer state are disjoint, so every other admitted
    group's tokens are among those exclusions - writing a complement after them
    would erase them. This used to live in the admission kernel, where one
    launch covered every configuration and the ordering came for free. It no
    longer does: admission runs in chunks so that its replay scratch is
    bounded, and a complement in the first chunk would land after the additive
    groups of the second. Two launches make the ordering the ordering of the
    launches, which is the only ordering CUDA gives without a barrier.
    """
    launched = tl.program_id(0)
    slot = tl.program_id(1)
    # Admission is recorded at the absolute row, since it runs in chunks and a
    # chunk-relative index would have the chunks overwrite each other.
    sequence = launched % BATCH
    row_index = sequence * CONFIGS + launched // BATCH
    if tl.load(admitted_ptr + row_index * MAX_GROUPS + slot) == 0:
        return
    state = tl.load(lexer_state_ptr + row_index)
    group = tl.load(group_offsets_ptr + state) + slot
    kind = tl.load(group_set_kind_ptr + group)
    if COMPLEMENTS_ONLY == 1:
        if kind != _COMPLEMENT:
            return
    elif kind == _COMPLEMENT:
        return
    offset = tl.load(group_set_offset_ptr + group)
    length = tl.load(group_set_length_ptr + group)
    row = mask_ptr + sequence * mask_words
    if kind == _COMPLEMENT:
        for start in range(0, mask_words, BLOCK):
            lane = start + tl.arange(0, BLOCK)
            live = lane < mask_words
            tl.atomic_or(row + lane, tl.full((BLOCK,), -1, tl.int32), mask=live)
        for start in range(0, length, BLOCK):
            lane = start + tl.arange(0, BLOCK)
            live = lane < length
            token = tl.load(set_payload_ptr + offset + lane, mask=live, other=0)
            tl.atomic_and(
                row + token // 32, (~(1 << (token % 32))).to(tl.int32), mask=live
            )
    elif kind == _SPARSE:
        for start in range(0, length, BLOCK):
            lane = start + tl.arange(0, BLOCK)
            live = lane < length
            token = tl.load(set_payload_ptr + offset + lane, mask=live, other=0)
            tl.atomic_or(
                row + token // 32, (1 << (token % 32)).to(tl.int32), mask=live
            )
    else:
        for start in range(0, mask_words, BLOCK):
            lane = start + tl.arange(0, BLOCK)
            live = lane < mask_words
            value = tl.load(set_payload_ptr + offset + lane, mask=live, other=0)
            tl.atomic_or(row + lane, value, mask=live)


@triton.jit
def _contains(kind, offset, length, payload_ptr, token):
    """Is `token` in this group's set?

    The three storages are the three shapes a set of tokens takes when it is
    stored exactly: a sorted list, a sorted list of exclusions, or a bitset.

    One exit, deliberately. A `return` inside a runtime branch of a jitted
    helper does not reliably do what it reads as - the branch is a predicate,
    not a jump - and written that way this silently reported that no group held
    the token, which stalls the parser instead of failing.
    """
    inside = 0
    if kind == _DENSE:
        word = tl.load(payload_ptr + offset + token // 32)
        inside = (word >> (token % 32)) & 1
    else:
        # The list is sorted, so its ends are its bounds. Almost every group of
        # a lexer state fails on them, and two adjacent loads settle that far
        # sooner than a search does - the search is a chain of dependent loads
        # into scattered memory, and the cost of this kernel is exactly how
        # many of those it performs.
        low = tl.load(payload_ptr + offset)
        high = tl.load(payload_ptr + offset + length - 1)
        found = -1
        if token >= low:
            if token <= high:
                found = _search(payload_ptr, offset, offset + length, token)
        if kind == _COMPLEMENT:
            if found < 0:
                inside = 1
        else:
            if found >= 0:
                inside = 1
    return inside


@triton.jit
def _candidate_kernel(
    group_offsets_ptr,
    group_set_kind_ptr,
    group_set_offset_ptr,
    group_set_length_ptr,
    set_payload_ptr,
    reading_offsets_ptr,
    reading_index_ptr,
    reading_next_state_ptr,
    reading_term_offsets_ptr,
    reading_terminals_ptr,
    action_offsets_ptr,
    action_terminals_ptr,
    action_values_ptr,
    goto_offsets_ptr,
    goto_nonterminals_ptr,
    goto_targets_ptr,
    production_lhs_ptr,
    production_arity_ptr,
    pending_offsets_ptr,
    pending_terminals_ptr,
    lexer_state_ptr,
    stack_ptr,
    stack_depth_ptr,
    config_count_ptr,
    widest_ptr,
    token_ptr,
    scratch_ptr,
    cand_valid_ptr,
    cand_lexer_ptr,
    cand_depth_ptr,
    cand_stack_ptr,
    overflow_ptr,
    mask_words,
    LIVE,
    BATCH: tl.constexpr,
    CONFIGS: tl.constexpr,
    GROUP_BLOCK: tl.constexpr,
    SEARCH_STEPS: tl.constexpr,
    MAX_READINGS: tl.constexpr,
    STACK_STRIDE: tl.constexpr,
    MAX_REDUCTIONS: tl.constexpr,
):
    """Where each configuration lands if the sampled token is accepted.

    The same replay the mask kernel runs, with two differences. It starts from
    the one group that holds the sampled token rather than sweeping all of
    them, and it keeps going after the first reading that survives: the mask
    only needed to know *whether* a token was admissible, while an advance
    needs every state the token could have led to, since each is a distinct
    parse that stays alive.

    The token arrives as a device pointer and is never read on the host. That
    is the whole point - a value the host has to see is a synchronisation, and
    a decode loop that synchronises once per token is the thing this design
    exists to remove.
    """
    launched = tl.program_id(0)
    block = tl.program_id(1)
    sequence = launched % BATCH
    config = launched // BATCH
    if config >= tl.load(widest_ptr):
        return
    if config >= tl.load(config_count_ptr + sequence):
        return
    row_index = sequence * CONFIGS + config
    state = tl.load(lexer_state_ptr + row_index)
    depth = tl.load(stack_depth_ptr + row_index)
    first = tl.load(group_offsets_ptr + state)
    last = tl.load(group_offsets_ptr + state + 1)

    # Find the group holding the token a block of groups at a time rather than
    # one per program. A state can have hundreds of groups and only one of them
    # holds any given token, so the search is nearly all rejection - and doing
    # it one program per group turns three contiguous arrays into three
    # scattered loads per group. Read as a block they are three coalesced
    # loads for the whole block, which is the difference this kernel is made
    # of: it is bound by how many scattered loads it issues, not by arithmetic.
    token = tl.load(token_ptr + sequence)
    glane = tl.arange(0, GROUP_BLOCK)
    group = first + block * GROUP_BLOCK + glane
    live_lane = group < last
    kind = tl.load(group_set_kind_ptr + group, mask=live_lane, other=0)
    offset = tl.load(group_set_offset_ptr + group, mask=live_lane, other=0)
    length = tl.load(group_set_length_ptr + group, mask=live_lane, other=1)

    dense = kind == _DENSE
    word = tl.load(
        set_payload_ptr + offset + token // 32, mask=live_lane & dense, other=0
    )
    in_dense = ((word >> (token % 32)) & 1) == 1

    # A sorted list's ends are its bounds, so most lanes are decided without a
    # search at all.
    listed = live_lane & (dense == 0)
    low = tl.load(set_payload_ptr + offset, mask=listed, other=1)
    high = tl.load(set_payload_ptr + offset + length - 1, mask=listed, other=0)
    searching = listed & (token >= low) & (token <= high)
    # A plain halving. A lane that has found its answer stops being active
    # rather than having its bounds collapsed: writing `hi = lo` on a hit made
    # `hi` depend on an already-updated `lo` within the same step, and a lane
    # could then search backwards. Lanes that finish early idle, which costs
    # the block nothing it was not already paying.
    lo = offset
    hi = offset + length
    at = tl.zeros((GROUP_BLOCK,), tl.int32) - 1
    for _ in range(0, SEARCH_STEPS):
        active = searching & (lo < hi) & (at < 0)
        middle = (lo + hi) // 2
        value = tl.load(set_payload_ptr + middle, mask=active, other=0)
        at = tl.where(active & (value == token), middle, at)
        lo = tl.where(active & (value < token), middle + 1, lo)
        hi = tl.where(active & (value > token), middle, hi)
    found = at >= 0
    complement = kind == _COMPLEMENT
    inside = tl.where(
        dense, in_dense, tl.where(complement, found == 0, found)
    ) & live_lane

    if tl.sum(inside.to(tl.int32)) == 0:
        return
    # The *first* group holding the token, not any of them. A complement group
    # excludes what the others hold, so they should be disjoint and the choice
    # should not matter - but "should" is not a guarantee the emitter makes,
    # and where two groups do overlap the reference matcher takes the earlier.
    # Taking the later instead produced three configurations it did not have.
    group = tl.min(tl.where(inside, group, last))

    scratch = (launched * tl.num_programs(1) + block) * 2 * STACK_STRIDE
    probe = scratch + STACK_STRIDE
    base = row_index * STACK_STRIDE
    out_base = row_index * MAX_READINGS

    use = tl.load(reading_offsets_ptr + group)
    use_end = tl.load(reading_offsets_ptr + group + 1)
    index = 0
    while use < use_end and index < MAX_READINGS:
        reading = tl.load(reading_index_ptr + use)
        term = tl.load(reading_term_offsets_ptr + reading)
        term_end = tl.load(reading_term_offsets_ptr + reading + 1)
        top = tl.load(stack_ptr + base + depth - 1)
        alive = 1
        lane = tl.arange(0, STACK_STRIDE)
        tl.store(
            scratch_ptr + scratch + lane,
            tl.load(stack_ptr + base + lane, mask=lane < depth, other=0),
            mask=lane < depth,
        )
        copy_depth = depth
        while term < term_end and alive == 1:
            terminal = tl.load(reading_terminals_ptr + term)
            settled = 0
            # Bounded, but not fixed. A reduction chain ends at the first
            # shift, and on real documents that is two to four steps, while the
            # bound has to cover the deepest chain the grammar admits - the
            # stack depth, 256. Running the bound every time did sixty times
            # the work a typical token needs. The counter is a guard against a
            # grammar that never settles, not the schedule.
            spins = 0
            while settled == 0 and alive == 1 and spins < MAX_REDUCTIONS:
                spins = spins + 1
                if settled == 0 and alive == 1:
                    row = tl.load(action_offsets_ptr + top)
                    row_end = tl.load(action_offsets_ptr + top + 1)
                    entry = _search(action_terminals_ptr, row, row_end, terminal)
                    if entry < 0:
                        alive = 0
                    else:
                        value = tl.load(action_values_ptr + entry)
                        if value == _ACCEPT:
                            alive = 0
                        elif value > 0:
                            if copy_depth >= STACK_STRIDE:
                                alive = 0
                                tl.store(overflow_ptr + sequence, 1)
                            else:
                                tl.store(scratch_ptr + scratch + copy_depth, value - 1)
                                copy_depth = copy_depth + 1
                                top = value - 1
                                settled = 1
                        else:
                            production = -value - 1
                            arity = tl.load(production_arity_ptr + production)
                            if copy_depth <= arity:
                                alive = 0
                            else:
                                copy_depth = copy_depth - arity
                                exposed = tl.load(scratch_ptr + scratch + copy_depth - 1)
                                lhs = tl.load(production_lhs_ptr + production)
                                grow = tl.load(goto_offsets_ptr + exposed)
                                grow_end = tl.load(goto_offsets_ptr + exposed + 1)
                                target = _search(
                                    goto_nonterminals_ptr, grow, grow_end, lhs
                                )
                                if target < 0:
                                    alive = 0
                                elif copy_depth >= STACK_STRIDE:
                                    alive = 0
                                    tl.store(overflow_ptr + sequence, 1)
                                else:
                                    top = tl.load(goto_targets_ptr + target)
                                    tl.store(scratch_ptr + scratch + copy_depth, top)
                                    copy_depth = copy_depth + 1
            if settled == 0:
                alive = 0
            term = term + 1

        next_state = tl.load(reading_next_state_ptr + reading)
        if alive == 1:
            pend = tl.load(pending_offsets_ptr + next_state)
            pend_end = tl.load(pending_offsets_ptr + next_state + 1)
            if pend < pend_end:
                any_ok = 0
                while pend < pend_end and any_ok == 0:
                    terminal = tl.load(pending_terminals_ptr + pend)
                    lane = tl.arange(0, STACK_STRIDE)
                    tl.store(
                        scratch_ptr + probe + lane,
                        tl.load(
                            scratch_ptr + scratch + lane,
                            mask=lane < copy_depth,
                            other=0,
                        ),
                        mask=lane < copy_depth,
                    )
                    probe_depth = copy_depth
                    probe_top = top
                    probe_alive = 1
                    probe_settled = 0
                    probe_spins = 0
                    while (
                        probe_settled == 0
                        and probe_alive == 1
                        and probe_spins < MAX_REDUCTIONS
                    ):
                        probe_spins = probe_spins + 1
                        if probe_settled == 0 and probe_alive == 1:
                            row = tl.load(action_offsets_ptr + probe_top)
                            row_end = tl.load(action_offsets_ptr + probe_top + 1)
                            entry = _search(action_terminals_ptr, row, row_end, terminal)
                            if entry < 0:
                                probe_alive = 0
                            else:
                                value = tl.load(action_values_ptr + entry)
                                if value == _ACCEPT:
                                    probe_settled = 1
                                elif value > 0:
                                    probe_settled = 1
                                else:
                                    production = -value - 1
                                    arity = tl.load(production_arity_ptr + production)
                                    if probe_depth <= arity:
                                        probe_alive = 0
                                    else:
                                        probe_depth = probe_depth - arity
                                        exposed = tl.load(
                                            scratch_ptr + probe + probe_depth - 1
                                        )
                                        lhs = tl.load(production_lhs_ptr + production)
                                        grow = tl.load(goto_offsets_ptr + exposed)
                                        grow_end = tl.load(goto_offsets_ptr + exposed + 1)
                                        target = _search(
                                            goto_nonterminals_ptr, grow, grow_end, lhs
                                        )
                                        if target < 0:
                                            probe_alive = 0
                                        else:
                                            probe_top = tl.load(goto_targets_ptr + target)
                                            tl.store(
                                                scratch_ptr + probe + probe_depth,
                                                probe_top,
                                            )
                                            probe_depth = probe_depth + 1
                    if probe_alive == 1 and probe_settled == 1:
                        any_ok = 1
                    pend = pend + 1
                alive = any_ok

        if alive == 1:
            tl.store(cand_valid_ptr + out_base + index, 1)
            tl.store(cand_lexer_ptr + out_base + index, next_state)
            tl.store(cand_depth_ptr + out_base + index, copy_depth)
            lane = tl.arange(0, STACK_STRIDE)
            tl.store(
                cand_stack_ptr + (out_base + index) * STACK_STRIDE + lane,
                tl.load(scratch_ptr + scratch + lane, mask=lane < copy_depth, other=0),
                mask=lane < copy_depth,
            )
            index = index + 1
        use = use + 1


@triton.jit
def _commit_kernel(
    lexer_state_ptr,
    stack_ptr,
    stack_depth_ptr,
    config_count_ptr,
    old_lexer_ptr,
    old_count_ptr,
    cand_valid_ptr,
    cand_lexer_ptr,
    cand_depth_ptr,
    cand_stack_ptr,
    terminated_ptr,
    widest_ptr,
    CONFIGS: tl.constexpr,
    MAX_READINGS: tl.constexpr,
    STACK_STRIDE: tl.constexpr,
):
    """Collect the surviving candidates into the next configuration set.

    Serial, one program per sequence, and deliberately so. The reference
    matcher deduplicates in a particular order and stops at its configuration
    ceiling, so a parallel collection that produced the same *set* could still
    produce a different *prefix* once the ceiling bites. Reproducing the order
    is what lets the two be compared for equality rather than for similarity.
    """
    sequence = tl.program_id(0)
    count = tl.load(old_count_ptr + sequence)
    lane = tl.arange(0, STACK_STRIDE)
    written = 0

    # Bounded by the count, not by the ceiling. Running to 128 when the parse
    # holds one configuration made the advance cost 88 us instead of 27 - the
    # loops are nested, so the ceiling enters squared.
    state_slot = 0
    while state_slot < count and written < CONFIGS:
        if 1 == 1:
            state = tl.load(old_lexer_ptr + sequence * CONFIGS + state_slot)
            # Only the first configuration carrying a lexer state introduces
            # it; a later one would repeat every candidate the first produced.
            seen = 0
            earlier = 0
            while earlier < state_slot:
                if tl.load(old_lexer_ptr + sequence * CONFIGS + earlier) == state:
                    seen = 1
                earlier = earlier + 1
            if seen == 0:
                source = 0
                while source < count and written < CONFIGS:
                    if 1 == 1:
                        if tl.load(old_lexer_ptr + sequence * CONFIGS + source) == state:
                            base = (sequence * CONFIGS + source) * MAX_READINGS
                            for index in range(0, MAX_READINGS):
                                if written < CONFIGS:
                                    if tl.load(cand_valid_ptr + base + index) == 1:
                                        next_state = tl.load(cand_lexer_ptr + base + index)
                                        depth = tl.load(cand_depth_ptr + base + index)
                                        values = tl.load(
                                            cand_stack_ptr
                                            + (base + index) * STACK_STRIDE
                                            + lane,
                                            mask=lane < depth,
                                            other=0,
                                        )
                                        duplicate = 0
                                        done = 0
                                        while done < written:
                                                out = sequence * CONFIGS + done
                                                if (
                                                    tl.load(lexer_state_ptr + out)
                                                    == next_state
                                                ) and (
                                                    tl.load(stack_depth_ptr + out)
                                                    == depth
                                                ):
                                                    held = tl.load(
                                                        stack_ptr + out * STACK_STRIDE + lane,
                                                        mask=lane < depth,
                                                        other=0,
                                                    )
                                                    if tl.sum(
                                                        tl.where(
                                                            (lane < depth)
                                                            & (held != values),
                                                            1,
                                                            0,
                                                        )
                                                    ) == 0:
                                                        duplicate = 1
                                                done = done + 1
                                        if duplicate == 0:
                                            out = sequence * CONFIGS + written
                                            tl.store(lexer_state_ptr + out, next_state)
                                            tl.store(stack_depth_ptr + out, depth)
                                            tl.store(
                                                stack_ptr + out * STACK_STRIDE + lane,
                                                values,
                                                mask=lane < depth,
                                            )
                                            written = written + 1
                        source = source + 1
        state_slot = state_slot + 1

    # No candidate survived: the token was refused. The set is left as it was
    # and the sequence is marked, because a mask filled from an empty set would
    # silently allow everything.
    if written == 0:
        tl.store(terminated_ptr + sequence, 1)
    else:
        tl.store(config_count_ptr + sequence, written)
    # The widest set in the batch, maintained on the device. The fill's grid is
    # sized for the ceiling because the host may not ask, but every program can
    # read this and return at once - which turns the ceiling from work into a
    # launch.
    tl.atomic_max(widest_ptr, written)


@triton.jit
def _hash_kernel(
    lexer_state_ptr,
    stack_ptr,
    stack_depth_ptr,
    config_count_ptr,
    hash_ptr,
    CONFIGS: tl.constexpr,
    STACK_STRIDE: tl.constexpr,
):
    """A fingerprint of one sequence's parse state.

    A serving batch runs many sequences against the same grammar at different
    points of their own documents, and there are only so many places to be: on
    this corpus a batch of 512 holds 24 to 34 distinct parse states, so 93% to
    95% of the fill recomputes an answer it already has. Finding the duplicates
    needs a cheap comparison first, which is what this is - the exact check
    follows, because a mask that is wrong is worse than a mask that is slow.
    """
    sequence = tl.program_id(0)
    count = tl.load(config_count_ptr + sequence)
    digest = 2166136261
    digest = (digest ^ count) * 16777619
    for config in range(0, CONFIGS):
        if config < count:
            row = sequence * CONFIGS + config
            depth = tl.load(stack_depth_ptr + row)
            digest = (digest ^ tl.load(lexer_state_ptr + row)) * 16777619
            digest = (digest ^ depth) * 16777619
            lane = tl.arange(0, STACK_STRIDE)
            values = tl.load(
                stack_ptr + row * STACK_STRIDE + lane, mask=lane < depth, other=0
            )
            # Order matters, so fold with a position-dependent weight rather
            # than a plain sum.
            digest = (digest ^ tl.sum(values * (lane + 1))) * 16777619
    tl.store(hash_ptr + sequence, digest)


@triton.jit
def _dedup_kernel(
    lexer_state_ptr,
    stack_ptr,
    stack_depth_ptr,
    config_count_ptr,
    hash_ptr,
    representative_ptr,
    BATCH: tl.constexpr,
    CONFIGS: tl.constexpr,
    STACK_STRIDE: tl.constexpr,
):
    """The lowest-numbered sequence holding the same parse state.

    The fingerprint narrows the search; this confirms it exactly, because two
    different parse states that happen to hash alike would otherwise share a
    mask and one of them would be wrong.
    """
    sequence = tl.program_id(0)
    mine = tl.load(hash_ptr + sequence)
    count = tl.load(config_count_ptr + sequence)
    found = sequence
    for other in range(0, BATCH):
        if other < sequence and found == sequence:
            if tl.load(hash_ptr + other) == mine:
                if tl.load(config_count_ptr + other) == count:
                    same = 1
                    for config in range(0, CONFIGS):
                        if config < count:
                            a = sequence * CONFIGS + config
                            b = other * CONFIGS + config
                            depth = tl.load(stack_depth_ptr + a)
                            if tl.load(lexer_state_ptr + a) != tl.load(
                                lexer_state_ptr + b
                            ):
                                same = 0
                            if depth != tl.load(stack_depth_ptr + b):
                                same = 0
                            lane = tl.arange(0, STACK_STRIDE)
                            left = tl.load(
                                stack_ptr + a * STACK_STRIDE + lane,
                                mask=lane < depth,
                                other=0,
                            )
                            right = tl.load(
                                stack_ptr + b * STACK_STRIDE + lane,
                                mask=lane < depth,
                                other=0,
                            )
                            if tl.sum(tl.where(left != right, 1, 0)) != 0:
                                same = 0
                    if same == 1:
                        found = other
    tl.store(representative_ptr + sequence, found)


@triton.jit
def _broadcast_kernel(
    representative_ptr,
    mask_ptr,
    mask_words,
    BLOCK: tl.constexpr,
):
    """Copy each duplicate's mask from the sequence that computed it."""
    sequence = tl.program_id(0)
    source = tl.load(representative_ptr + sequence)
    if source == sequence:
        return
    for start in range(0, mask_words, BLOCK):
        lane = start + tl.arange(0, BLOCK)
        live = lane < mask_words
        value = tl.load(mask_ptr + source * mask_words + lane, mask=live, other=0)
        tl.store(mask_ptr + sequence * mask_words + lane, value, mask=live)


class DeviceGrammar:
    """A compiled grammar, resident on the GPU."""

    def __init__(
        self,
        compiled,
        max_stack: int = 256,
        max_reductions: int | None = None,
        max_configs: int = 16,
    ):
        arrays = compiled.device_arrays()
        self.vocab_size = int(arrays["vocab_size"])
        self.mask_words = int(arrays["bitset_words"])
        self.start_parser_state = int(arrays["start_parser_state"])
        offsets = np.frombuffer(arrays["group_offsets"], dtype=np.uint32)
        self.max_groups_per_state = int(np.diff(offsets).max()) if offsets.size > 1 else 1
        # Sixty-four was a guess and real documents reach ninety. A stack that
        # overflows is not a slow parser, it is a wrong one: the replay is
        # declared dead and the mask silently narrows. Every write is now
        # bounds-checked and an overflow is recorded rather than absorbed, so
        # the limit being too small is something that can be found out.
        self.max_stack = max_stack
        # How many reductions a single terminal may take before the parser is
        # declared stuck. Sixteen was a guess, and it was wrong: a `}` closing
        # a document nested thirty-seven deep needs more, and a reading that
        # runs out is treated as dead, so the mask silently refused a token the
        # reference matcher allowed. It took a corpus document 137 bytes long
        # to reach; the earlier check stopped at 31.
        #
        # A settle is a shift. Every step before it either pops or replaces the
        # top, so a chain longer than the stack can hold is not making progress
        # towards one.
        self.max_reductions = max_reductions if max_reductions is not None else max_stack
        self.max_configs = max_configs
        readings = np.frombuffer(arrays["reading_offsets"], dtype=np.uint32)
        # An advance keeps every reading that survives, not just the first, so
        # the candidate buffers are sized by the widest group in the grammar.
        self.max_readings = int(np.diff(readings).max()) if readings.size > 1 else 1
        lengths = np.frombuffer(arrays["group_set_length"], dtype=np.uint32)
        longest = int(lengths.max()) if lengths.size else 1
        # The lanes of a block search in lockstep, so the loop runs a fixed
        # number of times and every lane must have finished by it. A scalar
        # search over `n` needs ceil(log2(n)) steps, but a masked one is not
        # a clean halving - a lane that has already found its answer still
        # carries its bounds through the remaining iterations - and a bound
        # tight enough for the scalar case left five schemas disagreeing with
        # the reference matcher. The margin is cheap: the extra iterations are
        # masked off for every lane that has finished.
        # Lanes search in lockstep, so the loop runs a fixed number of times
        # and every lane must have finished by then. A halving over `n` needs
        # ceil(log2(n)) steps; the margin covers the ends being inclusive.
        self.search_steps = max(2, int(np.ceil(np.log2(longest + 2))) + 2)

        def upload(name: str, dtype=torch.int32) -> torch.Tensor:
            return torch.frombuffer(bytearray(arrays[name]), dtype=dtype).cuda()

        for name in (
            "group_offsets",
            "reading_index",
            "group_set_kind",
            "group_set_offset",
            "group_set_length",
            "set_payload",
            "reading_offsets",
            "reading_next_state",
            "reading_term_offsets",
            "reading_terminals",
            "action_offsets",
            "action_terminals",
            "action_values",
            "goto_offsets",
            "goto_nonterminals",
            "goto_targets",
            "production_lhs",
            "production_arity",
            "pending_offsets",
            "pending_terminals",
        ):
            setattr(self, name, upload(name))

    def resident_bytes(self) -> int:
        return sum(
            getattr(self, name).numel() * 4
            for name in (
                "group_offsets",
                "group_set_kind",
                "group_set_offset",
                "group_set_length",
                "set_payload",
                "reading_offsets",
                "reading_next_state",
                "reading_term_offsets",
                "reading_terminals",
                "action_offsets",
                "action_terminals",
                "action_values",
                "goto_offsets",
                "goto_nonterminals",
                "goto_targets",
                "production_lhs",
                "production_arity",
                "pending_offsets",
                "pending_terminals",
            )
        )

    def new_batch(self, batch: int) -> "DeviceBatch":
        return DeviceBatch(self, batch)


class DeviceBatch:
    """Per-sequence parser state, in device memory."""

    def __init__(self, grammar: DeviceGrammar, batch: int):
        self.grammar = grammar
        self.batch = batch
        self.configs = grammar.max_configs
        self.live = 1
        self.graph: torch.cuda.CUDAGraph | None = None
        self.captured_live = 0
        self.advance_graph: torch.cuda.CUDAGraph | None = None

        readings = grammar.max_readings
        self.max_readings = readings
        slots = batch * self.configs * readings
        self.cand_valid = torch.zeros(slots, dtype=torch.int32, device="cuda")
        self.cand_lexer = torch.zeros(slots, dtype=torch.int32, device="cuda")
        self.cand_depth = torch.zeros(slots, dtype=torch.int32, device="cuda")
        self.cand_stack = torch.zeros(
            slots * grammar.max_stack, dtype=torch.int32, device="cuda"
        )
        self.old_lexer = torch.zeros(
            batch * self.configs, dtype=torch.int32, device="cuda"
        )
        self.old_count = torch.ones(batch, dtype=torch.int32, device="cuda")
        # One flag per sequence rather than one for the batch: a refusal is a
        # property of the sequence that hit it, and reading it back to find out
        # which would be the synchronisation this is all avoiding.
        self.terminated = torch.zeros(batch, dtype=torch.int32, device="cuda")
        self.overflow = torch.zeros(batch, dtype=torch.int32, device="cuda")
        self.widest = torch.ones(1, dtype=torch.int32, device="cuda")
        # The configuration width, copied back asynchronously and read one step
        # late so that no step ever synchronises to learn it.
        self.width_host = torch.ones(1, dtype=torch.int32).pin_memory()
        self.state_hash = torch.zeros(batch, dtype=torch.int32, device="cuda")
        self.representative = torch.arange(batch, dtype=torch.int32, device="cuda")
        self.pending_width = 1
        self.token = torch.zeros(batch, dtype=torch.int32, device="cuda")
        rows = batch * self.configs
        self.lexer_state = torch.zeros(rows, dtype=torch.int32, device="cuda")
        self.stack = torch.zeros(
            (rows, grammar.max_stack), dtype=torch.int32, device="cuda"
        )
        self.stack[:, 0] = grammar.start_parser_state
        self.depth = torch.ones(rows, dtype=torch.int32, device="cuda")
        self.config_count = torch.ones(batch, dtype=torch.int32, device="cuda")
        self.mask = torch.zeros(
            (batch, grammar.mask_words), dtype=torch.int32, device="cuda"
        )
        self.max_groups = grammar.max_groups_per_state
        self.advance_blocks = (self.max_groups + _GROUP_BLOCK - 1) // _GROUP_BLOCK
        self.admitted = torch.zeros(
            rows * grammar.max_groups_per_state, dtype=torch.int32, device="cuda"
        )
        self.config_count = torch.ones(batch, dtype=torch.int32, device="cuda")
        # Two stacks per program: one for the reading being replayed, one for
        # probing what a pending lexeme could still become.
        # Sized for the configurations actually in play, not for the ceiling.
        #
        # A program needs its own copy of the stack it is replaying, so the
        # scratch is one per (sequence, configuration, group) - and at the
        # ceiling of 128 on a wide schema at batch 512 that is 44 GB, which
        # simply does not allocate. The ceiling exists because some documents
        # reach it; the sets in play are almost always one or two, and `live`
        # already tracks that. Growing the buffer when `live` does costs a
        # reallocation on the rare step that widens a parse.
        self.scratch = self._scratch_for(_CONFIG_CHUNK)
        # The advance indexes its scratch by block, not by group - it sweeps
        # sixty-four groups per program where the fill takes one - so the two
        # cannot share a buffer. Sharing it had them writing over each other's
        # replays, which cost far more than the memory saved.
        self.advance_scratch = torch.zeros(
            (batch * self.advance_blocks, 2 * grammar.max_stack),
            dtype=torch.int32,
            device="cuda",
        )
        self.advance_live = 1

    def _scratch_for(self, live: int) -> torch.Tensor:
        return torch.zeros(
            (self.batch * live * self.max_groups, 2 * self.grammar.max_stack),
            dtype=torch.int32,
            device="cuda",
        )

    def set_configurations(
        self, sequence: int, configurations: list[tuple[int, list[int]]]
    ) -> None:
        """Put one sequence into a known set of parse states."""
        self.set_batch_configurations({sequence: configurations})

    def set_batch_configurations(
        self, per_sequence: dict[int, list[tuple[int, list[int]]]]
    ) -> None:
        """Put many sequences into known parse states, in one transfer each.

        Writing a row at a time is a host-to-device copy per row per step, and
        at a serving batch that dominates everything the kernel then does. The
        state is assembled on the host and sent once.
        """
        rows = max(per_sequence) + 1 if per_sequence else 0
        if rows == 0:
            return
        lexer = np.zeros(rows * self.configs, dtype=np.int32)
        stacks = np.zeros((rows * self.configs, self.grammar.max_stack), dtype=np.int32)
        depths = np.ones(rows * self.configs, dtype=np.int32)
        counts = np.ones(rows, dtype=np.int32)
        for sequence, configurations in per_sequence.items():
            if len(configurations) > self.configs:
                raise ValueError(
                    f"{len(configurations)} configurations exceeds the batch's "
                    f"limit of {self.configs}"
                )
            counts[sequence] = len(configurations)
            for index, (lexer_state, stack) in enumerate(configurations):
                row = sequence * self.configs + index
                lexer[row] = lexer_state
                stacks[row, : len(stack)] = stack
                depths[row] = len(stack)
        # The widest configuration set in the batch, recorded while it is still
        # on the host. Asking the device for it costs a synchronisation every
        # step - the one thing this design exists to avoid - and launching for
        # the ceiling instead wastes a grid dimension on programs that return
        # immediately. The host put the counts there; it can remember them.
        self.live = int(counts.max())
        self.lexer_state[: rows * self.configs].copy_(torch.from_numpy(lexer))
        self.stack[: rows * self.configs].copy_(torch.from_numpy(stacks))
        self.depth[: rows * self.configs].copy_(torch.from_numpy(depths))
        self.config_count[:rows].copy_(torch.from_numpy(counts))

    def advance(self, tokens: torch.Tensor) -> None:
        """Take one sampled token per sequence, entirely on device.

        `tokens` is a device tensor and its values are never read on the host.
        That is the requirement the rest of the design is in service of: a
        decode loop that has to look at a sampled token to advance its parser
        pays a device-to-host round trip per token, and no amount of making the
        parser itself faster removes it.
        """
        self.token.copy_(tokens.to(torch.int32).reshape(-1)[: self.batch])
        if self.advance_graph is not None:
            self.advance_graph.replay()
            return
        self._advance()

    def capture_advance(self) -> None:
        """Record the advance too, so a decode step launches two graphs."""
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                self._advance()
        torch.cuda.current_stream().wait_stream(stream)
        self.advance_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.advance_graph):
            self._advance()

    def _advance(self) -> None:
        grammar = self.grammar
        self.cand_valid.zero_()
        self.old_lexer.copy_(self.lexer_state)
        self.old_count.copy_(self.config_count)
        live = self.live
        if self.advance_live < live:
            self.advance_live = live
            self.advance_scratch = torch.zeros(
                (self.batch * live * self.advance_blocks, 2 * grammar.max_stack),
                dtype=torch.int32,
                device="cuda",
            )
        _candidate_kernel[(self.batch * live, self.advance_blocks)](
            grammar.group_offsets,
            grammar.group_set_kind,
            grammar.group_set_offset,
            grammar.group_set_length,
            grammar.set_payload,
            grammar.reading_offsets,
            grammar.reading_index,
            grammar.reading_next_state,
            grammar.reading_term_offsets,
            grammar.reading_terminals,
            grammar.action_offsets,
            grammar.action_terminals,
            grammar.action_values,
            grammar.goto_offsets,
            grammar.goto_nonterminals,
            grammar.goto_targets,
            grammar.production_lhs,
            grammar.production_arity,
            grammar.pending_offsets,
            grammar.pending_terminals,
            self.lexer_state,
            self.stack,
            self.depth,
            self.config_count,
            self.widest,
            self.token,
            self.advance_scratch,
            self.cand_valid,
            self.cand_lexer,
            self.cand_depth,
            self.cand_stack,
            self.overflow,
            grammar.mask_words,
            live,
            BATCH=self.batch,
            CONFIGS=self.configs,
            GROUP_BLOCK=_GROUP_BLOCK,
            SEARCH_STEPS=grammar.search_steps,
            MAX_READINGS=self.max_readings,
            STACK_STRIDE=grammar.max_stack,
            MAX_REDUCTIONS=grammar.max_reductions,
        )
        _commit_kernel[(self.batch,)](
            self.lexer_state,
            self.stack,
            self.depth,
            self.config_count,
            self.old_lexer,
            self.old_count,
            self.cand_valid,
            self.cand_lexer,
            self.cand_depth,
            self.cand_stack,
            self.terminated,
            self.widest,
            CONFIGS=self.configs,
            MAX_READINGS=self.max_readings,
            STACK_STRIDE=grammar.max_stack,
            num_warps=1,
        )
        # How wide the configuration sets now are, carried back one step late.
        #
        # The fill's grid is (sequence x configuration, group), so this number
        # is the whole cost of the fill: at batch 512 it is 291 us with a width
        # of one and 4,317 us with sixteen, in a straight line. A program past
        # the real width exits on its first instruction, having read the count
        # from device memory, but a launch that returns immediately is still a
        # launch. Real parses hold one or two configurations; the ceiling has
        # to be sixteen only because some documents reach it.
        #
        # Reading the width now would be a device-to-host synchronisation,
        # which is the one thing this design forbids. It does not have to be
        # read now. The copy is issued here and the host picks it up at the
        # start of the next fill, by which time it has long since landed - an
        # asynchronous copy into pinned memory costs about 4 us and saves four
        # thousand.
        #
        # Until then the ceiling is the only safe grid, because a stale width
        # is unsafe in exactly one direction: too small.
        self.width_host.copy_(self.config_count.max().reshape(1), non_blocking=True)
        self.live = self.configs
        self.pending_width = None

    def configurations(self, sequence: int) -> list[tuple[int, list[int]]]:
        """Read one sequence's parse state back. For tests only.

        Nothing in the decode loop calls this: it is a device-to-host copy, and
        the point of the loop is that it does not make one.
        """
        count = int(self.config_count[sequence])
        rows = []
        for index in range(count):
            row = sequence * self.configs + index
            depth = int(self.depth[row])
            stack = self.stack.reshape(-1)[
                row * self.grammar.max_stack : row * self.grammar.max_stack + depth
            ].tolist()
            rows.append((int(self.lexer_state[row]), stack))
        return rows

    def capture(self) -> None:
        """Record the fill as a CUDA graph and replay it thereafter.

        Two Triton launches cost about 110us of *host* time to issue - argument
        marshalling, not arithmetic - and on a small schema that is the entire
        measurement. It is also precisely the cost this design claims to remove
        from the critical path, so leaving it in would be answering the CPU
        bottleneck with a different CPU bottleneck.

        Capture is only possible because the fill no longer asks the device
        anything: every shape is fixed at construction and the one value that
        used to come back from the device, the live configuration count, is now
        remembered on the host. A graph cannot contain a synchronisation.
        """
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                self._fill()
        torch.cuda.current_stream().wait_stream(stream)

        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self._fill()
        self.captured_live = self.live

    def fill_mask(self) -> torch.Tensor:
        # Pick up the width the last advance measured. The copy was issued
        # then and has landed by now, so this reads pinned host memory rather
        # than synchronising with the device.
        if self.pending_width is None:
            self.pending_width = max(1, int(self.width_host[0]))
            live = min(self.configs, self.pending_width)
            if live != self.live:
                self.live = live
                if self.graph is not None:
                    self.graph = None
                    self._fill()
                    self.capture()
        # The graph baked in a configuration count. If the batch has since grown
        # a wider one the recording no longer covers it, so fall back rather
        # than silently mask too little.
        if self.graph is not None and self.live == self.captured_live:
            self.graph.replay()
            return self.mask
        return self._fill()

    def _fill(self) -> torch.Tensor:
        grammar = self.grammar
        self.mask.zero_()
        self.admitted.zero_()
        _hash_kernel[(self.batch,)](
            self.lexer_state,
            self.stack,
            self.depth,
            self.config_count,
            self.state_hash,
            CONFIGS=self.configs,
            STACK_STRIDE=grammar.max_stack,
            num_warps=1,
        )
        _dedup_kernel[(self.batch,)](
            self.lexer_state,
            self.stack,
            self.depth,
            self.config_count,
            self.state_hash,
            self.representative,
            BATCH=self.batch,
            CONFIGS=self.configs,
            STACK_STRIDE=grammar.max_stack,
            num_warps=1,
        )
        live = self.live
        # Wide parses run the admission pass in chunks so the replay scratch is
        # reused rather than sized for the widest set the batch might hold. One
        # program needs its own stack copy, and at a ceiling of 128 on a wide
        # schema at batch 512 that would be 44 GB - which does not allocate.
        # The chunk is only entered more than once when a parse has genuinely
        # widened, which is rare.
        for config_base in range(0, live, _CONFIG_CHUNK):
            span = min(_CONFIG_CHUNK, live - config_base)
            _mask_kernel[(self.batch * span, self.max_groups)](
                grammar.group_offsets,
                grammar.group_set_kind,
                grammar.group_set_offset,
                grammar.group_set_length,
                grammar.set_payload,
                grammar.reading_offsets,
                grammar.reading_index,
                grammar.reading_next_state,
                grammar.reading_term_offsets,
                grammar.reading_terminals,
                grammar.action_offsets,
                grammar.action_terminals,
                grammar.action_values,
                grammar.goto_offsets,
                grammar.goto_nonterminals,
                grammar.goto_targets,
                grammar.production_lhs,
                grammar.production_arity,
                grammar.pending_offsets,
                grammar.pending_terminals,
                self.lexer_state,
                self.stack,
                self.depth,
                self.config_count,
                self.widest,
                self.representative,
                self.scratch,
                self.admitted,
                self.mask,
                self.overflow,
                grammar.mask_words,
                live,
                config_base,
                BATCH=self.batch,
                CONFIGS=self.configs,
                MAX_GROUPS=self.max_groups,
                STACK_STRIDE=grammar.max_stack,
                MAX_REDUCTIONS=grammar.max_reductions,
                BLOCK=128,
                num_warps=1,
            )

        _scatter_kernel[(self.batch * live, self.max_groups)](
            grammar.group_offsets,
            grammar.group_set_kind,
            grammar.group_set_offset,
            grammar.group_set_length,
            grammar.set_payload,
            self.lexer_state,
            self.admitted,
            self.mask,
            grammar.mask_words,
            live,
            BATCH=self.batch,
            CONFIGS=self.configs,
            MAX_GROUPS=self.max_groups,
            COMPLEMENTS_ONLY=1,
            BLOCK=128,
            num_warps=1,
        )
        _scatter_kernel[(self.batch * live, self.max_groups)](
            grammar.group_offsets,
            grammar.group_set_kind,
            grammar.group_set_offset,
            grammar.group_set_length,
            grammar.set_payload,
            self.lexer_state,
            self.admitted,
            self.mask,
            grammar.mask_words,
            live,
            BATCH=self.batch,
            CONFIGS=self.configs,
            MAX_GROUPS=self.max_groups,
            COMPLEMENTS_ONLY=0,
            BLOCK=128,
            num_warps=1,
        )
        _broadcast_kernel[(self.batch,)](
            self.representative,
            self.mask,
            grammar.mask_words,
            BLOCK=128,
            num_warps=4,
        )
        # A complement sets the last word whole, so the bits past the final
        # token have to go: nothing may be allowed that is not a token.
        spare = grammar.mask_words * 32 - grammar.vocab_size
        if spare:
            self.mask[:, -1] &= 0xFFFFFFFF >> spare
        return self.mask


def _time(function, warmup: int = 5, iterations: int = 20) -> float:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        function()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000.0 / iterations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument(
        "--instances", type=Path, default=Path("results/jsonschemabench-instances.json")
    )
    parser.add_argument("--schema-index", type=int, default=0)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 32, 128, 512])
    parser.add_argument(
        "--states",
        choices=["visited", "start"],
        default="visited",
        help="which lexer states the sequences sit in. 'visited' replays the "
        "corpus document and samples the states it actually reaches; 'start' "
        "puts everything in the start state, which is the worst one - it has "
        "1,673 groups against a median of 11.",
    )
    parser.add_argument("--output", type=Path, default=None)
    arguments = parser.parse_args()

    import gpugrammar
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(arguments.model)
    vocabulary: list[bytes] = []
    for token_id in range(len(tokenizer)):
        piece = tokenizer.convert_ids_to_tokens(token_id)
        try:
            vocabulary.append(tokenizer.convert_tokens_to_string([piece]).encode())
        except Exception:  # noqa: BLE001
            vocabulary.append(b"")

    instances = json.loads(arguments.instances.read_text())["instances"]
    schema = instances[arguments.schema_index]["schema"]
    compiled = gpugrammar.Compiler(vocabulary).compile_json_schema(schema)
    grammar = DeviceGrammar(compiled)

    print(
        f"schema {arguments.schema_index}: {compiled.num_lexer_states} lexer states, "
        f"{compiled.num_groups} groups, {compiled.num_parser_states} parser states"
    )
    print(f"  resident on device: {grammar.resident_bytes() / 1024:.1f} KiB")

    # Agreement with the CPU matcher, which is the reference implementation.
    # Checked at every step of a real document rather than only at the start,
    # since the start state exercises none of the reduce path.
    matcher = compiled.matcher(0)
    probe = grammar.new_batch(1)
    reference = torch.zeros(grammar.mask_words, dtype=torch.int32)
    checked = 0
    for token in tokenizer.encode(
        instances[arguments.schema_index]["text"], add_special_tokens=False
    ):
        reference.zero_()
        matcher.fill_bitmask(reference)
        configurations = matcher.configurations()
        probe.set_configurations(0, configurations)
        device = probe.fill_mask()[0].cpu()
        if not torch.equal(device, reference):
            differing = int((device != reference).sum())
            raise SystemExit(
                f"device and CPU masks differ in {differing} words at step {checked}"
            )
        checked += 1
        if not matcher.accept_token(token):
            break
    print(f"  agrees with the CPU matcher at every one of {checked} steps")

    # Which states does a real document reach? Everything in the start state is
    # a worst case, not a workload.
    visited = [0]
    if arguments.states == "visited":
        matcher = compiled.matcher(0)
        for token in tokenizer.encode(
            instances[arguments.schema_index]["text"], add_special_tokens=False
        ):
            visited.append(matcher.lexer_state)
            if not matcher.accept_token(token):
                break
        groups_here = [
            int(grammar.group_offsets[state + 1] - grammar.group_offsets[state])
            for state in visited
        ]
        print(
            f"  a real document visits {len(set(visited))} states, "
            f"median {int(np.median(groups_here))} groups each"
        )

    results = []
    generator = np.random.default_rng(0)
    for size in arguments.batches:
        batch = grammar.new_batch(size)
        # One configuration each, in a state the document actually reached.
        chosen = generator.choice(np.array(visited, dtype=np.int32), size=size)
        batch.lexer_state.view(size, batch.configs)[:, 0] = torch.from_numpy(
            chosen
        ).cuda()
        microseconds = _time(batch.fill_mask)
        print(f"  batch {size:>4}: {microseconds:8.1f} us")
        results.append({"batch_size": size, "device_fill_us": microseconds})

    if arguments.output:
        arguments.output.write_text(
            json.dumps(
                {
                    "schema_index": arguments.schema_index,
                    "lexer_states": compiled.num_lexer_states,
                    "groups": compiled.num_groups,
                    "resident_bytes": grammar.resident_bytes(),
                    "measurements": results,
                },
                indent=2,
            )
        )
        print(f"wrote {arguments.output}")


if __name__ == "__main__":
    main()
