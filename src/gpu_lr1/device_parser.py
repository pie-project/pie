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
import bisect
import json
from pathlib import Path

import numpy as np
import torch
import triton
import triton.language as tl

_GROUP_BLOCK = 64
# Blocks that drain the sweep. Fixed, and deliberately so: the grid no longer
# depends on the batch, on how wide its parses are, or on the grammar, which is
# what a CUDA graph needs and what a batch of mixed grammars would need. It also
# bounds the replay scratch, which is now one window per block rather than one
# per (sequence, configuration, group).
_SWEEP_BLOCKS = 2048
# Sentinel for "no group of this configuration holds the sampled token". Above
# any group index, so an atomic minimum picks the earliest real finder.
_NO_GROUP = 2**31 - 1
ACCEPT = -(2**31)
SPARSE, COMPLEMENT, DENSE = 0, 1, 2

# Where each grammar's run of an array starts, inside the shared arena. One
# index per array whose elements a kernel addresses; arrays a grammar indexes
# together share a base.
_B_GROUP_OFFSETS = tl.constexpr(0)
_B_GROUPS = tl.constexpr(1)
_B_SET_PAYLOAD = tl.constexpr(2)
_B_READING_OFFSETS = tl.constexpr(3)
_B_READING_INDEX = tl.constexpr(4)
_B_READINGS = tl.constexpr(5)
_B_READING_TERM_OFFSETS = tl.constexpr(6)
_B_READING_TERMINALS = tl.constexpr(7)
_B_ACTION_OFFSETS = tl.constexpr(8)
_B_ACTIONS = tl.constexpr(9)
_B_GOTO_OFFSETS = tl.constexpr(10)
_B_GOTOS = tl.constexpr(11)
_B_PRODUCTIONS = tl.constexpr(12)
_B_PENDING_OFFSETS = tl.constexpr(13)
_B_PENDING_TERMINALS = tl.constexpr(14)
_NBASES = tl.constexpr(15)

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
def _peek(stack_ptr, base, window_ptr, floor, index):
    """Read stack entry `index` through a window laid over a shared prefix.

    A replay never writes below where its own pushes started, so the sequence's
    stack can stay where it is and be read in place; only what the replay adds
    needs to be private. The window holds `[floor, top)` and the entries below
    `floor` are still the sequence's own.

    Both addresses are in bounds whichever answer is wanted, so this selects
    rather than branches. The load it would have skipped is in L1, and a branch
    costs every lane in the block.
    """
    inside = index >= floor
    held = tl.load(window_ptr + tl.maximum(index - floor, 0))
    shared = tl.load(stack_ptr + base + index)
    return tl.where(inside, held, shared)


@triton.jit
def _replay_group(
    group_offsets_ptr,
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
    stack_ptr,
    scratch_ptr,
    overflow_ptr,
    sequence,
    base,
    depth,
    group,
    scratch,
    probe,
    STACK_STRIDE: tl.constexpr,
    MAX_REDUCTIONS: tl.constexpr,
    WINDOW: tl.constexpr,
):
    """Does the parser survive any reading of this group's tokens?

    A group asks one question and answers it from the stack alone, so this is
    the unit of work the sweep hands out, and the only state it needs is where
    the sequence's stack is and where its own window may go.
    """
    admitted = 0
    high_water = 0
    use = tl.load(reading_offsets_ptr + group)
    use_end = tl.load(reading_offsets_ptr + group + 1)
    while use < use_end:
        reading = tl.load(reading_index_ptr + use)
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

        # Nothing is copied. `floor` is where this replay's own writes begin,
        # so the window is empty until it pushes, and a reading that dies on
        # its first terminal - which most do - touches no scratch at all.
        copy_depth = depth
        floor = depth
        high_water = tl.maximum(high_water, 0)
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
                            if copy_depth >= STACK_STRIDE or copy_depth - floor >= WINDOW:
                                alive = 0
                                tl.store(overflow_ptr + sequence, 1)
                            else:
                                tl.store(
                                    scratch_ptr + scratch + copy_depth - floor,
                                    value - 1,
                                )
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
                                # Popping past where this replay started puts the
                                # top back in the sequence's own stack, and the
                                # window's contents are all above it and dead, so
                                # the window empties rather than being rewritten.
                                floor = tl.minimum(floor, copy_depth)
                                exposed = _peek(
                                    stack_ptr,
                                    base,
                                    scratch_ptr + scratch,
                                    floor,
                                    copy_depth - 1,
                                )
                                lhs = tl.load(production_lhs_ptr + production)
                                grow = tl.load(goto_offsets_ptr + exposed)
                                grow_end = tl.load(goto_offsets_ptr + exposed + 1)
                                target = _search(
                                    goto_nonterminals_ptr, grow, grow_end, lhs
                                )
                                if target < 0:
                                    alive = 0
                                elif (
                                    copy_depth >= STACK_STRIDE
                                    or copy_depth - floor >= WINDOW
                                ):
                                    alive = 0
                                    tl.store(overflow_ptr + sequence, 1)
                                else:
                                    top = tl.load(goto_targets_ptr + target)
                                    tl.store(
                                        scratch_ptr + scratch + copy_depth - floor, top
                                    )
                                    copy_depth = copy_depth + 1
            if settled == 0:
                alive = 0
            high_water = tl.maximum(high_water, copy_depth - floor)
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
                    # The probe reduces, which rewrites the stack, and it must
                    # not disturb the replay it starts from - so it gets its own
                    # window over the replay's window over the sequence's stack.
                    # Three levels, no copy at any of them.
                    probe_depth = copy_depth
                    probe_floor = copy_depth
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
                                        probe_floor = tl.minimum(
                                            probe_floor, probe_depth
                                        )
                                        under = _peek(
                                            stack_ptr,
                                            base,
                                            scratch_ptr + scratch,
                                            floor,
                                            tl.minimum(probe_depth - 1, copy_depth - 1),
                                        )
                                        held = tl.load(
                                            scratch_ptr
                                            + probe
                                            + tl.maximum(
                                                probe_depth - 1 - probe_floor, 0
                                            )
                                        )
                                        exposed = tl.where(
                                            probe_depth - 1 >= probe_floor, held, under
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
                                        elif (
                                            probe_depth >= STACK_STRIDE
                                            or probe_depth - probe_floor >= WINDOW
                                        ):
                                            probe_alive = 0
                                            tl.store(overflow_ptr + sequence, 1)
                                        else:
                                            probe_top = tl.load(
                                                goto_targets_ptr + target
                                            )
                                            tl.store(
                                                scratch_ptr
                                                + probe
                                                + probe_depth
                                                - probe_floor,
                                                probe_top,
                                            )
                                            probe_depth = probe_depth + 1
                            high_water = tl.maximum(
                                high_water, probe_depth - tl.minimum(probe_floor, floor)
                            )
                    if probe_alive == 1 and probe_settled == 1:
                        any_ok = 1
                    pend = pend + 1
                alive = any_ok

        if alive == 1:
            admitted = 1
            use = use_end
        else:
            use = use + 1
    return admitted, high_water


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
    work_offsets_ptr,
    grammar_ptr,
    bases_ptr,
    scratch_ptr,
    admitted_ptr,
    overflow_ptr,
    high_water_ptr,
    ROWS: tl.constexpr,
    CONFIGS: tl.constexpr,
    STACK_STRIDE: tl.constexpr,
    MAX_REDUCTIONS: tl.constexpr,
    WINDOW: tl.constexpr,
):
    """A fixed number of blocks draining a list of (configuration, group).

    The grid used to be one program per (sequence, configuration, group), sized
    for the ceilings rather than for the work: the configuration ceiling, and
    the largest number of groups any lexer state has - 411 on one schema and
    1,673 on another, against a median state's eleven. At batch 512 that is
    841,000 programs of which 93% to 95% exit at once, because a serving batch
    holds only a few dozen distinct parse states, and every one of them still
    needed its own private window, which is what made the scratch 1.7 GB.

    Here the live configurations are counted first and their groups laid end to
    end, so a block takes item `i`, then `i + blocks`, and so on. Three things
    follow. The scratch is one window per *block* rather than per program.
    The grid no longer depends on the batch, its width, or the grammar - which
    is what a CUDA graph needs, since a graph is a fixed sequence of launches
    and a serving batch changes composition every step. And the ceilings stop
    being paid for: what is enumerated is the work that exists.
    """
    block = tl.program_id(0)
    blocks = tl.num_programs(0)
    total = tl.load(work_offsets_ptr + ROWS)
    scratch = block * 2 * WINDOW
    probe = scratch + WINDOW
    high_water = 0

    item = block
    while item < total:
        # Which configuration owns this item. Rows that contribute nothing have
        # equal offsets, so taking the *last* row whose offset is at or below
        # the item lands past them and on the row that holds it.
        low = 0
        high = ROWS - 1
        while low < high:
            middle = (low + high + 1) // 2
            if tl.load(work_offsets_ptr + middle) <= item:
                low = middle
            else:
                high = middle - 1
        row_index = low
        slot = item - tl.load(work_offsets_ptr + row_index)
        sequence = row_index // CONFIGS
        state = tl.load(lexer_state_ptr + row_index)
        depth = tl.load(stack_depth_ptr + row_index)
        # The sequence's grammar decides where in the arena every table starts.
        # Rebasing the pointers rather than every index means the replay below
        # is the same code it was when there was only one grammar.
        at = bases_ptr + tl.load(grammar_ptr + sequence) * _NBASES
        my_group_offsets = group_offsets_ptr + tl.load(at + _B_GROUP_OFFSETS)
        group = tl.load(my_group_offsets + state) + slot

        admitted, reach = _replay_group(
            my_group_offsets,
            reading_offsets_ptr + tl.load(at + _B_READING_OFFSETS),
            reading_index_ptr + tl.load(at + _B_READING_INDEX),
            reading_next_state_ptr + tl.load(at + _B_READINGS),
            reading_term_offsets_ptr + tl.load(at + _B_READING_TERM_OFFSETS),
            reading_terminals_ptr + tl.load(at + _B_READING_TERMINALS),
            action_offsets_ptr + tl.load(at + _B_ACTION_OFFSETS),
            action_terminals_ptr + tl.load(at + _B_ACTIONS),
            action_values_ptr + tl.load(at + _B_ACTIONS),
            goto_offsets_ptr + tl.load(at + _B_GOTO_OFFSETS),
            goto_nonterminals_ptr + tl.load(at + _B_GOTOS),
            goto_targets_ptr + tl.load(at + _B_GOTOS),
            production_lhs_ptr + tl.load(at + _B_PRODUCTIONS),
            production_arity_ptr + tl.load(at + _B_PRODUCTIONS),
            pending_offsets_ptr + tl.load(at + _B_PENDING_OFFSETS),
            pending_terminals_ptr + tl.load(at + _B_PENDING_TERMINALS),
            stack_ptr,
            scratch_ptr,
            overflow_ptr,
            sequence,
            row_index * STACK_STRIDE,
            depth,
            group,
            scratch,
            probe,
            STACK_STRIDE=STACK_STRIDE,
            MAX_REDUCTIONS=MAX_REDUCTIONS,
            WINDOW=WINDOW,
        )
        # Written whether or not the group is admitted, so the buffer never has
        # to be cleared - at batch 512 that clear was 13 MB a step.
        tl.store(admitted_ptr + item, admitted.to(tl.int8))
        high_water = tl.maximum(high_water, reach)
        item = item + blocks

    # How much of the window this block actually needed. Recorded once, not per
    # push: the point is to know how loose the compile-time bound is, and an
    # atomic in the reduce loop would be measuring the measurement.
    tl.atomic_max(high_water_ptr, high_water)


@triton.jit
def _count_kernel(
    group_offsets_ptr,
    lexer_state_ptr,
    config_count_ptr,
    widest_ptr,
    representative_ptr,
    grammar_ptr,
    bases_ptr,
    counts_ptr,
    CONFIGS: tl.constexpr,
    ROWS: tl.constexpr,
    SKIP_DUPLICATES: tl.constexpr,
    UNIT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """How many groups each configuration puts into the sweep.

    Zero for a configuration past the width of its sequence, past the batch's
    width, or belonging to a sequence whose parse state an earlier one already
    holds - which in a serving batch is 93% to 95% of them. Everything the old
    grid spent on those exits is simply not enumerated here, and the running
    sum of this is what turns an item back into a configuration and a group.
    """
    lane = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    live = lane < ROWS
    sequence = lane // CONFIGS
    config = lane % CONFIGS
    keep = live
    keep = keep & (config < tl.load(widest_ptr))
    keep = keep & (config < tl.load(config_count_ptr + sequence, mask=live, other=0))
    if SKIP_DUPLICATES == 1:
        keep = keep & (
            tl.load(representative_ptr + sequence, mask=live, other=-1) == sequence
        )
    grammar = tl.load(grammar_ptr + sequence, mask=live, other=0)
    at = tl.load(bases_ptr + grammar * _NBASES + _B_GROUP_OFFSETS, mask=keep, other=0)
    state = tl.load(lexer_state_ptr + lane, mask=keep, other=0)
    if UNIT == 1:
        # The advance wants one entry per live configuration, not one per group:
        # it searches a whole state for the token at a time.
        tl.store(counts_ptr + lane, tl.where(keep, 1, 0), mask=live)
    else:
        first = tl.load(group_offsets_ptr + at + state, mask=keep, other=0)
        last = tl.load(group_offsets_ptr + at + state + 1, mask=keep, other=0)
        tl.store(counts_ptr + lane, tl.where(keep, last - first, 0), mask=live)


@triton.jit
def _snapshot_kernel(
    lexer_state_ptr,
    stack_ptr,
    config_count_ptr,
    live_offsets_ptr,
    old_lexer_ptr,
    old_count_ptr,
    old_stack_ptr,
    ROWS: tl.constexpr,
    CONFIGS: tl.constexpr,
    STACK_STRIDE: tl.constexpr,
):
    """Keep what the commit will overwrite, for the configurations in play.

    A candidate names the stack it came from rather than carrying a copy, and
    the commit builds the next set in place, so the sources have to survive.
    Copying the whole buffer was 8.4 MB and 13 us a step to preserve the one or
    two configurations a sequence actually holds.
    """
    program = tl.program_id(0)
    programs = tl.num_programs(0)
    total = tl.load(live_offsets_ptr + ROWS)
    slot = program
    while slot < total:
        low = 0
        high = ROWS - 1
        while low < high:
            middle = (low + high + 1) // 2
            if tl.load(live_offsets_ptr + middle) <= slot:
                low = middle
            else:
                high = middle - 1
        row_index = low
        tl.store(old_lexer_ptr + row_index, tl.load(lexer_state_ptr + row_index))
        sequence = row_index // CONFIGS
        tl.store(old_count_ptr + sequence, tl.load(config_count_ptr + sequence))
        lane = tl.arange(0, STACK_STRIDE)
        tl.store(
            old_stack_ptr + row_index * STACK_STRIDE + lane,
            tl.load(stack_ptr + row_index * STACK_STRIDE + lane),
        )
        slot = slot + programs


@triton.jit
def _scatter_kernel(
    group_offsets_ptr,
    group_set_kind_ptr,
    group_set_offset_ptr,
    group_set_length_ptr,
    set_payload_ptr,
    lexer_state_ptr,
    work_offsets_ptr,
    grammar_ptr,
    bases_ptr,
    admitted_ptr,
    mask_ptr,
    mask_words,
    ROWS: tl.constexpr,
    CONFIGS: tl.constexpr,
    COMPLEMENTS_ONLY: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Write the sets of the admitted groups. Launched twice.

    A complement sets the whole mask row and then punches its exclusions out,
    and the groups of a lexer state are disjoint, so every other admitted
    group's tokens are among those exclusions - writing a complement after them
    would erase them. Atomics give no ordering between programs, so this has to
    be the ordering of two launches, which is the only ordering CUDA offers
    without a barrier.

    Drains the same item list as the sweep, so an item means the same
    (configuration, group) in both and admission can be recorded against it.
    """
    block = tl.program_id(0)
    blocks = tl.num_programs(0)
    total = tl.load(work_offsets_ptr + ROWS)
    item = block
    while item < total:
        if tl.load(admitted_ptr + item) != 0:
            low = 0
            high = ROWS - 1
            while low < high:
                middle = (low + high + 1) // 2
                if tl.load(work_offsets_ptr + middle) <= item:
                    low = middle
                else:
                    high = middle - 1
            row_index = low
            sequence = row_index // CONFIGS
            state = tl.load(lexer_state_ptr + row_index)
            at = bases_ptr + tl.load(grammar_ptr + sequence) * _NBASES
            groups = tl.load(at + _B_GROUPS)
            payload = set_payload_ptr + tl.load(at + _B_SET_PAYLOAD)
            group = (
                tl.load(group_offsets_ptr + tl.load(at + _B_GROUP_OFFSETS) + state)
                + item
                - tl.load(work_offsets_ptr + row_index)
            )
            kind = tl.load(group_set_kind_ptr + groups + group)
            wanted = kind == _COMPLEMENT
            if COMPLEMENTS_ONLY == 0:
                wanted = kind != _COMPLEMENT
            if wanted:
                offset = tl.load(group_set_offset_ptr + groups + group)
                length = tl.load(group_set_length_ptr + groups + group)
                row = mask_ptr + sequence * mask_words
                if kind == _COMPLEMENT:
                    for start in range(0, mask_words, BLOCK):
                        lane = start + tl.arange(0, BLOCK)
                        live = lane < mask_words
                        tl.atomic_or(
                            row + lane, tl.full((BLOCK,), -1, tl.int32), mask=live
                        )
                    for start in range(0, length, BLOCK):
                        lane = start + tl.arange(0, BLOCK)
                        live = lane < length
                        token = tl.load(payload + offset + lane, mask=live, other=0)
                        tl.atomic_and(
                            row + token // 32,
                            (~(1 << (token % 32))).to(tl.int32),
                            mask=live,
                        )
                elif kind == _SPARSE:
                    for start in range(0, length, BLOCK):
                        lane = start + tl.arange(0, BLOCK)
                        live = lane < length
                        token = tl.load(payload + offset + lane, mask=live, other=0)
                        tl.atomic_or(
                            row + token // 32,
                            (1 << (token % 32)).to(tl.int32),
                            mask=live,
                        )
                else:
                    for start in range(0, mask_words, BLOCK):
                        lane = start + tl.arange(0, BLOCK)
                        live = lane < mask_words
                        value = tl.load(payload + offset + lane, mask=live, other=0)
                        tl.atomic_or(row + lane, value, mask=live)
        item = item + blocks


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
def _locate_kernel(
    group_offsets_ptr,
    group_set_kind_ptr,
    group_set_offset_ptr,
    group_set_length_ptr,
    set_payload_ptr,
    lexer_state_ptr,
    config_count_ptr,
    widest_ptr,
    token_ptr,
    grammar_ptr,
    bases_ptr,
    live_offsets_ptr,
    found_ptr,
    ROWS: tl.constexpr,
    CONFIGS: tl.constexpr,
    GROUP_BLOCK: tl.constexpr,
    SEARCH_STEPS: tl.constexpr,
):
    """Which group holds the sampled token, for each live configuration.

    Split out from the replay because at most one group of a lexer state holds
    any given token - the groups are disjoint - so of the blocks that search,
    all but one find nothing. Leaving the replay in the same kernel meant every
    searching block had to be given a private window in case it was the one, and
    at batch 512 that was 117 MB of scratch to serve a few hundred replays.

    It also settles which group wins. Within a block the smallest index was
    taken; across blocks two finders would both have gone on, and the reference
    matcher takes the earliest. An atomic minimum makes that the rule.
    """
    program = tl.program_id(0)
    programs = tl.num_programs(0)
    total = tl.load(live_offsets_ptr + ROWS)
    slot = program
    while slot < total:
        # Which configuration this slot is. Launching for the ceiling instead
        # was 38 us of a 163 us step at batch 512, nearly all of it blocks that
        # returned at once - the same mistake the fill's grid used to make.
        low = 0
        high = ROWS - 1
        while low < high:
            middle = (low + high + 1) // 2
            if tl.load(live_offsets_ptr + middle) <= slot:
                low = middle
            else:
                high = middle - 1
        row_index = low
        sequence = row_index // CONFIGS
        _locate_one(
            group_offsets_ptr,
            group_set_kind_ptr,
            group_set_offset_ptr,
            group_set_length_ptr,
            set_payload_ptr,
            lexer_state_ptr,
            token_ptr,
            grammar_ptr,
            bases_ptr,
            found_ptr,
            sequence,
            row_index,
            GROUP_BLOCK=GROUP_BLOCK,
            SEARCH_STEPS=SEARCH_STEPS,
        )
        slot = slot + programs


@triton.jit
def _locate_one(
    group_offsets_ptr,
    group_set_kind_ptr,
    group_set_offset_ptr,
    group_set_length_ptr,
    set_payload_ptr,
    lexer_state_ptr,
    token_ptr,
    grammar_ptr,
    bases_ptr,
    found_ptr,
    sequence,
    row_index,
    GROUP_BLOCK: tl.constexpr,
    SEARCH_STEPS: tl.constexpr,
):
    """Search one configuration's groups for the sampled token."""
    state = tl.load(lexer_state_ptr + row_index)
    at = bases_ptr + tl.load(grammar_ptr + sequence) * _NBASES
    groups = tl.load(at + _B_GROUPS)
    payload = set_payload_ptr + tl.load(at + _B_SET_PAYLOAD)
    my_group_offsets = group_offsets_ptr + tl.load(at + _B_GROUP_OFFSETS)
    first = tl.load(my_group_offsets + state)
    last = tl.load(my_group_offsets + state + 1)

    # Find the group holding the token a block of groups at a time rather than
    # one per program. A state can have hundreds of groups and only one of them
    # holds any given token, so the search is nearly all rejection - and doing
    # it one program per group turns three contiguous arrays into three
    # scattered loads per group. Read as a block they are three coalesced
    # loads for the whole block, which is the difference this kernel is made
    # of: it is bound by how many scattered loads it issues, not by arithmetic.
    token = tl.load(token_ptr + sequence)
    glane = tl.arange(0, GROUP_BLOCK)
    start = first
    while start < last:
        group = start + glane
        live_lane = group < last
        kind = tl.load(group_set_kind_ptr + groups + group, mask=live_lane, other=0)
        offset = tl.load(group_set_offset_ptr + groups + group, mask=live_lane, other=0)
        length = tl.load(group_set_length_ptr + groups + group, mask=live_lane, other=1)

        dense = kind == _DENSE
        word = tl.load(payload + offset + token // 32, mask=live_lane & dense, other=0)
        in_dense = ((word >> (token % 32)) & 1) == 1

        # A sorted list's ends are its bounds, so most lanes are decided without a
        # search at all.
        listed = live_lane & (dense == 0)
        low = tl.load(payload + offset, mask=listed, other=1)
        high = tl.load(payload + offset + length - 1, mask=listed, other=0)
        searching = listed & (token >= low) & (token <= high)
        # A plain halving. A lane that has found its answer stops being active
        # rather than having its bounds collapsed: writing `hi = lo` on a hit made
        # `hi` depend on an already-updated `lo` within the same step, and a lane
        # could then search backwards. Lanes that finish early idle, which costs
        # the block nothing it was not already paying.
        lo = offset
        hi = offset + length
        hit = tl.zeros((GROUP_BLOCK,), tl.int32) - 1
        for _ in range(0, SEARCH_STEPS):
            active = searching & (lo < hi) & (hit < 0)
            middle = (lo + hi) // 2
            value = tl.load(payload + middle, mask=active, other=0)
            hit = tl.where(active & (value == token), middle, hit)
            lo = tl.where(active & (value < token), middle + 1, lo)
            hi = tl.where(active & (value > token), middle, hi)
        found = hit >= 0
        complement = kind == _COMPLEMENT
        inside = tl.where(
            dense, in_dense, tl.where(complement, found == 0, found)
        ) & live_lane

        if tl.sum(inside.to(tl.int32)) != 0:
            tl.atomic_min(
                found_ptr + row_index, tl.min(tl.where(inside, group, last))
            )
        start = start + GROUP_BLOCK


@triton.jit
def _candidate_kernel(
    group_offsets_ptr,
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
    found_ptr,
    grammar_ptr,
    bases_ptr,
    scratch_ptr,
    cand_valid_ptr,
    cand_lexer_ptr,
    cand_depth_ptr,
    cand_floor_ptr,
    cand_window_ptr,
    overflow_ptr,
    ROWS: tl.constexpr,
    CONFIGS: tl.constexpr,
    MAX_READINGS: tl.constexpr,
    STACK_STRIDE: tl.constexpr,
    MAX_REDUCTIONS: tl.constexpr,
    WINDOW: tl.constexpr,
    NO_GROUP: tl.constexpr,
):
    """Where each configuration lands if the sampled token is accepted.

    The same replay the mask sweep runs, with two differences. It starts from
    the one group that holds the sampled token - which `_locate_kernel` has
    already found - rather than sweeping all of them, and it keeps going after
    the first reading that survives: the mask only needed to know *whether* a
    token was admissible, while an advance needs every state the token could
    have led to, since each is a distinct parse that stays alive.

    A fixed number of blocks drains the configurations, so the scratch is one
    window per block and the grid does not depend on the batch or on how wide
    its parses are - the same reason the mask sweep is shaped this way.

    The token arrives as a device pointer and is never read on the host. That
    is the whole point - a value the host has to see is a synchronisation, and
    a decode loop that synchronises once per token is the thing this design
    exists to remove.
    """
    block = tl.program_id(0)
    blocks = tl.num_programs(0)
    scratch = block * 2 * WINDOW
    probe = scratch + WINDOW

    row_index = block
    while row_index < ROWS:
        group = tl.load(found_ptr + row_index)
        if group < NO_GROUP:
            sequence = row_index // CONFIGS
            depth = tl.load(stack_depth_ptr + row_index)
            base = row_index * STACK_STRIDE
            out_base = row_index * MAX_READINGS
            at = bases_ptr + tl.load(grammar_ptr + sequence) * _NBASES
            reading_offsets = reading_offsets_ptr + tl.load(at + _B_READING_OFFSETS)
            reading_index = reading_index_ptr + tl.load(at + _B_READING_INDEX)
            reading_next_state = reading_next_state_ptr + tl.load(at + _B_READINGS)
            reading_term_offsets = reading_term_offsets_ptr + tl.load(
                at + _B_READING_TERM_OFFSETS
            )
            reading_terminals = reading_terminals_ptr + tl.load(
                at + _B_READING_TERMINALS
            )
            action_offsets = action_offsets_ptr + tl.load(at + _B_ACTION_OFFSETS)
            actions = tl.load(at + _B_ACTIONS)
            action_terminals = action_terminals_ptr + actions
            action_values = action_values_ptr + actions
            goto_offsets = goto_offsets_ptr + tl.load(at + _B_GOTO_OFFSETS)
            gotos = tl.load(at + _B_GOTOS)
            goto_nonterminals = goto_nonterminals_ptr + gotos
            goto_targets = goto_targets_ptr + gotos
            productions = tl.load(at + _B_PRODUCTIONS)
            production_lhs = production_lhs_ptr + productions
            production_arity = production_arity_ptr + productions
            pending_offsets = pending_offsets_ptr + tl.load(at + _B_PENDING_OFFSETS)
            pending_terminals = pending_terminals_ptr + tl.load(
                at + _B_PENDING_TERMINALS
            )
            use = tl.load(reading_offsets + group)
            use_end = tl.load(reading_offsets + group + 1)
            index = 0
            while use < use_end and index < MAX_READINGS:
                reading = tl.load(reading_index + use)
                term = tl.load(reading_term_offsets + reading)
                term_end = tl.load(reading_term_offsets + reading + 1)
                top = tl.load(stack_ptr + base + depth - 1)
                alive = 1
                copy_depth = depth
                floor = depth
                while term < term_end and alive == 1:
                    terminal = tl.load(reading_terminals + term)
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
                            row = tl.load(action_offsets + top)
                            row_end = tl.load(action_offsets + top + 1)
                            entry = _search(action_terminals, row, row_end, terminal)
                            if entry < 0:
                                alive = 0
                            else:
                                value = tl.load(action_values + entry)
                                if value == _ACCEPT:
                                    alive = 0
                                elif value > 0:
                                    if copy_depth >= STACK_STRIDE or copy_depth - floor >= WINDOW:
                                        alive = 0
                                        tl.store(overflow_ptr + sequence, 1)
                                    else:
                                        tl.store(
                                            scratch_ptr + scratch + copy_depth - floor,
                                            value - 1,
                                        )
                                        copy_depth = copy_depth + 1
                                        top = value - 1
                                        settled = 1
                                else:
                                    production = -value - 1
                                    arity = tl.load(production_arity + production)
                                    if copy_depth <= arity:
                                        alive = 0
                                    else:
                                        copy_depth = copy_depth - arity
                                        floor = tl.minimum(floor, copy_depth)
                                        exposed = _peek(
                                            stack_ptr,
                                            base,
                                            scratch_ptr + scratch,
                                            floor,
                                            copy_depth - 1,
                                        )
                                        lhs = tl.load(production_lhs + production)
                                        grow = tl.load(goto_offsets + exposed)
                                        grow_end = tl.load(goto_offsets + exposed + 1)
                                        target = _search(
                                            goto_nonterminals, grow, grow_end, lhs
                                        )
                                        if target < 0:
                                            alive = 0
                                        elif (
                                            copy_depth >= STACK_STRIDE
                                            or copy_depth - floor >= WINDOW
                                        ):
                                            alive = 0
                                            tl.store(overflow_ptr + sequence, 1)
                                        else:
                                            top = tl.load(goto_targets + target)
                                            tl.store(
                                                scratch_ptr + scratch + copy_depth - floor, top
                                            )
                                            copy_depth = copy_depth + 1
                    if settled == 0:
                        alive = 0
                    term = term + 1

                next_state = tl.load(reading_next_state + reading)
                if alive == 1:
                    pend = tl.load(pending_offsets + next_state)
                    pend_end = tl.load(pending_offsets + next_state + 1)
                    if pend < pend_end:
                        any_ok = 0
                        while pend < pend_end and any_ok == 0:
                            terminal = tl.load(pending_terminals + pend)
                            probe_depth = copy_depth
                            probe_floor = copy_depth
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
                                    row = tl.load(action_offsets + probe_top)
                                    row_end = tl.load(action_offsets + probe_top + 1)
                                    entry = _search(action_terminals, row, row_end, terminal)
                                    if entry < 0:
                                        probe_alive = 0
                                    else:
                                        value = tl.load(action_values + entry)
                                        if value == _ACCEPT:
                                            probe_settled = 1
                                        elif value > 0:
                                            probe_settled = 1
                                        else:
                                            production = -value - 1
                                            arity = tl.load(production_arity + production)
                                            if probe_depth <= arity:
                                                probe_alive = 0
                                            else:
                                                probe_depth = probe_depth - arity
                                                probe_floor = tl.minimum(
                                                    probe_floor, probe_depth
                                                )
                                                under = _peek(
                                                    stack_ptr,
                                                    base,
                                                    scratch_ptr + scratch,
                                                    floor,
                                                    tl.minimum(probe_depth - 1, copy_depth - 1),
                                                )
                                                held = tl.load(
                                                    scratch_ptr
                                                    + probe
                                                    + tl.maximum(
                                                        probe_depth - 1 - probe_floor, 0
                                                    )
                                                )
                                                exposed = tl.where(
                                                    probe_depth - 1 >= probe_floor, held, under
                                                )
                                                lhs = tl.load(production_lhs + production)
                                                grow = tl.load(goto_offsets + exposed)
                                                grow_end = tl.load(goto_offsets + exposed + 1)
                                                target = _search(
                                                    goto_nonterminals, grow, grow_end, lhs
                                                )
                                                if target < 0:
                                                    probe_alive = 0
                                                elif (
                                                    probe_depth >= STACK_STRIDE
                                                    or probe_depth - probe_floor >= WINDOW
                                                ):
                                                    probe_alive = 0
                                                    tl.store(overflow_ptr + sequence, 1)
                                                else:
                                                    probe_top = tl.load(goto_targets + target)
                                                    tl.store(
                                                        scratch_ptr
                                                        + probe
                                                        + probe_depth
                                                        - probe_floor,
                                                        probe_top,
                                                    )
                                                    probe_depth = probe_depth + 1
                            if probe_alive == 1 and probe_settled == 1:
                                any_ok = 1
                            pend = pend + 1
                        alive = any_ok

                if alive == 1:
                    # A candidate outlives the step that made it, so unlike a
                    # replay it does have to be written down - but not as a
                    # whole stack. It shares everything below its floor with the
                    # configuration it came from, and the commit can read that
                    # there, so what is stored is the floor and the window.
                    # Whole stacks made this 151 MB at batch 512, four fifths of
                    # everything a batch allocated.
                    tl.store(cand_valid_ptr + out_base + index, 1)
                    tl.store(cand_lexer_ptr + out_base + index, next_state)
                    tl.store(cand_depth_ptr + out_base + index, copy_depth)
                    tl.store(cand_floor_ptr + out_base + index, floor)
                    top_lane = tl.arange(0, WINDOW)
                    tl.store(
                        cand_window_ptr + (out_base + index) * WINDOW + top_lane,
                        tl.load(
                            scratch_ptr + scratch + top_lane,
                            mask=top_lane < copy_depth - floor,
                            other=0,
                        ),
                        mask=top_lane < copy_depth - floor,
                    )
                    index = index + 1
                use = use + 1
        row_index = row_index + blocks


@triton.jit
def _commit_kernel(
    lexer_state_ptr,
    stack_ptr,
    stack_depth_ptr,
    config_count_ptr,
    old_lexer_ptr,
    old_count_ptr,
    old_stack_ptr,
    cand_valid_ptr,
    cand_lexer_ptr,
    cand_depth_ptr,
    cand_floor_ptr,
    cand_window_ptr,
    terminated_ptr,
    overflow_ptr,
    widest_ptr,
    CONFIGS: tl.constexpr,
    MAX_READINGS: tl.constexpr,
    STACK_STRIDE: tl.constexpr,
    WINDOW: tl.constexpr,
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
    # Set when a surviving candidate has to be dropped for want of room. The
    # configuration ceiling is a policy, not a property of the grammar, and a
    # parse that outgrows it keeps a prefix of its states - which narrows the
    # mask. Narrowing is the failure this engine must never do quietly, so it
    # is reported through the same flag as a window that overran.
    saturated = 0

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
                                if tl.load(cand_valid_ptr + base + index) == 1:
                                    if written >= CONFIGS:
                                        saturated = 1
                                if written < CONFIGS:
                                    if tl.load(cand_valid_ptr + base + index) == 1:
                                        next_state = tl.load(cand_lexer_ptr + base + index)
                                        depth = tl.load(cand_depth_ptr + base + index)
                                        # The candidate's stack, put back
                                        # together: everything below its floor
                                        # is the source configuration's, which
                                        # is read from the copy taken before
                                        # this kernel began overwriting it.
                                        floor = tl.load(cand_floor_ptr + base + index)
                                        values = tl.where(
                                            lane < floor,
                                            tl.load(
                                                old_stack_ptr
                                                + (sequence * CONFIGS + source)
                                                * STACK_STRIDE
                                                + lane,
                                                mask=lane < floor,
                                                other=0,
                                            ),
                                            tl.load(
                                                cand_window_ptr
                                                + (base + index) * WINDOW
                                                + tl.maximum(lane - floor, 0),
                                                mask=(lane >= floor) & (lane < depth),
                                                other=0,
                                            ),
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
    if saturated == 1:
        tl.store(overflow_ptr + sequence, 1)
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
    grammar_ptr,
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
    # The grammar is part of the state. Two sequences under different schemas
    # can sit at the same parser state with the same stack and still admit
    # different tokens, so sharing a mask between them would be wrong.
    digest = (digest ^ tl.load(grammar_ptr + sequence)) * 16777619
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
    grammar_ptr,
    hash_ptr,
    representative_ptr,
    mask_ptr,
    mask_words,
    BATCH: tl.constexpr,
    CONFIGS: tl.constexpr,
    STACK_STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """The lowest-numbered sequence holding the same parse state.

    The fingerprint narrows the search; this confirms it exactly, because two
    different parse states that happen to hash alike would otherwise share a
    mask and one of them would be wrong.
    """
    sequence = tl.program_id(0)
    mine = tl.load(hash_ptr + sequence)
    count = tl.load(config_count_ptr + sequence)
    grammar = tl.load(grammar_ptr + sequence)
    found = sequence

    # Scanned a block of candidates at a time rather than one at a time. Every
    # sequence walks every earlier one, so this is quadratic in the batch and it
    # showed: 36 us of a 182 us step at batch 512, second only to the replay.
    # The scan itself is three loads a candidate and nothing else, which is
    # exactly the shape a block does for free.
    lane = tl.arange(0, BLOCK)
    start = 0
    while start < sequence and found == sequence:
        index = start + lane
        live = index < sequence
        alike = live
        alike = alike & (tl.load(hash_ptr + index, mask=live, other=0) == mine)
        alike = alike & (
            tl.load(config_count_ptr + index, mask=live, other=-1) == count
        )
        alike = alike & (tl.load(grammar_ptr + index, mask=live, other=-1) == grammar)
        other = tl.min(tl.where(alike, index, BATCH))
        if other >= BATCH:
            start = start + BLOCK
        else:
            # The fingerprint only narrows it. Two different parse states that
            # hash alike would otherwise share a mask and one would be wrong.
            same = 1
            for config in range(0, CONFIGS):
                if config < count:
                    a = sequence * CONFIGS + config
                    b = other * CONFIGS + config
                    depth = tl.load(stack_depth_ptr + a)
                    if tl.load(lexer_state_ptr + a) != tl.load(lexer_state_ptr + b):
                        same = 0
                    if depth != tl.load(stack_depth_ptr + b):
                        same = 0
                    slot = tl.arange(0, STACK_STRIDE)
                    left = tl.load(
                        stack_ptr + a * STACK_STRIDE + slot,
                        mask=slot < depth,
                        other=0,
                    )
                    right = tl.load(
                        stack_ptr + b * STACK_STRIDE + slot,
                        mask=slot < depth,
                        other=0,
                    )
                    if tl.sum(tl.where(left != right, 1, 0)) != 0:
                        same = 0
            if same == 1:
                found = other
            else:
                start = other + 1
    tl.store(representative_ptr + sequence, found)

    # Only a representative's row is built up by the scatter, and every other
    # row is overwritten wholesale by the broadcast, so those are the only rows
    # that have to start empty. Clearing all of them was 9.7 MB a step to make
    # room for 0.6 MB of answers.
    if found == sequence:
        for start in range(0, mask_words, BLOCK):
            slot = start + tl.arange(0, BLOCK)
            tl.store(
                mask_ptr + sequence * mask_words + slot,
                tl.zeros((BLOCK,), tl.int32),
                mask=slot < mask_words,
            )


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


def _run_rise(arrays: dict, cap: int = 48) -> tuple[dict[int, int], set[int]]:
    """How far a run of one terminal can push the stack above where it started.

    The window bound used to count every terminal of a reading as a possible
    push, which is what a reading may do in the worst case and is nowhere near
    what one does. A reading is a token's worth of terminals, and a token that
    is a hundred spaces is a hundred copies of one terminal - a *run*. What a
    run does to the stack is a property of the terminal and the automaton, not
    of how long the run is: with repetitions lowered left-recursively each copy
    shifts and is reduced away, so the run costs a constant.

    Simulated from every state a run can start in, with the window empty. A
    reduction that pops past the floor ends the segment - the state it exposes
    belongs to the sequence's stack and is unknown here - and the segment after
    it starts from some other state, which is covered because this takes the
    maximum over all of them. A terminal whose run is still growing when the cap
    is reached is reported as growing, and its run is then charged its length.
    """
    # Python lists and `bisect`, not numpy views. Every lookup here is a single
    # scalar search, and a numpy call costs about a microsecond of interpreter
    # before it does any work - which at tens of thousands of table entries was
    # half a second of grammar construction.
    action_offsets = np.frombuffer(arrays["action_offsets"], dtype=np.uint32).tolist()
    action_terminals = np.frombuffer(
        arrays["action_terminals"], dtype=np.uint32
    ).tolist()
    action_values = np.frombuffer(arrays["action_values"], dtype=np.int32).tolist()
    goto_offsets = np.frombuffer(arrays["goto_offsets"], dtype=np.uint32).tolist()
    goto_nonterminals = np.frombuffer(
        arrays["goto_nonterminals"], dtype=np.uint32
    ).tolist()
    goto_targets = np.frombuffer(arrays["goto_targets"], dtype=np.uint32).tolist()
    production_lhs = np.frombuffer(arrays["production_lhs"], dtype=np.uint32).tolist()
    production_arity = np.frombuffer(
        arrays["production_arity"], dtype=np.uint32
    ).tolist()

    def action(state: int, terminal: int) -> int | None:
        low, high = action_offsets[state], action_offsets[state + 1]
        at = bisect.bisect_left(action_terminals, terminal, low, high)
        if at >= high or action_terminals[at] != terminal:
            return None
        return action_values[at]

    def goto(state: int, nonterminal: int) -> int | None:
        low, high = goto_offsets[state], goto_offsets[state + 1]
        at = bisect.bisect_left(goto_nonterminals, nonterminal, low, high)
        if at >= high or goto_nonterminals[at] != nonterminal:
            return None
        return goto_targets[at]

    rise: dict[int, int] = {}
    growing: set[int] = set()
    for state in range(len(action_offsets) - 1):
        low, high = action_offsets[state], action_offsets[state + 1]
        for entry in range(low, high):
            value = action_values[entry]
            # A reduction that pops anything ends the segment on its first move,
            # so it can only ever contribute nothing. Most entries are one.
            if value < 0 and value != ACCEPT:
                if production_arity[-value - 1] > 0:
                    continue
            terminal = action_terminals[entry]
            window: list[int] = []
            best = 0
            alive = True
            step = 0
            # Where the trajectory has been, as (top, depth). Repeating one at
            # the same depth means it is going round without growing, which is
            # the ordinary case once repetitions are left-recursive - and there
            # is nothing further to learn from spinning the cap out.
            seen: set[tuple[int, int]] = set()
            while alive and step < cap:
                step += 1
                mark = (window[-1] if window else state, len(window))
                if mark in seen:
                    alive = False
                    break
                seen.add(mark)
                settled = False
                spins = 0
                while not settled and alive and spins < cap:
                    spins += 1
                    top = window[-1] if window else state
                    value = action(top, terminal)
                    if value is None or value == ACCEPT:
                        alive = False
                    elif value > 0:
                        window.append(value - 1)
                        settled = True
                    else:
                        production = -value - 1
                        arity = production_arity[production]
                        if arity > len(window):
                            # Past the floor: the exposed state is the
                            # sequence's, not ours, and the segment ends.
                            alive = False
                        else:
                            for _ in range(arity):
                                window.pop()
                            exposed = window[-1] if window else state
                            target = goto(exposed, production_lhs[production])
                            if target is None:
                                alive = False
                            else:
                                window.append(target)
                    best = max(best, len(window))
                if not settled:
                    alive = False
            if alive and step >= cap:
                growing.add(terminal)
            rise[terminal] = max(rise.get(terminal, 0), best)
    return rise, growing


def _window_bound(arrays: dict, nullable: int) -> int:
    """How much window the widest reading of this grammar can need.

    A reading is charged run by run: a run of a terminal that does not grow
    costs what one run of it costs whatever its length, and a run of one that
    does grow costs its length. Charging every terminal instead put three of
    twelve schemas at the 256 cap while the deepest excursion a document made
    was 27.
    """
    offsets = np.frombuffer(arrays["reading_term_offsets"], dtype=np.uint32)
    terminals = np.frombuffer(arrays["reading_terminals"], dtype=np.uint32)
    if terminals.size == 0:
        return 2
    rise, growing = _run_rise(arrays)

    # Where a run begins: the first terminal of a reading, or a change of
    # terminal within one.
    starts = np.ones(terminals.size, dtype=bool)
    starts[1:] = terminals[1:] != terminals[:-1]
    # A reading always begins a run. An empty one has no terminal to mark, and
    # its offset is the next reading's, so drop the ones that point past the end.
    heads_at = offsets[:-1].astype(np.int64)
    starts[heads_at[heads_at < terminals.size]] = True
    at = np.flatnonzero(starts)
    lengths = np.diff(np.append(at, terminals.size))
    heads = terminals[at]

    # A table indexed by terminal, not a comprehension over runs: a wide
    # grammar has a hundred thousand runs and only a few hundred terminals.
    default = max(rise.values()) if rise else 1
    width = int(terminals.max()) + 1
    rise_of = np.full(width, default, dtype=np.int64)
    for terminal, value in rise.items():
        if terminal < width:
            rise_of[terminal] = value
    grows_of = np.zeros(width, dtype=bool)
    for terminal in growing:
        if terminal < width:
            grows_of[terminal] = True
    per_run = rise_of[heads]
    grows = grows_of[heads]
    # A run of `k` copies cannot push more than `k` shifts and their nullable
    # reductions, whatever the saturated figure says: charging a run of one the
    # rise a run of fifty reaches is what left one schema at 202 for an
    # excursion of 27.
    ceiling = lengths * (1 + nullable)
    charge = np.minimum(np.where(grows, ceiling, per_run), ceiling)

    # Which reading each run belongs to, so the runs of one reading add up.
    owner = np.searchsorted(offsets, at, side="right") - 1
    totals = np.bincount(owner, weights=charge, minlength=offsets.size - 1)
    return int(totals.max()) + nullable + 2


def _nullable_chain(arrays: dict) -> int:
    """Longest run of reductions that push without popping anything.

    The replay window only has to hold what a replay writes above where it
    started, and almost nothing does: a reduction of arity one is net zero and
    one of arity two or more takes the top back below the start, emptying the
    window. Only an arity-0 reduction grows it, and only until the chain of
    them ends at a shift.

    A chain is deterministic - `ACTION[state, terminal]` is a function, and an
    arity-0 reduction leaves the top in place, so the goto is taken from the
    same state - which makes this the longest path in a functional graph, one
    per terminal. Counting the arity-0 productions instead, which is the easy
    bound, put every schema in this corpus at the 256 cap and saved nothing.
    """
    action_offsets = np.frombuffer(arrays["action_offsets"], dtype=np.uint32)
    action_terminals = np.frombuffer(arrays["action_terminals"], dtype=np.uint32)
    action_values = np.frombuffer(arrays["action_values"], dtype=np.int32)
    goto_offsets = np.frombuffer(arrays["goto_offsets"], dtype=np.uint32)
    goto_nonterminals = np.frombuffer(arrays["goto_nonterminals"], dtype=np.uint32)
    goto_targets = np.frombuffer(arrays["goto_targets"], dtype=np.uint32)
    production_lhs = np.frombuffer(arrays["production_lhs"], dtype=np.uint32)
    production_arity = np.frombuffer(arrays["production_arity"], dtype=np.uint32)

    reduce = (action_values < 0) & (action_values != ACCEPT)
    production = np.where(reduce, -action_values.astype(np.int64) - 1, 0)
    arity = np.where(reduce, production_arity[production], 1)
    entries = np.flatnonzero(reduce & (arity == 0))
    if entries.size == 0:
        return 0

    states = np.searchsorted(action_offsets, entries, side="right") - 1
    # Lists and `bisect` again: one scalar search per entry, and a numpy call
    # costs more interpreter than the search costs work.
    lhs_of = production_lhs[production].tolist()
    terminal_of = action_terminals.tolist()
    goto_low = goto_offsets.tolist()
    nonterminals = goto_nonterminals.tolist()
    targets = goto_targets.tolist()
    step: dict[tuple[int, int], int] = {}
    for entry, state in zip(entries.tolist(), states.tolist()):
        lhs = lhs_of[entry]
        low, high = goto_low[state], goto_low[state + 1]
        at = bisect.bisect_left(nonterminals, lhs, low, high)
        if at >= high or nonterminals[at] != lhs:
            continue
        step[(terminal_of[entry], state)] = targets[at]

    longest = 0
    settled: dict[tuple[int, int], int] = {}
    for start in step:
        if start in settled:
            continue
        path: list[tuple[int, int]] = []
        on_path = set()
        node = start
        while node in step and node not in settled and node not in on_path:
            path.append(node)
            on_path.add(node)
            node = (node[0], step[node])
        if node in on_path:
            # A cycle is a parser that never settles, which the reduction limit
            # already refuses. Charge the longest simple path rather than
            # looping here.
            tail = len(step)
        else:
            tail = settled.get(node, 0)
        for depth, visited in enumerate(reversed(path)):
            settled[visited] = tail + depth + 1
            longest = max(longest, settled[visited])
    return longest




# Which base each uploaded array is addressed through.
_ARENA = {
    "group_offsets": _B_GROUP_OFFSETS,
    "group_set_kind": _B_GROUPS,
    "group_set_offset": _B_GROUPS,
    "group_set_length": _B_GROUPS,
    "set_payload": _B_SET_PAYLOAD,
    "reading_offsets": _B_READING_OFFSETS,
    "reading_index": _B_READING_INDEX,
    "reading_next_state": _B_READINGS,
    "reading_term_offsets": _B_READING_TERM_OFFSETS,
    "reading_terminals": _B_READING_TERMINALS,
    "action_offsets": _B_ACTION_OFFSETS,
    "action_terminals": _B_ACTIONS,
    "action_values": _B_ACTIONS,
    "goto_offsets": _B_GOTO_OFFSETS,
    "goto_nonterminals": _B_GOTOS,
    "goto_targets": _B_GOTOS,
    "production_lhs": _B_PRODUCTIONS,
    "production_arity": _B_PRODUCTIONS,
    "pending_offsets": _B_PENDING_OFFSETS,
    "pending_terminals": _B_PENDING_TERMINALS,
}

# One base per array, but the CSR offset arrays carry a sentinel element, so
# concatenating them shifts every following grammar by one. The base is
# whatever the concatenation actually produced, computed rather than derived.
_SIGNED = {"action_values"}


class DeviceGrammar:
    """One or more compiled grammars, resident on the GPU as one arena.

    A serving batch does not hold one grammar. Requests arrive with their own
    schemas, so the sequences in a step are under different grammars, and a
    design whose kernels are shaped by one grammar's tables cannot serve them.

    The tables are therefore laid end to end and a sequence carries the index of
    the grammar it is under. Every lookup a kernel makes is into a run that
    starts at `bases[grammar, which array]`, so the arithmetic is one addition
    and no branch; the replay itself does not know there is more than one
    grammar. What has to be shared is the vocabulary, since the mask is a row
    over it - which is what a serving engine has anyway.
    """

    def __init__(
        self,
        compiled,
        max_stack: int = 256,
        max_reductions: int | None = None,
        max_configs: int = 16,
        window: int | None = None,
    ):
        many = list(compiled) if isinstance(compiled, (list, tuple)) else [compiled]
        self.count = len(many)
        every = [item.device_arrays() for item in many]

        self.vocab_size = int(every[0]["vocab_size"])
        self.mask_words = int(every[0]["bitset_words"])
        for arrays in every:
            if int(arrays["vocab_size"]) != self.vocab_size:
                raise ValueError("grammars in one batch must share a vocabulary")
        self.start_parser_states = [int(a["start_parser_state"]) for a in every]
        self.start_parser_state = self.start_parser_states[0]

        # Ceilings are the maximum over the pool. They cost launch shape, not
        # memory, now that the scratch is per block rather than per program.
        self.max_stack = max_stack
        self.max_reductions = (
            max_reductions if max_reductions is not None else max_stack
        )
        self.max_configs = max_configs

        def widest(arrays, name):
            values = np.frombuffer(arrays[name], dtype=np.uint32)
            return int(np.diff(values).max()) if values.size > 1 else 1

        self.max_groups_per_state = max(widest(a, "group_offsets") for a in every)
        self.max_readings = max(widest(a, "reading_offsets") for a in every)
        self.max_reading_terms = max(widest(a, "reading_term_offsets") for a in every)
        self.nullable_chain = max(_nullable_chain(a) for a in every)
        needed = max(_window_bound(a, self.nullable_chain) for a in every)
        # Capped at the stack: a window that large is no worse than the copy it
        # replaces, and the overflow flag makes a grammar that genuinely needs
        # more visible rather than silently mis-masked.
        self.window = window or min(
            max_stack, 1 << max(3, int(np.ceil(np.log2(max(needed, 2)))))
        )
        self.window_bound = needed

        longest = 1
        for arrays in every:
            lengths = np.frombuffer(arrays["group_set_length"], dtype=np.uint32)
            if lengths.size:
                longest = max(longest, int(lengths.max()))
        # The lanes of a block search in lockstep, so the loop runs a fixed
        # number of times and every lane must have finished by it. A halving
        # over `n` needs ceil(log2(n)) steps; the margin covers the ends being
        # inclusive, and a bound tight enough for the scalar case left five
        # schemas disagreeing with the reference matcher.
        self.search_steps = max(2, int(np.ceil(np.log2(longest + 2))) + 2)

        bases = np.zeros((self.count, int(_NBASES.value)), dtype=np.int32)
        for name, slot in _ARENA.items():
            dtype = np.int32 if name in _SIGNED else np.uint32
            runs = [np.frombuffer(arrays[name], dtype=dtype) for arrays in every]
            at = 0
            for index, run in enumerate(runs):
                bases[index, int(slot.value)] = at
                at += run.size
            joined = np.concatenate(runs) if runs else np.zeros(1, dtype=dtype)
            setattr(
                self,
                name,
                torch.from_numpy(joined.astype(np.int32).copy()).cuda(),
            )
        self.bases = torch.from_numpy(bases.reshape(-1).copy()).cuda()
        self._resident = sum(
            getattr(self, name).numel() * 4 for name in _ARENA
        ) + self.bases.numel() * 4

    def resident_bytes(self) -> int:
        return self._resident

    def new_batch(self, batch: int) -> "DeviceBatch":
        return DeviceBatch(self, batch)


class DeviceBatch:
    """Per-sequence parser state, in device memory."""

    def __init__(self, grammar: DeviceGrammar, batch: int):
        self.grammar = grammar
        self.batch = batch
        self.configs = grammar.max_configs
        self.graph: torch.cuda.CUDAGraph | None = None
        self.advance_graph: torch.cuda.CUDAGraph | None = None

        readings = grammar.max_readings
        self.max_readings = readings
        slots = batch * self.configs * readings
        self.cand_valid = torch.zeros(slots, dtype=torch.int32, device="cuda")
        self.cand_lexer = torch.zeros(slots, dtype=torch.int32, device="cuda")
        self.cand_depth = torch.zeros(slots, dtype=torch.int32, device="cuda")
        self.cand_floor = torch.zeros(slots, dtype=torch.int32, device="cuda")
        self.cand_window = torch.zeros(
            slots * grammar.window, dtype=torch.int32, device="cuda"
        )
        self.old_lexer = torch.zeros(
            batch * self.configs, dtype=torch.int32, device="cuda"
        )
        self.old_count = torch.ones(batch, dtype=torch.int32, device="cuda")
        # The commit builds the next configuration set in place, and a candidate
        # now names the stack it came from rather than carrying a copy, so the
        # source has to survive being overwritten. One copy of the stacks a step
        # against 131 MB of candidate stacks is the trade.
        self.old_stack = torch.zeros(
            batch * self.configs * grammar.max_stack, dtype=torch.int32, device="cuda"
        )
        # One flag per sequence rather than one for the batch: a refusal is a
        # property of the sequence that hit it, and reading it back to find out
        # which would be the synchronisation this is all avoiding.
        #
        # `terminated` means the token was refused - the parse is over.
        # `overflow` means this sequence's mask may be *narrower* than the
        # grammar allows, from a replay that outran its window or a parse that
        # outgrew the configuration ceiling. Neither is meant to happen; both
        # are recorded rather than absorbed, because a narrow mask does not
        # crash, it quietly forbids a legal token. See `problems()`.
        self.terminated = torch.zeros(batch, dtype=torch.int32, device="cuda")
        self.overflow = torch.zeros(batch, dtype=torch.int32, device="cuda")
        # The deepest excursion any replay in this batch has made above where it
        # started. Not used by the step - it is how the compile-time window
        # bound gets checked against what documents actually need.
        self.high_water = torch.zeros(1, dtype=torch.int32, device="cuda")
        self.widest = torch.ones(1, dtype=torch.int32, device="cuda")
        self.state_hash = torch.zeros(batch, dtype=torch.int32, device="cuda")
        self.representative = torch.arange(batch, dtype=torch.int32, device="cuda")
        # Which grammar each sequence is under. A serving batch mixes them, and
        # everything else in the step reads this to find its tables.
        self.grammar_of = torch.zeros(batch, dtype=torch.int32, device="cuda")
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
        # How many blocks drain the sweep. Deriving this from the batch was
        # tried and is worse: a batch of eight went from 74 us to 154 us on 64
        # blocks, because fewer blocks means more items each, while a batch of
        # one was unchanged - its cost is the eight launches, not the grid.
        self.sweep_blocks = _SWEEP_BLOCKS
        self.max_groups = grammar.max_groups_per_state
        self.advance_blocks = (self.max_groups + _GROUP_BLOCK - 1) // _GROUP_BLOCK
        # One entry per work item, so nothing has to be cleared between steps:
        # the sweep writes every item it enumerates, admitted or not. The
        # allocation is still the ceiling because the host cannot know the item
        # count without asking the device, but only the items that exist are
        # ever touched - and a byte says everything an admission has to say,
        # which is 55 MB against 14 for a batch of 512 over ten grammars.
        self.admitted = torch.zeros(
            rows * grammar.max_groups_per_state, dtype=torch.int8, device="cuda"
        )
        self.counts = torch.zeros(rows, dtype=torch.int32, device="cuda")
        self.work_offsets = torch.zeros(rows + 1, dtype=torch.int32, device="cuda")
        self.live_counts = torch.zeros(rows, dtype=torch.int32, device="cuda")
        self.live_offsets = torch.zeros(rows + 1, dtype=torch.int32, device="cuda")
        # Two windows per block: one for the reading being replayed, one for
        # probing what a pending lexeme could still become. Per *block*, not per
        # program - which is the point of the sweep. This used to be one per
        # (sequence, configuration, group) and reach 1.7 GB at batch 512.
        self.scratch = torch.zeros(
            (self.sweep_blocks, 2 * grammar.window), dtype=torch.int32, device="cuda"
        )
        # The advance indexes its scratch by block, like the sweep, so it too
        # stops depending on the batch. It cannot share the sweep's buffer:
        # both may be in flight on the same stream and they would write over
        # each other's replays.
        self.advance_scratch = torch.zeros(
            (self.sweep_blocks, 2 * grammar.window),
            dtype=torch.int32,
            device="cuda",
        )
        self.found = torch.full(
            (rows,), _NO_GROUP, dtype=torch.int32, device="cuda"
        )

    def set_grammars(self, ids) -> None:
        """Say which grammar each sequence is under, and reset it to that start.

        A serving batch is heterogeneous by default - requests bring their own
        schemas - so this is the ordinary case rather than a special one. The
        step's shape does not change with the mixture: the tables are one arena,
        the work list is built from whatever the sequences are, and the grid is
        fixed, so the same CUDA graph covers any assignment.
        """
        values = torch.as_tensor(ids, dtype=torch.int32).reshape(-1)
        if values.numel() != self.batch:
            raise ValueError(
                f"{values.numel()} grammar ids for a batch of {self.batch}"
            )
        if int(values.max()) >= self.grammar.count:
            raise ValueError("grammar id past the end of the pool")
        self.grammar_of.copy_(values.cuda())
        starts = torch.tensor(
            self.grammar.start_parser_states, dtype=torch.int32
        )[values.long()]
        rows = self.stack.reshape(self.batch, self.configs, -1)
        rows[:, :, 0] = starts.reshape(self.batch, 1).cuda()
        self.depth.fill_(1)
        self.config_count.fill_(1)
        self.lexer_state.zero_()

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
        # The kernels gate on `widest`, which lives on the device because the
        # advance is the thing that normally sets it. Loading state from the
        # host has to set it too, or every configuration past the first is
        # skipped and the mask comes back narrower than the grammar allows -
        # which is not a slow parser but a wrong one. Only the fill-only path
        # reaches this, so an advance was hiding it: the commit kernel writes
        # `widest` on its way past, and checking both together let one claim
        # cover for the other.
        self.widest.fill_(int(counts.max()))
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
        self.found.fill_(_NO_GROUP)
        rows = self.batch * self.configs
        # One entry per live configuration, enumerated the way the fill
        # enumerates its groups. Sizing the grid by the width instead meant a
        # recorded advance was only valid while the parse stayed that wide, and
        # nothing checked; sizing it by the ceiling meant launching for every
        # configuration and returning at once from fifteen of sixteen, which was
        # 38 us of a 163 us step.
        _count_kernel[((rows + 255) // 256,)](
            grammar.group_offsets,
            self.lexer_state,
            self.config_count,
            self.widest,
            self.representative,
            self.grammar_of,
            grammar.bases,
            self.live_counts,
            CONFIGS=self.configs,
            ROWS=rows,
            SKIP_DUPLICATES=0,
            UNIT=1,
            BLOCK=256,
        )
        # Torch's scan rather than one of our own: a single program carrying a
        # running total across eight thousand words is one multiprocessor doing
        # what thirty could, and measured 16 us against 9.
        torch.cumsum(self.live_counts, 0, out=self.live_offsets[1:])
        _snapshot_kernel[(self.sweep_blocks,)](
            self.lexer_state,
            self.stack,
            self.config_count,
            self.live_offsets,
            self.old_lexer,
            self.old_count,
            self.old_stack,
            ROWS=rows,
            CONFIGS=self.configs,
            STACK_STRIDE=grammar.max_stack,
        )
        _locate_kernel[(self.sweep_blocks,)](
            grammar.group_offsets,
            grammar.group_set_kind,
            grammar.group_set_offset,
            grammar.group_set_length,
            grammar.set_payload,
            self.lexer_state,
            self.config_count,
            self.widest,
            self.token,
            self.grammar_of,
            grammar.bases,
            self.live_offsets,
            self.found,
            ROWS=rows,
            CONFIGS=self.configs,
            GROUP_BLOCK=_GROUP_BLOCK,
            SEARCH_STEPS=grammar.search_steps,
        )
        _candidate_kernel[(self.sweep_blocks,)](
            grammar.group_offsets,
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
            self.found,
            self.grammar_of,
            grammar.bases,
            self.advance_scratch,
            self.cand_valid,
            self.cand_lexer,
            self.cand_depth,
            self.cand_floor,
            self.cand_window,
            self.overflow,
            ROWS=rows,
            CONFIGS=self.configs,
            MAX_READINGS=self.max_readings,
            STACK_STRIDE=grammar.max_stack,
            MAX_REDUCTIONS=grammar.max_reductions,
            WINDOW=grammar.window,
            NO_GROUP=_NO_GROUP,
            num_warps=1,
        )
        _commit_kernel[(self.batch,)](
            self.lexer_state,
            self.stack,
            self.depth,
            self.config_count,
            self.old_lexer,
            self.old_count,
            self.old_stack,
            self.cand_valid,
            self.cand_lexer,
            self.cand_depth,
            self.cand_floor,
            self.cand_window,
            self.terminated,
            self.overflow,
            self.widest,
            CONFIGS=self.configs,
            MAX_READINGS=self.max_readings,
            STACK_STRIDE=grammar.max_stack,
            WINDOW=grammar.window,
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

    def problems(self) -> tuple[torch.Tensor, torch.Tensor]:
        """`(terminated, overflow)`, on the device, one entry per sequence.

        Deliberately returns the tensors rather than their contents: reading
        them is a device-to-host synchronisation, and the decode loop must not
        make one. A serving engine already reads `terminated` on its own
        schedule to retire sequences, and should read `overflow` on the same
        one - it should always be zero, and a sequence where it is not has been
        given a mask that may forbid something the grammar allows.
        """
        return self.terminated, self.overflow

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

        And the recording no longer goes stale when a parse widens. The sweep's
        grid is a fixed number of blocks draining a list the device builds, so
        one recording covers any configuration width - which is what continuous
        batching needs, since a batch changes composition every step and
        re-recording per composition is not a thing a serving loop can do.
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

    def fill_mask(self) -> torch.Tensor:
        if self.graph is not None:
            self.graph.replay()
            return self.mask
        return self._fill()

    def _fill(self) -> torch.Tensor:
        grammar = self.grammar
        # The mask is not cleared here. Deduplication knows which rows the
        # scatter will build up and clears only those; the rest are overwritten
        # whole by the broadcast.
        _hash_kernel[(self.batch,)](
            self.lexer_state,
            self.stack,
            self.depth,
            self.config_count,
            self.grammar_of,
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
            self.grammar_of,
            self.state_hash,
            self.representative,
            self.mask,
            grammar.mask_words,
            BATCH=self.batch,
            CONFIGS=self.configs,
            STACK_STRIDE=grammar.max_stack,
            BLOCK=128,
            num_warps=4,
        )
        rows = self.batch * self.configs
        _count_kernel[((rows + 255) // 256,)](
            grammar.group_offsets,
            self.lexer_state,
            self.config_count,
            self.widest,
            self.representative,
            self.grammar_of,
            grammar.bases,
            self.counts,
            CONFIGS=self.configs,
            ROWS=rows,
            SKIP_DUPLICATES=1,
            UNIT=0,
            BLOCK=256,
        )
        # The running sum turns an item back into a configuration and a group.
        # It is a device op on a device value, so the total never comes to the
        # host and the launches below do not depend on it.
        torch.cumsum(self.counts, 0, out=self.work_offsets[1:])
        _mask_kernel[(self.sweep_blocks,)](
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
            self.work_offsets,
            self.grammar_of,
            grammar.bases,
            self.scratch,
            self.admitted,
            self.overflow,
            self.high_water,
            ROWS=rows,
            CONFIGS=self.configs,
            STACK_STRIDE=grammar.max_stack,
            MAX_REDUCTIONS=grammar.max_reductions,
            WINDOW=grammar.window,
            num_warps=1,
        )
        for complements_only in (1, 0):
            _scatter_kernel[(self.sweep_blocks,)](
                grammar.group_offsets,
                grammar.group_set_kind,
                grammar.group_set_offset,
                grammar.group_set_length,
                grammar.set_payload,
                self.lexer_state,
                self.work_offsets,
                self.grammar_of,
                grammar.bases,
                self.admitted,
                self.mask,
                grammar.mask_words,
                ROWS=rows,
                CONFIGS=self.configs,
                COMPLEMENTS_ONLY=complements_only,
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
