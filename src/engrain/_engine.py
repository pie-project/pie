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

import bisect
import hashlib
import dataclasses
from dataclasses import dataclass

import numpy as np
import os
import torch
import triton
import triton.language as tl

_GROUP_BLOCK = 64

# Slots in the per-row filter that stops a group being unioned twice. A power
# of two, and small: what a row admits is a handful, and a collision costs a
# repeated union rather than a wrong answer. Sixty-four is 16 MiB at batch 512
# against 5.16 GiB for the bit-per-group shape it replaces.
_GROUP_FILTER = 64

# Window entries one sequence may pack into its candidate arena in a step.
# Measured over a real step at batch 512 across forty-nine corpus schemas, the
# most any sequence wanted was 8 configurations x 4 candidates x 7 entries, so
# 224; this is that with room, and it is 2 MiB at batch 512 against the 8 GiB
# the old grid of ceilings reserved.
_CANDIDATE_ARENA = 1024
# Threads per block for the CUDA kernels that give a warp to a configuration.
# Four warps, which is what the Triton locate uses, so the two are comparable.
_LOCATE_THREADS = 128
# A thread replays a configuration, so the launch is sized by threads and the
# scratch follows it. 128 keeps the block small enough that a heavy row does
# not hold a whole multiprocessor.
_CANDIDATE_THREADS = 128
# Blocks that drain the sweep. Fixed, and deliberately so: the grid no longer
# depends on the batch, on how wide its parses are, or on the grammar, which is
# what a CUDA graph needs and what a batch of mixed grammars would need. It also
# bounds the replay scratch, which is now one window per block rather than one
# per (sequence, configuration, group).
# Which implementation of the step runs. Triton is what everything has been
# measured and verified with; CUDA is the port in progress; `differential`
# runs both on the same input and refuses to continue if they disagree, which
# is the only check that can catch a difference the reference matcher would
# not - because a wrong answer both backends share is not a disagreement.
_TRITON, _CUDA, _DIFFERENTIAL = "triton", "cuda", "differential"
_BACKENDS = (_TRITON, _CUDA, _DIFFERENTIAL)

# Which paths the CUDA backend actually implements. Empty while the port is
# starting: `cuda` then delegates to Triton for everything, which is what lets
# the whole suite run under either name from the first day and makes each
# kernel's arrival a one-line change here rather than a switch of everything at
# once. Anything not named is Triton, honestly and by construction.
_PORTED: frozenset[str] = frozenset(
    {
        "candidate",
        "commit",
        "mask",
        "hash",
        "probe",
        "copy",
        "store",
        "restore",
        "advance_fused",
        "fill_split",
    }
)


def ported() -> frozenset[str]:
    """Which step paths the CUDA backend really runs. For tests and reports."""
    return _PORTED


def _chosen_backend() -> str:
    name = os.environ.get("ENGRAIN_BACKEND", _TRITON).strip().lower()
    if name not in _BACKENDS:
        raise ValueError(
            f"ENGRAIN_BACKEND={name!r} is not one of {', '.join(_BACKENDS)}"
        )
    return name


# Rows below which one block counting and scanning beats three launches. Set
# from measurement rather than reasoning: at 4,096 rows the fused kernel is
# 2 us against 15, and by 65,536 the parallel scan has caught up.
_SCAN_ALONE = 16384
# One chunk rather than four: the scan is sequential across chunks, so a
# narrower block turns a log-depth reduction into a serial chain. Measured at
# batch 32, 8.8 us against 12.4 at a block of 1,024 and 15.0 for the three
# launches this replaces.
_SCAN_BLOCK = 4096
_SCAN_WARPS = 8

#: The blocks are one warp each, so what a machine can hold at once is its
#: multiprocessors times its resident warps - 64 on everything since Volta.
_WARPS_PER_SM = 64
#: Blocks per sequence. The drain loops are correct at any width, so this only
#: decides how much of the machine to occupy, and the right answer turns out to
#: follow the *batch* at least as much as the device. Swept per block count in
#: its own process, since the replay scratch is sized with the grid and sharing
#: a process across widths reads out of bounds:
#:
#:     blocks    b1     b8     b32    b128     (us a step, schema 1)
#:        512  41.0  121.2   144.5   470.9
#:       1024  42.0   87.6   107.6   324.1
#:       2048  44.1   89.5   101.2   315.9
#:       4096  48.6   93.2   100.6   210.3
#:       8192  57.5  100.3   107.6   173.3
#:
#: Every optimum on that grid is `batch * 128` rounded up, held between a floor
#: that keeps a batch of one from running on a sliver of the device and the
#: ceiling the machine itself sets. A fixed 4,096 - which is what this was -
#: costs 16% at batch 1 and 18% at batch 128 on the very card it was swept on.
_BLOCKS_PER_SEQUENCE = 128
_MIN_SWEEP_BLOCKS = 512
# The memo of masks already computed. 256 entries because the corpus reaches 27
# to 551 distinct parse states over a whole document at batch 512, so this holds
# the working set of any one batch while costing 256 rows of mask - 4.9 MB
# against the 142 MB a batch of 128 already holds. The configuration and depth
# bounds keep the state beside each entry small; a parse too wide or too deep
# for them is not remembered, which costs a recomputation and nothing else.
_MEMO_SLOTS = 256
# How many stack suffixes to key on. The sweep measures how far down each mask
# looked; on this corpus and on a recursive schema that is 1 or 2 for two
# thirds to three quarters of configurations, with a long thin tail that the
# whole-stack key still catches.
_MEMO_SUFFIXES = 1

# Threads per block for the fused fill. Swept on the skewed draft walk, which
# is the workload the width shows up in - batch 128, one schema, sequences at
# six different points of a document:
#
#     32 threads   4081 us      256 threads    950 us
#     64 threads   2315 us      512 threads    launch fails, out of registers
#    128 threads   1606 us
#
# So it is the ceiling the hardware allows, and it has nothing to do with the
# stack depth that sizes the advance.
_FILL_THREADS = 256

# How much the fill's replay windows may take. The sweep spreads a sequence
# over several blocks and a thread owns two windows, so this is what decides
# how many blocks a sequence may have. See `_fill_chunks`.
_FILL_SCRATCH_BUDGET = 64 * 1024 * 1024
_MEMO_CONFIGS = 64
_MEMO_DEPTH = 64
# No real fingerprint is asked to be this, and an empty entry has to miss.
_MEMO_EMPTY = 0x7EEDFACE

#: What the memo may spend, before the machine has a say. Sized against the
#: batch buffers it sits beside - 142 MB at batch 128 - rather than against
#: anything the device promises.
_MEMO_BUDGET = 16 << 20

# Sentinel for "no group of this configuration holds the sampled token". Above
# any group index, so an atomic minimum picks the earliest real finder.
_NO_GROUP = 2**31 - 1


def _round_up(value: int) -> int:
    return 1 << max(value - 1, 1).bit_length()


def _sweep_blocks(batch: int) -> int:
    """How many blocks the drain loops should run, here and for this batch.

    Every kernel shaped this way drains a list, so any width is correct and
    only decides how much of the machine is occupied. It was a constant swept
    on one card - the kind of thing that is wrong on the next one, and was
    already wrong at both ends of the batch on that one.
    """
    override = os.environ.get("ENGRAIN_SWEEP_BLOCKS")
    if override:
        # For sweeping the curve on a machine, and for a deployment that has.
        return max(1, int(override))
    device = torch.cuda.get_device_properties(torch.cuda.current_device())
    ceiling = _round_up(device.multi_processor_count * _WARPS_PER_SM)
    return max(_MIN_SWEEP_BLOCKS, min(ceiling, _round_up(batch * _BLOCKS_PER_SEQUENCE)))


def _memo_slots(per_slot: int) -> int:
    """How many masks to remember, given what one costs here.

    A slot holds a mask row and the parse state that produced it, so its size
    follows the vocabulary and the grammar rather than anything fixed. Sizing
    the table by a count meant a schema with a large vocabulary quietly spent
    several times what one with a small vocabulary did, and a small card paid
    the same as a large one.
    """
    device = torch.cuda.get_device_properties(torch.cuda.current_device())
    budget = min(_MEMO_BUDGET, device.total_memory // 1024)
    return max(32, min(_MEMO_SLOTS, budget // max(per_slot, 1)))


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
_B_ACTION_EXTRA_OFFSETS = tl.constexpr(15)
_B_ACTION_EXTRA = tl.constexpr(16)
_B_VERDICT_OFFSETS = tl.constexpr(17)
_B_VERDICTS = tl.constexpr(18)
_B_VERDICT_STRIDE = tl.constexpr(19)
_NBASES = tl.constexpr(20)

# Matches `MAX_PATHS` in the reference matcher. Both refuse the derivations
# past it, and refusing the same ones is what keeps device and reference in
# step - a bound that differed would show up as a mask disagreement.
_MAX_PATHS = 16

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
def _owner(offsets_ptr, rows, item):
    """Which row owns work item `item`, given that row's CSR offsets.

    The last row whose offset is at or below the item. Rows that contribute
    nothing have equal offsets, so taking the *last* such row steps past them
    and lands on the one that actually holds it - which is why this is an
    upper bound rather than the exact-match `_search`.

    Written out at five launch sites before this existed. `@triton.jit`
    helpers are inlined, so the generated code is the same; what changes is
    that there is one copy of the off-by-one to get right.
    """
    low = 0
    high = rows - 1
    while low < high:
        middle = (low + high + 1) // 2
        if tl.load(offsets_ptr + middle) <= item:
            low = middle
        else:
            high = middle - 1
    return low


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
    action_extra_offsets_ptr,
    action_extra_ptr,
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
    PATHS: tl.constexpr,
):
    """Does the parser survive any reading of this group's tokens?

    A group asks one question and answers it from the stack alone, so this is
    the unit of work the sweep hands out, and the only state it needs is where
    the sequence's stack is and where its own window may go.
    """
    admitted = 0
    high_water = 0
    # The lowest stack entry any reading of this group looks at. A replay pops
    # to expose what is underneath, and how far down it goes is what the answer
    # actually depends on - everything below is untouched and cannot change it.
    # Reported so the memo can key on that much of the stack instead of all of
    # it, which is the difference between hitting and never hitting on a
    # grammar whose stack grows with the document.
    deepest = depth
    # The block is length-prefixed: a reading list is shared with every group
    # that wants the same one, wherever in the pool it is, so it cannot say how
    # long it is by where the next one starts.
    use = tl.load(reading_offsets_ptr + group)
    use_end = use + 1 + tl.load(reading_index_ptr + use)
    use = use + 1
    while use < use_end:
        reading = tl.load(reading_index_ptr + use)
        # One derivation per path, in the mixed radix each conflicted cell
        # contributes a digit to. A reading is admitted if any of them
        # survives, so this stops at the first. Where nothing conflicts -
        # every grammar that compiled before - PATHS is 1 and the loop is
        # one iteration Triton folds away.
        found = 0
        path = 0
        # How many derivations the trajectories seen so far actually had. A
        # path beyond it repeats one already run, so this is where the loop
        # stops - which is after the first path whenever nothing conflicted.
        span = 1
        while path < PATHS and found == 0 and path < span:
            rest = path
            radix = 1
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
                            if PATHS > 1:
                                # A cell holding several actions is a grammar that is
                                # ambiguous here, which these are: `oneOf` branches
                                # overlap, so two derivations reach the same string.
                                # A mask does not need the ambiguity resolved - only
                                # whether *some* derivation admits the token - so the
                                # replay runs once per combination of choices, with
                                # `rest` carrying which one in mixed radix.
                                low = tl.load(action_extra_offsets_ptr + entry)
                                high = tl.load(action_extra_offsets_ptr + entry + 1)
                                count = 1 + high - low
                                if count > 1:
                                    radix = radix * count
                                    pick = rest % count
                                    rest = rest // count
                                    if pick > 0:
                                        value = tl.load(
                                            action_extra_ptr + low + pick - 1
                                        )
                            if value == _ACCEPT:
                                alive = 0
                            elif value > 0:
                                if (
                                    copy_depth >= STACK_STRIDE
                                    or copy_depth - floor >= WINDOW
                                ):
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
                                    deepest = tl.minimum(deepest, copy_depth)
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
                                            scratch_ptr + scratch + copy_depth - floor,
                                            top,
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
                        # The path already spent its choices getting here; what
                        # is left of the radix is what the probe forks on, so
                        # one path is one derivation through both.
                        probe_rest = rest
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
                                    if PATHS > 1:
                                        low = tl.load(action_extra_offsets_ptr + entry)
                                        high = tl.load(
                                            action_extra_offsets_ptr + entry + 1
                                        )
                                        count = 1 + high - low
                                        if count > 1:
                                            radix = radix * count
                                            pick = probe_rest % count
                                            probe_rest = probe_rest // count
                                            if pick > 0:
                                                value = tl.load(
                                                    action_extra_ptr + low + pick - 1
                                                )
                                    if value == _ACCEPT:
                                        probe_settled = 1
                                    elif value > 0:
                                        probe_settled = 1
                                    else:
                                        production = -value - 1
                                        arity = tl.load(
                                            production_arity_ptr + production
                                        )
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
                                                tl.minimum(
                                                    probe_depth - 1, copy_depth - 1
                                                ),
                                            )
                                            held = tl.load(
                                                scratch_ptr
                                                + probe
                                                + tl.maximum(
                                                    probe_depth - 1 - probe_floor, 0
                                                )
                                            )
                                            exposed = tl.where(
                                                probe_depth - 1 >= probe_floor,
                                                held,
                                                under,
                                            )
                                            lhs = tl.load(
                                                production_lhs_ptr + production
                                            )
                                            grow = tl.load(goto_offsets_ptr + exposed)
                                            grow_end = tl.load(
                                                goto_offsets_ptr + exposed + 1
                                            )
                                            target = _search(
                                                goto_nonterminals_ptr,
                                                grow,
                                                grow_end,
                                                lhs,
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
                                    high_water,
                                    probe_depth - tl.minimum(probe_floor, floor),
                                )
                        if probe_alive == 1 and probe_settled == 1:
                            any_ok = 1
                        pend = pend + 1
                    alive = any_ok

            span = tl.maximum(span, radix)
            if alive == 1:
                found = 1
            path = path + 1
        if found == 1:
            admitted = 1
            use = use_end
        else:
            use = use + 1
    return admitted, high_water, deepest


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
    action_extra_offsets_ptr,
    action_extra_ptr,
    goto_offsets_ptr,
    goto_nonterminals_ptr,
    goto_targets_ptr,
    production_lhs_ptr,
    production_arity_ptr,
    pending_offsets_ptr,
    pending_terminals_ptr,
    verdict_offsets_ptr,
    verdicts_ptr,
    verdict_stride_ptr,
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
    row_floor_ptr,
    ROWS: tl.constexpr,
    CONFIGS: tl.constexpr,
    STACK_STRIDE: tl.constexpr,
    MAX_REDUCTIONS: tl.constexpr,
    WINDOW: tl.constexpr,
    PATHS: tl.constexpr,
    HAS_VERDICTS: tl.constexpr,
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

    # Every `blocks`-th item, not a contiguous run of them. A run would let a
    # block resolve which configuration it is in once instead of searching per
    # item, which was tried and is worse: the groups of one state cost wildly
    # different amounts - most die on their first terminal, a few replay whole
    # readings - so a block handed a run gets all of one state's expensive ones.
    # Striding mixes them. 37 us against 49.
    item = block
    while item < total:
        # Which configuration owns this item. Rows that contribute nothing have
        # equal offsets, so taking the *last* row whose offset is at or below
        # the item lands past them and on the row that holds it.
        row_index = _owner(work_offsets_ptr, ROWS, item)
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

        # Most of this answer does not depend on the stack. A group whose every
        # reading dies on a missing action dies for any stack, and that is 91%
        # of all replays on real grammars, so it is settled when the tables are
        # built and read here instead of run. Two bits a group; 1 means refused
        # and there is nothing to replay.
        settled = 0
        if HAS_VERDICTS == 1:
            stride = tl.load(
                verdict_stride_ptr + tl.load(at + _B_VERDICT_STRIDE) + state
            )
            if stride > 0:
                top = tl.load(stack_ptr + row_index * STACK_STRIDE + depth - 1)
                word = (
                    verdicts_ptr
                    + tl.load(at + _B_VERDICTS)
                    + tl.load(
                        verdict_offsets_ptr + tl.load(at + _B_VERDICT_OFFSETS) + state
                    )
                    + top * stride
                    + slot // 16
                )
                settled = (tl.load(word) >> (2 * (slot % 16))) & 3

        admitted = 0
        reach = 0
        # A settled group is refused without looking at the stack at all, so it
        # constrains nothing; only a replay can widen how much of the stack the
        # answer depends on.
        read = depth
        if settled == 0:
            admitted, reach, read = _replay_group(
                my_group_offsets,
                reading_offsets_ptr + tl.load(at + _B_READING_OFFSETS),
                reading_index_ptr + tl.load(at + _B_READING_INDEX),
                reading_next_state_ptr + tl.load(at + _B_READINGS),
                reading_term_offsets_ptr + tl.load(at + _B_READING_TERM_OFFSETS),
                reading_terminals_ptr + tl.load(at + _B_READING_TERMINALS),
                action_offsets_ptr + tl.load(at + _B_ACTION_OFFSETS),
                action_terminals_ptr + tl.load(at + _B_ACTIONS),
                action_values_ptr + tl.load(at + _B_ACTIONS),
                action_extra_offsets_ptr + tl.load(at + _B_ACTION_EXTRA_OFFSETS),
                action_extra_ptr + tl.load(at + _B_ACTION_EXTRA),
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
                PATHS=PATHS,
            )
        # Written whether or not the group is admitted, so the buffer never has
        # to be cleared - at batch 512 that clear was 13 MB a step.
        tl.store(admitted_ptr + item, admitted.to(tl.int8))
        high_water = tl.maximum(high_water, reach)
        # The sweep asks every group, so the deepest any of them looked is how
        # much of this configuration's stack the finished mask depends on.
        # Atomic because a row's groups are spread across blocks - but only
        # when there is something to say. A group that is settled, or whose
        # readings die on their first terminal, never looks below the top and
        # leaves `read` at the depth it was seeded with; those are most items,
        # and doing the atomic for them anyway cost more than everything the
        # suffix key buys - 40 to 146 us at batch 32 on the widest schema.
        if read < depth:
            tl.atomic_min(row_floor_ptr + row_index, read)
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
    memo_slot_ptr,
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
        # A remembered state enumerates nothing at all: its answer is already
        # written and the restore puts it back.
        keep = keep & (tl.load(memo_slot_ptr + sequence, mask=live, other=0) < 0)
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
def _count_scan_kernel(
    group_offsets_ptr,
    lexer_state_ptr,
    config_count_ptr,
    widest_ptr,
    representative_ptr,
    memo_slot_ptr,
    grammar_ptr,
    bases_ptr,
    offsets_ptr,
    CONFIGS: tl.constexpr,
    ROWS: tl.constexpr,
    SKIP_DUPLICATES: tl.constexpr,
    UNIT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """The same counts as `_count_kernel`, already summed.

    One launch instead of three. The counts were written out, read back by
    cuBLAS's two-kernel scan and written again, which at batch 32 was 15 us of
    a 124 us step to prefix-sum four thousand numbers most of which are zero.
    One block is enough for that, and a block that has just computed a value
    can scan it without it ever reaching memory.

    Deliberately one program: the scan is sequential across chunks, so a wider
    grid would need a second pass, which is the thing being removed. At large
    `ROWS` a single block loses to the parallel scan, so the caller picks.
    """
    running = 0
    start = 0
    tl.store(offsets_ptr, 0)
    while start < ROWS:
        lane = start + tl.arange(0, BLOCK)
        live = lane < ROWS
        sequence = lane // CONFIGS
        config = lane % CONFIGS
        keep = live
        keep = keep & (config < tl.load(widest_ptr))
        keep = keep & (
            config < tl.load(config_count_ptr + sequence, mask=live, other=0)
        )
        if SKIP_DUPLICATES == 1:
            keep = keep & (
                tl.load(representative_ptr + sequence, mask=live, other=-1) == sequence
            )
            keep = keep & (tl.load(memo_slot_ptr + sequence, mask=live, other=0) < 0)
        if UNIT == 1:
            value = tl.where(keep, 1, 0)
        else:
            grammar = tl.load(grammar_ptr + sequence, mask=live, other=0)
            at = tl.load(
                bases_ptr + grammar * _NBASES + _B_GROUP_OFFSETS, mask=keep, other=0
            )
            state = tl.load(lexer_state_ptr + lane, mask=keep, other=0)
            first = tl.load(group_offsets_ptr + at + state, mask=keep, other=0)
            last = tl.load(group_offsets_ptr + at + state + 1, mask=keep, other=0)
            value = tl.where(keep, last - first, 0)
        tl.store(offsets_ptr + 1 + lane, running + tl.cumsum(value, axis=0), mask=live)
        running = running + tl.sum(value)
        start = start + BLOCK


@triton.jit
def _history_kernel(
    lexer_state_ptr,
    stack_ptr,
    stack_depth_ptr,
    config_count_ptr,
    live_offsets_ptr,
    slot_ptr,
    hist_lexer_ptr,
    hist_stack_ptr,
    hist_depth_ptr,
    hist_count_ptr,
    ROWS: tl.constexpr,
    CONFIGS: tl.constexpr,
    STACK_STRIDE: tl.constexpr,
    DEPTH: tl.constexpr,
):
    """Keep this step's parse state so a later one can be undone.

    Speculative decoding advances through a draft and then keeps only the prefix
    the model accepted, so the parser has to go back - and going back by asking
    the host to replay the tokens is the round trip this design exists not to
    make.

    The slot to write is read from the device rather than passed in, which is
    what lets the advance stay inside a CUDA graph: a graph records the
    arguments it was given, so a slot that arrived as a scalar would be frozen
    at whatever it was when the recording was made.
    """
    program = tl.program_id(0)
    programs = tl.num_programs(0)
    slot = tl.load(slot_ptr) % DEPTH
    at = slot * ROWS
    total = tl.load(live_offsets_ptr + ROWS)
    lane = tl.arange(0, STACK_STRIDE)

    # Only the configurations in play, and only as deep as they go. Writing
    # every row to its full stack depth is 67 MB a step at batch 512 with 128
    # configurations, to preserve the one or two configurations a sequence
    # actually holds - which took the advance from 133 us to 1,200.
    item = program
    while item < total:
        row_index = _owner(live_offsets_ptr, ROWS, item)
        depth = tl.load(stack_depth_ptr + row_index)
        tl.store(hist_lexer_ptr + at + row_index, tl.load(lexer_state_ptr + row_index))
        tl.store(hist_depth_ptr + at + row_index, depth)
        sequence = row_index // CONFIGS
        tl.store(
            hist_count_ptr + slot * (ROWS // CONFIGS) + sequence,
            tl.load(config_count_ptr + sequence),
        )
        tl.store(
            hist_stack_ptr + (at + row_index) * STACK_STRIDE + lane,
            tl.load(
                stack_ptr + row_index * STACK_STRIDE + lane, mask=lane < depth, other=0
            ),
            mask=lane < depth,
        )
        item = item + programs


@triton.jit
def _restore_kernel(
    lexer_state_ptr,
    stack_ptr,
    stack_depth_ptr,
    config_count_ptr,
    widest_ptr,
    hist_lexer_ptr,
    hist_stack_ptr,
    hist_depth_ptr,
    hist_count_ptr,
    slot,
    ROWS: tl.constexpr,
    CONFIGS: tl.constexpr,
    STACK_STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Put a kept parse state back. The slot is known on the host here.

    Unlike the advance, a rollback is not part of a captured decode step - it
    happens when a draft is rejected, which the host already knows about - so
    the slot may be an argument.
    """
    row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    live = row < ROWS
    at = slot * ROWS
    tl.store(
        lexer_state_ptr + row,
        tl.load(hist_lexer_ptr + at + row, mask=live, other=0),
        mask=live,
    )
    tl.store(
        stack_depth_ptr + row,
        tl.load(hist_depth_ptr + at + row, mask=live, other=1),
        mask=live,
    )
    counted = row < ROWS // CONFIGS
    counts = tl.load(
        hist_count_ptr + slot * (ROWS // CONFIGS) + row, mask=counted, other=1
    )
    tl.store(config_count_ptr + row, counts, mask=counted)
    tl.atomic_max(widest_ptr, tl.max(tl.where(counted, counts, 1)))
    for start in range(0, STACK_STRIDE):
        tl.store(
            stack_ptr + row * STACK_STRIDE + start,
            tl.load(
                hist_stack_ptr + (at + row) * STACK_STRIDE + start, mask=live, other=0
            ),
            mask=live,
        )


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
    BLOCK: tl.constexpr,
):
    """Write the sets of the admitted groups.

    Every kind writes with an OR of a value it decided by itself, so the result
    does not depend on the order the items are drained in. It used to take two
    launches - complements, then the rest - because a complement set the row
    and then punched its exclusions out, which erased whatever had already been
    written. That ordering was not enough either: two complements erase each
    other, and a sequence with configurations at two lexer states has two.

    Drains the same item list as the sweep, so an item means the same
    (configuration, group) in both and admission can be recorded against it.
    """
    block = tl.program_id(0)
    blocks = tl.num_programs(0)
    total = tl.load(work_offsets_ptr + ROWS)
    item = block
    while item < total:
        if tl.load(admitted_ptr + item) != 0:
            row_index = _owner(work_offsets_ptr, ROWS, item)
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
            offset = tl.load(group_set_offset_ptr + groups + group)
            length = tl.load(group_set_length_ptr + groups + group)
            row = mask_ptr + sequence * mask_words
            if kind == _COMPLEMENT:
                # Every word's value is decided before it is written, and
                # the write is an OR, so no ordering between programs can
                # undo it. Setting the row and then punching the exclusions
                # out is only correct while there is one complement: a
                # sequence holding configurations at two lexer states can
                # have two, and the second punched out what the first
                # admitted. It cost half the mask on one schema, and it
                # narrowed rather than widened, which is the failure this
                # engine must never make.
                cursor = offset
                stop = offset + length
                for start in range(0, mask_words, BLOCK):
                    lane = start + tl.arange(0, BLOCK)
                    live = lane < mask_words
                    value = tl.full((BLOCK,), -1, tl.int32)
                    # The exclusions are stored ascending, so one pass over
                    # them serves every block in turn.
                    limit = (start + BLOCK) * 32
                    going = 1
                    while cursor < stop and going == 1:
                        token = tl.load(payload + cursor)
                        if token < limit:
                            value = tl.where(
                                lane == token // 32,
                                value & (~(1 << (token % 32))).to(tl.int32),
                                value,
                            )
                            cursor = cursor + 1
                        else:
                            going = 0
                    tl.atomic_or(row + lane, value, mask=live)
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
def _locate_kernel(
    group_offsets_ptr,
    group_set_kind_ptr,
    group_set_offset_ptr,
    group_set_length_ptr,
    set_payload_ptr,
    verdict_offsets_ptr,
    verdicts_ptr,
    verdict_stride_ptr,
    lexer_state_ptr,
    stack_ptr,
    stack_depth_ptr,
    config_count_ptr,
    widest_ptr,
    token_ptr,
    grammar_ptr,
    bases_ptr,
    live_offsets_ptr,
    found_ptr,
    old_lexer_ptr,
    old_count_ptr,
    old_stack_ptr,
    ROWS: tl.constexpr,
    CONFIGS: tl.constexpr,
    GROUP_BLOCK: tl.constexpr,
    SEARCH_STEPS: tl.constexpr,
    STACK_STRIDE: tl.constexpr,
    HAS_VERDICTS: tl.constexpr,
    NO_GROUP: tl.constexpr,
    VOCAB: tl.constexpr,
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
        row_index = _owner(live_offsets_ptr, ROWS, slot)
        sequence = row_index // CONFIGS
        # The old state, saved here rather than in a launch of its own. The
        # advance writes new configurations while reading old ones, so the copy
        # has to happen before `_candidate_kernel` - but it reads the same row
        # this program is already holding, and one program owns a row.
        tl.store(old_lexer_ptr + row_index, tl.load(lexer_state_ptr + row_index))
        tl.store(old_count_ptr + sequence, tl.load(config_count_ptr + sequence))
        lane = tl.arange(0, STACK_STRIDE)
        tl.store(
            old_stack_ptr + row_index * STACK_STRIDE + lane,
            tl.load(stack_ptr + row_index * STACK_STRIDE + lane),
        )
        tl.store(
            found_ptr + row_index,
            _locate_one(
                group_offsets_ptr,
                group_set_kind_ptr,
                group_set_offset_ptr,
                group_set_length_ptr,
                set_payload_ptr,
                verdict_offsets_ptr,
                verdicts_ptr,
                verdict_stride_ptr,
                lexer_state_ptr,
                stack_ptr,
                stack_depth_ptr,
                token_ptr,
                grammar_ptr,
                bases_ptr,
                sequence,
                row_index,
                GROUP_BLOCK=GROUP_BLOCK,
                SEARCH_STEPS=SEARCH_STEPS,
                STACK_STRIDE=STACK_STRIDE,
                HAS_VERDICTS=HAS_VERDICTS,
                NO_GROUP=NO_GROUP,
                VOCAB=VOCAB,
            ),
        )
        slot = slot + programs


@triton.jit
def _locate_one(
    group_offsets_ptr,
    group_set_kind_ptr,
    group_set_offset_ptr,
    group_set_length_ptr,
    set_payload_ptr,
    verdict_offsets_ptr,
    verdicts_ptr,
    verdict_stride_ptr,
    lexer_state_ptr,
    stack_ptr,
    stack_depth_ptr,
    token_ptr,
    grammar_ptr,
    bases_ptr,
    sequence,
    row_index,
    GROUP_BLOCK: tl.constexpr,
    SEARCH_STEPS: tl.constexpr,
    STACK_STRIDE: tl.constexpr,
    HAS_VERDICTS: tl.constexpr,
    NO_GROUP: tl.constexpr,
    VOCAB: tl.constexpr,
):
    """Search one configuration's groups for the sampled token."""
    state = tl.load(lexer_state_ptr + row_index)
    at = bases_ptr + tl.load(grammar_ptr + sequence) * _NBASES
    groups = tl.load(at + _B_GROUPS)
    payload = set_payload_ptr + tl.load(at + _B_SET_PAYLOAD)
    my_group_offsets = group_offsets_ptr + tl.load(at + _B_GROUP_OFFSETS)
    first = tl.load(my_group_offsets + state)
    last = tl.load(my_group_offsets + state + 1)
    # One program owns a row - the advance enumerates one item per live
    # configuration - so the smallest group holding the token is a running
    # minimum in a register. It used to be an atomic into an array the step
    # had to clear first, which was a launch and a pass over the ceiling to
    # initialise what this returns anyway.
    best = NO_GROUP

    # Find the group holding the token a block of groups at a time rather than
    # one per program. A state can have hundreds of groups and only one of them
    # holds any given token, so the search is nearly all rejection - and doing
    # it one program per group turns three contiguous arrays into three
    # scattered loads per group. Read as a block they are three coalesced
    # loads for the whole block, which is the difference this kernel is made
    # of: it is bound by how many scattered loads it issues, not by arithmetic.
    token = tl.load(token_ptr + sequence)
    # A token id outside the vocabulary is in no group. Bounded here rather
    # than on the host because the sampled tokens are never read there, and a
    # dense set is indexed by the id: one too large read past the payload and
    # took the CUDA context with it.
    in_vocabulary = (token >= 0) & (token < VOCAB)
    verdict_row = verdicts_ptr
    verdict_stride = 0
    if HAS_VERDICTS == 1:
        verdict_stride = tl.load(
            verdict_stride_ptr + tl.load(at + _B_VERDICT_STRIDE) + state
        )
        depth = tl.load(stack_depth_ptr + row_index)
        if verdict_stride > 0 and depth > 0:
            top = tl.load(stack_ptr + row_index * STACK_STRIDE + depth - 1)
            verdict_row = (
                verdicts_ptr
                + tl.load(at + _B_VERDICTS)
                + tl.load(
                    verdict_offsets_ptr + tl.load(at + _B_VERDICT_OFFSETS) + state
                )
                + top * verdict_stride
            )
    glane = tl.arange(0, GROUP_BLOCK)
    start = first
    while start < last:
        group = start + glane
        live_lane = group < last
        kind = tl.load(group_set_kind_ptr + groups + group, mask=live_lane, other=0)
        offset = tl.load(group_set_offset_ptr + groups + group, mask=live_lane, other=0)
        length = tl.load(group_set_length_ptr + groups + group, mask=live_lane, other=1)

        dense = kind == _DENSE
        word = tl.load(
            payload + offset + token // 32,
            mask=live_lane & dense & in_vocabulary,
            other=0,
        )
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
        inside = (
            tl.where(dense, in_dense, tl.where(complement, found == 0, found))
            & live_lane
            & in_vocabulary
        )
        # A group the tables already refused for this parser state cannot be
        # the one that advances, and on real grammars 91% of them are. Applied
        # to the decision rather than to the loads, because the loads are
        # masked already and narrowing them changes what the minimum below
        # reduces over.
        if HAS_VERDICTS == 1:
            if verdict_stride > 0 and depth > 0:
                at_slot = group - first
                packed = tl.load(verdict_row + at_slot // 16, mask=live_lane, other=0)
                inside = inside & (((packed >> (2 * (at_slot % 16))) & 3) != 1)

        if tl.sum(inside.to(tl.int32)) != 0:
            best = tl.minimum(best, tl.min(tl.where(inside, group, last)))
        start = start + GROUP_BLOCK
    return best


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
    action_extra_offsets_ptr,
    action_extra_ptr,
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
    cand_count_ptr,
    cand_lexer_ptr,
    cand_depth_ptr,
    cand_floor_ptr,
    cand_window_ptr,
    cand_at_ptr,
    cand_used_ptr,
    overflow_ptr,
    ROWS: tl.constexpr,
    CONFIGS: tl.constexpr,
    MAX_READINGS: tl.constexpr,
    STACK_STRIDE: tl.constexpr,
    MAX_REDUCTIONS: tl.constexpr,
    WINDOW: tl.constexpr,
    ARENA: tl.constexpr,
    NO_GROUP: tl.constexpr,
    PATHS: tl.constexpr,
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
        # Written for every row, not only the ones that found a group. The
        # count replaced a buffer of flags that was cleared each step, and
        # clearing is exactly what a row skipped here does not get - so a
        # configuration whose token is in no group kept whatever count the
        # previous step left, and the commit read candidates that were not
        # there. It survived because a row almost always finds a group.
        if group >= NO_GROUP:
            tl.store(cand_count_ptr + row_index, 0)
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
            action_extra_offsets = action_extra_offsets_ptr + tl.load(
                at + _B_ACTION_EXTRA_OFFSETS
            )
            action_extra = action_extra_ptr + tl.load(at + _B_ACTION_EXTRA)
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
            use_end = use + 1 + tl.load(reading_index + use)
            use = use + 1
            index = 0
            while use < use_end and index < MAX_READINGS:
                reading = tl.load(reading_index + use)
                # Unlike the mask, every surviving derivation is kept: two
                # of them reach different stacks, and both are states the
                # next token may be read from. A path past `radix` repeats
                # a trajectory already taken, so it is not emitted.
                path = 0
                span = 1
                while path < PATHS and index < MAX_READINGS and path < span:
                    rest = path
                    radix = 1
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
                                entry = _search(
                                    action_terminals, row, row_end, terminal
                                )
                                if entry < 0:
                                    alive = 0
                                else:
                                    value = tl.load(action_values + entry)
                                    if PATHS > 1:
                                        low = tl.load(action_extra_offsets + entry)
                                        high = tl.load(action_extra_offsets + entry + 1)
                                        count = 1 + high - low
                                        if count > 1:
                                            radix = radix * count
                                            pick = rest % count
                                            rest = rest // count
                                            if pick > 0:
                                                value = tl.load(
                                                    action_extra + low + pick - 1
                                                )
                                    if value == _ACCEPT:
                                        alive = 0
                                    elif value > 0:
                                        if (
                                            copy_depth >= STACK_STRIDE
                                            or copy_depth - floor >= WINDOW
                                        ):
                                            alive = 0
                                            tl.store(overflow_ptr + sequence, 1)
                                        else:
                                            tl.store(
                                                scratch_ptr
                                                + scratch
                                                + copy_depth
                                                - floor,
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
                                            grow_end = tl.load(
                                                goto_offsets + exposed + 1
                                            )
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
                                                    scratch_ptr
                                                    + scratch
                                                    + copy_depth
                                                    - floor,
                                                    top,
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
                                probe_rest = rest
                                while (
                                    probe_settled == 0
                                    and probe_alive == 1
                                    and probe_spins < MAX_REDUCTIONS
                                ):
                                    probe_spins = probe_spins + 1
                                    if probe_settled == 0 and probe_alive == 1:
                                        row = tl.load(action_offsets + probe_top)
                                        row_end = tl.load(
                                            action_offsets + probe_top + 1
                                        )
                                        entry = _search(
                                            action_terminals, row, row_end, terminal
                                        )
                                        if entry < 0:
                                            probe_alive = 0
                                        else:
                                            value = tl.load(action_values + entry)
                                            if PATHS > 1:
                                                low = tl.load(
                                                    action_extra_offsets + entry
                                                )
                                                high = tl.load(
                                                    action_extra_offsets + entry + 1
                                                )
                                                count = 1 + high - low
                                                if count > 1:
                                                    radix = radix * count
                                                    pick = probe_rest % count
                                                    probe_rest = probe_rest // count
                                                    if pick > 0:
                                                        value = tl.load(
                                                            action_extra
                                                            + low
                                                            + pick
                                                            - 1
                                                        )
                                            if value == _ACCEPT:
                                                probe_settled = 1
                                            elif value > 0:
                                                probe_settled = 1
                                            else:
                                                production = -value - 1
                                                arity = tl.load(
                                                    production_arity + production
                                                )
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
                                                        tl.minimum(
                                                            probe_depth - 1,
                                                            copy_depth - 1,
                                                        ),
                                                    )
                                                    held = tl.load(
                                                        scratch_ptr
                                                        + probe
                                                        + tl.maximum(
                                                            probe_depth
                                                            - 1
                                                            - probe_floor,
                                                            0,
                                                        )
                                                    )
                                                    exposed = tl.where(
                                                        probe_depth - 1 >= probe_floor,
                                                        held,
                                                        under,
                                                    )
                                                    lhs = tl.load(
                                                        production_lhs + production
                                                    )
                                                    grow = tl.load(
                                                        goto_offsets + exposed
                                                    )
                                                    grow_end = tl.load(
                                                        goto_offsets + exposed + 1
                                                    )
                                                    target = _search(
                                                        goto_nonterminals,
                                                        grow,
                                                        grow_end,
                                                        lhs,
                                                    )
                                                    if target < 0:
                                                        probe_alive = 0
                                                    elif (
                                                        probe_depth >= STACK_STRIDE
                                                        or probe_depth - probe_floor
                                                        >= WINDOW
                                                    ):
                                                        probe_alive = 0
                                                        tl.store(
                                                            overflow_ptr + sequence, 1
                                                        )
                                                    else:
                                                        probe_top = tl.load(
                                                            goto_targets + target
                                                        )
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

                    if alive == 1 and path < radix:
                        # A candidate outlives the step that made it, so unlike a
                        # replay it does have to be written down - but not as a
                        # whole stack. It shares everything below its floor with the
                        # configuration it came from, and the commit can read that
                        # there, so what is stored is the floor and the window.
                        # Whole stacks made this 151 MB at batch 512, four fifths of
                        # everything a batch allocated.
                        # Packed into the sequence's arena, not placed in a
                        # grid of worst cases. The grid was `rows x readings x
                        # window` and reserved 8.00 GiB at batch 512 to write
                        # 0.01 MiB of it.
                        need = copy_depth - floor
                        bump = tl.atomic_add(cand_used_ptr + sequence, need)
                        if bump + need > ARENA:
                            tl.store(overflow_ptr + sequence, 1)
                        else:
                            tl.store(cand_lexer_ptr + out_base + index, next_state)
                            tl.store(cand_depth_ptr + out_base + index, copy_depth)
                            tl.store(cand_floor_ptr + out_base + index, floor)
                            base_at = sequence * ARENA + bump
                            tl.store(cand_at_ptr + out_base + index, base_at)
                            top_lane = tl.arange(0, WINDOW)
                            tl.store(
                                cand_window_ptr + base_at + top_lane,
                                tl.load(
                                    scratch_ptr + scratch + top_lane,
                                    mask=top_lane < need,
                                    other=0,
                                ),
                                mask=top_lane < need,
                            )
                            index = index + 1
                    span = tl.maximum(span, radix)
                    path = path + 1
                use = use + 1
            # How many candidates this configuration produced, so the commit
            # reads that many rather than the ceiling - and so nothing has to
            # be cleared between steps. Clearing the ceiling was 2 MB a step at
            # batch 32 to make room for a few dozen answers.
            tl.store(cand_count_ptr + row_index, index)
            # A configuration with more derivations than there is room for keeps
            # a prefix of them, which narrows the mask at the next token. That
            # is the one failure this engine must not do quietly, and the slot
            # count is a ceiling rather than a property of the grammar, so
            # reaching it is reported through the same flag as a replay that
            # overran its window.
            if use < use_end and index >= MAX_READINGS:
                tl.store(overflow_ptr + sequence, 1)
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
    cand_count_ptr,
    cand_lexer_ptr,
    cand_depth_ptr,
    cand_floor_ptr,
    cand_window_ptr,
    cand_at_ptr,
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
                        if (
                            tl.load(old_lexer_ptr + sequence * CONFIGS + source)
                            == state
                        ):
                            base = (sequence * CONFIGS + source) * MAX_READINGS
                            made = tl.load(cand_count_ptr + sequence * CONFIGS + source)
                            index = 0
                            while index < made:
                                if written >= CONFIGS:
                                    saturated = 1
                                if written < CONFIGS:
                                    if 1 == 1:
                                        next_state = tl.load(
                                            cand_lexer_ptr + base + index
                                        )
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
                                                + tl.load(cand_at_ptr + base + index)
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
                                                tl.load(stack_depth_ptr + out) == depth
                                            ):
                                                held = tl.load(
                                                    stack_ptr
                                                    + out * STACK_STRIDE
                                                    + lane,
                                                    mask=lane < depth,
                                                    other=0,
                                                )
                                                if (
                                                    tl.sum(
                                                        tl.where(
                                                            (lane < depth)
                                                            & (held != values),
                                                            1,
                                                            0,
                                                        )
                                                    )
                                                    == 0
                                                ):
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
                                index = index + 1
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
    suffix_hash_ptr,
    CONFIGS: tl.constexpr,
    STACK_STRIDE: tl.constexpr,
    SUFFIXES: tl.constexpr,
):
    """A fingerprint of one sequence's parse state, whole and by suffix.

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
    # Over the configurations that exist, not the ceiling. A `range` over a
    # constexpr is unrolled whole, so this was 128 bodies each loading 256
    # stack slots to fold the one to twelve a sequence actually holds - the
    # same mistake as the fill grid and the replay scratch, in a kernel small
    # enough that nobody looked.
    config = 0
    while config < count:
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
        config = config + 1
    tl.store(hash_ptr + sequence, digest)

    # The same fingerprint over only the top `k` of every stack, for every k at
    # once. A mask depends on the stack only as far down as its replay looked,
    # which the sweep measures - so an entry saved under the suffix it actually
    # needs is found again by whatever agrees on that much, however different
    # the two stacks are underneath. That is the whole of the difference on a
    # grammar that nests: the stack grows with the document and the answer does
    # not.
    #
    # All the suffixes in one pass over the configurations. A pass apiece reads
    # the stack `SUFFIXES` times over, and a sequence here can hold sixty-four
    # configurations - that cost more than the suffix key buys, 40 to 146 us at
    # batch 32 on the widest schema.
    width = tl.arange(0, SUFFIXES) + 1
    digests = tl.full((SUFFIXES,), 2166136261, tl.int32)
    digests = (digests ^ tl.load(grammar_ptr + sequence)) * 16777619
    digests = (digests ^ count) * 16777619
    digests = (digests ^ width) * 16777619
    config = 0
    while config < count:
        row = sequence * CONFIGS + config
        depth = tl.load(stack_depth_ptr + row)
        lane = tl.arange(0, STACK_STRIDE)
        values = tl.load(
            stack_ptr + row * STACK_STRIDE + lane, mask=lane < depth, other=0
        )
        kept = tl.minimum(depth, width)
        floor = depth - kept
        inside = (lane[None, :] >= floor[:, None]) & (lane[None, :] < depth)
        weight = lane[None, :] - floor[:, None] + 1
        folded = tl.sum(tl.where(inside, values[None, :] * weight, 0), axis=1)
        digests = (digests ^ tl.load(lexer_state_ptr + row)) * 16777619
        digests = (digests ^ kept) * 16777619
        # A stack shorter than `k` is folded whole, so it can never look like a
        # longer one that happens to end the same way.
        digests = (digests ^ tl.where(depth <= width, 1, 0)) * 16777619
        digests = (digests ^ folded) * 16777619
        config = config + 1
    tl.store(suffix_hash_ptr + sequence * SUFFIXES + tl.arange(0, SUFFIXES), digests)


@triton.jit
def _probe_kernel(
    lexer_state_ptr,
    stack_ptr,
    stack_depth_ptr,
    config_count_ptr,
    grammar_ptr,
    hash_ptr,
    suffix_hash_ptr,
    memo_hash_ptr,
    memo_lexer_ptr,
    memo_stack_ptr,
    memo_depth_ptr,
    memo_count_ptr,
    memo_grammar_ptr,
    memo_read_ptr,
    memo_slot_ptr,
    representative_ptr,
    memo_store_ptr,
    row_floor_ptr,
    mask_ptr,
    mask_words,
    BATCH: tl.constexpr,
    CONFIGS: tl.constexpr,
    STACK_STRIDE: tl.constexpr,
    SLOTS: tl.constexpr,
    MEMO_CONFIGS: tl.constexpr,
    MEMO_STRIDE: tl.constexpr,
    SUFFIXES: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Where this sequence's mask is coming from: a table, a neighbour, or work.

    One question - has this parse state been masked already - asked in the two
    places an answer can be. The table holds what earlier steps computed; the
    batch holds what *this* step is about to. These were separate mechanisms
    with separate kernels, and the second is only the first with its provider
    written moments ago rather than moments before.

    Three outcomes, and every sequence gets exactly one:

        a slot        the table holds this state; take the mask out of it
        a neighbour   a lower sequence holds it and is about to compute it
        neither       compute it, and save it if the slot is free

    The fingerprint only narrows; both lookups confirm the state exactly,
    because a collision would hand back another state's mask and a wrong mask
    is worse than a slow one. Two sequences with the same state have the same
    fingerprint and probe the same slot, so a neighbour that matches is one
    that missed the table too - nothing ends up following a follower.
    """
    sequence = tl.program_id(0)
    count = tl.load(config_count_ptr + sequence)
    mine = tl.load(grammar_ptr + sequence)

    found = -1
    attempt = 0
    while attempt <= SUFFIXES and found < 0:
        if attempt == 0:
            want = -1
            digest = tl.load(hash_ptr + sequence)
        else:
            want = attempt
            digest = tl.load(suffix_hash_ptr + sequence * SUFFIXES + attempt - 1)
        slot = (digest & 0x7FFFFFFF) % SLOTS
        if count <= MEMO_CONFIGS:
            same = 1
            if tl.load(memo_hash_ptr + slot) != digest:
                same = 0
            if tl.load(memo_read_ptr + slot) != want:
                same = 0
            if tl.load(memo_count_ptr + slot) != count:
                same = 0
            if tl.load(memo_grammar_ptr + slot) != mine:
                same = 0
            config = 0
            while config < count and same == 1:
                row = sequence * CONFIGS + config
                held = slot * MEMO_CONFIGS + config
                depth = tl.load(stack_depth_ptr + row)
                kept = depth
                if want > 0:
                    kept = tl.minimum(depth, want)
                if tl.load(lexer_state_ptr + row) != tl.load(memo_lexer_ptr + held):
                    same = 0
                if kept != tl.load(memo_depth_ptr + held):
                    same = 0
                lane = tl.arange(0, STACK_STRIDE)
                live = (lane < kept) & (lane < MEMO_STRIDE)
                left = tl.load(
                    stack_ptr + row * STACK_STRIDE + depth - kept + lane,
                    mask=live,
                    other=0,
                )
                right = tl.load(
                    memo_stack_ptr + held * MEMO_STRIDE + lane, mask=live, other=0
                )
                if tl.sum(tl.where(left != right, 1, 0)) != 0:
                    same = 0
                config = config + 1
            if same == 1:
                found = slot
        attempt = attempt + 1
    tl.store(memo_slot_ptr + sequence, found)

    # The batch, scanned a block of candidates at a time: every sequence walks
    # every earlier one, so this is quadratic, and one at a time it showed -
    # 36 us of a 182 us step at batch 512.
    neighbour = sequence
    if found < 0:
        digest = tl.load(hash_ptr + sequence)
        lane = tl.arange(0, BLOCK)
        begin = 0
        while begin < sequence and neighbour == sequence:
            index = begin + lane
            live = index < sequence
            alike = live
            alike = alike & (tl.load(hash_ptr + index, mask=live, other=0) == digest)
            alike = alike & (
                tl.load(config_count_ptr + index, mask=live, other=-1) == count
            )
            alike = alike & (tl.load(grammar_ptr + index, mask=live, other=-1) == mine)
            other = tl.min(tl.where(alike, index, BATCH))
            if other >= BATCH:
                begin = begin + BLOCK
            else:
                same = 1
                config = 0
                while config < count and same == 1:
                    a = sequence * CONFIGS + config
                    b = other * CONFIGS + config
                    depth = tl.load(stack_depth_ptr + a)
                    if tl.load(lexer_state_ptr + a) != tl.load(lexer_state_ptr + b):
                        same = 0
                    if depth != tl.load(stack_depth_ptr + b):
                        same = 0
                    place = tl.arange(0, STACK_STRIDE)
                    left = tl.load(
                        stack_ptr + a * STACK_STRIDE + place,
                        mask=place < depth,
                        other=0,
                    )
                    right = tl.load(
                        stack_ptr + b * STACK_STRIDE + place,
                        mask=place < depth,
                        other=0,
                    )
                    if tl.sum(tl.where(left != right, 1, 0)) != 0:
                        same = 0
                    config = config + 1
                if same == 1:
                    neighbour = other
                else:
                    begin = other + 1
    tl.store(representative_ptr + sequence, neighbour)

    # Only what computes has to start empty; everything else is written whole
    # by the copy. Clearing all of them was 9.7 MB a step to make room for
    # 0.6 MB of answers.
    computes = (found < 0) & (neighbour == sequence)
    if computes:
        for start in range(0, mask_words, BLOCK):
            place = start + tl.arange(0, BLOCK)
            tl.store(
                mask_ptr + sequence * mask_words + place,
                tl.zeros((BLOCK,), tl.int32),
                mask=place < mask_words,
            )

    # Seed the floors the sweep reduces.
    seed = 0
    while seed < count:
        row = sequence * CONFIGS + seed
        tl.store(row_floor_ptr + row, tl.load(stack_depth_ptr + row))
        seed = seed + 1

    tl.store(
        memo_store_ptr + sequence,
        tl.where(computes & (count <= MEMO_CONFIGS), 1, -1),
    )


@triton.jit
def _claim_kernel(
    stack_depth_ptr,
    config_count_ptr,
    row_floor_ptr,
    memo_store_ptr,
    memo_want_ptr,
    CONFIGS: tl.constexpr,
    MEMO_STRIDE: tl.constexpr,
    SUFFIXES: tl.constexpr,
):
    """How much of the stack each sequence's mask turned out to depend on.

    The sweep reduces `row_floor` to the lowest entry any group looked at, so
    `depth - floor + 1` is what the answer needs. Within the suffix bound the
    entry is keyed on that much and will match any stack agreeing there, which
    is what lets a nesting document hit; past it the whole stack is the key.

    Separate from the store because the store has to know what *other*
    sequences chose in order to give each slot one writer, and a kernel cannot
    read its siblings' writes.
    """
    sequence = tl.program_id(0)
    want = -1
    keep = tl.load(memo_store_ptr + sequence) >= 0
    if keep:
        count = tl.load(config_count_ptr + sequence)
        need = 1
        config = 0
        while config < count:
            row = sequence * CONFIGS + config
            depth = tl.load(stack_depth_ptr + row)
            need = tl.maximum(need, depth - tl.load(row_floor_ptr + row) + 1)
            if depth > MEMO_STRIDE:
                keep = False
            config = config + 1
        if need <= SUFFIXES:
            want = need
    tl.store(memo_want_ptr + sequence, tl.where(keep, want, -2))


@triton.jit
def _copy_kernel(
    memo_slot_ptr,
    representative_ptr,
    memo_mask_ptr,
    mask_ptr,
    mask_words,
    BLOCK: tl.constexpr,
):
    """Give every sequence that did not compute the mask it was promised.

    Two sources and one pass, because the probe already decided which: a table
    slot for a state an earlier step masked, or a neighbour's row for one this
    step is masking anyway. These were two kernels, and the difference between
    them was never more than where the bytes were.

    Two dimensions, because this is a copy and copies are bandwidth. One
    program per sequence walked a whole 19 KiB row in chunks and left a batch
    of 32 running 32 programs on 108 multiprocessors.

    Before the store, so a sequence reading a slot reads the entry it matched
    rather than one a provider has since replaced.
    """
    sequence = tl.program_id(0)
    slot = tl.load(memo_slot_ptr + sequence)
    source = tl.load(representative_ptr + sequence)
    if (slot >= 0) or (source != sequence):
        lane = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        live = lane < mask_words
        if slot >= 0:
            value = tl.load(
                memo_mask_ptr + slot * mask_words + lane, mask=live, other=0
            )
        else:
            value = tl.load(mask_ptr + source * mask_words + lane, mask=live, other=0)
        tl.store(mask_ptr + sequence * mask_words + lane, value, mask=live)


@triton.jit
def _store_kernel(
    lexer_state_ptr,
    stack_ptr,
    stack_depth_ptr,
    config_count_ptr,
    grammar_ptr,
    hash_ptr,
    representative_ptr,
    memo_want_ptr,
    suffix_hash_ptr,
    memo_read_ptr,
    memo_hash_ptr,
    memo_lexer_ptr,
    memo_stack_ptr,
    memo_depth_ptr,
    memo_count_ptr,
    memo_grammar_ptr,
    memo_mask_ptr,
    mask_ptr,
    mask_words,
    CONFIGS: tl.constexpr,
    STACK_STRIDE: tl.constexpr,
    SLOTS: tl.constexpr,
    MEMO_CONFIGS: tl.constexpr,
    MEMO_STRIDE: tl.constexpr,
    SUFFIXES: tl.constexpr,
    BATCH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Remember a mask this step had to compute.

    Written after the scatter, so what is stored is the finished row. Only a
    representative that missed is stored - a hit is already there, and a
    duplicate has not computed anything.

    A state too wide or too deep for an entry is simply not remembered. The
    bound keeps the table small enough to be worth having, and the states that
    exceed it are rare; leaving them out costs a recomputation rather than a
    wrong answer.
    """
    sequence = tl.program_id(0)
    count = tl.load(config_count_ptr + sequence)
    want = tl.load(memo_want_ptr + sequence)
    wants = want != -2
    digest = tl.load(hash_ptr + sequence)
    if want > 0:
        digest = tl.load(suffix_hash_ptr + sequence * SUFFIXES + want - 1)
    slot = (digest & 0x7FFFFFFF) % SLOTS

    # One writer per slot. Two sequences whose fingerprints collide would
    # otherwise interleave and leave an entry holding one state and the
    # other's mask, which a later probe matches and hands back. Decided by a
    # scan rather than an atomic so the answer does not depend on block order.
    if wants:
        # A block of candidates at a time. Walking them one by one is quadratic
        # in the batch, and it showed the moment it was written that way: the
        # fill went from 40 to 148 us at batch 32 on the schema with the widest
        # parses.
        lane = tl.arange(0, BATCH)
        live = lane < sequence
        theirs_k = tl.load(memo_want_ptr + lane, mask=live, other=-2)
        live = live & (theirs_k != -2)
        theirs = tl.load(hash_ptr + lane, mask=live, other=0)
        by_suffix = tl.load(
            suffix_hash_ptr + lane * SUFFIXES + tl.maximum(theirs_k, 1) - 1,
            mask=live & (theirs_k > 0),
            other=0,
        )
        theirs = tl.where(theirs_k > 0, by_suffix, theirs)
        rival = live & (((theirs & 0x7FFFFFFF) % SLOTS) == slot)
        if tl.sum(rival.to(tl.int32)) != 0:
            wants = False

    if wants:
        if tl.program_id(1) == 0:
            config = 0
            while config < count:
                row = sequence * CONFIGS + config
                held = slot * MEMO_CONFIGS + config
                depth = tl.load(stack_depth_ptr + row)
                kept = depth
                if want > 0:
                    kept = tl.minimum(depth, want)
                tl.store(memo_lexer_ptr + held, tl.load(lexer_state_ptr + row))
                tl.store(memo_depth_ptr + held, kept)
                lane = tl.arange(0, STACK_STRIDE)
                live = (lane < kept) & (lane < MEMO_STRIDE)
                tl.store(
                    memo_stack_ptr + held * MEMO_STRIDE + lane,
                    tl.load(
                        stack_ptr + row * STACK_STRIDE + depth - kept + lane,
                        mask=live,
                        other=0,
                    ),
                    mask=live,
                )
                config = config + 1
            tl.store(memo_count_ptr + slot, count)
            tl.store(memo_grammar_ptr + slot, tl.load(grammar_ptr + sequence))
            tl.store(memo_read_ptr + slot, want)
            # The fingerprint last, so a reader that sees it finds the rest of
            # the entry already written.
            tl.debug_barrier()
            tl.store(memo_hash_ptr + slot, digest)

        lane = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        live = lane < mask_words
        value = tl.load(mask_ptr + sequence * mask_words + lane, mask=live, other=0)
        tl.store(memo_mask_ptr + slot * mask_words + lane, value, mask=live)


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
    "action_extra_offsets": _B_ACTION_EXTRA_OFFSETS,
    "action_extra": _B_ACTION_EXTRA,
    "verdict_offsets": _B_VERDICT_OFFSETS,
    "verdicts": _B_VERDICTS,
    "verdict_stride": _B_VERDICT_STRIDE,
}


def _blocks(raw: bytes, starts, lengths):
    """Which distinct blocks an offset array points at, by content.

    Returns one digest and one span per distinct block, and which block each
    slot wants. The *bodies* are not built here: a block that some other
    grammar already holds is never copied, and building every one of them
    would be the cost this exists to avoid. Digested straight off the raw
    buffer for the same reason - a numpy slice and a cast per block doubled a
    compile, on the path a request waits on.
    """
    at = np.asarray(starts, dtype=np.int64)
    size = np.asarray(lengths, dtype=np.int64)
    view = memoryview(raw)
    digests: list[bytes] = []
    spans: list[tuple[int, int]] = []
    of_slot = np.zeros(at.size, dtype=np.int64)
    seen: dict[bytes, int] = {}
    for slot, (block_at, block_size) in enumerate(zip(at.tolist(), size.tolist())):
        digest = hashlib.blake2b(
            view[4 * block_at : 4 * (block_at + block_size)], digest_size=16
        ).digest()
        found = seen.get(digest)
        if found is None:
            found = len(digests)
            seen[digest] = found
            digests.append(digest)
            spans.append((block_at, block_size))
        of_slot[slot] = found
    return digests, spans, of_slot


@dataclass
class ResidentTables:
    """A grammar as the pool wants it: arrays to upload and ceilings to raise.

    Made once by `DeviceGrammar.prepare` and kept by whoever may need to admit
    the same grammar again. Under a memory budget that is the ordinary case -
    a schema no request is using is evicted and comes back when one arrives -
    and doing this work again each time is what made re-admission twelve times
    the cost of the copy it performs.
    """

    runs: dict
    vocab_size: int
    mask_words: int
    start_parser_state: int
    max_groups_per_state: int
    # Every group entry the grammar has, across all lexer states - the bound on
    # the global group index a sweep writes, and so the width of the per-row
    # record of which groups it has already been given.
    num_groups: int
    max_readings: int
    max_reading_terms: int
    nullable_chain: int
    window_bound: int
    paths: int
    longest_set: int
    has_verdicts: int
    # The tokenizer this was compiled against, not just its size. Defaulted
    # so a caller holding a `ResidentTables` from before this existed still
    # works; 0 means "unknown" and matches anything.
    vocabulary_digest: int = 0
    # The arrays whose blocks are shared across the pool, as content: for each
    # store, one digest and one body per distinct block, and which block each
    # group wants. Digested here because a digest of two megabytes is
    # milliseconds and admission happens far more often than compilation.
    shared: dict = dataclasses.field(default_factory=dict)

    @property
    def words(self) -> int:
        return sum(run.numel() for run in self.runs.values())


# One base per array, but the CSR offset arrays carry a sentinel element, so
# concatenating them shifts every following grammar by one. The base is
# whatever the concatenation actually produced, computed rather than derived.
_SIGNED = {"action_values", "action_extra"}

# Whether to keep the verdict shortcut resident. Off trades a replay per group
# for a quarter of the arena.
_VERDICTS = os.environ.get("ENGRAIN_VERDICTS", "1") != "0"

# Arrays whose blocks are shared across the pool, and the array that names
# where each block is. A pointer into a shared store is absolute, so the
# grammar's base for that store is zero.
_INTERNED = {
    "set_payload": "group_set_offset",
    "reading_index": "reading_offsets",
}

# Stores whose blocks carry their length in front of them. A block placed
# wherever it fits cannot say how long it is by where the next one starts, and
# `group_set_length` already says it for the token sets.
_PREFIXED = {"reading_index"}


class ConfigurationsExceeded(ValueError):
    """A parse carries more configurations at once than the batch was built for.

    Every per-configuration array is `batch x configurations x something`, and
    the something is another ceiling - so this factor is 26% of a batch's
    memory in `admitted` alone, and 100% of it in the sense that nothing else
    would shrink without it. Measured over a real step at batch 512 across
    forty-nine corpus schemas, a row carries 2.5 configurations at the mean and
    8 at the worst, against a ceiling of 128.

    So a caller may start small and grow, which is what this is for: it carries
    the width that was wanted so the batch can be rebuilt at it rather than at
    a guess.
    """

    def __init__(self, needed: int, limit: int) -> None:
        super().__init__(
            f"{needed} configurations exceeds the batch's limit of {limit}"
        )
        self.needed = needed
        self.limit = limit


class WindowTooWide(ValueError):
    """A grammar's replay window is wider than the pool will hold for it.

    The window is the part of a stack one reduction chain rewrites, and the
    batch allocates `batch x configurations x readings x window` for it -
    which is 95% of a batch's memory and the only factor there that a single
    grammar can multiply for every row. Measured over 116 corpus schemas the
    bound is 20 at the median and 32 at the p90, and then 349 at the p99: one
    schema in ten costs everyone else an eightfold buffer.

    So a grammar past the cap is not admitted, and the caller keeps it on the
    reference matcher - exact, and slower for that request alone rather than
    larger for all of them.
    """

    def __init__(self, needed: int, limit: int) -> None:
        super().__init__(
            f"a replay window of {needed} exceeds the pool's cap of {limit}"
        )
        self.needed = needed
        self.limit = limit


class StackTooDeep(ValueError):
    """A parse went deeper than the batch was built for.

    A run-time ceiling rather than an admission-time one: the depth a parse
    reaches is a property of the document, and a schema with an unbounded array
    reaches any depth given a long enough one. Carries the numbers so a caller
    can build a deeper batch and carry on, rather than parse them back out of a
    message or rescan every row to find out who is responsible - which at a
    serving batch costs more per step than the fill it is protecting.
    """

    def __init__(self, needed: int, limit: int) -> None:
        super().__init__(f"a stack of {needed} exceeds the batch's limit of {limit}")
        self.needed = needed
        self.limit = limit


class DeviceGrammar:
    """A pool of compiled grammars, resident on the GPU as one arena.

    A serving batch does not hold one grammar, and it does not hold a fixed set
    of them either. Requests arrive with their own schemas and leave when they
    are done, so the pool has to admit and release grammars while the engine is
    running - not be handed a list at construction.

    The tables are laid end to end and a sequence carries the index of the
    grammar it is under. Every lookup a kernel makes is into a run that starts
    at `bases[grammar, which array]`, so the arithmetic is one addition and no
    branch; the replay itself does not know there is more than one grammar. What
    has to be shared is the vocabulary, since the mask is a row over it - which
    is what a serving engine has anyway.

    Admission appends, which is why the arrays are kept with spare capacity: a
    tensor that has to grow is a new allocation at a new address, and a CUDA
    graph holds the address it recorded. Growth therefore bumps `revision`, and
    a batch re-records when it sees one it did not record against. Admitting
    into spare capacity does not, so the ordinary case does not disturb a
    running decode loop.
    """

    def __init__(
        self,
        compiled=None,
        max_stack: int = 256,
        window_cap: int | None = None,
        max_reductions: int | None = None,
        max_configs: int = 128,
        window: int | None = None,
        capacity: int = 16,
        budget_bytes: int | None = None,
    ):
        # Ceilings are policy, not properties of any one grammar, so they are
        # fixed here rather than derived from whatever the pool happens to hold.
        # A grammar that needs more than the window allows is admitted anyway
        # and its replays raise the overflow flag, which is the honest failure:
        # visible rather than silent. See `problems()`.
        self.max_stack = max_stack
        # What a grammar's replay window may be before the pool refuses it.
        # `None` - the default - keeps every grammar and lets the widest of
        # them size the batch, which is what a library should do: refusing a
        # schema is the caller's decision, not the compiler's. A serving
        # backend has the opposite priority and asks for a cap, because 95% of
        # a batch's memory is behind this number and over 116 corpus schemas it
        # is 20 at the median, 32 at the p90 and 349 at the p99.
        self.window_cap = window_cap
        self.max_reductions = (
            max_reductions if max_reductions is not None else max_stack
        )
        # Matches the reference matcher. Dropping configurations can only make a
        # parser stricter, so a ceiling below the matcher's is a source of masks
        # that are narrower than the grammar allows - and sixteen was reached at
        # 17.7% of the corpus's steps, in 36% of its documents. It is affordable
        # because the sweep enumerates the configurations that exist rather than
        # the ceiling: 128 against 16 costs 340 MB against 52 at batch 512, and
        # 70 us against 52.
        self.max_configs = max_configs
        self._forced_window = window

        self.count = 0
        self.revision = 0
        # Which grammars occupy which slots, as a counter. Distinct from
        # `revision`, which says the arrays have *moved* and a recorded graph is
        # stale; this says a slot has changed hands and anything keyed on a
        # grammar identifier is stale. Admitting into spare capacity moves this
        # and deliberately does not move `revision`, because forcing a graph
        # re-record on every arriving request is what residency exists to avoid.
        self.tenancy = 0
        self.vocab_size = 0
        self.vocabulary_digest = 0
        # Every allocation below says `device="cuda"`, which means the *current*
        # device rather than a fixed one. That is right for one process to one
        # GPU, which is how a serving engine runs its ranks - and wrong the
        # moment a compiler thread has switched the current device, since the
        # arena would gain an array on another card. Recorded here and made
        # current around anything that allocates or launches.
        self.device = torch.device("cuda", torch.cuda.current_device())
        self.mask_words = 0
        self.window = window or 8
        self.window_bound = 0
        # Derivations a replay follows. One where nothing conflicts, which is
        # every grammar that compiled before the tables kept conflicts, and the
        # path loop is then a single iteration around unchanged code.
        self.paths = 1
        # Whether every grammar in the pool carries the precomputed verdicts. A
        # pool is a mixture, and a kernel is compiled once for it, so one
        # grammar too large for the table turns the shortcut off for all - the
        # alternative is a branch per item on a value the compiler cannot see.
        self.has_verdicts = 1
        self.search_steps = 2
        self.max_groups_per_state = 1
        self.num_groups = 1
        self.max_readings = 1
        self.max_reading_terms = 1
        self.nullable_chain = 0
        self.start_parser_states: list[int] = []
        self._live: list[bool] = []
        self._extent: list[dict[str, tuple[int, int]]] = []
        self._dead_words = 0

        # What the tables may occupy, capacity included. Past it a grammar
        # nothing is running under is evicted to make room. None means the pool
        # grows until the allocator says no, which is what a benchmark wants and
        # not what a serving engine does.
        self.budget_bytes = budget_bytes
        self.evictions = 0
        self.admissions = 0

        self._capacity = max(1, capacity)
        # A bump pointer with wholesale compaction was enough while grammars
        # only arrived. Under continuous batching they leave too, and
        # compaction renumbers every survivor and bumps `revision`, which
        # re-records every CUDA graph in the engine. So each array carries a
        # free list instead: a released run goes back to it and the next
        # admission takes it, nothing moves, and no identifier changes.
        self._used = {name: 0 for name in _ARENA}
        self._free = {name: [] for name in _ARENA}
        self._free_ids: list[int] = []
        # The shared stores: array -> digest -> [offset, words, holders]. Every
        # grammar in a pool is compiled against the same tokenizer, so a block
        # two of them hold is the same words twice; here it is the same words
        # once, and a group points at it from wherever it is wanted.
        self._interned: dict[str, dict[bytes, list[int]]] = {}
        self._interned_of: dict[int, dict[str, tuple]] = {}
        self._stamp = 0
        self._used_at: list[int] = []
        self._pinned: list[int] = []
        # Bumped every time a slot is admitted into. A holder that kept an
        # identifier across an eviction would otherwise be masked against
        # whatever grammar took the slot, which is a wrong mask that looks like
        # a working one. See `holds`.
        self._generation: list[int] = []
        for name in _ARENA:
            setattr(self, name, torch.zeros(1024, dtype=torch.int32, device="cuda"))
        self.bases = torch.zeros(
            self._capacity * int(_NBASES.value), dtype=torch.int32, device="cuda"
        )

        if compiled is not None:
            for item in compiled if isinstance(compiled, (list, tuple)) else [compiled]:
                self.admit(item)

    @property
    def start_parser_state(self) -> int:
        return self.start_parser_states[0] if self.start_parser_states else 0

    def _reserve(self, name: str, extra: int) -> int:
        """Room for `extra` more words of `name`, growing if there is not.

        First fit in whatever released grammars left behind, and only past the
        high-water mark when nothing fits. Growing moves the array, which is
        the one thing here that invalidates a recorded graph, so it is also the
        one thing this tries to avoid.
        """
        if extra == 0:
            return self._used[name]
        holes = self._free[name]
        for index, (start, size) in enumerate(holes):
            if size >= extra:
                if size == extra:
                    holes.pop(index)
                else:
                    holes[index] = (start + extra, size - extra)
                return start
        held = getattr(self, name)
        at = self._used[name]
        if at + extra > held.numel():
            size = held.numel()
            while size < at + extra:
                size *= 2
            grown = torch.zeros(size, dtype=torch.int32, device="cuda")
            grown[:at] = held[:at]
            setattr(self, name, grown)
            self.revision += 1
        self._used[name] = at + extra
        return at

    def _intern(self, tables) -> dict:
        """Place this grammar's shared blocks and say where they landed.

        Returns the arrays whose contents depend on the placement - the offset
        array of each store, since an offset into a shared store is absolute
        rather than relative to a base - and the digests this grammar now holds
        a reference to.

        Misses are copied in one run rather than block by block: a grammar has
        thousands of distinct blocks, and thousands of small copies is the
        latency rather than the bytes.
        """
        out: dict[str, object] = {}
        held: dict[str, tuple] = {}
        for store, (digests, spans, of_slot) in tables.shared.items():
            prefix = 1 if store in _PREFIXED else 0
            source = tables.runs[store].numpy()
            table = self._interned.setdefault(store, {})
            placed = [0] * len(digests)
            missing = []
            for block, digest in enumerate(digests):
                entry = table.get(digest)
                if entry is None:
                    missing.append(block)
                else:
                    entry[2] += 1
                    placed[block] = entry[0]
            if missing:
                wanted = sum(spans[block][1] + prefix for block in missing)
                base = self._reserve(store, wanted)
                staged = np.empty(wanted, dtype=np.int32)
                cursor = 0
                for block in missing:
                    at, size = spans[block]
                    if prefix:
                        staged[cursor] = size
                    staged[cursor + prefix : cursor + prefix + size] = source[
                        at : at + size
                    ]
                    placed[block] = base + cursor
                    table[digests[block]] = [base + cursor, size + prefix, 1]
                    cursor += size + prefix
                getattr(self, store)[base : base + wanted] = torch.from_numpy(
                    staged
                ).cuda(non_blocking=False)
            pointer = _INTERNED[store]
            rewritten = np.asarray(placed, dtype=np.int64)[of_slot].astype(np.int32)
            # Kept the same length as the run it replaces. `reading_offsets`
            # carries a CSR sentinel nothing reads any more, and a run that
            # changed size would make the room reserved for it a guess.
            wide = tables.runs[pointer].numel()
            if rewritten.size < wide:
                rewritten = np.concatenate(
                    [rewritten, np.zeros(wide - rewritten.size, dtype=np.int32)]
                )
            out[pointer] = torch.from_numpy(rewritten)
            held[store] = (tuple(digests), of_slot)
        out["__held"] = held
        return out

    def _compact_shared(self, store: str, previous: torch.Tensor) -> None:
        """Rebuild a shared store densely and point every grammar at it again.

        A block in it is named by an absolute offset, so this is the one place
        a pointer array has to be written twice. Doing it here keeps
        `dead_fraction` honest: a store that only ever grew would make the
        number that decides whether to compact blind to the largest arrays in
        the arena.
        """
        table = self._interned.get(store, {})
        wanted = sum(size for _, size, _ in table.values())
        self._used[store] = 0
        self._free[store] = []
        setattr(
            self,
            store,
            torch.zeros(
                max(1024, int(wanted * 1.25)), dtype=torch.int32, device="cuda"
            ),
        )
        rebuilt = getattr(self, store)
        for entry in table.values():
            at, size, _ = entry
            to = self._reserve(store, size)
            rebuilt[to : to + size] = previous[at : at + size]
            entry[0] = to
        pointer = _INTERNED[store]
        for identifier, holds in self._interned_of.items():
            digests, of_slot = holds[store]
            placed = np.asarray(
                [table[digest][0] for digest in digests], dtype=np.int64
            )
            at, size = self._extent[identifier][pointer]
            rewritten = placed[of_slot].astype(np.int32)
            if rewritten.size < size:
                rewritten = np.concatenate(
                    [rewritten, np.zeros(size - rewritten.size, dtype=np.int32)]
                )
            getattr(self, pointer)[at : at + size] = torch.from_numpy(rewritten).cuda()

    def _return(self, name: str, at: int, size: int) -> None:
        """Give a run back, joined to whatever it now touches.

        Coalescing is what keeps a pool that has churned for hours from holding
        its memory in pieces too small to admit anything into.
        """
        if size == 0:
            return
        holes = self._free[name]
        holes.append((at, size))
        holes.sort()
        merged: list[tuple[int, int]] = []
        for start, length in holes:
            if merged and merged[-1][0] + merged[-1][1] == start:
                previous, held = merged[-1]
                merged[-1] = (previous, held + length)
            else:
                merged.append((start, length))
        # A hole against the high-water mark is not a hole, it is the mark in
        # the wrong place.
        while merged and merged[-1][0] + merged[-1][1] == self._used[name]:
            start, length = merged.pop()
            self._used[name] = start
        self._free[name] = merged

    def _room_for(self, sizes: dict[str, int]) -> bool:
        """Would admitting a grammar of these sizes stay inside the budget?"""
        if self.budget_bytes is None:
            return True
        wanted = 0
        for name, size in sizes.items():
            held = getattr(self, name).numel()
            if any(hole >= size for _, hole in self._free[name]):
                continue
            need = self._used[name] + size
            while held < need:
                held *= 2
            wanted += held - getattr(self, name).numel()
        return self.resident_bytes() + wanted * 4 <= self.budget_bytes

    def _evict_for(self, sizes: dict[str, int]) -> None:
        """Release grammars nothing is running under until this one fits.

        Least recently used first, and never one a sequence is still under -
        that is what `pin` says. A pool that cannot evict enough admits anyway
        and goes over budget, because refusing to admit a grammar is a refused
        request and being over budget is not.
        """
        while not self._room_for(sizes):
            victims = [
                index
                for index in range(len(self._live))
                if self._live[index] and not self._pinned[index]
            ]
            if not victims:
                return
            self.release(min(victims, key=lambda index: self._used_at[index]))
            self.evictions += 1

    def pin(self, identifier: int) -> None:
        """Say a sequence is running under this grammar, so it cannot be evicted."""
        self._pinned[identifier] += 1

    def unpin(self, identifier: int) -> None:
        """Say one sequence has finished with it.

        The eviction order is stamped here rather than at every step. Only
        unpinned grammars are ever evicted, so what orders them is when they
        stopped being used - and reading that off the last unpin costs nothing,
        where stamping a batch of 512 every step would be 512 lines of Python
        on the path this design exists to keep clear.
        """
        if self._pinned[identifier] > 0:
            self._pinned[identifier] -= 1
            if self._pinned[identifier] == 0:
                self._stamp += 1
                self._used_at[identifier] = self._stamp

    def holds(self, identifier: int, generation: int) -> bool:
        """Is this still the grammar that was admitted into this slot?

        A slot freed by an eviction is reused, so an identifier alone does not
        identify a grammar across one. Whoever cached an identifier has to ask.
        """
        return (
            0 <= identifier < len(self._live)
            and self._live[identifier]
            and self._generation[identifier] == generation
        )

    def generation(self, identifier: int) -> int:
        return self._generation[identifier]

    def is_live(self, identifier: int) -> bool:
        return 0 <= identifier < len(self._live) and self._live[identifier]

    @staticmethod
    def prepare(compiled) -> ResidentTables:
        """Everything about a grammar that does not depend on where it lands.

        Separated from `admit` because an evicted grammar comes back. Doing
        this at every admission made re-admission 8.2 ms where the device copy
        it exists to perform is 0.7 - the rest was materialising arrays from
        Rust and recomputing ceilings that had not changed. Held in pinned host
        memory, which is not the scarce resource here: the whole point of a
        budget is that device memory is.
        """
        arrays = compiled.device_arrays()
        runs = {}
        for name in _ARENA:
            dtype = np.int32 if name in _SIGNED else np.uint32
            run = np.frombuffer(arrays[name], dtype=dtype).astype(np.int32)
            # The verdict table is a shortcut, not an answer: it says whether a
            # (lexer state, parser state) pair admits a group without replaying
            # the group's readings, and the kernel does the replay anyway where
            # it is absent. It is also a quarter of the arena, being lexer
            # states times parser states times groups - so a pool with more
            # schemas than room is offered the trade the table exists to make.
            if (
                name in ("verdicts", "verdict_stride", "verdict_offsets")
                and not _VERDICTS
            ):
                run = run[:0]
            runs[name] = torch.from_numpy(run).pin_memory()

        def widest(name):
            values = np.frombuffer(arrays[name], dtype=np.uint32)
            return int(np.diff(values).max()) if values.size > 1 else 1

        lengths = np.frombuffer(arrays["group_set_length"], dtype=np.uint32)
        nullable = _nullable_chain(arrays)
        # Every grammar in a pool is compiled against one tokenizer, so a block
        # two of them hold is the same words twice. Measured over the corpus,
        # `reading_index` repeats 5.06x - a group's list of ways to read its
        # tokens is the same list in state after state - and `set_payload`
        # 1.56x. Both are stored once here and pointed at from wherever they
        # are wanted, which costs nothing at all to run.
        starts = np.frombuffer(arrays["reading_offsets"], dtype=np.uint32)
        shared = {
            "set_payload": _blocks(
                arrays["set_payload"],
                np.frombuffer(arrays["group_set_offset"], dtype=np.uint32),
                lengths,
            ),
            # Length-prefixed, because a shared block cannot say how long it is
            # by where the next one starts. One word per *distinct* block is
            # cheaper than one per group, and it keeps the arena's array count
            # where it was.
            "reading_index": _blocks(
                arrays["reading_index"], starts[:-1], np.diff(starts)
            ),
        }
        return ResidentTables(
            shared=shared,
            runs=runs,
            vocab_size=int(arrays["vocab_size"]),
            mask_words=int(arrays["bitset_words"]),
            start_parser_state=int(arrays["start_parser_state"]),
            max_groups_per_state=widest("group_offsets"),
            num_groups=int(
                np.frombuffer(arrays["group_set_kind"], dtype=np.uint32).size
            ),
            max_readings=max(1, int(np.diff(starts).max()) if starts.size > 1 else 1),
            max_reading_terms=widest("reading_term_offsets"),
            nullable_chain=nullable,
            window_bound=_window_bound(arrays, nullable),
            paths=min(_MAX_PATHS, max(1, int(arrays.get("max_actions", 1)))),
            longest_set=int(lengths.max()) if lengths.size else 1,
            has_verdicts=1 if len(arrays["verdicts"]) else 0,
            vocabulary_digest=getattr(compiled, "vocabulary_digest", 0),
        )

    def admit(self, compiled) -> int:
        """Add one compiled grammar to the pool and return the id to use for it.

        Takes either a compiled grammar or what `prepare` made of one. A
        serving engine wanting cheap re-admission after an eviction keeps the
        latter.
        """
        with torch.cuda.device(self.device):
            return self._admit(compiled)

    def _admit(self, compiled) -> int:
        tables = (
            compiled if isinstance(compiled, ResidentTables) else self.prepare(compiled)
        )
        # Before anything is written, because a refusal must leave the pool as
        # it was. The window is the one ceiling a single grammar can multiply
        # for every row of every batch, and 95% of a batch's memory is behind
        # it.
        if self.window_cap is not None and tables.window_bound > self.window_cap:
            raise WindowTooWide(tables.window_bound, self.window_cap)
        if self.count == 0 and not self.vocab_size:
            self.vocab_size = tables.vocab_size
            self.mask_words = tables.mask_words
            self.vocabulary_digest = tables.vocabulary_digest
        elif tables.vocab_size != self.vocab_size:
            raise ValueError("grammars in one pool must share a vocabulary")
        elif (
            tables.vocabulary_digest
            and self.vocabulary_digest
            and tables.vocabulary_digest != self.vocabulary_digest
        ):
            # Same size is not the same tokenizer. A grammar's groups are token
            # ids, so one compiled against a different ordering yields a mask
            # that is wrong token by token, with nothing in the parse to notice
            # - the one failure mode no verification downstream can catch.
            raise ValueError(
                "this grammar was compiled against a different vocabulary "
                "from the rest of the pool"
            )

        runs = tables.runs
        # Room is made before anything is written, because reserving array by
        # array and running out halfway would leave a grammar half in the pool.
        self._evict_for({name: run.numel() for name, run in runs.items()})

        # An identifier freed by an eviction is reused rather than renumbered.
        # Renumbering is what `compact` does and why it is a last resort: every
        # holder of a `grammar_of` has to be told, and every recorded graph
        # dies. Under continuous batching that would be every few requests.
        if self._free_ids:
            identifier = self._free_ids.pop(0)
        else:
            identifier = len(self._live)
            self._live.append(False)
            self._extent.append({})
            self.start_parser_states.append(0)
            self._used_at.append(0)
            self._pinned.append(0)
            self._generation.append(0)
        if identifier >= self._capacity:
            while identifier >= self._capacity:
                self._capacity *= 2
            grown = torch.zeros(
                self._capacity * int(_NBASES.value), dtype=torch.int32, device="cuda"
            )
            grown[: self.bases.numel()] = self.bases
            self.bases = grown
            self.revision += 1

        rows = np.zeros(int(_NBASES.value), dtype=np.int32)
        extent: dict[str, tuple[int, int]] = {}
        # `set_payload` is shared, so its base is zero and a group's offset is
        # absolute. Everything else is a per-grammar run at a base.
        held = self._intern(tables) if tables.shared else None
        for name, slot in _ARENA.items():
            if held is not None and name in _INTERNED:
                rows[int(slot.value)] = 0
                continue
            run = held.get(name, runs[name]) if held is not None else runs[name]
            size = run.numel()
            at = self._reserve(name, size)
            getattr(self, name)[at : at + size] = run.cuda(non_blocking=True)
            rows[int(slot.value)] = at
            extent[name] = (at, size)
        self.bases[
            identifier * int(_NBASES.value) : (identifier + 1) * int(_NBASES.value)
        ] = torch.from_numpy(rows).cuda()

        self.max_groups_per_state = max(
            self.max_groups_per_state, tables.max_groups_per_state
        )
        self.num_groups = max(self.num_groups, tables.num_groups)
        self.max_readings = max(self.max_readings, tables.max_readings)
        self.max_reading_terms = max(self.max_reading_terms, tables.max_reading_terms)
        self.nullable_chain = max(self.nullable_chain, tables.nullable_chain)
        # A conflicted cell holds up to `max_actions` actions and a reading
        # meets several, so the product is what enumerating them all would
        # cost. Bounded at the reference matcher's own bound: past it both
        # refuse the same derivations, which is what keeps them in step.
        # One grammar too large for a verdict table turns the shortcut off for
        # the whole pool: the kernel is compiled once for a pool and a mixture
        # would otherwise index a table that is not there. It went in with the
        # assignment inverted, which held until a corpus schema exceeded the
        # budget and the kernel read an empty array.
        if tables.has_verdicts == 0 and self.has_verdicts == 1:
            self.has_verdicts = 0
            self.revision += 1
        wanted_paths = tables.paths
        if wanted_paths > self.paths:
            self.paths = wanted_paths
            self.revision += 1
        self.window_bound = max(self.window_bound, tables.window_bound)
        longest = tables.longest_set
        # The lanes of a block search in lockstep, so the loop runs a fixed
        # number of times and every lane must have finished by it. A halving
        # over `n` needs ceil(log2(n)) steps; the margin covers the ends being
        # inclusive, and a bound tight enough for the scalar case left five
        # schemas disagreeing with the reference matcher.
        steps = max(2, int(np.ceil(np.log2(longest + 2))) + 2)
        if steps > self.search_steps:
            self.search_steps = steps
            self.revision += 1
        if self._forced_window is None:
            wanted = min(
                self.max_stack,
                1 << max(3, int(np.ceil(np.log2(max(self.window_bound, 2))))),
            )
            if wanted > self.window:
                self.window = wanted
                self.revision += 1

        if held is not None:
            self._interned_of[identifier] = held["__held"]
        self.start_parser_states[identifier] = tables.start_parser_state
        self._live[identifier] = True
        self._extent[identifier] = extent
        self._pinned[identifier] = 0
        self._generation[identifier] += 1
        self._stamp += 1
        self._used_at[identifier] = self._stamp
        self.count += 1
        self.tenancy += 1
        self.admissions += 1
        return identifier

    def release(self, identifier: int) -> None:
        """Say that nothing is under this grammar any more.

        The space *is* reclaimed, into each array's free list, and the
        identifier goes back too. Nothing moves and nothing is renumbered, so a
        recorded graph is still valid - which is the property that makes
        admitting and evicting affordable at the rate requests arrive at.

        What a release cannot lower is a ceiling: the window, the readings a
        group can have, the paths a replay follows. Those are maxima over every
        grammar the pool has *ever* held, because lowering one would resize the
        batch buffers a running step is reading. A pool that has seen a large
        grammar keeps its shape after it leaves.
        """
        if not self._live[identifier]:
            return
        self._live[identifier] = False
        self._pinned[identifier] = 0
        for name, (at, size) in self._extent[identifier].items():
            self._return(name, at, size)
        self._extent[identifier] = {}
        for store, (digests, _) in self._interned_of.pop(identifier, {}).items():
            table = self._interned[store]
            for digest in digests:
                entry = table[digest]
                entry[2] -= 1
                if entry[2] == 0:
                    self._return(store, entry[0], entry[1])
                    del table[digest]
        self._free_ids.append(identifier)
        self.count -= 1
        self.tenancy += 1

    @property
    def dead_fraction(self) -> float:
        """How much of the arena is holes rather than tables.

        Free space below the high-water mark, which a release put there and a
        later admission may or may not be able to use - a hole only takes a
        grammar whose run fits in it. This is the fragmentation `compact`
        exists to answer, and the number that says whether it is worth it.
        """
        total = sum(self._used.values())
        holes = sum(size for name in _ARENA for _, size in self._free[name])
        return holes / total if total else 0.0

    def compact(self) -> dict[int, int]:
        """Rebuild the arena around the grammars still in use.

        Returns the new id of each surviving grammar, since compaction renumbers
        them and whoever holds a `grammar_of` has to be told. Bumps `revision`,
        so any recorded graph is re-recorded.
        """
        keep = [index for index in range(len(self._live)) if self._live[index]]
        remap = {old: new for new, old in enumerate(keep)}
        held = {
            name: getattr(self, name)[: self._used[name]].clone() for name in _ARENA
        }
        old_bases = self.bases.reshape(-1, int(_NBASES.value)).clone()
        starts = [self.start_parser_states[index] for index in keep]
        extents = [self._extent[index] for index in keep]

        # Shrink to fit while rebuilding, with a little slack so the next few
        # admissions do not immediately grow it back. Compaction is the only
        # moment the capacity can come down: everywhere else a live batch may
        # be reading the arrays.
        # The shared store is renumbered like everything else, but a block in
        # it is pointed at by absolute offset from every grammar that admits
        # the same set - so moving one means rewriting those grammars'
        # `group_set_offset`, which is done below rather than through `rows`.
        shared = set(_INTERNED) if self._interned else set()
        for name in _ARENA:
            if name in shared:
                continue
            wanted = sum(extent[name][1] for extent in extents)
            self._used[name] = 0
            self._free[name] = []
            setattr(
                self,
                name,
                torch.zeros(
                    max(1024, int(wanted * 1.25)), dtype=torch.int32, device="cuda"
                ),
            )
        rows = np.zeros((max(len(keep), 1), int(_NBASES.value)), dtype=np.int32)
        for new, extent in enumerate(extents):
            for name, slot in _ARENA.items():
                if name in shared:
                    continue
                at, size = extent[name]
                to = self._reserve(name, size)
                getattr(self, name)[to : to + size] = held[name][at : at + size]
                rows[new, int(slot.value)] = to
        if keep:
            self.bases.reshape(-1, int(_NBASES.value))[: len(keep)] = torch.from_numpy(
                rows
            ).cuda()
        del old_bases

        self.count = len(keep)
        self.start_parser_states = starts
        self._live = [True] * len(keep)
        self._free_ids = []
        self._used_at = [self._used_at[index] for index in keep]
        self._pinned = [self._pinned[index] for index in keep]
        self._generation = [self._generation[index] + 1 for index in keep]
        self._extent = [
            {
                name: (int(rows[new, int(slot.value)]), extents[new][name][1])
                for name, slot in _ARENA.items()
                if name not in shared
            }
            for new in range(len(keep))
        ]
        self._interned_of = {
            remap[old]: entry
            for old, entry in self._interned_of.items()
            if old in remap
        }
        for store in shared:
            self._compact_shared(store, held[store])
        self.revision += 1
        self.tenancy += 1
        return remap

    def resident_bytes(self) -> int:
        """What the pool's tables occupy, capacity and all."""
        return (
            sum(getattr(self, name).numel() * 4 for name in _ARENA)
            + self.bases.numel() * 4
        )

    def used_bytes(self) -> int:
        """What the grammars in the pool actually take up."""
        return sum(self._used.values()) * 4

    # The arena as the CUDA kernels want it: one struct of pointers instead of
    # twenty-six kernel arguments. Built from `_ARENA` so that the two backends
    # cannot disagree about the order, and rebuilt when `revision` moves -
    # which is exactly when an array has been reallocated and the addresses in
    # it are stale, the same signal a recorded graph uses.
    _ARENA_FIELDS = (*_ARENA, "bases")

    def arena_struct(self) -> torch.Tensor:
        """Device memory holding one `en::Arena`. Cached until the pool moves."""
        held = getattr(self, "_arena_struct", None)
        if held is not None and self._arena_struct_revision == self.revision:
            return held
        addresses = [getattr(self, name).data_ptr() for name in self._ARENA_FIELDS]
        packed = torch.tensor(addresses, dtype=torch.int64, device=self.device)
        self._arena_struct = packed
        self._arena_struct_revision = self.revision
        return packed

    @property
    def arena_slots(self) -> int:
        """How many pointers `arena_struct` packs. Checked against the kernel."""
        return len(self._ARENA_FIELDS)

    @property
    def slots(self) -> int:
        """How many identifiers the pool has ever handed out.

        Not `count`, which is how many are *live*. An eviction frees a slot and
        decrements the count, but the identifier space does not shrink: a freed
        id goes on the free list to be reused, and a fresh one is allocated past
        the high-water mark. So after any eviction the largest live identifier
        can exceed `count` - measured at 29 against a count of 21 with forty
        schemas under an 8 MB budget - and validating an id against `count`
        rejects a perfectly live grammar. A workload of 425 real schemas under a
        table budget is what found it.
        """
        return len(self._live)

    def new_batch(self, batch: int, rollback: int = 0) -> DeviceBatch:
        return DeviceBatch(self, batch, rollback)

    def footprint(self, batch: int, rollback: int = 0) -> dict[str, int]:
        """Exactly what a batch of this size will allocate, before it is built.

        Every number this depends on is known once the grammars are admitted:
        `window` and `max_readings` are computed from each grammar's tables at
        admission, `mask_words` from the vocabulary, and `max_configs` is a
        policy the caller set. So a serving engine can be told what a batch
        costs when a request arrives rather than when the allocation fails -
        which is the difference between budgeting and discovering.

        Checked against the allocation itself by a test, because a prediction
        that drifts from what it predicts is worse than none.
        """
        rows = batch * self.max_configs
        # The same clamp the batch applies: a configuration offering more
        # candidates than the commit can keep has already said everything that
        # can be kept.
        slots = rows * min(self.max_readings * self.paths, self.max_configs)
        words = self.mask_words or (self.vocab_size + 31) // 32
        memo_configs = min(self.max_configs, _MEMO_CONFIGS)
        memo_stride = min(self.max_stack, _MEMO_DEPTH)
        memo_slots = _memo_slots(words * 4 + memo_configs * (8 + memo_stride * 4) + 16)
        blocks = _sweep_blocks(batch)
        replayers = max(
            blocks,
            (blocks + _CANDIDATE_THREADS - 1)
            // _CANDIDATE_THREADS
            * _CANDIDATE_THREADS,
        )
        sizes = {
            "admitted": rows * self.max_groups_per_state,
            "stack": rows * self.max_stack * 4,
            "old_stack": rows * self.max_stack * 4,
            # Two replay scratches, one for the sweep and one for the advance,
            # each two windows per replayer.
            "scratch": 2 * replayers * 2 * self.window * 4,
            "memo_mask": memo_slots * words * 4,
            "memo_stack": memo_slots * memo_configs * memo_stride * 4,
            "cand_window": batch * max(self.window, _CANDIDATE_ARENA) * 4,
            "cand_at": slots * 4,
            "cand_depth": slots * 4,
            "cand_floor": slots * 4,
            "cand_lexer": slots * 4,
            "mask": batch * words * 4,
            "group_given": rows * _GROUP_FILTER * 4,
            # Everything else is per row or per sequence and small beside these:
            # lexer states, depths, counts, offsets, the memo's own indices.
            "the rest": rows * 24 + batch * 32 + memo_slots * memo_configs * 8,
        }
        sizes["total"] = sum(sizes.values())
        return sizes


class DeviceBatch:
    """Per-sequence parser state, in device memory."""

    def __init__(self, grammar: DeviceGrammar, batch: int, rollback: int = 0):
        # See the note on `DeviceGrammar.device`: everything below allocates
        # against the current device, and the pool's arrays are on the pool's.
        self.device = grammar.device
        # Read once per batch rather than per step: a mode that changed under a
        # recorded graph would replay one backend while claiming another.
        self.backend = _chosen_backend()
        with torch.cuda.device(self.device):
            self._build(grammar, batch, rollback)

    def _build(self, grammar: DeviceGrammar, batch: int, rollback: int = 0):
        self.grammar = grammar
        self.batch = batch
        self.configs = grammar.max_configs
        self.graph: torch.cuda.CUDAGraph | None = None
        self.advance_graph: torch.cuda.CUDAGraph | None = None
        self.step_graph: torch.cuda.CUDAGraph | None = None
        # A whole draft walk as one recording. See `capture_draft`.
        self.draft_graph: torch.cuda.CUDAGraph | None = None
        self.draft_length = 0
        # Which shape of the pool the graphs were recorded against. Admitting a
        # grammar into spare capacity leaves this alone, but one that makes an
        # array grow moves it to a new address, and a graph holds the address it
        # recorded. Replaying then reads freed memory, which is a wrong mask if
        # it is anything at all.
        self.recorded = -1
        # The pool as it was when this batch's buffers were sized. Admitting a
        # grammar can raise a ceiling - the window, the widest state's group
        # count, the readings a group can have - and buffers sized against the
        # old ones are too small for the new. Silently, since a kernel indexes
        # what it is given.
        self.pool_revision = grammar.revision
        # And the ceilings themselves, because `revision` tracks where the
        # arrays *are* and these are how big they have to be - a grammar can
        # raise one without moving anything. Compared before every operation
        # by `_check_shape`, since the alternative is a kernel indexing past a
        # buffer with whatever it finds there.
        self.sized_for = self._ceilings()

        # A conflicted reading yields one candidate per surviving derivation,
        # so the slot count would be readings times paths - a product of two
        # ceilings, which on one schema was 1,104 slots per configuration and
        # 9.3 GB of window. The commit keeps at most `configs` configurations
        # in all, so a single configuration offering more than that has already
        # said everything that can be kept, and the rest is a ceiling being
        # paid for as though it were the work.
        readings = min(grammar.max_readings * grammar.paths, self.configs)
        self.max_readings = readings
        slots = batch * self.configs * readings
        # One count per (sequence, configuration), not one flag per slot. The
        # candidate kernel writes every row the commit will read, so nothing
        # here has to start at a known value - which is what removes a 2 MB
        # clear from every step at batch 32.
        self.cand_count = torch.zeros(
            batch * self.configs, dtype=torch.int32, device="cuda"
        )
        self.cand_lexer = torch.zeros(slots, dtype=torch.int32, device="cuda")
        self.cand_depth = torch.zeros(slots, dtype=torch.int32, device="cuda")
        self.cand_floor = torch.zeros(slots, dtype=torch.int32, device="cuda")
        # A budget per sequence, not a product of ceilings. The old shape was
        # `batch x configurations x readings x window` - four independent worst
        # cases that never co-occur - and over one real step at batch 512 it
        # reserved 8.00 GiB and wrote 0.01 MiB. Candidates are bumped into
        # their sequence's slice instead, and a sequence that runs out drops
        # the candidate and raises `overflow`, which is the narrowing signal a
        # caller already refills from the reference matcher.
        self.window_budget = max(grammar.window, _CANDIDATE_ARENA)
        self.cand_at = torch.zeros(slots, dtype=torch.int32, device="cuda")
        self.cand_used = torch.zeros(batch, dtype=torch.int32, device="cuda")
        self.cand_window = torch.zeros(
            batch * self.window_budget, dtype=torch.int32, device="cuda"
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
        # Masks this batch has already computed, keyed by the parse state that
        # produced them. Deduplication finds the sequences of one step that
        # agree; this finds the far larger overlap with the steps before, which
        # measures at 92% to 96% on this corpus and does not fall off at batch
        # one - a document keeps returning to the same configuration, and every
        # character inside a string is the same one.
        #
        # Direct-mapped, so a lookup is one load and eviction needs no policy.
        # An entry holds the state as well as the mask because the fingerprint
        # only narrows the search, exactly as in deduplication.
        # Both bounds follow the grammar. A parse that cannot be as wide or as
        # deep as the ceiling allows should not be charged for it, and every
        # byte here is one the table cannot spend on another entry.
        self.memo_configs = min(self.configs, _MEMO_CONFIGS)
        self.memo_stride = min(grammar.max_stack, _MEMO_DEPTH)
        self.memo_slots = _memo_slots(
            grammar.mask_words * 4 + self.memo_configs * (8 + self.memo_stride * 4) + 16
        )
        held = self.memo_slots * self.memo_configs
        self.memo_hash = torch.full(
            (self.memo_slots,), _MEMO_EMPTY, dtype=torch.int32, device="cuda"
        )
        self.memo_count = torch.zeros(self.memo_slots, dtype=torch.int32, device="cuda")
        self.memo_grammar = torch.zeros(
            self.memo_slots, dtype=torch.int32, device="cuda"
        )
        self.memo_lexer = torch.zeros(held, dtype=torch.int32, device="cuda")
        self.memo_depth = torch.zeros(held, dtype=torch.int32, device="cuda")
        self.memo_stack = torch.zeros(
            held * self.memo_stride, dtype=torch.int32, device="cuda"
        )
        self.memo_mask = torch.zeros(
            self.memo_slots * grammar.mask_words, dtype=torch.int32, device="cuda"
        )
        # How far down its own stack each configuration's mask actually looked.
        # Set to the depth before the sweep and reduced by it.
        self.row_floor = torch.zeros(
            batch * self.configs, dtype=torch.int32, device="cuda"
        )
        self.suffix_hash = torch.zeros(
            batch * _MEMO_SUFFIXES, dtype=torch.int32, device="cuda"
        )
        # What each entry was keyed on: `k` entries from the top, or -1 for the
        # whole stack.
        self.memo_read = torch.zeros(self.memo_slots, dtype=torch.int32, device="cuda")
        self.memo_slot = torch.full((batch,), -1, dtype=torch.int32, device="cuda")
        self.memo_store = torch.full((batch,), -1, dtype=torch.int32, device="cuda")
        self.memo_want = torch.full((batch,), -2, dtype=torch.int32, device="cuda")
        self.memo_tenancy = grammar.tenancy
        # Which grammar each sequence is under. A serving batch mixes them, and
        # everything else in the step reads this to find its tables.
        self.grammar_of = torch.zeros(batch, dtype=torch.int32, device="cuda")
        # Nothing has said which grammar each sequence is under yet, and
        # zero is a real identifier - so a fill before `set_grammars` or
        # `set_batch_configurations` would mask every sequence against
        # whichever grammar happens to hold slot 0.
        self.assigned = False
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
        # Swept again once the small kernels were out of the way: 256 blocks is
        # 180 us at batch 32 and 516 at 512, 2,048 is 103 and 175, and 4,096 is
        # 103 and 149. Past that batch 32 loses more to the launch than batch
        # 512 gains. The blocks are not the floor; the items are.
        self.sweep_blocks = _sweep_blocks(batch)
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
        # One bit per group entry per row: has this row already been given this
        # group's tokens? A group admitted by several configurations writes the
        # same tokens each time and the row is a union, so only the first write
        # is work. Measured on the corpus schema that forks hardest, sixty-four
        # configurations produce four distinct rows and write sixty times the
        # bits that end up set.
        # Slots in each row's filter, a power of two so the probe can mask.
        # Not a bit per group: `num_groups` is the largest schema the pool
        # holds, so that shape was 5.16 GiB at batch 512 over sixty-four corpus
        # schemas and could not be built at all at 1,024. A row admits a
        # handful of groups, so a fixed set holds them, and a collision costs
        # one repeated union rather than a wrong answer.
        self.group_words = _GROUP_FILTER
        self.group_given = torch.zeros(
            rows * self.group_words, dtype=torch.int32, device="cuda"
        )
        self.counts = torch.zeros(rows, dtype=torch.int32, device="cuda")
        self.work_offsets = torch.zeros(rows + 1, dtype=torch.int32, device="cuda")
        self.live_counts = torch.zeros(rows, dtype=torch.int32, device="cuda")
        self.live_offsets = torch.zeros(rows + 1, dtype=torch.int32, device="cuda")
        replayers = max(
            self.sweep_blocks,
            (self.sweep_blocks + _CANDIDATE_THREADS - 1)
            // _CANDIDATE_THREADS
            * _CANDIDATE_THREADS,
        )
        # Two windows per block: one for the reading being replayed, one for
        # probing what a pending lexeme could still become. Per *block*, not per
        # program - which is the point of the sweep. This used to be one per
        # (sequence, configuration, group) and reach 1.7 GB at batch 512.
        # Rounded up to the CUDA launch's thread count, for the same reason as
        # `advance_scratch`: that backend gives a replay to a thread.
        self.scratch = torch.zeros(
            (replayers, 2 * grammar.window), dtype=torch.int32, device="cuda"
        )
        # The advance indexes its scratch by block, like the sweep, so it too
        # stops depending on the batch. It cannot share the sweep's buffer:
        # both may be in flight on the same stream and they would write over
        # each other's replays.
        #
        # Rounded up to the CUDA launch's thread count, because that backend
        # gives a *thread* a replay rather than a block - which is the 3.42x
        # lever - and a grid of `ceil(blocks/threads) * threads` is a little
        # wider than `blocks`. Sizing for the smaller of the two would let the
        # last few threads write past the end.
        self.advance_scratch = torch.zeros(
            (replayers, 2 * grammar.window),
            dtype=torch.int32,
            device="cuda",
        )
        self.found = torch.full((rows,), _NO_GROUP, dtype=torch.int32, device="cuda")

        # How many steps of parse state to keep so they can be undone.
        #
        # Zero by default because it is not free: one kept step is the same size
        # as the live state, which at batch 512 with 128 configurations is 67 MB.
        # Speculative decoding is what needs it - advance through a draft, then
        # keep only the prefix the model accepted - and a decode loop that does
        # not speculate should not pay for it.
        self.rollback_depth = rollback
        self.history_length = 0
        # The fused advance writes the entry itself, so it takes these pointers
        # whether or not there is a ring behind them. A single word each when
        # there is not, so the launch has one shape rather than two.
        held = rollback if rollback > 0 else 0
        self.hist_lexer = torch.zeros(
            max(1, held * rows), dtype=torch.int32, device="cuda"
        )
        self.hist_depth = torch.ones(
            max(1, held * rows), dtype=torch.int32, device="cuda"
        )
        self.hist_stack = torch.zeros(
            max(1, held * rows * grammar.max_stack), dtype=torch.int32, device="cuda"
        )
        self.hist_count = torch.ones(
            max(1, held * batch), dtype=torch.int32, device="cuda"
        )
        # On the device, so that a captured advance stays valid: a graph
        # records the arguments it was launched with, and a slot passed as a
        # scalar would be frozen at whatever it was when it was recorded.
        self.hist_slot = torch.zeros(1, dtype=torch.int32, device="cuda")

    def rollback(self, steps: int) -> None:
        """Undo `steps` advances, putting the parse state back where it was.

        The state comes from the ring the advance has been filling, so nothing
        is replayed and the host says only how far to go - which it already
        knows, since it is the one that decided the draft was rejected.
        """
        if steps <= 0:
            return
        if self.rollback_depth == 0:
            raise ValueError(
                "this batch keeps no history; construct it with rollback=k"
            )
        if steps > self.history_length:
            raise ValueError(
                f"cannot undo {steps} advances; only {self.history_length} are kept"
            )
        rows = self.batch * self.configs
        slot = (int(self.hist_slot.item()) - steps) % self.rollback_depth
        if "restore" in _PORTED:
            self._restore_cuda(rows, slot)
            self.hist_slot.fill_(slot)
            self.history_length -= steps
            return
        _restore_kernel[((rows + 255) // 256,)](
            self.lexer_state,
            self.stack,
            self.depth,
            self.config_count,
            self.widest,
            self.hist_lexer,
            self.hist_stack,
            self.hist_depth,
            self.hist_count,
            slot,
            ROWS=rows,
            CONFIGS=self.configs,
            STACK_STRIDE=self.grammar.max_stack,
            BLOCK=256,
        )
        self.hist_slot.fill_(slot)
        self.history_length -= steps

    def set_grammars(self, ids) -> None:
        """Say which grammar each sequence is under, and reset it to that start.

        A serving batch is heterogeneous by default - requests bring their own
        schemas - so this is the ordinary case rather than a special one. The
        step's shape does not change with the mixture: the tables are one arena,
        the work list is built from whatever the sequences are, and the grid is
        fixed, so the same CUDA graph covers any assignment.
        """
        values = torch.as_tensor(ids, dtype=torch.int32).reshape(-1)
        self._check_shape()
        if values.numel() != self.batch:
            raise ValueError(
                f"{values.numel()} grammar ids for a batch of {self.batch}"
            )
        if int(values.min()) < 0:
            raise ValueError("negative grammar id")
        if int(values.max()) >= self.grammar.slots:
            raise ValueError("grammar id past the end of the pool")
        # `count` is how many slots exist, not which of them hold anything. A
        # slot an eviction freed is still inside `count`, and a sequence under
        # it would be masked against whatever the arena last left there.
        dead = sorted(
            {
                identifier
                for identifier in values.tolist()
                if not self.grammar.is_live(identifier)
            }
        )
        if dead:
            raise ValueError(f"grammar ids no longer in the pool: {dead}")
        self.grammar_of.copy_(values.cuda())
        starts = torch.tensor(self.grammar.start_parser_states, dtype=torch.int32)[
            values.long()
        ]
        rows = self.stack.reshape(self.batch, self.configs, -1)
        rows[:, :, 0] = starts.reshape(self.batch, 1).cuda()
        self.depth.fill_(1)
        self.config_count.fill_(1)
        self.lexer_state.zero_()
        self.assigned = True
        # This is a reset, so what the previous parse reported about itself is
        # no longer about anything. Leaving the flags would carry a refusal, or
        # an overflow, into a sequence that has not taken a step yet.
        self.terminated.zero_()
        self.overflow.zero_()
        if self.rollback_depth > 0:
            self.hist_slot.zero_()
            self.history_length = 0

    def set_matchers(self, matchers: list) -> None:
        """Load the parse state of many reference matchers, in one transfer.

        The same thing `set_batch_configurations` does, with the conversion in
        Rust: building the arrays a row at a time in Python costs 2.3 ms at
        batch 512, against 84 us for the fill it prepares, and most of that is
        not the copy but turning each matcher's state into Python objects only
        to write them straight back out.
        """
        from engrain._engrain import pack_configurations

        rows = len(matchers)
        if rows == 0:
            return
        try:
            lexer, depths, stacks, counts, width, deep = pack_configurations(
                matchers, self.configs
            )
        except ValueError as refusal:
            # The packer refuses on the same ceiling and says so in a message.
            # A caller that can grow needs the number, not the sentence.
            widest = max(
                (len(matcher.configurations()) for matcher in matchers), default=0
            )
            if widest > self.configs:
                raise ConfigurationsExceeded(widest, self.configs) from refusal
            raise
        if deep > self.grammar.max_stack:
            raise StackTooDeep(deep, self.grammar.max_stack)

        def view(blob, shape):
            # The packer hands back a `bytearray`, which `frombuffer` takes as
            # it is. It used to hand back `bytes`, which `frombuffer` refuses
            # as read-only - so every array was copied a second time in Python
            # purely to make it writable, 284 us a step at batch 512.
            return torch.frombuffer(blob, dtype=torch.int32).view(*shape)

        self.lexer_state.view(self.batch, self.configs)[:rows, :width].copy_(
            view(lexer, (rows, width))
        )
        self.depth.view(self.batch, self.configs)[:rows, :width].copy_(
            view(depths, (rows, width))
        )
        self.stack.view(self.batch, self.configs, -1)[:rows, :width, :deep].copy_(
            view(stacks, (rows, width, deep))
        )
        counted = view(counts, (rows,))
        self.config_count[:rows].copy_(counted)
        self.widest.fill_(int(counted.max()))
        # A row given a parse state is in that state: it has not refused a
        # token and it has not met a ceiling. Clearing here rather than leaving
        # it to `set_grammars` is what lets a caller skip that call on a step
        # where the assignment has not changed - and `set_grammars` resets the
        # whole batch, which at 512 rows and a deep stack was 3,282 us against
        # the 421 this costs.
        self.terminated[:rows].zero_()
        self.overflow[:rows].zero_()

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
        # Sized by what the sequences hold rather than by the ceilings. Sending
        # `rows x max_configs x max_stack` was 4 MB a step at a serving batch to
        # carry a couple of dozen words: real parses hold one or two
        # configurations and reach a depth of tens, against a ceiling of sixteen
        # and 256.
        width = 1
        deep = 1
        for configurations in per_sequence.values():
            width = max(width, len(configurations))
            for _, stack in configurations:
                deep = max(deep, len(stack))
        if width > self.configs:
            raise ConfigurationsExceeded(width, self.configs)
        if deep > self.grammar.max_stack:
            raise StackTooDeep(deep, self.grammar.max_stack)

        # Built as one flat list and converted once. Writing each stack into a
        # numpy slice instead is a numpy call per configuration, and at a
        # serving batch that is thousands of them. Keying rows by their state to
        # skip the duplicates was tried and is worse - building a hashable key
        # out of a configuration set costs more than writing it out.
        #
        # `set_matchers` avoids this path entirely and should be preferred where
        # the caller has the matchers.
        flat: list[int] = []
        lexer_flat: list[int] = []
        depth_flat: list[int] = []
        counts = np.ones(rows, dtype=np.int32)
        blank = [0] * deep
        for sequence in range(rows):
            configurations = per_sequence.get(sequence) or ()
            counts[sequence] = max(1, len(configurations))
            for index in range(width):
                if index < len(configurations):
                    lexer_state, stack = configurations[index]
                    lexer_flat.append(lexer_state)
                    depth_flat.append(len(stack))
                    flat.extend(stack)
                    flat.extend(blank[len(stack) :])
                else:
                    lexer_flat.append(0)
                    depth_flat.append(1)
                    flat.extend(blank)
        lexer = np.array(lexer_flat, dtype=np.int32).reshape(rows, width)
        depths = np.array(depth_flat, dtype=np.int32).reshape(rows, width)
        stacks = np.array(flat, dtype=np.int32).reshape(rows, width, deep)

        # Rows past `width` are never read, because `config_count` bounds every
        # sweep, so only the prefix has to be written.
        self.lexer_state.view(self.batch, self.configs)[:rows, :width].copy_(
            torch.from_numpy(lexer)
        )
        self.depth.view(self.batch, self.configs)[:rows, :width].copy_(
            torch.from_numpy(depths)
        )
        self.stack.view(self.batch, self.configs, -1)[:rows, :width, :deep].copy_(
            torch.from_numpy(stacks)
        )
        self.config_count[:rows].copy_(torch.from_numpy(counts))
        # The widest configuration set in the batch, recorded while it is still
        # on the host. Asking the device for it costs a synchronisation every
        # step - the one thing this design exists to avoid.
        #
        # The kernels gate on `widest`, which lives on the device because the
        # advance is the thing that normally sets it. Loading state from the
        # host has to set it too, or every configuration past the first is
        # skipped and the mask comes back narrower than the grammar allows -
        # which is not a slow parser but a wrong one. Only the fill-only path
        # reaches this, so an advance was hiding it: the commit kernel writes
        # `widest` on its way past, and checking both together let one claim
        # cover for the other.
        self.widest.fill_(int(counts.max()))

    def advance(self, tokens: torch.Tensor) -> None:
        """Take one sampled token per sequence, entirely on device.

        `tokens` is a device tensor and its values are never read on the host.
        That is the requirement the rest of the design is in service of: a
        decode loop that has to look at a sampled token to advance its parser
        pays a device-to-host round trip per token, and no amount of making the
        parser itself faster removes it.
        """
        self._check_assigned()
        self.token.copy_(tokens.to(torch.int32).reshape(-1)[: self.batch])
        if self.rollback_depth > 0:
            self.history_length = min(self.history_length + 1, self.rollback_depth)
        if self.advance_graph is not None and self.recorded == self.grammar.revision:
            self.advance_graph.replay()
            return
        self._advance()

    @property
    def outgrown(self) -> bool:
        """Has the pool outgrown the buffers this batch was sized for?

        True when a grammar admitted since raised a ceiling. The batch cannot
        answer by resizing - a recorded graph holds the addresses it recorded -
        so a caller that keeps a batch across admissions asks this and makes a
        new one. Every operation checks it too, and refuses rather than letting
        a kernel index past a buffer.
        """
        return self.sized_for != self._ceilings()

    def _ceilings(self) -> tuple[int, ...]:
        """Every pool-wide maximum this batch's buffers were sized from."""
        grammar = self.grammar
        return (
            grammar.max_configs,
            grammar.max_readings,
            grammar.paths,
            grammar.window,
            grammar.max_stack,
            grammar.max_groups_per_state,
            grammar.num_groups,
        )

    def _check_shape(self) -> None:
        """Refuse to run against a pool that outgrew this batch.

        Admitting a grammar can raise a ceiling, and a kernel indexes what it
        is given: buffers sized against the old maxima do not overflow, they
        read past themselves. This engine's one rule is that a mask may be
        wider than the grammar and never narrower, and neither a wrong mask
        nor a wrong address is something to discover from a flag.

        A batch cannot resize itself - a recorded graph holds the addresses it
        recorded, and the caller holds the mask tensor - so this raises and
        names the fix.
        """
        if self.outgrown:
            raise RuntimeError(
                "a grammar admitted since this batch was made needs more room "
                f"than it has (sized for {self.sized_for}, pool now needs "
                f"{self._ceilings()}): make a new batch from the engine"
            )

    def _check_assigned(self) -> None:
        """Refuse to read an assignment nothing has written.

        Zero is a real identifier, so an unassigned batch does not fail - it
        masks every sequence against whichever grammar holds slot 0. Only a
        question when the pool holds more than one: with a single grammar zero
        is right for everyone, which is what a batch loaded straight from
        `set_configurations` relies on.
        """
        self._check_shape()
        if not self.assigned and self.grammar.count > 1:
            raise RuntimeError(
                "the pool holds several grammars and this batch has not been "
                "told which one each sequence is under: call set_grammars first"
            )

    def _refuse_capture_in_differential(self) -> None:
        """A differential run compares on the host, which a graph cannot hold.

        Recording it would capture whichever backend happened to run last and
        drop the comparison entirely - a graph that silently checks nothing,
        which is worse than not having the mode.
        """
        if self.backend == _DIFFERENTIAL:
            raise RuntimeError(
                "ENGRAIN_BACKEND=differential compares the two backends on "
                "the host, so it cannot be captured: run it eagerly"
            )

    def _snapshot_live(self) -> dict[str, torch.Tensor]:
        """Everything a rehearsal disturbs and a caller can observe.

        By name rather than by position, because it was a positional tuple
        that let `terminated` and `overflow` fall out of the restore: a
        rehearsal advances on a synthetic token the grammar refuses, so after
        `capture()` every sequence reported itself terminated and a serving
        engine polling `problems` would retire the whole batch.
        """
        return {
            name: getattr(self, name).clone()
            for name in (
                "lexer_state",
                "stack",
                "depth",
                "config_count",
                "widest",
                "terminated",
                "overflow",
            )
        }

    def _restore_live(self, held: dict[str, torch.Tensor]) -> None:
        for name, value in held.items():
            getattr(self, name).copy_(value)

    def capture_advance(self) -> None:
        """Record the advance too, so a decode step launches two graphs.

        Warming up and recording both *run* the advance, four times over, and an
        advance moves the parse. Leaving that in place would mean capturing
        silently consumed four tokens - which nothing noticed while callers
        happened to load their state afterwards, and which is wrong the moment
        one does not. The live state is put back afterwards, and the history the
        rehearsal wrote is discarded with it.
        """
        self._refuse_capture_in_differential()
        held = self._snapshot_live()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                self._advance()
        torch.cuda.current_stream().wait_stream(stream)
        self.recorded = self.grammar.revision
        self.advance_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.advance_graph):
            self._advance()

        self._restore_live(held)
        if self.rollback_depth > 0:
            self.hist_slot.zero_()
            self.history_length = 0

    def capture_step(self) -> None:
        """Record the advance and the *next* fill as one graph.

        A decode step is fill, sample, advance - and then the next fill. Only
        the sample sits between the fill and the advance; nothing at all sits
        between the advance and the fill that follows it, so those two are one
        graph and a step costs one replay instead of two. What that saves is
        not kernel time but the fixed cost of a replay and the Python around
        it, which at batch 8 is a fifth of the step.

        The rehearsal runs the advance, and an advance moves the parse, so the
        live state is put back afterwards exactly as `capture_advance` does.
        """
        self._refuse_capture_in_differential()
        held = self._snapshot_live()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                self._advance()
                self._fill()
        torch.cuda.current_stream().wait_stream(stream)
        self.recorded = self.grammar.revision
        self.step_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.step_graph):
            self._advance()
            self._fill()

        self._restore_live(held)
        if self.rollback_depth > 0:
            self.hist_slot.zero_()
            self.history_length = 0

    def advance_and_fill(self, tokens: torch.Tensor) -> torch.Tensor:
        """Accept one sampled token per sequence and return the next mask.

        The whole of a decode step after sampling, in one launch. Equivalent to
        `advance(tokens)` followed by `fill_mask()`, and the mask it returns is
        the same tensor `fill_mask` returns.
        """
        self._check_assigned()
        self._refresh_memo()
        self.token.copy_(tokens.to(torch.int32).reshape(-1)[: self.batch])
        if self.rollback_depth > 0:
            self.history_length = min(self.history_length + 1, self.rollback_depth)
        if self.step_graph is not None and self.recorded == self.grammar.revision:
            self.step_graph.replay()
            return self.mask
        self._advance()
        return self._fill()

    # The per-batch state as the CUDA kernels want it, in the order
    # `en::BatchState` declares. Built once: these buffers do not move, which
    # is the same property that lets a recorded graph hold their addresses.
    # `token` and `mask` are deliberately not here: they are rebound - the
    # draft walk points them at a row of its own arrays per position - and a
    # struct cached past that aims a kernel at the wrong tensor, while
    # rebuilding it inside a capture is a host-to-device copy a capture
    # forbids. They are passed per launch instead.
    _BATCH_FIELDS = (
        "lexer_state",
        "stack",
        "depth",
        "config_count",
        "widest",
        "grammar_of",
        "terminated",
        "overflow",
    )

    def state_struct(self) -> torch.Tensor:
        """Device memory holding one `en::BatchState`. Built once.

        Everything in it is a buffer made at construction whose address a
        recorded graph holds, so it never moves. What does move is passed per
        launch - see `_BATCH_FIELDS`.
        """
        held = getattr(self, "_state_struct", None)
        if held is None:
            held = torch.tensor(
                [getattr(self, name).data_ptr() for name in self._BATCH_FIELDS],
                dtype=torch.int64,
                device=self.device,
            )
            self._state_struct = held
        return held

    def _locate_cuda(self, grammar, rows) -> None:
        """`en_locate`: one warp per live configuration."""
        from engrain import _engrain

        _engrain.cuda_launch(
            "en_locate",
            self.sweep_blocks,
            _LOCATE_THREADS,
            torch.cuda.current_stream().cuda_stream,
            [
                self.grammar.arena_struct().data_ptr(),
                self.state_struct().data_ptr(),
                self.token.data_ptr(),
                self.live_offsets.data_ptr(),
                self.found.data_ptr(),
                self.old_lexer.data_ptr(),
                self.old_count.data_ptr(),
                self.old_stack.data_ptr(),
            ],
            [
                self.batch,
                self.configs,
                grammar.max_stack,
                rows,
                grammar.has_verdicts,
                grammar.vocab_size,
            ],
        )

    def _commit_cuda(self, grammar) -> None:
        """`en_commit`: one block per sequence, a thread per stack entry."""
        from engrain import _engrain

        _engrain.cuda_launch(
            "en_commit",
            self.batch,
            # A thread owns every `blockDim.x`-th stack entry, so this is a
            # throughput choice rather than a bound the stack imposes.
            self._fused_threads(grammar),
            torch.cuda.current_stream().cuda_stream,
            [
                self.state_struct().data_ptr(),
                self.old_lexer.data_ptr(),
                self.old_count.data_ptr(),
                self.old_stack.data_ptr(),
                self.cand_count.data_ptr(),
                self.cand_lexer.data_ptr(),
                self.cand_depth.data_ptr(),
                self.cand_floor.data_ptr(),
                self.cand_window.data_ptr(),
                self.cand_at.data_ptr(),
            ],
            [
                self.configs,
                self.max_readings,
                grammar.max_stack,
            ],
        )

    def _candidate_cuda(self, grammar, rows) -> None:
        """`en_candidate`: one thread per configuration.

        The launch is sized so that the *total thread count* equals the block
        count Triton used, because the replay scratch is two windows per
        replayer and Triton's replayer is a block. Thread-per-item is the
        3.42x lever, and this is what buys it without buying memory too.
        """
        from engrain import _engrain

        threads = _CANDIDATE_THREADS
        blocks = max(1, (self.sweep_blocks + threads - 1) // threads)
        _engrain.cuda_launch(
            "en_candidate",
            blocks,
            threads,
            torch.cuda.current_stream().cuda_stream,
            [
                self.grammar.arena_struct().data_ptr(),
                self.state_struct().data_ptr(),
                self.found.data_ptr(),
                self._fused_scratch(grammar).data_ptr(),
                self.cand_count.data_ptr(),
                self.cand_lexer.data_ptr(),
                self.cand_depth.data_ptr(),
                self.cand_floor.data_ptr(),
                self.cand_window.data_ptr(),
                self.cand_at.data_ptr(),
                self.cand_used.data_ptr(),
            ],
            [
                rows,
                self.configs,
                self.max_readings,
                grammar.max_stack,
                grammar.max_reductions,
                grammar.window,
                self.window_budget,
                grammar.paths,
            ],
        )

    def _mask_triton(self, grammar, rows) -> None:
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
            grammar.action_extra_offsets,
            grammar.action_extra,
            grammar.goto_offsets,
            grammar.goto_nonterminals,
            grammar.goto_targets,
            grammar.production_lhs,
            grammar.production_arity,
            grammar.pending_offsets,
            grammar.pending_terminals,
            grammar.verdict_offsets,
            grammar.verdicts,
            grammar.verdict_stride,
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
            self.row_floor,
            ROWS=rows,
            CONFIGS=self.configs,
            STACK_STRIDE=grammar.max_stack,
            MAX_REDUCTIONS=grammar.max_reductions,
            WINDOW=grammar.window,
            PATHS=grammar.paths,
            HAS_VERDICTS=grammar.has_verdicts,
            num_warps=1,
        )

    def _mask_cuda(self, grammar, rows) -> None:
        """`en_mask`: one thread per (configuration, group) item.

        Sized like the candidate: total threads equal the block count Triton
        used, since the scratch is two windows per replayer.
        """
        from engrain import _engrain

        threads = _CANDIDATE_THREADS
        blocks = max(1, (self.sweep_blocks + threads - 1) // threads)
        _engrain.cuda_launch(
            "en_mask",
            blocks,
            threads,
            torch.cuda.current_stream().cuda_stream,
            [
                self.grammar.arena_struct().data_ptr(),
                self.state_struct().data_ptr(),
                self.work_offsets.data_ptr(),
                self.scratch.data_ptr(),
                self.admitted.data_ptr(),
                self.high_water.data_ptr(),
                self.row_floor.data_ptr(),
            ],
            [
                rows,
                self.configs,
                grammar.max_stack,
                grammar.max_reductions,
                grammar.window,
                grammar.paths,
                grammar.has_verdicts,
            ],
        )

    # A power of two, because the digest folds a stack with a tree reduction
    # over the block. 128 covers the common stack depth without giving a
    # sequence a whole multiprocessor.
    _MEMO_THREADS = 128

    def _hash_triton(self, grammar) -> None:
        _hash_kernel[(self.batch,)](
            self.lexer_state,
            self.stack,
            self.depth,
            self.config_count,
            self.grammar_of,
            self.state_hash,
            self.suffix_hash,
            CONFIGS=self.configs,
            STACK_STRIDE=grammar.max_stack,
            SUFFIXES=_MEMO_SUFFIXES,
            num_warps=1,
        )

    def _hash_cuda(self, grammar) -> None:
        from engrain import _engrain

        _engrain.cuda_launch(
            "en_hash",
            self.batch,
            # A warp per configuration rather than one per sequence: the fold
            # is sequential in configuration order but the stack sums it folds
            # are not, and a sequence can hold sixty-four configurations.
            256,
            torch.cuda.current_stream().cuda_stream,
            [
                self.state_struct().data_ptr(),
                self.state_hash.data_ptr(),
                self.suffix_hash.data_ptr(),
            ],
            [self.configs, grammar.max_stack, _MEMO_SUFFIXES],
            shared_bytes=self.configs * 4,
        )

    def _probe_triton(self, grammar) -> None:
        _probe_kernel[(self.batch,)](
            self.lexer_state,
            self.stack,
            self.depth,
            self.config_count,
            self.grammar_of,
            self.state_hash,
            self.suffix_hash,
            self.memo_hash,
            self.memo_lexer,
            self.memo_stack,
            self.memo_depth,
            self.memo_count,
            self.memo_grammar,
            self.memo_read,
            self.memo_slot,
            self.representative,
            self.memo_store,
            self.row_floor,
            self.mask,
            grammar.mask_words,
            BATCH=triton.next_power_of_2(self.batch),
            CONFIGS=self.configs,
            STACK_STRIDE=grammar.max_stack,
            SLOTS=self.memo_slots,
            MEMO_CONFIGS=self.memo_configs,
            MEMO_STRIDE=self.memo_stride,
            SUFFIXES=_MEMO_SUFFIXES,
            BLOCK=128,
            num_warps=4,
        )

    def _probe_cuda(self, grammar) -> None:
        from engrain import _engrain

        _engrain.cuda_launch(
            "en_probe",
            self.batch,
            32,  # one warp, as `en_hash`: `__any_sync` instead of a barrier
            torch.cuda.current_stream().cuda_stream,
            [
                self.state_struct().data_ptr(),
                self.state_hash.data_ptr(),
                self.suffix_hash.data_ptr(),
                self.memo_hash.data_ptr(),
                self.memo_lexer.data_ptr(),
                self.memo_stack.data_ptr(),
                self.memo_depth.data_ptr(),
                self.memo_count.data_ptr(),
                self.memo_grammar.data_ptr(),
                self.memo_read.data_ptr(),
                self.mask.data_ptr(),
                self.memo_slot.data_ptr(),
                self.representative.data_ptr(),
                self.row_floor.data_ptr(),
                self.memo_store.data_ptr(),
            ],
            [
                self.batch,
                self.configs,
                grammar.max_stack,
                self.memo_slots,
                self.memo_configs,
                self.memo_stride,
                _MEMO_SUFFIXES,
                grammar.mask_words,
            ],
        )

    def _copy_triton(self, grammar) -> None:
        _copy_kernel[(self.batch, (grammar.mask_words + 511) // 512)](
            self.memo_slot,
            self.representative,
            self.memo_mask,
            self.mask,
            grammar.mask_words,
            BLOCK=512,
            num_warps=4,
        )

    def _copy_cuda(self, grammar) -> None:
        from engrain import _engrain

        # One row is up to 19 KiB, so this is bandwidth and wants width. The
        # second grid dimension is the sequence, which is why `cuda_launch`
        # takes a pair here.
        _engrain.cuda_launch(
            "en_copy",
            max(1, (grammar.mask_words + 255) // 256),
            256,
            torch.cuda.current_stream().cuda_stream,
            [
                self.mask.data_ptr(),
                self.memo_slot.data_ptr(),
                self.representative.data_ptr(),
                self.memo_mask.data_ptr(),
                self.state_struct().data_ptr(),
                self.row_floor.data_ptr(),
                self.memo_want.data_ptr(),
            ],
            [
                grammar.mask_words,
                self.configs,
                self.memo_stride,
                _MEMO_SUFFIXES,
            ],
            grid_y=self.batch,
        )

    def _claim_triton(self, grammar) -> None:
        _claim_kernel[(self.batch,)](
            self.depth,
            self.config_count,
            self.row_floor,
            self.memo_store,
            self.memo_want,
            CONFIGS=self.configs,
            MEMO_STRIDE=self.memo_stride,
            SUFFIXES=_MEMO_SUFFIXES,
            num_warps=1,
        )

    def _store_triton(self, grammar) -> None:
        _store_kernel[(self.batch, (grammar.mask_words + 511) // 512)](
            self.lexer_state,
            self.stack,
            self.depth,
            self.config_count,
            self.grammar_of,
            self.state_hash,
            self.representative,
            self.memo_want,
            self.suffix_hash,
            self.memo_read,
            self.memo_hash,
            self.memo_lexer,
            self.memo_stack,
            self.memo_depth,
            self.memo_count,
            self.memo_grammar,
            self.memo_mask,
            self.mask,
            grammar.mask_words,
            CONFIGS=self.configs,
            STACK_STRIDE=grammar.max_stack,
            SLOTS=self.memo_slots,
            MEMO_CONFIGS=self.memo_configs,
            MEMO_STRIDE=self.memo_stride,
            SUFFIXES=_MEMO_SUFFIXES,
            BATCH=triton.next_power_of_2(self.batch),
            BLOCK=512,
            num_warps=4,
        )

    def _restore_cuda(self, rows, slot) -> None:
        from engrain import _engrain

        _engrain.cuda_launch(
            "en_restore",
            max(1, (rows + 255) // 256),
            256,
            torch.cuda.current_stream().cuda_stream,
            [
                self.lexer_state.data_ptr(),
                self.stack.data_ptr(),
                self.depth.data_ptr(),
                self.config_count.data_ptr(),
                self.widest.data_ptr(),
                self.hist_lexer.data_ptr(),
                self.hist_stack.data_ptr(),
                self.hist_depth.data_ptr(),
                self.hist_count.data_ptr(),
            ],
            [slot, rows, self.configs, self.grammar.max_stack],
        )

    def _store_cuda(self, grammar) -> None:
        from engrain import _engrain

        # The rival scan is over the batch and the mask copy is over the words,
        # so the block is wide enough for both and the grid carries the row.
        _engrain.cuda_launch(
            "en_store",
            max(1, (grammar.mask_words + 255) // 256),
            256,
            torch.cuda.current_stream().cuda_stream,
            [
                self.lexer_state.data_ptr(),
                self.stack.data_ptr(),
                self.depth.data_ptr(),
                self.config_count.data_ptr(),
                self.grammar_of.data_ptr(),
                self.state_hash.data_ptr(),
                self.memo_want.data_ptr(),
                self.suffix_hash.data_ptr(),
                self.memo_read.data_ptr(),
                self.memo_hash.data_ptr(),
                self.memo_lexer.data_ptr(),
                self.memo_stack.data_ptr(),
                self.memo_depth.data_ptr(),
                self.memo_count.data_ptr(),
                self.memo_grammar.data_ptr(),
                self.memo_mask.data_ptr(),
                self.mask.data_ptr(),
            ],
            [
                grammar.mask_words,
                self.configs,
                grammar.max_stack,
                self.memo_slots,
                self.memo_configs,
                self.memo_stride,
                _MEMO_SUFFIXES,
            ],
            grid_y=self.batch,
        )

    def _fill_split_cuda(self, grammar) -> None:
        """The fill as a probe and a sweep, so a wide sequence is not one block.

        Two nodes rather than one. The probe keeps a block per sequence, which
        is right for it - a table probe and a scan over earlier sequences is
        O(configs x depth) and every sequence costs about the same. The sweep
        is O(configs x groups) and does not, so it takes a second grid
        dimension and a sequence spreads over as many blocks as the scratch
        will pay for.

        They cannot be one kernel with a second dimension, because the probe
        clears the rows that will be built up and a block clearing a row while
        another is already OR-ing into it loses bits.
        """
        from engrain import _engrain

        _engrain.cuda_launch(
            "en_fill_probe",
            self.batch,
            self._fill_threads(grammar),
            torch.cuda.current_stream().cuda_stream,
            [
                self.state_struct().data_ptr(),
                self.mask.data_ptr(),
                self.state_hash.data_ptr(),
                self.suffix_hash.data_ptr(),
                self.memo_hash.data_ptr(),
                self.memo_lexer.data_ptr(),
                self.memo_stack.data_ptr(),
                self.memo_depth.data_ptr(),
                self.memo_count.data_ptr(),
                self.memo_grammar.data_ptr(),
                self.memo_read.data_ptr(),
                self.memo_slot.data_ptr(),
                self.representative.data_ptr(),
                self.memo_store.data_ptr(),
                self.memo_want.data_ptr(),
                self.row_floor.data_ptr(),
                self.group_given.data_ptr(),
            ],
            [
                self.configs,
                grammar.max_stack,
                self.memo_slots,
                self.memo_configs,
                self.memo_stride,
                _MEMO_SUFFIXES,
                grammar.mask_words,
                self.group_words,
            ],
        )
        _engrain.cuda_launch(
            "en_fill_sweep",
            self.batch,
            self._fill_threads(grammar),
            torch.cuda.current_stream().cuda_stream,
            [
                self.grammar.arena_struct().data_ptr(),
                self.state_struct().data_ptr(),
                self.mask.data_ptr(),
                self.memo_slot.data_ptr(),
                self.representative.data_ptr(),
                self.row_floor.data_ptr(),
                self._fused_scratch(grammar).data_ptr(),
                self.high_water.data_ptr(),
                self.group_given.data_ptr(),
            ],
            [
                self.configs,
                grammar.max_stack,
                grammar.max_reductions,
                grammar.window,
                grammar.paths,
                grammar.has_verdicts,
                grammar.mask_words,
                self.group_words,
            ],
            # One offset per configuration and a total, built per block instead
            # of by a counting kernel and a scan.
            shared_bytes=(self.configs + 1) * 4,
            grid_y=self._fill_chunks(grammar),
        )

    # What the commit phase is launched with once the stack is deeper than it.
    # The threads walk the stack in a strided loop, so this is a throughput
    # choice rather than a correctness one - and it has to be a choice, because
    # the fused kernel cannot be launched with a thread per entry past about
    # 512: it asks for more registers than a block may hold. 256 is what a
    # 256-deep batch used before this became a loop, so nothing that fits today
    # changes shape.
    _COMMIT_THREADS = 256

    def _fused_threads(self, grammar) -> int:
        """Threads per block for the fused kernels.

        Enough to cover the stack where the stack is small, and capped where it
        is not. A thread owns every `blockDim.x`-th entry of the commit phase,
        so a narrower block costs iterations rather than correctness.
        """
        wanted = max(32, (grammar.max_stack + 31) // 32 * 32)
        return min(self._COMMIT_THREADS, wanted)

    def _fill_threads(self, grammar) -> int:
        """Threads per block for the fused fill, which has no commit phase.

        The advance's width is set by the stack, because a thread owns a stack
        entry when the commit runs. The fill has no such phase: a thread owns a
        *replay*, and what it sweeps is a lexer state's groups - hundreds of
        them - so sizing it by the stack leaves a narrow grammar sweeping seven
        hundred groups with one warp.
        """
        return _FILL_THREADS

    def _fill_chunks(self, grammar) -> int:
        """How many blocks share one sequence's sweep.

        One block per sequence made the fill cost the *widest* sequence rather
        than the total, and a serving batch is skewed by construction - every
        request sits at its own point in its own document. Chunking is what the
        global work list used to buy, without the counting kernel and the
        prefix sum on a grid of one that paid for it.

        Bounded by the scratch, not by the batch: a thread owns two replay
        windows, so the buffer is `batch x chunks x threads` and would be
        268 MB at batch 512 with eight chunks. A budget keeps it flat and lets
        the small batches - the ones with too few blocks to fill the machine,
        and so the ones that need chunking most - have the most.
        """
        rows = _FILL_SCRATCH_BUDGET // max(1, 2 * grammar.window * 4)
        return max(1, min(32, rows // max(1, self.batch * self._fill_threads(grammar))))

    def _fused_scratch(self, grammar) -> torch.Tensor:
        """Two replay windows per *thread*, since a thread owns a replay.

        Made on first use rather than at construction: the fused shape earns
        most at small batches - where graph-node dispatch is 44% of a step -
        and a batch that never takes it should not pay for the possibility.
        """
        held = getattr(self, "_fused_scratch_buffer", None)
        if held is None:
            held = torch.zeros(
                # The fill indexes by thread and the advance by configuration,
                # and either can be the wider of the two, so the buffer covers
                # both rather than whichever the caller happens to be.
                (
                    self.batch
                    * max(
                        self._fused_threads(grammar),
                        self._fill_threads(grammar) * self._fill_chunks(grammar),
                        self.configs,
                    ),
                    2 * grammar.window,
                ),
                dtype=torch.int32,
                device=self.device,
            )
            self._fused_scratch_buffer = held
        return held

    def _advance_fused_cuda(self, grammar) -> None:
        """The whole advance as one kernel: locate, replay, commit.

        Four graph nodes become one, because a block owns a sequence and its
        configurations are already its own - there is no global work list to
        count and prefix-sum, and that scan is what a kernel boundary was for.
        """
        from engrain import _engrain

        _engrain.cuda_launch(
            "en_advance_fused",
            self.batch,
            self._fused_threads(grammar),
            torch.cuda.current_stream().cuda_stream,
            [
                self.grammar.arena_struct().data_ptr(),
                self.state_struct().data_ptr(),
                self.token.data_ptr(),
                self.old_lexer.data_ptr(),
                self.old_count.data_ptr(),
                self.old_stack.data_ptr(),
                self.found.data_ptr(),
                self._fused_scratch(grammar).data_ptr(),
                self.cand_count.data_ptr(),
                self.cand_lexer.data_ptr(),
                self.cand_depth.data_ptr(),
                self.cand_floor.data_ptr(),
                self.cand_window.data_ptr(),
                self.cand_at.data_ptr(),
                self.cand_used.data_ptr(),
                self.hist_slot.data_ptr(),
                self.hist_lexer.data_ptr(),
                self.hist_stack.data_ptr(),
                self.hist_depth.data_ptr(),
                self.hist_count.data_ptr(),
            ],
            [
                self.configs,
                self.max_readings,
                grammar.max_stack,
                grammar.max_reductions,
                grammar.window,
                self.window_budget,
                grammar.paths,
                grammar.has_verdicts,
                self.rollback_depth,
                grammar.vocab_size,
            ],
        )
        if self.rollback_depth > 0:
            # A device op on a device value, so a captured advance keeps
            # stepping the ring rather than freezing on the slot it recorded.
            self.hist_slot += 1

    def _advance_triton(self) -> None:
        grammar = self.grammar
        rows = self.batch * self.configs
        self._advance_prepare_triton(grammar, rows)
        self._commit_triton(grammar)

    def _advance_prepare_cuda(self, grammar, rows) -> None:
        """The front half with each kernel taken from CUDA where it is ported.

        Written out rather than flagged inside the Triton path, for the reason
        the differential exists: the two have to be able to run the same input
        independently, or the comparison is a backend against itself.
        """
        self._count_and_scan(
            grammar, rows, self.live_counts, self.live_offsets, skip=0, unit=1
        )
        if self.rollback_depth > 0:
            self._history_triton(grammar, rows)
        if "locate" in _PORTED:
            self._locate_cuda(grammar, rows)
        else:
            self._locate_triton(grammar, rows)
        if "candidate" in _PORTED:
            # The candidates are bumped into each sequence's arena, so the
            # bump starts at zero. The fused kernel does this itself, in the
            # block that owns the sequence; the unfused path has no such
            # moment, so it happens here.
            self.cand_used.zero_()
            self._candidate_cuda(grammar, rows)
        else:
            self.cand_used.zero_()
            self._candidate_triton(grammar, rows)

    def _advance_prepare_triton(self, grammar, rows) -> None:
        """Everything the advance does before the commit: the history entry,
        the work list, the group locate and the candidates.

        Split out so the CUDA path can take the kernels that have landed and
        leave the rest, rather than the whole advance being one switch.
        """
        # One entry per live configuration, enumerated the way the fill
        # enumerates its groups. Sizing the grid by the width instead meant a
        # recorded advance was only valid while the parse stayed that wide, and
        # nothing checked; sizing it by the ceiling meant launching for every
        # configuration and returning at once from fifteen of sixteen, which was
        # 38 us of a 163 us step.
        self._count_and_scan(
            grammar, rows, self.live_counts, self.live_offsets, skip=0, unit=1
        )
        # After the running sum, so the history knows which configurations are
        # in play: before it, `live_offsets` still describes the previous step.
        if self.rollback_depth > 0:
            self._history_triton(grammar, rows)
        self._locate_triton(grammar, rows)
        self.cand_used.zero_()
        self._candidate_triton(grammar, rows)

        self._commit_triton(grammar)

    def _history_triton(self, grammar, rows) -> None:
        _history_kernel[(self.sweep_blocks,)](
            self.lexer_state,
            self.stack,
            self.depth,
            self.config_count,
            self.live_offsets,
            self.hist_slot,
            self.hist_lexer,
            self.hist_stack,
            self.hist_depth,
            self.hist_count,
            ROWS=rows,
            CONFIGS=self.configs,
            STACK_STRIDE=grammar.max_stack,
            DEPTH=self.rollback_depth,
        )
        self.hist_slot += 1

    def _locate_triton(self, grammar, rows) -> None:
        _locate_kernel[(self.sweep_blocks,)](
            grammar.group_offsets,
            grammar.group_set_kind,
            grammar.group_set_offset,
            grammar.group_set_length,
            grammar.set_payload,
            grammar.verdict_offsets,
            grammar.verdicts,
            grammar.verdict_stride,
            self.lexer_state,
            self.stack,
            self.depth,
            self.config_count,
            self.widest,
            self.token,
            self.grammar_of,
            grammar.bases,
            self.live_offsets,
            self.found,
            self.old_lexer,
            self.old_count,
            self.old_stack,
            ROWS=rows,
            CONFIGS=self.configs,
            GROUP_BLOCK=_GROUP_BLOCK,
            SEARCH_STEPS=grammar.search_steps,
            STACK_STRIDE=grammar.max_stack,
            HAS_VERDICTS=grammar.has_verdicts,
            NO_GROUP=_NO_GROUP,
            VOCAB=grammar.vocab_size,
        )

    def _candidate_triton(self, grammar, rows) -> None:
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
            grammar.action_extra_offsets,
            grammar.action_extra,
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
            self.cand_count,
            self.cand_lexer,
            self.cand_depth,
            self.cand_floor,
            self.cand_window,
            self.cand_at,
            self.cand_used,
            self.overflow,
            ROWS=rows,
            CONFIGS=self.configs,
            MAX_READINGS=self.max_readings,
            STACK_STRIDE=grammar.max_stack,
            MAX_REDUCTIONS=grammar.max_reductions,
            WINDOW=grammar.window,
            ARENA=self.window_budget,
            NO_GROUP=_NO_GROUP,
            PATHS=grammar.paths,
            num_warps=1,
        )

    def _commit_triton(self, grammar) -> None:
        _commit_kernel[(self.batch,)](
            self.lexer_state,
            self.stack,
            self.depth,
            self.config_count,
            self.old_lexer,
            self.old_count,
            self.old_stack,
            self.cand_count,
            self.cand_lexer,
            self.cand_depth,
            self.cand_floor,
            self.cand_window,
            self.cand_at,
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

    def warmup(self) -> None:
        """Compile the kernels now rather than during a decode step.

        Triton compiles on first use, and first use would otherwise be inside a
        step - a serving engine sees that as a latency spike of tens of
        milliseconds on one token. Nothing here depends on the state, so it can
        be run against whatever the batch currently holds.

        The advance is a real advance, so the live state is put back the same
        way the recordings put theirs back.
        """
        held = self._snapshot_live()
        self._fill()
        self._advance()
        torch.cuda.synchronize()
        self._restore_live(held)
        if self.rollback_depth > 0:
            self.hist_slot.zero_()
            self.history_length = 0

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
        self._refuse_capture_in_differential()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                self._fill()
        torch.cuda.current_stream().wait_stream(stream)

        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self._fill()
        self.recorded = self.grammar.revision

    def capture_draft(self, length: int) -> torch.Tensor:
        """Record a whole draft walk - every position's mask - as one graph.

        A draft of `k` tokens was `k` fills and `k` advances, which is `2k`
        graph replays and a rollback, and linear in `k` in the one cost this
        design exists to remove. XGrammar's `traverse_draft_tree` walks the
        whole tree in one call and is flat in `k`, and that is the comparison
        this loses at `k = 4` and beyond.

        The walk is the same work either way; what is not the same is issuing
        it. Here it is one recording: the state is saved, every position is
        advanced and filled into its own row, and the state is put back - all
        of it device-side, so the parse is where it started and nothing came to
        the host. Replaying it costs one launch whatever `k` is.

        Returns the buffer the masks land in, shaped (length, batch, words).
        """
        self.draft_length = length
        self.draft_tokens = torch.zeros(
            (length, self.batch), dtype=torch.int32, device="cuda"
        )
        self.draft_mask = torch.zeros(
            (length, self.batch, self.grammar.mask_words),
            dtype=torch.int32,
            device="cuda",
        )
        live = (
            self.lexer_state,
            self.stack,
            self.depth,
            self.config_count,
            self.widest,
        )
        # A copy of the parse to come back to. Nothing to do with the rollback
        # ring: a draft walk always returns to exactly where it began, so it
        # needs one slot and no arithmetic, and it stays capturable.
        self._draft_saved = tuple(item.clone() for item in live)

        def walk() -> None:
            for saved, item in zip(self._draft_saved, live):
                saved.copy_(item)
            for position in range(length):
                self.token = self.draft_tokens[position]
                self._advance()
                self.mask = self.draft_mask[position]
                self._fill()
            for saved, item in zip(self._draft_saved, live):
                item.copy_(saved)

        held_token, held_mask = self.token, self.mask
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                walk()
        torch.cuda.current_stream().wait_stream(stream)

        self.draft_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.draft_graph):
            walk()
        self.token, self.mask = held_token, held_mask
        self.recorded = self.grammar.revision
        return self.draft_mask

    def walk_draft(self, tokens: torch.Tensor) -> torch.Tensor:
        """Every position of a draft, in one replay. Leaves the parse alone.

        `tokens` is (length, batch) on the device and is never read on the host.
        """
        self.draft_tokens.copy_(tokens.to(torch.int32).reshape(self.draft_length, -1))
        if self.draft_graph is not None and self.recorded == self.grammar.revision:
            self.draft_graph.replay()
            return self.draft_mask
        raise RuntimeError("call capture_draft first")

    def _refresh_memo(self) -> None:
        """Empty the memo if a slot has changed hands since it was filled.

        An entry names a grammar by its slot, and a slot changes hands:
        compaction renumbers the survivors, and an eviction frees a slot that
        the next admission reuses. Either way an entry saying "grammar 2" now
        names a different grammar, and the state stored beside it cannot catch
        that - the identifier compares equal - so the mask of the schema that
        left is handed to the one that arrived. It is always *wider* than the
        truth rather than narrower, so it does not trip the overflow flag and
        surfaces as the model emitting a token the matcher then refuses.

        `revision` is the wrong signal for this. It says the arrays moved and a
        recorded graph is stale, which admitting into spare capacity does not
        do - so under a table budget a slot could be recycled with the memo
        left holding the previous tenant's masks. Found by 409 distinct schemas
        at batch 128, where a row that agreed with its own matcher when
        computed alone had 455 words of extra bits in the batch.
        """
        if self.memo_tenancy != self.grammar.tenancy:
            self.memo_hash.fill_(_MEMO_EMPTY)
            self.memo_tenancy = self.grammar.tenancy

    def fill_mask(self) -> torch.Tensor:
        self._check_assigned()
        self._refresh_memo()
        if self.graph is not None and self.recorded != self.grammar.revision:
            # The pool moved under us. Re-record rather than replay a graph that
            # points at where the tables used to be.
            self.graph = None
            self.advance_graph = None
            self.step_graph = None
            self.draft_graph = None
        if self.graph is not None:
            self.graph.replay()
            return self.mask
        return self._fill()

    def compact(
        self, capacity: int, both: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The mask as sorted token ids, `(ids, counts)`, both on the device.

        A sampler that is handed a vocabulary-wide mask sorts, scans and
        normalises the whole vocabulary to draw one token from a few hundred.
        Handed the set instead it does the work the constraint actually
        implies - which is why a constrained step can be *cheaper* than an
        unconstrained one, and why nobody has done it: the set has to already
        be on the device, and everywhere else it is a mask on its way to the
        host.

        `counts` is the true size of each set whatever `capacity` is, so a
        caller can always tell that a list was truncated rather than silently
        sampling from a prefix of it.
        """
        from engrain import _engrain

        if capacity <= 0:
            raise ValueError("capacity must be positive")
        held = getattr(self, "_allowed", None)
        if held is None or held.shape != (self.batch, capacity):
            self._allowed = torch.zeros(
                (self.batch, capacity), dtype=torch.int32, device=self.device
            )
            self._allowed_count = torch.zeros(
                self.batch, dtype=torch.int32, device=self.device
            )
            self._allowed_kind = torch.zeros(
                self.batch, dtype=torch.int32, device=self.device
            )
        _engrain.cuda_launch(
            "en_compact",
            self.batch,
            256,
            torch.cuda.current_stream().cuda_stream,
            [
                self.mask.data_ptr(),
                self._allowed.data_ptr(),
                self._allowed_count.data_ptr(),
                self._allowed_kind.data_ptr(),
            ],
            [
                self.grammar.mask_words,
                capacity,
                self.grammar.vocab_size,
                1 if both else 0,
            ],
        )
        return self._allowed, self._allowed_count, self._allowed_kind

    def _count_and_scan(self, grammar, rows, counts, offsets, skip, unit) -> None:
        """Count each configuration's work and prefix-sum it.

        One block does both when the row count is small, which is where the
        three launches this replaces were most of the step. Past the threshold
        a single program carrying a running total is one multiprocessor doing
        what thirty could - measured 16 us against 9 - so the parallel scan
        keeps that end. The choice is made from the batch shape, which is fixed
        when the graph is recorded, so it is not a branch inside a step.
        """
        if rows <= _SCAN_ALONE:
            _count_scan_kernel[(1,)](
                grammar.group_offsets,
                self.lexer_state,
                self.config_count,
                self.widest,
                self.representative,
                self.memo_slot,
                self.grammar_of,
                grammar.bases,
                offsets,
                CONFIGS=self.configs,
                ROWS=rows,
                SKIP_DUPLICATES=skip,
                UNIT=unit,
                BLOCK=_SCAN_BLOCK,
                num_warps=_SCAN_WARPS,
            )
            return
        _count_kernel[((rows + 255) // 256,)](
            grammar.group_offsets,
            self.lexer_state,
            self.config_count,
            self.widest,
            self.representative,
            self.memo_slot,
            self.grammar_of,
            grammar.bases,
            counts,
            CONFIGS=self.configs,
            ROWS=rows,
            SKIP_DUPLICATES=skip,
            UNIT=unit,
            BLOCK=256,
        )
        torch.cumsum(counts, 0, out=offsets[1:])

    def _fill(self) -> torch.Tensor:
        """The mask, through whichever backend this batch was built with."""
        if self.backend == _TRITON:
            return self._fill_triton()
        if self.backend == _CUDA:
            return self._fill_cuda()
        return self._differential("fill")

    def _advance(self) -> None:
        """One token per sequence, through whichever backend."""
        if self.backend == _TRITON:
            self._advance_triton()
        elif self.backend == _CUDA:
            self._advance_cuda()
        else:
            self._differential("advance")

    def _fill_cuda(self) -> torch.Tensor:
        """The fill with each kernel taken from CUDA where it is ported.

        Only the sweep so far; the memo and the scatter are still Triton, and
        the dispatch is written out rather than flagged inside `_fill_triton`
        so the two paths can run the same input independently.
        """
        if not _PORTED:
            return self._fill_triton()
        if "fill_split" in _PORTED:
            grammar = self.grammar
            self._hash_cuda(grammar)
            self._fill_split_cuda(grammar)
            self._copy_cuda(grammar)
            (self._store_cuda if "store" in _PORTED else self._store_triton)(grammar)
            return self.mask
        return self._fill_triton(
            mask=self._mask_cuda if "mask" in _PORTED else None,
            hash=self._hash_cuda if "hash" in _PORTED else None,
            probe=self._probe_cuda if "probe" in _PORTED else None,
            copy=self._copy_cuda if "copy" in _PORTED else None,
            store=self._store_cuda if "store" in _PORTED else None,
        )

    def _advance_cuda(self) -> None:
        """The advance with each kernel taken from CUDA where it is ported.

        Deliberately a separate path rather than a flag inside
        `_advance_triton`: the two have to be able to run the same input
        independently or the differential compares a backend with itself. A
        kernel not in `_PORTED` falls back here, so the two paths differ by
        exactly the kernels that have landed.
        """
        grammar = self.grammar
        rows = self.batch * self.configs
        if not _PORTED:
            self._advance_triton()
            return
        if "advance_fused" in _PORTED:
            # The history entry is written inside the fused kernel, before it
            # changes anything, so the rollback ring no longer costs a count, a
            # scan on a grid of one, and a kernel of its own.
            self._advance_fused_cuda(grammar)
            return
        self._advance_prepare_cuda(grammar, rows)
        if "commit" in _PORTED:
            self._commit_cuda(grammar)
        else:
            self._commit_triton(grammar)

    # What each path is allowed to change, and therefore what a differential
    # run has to put back between the two and compare afterwards. Naming them
    # is the whole mechanism: a tensor missing from here is one the comparison
    # cannot see, which is exactly how a port ships a difference nobody caught.
    _FILL_OUTPUTS = ("mask",)
    _ADVANCE_OUTPUTS = (
        "lexer_state",
        "stack",
        "depth",
        "config_count",
        "widest",
        "terminated",
        "overflow",
    )

    def _differential(self, path: str):
        """Run both backends on the same input and compare what they produced.

        The port's safety net, and the only thing that can catch a CUDA-only
        difference: the verifications compare a backend against the reference
        matcher, which finds a wrong answer but not a wrong answer both
        backends would have to agree on. This finds any disagreement at all.

        Eager only. Two backends and a host-side comparison cannot go in a
        graph, so `capture` refuses this mode rather than recording half of it.

        Note what this does *not* prove while only one backend exists: running
        Triton against Triton always agrees. It is still worth doing then,
        because it is a real test of the snapshot and restore below - and if
        those are incomplete, every later comparison is meaningless.
        """
        outputs = self._FILL_OUTPUTS if path == "fill" else self._ADVANCE_OUTPUTS
        entry = self._snapshot_live()
        # The memo answers a repeated state for free, so the second backend
        # would read the first one's answer instead of computing its own.
        memo = self.memo_hash.clone()
        # And the advance writes a rollback entry, so running it twice would
        # write two - which the first version of this did, and which the
        # rollback tests caught immediately. Anything a path *writes* has to be
        # put back, not just what a caller reads.
        history = None
        if self.rollback_depth > 0:
            history = {
                name: getattr(self, name).clone()
                for name in (
                    "hist_lexer",
                    "hist_depth",
                    "hist_stack",
                    "hist_count",
                    "hist_slot",
                )
            }

        first = self._fill_triton() if path == "fill" else self._advance_triton()
        theirs = {name: getattr(self, name).clone() for name in outputs}
        after = None
        if history is not None:
            after = {name: getattr(self, name).clone() for name in history}

        self._restore_live(entry)
        self.memo_hash.copy_(memo)
        if history is not None:
            for name, value in history.items():
                getattr(self, name).copy_(value)
        second = self._fill_cuda() if path == "fill" else self._advance_cuda()
        del first, second

        wrong = []
        for name in outputs:
            mine = getattr(self, name)
            if not bool(torch.equal(mine, theirs[name])):
                differing = int((mine != theirs[name]).sum())
                wrong.append(f"{name} differs in {differing} of {mine.numel()} entries")
        # The history is an output too: a backend that recorded a different
        # rollback entry would pass every mask comparison and then undo wrongly.
        for name, value in (after or {}).items():
            if not bool(torch.equal(getattr(self, name), value)):
                wrong.append(f"{name} differs")
        if wrong:
            raise AssertionError(
                f"the backends disagree on {path}: " + "; ".join(wrong)
            )
        return self.mask if path == "fill" else None

    def _fill_triton(
        self, mask=None, hash=None, probe=None, copy=None, store=None
    ) -> torch.Tensor:
        grammar = self.grammar
        # The mask is not cleared here. The probe knows which rows the scatter
        # will build up - the ones with no answer anywhere - and clears only
        # those; the rest are overwritten whole by the copy.
        (hash or self._hash_triton)(grammar)
        (probe or self._probe_triton)(grammar)
        rows = self.batch * self.configs
        # The running sum turns an item back into a configuration and a group.
        # It is a device op on a device value, so the total never comes to the
        # host and the launches below do not depend on it.
        self._count_and_scan(
            grammar, rows, self.counts, self.work_offsets, skip=1, unit=0
        )
        (mask or self._mask_triton)(grammar, rows)
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
            BLOCK=128,
            num_warps=1,
        )
        self._claim_triton(grammar)
        (copy or self._copy_triton)(grammar)
        (store or self._store_triton)(grammar)
        return self.mask
