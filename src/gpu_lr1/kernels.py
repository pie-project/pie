from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
import triton.language as tl


@triton.jit
def _dense_mask_logits_kernel(
    logits_ptr,
    dense_mask_ptr,
    rows_ptr,
    output_ptr,
    vocab_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    batch_id = tl.program_id(0)
    block_id = tl.program_id(1)
    token_offsets = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid = token_offsets < vocab_size
    row = tl.load(rows_ptr + batch_id)
    logits = tl.load(
        logits_ptr + batch_id * vocab_size + token_offsets,
        mask=valid,
        other=-float("inf"),
    )
    allowed = tl.load(
        dense_mask_ptr + row * vocab_size + token_offsets,
        mask=valid,
        other=0,
    )
    masked = tl.where(allowed != 0, logits, -float("inf"))
    tl.store(
        output_ptr + batch_id * vocab_size + token_offsets,
        masked,
        mask=valid,
    )


@triton.jit
def _bitset_mask_logits_kernel(
    logits_ptr,
    bitset_ptr,
    rows_ptr,
    output_ptr,
    vocab_size: tl.constexpr,
    words_per_row: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    batch_id = tl.program_id(0)
    block_id = tl.program_id(1)
    token_offsets = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid = token_offsets < vocab_size
    row = tl.load(rows_ptr + batch_id)
    words = tl.load(
        bitset_ptr + row * words_per_row + token_offsets // 32,
        mask=valid,
        other=0,
    )
    allowed = (words >> (token_offsets & 31)) & 1
    logits = tl.load(
        logits_ptr + batch_id * vocab_size + token_offsets,
        mask=valid,
        other=-float("inf"),
    )
    masked = tl.where(allowed != 0, logits, -float("inf"))
    tl.store(
        output_ptr + batch_id * vocab_size + token_offsets,
        masked,
        mask=valid,
    )


@triton.jit
def _dense_argmax_stage1_kernel(
    logits_ptr,
    dense_mask_ptr,
    rows_ptr,
    partial_values_ptr,
    partial_tokens_ptr,
    vocab_size: tl.constexpr,
    blocks_per_row: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    batch_id = tl.program_id(0)
    block_id = tl.program_id(1)
    token_offsets = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid = token_offsets < vocab_size
    row = tl.load(rows_ptr + batch_id)
    values = tl.load(
        logits_ptr + batch_id * vocab_size + token_offsets,
        mask=valid,
        other=-float("inf"),
    ).to(tl.float32)
    allowed = tl.load(
        dense_mask_ptr + row * vocab_size + token_offsets,
        mask=valid,
        other=0,
    )
    values = tl.where(allowed != 0, values, -float("inf"))
    local_offset = tl.argmax(values, axis=0)
    output_offset = batch_id * blocks_per_row + block_id
    tl.store(partial_values_ptr + output_offset, tl.max(values, axis=0))
    tl.store(
        partial_tokens_ptr + output_offset,
        block_id * BLOCK_SIZE + local_offset,
    )


@triton.jit
def _bitset_argmax_stage1_kernel(
    logits_ptr,
    bitset_ptr,
    rows_ptr,
    partial_values_ptr,
    partial_tokens_ptr,
    vocab_size: tl.constexpr,
    words_per_row: tl.constexpr,
    blocks_per_row: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    batch_id = tl.program_id(0)
    block_id = tl.program_id(1)
    token_offsets = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid = token_offsets < vocab_size
    row = tl.load(rows_ptr + batch_id)
    words = tl.load(
        bitset_ptr + row * words_per_row + token_offsets // 32,
        mask=valid,
        other=0,
    )
    allowed = (words >> (token_offsets & 31)) & 1
    values = tl.load(
        logits_ptr + batch_id * vocab_size + token_offsets,
        mask=valid,
        other=-float("inf"),
    ).to(tl.float32)
    values = tl.where(allowed != 0, values, -float("inf"))
    local_offset = tl.argmax(values, axis=0)
    output_offset = batch_id * blocks_per_row + block_id
    tl.store(partial_values_ptr + output_offset, tl.max(values, axis=0))
    tl.store(
        partial_tokens_ptr + output_offset,
        block_id * BLOCK_SIZE + local_offset,
    )


@triton.jit
def _argmax_stage2_kernel(
    partial_values_ptr,
    partial_tokens_ptr,
    output_tokens_ptr,
    blocks_per_row: tl.constexpr,
    REDUCE_SIZE: tl.constexpr,
):
    batch_id = tl.program_id(0)
    offsets = tl.arange(0, REDUCE_SIZE)
    valid = offsets < blocks_per_row
    values = tl.load(
        partial_values_ptr + batch_id * blocks_per_row + offsets,
        mask=valid,
        other=-float("inf"),
    )
    local_offset = tl.argmax(values, axis=0)
    token = tl.load(
        partial_tokens_ptr + batch_id * blocks_per_row + local_offset
    )
    tl.store(output_tokens_ptr + batch_id, token)


@triton.jit
def _csr_argmax_kernel(
    logits_ptr,
    csr_indptr_ptr,
    csr_indices_ptr,
    rows_ptr,
    output_tokens_ptr,
    vocab_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    batch_id = tl.program_id(0)
    row = tl.load(rows_ptr + batch_id)
    start = tl.load(csr_indptr_ptr + row)
    end = tl.load(csr_indptr_ptr + row + 1)
    length = end - start
    offsets = tl.arange(0, BLOCK_SIZE)
    valid = offsets < length
    tokens = tl.load(
        csr_indices_ptr + start + offsets,
        mask=valid,
        other=0,
    )
    values = tl.load(
        logits_ptr + batch_id * vocab_size + tokens,
        mask=valid,
        other=-float("inf"),
    ).to(tl.float32)
    local_offset = tl.argmax(values, axis=0)
    selected = tl.load(
        csr_indices_ptr + start + local_offset,
        mask=length > 0,
        other=-1,
    )
    tl.store(output_tokens_ptr + batch_id, selected)


@triton.jit
def _csr_argmax_advance_kernel(
    logits_ptr,
    csr_indptr_ptr,
    csr_indices_ptr,
    csr_next_state_ptr,
    rows_ptr,
    output_tokens_ptr,
    output_states_ptr,
    vocab_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    batch_id = tl.program_id(0)
    row = tl.load(rows_ptr + batch_id)
    start = tl.load(csr_indptr_ptr + row)
    end = tl.load(csr_indptr_ptr + row + 1)
    length = end - start
    offsets = tl.arange(0, BLOCK_SIZE)
    valid = offsets < length
    tokens = tl.load(
        csr_indices_ptr + start + offsets,
        mask=valid,
        other=0,
    )
    values = tl.load(
        logits_ptr + batch_id * vocab_size + tokens,
        mask=valid,
        other=-float("inf"),
    ).to(tl.float32)
    local_offset = tl.argmax(values, axis=0)
    selected_offset = start + local_offset
    selected_token = tl.load(
        csr_indices_ptr + selected_offset,
        mask=length > 0,
        other=-1,
    )
    selected_state = tl.load(
        csr_next_state_ptr + selected_offset,
        mask=length > 0,
        other=0,
    )
    tl.store(output_tokens_ptr + batch_id, selected_token)
    tl.store(output_states_ptr + batch_id, selected_state)


@triton.jit
def _csr_argmax_advance_packed_kernel(
    logits_ptr,
    csr_indptr_ptr,
    csr_indices_ptr,
    csr_next_state_ptr,
    rows_ptr,
    output_tokens_ptr,
    output_states_ptr,
    batch_size,
    vocab_size: tl.constexpr,
    ROWS_PER_PROGRAM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    batch_offsets = (
        tl.program_id(0) * ROWS_PER_PROGRAM
        + tl.arange(0, ROWS_PER_PROGRAM)
    )
    lane_offsets = tl.arange(0, BLOCK_SIZE)
    batch_valid = batch_offsets < batch_size
    rows = tl.load(
        rows_ptr + batch_offsets,
        mask=batch_valid,
        other=0,
    )
    starts = tl.load(
        csr_indptr_ptr + rows,
        mask=batch_valid,
        other=0,
    )
    ends = tl.load(
        csr_indptr_ptr + rows + 1,
        mask=batch_valid,
        other=0,
    )
    lengths = ends - starts
    entry_offsets = starts[:, None] + lane_offsets[None, :]
    valid = batch_valid[:, None] & (lane_offsets[None, :] < lengths[:, None])
    tokens = tl.load(
        csr_indices_ptr + entry_offsets,
        mask=valid,
        other=0,
    )
    values = tl.load(
        logits_ptr
        + batch_offsets[:, None] * vocab_size
        + tokens,
        mask=valid,
        other=-float("inf"),
    ).to(tl.float32)
    selected_lanes = tl.argmax(values, axis=1)
    selected = valid & (
        lane_offsets[None, :] == selected_lanes[:, None]
    )
    selected_tokens = tl.sum(tl.where(selected, tokens, 0), axis=1)
    selected_states = tl.sum(
        tl.load(
            csr_next_state_ptr + entry_offsets,
            mask=selected,
            other=0,
        ),
        axis=1,
    )
    has_values = batch_valid & (lengths > 0)
    tl.store(
        output_tokens_ptr + batch_offsets,
        tl.where(has_values, selected_tokens, -1),
        mask=batch_valid,
    )
    tl.store(
        output_states_ptr + batch_offsets,
        selected_states,
        mask=batch_valid,
    )


@triton.jit
def _ell_argmax_advance_kernel(
    logits_ptr,
    row_lengths_ptr,
    token_table_ptr,
    next_state_table_ptr,
    rows_ptr,
    output_tokens_ptr,
    output_states_ptr,
    vocab_size: tl.constexpr,
    row_width: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    batch_id = tl.program_id(0)
    row = tl.load(rows_ptr + batch_id)
    length = tl.load(row_lengths_ptr + row)
    offsets = tl.arange(0, BLOCK_SIZE)
    valid = offsets < length
    table_offsets = row * row_width + offsets
    tokens = tl.load(
        token_table_ptr + table_offsets,
        mask=valid,
        other=0,
    )
    values = tl.load(
        logits_ptr + batch_id * vocab_size + tokens,
        mask=valid,
        other=-float("inf"),
    ).to(tl.float32)
    selected_lane = tl.argmax(values, axis=0)
    selected_offset = row * row_width + selected_lane
    selected_token = tl.load(
        token_table_ptr + selected_offset,
        mask=length > 0,
        other=-1,
    )
    selected_state = tl.load(
        next_state_table_ptr + selected_offset,
        mask=length > 0,
        other=0,
    )
    tl.store(output_tokens_ptr + batch_id, selected_token)
    tl.store(output_states_ptr + batch_id, selected_state)


@triton.jit
def _dense_advance_kernel(
    states_ptr,
    token_ids_ptr,
    next_state_ptr,
    output_states_ptr,
    batch_size,
    vocab_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid = offsets < batch_size
    states = tl.load(states_ptr + offsets, mask=valid)
    token_ids = tl.load(token_ids_ptr + offsets, mask=valid)
    next_states = tl.load(
        next_state_ptr + states * vocab_size + token_ids,
        mask=valid,
    )
    tl.store(output_states_ptr + offsets, next_states, mask=valid)


@triton.jit
def _byte_dfa_advance_kernel(
    states_ptr,
    token_ids_ptr,
    token_bytes_ptr,
    token_lengths_ptr,
    byte_transitions_ptr,
    output_states_ptr,
    batch_size,
    max_token_bytes: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid = offsets < batch_size
    states = tl.load(states_ptr + offsets, mask=valid, other=0)
    token_ids = tl.load(token_ids_ptr + offsets, mask=valid, other=0)
    lengths = tl.load(token_lengths_ptr + token_ids, mask=valid, other=0)

    for byte_index in tl.static_range(0, max_token_bytes):
        byte_values = tl.load(
            token_bytes_ptr + token_ids * max_token_bytes + byte_index,
            mask=valid,
            other=0,
        ).to(tl.int32)
        advanced = tl.load(
            byte_transitions_ptr + states * 256 + byte_values,
            mask=valid,
            other=0,
        )
        states = tl.where(lengths > byte_index, advanced, states)

    tl.store(output_states_ptr + offsets, states, mask=valid)


@triton.jit
def _stack_update_row_major_kernel(
    stack_ptr,
    stack_pointers_ptr,
    actions_ptr,
    push_symbols_ptr,
    output_pointers_ptr,
    output_tops_ptr,
    batch_size,
    max_depth: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    batch_offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid = batch_offsets < batch_size
    stack_pointers = tl.load(
        stack_pointers_ptr + batch_offsets,
        mask=valid,
        other=1,
    )
    actions = tl.load(actions_ptr + batch_offsets, mask=valid, other=0)
    top_indices = batch_offsets * max_depth + stack_pointers - 1
    tops = tl.load(stack_ptr + top_indices, mask=valid, other=0)
    push_symbols = tl.load(
        push_symbols_ptr + batch_offsets,
        mask=valid,
        other=0,
    )
    push_indices = batch_offsets * max_depth + stack_pointers
    tl.store(
        stack_ptr + push_indices,
        push_symbols,
        mask=valid & (actions > 0) & (stack_pointers < max_depth),
    )
    next_pointers = stack_pointers + actions
    next_pointers = tl.maximum(1, tl.minimum(next_pointers, max_depth))
    tl.store(output_pointers_ptr + batch_offsets, next_pointers, mask=valid)
    tl.store(output_tops_ptr + batch_offsets, tops, mask=valid)


@triton.jit
def _stack_update_depth_major_kernel(
    stack_ptr,
    stack_pointers_ptr,
    actions_ptr,
    push_symbols_ptr,
    output_pointers_ptr,
    output_tops_ptr,
    batch_size,
    max_depth: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    batch_offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid = batch_offsets < batch_size
    stack_pointers = tl.load(
        stack_pointers_ptr + batch_offsets,
        mask=valid,
        other=1,
    )
    actions = tl.load(actions_ptr + batch_offsets, mask=valid, other=0)
    top_indices = (stack_pointers - 1) * batch_size + batch_offsets
    tops = tl.load(stack_ptr + top_indices, mask=valid, other=0)
    push_symbols = tl.load(
        push_symbols_ptr + batch_offsets,
        mask=valid,
        other=0,
    )
    push_indices = stack_pointers * batch_size + batch_offsets
    tl.store(
        stack_ptr + push_indices,
        push_symbols,
        mask=valid & (actions > 0) & (stack_pointers < max_depth),
    )
    next_pointers = stack_pointers + actions
    next_pointers = tl.maximum(1, tl.minimum(next_pointers, max_depth))
    tl.store(output_pointers_ptr + batch_offsets, next_pointers, mask=valid)
    tl.store(output_tops_ptr + batch_offsets, tops, mask=valid)


@triton.jit
def _lr1_csr_lookup(
    indptr_ptr,
    symbols_ptr,
    values_ptr,
    row,
    symbol,
    BLOCK_SIZE: tl.constexpr,
    DEFAULT_VALUE: tl.constexpr,
):
    start = tl.load(indptr_ptr + row)
    end = tl.load(indptr_ptr + row + 1)
    offsets = tl.arange(0, BLOCK_SIZE)
    valid = offsets < end - start
    columns = tl.load(
        symbols_ptr + start + offsets,
        mask=valid,
        other=-1,
    )
    values = tl.load(
        values_ptr + start + offsets,
        mask=valid,
        other=0,
    ).to(tl.int32)
    matches = valid & (columns == symbol)
    found = tl.sum(matches.to(tl.int32), axis=0)
    value = tl.sum(tl.where(matches, values, 0), axis=0)
    return tl.where(found != 0, value, DEFAULT_VALUE)


@triton.jit
def _lr1_execute_selected(
    stack_ptr,
    stack_offsets_ptr,
    action_indptr_ptr,
    action_symbols_ptr,
    action_values_ptr,
    goto_indptr_ptr,
    goto_symbols_ptr,
    goto_targets_ptr,
    production_lhs_ptr,
    production_rhs_len_ptr,
    batch_id,
    terminal,
    initial_action,
    initial_status,
    pointer,
    max_reductions,
    ACTION_BLOCK_SIZE: tl.constexpr,
    GOTO_BLOCK_SIZE: tl.constexpr,
):
    base = tl.load(stack_offsets_ptr + batch_id)
    capacity = tl.load(stack_offsets_ptr + batch_id + 1) - base
    action = initial_action
    status = initial_status
    reductions = 0
    execute = status == -1

    is_error = execute & (action == 0)
    is_accept = execute & (action == -2147483648)
    is_shift = execute & (action > 0)
    shift_has_room = is_shift & (pointer < capacity)
    tl.store(
        stack_ptr + base + pointer,
        action - 1,
        mask=shift_has_room,
    )
    pointer += shift_has_room.to(tl.int32)
    status = tl.where(is_error, 2, status)
    status = tl.where(is_accept, 1, status)
    status = tl.where(is_shift & ~shift_has_room, 3, status)
    status = tl.where(shift_has_room, 0, status)

    while status == -1:
        at_limit = reductions >= max_reductions
        status = tl.where(at_limit, 4, status)
        active = status == -1

        safe_production = tl.where(active, -action - 1, 0)
        pop_count = tl.load(
            production_rhs_len_ptr + safe_production,
            mask=active,
            other=0,
        )
        valid_pop = active & (pop_count < pointer)
        status = tl.where(active & ~valid_pop, 2, status)

        reduced_pointer = pointer - pop_count
        safe_reduced_pointer = tl.where(valid_pop, reduced_pointer, 1)
        exposed_state = tl.load(
            stack_ptr + base + safe_reduced_pointer - 1,
            mask=valid_pop,
            other=0,
        )
        lhs = tl.load(
            production_lhs_ptr + safe_production,
            mask=valid_pop,
            other=0,
        )
        safe_exposed_state = tl.where(valid_pop, exposed_state, 0)
        goto_target = _lr1_csr_lookup(
            goto_indptr_ptr,
            goto_symbols_ptr,
            goto_targets_ptr,
            safe_exposed_state,
            lhs,
            BLOCK_SIZE=GOTO_BLOCK_SIZE,
            DEFAULT_VALUE=-1,
        )
        valid_goto = valid_pop & (goto_target >= 0)
        status = tl.where(valid_pop & ~valid_goto, 2, status)
        goto_has_room = valid_goto & (reduced_pointer < capacity)
        status = tl.where(valid_goto & ~goto_has_room, 3, status)
        tl.store(
            stack_ptr + base + reduced_pointer,
            goto_target,
            mask=goto_has_room,
        )
        pointer = tl.where(
            goto_has_room,
            reduced_pointer + 1,
            pointer,
        )
        reductions += goto_has_room.to(tl.int32)

        safe_goto_target = tl.where(goto_has_room, goto_target, 0)
        next_action = _lr1_csr_lookup(
            action_indptr_ptr,
            action_symbols_ptr,
            action_values_ptr,
            safe_goto_target,
            terminal,
            BLOCK_SIZE=ACTION_BLOCK_SIZE,
            DEFAULT_VALUE=0,
        )
        next_error = goto_has_room & (next_action == 0)
        next_accept = goto_has_room & (next_action == -2147483648)
        next_shift = goto_has_room & (next_action > 0)
        next_shift_has_room = next_shift & (pointer < capacity)
        tl.store(
            stack_ptr + base + pointer,
            next_action - 1,
            mask=next_shift_has_room,
        )
        pointer += next_shift_has_room.to(tl.int32)
        status = tl.where(next_error, 2, status)
        status = tl.where(next_accept, 1, status)
        status = tl.where(next_shift & ~next_shift_has_room, 3, status)
        status = tl.where(next_shift_has_room, 0, status)
        action = next_action

    return pointer, status, reductions


@triton.jit
def _lr1_fused_step_kernel(
    logits_ptr,
    action_indptr_ptr,
    action_symbols_ptr,
    action_values_ptr,
    goto_indptr_ptr,
    goto_symbols_ptr,
    goto_targets_ptr,
    production_lhs_ptr,
    production_rhs_len_ptr,
    stack_ptr,
    stack_offsets_ptr,
    stack_pointers_ptr,
    output_terminals_ptr,
    output_statuses_ptr,
    output_reductions_ptr,
    terminal_count: tl.constexpr,
    max_reductions,
    ACTION_BLOCK_SIZE: tl.constexpr,
    GOTO_BLOCK_SIZE: tl.constexpr,
):
    batch_id = tl.program_id(0)
    base = tl.load(stack_offsets_ptr + batch_id)
    pointer = tl.load(stack_pointers_ptr + batch_id)
    state = tl.load(stack_ptr + base + pointer - 1)
    start = tl.load(action_indptr_ptr + state)
    end = tl.load(action_indptr_ptr + state + 1)
    length = end - start
    offsets = tl.arange(0, ACTION_BLOCK_SIZE)
    valid = offsets < length
    terminals = tl.load(
        action_symbols_ptr + start + offsets,
        mask=valid,
        other=0,
    )
    values = tl.load(
        logits_ptr + batch_id * terminal_count + terminals,
        mask=valid,
        other=-float("inf"),
    ).to(tl.float32)
    actions = tl.load(
        action_values_ptr + start + offsets,
        mask=valid,
        other=0,
    )
    maximum = tl.max(values, axis=0)
    tied_terminals = tl.where(
        valid & (values == maximum),
        terminals,
        2147483647,
    )
    selected_terminal = tl.min(tied_terminals, axis=0)
    selected = valid & (terminals == selected_terminal)
    terminal = tl.where(length > 0, selected_terminal, -1)
    action = tl.sum(tl.where(selected, actions, 0), axis=0)
    pointer, status, reductions = _lr1_execute_selected(
        stack_ptr,
        stack_offsets_ptr,
        action_indptr_ptr,
        action_symbols_ptr,
        action_values_ptr,
        goto_indptr_ptr,
        goto_symbols_ptr,
        goto_targets_ptr,
        production_lhs_ptr,
        production_rhs_len_ptr,
        batch_id,
        terminal,
        action,
        -1,
        pointer,
        max_reductions,
        ACTION_BLOCK_SIZE=ACTION_BLOCK_SIZE,
        GOTO_BLOCK_SIZE=GOTO_BLOCK_SIZE,
    )
    tl.store(stack_pointers_ptr + batch_id, pointer)
    tl.store(output_terminals_ptr + batch_id, terminal)
    tl.store(output_statuses_ptr + batch_id, status)
    tl.store(output_reductions_ptr + batch_id, reductions)


@triton.jit
def _lr1_select_fast_kernel(
    logits_ptr,
    action_indptr_ptr,
    action_symbols_ptr,
    action_values_ptr,
    stack_ptr,
    stack_offsets_ptr,
    stack_pointers_ptr,
    output_terminals_ptr,
    output_actions_ptr,
    output_statuses_ptr,
    output_reductions_ptr,
    terminal_count: tl.constexpr,
    ACTION_BLOCK_SIZE: tl.constexpr,
):
    batch_id = tl.program_id(0)
    base = tl.load(stack_offsets_ptr + batch_id)
    capacity = tl.load(stack_offsets_ptr + batch_id + 1) - base
    pointer = tl.load(stack_pointers_ptr + batch_id)
    state = tl.load(stack_ptr + base + pointer - 1)
    start = tl.load(action_indptr_ptr + state)
    end = tl.load(action_indptr_ptr + state + 1)
    length = end - start
    offsets = tl.arange(0, ACTION_BLOCK_SIZE)
    valid = offsets < length
    terminals = tl.load(
        action_symbols_ptr + start + offsets,
        mask=valid,
        other=0,
    )
    values = tl.load(
        logits_ptr + batch_id * terminal_count + terminals,
        mask=valid,
        other=-float("inf"),
    ).to(tl.float32)
    actions = tl.load(
        action_values_ptr + start + offsets,
        mask=valid,
        other=0,
    )
    maximum = tl.max(values, axis=0)
    tied_terminals = tl.where(
        valid & (values == maximum),
        terminals,
        2147483647,
    )
    selected_terminal = tl.min(tied_terminals, axis=0)
    selected = valid & (terminals == selected_terminal)
    terminal = tl.where(length > 0, selected_terminal, -1)
    action = tl.sum(tl.where(selected, actions, 0), axis=0)

    is_error = action == 0
    is_accept = action == -2147483648
    is_shift = action > 0
    shift_has_room = is_shift & (pointer < capacity)
    tl.store(
        stack_ptr + base + pointer,
        action - 1,
        mask=shift_has_room,
    )
    pointer += shift_has_room.to(tl.int32)
    status = tl.where(is_error, 2, -1)
    status = tl.where(is_accept, 1, status)
    status = tl.where(is_shift & ~shift_has_room, 3, status)
    status = tl.where(shift_has_room, 0, status)

    tl.store(stack_pointers_ptr + batch_id, pointer)
    tl.store(output_terminals_ptr + batch_id, terminal)
    tl.store(output_actions_ptr + batch_id, action)
    tl.store(output_statuses_ptr + batch_id, status)
    tl.store(output_reductions_ptr + batch_id, 0)


@triton.jit
def _lr1_reduce_slow_kernel(
    action_indptr_ptr,
    action_symbols_ptr,
    action_values_ptr,
    goto_indptr_ptr,
    goto_symbols_ptr,
    goto_targets_ptr,
    production_lhs_ptr,
    production_rhs_len_ptr,
    stack_ptr,
    stack_offsets_ptr,
    stack_pointers_ptr,
    terminals_ptr,
    actions_ptr,
    statuses_ptr,
    reductions_ptr,
    max_reductions,
    ACTION_BLOCK_SIZE: tl.constexpr,
    GOTO_BLOCK_SIZE: tl.constexpr,
):
    batch_id = tl.program_id(0)
    pointer = tl.load(stack_pointers_ptr + batch_id)
    terminal = tl.load(terminals_ptr + batch_id)
    action = tl.load(actions_ptr + batch_id)
    status = tl.load(statuses_ptr + batch_id)
    pointer, status, reductions = _lr1_execute_selected(
        stack_ptr,
        stack_offsets_ptr,
        action_indptr_ptr,
        action_symbols_ptr,
        action_values_ptr,
        goto_indptr_ptr,
        goto_symbols_ptr,
        goto_targets_ptr,
        production_lhs_ptr,
        production_rhs_len_ptr,
        batch_id,
        terminal,
        action,
        status,
        pointer,
        max_reductions,
        ACTION_BLOCK_SIZE=ACTION_BLOCK_SIZE,
        GOTO_BLOCK_SIZE=GOTO_BLOCK_SIZE,
    )
    tl.store(stack_pointers_ptr + batch_id, pointer)
    tl.store(statuses_ptr + batch_id, status)
    tl.store(reductions_ptr + batch_id, reductions)


@dataclass(frozen=True)
class ArgmaxWorkspace:
    partial_values: torch.Tensor
    partial_tokens: torch.Tensor
    blocks_per_row: int


@dataclass(frozen=True)
class LR1StepWorkspace:
    terminals: torch.Tensor
    actions: torch.Tensor
    statuses: torch.Tensor
    reductions: torch.Tensor


@dataclass
class CSRArgmaxAdvancePlan:
    csr_indptr: torch.Tensor
    csr_indices: torch.Tensor
    csr_next_state: torch.Tensor
    rows: torch.Tensor
    output_tokens: torch.Tensor
    output_states: torch.Tensor
    batch_size: int
    vocab_size: int
    block_size: int
    num_warps: int
    rows_per_program: int = 1
    packed: bool = False
    autotune_cuda_us: float | None = None

    @property
    def strategy(self) -> str:
        if self.packed:
            return (
                f"packed_r{self.rows_per_program}_w{self.num_warps}"
            )
        return f"single_w{self.num_warps}"

    def __call__(
        self,
        logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.packed:
            grid = (triton.cdiv(self.batch_size, self.rows_per_program),)
            _csr_argmax_advance_packed_kernel[grid](
                logits,
                self.csr_indptr,
                self.csr_indices,
                self.csr_next_state,
                self.rows,
                self.output_tokens,
                self.output_states,
                self.batch_size,
                vocab_size=self.vocab_size,
                ROWS_PER_PROGRAM=self.rows_per_program,
                BLOCK_SIZE=self.block_size,
                num_warps=self.num_warps,
            )
        else:
            _csr_argmax_advance_kernel[(self.batch_size,)](
                logits,
                self.csr_indptr,
                self.csr_indices,
                self.csr_next_state,
                self.rows,
                self.output_tokens,
                self.output_states,
                vocab_size=self.vocab_size,
                BLOCK_SIZE=self.block_size,
                num_warps=self.num_warps,
            )
        return self.output_tokens, self.output_states

    def capture(
        self,
        logits: torch.Tensor,
    ) -> CSRArgmaxAdvanceGraph:
        inplace_states = (
            self.output_states.data_ptr() == self.rows.data_ptr()
        )
        saved_states = self.rows.clone() if inplace_states else None
        self(logits)
        torch.cuda.synchronize(logits.device)
        if saved_states is not None:
            self.rows.copy_(saved_states)
            torch.cuda.synchronize(logits.device)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            self(logits)
        if saved_states is not None:
            self.rows.copy_(saved_states)
            torch.cuda.synchronize(logits.device)
        return CSRArgmaxAdvanceGraph(
            graph=graph,
            output_tokens=self.output_tokens,
            output_states=self.output_states,
            strategy=self.strategy,
        )


@dataclass(frozen=True)
class CSRArgmaxAdvanceGraph:
    graph: torch.cuda.CUDAGraph
    output_tokens: torch.Tensor
    output_states: torch.Tensor
    strategy: str

    def replay(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.graph.replay()
        return self.output_tokens, self.output_states


@dataclass
class ELLArgmaxAdvancePlan:
    row_lengths: torch.Tensor
    token_table: torch.Tensor
    next_state_table: torch.Tensor
    rows: torch.Tensor
    output_tokens: torch.Tensor
    output_states: torch.Tensor
    batch_size: int
    vocab_size: int
    row_width: int
    block_size: int
    num_warps: int

    @property
    def strategy(self) -> str:
        return f"ell_w{self.num_warps}"

    def __call__(
        self,
        logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _ell_argmax_advance_kernel[(self.batch_size,)](
            logits,
            self.row_lengths,
            self.token_table,
            self.next_state_table,
            self.rows,
            self.output_tokens,
            self.output_states,
            vocab_size=self.vocab_size,
            row_width=self.row_width,
            BLOCK_SIZE=self.block_size,
            num_warps=self.num_warps,
        )
        return self.output_tokens, self.output_states

    def capture(
        self,
        logits: torch.Tensor,
    ) -> CSRArgmaxAdvanceGraph:
        inplace_states = (
            self.output_states.data_ptr() == self.rows.data_ptr()
        )
        saved_states = self.rows.clone() if inplace_states else None
        self(logits)
        torch.cuda.synchronize(logits.device)
        if saved_states is not None:
            self.rows.copy_(saved_states)
            torch.cuda.synchronize(logits.device)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            self(logits)
        if saved_states is not None:
            self.rows.copy_(saved_states)
            torch.cuda.synchronize(logits.device)
        return CSRArgmaxAdvanceGraph(
            graph=graph,
            output_tokens=self.output_tokens,
            output_states=self.output_states,
            strategy=self.strategy,
        )


def torch_dense_mask_logits(
    logits: torch.Tensor,
    dense_mask: torch.Tensor,
    rows: torch.Tensor,
) -> torch.Tensor:
    return logits.masked_fill(dense_mask[rows].logical_not(), -float("inf"))


def triton_dense_mask_logits(
    logits: torch.Tensor,
    dense_mask: torch.Tensor,
    rows: torch.Tensor,
    *,
    block_size: int = 256,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    _validate_logits(logits, rows)
    output = output if output is not None else torch.empty_like(logits)
    grid = (logits.shape[0], triton.cdiv(logits.shape[1], block_size))
    _dense_mask_logits_kernel[grid](
        logits,
        dense_mask,
        rows,
        output,
        vocab_size=logits.shape[1],
        BLOCK_SIZE=block_size,
    )
    return output


def triton_bitset_mask_logits(
    logits: torch.Tensor,
    bitset_mask: torch.Tensor,
    rows: torch.Tensor,
    *,
    block_size: int = 256,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    _validate_logits(logits, rows)
    output = output if output is not None else torch.empty_like(logits)
    words_per_row = bitset_mask.shape[1]
    grid = (logits.shape[0], triton.cdiv(logits.shape[1], block_size))
    _bitset_mask_logits_kernel[grid](
        logits,
        bitset_mask,
        rows,
        output,
        vocab_size=logits.shape[1],
        words_per_row=words_per_row,
        BLOCK_SIZE=block_size,
    )
    return output


def triton_dense_argmax(
    logits: torch.Tensor,
    dense_mask: torch.Tensor,
    rows: torch.Tensor,
    *,
    block_size: int = 1024,
    workspace: ArgmaxWorkspace | None = None,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    return _triton_table_argmax(
        logits,
        dense_mask,
        rows,
        block_size=block_size,
        workspace=workspace,
        output=output,
        bitset=False,
    )


def triton_bitset_argmax(
    logits: torch.Tensor,
    bitset_mask: torch.Tensor,
    rows: torch.Tensor,
    *,
    block_size: int = 1024,
    workspace: ArgmaxWorkspace | None = None,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    return _triton_table_argmax(
        logits,
        bitset_mask,
        rows,
        block_size=block_size,
        workspace=workspace,
        output=output,
        bitset=True,
    )


def make_argmax_workspace(
    batch_size: int,
    vocab_size: int,
    *,
    block_size: int = 1024,
    device: torch.device | str = "cuda",
) -> ArgmaxWorkspace:
    blocks_per_row = triton.cdiv(vocab_size, block_size)
    return ArgmaxWorkspace(
        partial_values=torch.empty(
            (batch_size, blocks_per_row),
            dtype=torch.float32,
            device=device,
        ),
        partial_tokens=torch.empty(
            (batch_size, blocks_per_row),
            dtype=torch.int32,
            device=device,
        ),
        blocks_per_row=blocks_per_row,
    )


def _triton_table_argmax(
    logits: torch.Tensor,
    table: torch.Tensor,
    rows: torch.Tensor,
    *,
    block_size: int,
    workspace: ArgmaxWorkspace | None,
    output: torch.Tensor | None,
    bitset: bool,
) -> torch.Tensor:
    _validate_logits(logits, rows)
    batch_size, vocab_size = logits.shape
    workspace = workspace or make_argmax_workspace(
        batch_size,
        vocab_size,
        block_size=block_size,
        device=logits.device,
    )
    blocks_per_row = triton.cdiv(vocab_size, block_size)
    if workspace.partial_values.shape != (batch_size, blocks_per_row):
        raise ValueError("argmax workspace shape does not match logits")

    grid = (batch_size, blocks_per_row)
    if bitset:
        _bitset_argmax_stage1_kernel[grid](
            logits,
            table,
            rows,
            workspace.partial_values,
            workspace.partial_tokens,
            vocab_size=vocab_size,
            words_per_row=table.shape[1],
            blocks_per_row=blocks_per_row,
            BLOCK_SIZE=block_size,
        )
    else:
        _dense_argmax_stage1_kernel[grid](
            logits,
            table,
            rows,
            workspace.partial_values,
            workspace.partial_tokens,
            vocab_size=vocab_size,
            blocks_per_row=blocks_per_row,
            BLOCK_SIZE=block_size,
        )

    output = (
        output
        if output is not None
        else torch.empty(batch_size, dtype=torch.int32, device=logits.device)
    )
    reduce_size = triton.next_power_of_2(blocks_per_row)
    _argmax_stage2_kernel[(batch_size,)](
        workspace.partial_values,
        workspace.partial_tokens,
        output,
        blocks_per_row=blocks_per_row,
        REDUCE_SIZE=reduce_size,
    )
    return output


def triton_csr_argmax(
    logits: torch.Tensor,
    csr_indptr: torch.Tensor,
    csr_indices: torch.Tensor,
    rows: torch.Tensor,
    *,
    max_row_nnz: int,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    """Select from CSR rows; max_row_nnz must bound every selected row."""
    _validate_logits(logits, rows)
    if max_row_nnz <= 0:
        raise ValueError("CSR argmax requires at least one allowed token per row")
    block_size = triton.next_power_of_2(max_row_nnz)
    if block_size > 32768:
        raise ValueError("CSR row is too dense for the single-program kernel")
    output = (
        output
        if output is not None
        else torch.empty(logits.shape[0], dtype=torch.int32, device=logits.device)
    )
    _csr_argmax_kernel[(logits.shape[0],)](
        logits,
        csr_indptr,
        csr_indices,
        rows,
        output,
        vocab_size=logits.shape[1],
        BLOCK_SIZE=block_size,
        num_warps=_csr_num_warps(block_size),
    )
    return output


def triton_csr_argmax_advance(
    logits: torch.Tensor,
    csr_indptr: torch.Tensor,
    csr_indices: torch.Tensor,
    csr_next_state: torch.Tensor,
    rows: torch.Tensor,
    *,
    max_row_nnz: int,
    output_tokens: torch.Tensor | None = None,
    output_states: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select and advance; max_row_nnz must bound every selected CSR row."""
    _validate_logits(logits, rows)
    if max_row_nnz <= 0:
        raise ValueError("CSR argmax requires at least one allowed token per row")
    block_size = triton.next_power_of_2(max_row_nnz)
    if block_size > 32768:
        raise ValueError("CSR row is too dense for the single-program kernel")
    output_tokens = (
        output_tokens
        if output_tokens is not None
        else torch.empty(logits.shape[0], dtype=torch.int32, device=logits.device)
    )
    output_states = (
        output_states
        if output_states is not None
        else torch.empty_like(rows)
    )
    _csr_argmax_advance_kernel[(logits.shape[0],)](
        logits,
        csr_indptr,
        csr_indices,
        csr_next_state,
        rows,
        output_tokens,
        output_states,
        vocab_size=logits.shape[1],
        BLOCK_SIZE=block_size,
        num_warps=_csr_num_warps(block_size),
    )
    return output_tokens, output_states


def make_csr_argmax_advance_plan(
    logits: torch.Tensor,
    csr_indptr: torch.Tensor,
    csr_indices: torch.Tensor,
    csr_next_state: torch.Tensor,
    rows: torch.Tensor,
    *,
    max_row_nnz: int,
    output_tokens: torch.Tensor | None = None,
    output_states: torch.Tensor | None = None,
    autotune: bool = False,
    autotune_warmup: int = 5,
    autotune_iterations: int = 50,
) -> CSRArgmaxAdvancePlan:
    """Validate once and bind a low-overhead CSR selection launch plan."""
    _validate_logits(logits, rows)
    for name, tensor in (
        ("csr_indptr", csr_indptr),
        ("csr_indices", csr_indices),
        ("csr_next_state", csr_next_state),
    ):
        if not tensor.is_cuda or tensor.dtype != torch.int32:
            raise ValueError(f"{name} must be a CUDA int32 tensor")
    if csr_indices.shape != csr_next_state.shape:
        raise ValueError("CSR token and next-state arrays must match")
    if max_row_nnz <= 0:
        raise ValueError("CSR argmax requires at least one allowed token per row")
    block_size = triton.next_power_of_2(max_row_nnz)
    if block_size > 32768:
        raise ValueError("CSR row is too dense for the single-program kernel")
    if autotune_warmup < 0 or autotune_iterations <= 0:
        raise ValueError("invalid CSR autotune iteration counts")
    output_tokens = (
        output_tokens
        if output_tokens is not None
        else torch.empty(logits.shape[0], dtype=torch.int32, device=logits.device)
    )
    output_states = (
        output_states
        if output_states is not None
        else torch.empty_like(rows)
    )
    if output_tokens.shape != rows.shape or output_tokens.dtype != torch.int32:
        raise ValueError("CSR output tokens must match rows and use int32")
    if output_states.shape != rows.shape or output_states.dtype != torch.int32:
        raise ValueError("CSR output states must match rows and use int32")
    if not output_tokens.is_cuda or not output_states.is_cuda:
        raise ValueError("CSR outputs must be CUDA tensors")

    candidates = [
        CSRArgmaxAdvancePlan(
            csr_indptr=csr_indptr,
            csr_indices=csr_indices,
            csr_next_state=csr_next_state,
            rows=rows,
            output_tokens=output_tokens,
            output_states=output_states,
            batch_size=logits.shape[0],
            vocab_size=logits.shape[1],
            block_size=block_size,
            num_warps=_csr_num_warps(block_size),
        )
    ]
    if autotune:
        existing = {(False, 1, candidates[0].num_warps)}
        for num_warps in (1, 2, 4, 8):
            key = (False, 1, num_warps)
            if key not in existing:
                candidates.append(
                    CSRArgmaxAdvancePlan(
                        csr_indptr=csr_indptr,
                        csr_indices=csr_indices,
                        csr_next_state=csr_next_state,
                        rows=rows,
                        output_tokens=output_tokens,
                        output_states=output_states,
                        batch_size=logits.shape[0],
                        vocab_size=logits.shape[1],
                        block_size=block_size,
                        num_warps=num_warps,
                    )
                )
                existing.add(key)
        if block_size <= 64 and logits.shape[0] >= 2:
            for rows_per_program in (2, 4, 8, 16):
                active_lanes = rows_per_program * block_size
                recommended_warps = min(
                    8,
                    max(1, triton.next_power_of_2(max(1, active_lanes // 32))),
                )
                for num_warps in {
                    recommended_warps,
                    max(1, recommended_warps // 2),
                }:
                    key = (True, rows_per_program, num_warps)
                    if key in existing:
                        continue
                    candidates.append(
                        CSRArgmaxAdvancePlan(
                            csr_indptr=csr_indptr,
                            csr_indices=csr_indices,
                            csr_next_state=csr_next_state,
                            rows=rows,
                            output_tokens=output_tokens,
                            output_states=output_states,
                            batch_size=logits.shape[0],
                            vocab_size=logits.shape[1],
                            block_size=block_size,
                            num_warps=num_warps,
                            rows_per_program=rows_per_program,
                            packed=True,
                        )
                    )
                    existing.add(key)

        best_plan = candidates[0]
        best_us = float("inf")
        for candidate in candidates:
            elapsed_us = float(
                triton.testing.do_bench(
                    lambda candidate=candidate: candidate(logits),
                    warmup=autotune_warmup,
                    rep=autotune_iterations,
                )
                * 1_000
            )
            if elapsed_us < best_us:
                best_us = elapsed_us
                best_plan = candidate
        best_plan.autotune_cuda_us = float(best_us)
        return best_plan
    return candidates[0]


def make_ell_argmax_advance_plan(
    logits: torch.Tensor,
    row_lengths: torch.Tensor,
    token_table: torch.Tensor,
    next_state_table: torch.Tensor,
    rows: torch.Tensor,
    *,
    output_tokens: torch.Tensor | None = None,
    output_states: torch.Tensor | None = None,
) -> ELLArgmaxAdvancePlan:
    _validate_logits(logits, rows)
    for name, tensor in (
        ("row_lengths", row_lengths),
        ("token_table", token_table),
        ("next_state_table", next_state_table),
    ):
        if not tensor.is_cuda:
            raise ValueError(f"{name} must be a CUDA tensor")
    if row_lengths.ndim != 1 or row_lengths.dtype != torch.int32:
        raise ValueError("ELL row lengths must be one-dimensional int32")
    if (
        token_table.ndim != 2
        or next_state_table.shape != token_table.shape
        or token_table.dtype != torch.int32
        or next_state_table.dtype != torch.int32
    ):
        raise ValueError("ELL token and next-state tables must match int32")
    if token_table.shape[0] != row_lengths.shape[0]:
        raise ValueError("ELL tables require one length per row")
    row_width = token_table.shape[1]
    if row_width <= 0:
        raise ValueError("ELL row width must be positive")
    block_size = triton.next_power_of_2(row_width)
    if block_size > 32768:
        raise ValueError("ELL row is too wide for one Triton program")
    output_tokens = (
        output_tokens
        if output_tokens is not None
        else torch.empty(logits.shape[0], dtype=torch.int32, device=logits.device)
    )
    output_states = (
        output_states
        if output_states is not None
        else torch.empty_like(rows)
    )
    if not output_tokens.is_cuda or not output_states.is_cuda:
        raise ValueError("ELL outputs must be CUDA tensors")
    return ELLArgmaxAdvancePlan(
        row_lengths=row_lengths,
        token_table=token_table,
        next_state_table=next_state_table,
        rows=rows,
        output_tokens=output_tokens,
        output_states=output_states,
        batch_size=logits.shape[0],
        vocab_size=logits.shape[1],
        row_width=row_width,
        block_size=block_size,
        num_warps=_csr_num_warps(block_size),
    )


def triton_csr_argmax_advance_packed(
    logits: torch.Tensor,
    csr_indptr: torch.Tensor,
    csr_indices: torch.Tensor,
    csr_next_state: torch.Tensor,
    rows: torch.Tensor,
    *,
    max_row_nnz: int,
    rows_per_program: int,
    num_warps: int,
    output_tokens: torch.Tensor | None = None,
    output_states: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select and advance several tiny CSR rows in each Triton program."""
    _validate_logits(logits, rows)
    if max_row_nnz <= 0:
        raise ValueError("CSR argmax requires at least one allowed token per row")
    if rows_per_program not in (1, 2, 4, 8, 16):
        raise ValueError("rows_per_program must be a power of two from 1 to 16")
    if num_warps not in (1, 2, 4, 8):
        raise ValueError("num_warps must be 1, 2, 4, or 8")
    block_size = triton.next_power_of_2(max_row_nnz)
    if block_size > 64:
        raise ValueError("packed CSR kernel only supports rows up to 64 entries")
    output_tokens = (
        output_tokens
        if output_tokens is not None
        else torch.empty(logits.shape[0], dtype=torch.int32, device=logits.device)
    )
    output_states = (
        output_states
        if output_states is not None
        else torch.empty_like(rows)
    )
    grid = (triton.cdiv(logits.shape[0], rows_per_program),)
    _csr_argmax_advance_packed_kernel[grid](
        logits,
        csr_indptr,
        csr_indices,
        csr_next_state,
        rows,
        output_tokens,
        output_states,
        logits.shape[0],
        vocab_size=logits.shape[1],
        ROWS_PER_PROGRAM=rows_per_program,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )
    return output_tokens, output_states


def triton_dense_advance(
    states: torch.Tensor,
    token_ids: torch.Tensor,
    next_state: torch.Tensor,
    *,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    if states.shape != token_ids.shape:
        raise ValueError("states and token_ids must have matching shapes")
    output = output if output is not None else torch.empty_like(states)
    block_size = 256
    grid = (triton.cdiv(states.numel(), block_size),)
    _dense_advance_kernel[grid](
        states,
        token_ids,
        next_state,
        output,
        states.numel(),
        vocab_size=next_state.shape[1],
        BLOCK_SIZE=block_size,
    )
    return output


def triton_byte_dfa_advance(
    states: torch.Tensor,
    token_ids: torch.Tensor,
    token_bytes: torch.Tensor,
    token_lengths: torch.Tensor,
    byte_transitions: torch.Tensor,
    *,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    if states.shape != token_ids.shape:
        raise ValueError("states and token_ids must have matching shapes")
    output = output if output is not None else torch.empty_like(states)
    block_size = 256
    grid = (triton.cdiv(states.numel(), block_size),)
    _byte_dfa_advance_kernel[grid](
        states,
        token_ids,
        token_bytes,
        token_lengths,
        byte_transitions,
        output,
        states.numel(),
        max_token_bytes=token_bytes.shape[1],
        BLOCK_SIZE=block_size,
    )
    return output


def triton_stack_update(
    stack: torch.Tensor,
    stack_pointers: torch.Tensor,
    actions: torch.Tensor,
    push_symbols: torch.Tensor,
    *,
    layout: str,
    output_pointers: torch.Tensor | None = None,
    output_tops: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if stack_pointers.shape != actions.shape or actions.shape != push_symbols.shape:
        raise ValueError("stack metadata tensors must have matching shapes")
    batch_size = stack_pointers.numel()
    if layout == "row":
        if stack.ndim != 2 or stack.shape[0] != batch_size:
            raise ValueError("row-major stack must have shape [batch, depth]")
        max_depth = stack.shape[1]
        kernel = _stack_update_row_major_kernel
    elif layout == "depth":
        if stack.ndim != 2 or stack.shape[1] != batch_size:
            raise ValueError("depth-major stack must have shape [depth, batch]")
        max_depth = stack.shape[0]
        kernel = _stack_update_depth_major_kernel
    else:
        raise ValueError("layout must be 'row' or 'depth'")

    output_pointers = (
        output_pointers
        if output_pointers is not None
        else torch.empty_like(stack_pointers)
    )
    output_tops = (
        output_tops
        if output_tops is not None
        else torch.empty_like(push_symbols)
    )
    block_size = 256
    grid = (triton.cdiv(batch_size, block_size),)
    kernel[grid](
        stack,
        stack_pointers,
        actions,
        push_symbols,
        output_pointers,
        output_tops,
        batch_size,
        max_depth=max_depth,
        BLOCK_SIZE=block_size,
    )
    return output_pointers, output_tops


def make_lr1_step_workspace(
    batch_size: int,
    *,
    device: torch.device | str = "cuda",
) -> LR1StepWorkspace:
    target = torch.device(device)
    return LR1StepWorkspace(
        terminals=torch.empty(batch_size, dtype=torch.int32, device=target),
        actions=torch.empty(batch_size, dtype=torch.int32, device=target),
        statuses=torch.empty(batch_size, dtype=torch.int32, device=target),
        reductions=torch.empty(batch_size, dtype=torch.int32, device=target),
    )


def triton_lr1_step_fused(
    logits: torch.Tensor,
    tables,
    stacks,
    *,
    max_action_row_nnz: int | None = None,
    max_goto_row_nnz: int | None = None,
    max_reductions: int = 128,
    workspace: LR1StepWorkspace | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select one grammar terminal and execute its LR reduction/shift closure."""
    max_action_row_nnz, max_goto_row_nnz = _resolve_lr1_row_bounds(
        tables,
        max_action_row_nnz,
        max_goto_row_nnz,
    )
    workspace = _prepare_lr1_step(
        logits,
        tables,
        stacks,
        max_action_row_nnz=max_action_row_nnz,
        max_goto_row_nnz=max_goto_row_nnz,
        max_reductions=max_reductions,
        workspace=workspace,
    )
    action_block = triton.next_power_of_2(max_action_row_nnz)
    goto_block = triton.next_power_of_2(max_goto_row_nnz)
    _lr1_fused_step_kernel[(logits.shape[0],)](
        logits,
        tables.action_indptr,
        tables.action_symbols,
        tables.action_values,
        tables.goto_indptr,
        tables.goto_symbols,
        tables.goto_targets,
        tables.production_lhs,
        tables.production_rhs_len,
        stacks.values,
        stacks.offsets,
        stacks.pointers,
        workspace.terminals,
        workspace.statuses,
        workspace.reductions,
        terminal_count=logits.shape[1],
        max_reductions=max_reductions,
        ACTION_BLOCK_SIZE=action_block,
        GOTO_BLOCK_SIZE=goto_block,
        num_warps=_num_warps(action_block),
    )
    return workspace.terminals, workspace.statuses, workspace.reductions


def triton_lr1_step_split(
    logits: torch.Tensor,
    tables,
    stacks,
    *,
    max_action_row_nnz: int | None = None,
    max_goto_row_nnz: int | None = None,
    max_reductions: int = 128,
    workspace: LR1StepWorkspace | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run a shift fast path, then execute only pending reductions."""
    max_action_row_nnz, max_goto_row_nnz = _resolve_lr1_row_bounds(
        tables,
        max_action_row_nnz,
        max_goto_row_nnz,
    )
    workspace = _prepare_lr1_step(
        logits,
        tables,
        stacks,
        max_action_row_nnz=max_action_row_nnz,
        max_goto_row_nnz=max_goto_row_nnz,
        max_reductions=max_reductions,
        workspace=workspace,
    )
    action_block = triton.next_power_of_2(max_action_row_nnz)
    goto_block = triton.next_power_of_2(max_goto_row_nnz)
    grid = (logits.shape[0],)
    _lr1_select_fast_kernel[grid](
        logits,
        tables.action_indptr,
        tables.action_symbols,
        tables.action_values,
        stacks.values,
        stacks.offsets,
        stacks.pointers,
        workspace.terminals,
        workspace.actions,
        workspace.statuses,
        workspace.reductions,
        terminal_count=logits.shape[1],
        ACTION_BLOCK_SIZE=action_block,
        num_warps=_num_warps(action_block),
    )
    _lr1_reduce_slow_kernel[grid](
        tables.action_indptr,
        tables.action_symbols,
        tables.action_values,
        tables.goto_indptr,
        tables.goto_symbols,
        tables.goto_targets,
        tables.production_lhs,
        tables.production_rhs_len,
        stacks.values,
        stacks.offsets,
        stacks.pointers,
        workspace.terminals,
        workspace.actions,
        workspace.statuses,
        workspace.reductions,
        max_reductions=max_reductions,
        ACTION_BLOCK_SIZE=action_block,
        GOTO_BLOCK_SIZE=goto_block,
        num_warps=_num_warps(action_block),
    )
    return workspace.terminals, workspace.statuses, workspace.reductions


def _resolve_lr1_row_bounds(
    tables,
    max_action_row_nnz: int | None,
    max_goto_row_nnz: int | None,
) -> tuple[int, int]:
    actual_action = int(tables.max_action_row_nnz)
    actual_goto = int(tables.max_goto_row_nnz)
    action_bound = (
        actual_action
        if max_action_row_nnz is None
        else int(max_action_row_nnz)
    )
    goto_bound = (
        actual_goto if max_goto_row_nnz is None else int(max_goto_row_nnz)
    )
    if action_bound < actual_action:
        raise ValueError(
            "max_action_row_nnz does not cover the packed ACTION table"
        )
    if goto_bound < actual_goto:
        raise ValueError(
            "max_goto_row_nnz does not cover the packed GOTO table"
        )
    return action_bound, goto_bound


def _prepare_lr1_step(
    logits: torch.Tensor,
    tables,
    stacks,
    *,
    max_action_row_nnz: int,
    max_goto_row_nnz: int,
    max_reductions: int,
    workspace: LR1StepWorkspace | None,
) -> LR1StepWorkspace:
    if not logits.is_cuda:
        raise ValueError("LR(1) Triton kernels require CUDA logits")
    if logits.ndim != 2:
        raise ValueError("LR(1) logits must have shape [batch, terminals]")
    if logits.shape[1] < int(tables.num_terminals):
        raise ValueError("LR(1) logits do not cover all packed terminals")
    batch_size = logits.shape[0]
    if stacks.pointers.shape != (batch_size,):
        raise ValueError("LR(1) stack pointers must match logits batch size")
    if stacks.offsets.shape != (batch_size + 1,):
        raise ValueError("LR(1) stack offsets must have batch_size + 1 entries")
    tensors = (
        logits,
        tables.action_indptr,
        tables.action_symbols,
        tables.action_values,
        tables.goto_indptr,
        tables.goto_symbols,
        tables.goto_targets,
        tables.production_lhs,
        tables.production_rhs_len,
        stacks.values,
        stacks.offsets,
        stacks.pointers,
    )
    if any(not tensor.is_cuda for tensor in tensors):
        raise ValueError("all LR(1) tables and stacks must be CUDA tensors")
    int_tensors = tensors[1:]
    if any(tensor.dtype != torch.int32 for tensor in int_tensors):
        raise TypeError("LR(1) tables and stacks must use int32")
    if max_action_row_nnz <= 0 or max_goto_row_nnz <= 0:
        raise ValueError("LR(1) sparse row bounds must be positive")
    if max_reductions < 0:
        raise ValueError("max_reductions must be non-negative")
    action_block = triton.next_power_of_2(max_action_row_nnz)
    goto_block = triton.next_power_of_2(max_goto_row_nnz)
    if action_block > 32768 or goto_block > 32768:
        raise ValueError("LR(1) sparse row is too wide for one Triton program")
    workspace = workspace or make_lr1_step_workspace(
        batch_size,
        device=logits.device,
    )
    expected = (batch_size,)
    for tensor in (
        workspace.terminals,
        workspace.actions,
        workspace.statuses,
        workspace.reductions,
    ):
        if tensor.shape != expected or tensor.dtype != torch.int32:
            raise ValueError("LR(1) workspace does not match batch size")
        if not tensor.is_cuda:
            raise ValueError("LR(1) workspace must be on CUDA")
    return workspace


def _validate_logits(logits: torch.Tensor, rows: torch.Tensor) -> None:
    if not logits.is_cuda or not rows.is_cuda:
        raise ValueError("Triton kernels require CUDA tensors")
    if logits.ndim != 2 or rows.shape != (logits.shape[0],):
        raise ValueError("rows must contain one table row per logits row")
    if rows.dtype != torch.int32:
        raise TypeError("rows must use int32")


def _num_warps(block_size: int) -> int:
    if block_size <= 256:
        return 4
    if block_size <= 2048:
        return 8
    return 8


def _csr_num_warps(block_size: int) -> int:
    if block_size <= 64:
        return 4
    if block_size <= 512:
        return 2
    return 8
