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
    scratch_ptr,
    admitted_ptr,
    mask_ptr,
    mask_words,
    LIVE,
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
    sequence = launched // LIVE
    config = launched % LIVE
    if config >= tl.load(config_count_ptr + sequence):
        return
    # The grid covers only the configurations in use; the arrays are strided by
    # the batch's ceiling, so the two indices are not the same.
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
    reading = tl.load(reading_offsets_ptr + group)
    reading_end = tl.load(reading_offsets_ptr + group + 1)
    while reading < reading_end:
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
            for _ in range(0, MAX_REDUCTIONS):
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
                    for _ in range(0, MAX_REDUCTIONS):
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
            reading = reading_end
        else:
            reading = reading + 1

    if admitted == 1:
        tl.store(admitted_ptr + launched * MAX_GROUPS + slot, 1)
        # A complement group is written here, before anything else: it sets the
        # whole mask and then punches its exclusions out, and the groups of one
        # lexer state are disjoint, so every other admitted group's tokens are
        # among those exclusions. Punching after them would erase them. The
        # additive groups follow in a second pass.
        kind = tl.load(group_set_kind_ptr + group)
        if kind == _COMPLEMENT:
            offset = tl.load(group_set_offset_ptr + group)
            length = tl.load(group_set_length_ptr + group)
            row = mask_ptr + sequence * mask_words
            for start in range(0, mask_words, BLOCK):
                lane = start + tl.arange(0, BLOCK)
                live = lane < mask_words
                tl.atomic_or(row + lane, tl.full((BLOCK,), -1, tl.int32), mask=live)
            for start in range(0, length, BLOCK):
                lane = start + tl.arange(0, BLOCK)
                live = lane < length
                token = tl.load(set_payload_ptr + offset + lane, mask=live, other=0)
                tl.atomic_and(
                    row + token // 32,
                    (~(1 << (token % 32))).to(tl.int32),
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
    admitted_ptr,
    mask_ptr,
    mask_words,
    LIVE,
    CONFIGS: tl.constexpr,
    MAX_GROUPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Union the additive sets of the groups the first kernel admitted."""
    launched = tl.program_id(0)
    slot = tl.program_id(1)
    if tl.load(admitted_ptr + launched * MAX_GROUPS + slot) == 0:
        return
    sequence = launched // LIVE
    row_index = sequence * CONFIGS + launched % LIVE
    state = tl.load(lexer_state_ptr + row_index)
    group = tl.load(group_offsets_ptr + state) + slot
    kind = tl.load(group_set_kind_ptr + group)
    if kind == _COMPLEMENT:
        return
    offset = tl.load(group_set_offset_ptr + group)
    length = tl.load(group_set_length_ptr + group)
    row = mask_ptr + sequence * mask_words
    if kind == _SPARSE:
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


class DeviceGrammar:
    """A compiled grammar, resident on the GPU."""

    def __init__(
        self,
        compiled,
        max_stack: int = 64,
        max_reductions: int = 16,
        max_configs: int = 16,
    ):
        arrays = compiled.device_arrays()
        self.vocab_size = int(arrays["vocab_size"])
        self.mask_words = int(arrays["bitset_words"])
        self.start_parser_state = int(arrays["start_parser_state"])
        offsets = np.frombuffer(arrays["group_offsets"], dtype=np.uint32)
        self.max_groups_per_state = int(np.diff(offsets).max()) if offsets.size > 1 else 1
        self.max_stack = max_stack
        self.max_reductions = max_reductions
        self.max_configs = max_configs

        def upload(name: str, dtype=torch.int32) -> torch.Tensor:
            return torch.frombuffer(bytearray(arrays[name]), dtype=dtype).cuda()

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
        self.admitted = torch.zeros(
            rows * grammar.max_groups_per_state, dtype=torch.int32, device="cuda"
        )
        self.config_count = torch.ones(batch, dtype=torch.int32, device="cuda")
        # Two stacks per program: one for the reading being replayed, one for
        # probing what a pending lexeme could still become.
        self.scratch = torch.zeros(
            (rows * self.max_groups, 2 * grammar.max_stack),
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
        self.lexer_state[: rows * self.configs].copy_(torch.from_numpy(lexer))
        self.stack[: rows * self.configs].copy_(torch.from_numpy(stacks))
        self.depth[: rows * self.configs].copy_(torch.from_numpy(depths))
        self.config_count[:rows].copy_(torch.from_numpy(counts))

    def fill_mask(self) -> torch.Tensor:
        grammar = self.grammar
        self.mask.zero_()
        self.admitted.zero_()
        live = int(self.config_count.max())
        _mask_kernel[(self.batch * live, self.max_groups)](
            grammar.group_offsets,
            grammar.group_set_kind,
            grammar.group_set_offset,
            grammar.group_set_length,
            grammar.set_payload,
            grammar.reading_offsets,
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
            self.scratch,
            self.admitted,
            self.mask,
            grammar.mask_words,
            live,
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
            CONFIGS=self.configs,
            MAX_GROUPS=self.max_groups,
            BLOCK=128,
            num_warps=1,
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
