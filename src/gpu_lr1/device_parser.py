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
    scratch_ptr,
    mask_ptr,
    mask_words,
    STACK_STRIDE: tl.constexpr,
    MAX_REDUCTIONS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """One program per (sequence, group).

    Groups are independent - each asks whether the parser survives one reading
    of one set of tokens - so they are the axis to parallelise over. A sequence
    can have hundreds of them, and deciding them in series inside one program is
    what makes a large schema slow however few sequences there are.
    """
    sequence = tl.program_id(0)
    slot = tl.program_id(1)
    state = tl.load(lexer_state_ptr + sequence)
    depth = tl.load(stack_depth_ptr + sequence)
    first = tl.load(group_offsets_ptr + state)
    last = tl.load(group_offsets_ptr + state + 1)
    group = first + slot
    if group >= last:
        return

    # Replaying may push, so each program needs its own stack copy. The live
    # prefix is shared and read-only; the copy only has to hold what a replay
    # adds, which is bounded by the reduction limit.
    scratch = (sequence * tl.num_programs(1) + slot) * STACK_STRIDE
    base = sequence * STACK_STRIDE

    admitted = 0
    reading = tl.load(reading_offsets_ptr + group)
    reading_end = tl.load(reading_offsets_ptr + group + 1)
    while reading < reading_end:
        for index in range(0, STACK_STRIDE):
            if index < depth:
                tl.store(
                    scratch_ptr + scratch + index,
                    tl.load(stack_ptr + base + index),
                )
        copy_depth = depth
        top = tl.load(scratch_ptr + scratch + copy_depth - 1)
        alive = 1
        term = tl.load(reading_term_offsets_ptr + reading)
        term_end = tl.load(reading_term_offsets_ptr + reading + 1)
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
        # opening of another.
        if alive == 1:
            next_state = tl.load(reading_next_state_ptr + reading)
            pend = tl.load(pending_offsets_ptr + next_state)
            pend_end = tl.load(pending_offsets_ptr + next_state + 1)
            if pend < pend_end:
                any_ok = 0
                while pend < pend_end:
                    terminal = tl.load(pending_terminals_ptr + pend)
                    row = tl.load(action_offsets_ptr + top)
                    row_end = tl.load(action_offsets_ptr + top + 1)
                    if _search(action_terminals_ptr, row, row_end, terminal) >= 0:
                        any_ok = 1
                        pend = pend_end
                    else:
                        pend = pend + 1
                alive = any_ok

        if alive == 1:
            admitted = 1
            reading = reading_end
        else:
            reading = reading + 1

    if admitted == 1:
        kind = tl.load(group_set_kind_ptr + group)
        offset = tl.load(group_set_offset_ptr + group)
        length = tl.load(group_set_length_ptr + group)
        row = mask_ptr + sequence * mask_words
        # Written a block at a time rather than a token at a time: a program is
        # a whole warp, and a serial loop over the set uses one lane of it.
        if kind == _SPARSE:
            for start in range(0, length, BLOCK):
                lane = start + tl.arange(0, BLOCK)
                live = lane < length
                token = tl.load(set_payload_ptr + offset + lane, mask=live, other=0)
                tl.atomic_or(
                    row + token // 32,
                    (1 << (token % 32)).to(tl.int32),
                    mask=live,
                )
        elif kind == _DENSE:
            for start in range(0, mask_words, BLOCK):
                lane = start + tl.arange(0, BLOCK)
                live = lane < mask_words
                value = tl.load(set_payload_ptr + offset + lane, mask=live, other=0)
                tl.atomic_or(row + lane, value, mask=live)
        else:
            # A complement admits nearly everything, so it is set wholesale and
            # then punched through.
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


class DeviceGrammar:
    """A compiled grammar, resident on the GPU."""

    def __init__(self, compiled, max_stack: int = 64, max_reductions: int = 16):
        arrays = compiled.device_arrays()
        self.vocab_size = int(arrays["vocab_size"])
        self.mask_words = int(arrays["bitset_words"])
        self.start_parser_state = int(arrays["start_parser_state"])
        offsets = np.frombuffer(arrays["group_offsets"], dtype=np.uint32)
        self.max_groups_per_state = int(np.diff(offsets).max()) if offsets.size > 1 else 1
        self.max_stack = max_stack
        self.max_reductions = max_reductions

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
        self.lexer_state = torch.zeros(batch, dtype=torch.int32, device="cuda")
        self.stack = torch.zeros(
            (batch, grammar.max_stack), dtype=torch.int32, device="cuda"
        )
        self.stack[:, 0] = grammar.start_parser_state
        self.depth = torch.ones(batch, dtype=torch.int32, device="cuda")
        self.mask = torch.zeros(
            (batch, grammar.mask_words), dtype=torch.int32, device="cuda"
        )
        self.max_groups = grammar.max_groups_per_state
        self.scratch = torch.zeros(
            (batch * self.max_groups, grammar.max_stack),
            dtype=torch.int32,
            device="cuda",
        )

    def fill_mask(self) -> torch.Tensor:
        grammar = self.grammar
        self.mask.zero_()
        _mask_kernel[(self.batch, self.max_groups)](
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
            self.scratch,
            self.mask,
            grammar.mask_words,
            STACK_STRIDE=grammar.max_stack,
            MAX_REDUCTIONS=grammar.max_reductions,
            BLOCK=128,
            num_warps=1,
        )
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
    matcher = compiled.matcher(0)
    reference = torch.zeros(grammar.mask_words, dtype=torch.int32)
    matcher.fill_bitmask(reference)
    batch = grammar.new_batch(1)
    device = batch.fill_mask()[0].cpu()
    if torch.equal(device, reference):
        allowed = sum(bin(word & 0xFFFFFFFF).count("1") for word in reference.tolist())
        print(f"  agrees with the CPU matcher at the start state ({allowed} tokens)")
    else:
        differing = int((device != reference).sum())
        raise SystemExit(f"device and CPU masks differ in {differing} words")

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
        batch.lexer_state.copy_(
            torch.from_numpy(
                generator.choice(np.array(visited, dtype=np.int32), size=size)
            ).cuda()
        )
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
