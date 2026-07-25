"""Walk the vocabulary through the lexer on device, instead of tabulating it.

The current artifact answers "is this token allowed" by lookup: for every lexer
state it stores, per group, a bitset over the whole vocabulary. That is why one
real schema costs 440 MB — the masks scale as `groups x vocabulary / 8`, and the
group count grows with the lexer.

The alternative is to carry the automaton and walk. A transition table is
`states x 256 x 4` bytes and does not scale with the vocabulary at all: 1.2 MB
for the same schema, 365x smaller. The cost moves from memory to compute, and
the question this module answers is how much compute.

The walk is maximal munch with restart: consume bytes; when a byte cannot extend
the current lexeme, settle it if the state accepts and start the next lexeme
from the same byte. A token is lexically admissible when its bytes can be
consumed that way. This is only the lexer half of admissibility - the parser
half is a handful of ACTION lookups on the terminals emitted, which does not
scale with the vocabulary and is measured separately.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import triton
import triton.language as tl


NO_STATE = 0xFFFFFFFF
DEAD = tl.constexpr(-1)


@triton.jit
def _walk_kernel(
    transitions_ptr,
    accepting_ptr,
    token_bytes_ptr,
    token_offsets_ptr,
    states_ptr,
    mask_ptr,
    vocab_size,
    mask_words,
    START: tl.constexpr,
    BLOCK: tl.constexpr,
    MAX_TOKEN_BYTES: tl.constexpr,
):
    """One program per (sequence, block of tokens)."""
    sequence = tl.program_id(0)
    block = tl.program_id(1)

    token = block * BLOCK + tl.arange(0, BLOCK)
    live = token < vocab_size

    start = tl.load(token_offsets_ptr + token, mask=live, other=0)
    end = tl.load(token_offsets_ptr + token + 1, mask=live, other=0)

    state = tl.load(states_ptr + sequence)
    state = tl.where(live, state, DEAD)
    cursor = start

    # Run while any lane still has bytes. A fixed bound would charge every lane
    # for the longest token in the vocabulary - 128 bytes against a mean of 6.5 -
    # so the loop has to end when the block does, not when the vocabulary does.
    running = live & (cursor < end) & (state != DEAD)
    while tl.sum(running.to(tl.int32)) > 0:
        byte = tl.load(token_bytes_ptr + cursor, mask=running, other=0)
        nxt = tl.load(
            transitions_ptr + state.to(tl.int64) * 256 + byte.to(tl.int64),
            mask=running,
            other=DEAD,
        )
        # A byte that cannot extend the lexeme settles it, if the state accepts,
        # and the next lexeme starts from the same byte.
        blocked = running & (nxt == DEAD)
        accepts = tl.load(accepting_ptr + state, mask=blocked, other=0)
        restart = tl.load(
            transitions_ptr + START * 256 + byte.to(tl.int64),
            mask=blocked & (accepts != 0),
            other=DEAD,
        )
        nxt = tl.where(blocked, tl.where(accepts != 0, restart, DEAD), nxt)
        state = tl.where(running, nxt, state)
        cursor = tl.where(running & (state != DEAD), cursor + 1, cursor)
        running = live & (cursor < end) & (state != DEAD)

    allowed = live & (state != DEAD) & (cursor >= end)
    word = token // 32
    bit = (allowed.to(tl.uint32) << (token % 32).to(tl.uint32)).to(tl.uint32)
    tl.atomic_or(
        mask_ptr + sequence * mask_words + word,
        bit,
        mask=live & allowed,
    )


def walk(
    transitions: torch.Tensor,
    accepting: torch.Tensor,
    token_bytes: torch.Tensor,
    token_offsets: torch.Tensor,
    states: torch.Tensor,
    mask: torch.Tensor,
    vocab_size: int,
    block: int = 1024,
    max_token_bytes: int = 32,
) -> None:
    mask.zero_()
    grid = (states.numel(), triton.cdiv(vocab_size, block))
    _walk_kernel[grid](
        transitions,
        accepting,
        token_bytes,
        token_offsets,
        states,
        mask,
        vocab_size,
        mask.shape[1],
        START=0,
        BLOCK=block,
        MAX_TOKEN_BYTES=max_token_bytes,
    )


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
    parser.add_argument("--schema-index", type=int, default=6)
    parser.add_argument(
        "--distinct-states",
        type=int,
        nargs="+",
        default=[1, 8, 32, 128, 512],
        help="sequences sharing a lexer state share an answer, so the walk runs "
        "once per distinct state, not once per sequence",
    )
    parser.add_argument(
        "--states",
        choices=["start", "random"],
        default="start",
        help="which lexer states the sequences sit in. 'start' is the worst "
        "case: almost every token is still alive, so every walk runs to the "
        "end of its token. A random state is usually mid-codepoint and kills "
        "tokens on the first byte, which flatters the measurement.",
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

    states = compiled.num_lexer_states
    transitions = torch.frombuffer(
        bytearray(compiled.lexer_transitions()), dtype=torch.int32
    ).cuda()
    accepting = torch.frombuffer(
        bytearray(compiled.lexer_accepting()), dtype=torch.uint8
    ).cuda()

    flat = b"".join(vocabulary)
    offsets = np.zeros(len(vocabulary) + 1, dtype=np.int32)
    np.cumsum([len(piece) for piece in vocabulary], out=offsets[1:])
    token_bytes = torch.frombuffer(bytearray(flat), dtype=torch.uint8).cuda()
    token_offsets = torch.from_numpy(offsets).cuda()

    vocab_size = len(vocabulary)
    words = (vocab_size + 31) // 32

    transition_bytes = states * 256 * 4
    mask_bytes = compiled.num_groups * compiled.bitset_words * 4
    print(
        f"schema {arguments.schema_index}: {states} lexer states, "
        f"{compiled.num_groups} groups, {vocab_size} tokens"
    )
    print(
        f"  transition table {transition_bytes / 1048576:.2f} MB   "
        f"precomputed masks {mask_bytes / 1048576:.1f} MB   "
        f"{mask_bytes / max(transition_bytes, 1):.0f}x"
    )

    longest = max(len(piece) for piece in vocabulary)
    print(f"  longest token {longest} bytes")

    results = []
    generator = torch.Generator(device="cuda").manual_seed(0)
    for batch in arguments.distinct_states:
        if arguments.states == "start":
            sequence_states = torch.zeros(batch, device="cuda", dtype=torch.int32)
        else:
            sequence_states = torch.randint(
                0, states, (batch,), device="cuda", dtype=torch.int32, generator=generator
            )
        mask = torch.zeros((batch, words), dtype=torch.int32, device="cuda")
        walk(
            transitions,
            accepting,
            token_bytes,
            token_offsets,
            sequence_states,
            mask,
            vocab_size,
            max_token_bytes=longest,
        )
        microseconds = _time(
            lambda: walk(
                transitions,
                accepting,
                token_bytes,
                token_offsets,
                sequence_states,
                mask,
                vocab_size,
                max_token_bytes=longest,
            )
        )
        counts = [
            sum(bin(word & 0xFFFFFFFF).count("1") for word in row)
            for row in mask.cpu().tolist()
        ]
        median = sorted(counts)[len(counts) // 2]
        print(
            f"  {batch:>4} distinct states: {microseconds:8.1f} us   "
            f"(median {median} of {vocab_size} tokens allowed)"
        )
        results.append(
            {
                "distinct_states": batch,
                "walk_us": microseconds,
                "median_allowed": median,
            }
        )

    if arguments.output:
        arguments.output.write_text(
            json.dumps(
                {
                    "schema_index": arguments.schema_index,
                    "lexer_states": states,
                    "groups": compiled.num_groups,
                    "vocab_size": vocab_size,
                    "transition_bytes": transition_bytes,
                    "mask_bytes": mask_bytes,
                    "measurements": results,
                },
                indent=2,
            )
        )
        print(f"wrote {arguments.output}")


if __name__ == "__main__":
    main()
