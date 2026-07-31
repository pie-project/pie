"""Time a captured fill at several batch sizes, on real parse states.

The measurement that lived at the bottom of the engine module until the
library had to ship: it pulled `argparse`, `json` and `pathlib` into a wheel
that needs none of them, and put a `transformers` import inside the file a
serving engine loads.

    python -m engrain_lab.rigor.fill --batches 1 32 128 512
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from engrain._engine import DeviceGrammar


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

    import engrain
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
    compiled = engrain.Compiler(vocabulary).compile_json_schema(schema)
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
