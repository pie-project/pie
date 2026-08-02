"""How unevenly the sweep's work is spread across a batch's sequences.

The number that decides the device runtime's block structure. Two ways to
sweep the same work:

- **A global work list.** Every block takes a contiguous slice of *all* the
  items, so the load is perfectly balanced. That is what the engine does, and
  it costs a global prefix sum - which is a kernel boundary, and a kernel
  boundary is 1.10 us of graph-node dispatch that no kernel can win back.
- **A block per sequence.** Nothing crosses a block, so every phase of the
  step can be fused into one kernel with `__syncthreads()` between them. The
  price is whatever imbalance the corpus actually has.

The raw spread looks alarming - measured on twelve JSONSchemaBench schemas at
batch 128, one sequence does 11x the mean and 48x the median. It matters much
less than that: the hardware backfills blocks as they retire, and even a heavy
sequence spreads its items over the block's threads. Measured cost of choosing
block-per-sequence: **0.90x at batch 1-4 (it is faster), 1.28x at batch 128**.

Against 12.1 us of dispatch saved by fusing twelve nodes into one, that makes
the fused shape clearly right at small batch and roughly neutral at large -
which is why the runtime picks between them from the batch size, fixed when
the graph is recorded.

    python -m engrain_lab.rigor.imbalance --schemas 12 --batch 128

Re-run this on any new card before trusting the block structure: it is the
one measurement the design is fitted to.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

INSTANCES = Path("results/jsonschemabench-instances.json")


def distribution(schemas: int, batch: int, model: str) -> list[int]:
    """Work items per sequence, with the batch in real parse states."""
    import torch
    from transformers import AutoTokenizer

    import engrain
    import engrain.internals
    from engrain._engine import DeviceGrammar

    tokenizer = AutoTokenizer.from_pretrained(model)
    vocabulary: list[bytes] = []
    for token_id in range(len(tokenizer)):
        piece = tokenizer.convert_ids_to_tokens(token_id)
        try:
            vocabulary.append(tokenizer.convert_tokens_to_string([piece]).encode())
        except Exception:  # noqa: BLE001
            vocabulary.append(b"")

    instances = json.loads(INSTANCES.read_text())["instances"]
    compiler = engrain.internals.Compiler(vocabulary)
    pool = DeviceGrammar()

    admitted = []
    for item in instances:
        if len(admitted) >= schemas:
            break
        try:
            grammar = compiler.compile_json_schema(item["schema"])
        except Exception:  # noqa: BLE001
            continue
        tokens = tokenizer.encode(item["text"], add_special_tokens=False)
        admitted.append((grammar, pool.admit(grammar), tokens))
    if not admitted:
        raise SystemExit("no schema in the corpus compiled")

    device_batch = pool.new_batch(batch)
    device_batch.set_grammars([admitted[i % len(admitted)][1] for i in range(batch)])
    # Walked into real states, and to *different* depths, because a batch where
    # every sequence sits at the same point is the one case with no imbalance
    # to find - and it is not what a serving batch looks like.
    states = {}
    for index in range(batch):
        grammar, _, tokens = admitted[index % len(admitted)]
        matcher = grammar.matcher(32)
        for token in tokens[: 3 + (index * 7) % 23]:
            if not matcher.accept_token(token):
                break
        states[index] = matcher.configurations()
    device_batch.set_batch_configurations(states)
    device_batch.fill_mask()
    torch.cuda.synchronize()

    offsets = device_batch.work_offsets.cpu().numpy().astype("int64")
    per_row = offsets[1:] - offsets[:-1]
    configs = device_batch.configs
    return [
        int(per_row[index * configs : (index + 1) * configs].sum())
        for index in range(batch)
    ]


def report(name: str, values: list[int]) -> None:
    ordered = sorted(values)
    mean = statistics.mean(ordered) or 1
    high = ordered[min(len(ordered) - 1, int(len(ordered) * 0.99))]
    print(
        f"{name:22} n={len(ordered):4}  mean {mean:8.0f}  p50 {ordered[len(ordered)//2]:8.0f}"
        f"  p99 {high:8.0f}  max {ordered[-1]:8.0f}  max/mean {ordered[-1]/mean:5.2f}x"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schemas", type=int, default=12)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--output", type=Path, default=None)
    arguments = parser.parse_args()

    per_sequence = distribution(arguments.schemas, arguments.batch, arguments.model)
    report("per sequence", per_sequence)
    # A block owning several sequences sums their work, so what a tile design
    # would actually see is the spread of these, not of the singles.
    for tile in (2, 4, 8, 16, 32):
        if tile > len(per_sequence):
            break
        report(
            f"per tile of {tile}",
            [
                sum(per_sequence[at : at + tile])
                for at in range(0, len(per_sequence), tile)
            ],
        )
    if arguments.output:
        arguments.output.write_text(json.dumps(per_sequence))
        print(f"wrote {arguments.output}")


if __name__ == "__main__":
    main()
