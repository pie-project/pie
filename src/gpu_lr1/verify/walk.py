"""Walk the grammar at random and compare masks at every state reached.

A corpus document visits one path through a grammar. The serving path found a
disagreement at a state that path never reaches, which is the failure mode a
fixed document cannot catch: the model tokenises differently from the corpus, so
it arrives at states by routes the corpus does not take.

So the walk is random and driven by the mask itself - at every step choose one
of the tokens the reference matcher admits, feed it to both, and compare. Over
many restarts that covers far more of the reachable state space than any
document does.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import torch

INSTANCES = Path("results/jsonschemabench-instances.json")


def allowed_tokens(row: torch.Tensor, vocabulary: int) -> list[int]:
    bits = torch.nonzero(
        ((row.view(-1, 1) >> torch.arange(32)) & 1).reshape(-1)[:vocabulary]
    ).reshape(-1)
    return bits.tolist()


def main() -> None:
    import gpugrammar
    from transformers import AutoTokenizer

    from gpu_lr1.device_parser import DeviceGrammar

    schemas = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    walks = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    length = int(sys.argv[3]) if len(sys.argv) > 3 else 40

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    vocabulary: list[bytes] = []
    for token_id in range(len(tokenizer)):
        piece = tokenizer.convert_ids_to_tokens(token_id)
        try:
            vocabulary.append(tokenizer.convert_tokens_to_string([piece]).encode())
        except Exception:  # noqa: BLE001
            vocabulary.append(b"")

    if "--smoke" in sys.argv:
        # The schemas the vLLM smoke test uses. Closed objects, which the corpus
        # has few of, and which is where the serving path disagreed.
        instances = [
            {"schema": json.dumps(schema)}
            for schema in (
                {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "age": {"type": "integer"},
                        "active": {"type": "boolean"},
                    },
                    "required": ["name", "age", "active"],
                    "additionalProperties": False,
                },
                {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "pages": {"type": "integer"},
                    },
                    "required": ["title", "pages"],
                    "additionalProperties": False,
                },
                {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                        "population": {"type": "integer"},
                    },
                    "required": ["city", "population"],
                    "additionalProperties": False,
                },
            )
        ]
    else:
        instances = json.loads(INSTANCES.read_text())["instances"]
    compiler = gpugrammar.Compiler(vocabulary)
    rng = random.Random(20260727)

    total_steps = 0
    failures = []
    for index in range(schemas):
        try:
            compiled = compiler.compile_json_schema(instances[index]["schema"])
        except Exception:  # noqa: BLE001
            continue
        pool = DeviceGrammar(compiled)
        batch = pool.new_batch(1)
        reference = torch.zeros(pool.mask_words, dtype=torch.int32)
        steps = 0
        seen = set()
        for _ in range(walks):
            matcher = compiled.matcher(0)
            for _ in range(length):
                configurations = matcher.configurations()
                if len(configurations) > pool.max_configs:
                    break
                seen.add(
                    tuple(sorted((s, tuple(k)) for s, k in configurations))
                )
                reference.zero_()
                matcher.fill_bitmask(reference)
                batch.set_configurations(0, configurations)
                device = batch.fill_mask()[0].cpu()
                if not torch.equal(device, reference):
                    extra = int(((device & ~reference) != 0).sum())
                    missing = int(((reference & ~device) != 0).sum())
                    failures.append(
                        f"schema {index}: {extra} words with extra bits, "
                        f"{missing} with missing, at {configurations}"
                    )
                    break
                choices = allowed_tokens(reference, pool.vocab_size)
                if not choices:
                    break
                if not matcher.accept_token(rng.choice(choices)):
                    break
                steps += 1
        total_steps += steps
        print(
            f"schema {index:>3}: {steps:>5} steps over {walks} walks, "
            f"{len(seen)} distinct states, "
            f"{sum(1 for f in failures if f.startswith(f'schema {index}:'))} failures",
            flush=True,
        )

    print(f"\n{total_steps} steps, {len(failures)} failures")
    for line in failures[:10]:
        print("  " + line)
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
