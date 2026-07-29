"""Does the one-launch draft walk agree with walking it a step at a time?

`capture_draft` records a whole draft - every position's advance and mask - as
one graph, so a speculative step is one replay whatever the draft length is.
That is only worth anything if it produces the same masks as doing it the slow
way, and leaves the parse exactly where it found it.

Checked against the reference matcher rather than against ourselves: for each
position the mask must be what a matcher that accepted the draft so far would
fill, and after the walk the configuration set must be untouched.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

INSTANCES = Path("results/jsonschemabench-instances.json")


def main() -> None:
    import gpugrammar
    from transformers import AutoTokenizer

    from gpugrammar._engine import DeviceGrammar

    count = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    length = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    batch = int(sys.argv[3]) if len(sys.argv) > 3 else 8

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    vocabulary: list[bytes] = []
    for token_id in range(len(tokenizer)):
        piece = tokenizer.convert_ids_to_tokens(token_id)
        try:
            vocabulary.append(tokenizer.convert_tokens_to_string([piece]).encode())
        except Exception:  # noqa: BLE001
            vocabulary.append(b"")

    instances = json.loads(INSTANCES.read_text())["instances"]
    compiler = gpugrammar.Compiler(vocabulary)

    failures: list[str] = []
    checked = 0
    for index in range(count):
        instance = instances[index]
        try:
            compiled = compiler.compile_json_schema(instance["schema"])
        except Exception:  # noqa: BLE001
            continue
        tokens = tokenizer.encode(instance["text"], add_special_tokens=False)
        if len(tokens) < length + 4:
            continue
        pool = DeviceGrammar(compiled)
        device = pool.new_batch(batch)

        # Every sequence at its own point in the document, which is what a
        # serving batch looks like and what makes the walk interesting.
        matchers = []
        for row in range(batch):
            matcher = compiled.matcher(0)
            for token in tokens[: 1 + row % 6]:
                if not matcher.accept_token(token):
                    break
            matchers.append(matcher)
        device.set_batch_configurations(
            {row: matcher.configurations() for row, matcher in enumerate(matchers)}
        )
        before = [
            sorted((s, tuple(stack)) for s, stack in device.configurations(row))
            for row in range(batch)
        ]

        device.capture_draft(length)
        draft = torch.tensor(
            [[tokens[(position + row) % len(tokens)] for row in range(batch)]
             for position in range(length)],
            dtype=torch.int32,
            device="cuda",
        )
        masks = device.walk_draft(draft).cpu()

        # The reference: accept the draft one token at a time and fill.
        reference = torch.zeros(pool.mask_words, dtype=torch.int32)
        for row, matcher in enumerate(matchers):
            alive = True
            for position in range(length):
                token = int(draft[position, row])
                if alive:
                    alive = matcher.accept_token(token)
                if not alive:
                    break
                reference.zero_()
                matcher.fill_bitmask(reference)
                if not torch.equal(masks[position, row], reference):
                    extra = int(((masks[position, row] & ~reference) != 0).sum())
                    missing = int(((reference & ~masks[position, row]) != 0).sum())
                    failures.append(
                        f"schema {index} row {row} position {position}: "
                        f"{extra} words too wide, {missing} too narrow"
                    )
                    break
                checked += 1

        after = [
            sorted((s, tuple(stack)) for s, stack in device.configurations(row))
            for row in range(batch)
        ]
        if before != after:
            failures.append(f"schema {index}: the walk moved the parse")
        print(f"schema {index:>3}: {length} positions x {batch} rows", flush=True)
        del device, pool
        torch.cuda.empty_cache()

    print()
    print(f"{checked} draft masks checked against the matcher")
    for line in failures[:10]:
        print("  " + line)
    print("FAIL" if failures else "PASS")


if __name__ == "__main__":
    main()
