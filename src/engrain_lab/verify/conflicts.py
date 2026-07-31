"""Verify the device parser on the grammars that fork.

`verify_device.py` walks the schemas that compiled before the tables kept
conflicts, and on those `PATHS` is 1 and the fork is inert. This walks the
other set: schemas whose ACTION table holds a cell with more than one action,
which are exactly the ones GLR-lite exists for.

The device enumerates derivations in mixed radix; the reference matcher runs an
agenda. They are different searches over the same tree, so agreement is a
claim, not a construction - which is why it is measured here.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

INSTANCES = Path("results/jsonschemabench-instances.json")


def main() -> None:
    import engrain
    from transformers import AutoTokenizer

    from engrain._engine import DeviceGrammar

    want = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    scan = int(sys.argv[2]) if len(sys.argv) > 2 else 200

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    vocabulary: list[bytes] = []
    for token_id in range(len(tokenizer)):
        piece = tokenizer.convert_ids_to_tokens(token_id)
        try:
            vocabulary.append(tokenizer.convert_tokens_to_string([piece]).encode())
        except Exception:  # noqa: BLE001
            vocabulary.append(b"")

    instances = json.loads(INSTANCES.read_text())["instances"]
    compiler = engrain.Compiler(vocabulary)

    found = 0
    failures: list[str] = []
    total_steps = 0
    overflows = 0
    for index in range(min(scan, len(instances))):
        instance = instances[index]
        started = time.time()
        try:
            compiled = compiler.compile_json_schema(instance["schema"])
        except Exception as error:  # noqa: BLE001
            print(f"schema {index:>3}: refused after {time.time() - started:.1f}s "
                  f"({type(error).__name__})", flush=True)
            continue
        arrays = compiled.device_arrays()
        actions = int(arrays.get("max_actions", 1))
        print(f"schema {index:>3}: compiled in {time.time() - started:.1f}s, "
              f"max_actions {actions}", flush=True)
        if actions < 2:
            continue
        found += 1

        grammar = DeviceGrammar(compiled)
        batch = grammar.new_batch(1)
        matcher = compiled.matcher(0)
        reference = torch.zeros(grammar.mask_words, dtype=torch.int32)
        steps = 0
        note = "ok"
        for token in tokenizer.encode(instance["text"], add_special_tokens=False):
            configurations = matcher.configurations()
            if not configurations or len(configurations) > grammar.max_configs:
                note = "configuration ceiling"
                break
            reference.zero_()
            matcher.fill_bitmask(reference)
            batch.set_configurations(0, configurations)
            device = batch.fill_mask()[0].cpu()
            if not torch.equal(device, reference):
                extra = int(((device & ~reference) != 0).sum())
                missing = int(((reference & ~device) != 0).sum())
                failures.append(
                    f"schema {index} step {steps}: mask differs, "
                    f"{extra} words too wide, {missing} too narrow"
                )
                note = "MASK"
                break

            accepted = matcher.accept_token(token)
            batch.advance(torch.tensor([token], dtype=torch.int32, device="cuda"))
            if accepted:
                expected = sorted(
                    (state, tuple(stack)) for state, stack in matcher.configurations()
                )
                got = sorted(
                    (state, tuple(stack)) for state, stack in batch.configurations(0)
                )
                if expected != got and not (
                    len(got) == grammar.max_configs and len(expected) > len(got)
                ):
                    only_device = sorted(set(got) - set(expected))
                    only_host = sorted(set(expected) - set(got))
                    failures.append(
                        f"schema {index} step {steps}: {len(got)} configurations "
                        f"against {len(expected)}; {len(only_device)} only on the "
                        f"device, {len(only_host)} only on the host"
                    )
                    note = "ADVANCE"
                    break
            steps += 1
            if not accepted:
                break
        overflows += int(batch.overflow.sum().item())
        total_steps += steps
        print(
            f"schema {index:>3}: max_actions {int(arrays['max_actions']):>2}  "
            f"paths {grammar.paths:>2}  {steps:>4} steps  {note}",
            flush=True,
        )
        if found >= want:
            break

    print()
    print(f"{found} conflicted schemas, {total_steps} steps, {overflows} overflows")
    for line in failures:
        print("  " + line)
    print("FAIL" if failures else "PASS")


if __name__ == "__main__":
    main()
