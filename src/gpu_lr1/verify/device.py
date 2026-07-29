"""Verify the device parser against the CPU matcher, mask and advance both.

The regression test for any change to the replay. For each schema it walks the
corpus document one token at a time and checks

  - the mask, word for word, against `matcher.fill_bitmask`
  - the configuration set after `advance`, against `matcher.configurations()`

and reports any window or stack overflow the kernels flagged. A change that
narrows the mask shows up here and nowhere else.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve()
INSTANCES = Path("results/jsonschemabench-instances.json")


def main() -> None:
    import gpugrammar
    from transformers import AutoTokenizer

    from gpu_lr1.device_parser import DeviceGrammar

    count = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    # The fill and the advance are separate claims. Checking them together lets
    # one hide the other: an advance rewrites the configuration width, so a fill
    # that would have run too narrow runs wide enough by accident.
    with_advance = "--no-advance" not in sys.argv[2:]
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

    total_steps = 0
    schemas = 0
    deepest_all = 0
    failures = []
    overflows = 0
    for index in range(min(count, len(instances))):
        instance = instances[index]
        try:
            compiled = compiler.compile_json_schema(
                instance["schema"], None, "--exact" in sys.argv[2:]
            )
        except Exception:  # noqa: BLE001
            continue
        grammar = DeviceGrammar(compiled)
        batch = grammar.new_batch(1)
        matcher = compiled.matcher(0)
        reference = torch.zeros(grammar.mask_words, dtype=torch.int32)
        steps = 0
        deepest = 0
        for token in tokenizer.encode(instance["text"], add_special_tokens=False):
            configurations = matcher.configurations()
            deepest = max([deepest] + [len(stack) for _, stack in configurations])
            if len(configurations) > grammar.max_configs:
                break
            reference.zero_()
            matcher.fill_bitmask(reference)
            batch.set_configurations(0, configurations)
            device = batch.fill_mask()[0].cpu()
            if not torch.equal(device, reference):
                differing = int((device != reference).sum())
                extra = int(((device & ~reference) != 0).sum())
                missing = int(((reference & ~device) != 0).sum())
                failures.append(
                    f"schema {index} step {steps}: mask differs in {differing} words "
                    f"({extra} words with extra bits, {missing} with missing bits)"
                )
                break

            accepted = matcher.accept_token(token)
            if with_advance:
                batch.advance(torch.tensor([token], dtype=torch.int32, device="cuda"))
                if accepted:
                    expected = sorted(
                        (state, tuple(stack))
                        for state, stack in matcher.configurations()
                    )
                    got = sorted(
                        (state, tuple(stack)) for state, stack in batch.configurations(0)
                    )
                    if expected != got and not (
                        len(got) == grammar.max_configs and len(expected) > len(got)
                    ):
                        failures.append(
                            f"schema {index} step {steps}: advance gives {len(got)} "
                            f"configurations, matcher gives {len(expected)}"
                        )
                        break
            steps += 1
            if not accepted:
                break
        overflows += int(batch.overflow.sum().item())
        total_steps += steps
        schemas += 1
        high = int(batch.high_water.item())
        print(
            f"schema {index:>3}: {steps:>4} steps ok  "
            f"window {grammar.window:>3} (bound {grammar.window_bound:>5} run by run, "
            f"widest reading {grammar.max_reading_terms} terminals), "
            f"deepest excursion used {high:>3}, "
            f"deepest stack {deepest:>3}, "
            f"scratch {batch.scratch.numel() * 4 / 1e6:.2f} MB",
            flush=True,
        )
        deepest_all = max(deepest_all, deepest)

    print(f"\n{schemas} schemas, {total_steps} steps, {len(failures)} failures")
    print(f"deepest parser stack over all of them: {deepest_all}")
    print(f"overflow flags raised: {overflows}")
    for line in failures:
        print("  " + line)
    if failures or overflows:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
