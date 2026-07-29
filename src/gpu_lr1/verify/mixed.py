"""A batch whose sequences are under different grammars.

The case a serving engine actually has: requests arrive with their own schemas,
so the sequences in one decode step are under different grammars. Each sequence
here walks its own document and is checked against its own CPU matcher, mask row
by mask row and configuration set by configuration set.

Also checks the thing that makes the mixture safe: a sequence must not take
another's mask because their parse states happen to look alike. Two grammars can
have the same state number and the same stack and still admit different tokens.
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

    schemas = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    batch = int(sys.argv[2]) if len(sys.argv) > 2 else 24

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

    pool = []
    chosen = []
    for index in range(len(instances)):
        if len(pool) >= schemas:
            break
        try:
            pool.append(compiler.compile_json_schema(instances[index]["schema"]))
            chosen.append(index)
        except Exception:  # noqa: BLE001
            continue
    print(f"pool of {len(pool)} grammars: schemas {chosen}", flush=True)

    grammar = DeviceGrammar(pool)
    print(
        f"arena: {grammar.resident_bytes()/1e6:.2f} MB, window {grammar.window}, "
        f"groups/state {grammar.max_groups_per_state}, readings {grammar.max_readings}"
    )
    device = grammar.new_batch(batch)
    assignment = [index % len(pool) for index in range(batch)]
    device.set_grammars(assignment)
    print(f"assignment: {assignment}")

    matchers = [pool[assignment[i]].matcher(0) for i in range(batch)]
    streams = [
        tokenizer.encode(instances[chosen[assignment[i]]]["text"], add_special_tokens=False)
        for i in range(batch)
    ]
    alive = [True] * batch

    reference = torch.zeros(grammar.mask_words, dtype=torch.int32)
    failures = []
    checked = 0
    step = 0
    while any(alive) and step < 64:
        live = [i for i in range(batch) if alive[i] and step < len(streams[i])]
        if not live:
            break
        states = {}
        for index in live:
            configurations = matchers[index].configurations()
            if len(configurations) > grammar.max_configs:
                alive[index] = False
                continue
            states[index] = configurations
        if not states:
            break
        device.set_batch_configurations(states)
        masks = device.fill_mask().cpu()
        for index in states:
            reference.zero_()
            matchers[index].fill_bitmask(reference)
            if not torch.equal(masks[index], reference):
                extra = int(((masks[index] & ~reference) != 0).sum())
                missing = int(((reference & ~masks[index]) != 0).sum())
                failures.append(
                    f"step {step} sequence {index} (grammar {assignment[index]}): "
                    f"{extra} words with extra bits, {missing} with missing bits"
                )
                alive[index] = False
            checked += 1

        tokens = torch.zeros(batch, dtype=torch.int32)
        for index in states:
            tokens[index] = streams[index][step]
        device.advance(tokens.cuda())
        for index in list(states):
            if not matchers[index].accept_token(streams[index][step]):
                alive[index] = False
                continue
            expected = sorted(
                (state, tuple(stack)) for state, stack in matchers[index].configurations()
            )
            got = sorted(
                (state, tuple(stack)) for state, stack in device.configurations(index)
            )
            if expected != got and len(got) != grammar.max_configs:
                failures.append(
                    f"step {step} sequence {index} (grammar {assignment[index]}): "
                    f"advance gives {len(got)} configurations, matcher gives "
                    f"{len(expected)}"
                )
                alive[index] = False
        step += 1

    print(f"\n{checked} mask rows checked over {step} steps, {len(failures)} failures")
    for line in failures[:20]:
        print("  " + line)
    if failures:
        raise SystemExit(1)

    # The structural claim: one recording covers any mixture. A CUDA graph is a
    # fixed sequence of launches, and continuous batching changes the batch's
    # composition every step, so a grid derived from the grammars in it could
    # never be recorded once. Here the grid is fixed and the work list is built
    # on the device, so the same graph should serve an assignment it has never
    # seen.
    import random

    rng = random.Random(20260727)
    fresh = [rng.randrange(len(pool)) for _ in range(batch)]
    device.set_grammars(fresh)
    matchers = [pool[fresh[i]].matcher(0) for i in range(batch)]
    device.set_batch_configurations(
        {i: matchers[i].configurations() for i in range(batch)}
    )
    device.fill_mask()
    device.capture()

    swapped = [rng.randrange(len(pool)) for _ in range(batch)]
    device.set_grammars(swapped)
    matchers = [pool[swapped[i]].matcher(0) for i in range(batch)]
    device.set_batch_configurations(
        {i: matchers[i].configurations() for i in range(batch)}
    )
    replayed = device.fill_mask().cpu()
    wrong = 0
    for index in range(batch):
        reference.zero_()
        matchers[index].fill_bitmask(reference)
        if not torch.equal(replayed[index], reference):
            wrong += 1
    print(
        f"graph recorded on assignment {fresh[:8]}..., replayed on "
        f"{swapped[:8]}...: {wrong} of {batch} rows wrong"
    )
    if wrong:
        raise SystemExit(1)

    # What the mixture costs at serving scale.
    for size in (128, 512):
        big = grammar.new_batch(size)
        torch.cuda.synchronize()

        big.set_grammars([rng.randrange(len(pool)) for _ in range(size)])
        states = {}
        for index in range(size):
            matcher = pool[int(big.grammar_of[index])].matcher(0)
            states[index] = matcher.configurations()
        big.set_batch_configurations(states)
        torch.cuda.synchronize()


        def timed(call):
            for _ in range(5):
                call()
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(20):
                call()
            end.record()
            torch.cuda.synchronize()
            return start.elapsed_time(end) * 1e3 / 20

        sampled = torch.zeros(size, dtype=torch.int32, device="cuda")
        # Captured, because that is how a serving loop runs it and because an
        # uncaptured Triton launch costs tens of microseconds of Python before
        # any kernel starts - which measures the launcher, not the step.
        big.fill_mask()
        big.capture()
        big.advance(sampled)
        big.capture_advance()
        fill = timed(big.fill_mask)
        advance = timed(lambda big=big, sampled=sampled: big.advance(sampled))
        total = sum(
            v.numel() * v.element_size()
            for v in vars(big).values()
            if isinstance(v, torch.Tensor) and v.is_cuda
        )
        print(
            f"batch {size:>4} over {len(pool)} grammars: fill {fill:7.1f} us, "
            f"advance {advance:7.1f} us, batch buffers {total/1e6:7.2f} MB, "
            f"arena {grammar.used_bytes()/1e6:.2f} MB used of "
            f"{grammar.resident_bytes()/1e6:.2f} MB held"
        )
        del big
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
