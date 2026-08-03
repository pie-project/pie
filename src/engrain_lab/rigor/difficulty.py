"""What makes a schema expensive for us and not for XGrammar.

End to end we win on one easy schema at batch 512 and lose 1.39x on one
corpus schema, with the two engines generating almost the same number of
tokens - so that gap is per step, and "the schema is harder" is not an
explanation. This measures what actually differs.

Each schema is driven the way serving drives it: 512 independent rows, each
walking its own random path through its own mask, so the rows diverge exactly
as separate requests do. Per step it records what the fill has to do - how
many configurations a row carries, and how many distinct parse states the
batch holds, which is what the fill deduplicates - beside the time both
engines take.

    python -m engrain_lab.rigor.difficulty --schemas 40 --steps 48
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np
import torch

RESULTS = Path("results")
OUT = RESULTS / "difficulty.json"

# The schema the end-to-end run calls easy: three required scalars, closed,
# and nothing in it that can grow without bound.
EASY = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
        "active": {"type": "boolean"},
    },
    "required": ["name", "age", "active"],
    "additionalProperties": False,
}


def _unbounded(node, depth: int = 0) -> bool:
    """Whether the schema's language contains arbitrarily long documents.

    An array with no `maxItems` and an object with open `additionalProperties`
    both admit documents of any length, which is what decides how many decode
    steps a request takes rather than how much each one costs.
    """
    if depth > 12 or not isinstance(node, dict):
        return False
    if node.get("type") == "array" and "maxItems" not in node:
        return True
    if node.get("type") == "object" and node.get("additionalProperties") is not False:
        return True
    for key in ("properties", "$defs", "definitions", "patternProperties"):
        child = node.get(key)
        if isinstance(child, dict) and any(
            _unbounded(value, depth + 1) for value in child.values()
        ):
            return True
    for key in ("items", "additionalProperties", "contains"):
        if _unbounded(node.get(key), depth + 1):
            return True
    for key in ("anyOf", "oneOf", "allOf"):
        branch = node.get(key)
        if isinstance(branch, list) and any(
            _unbounded(value, depth + 1) for value in branch
        ):
            return True
    return False


def _walk_xgrammar(matchers, bitmask, steps, rng, vocabulary_size):
    """The same walk through XGrammar, timed the way vLLM calls it.

    Their backend has no batched entry point: a step is one call per row, on
    the host. That is the cost our fill is meant to replace, so it belongs
    beside it rather than in a separate run.
    """
    import time

    import xgrammar as xg

    micros = []
    accepts = []
    alive = list(range(len(matchers)))
    for _ in range(steps):
        if not alive:
            break
        start = time.perf_counter()
        for row in range(len(matchers)):
            matchers[row].fill_next_token_bitmask(bitmask, row)
        micros.append((time.perf_counter() - start) * 1e6)

        mask = bitmask.numpy()
        still = []
        chosen = []
        for row in alive:
            words = np.flatnonzero(mask[row])
            if words.size == 0:
                continue
            word = int(words[rng.integers(words.size)])
            bits = np.flatnonzero(
                np.unpackbits(
                    np.array([mask[row][word]], dtype=np.int32).view(np.uint8),
                    bitorder="little",
                )
            )
            allowed = word * 32 + bits
            allowed = allowed[allowed < vocabulary_size]
            if allowed.size == 0:
                continue
            chosen.append((row, int(allowed[rng.integers(allowed.size)])))
        accepted = time.perf_counter()
        for row, token in chosen:
            if matchers[row].accept_token(token):
                still.append(row)
        accepts.append((time.perf_counter() - accepted) * 1e6)
        alive = still
    del xg
    if not micros:
        return None
    return statistics.median(micros), statistics.median(accepts)


def _walk(matchers, device, batch, steps, rng, vocabulary_size):
    """Drive every row down its own path, recording what the batch looks like.

    Returns the per-step medians. A row that can no longer move is left where
    it is; that is what a finished request does too.
    """
    import time as _time

    configs = []
    distinct = []
    micros = []
    seeds = []
    accepts = []
    alive = list(range(len(matchers)))
    for _ in range(steps):
        if not alive:
            break
        seeded = _time.perf_counter()
        device.set_matchers(matchers)
        torch.cuda.synchronize()
        seeds.append((_time.perf_counter() - seeded) * 1e6)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(5):
            device.fill_mask()
        end.record()
        torch.cuda.synchronize()
        micros.append(start.elapsed_time(end) * 1000.0 / 5)

        states = set()
        total = 0
        for matcher in matchers:
            held = matcher.configurations()
            total += len(held)
            for lexer_state, stack in held:
                states.add((lexer_state, tuple(stack)))
        configs.append(total / len(matchers))
        distinct.append(len(states))

        mask = device.fill_mask()[: len(matchers)].cpu().numpy()
        still = []
        chosen = []
        for row in alive:
            # Only the word the token comes from is unpacked. Unpacking the
            # whole row is 151k bits per row per step and turns the probe into
            # a measurement of numpy.
            words = np.flatnonzero(mask[row])
            if words.size == 0:
                continue
            word = int(words[rng.integers(words.size)])
            bits = np.flatnonzero(
                np.unpackbits(
                    np.array([mask[row][word]], dtype=np.uint32).view(np.uint8),
                    bitorder="little",
                )
            )
            allowed = word * 32 + bits
            allowed = allowed[allowed < vocabulary_size]
            if allowed.size == 0:
                continue
            chosen.append((row, int(allowed[rng.integers(allowed.size)])))
        # vLLM advances the matcher itself, once per row per step, because the
        # interface owns token acceptance. It is the one part of our step that
        # is not on the device, and nothing else in this file measures it.
        accepted = _time.perf_counter()
        for row, token in chosen:
            if matchers[row].accept_token(token):
                still.append(row)
        accepts.append((_time.perf_counter() - accepted) * 1e6)
        alive = still
    if not micros:
        return None
    return {
        "fill_us": statistics.median(micros),
        "seed_us": statistics.median(seeds),
        "accept_us": statistics.median(accepts),
        "configs_per_row": statistics.median(configs),
        "distinct_states": statistics.median(distinct),
        "steps_walked": len(micros),
    }


def main() -> int:
    from transformers import AutoTokenizer

    from engrain.internals import Compiler, DeviceGrammar

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--steps", type=int, default=48)
    parser.add_argument("--schemas", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260803)
    arguments = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(arguments.model)
    vocabulary = []
    for token_id in range(len(tokenizer)):
        piece = tokenizer.convert_ids_to_tokens(token_id)
        try:
            vocabulary.append(tokenizer.convert_tokens_to_string([piece]).encode())
        except Exception:  # noqa: BLE001
            vocabulary.append(b"")

    import xgrammar as xg

    ours = Compiler(vocabulary)
    info = xg.TokenizerInfo.from_huggingface(tokenizer, vocab_size=len(tokenizer))
    xgc = xg.GrammarCompiler(info)
    instances = json.loads(
        (RESULTS / "jsonschemabench-instances.json").read_text()
    )["instances"]

    candidates = [("easy", EASY)]
    for index, instance in enumerate(instances):
        if len(candidates) > arguments.schemas:
            break
        try:
            candidates.append((f"corpus{index}", json.loads(instance["schema"])))
        except Exception:  # noqa: BLE001
            continue

    rng = np.random.default_rng(arguments.seed)
    rows = []
    print(
        f"{'schema':>10} {'fill':>8} {'seed':>8} {'accept':>9} {'ours':>9} | "
        f"{'xg fill':>9} {'xg acc':>8} {'xg':>9} {'ratio':>6} | "
        f"{'p.states':>9} {'relaxed':>8} {'unbnd':>6}"
    )
    for name, schema in candidates:
        try:
            grammar = ours.compile_json_schema(json.dumps(schema), max_digits=8)
        except Exception:  # noqa: BLE001
            continue
        pool = DeviceGrammar()
        try:
            pool.admit(grammar)
            device = pool.new_batch(arguments.batch)
            device.set_grammars([0] * arguments.batch)
            matchers = [grammar.matcher(0) for _ in range(arguments.batch)]
            walked = _walk(
                matchers,
                device,
                arguments.batch,
                arguments.steps,
                rng,
                len(tokenizer),
            )
        finally:
            del pool
            torch.cuda.empty_cache()
        if walked is None:
            continue
        theirs = None
        try:
            compiled = xgc.compile_json_schema(json.dumps(schema), any_whitespace=True)
            bitmask = xg.allocate_token_bitmask(arguments.batch, len(tokenizer))
            theirs, their_accept = _walk_xgrammar(
                [
                    xg.GrammarMatcher(compiled, terminate_without_stop_token=True)
                    for _ in range(arguments.batch)
                ],
                bitmask,
                arguments.steps,
                np.random.default_rng(arguments.seed),
                len(tokenizer),
            )
        except Exception:  # noqa: BLE001
            theirs = their_accept = None
        row = {
            "schema": name,
            "xgrammar_fill_us": theirs,
            "xgrammar_accept_us": their_accept,
            "xgrammar_us": (theirs + their_accept) if theirs is not None else None,
            "ours_us": (
                walked["fill_us"] + walked["seed_us"] + walked["accept_us"]
            ),
            **walked,
            "groups": grammar.num_groups,
            "parser_states": grammar.num_parser_states,
            "resident_kib": grammar.resident_bytes / 1024,
            "relaxed": bool(grammar.relaxations),
            "unbounded": _unbounded(schema),
        }
        rows.append(row)
        ratio = (
            row["ours_us"] / row["xgrammar_us"] if row["xgrammar_us"] else float("nan")
        )
        nan = float("nan")
        print(
            f"{name:>10} {row['fill_us']:>8.1f} {row['seed_us']:>8.1f} "
            f"{row['accept_us']:>9.1f} {row['ours_us']:>9.1f} | "
            f"{(row['xgrammar_fill_us'] or nan):>9.1f} "
            f"{(row['xgrammar_accept_us'] or nan):>8.1f} "
            f"{(row['xgrammar_us'] or nan):>9.1f} {ratio:>6.2f} | "
            f"{row['parser_states']:>9} "
            f"{str(row['relaxed']):>8} {str(row['unbounded']):>6}"
        )

    RESULTS.mkdir(exist_ok=True)
    OUT.write_text(json.dumps({"batch": arguments.batch, "rows": rows}, indent=2))
    print(f"\nwritten to {OUT}")

    if len(rows) > 3:
        fill = np.array([row["ours_us"] for row in rows])
        print("\nwhat predicts our per-step cost, by correlation:")
        for key in (
            "configs_per_row",
            "distinct_states",
            "groups",
            "parser_states",
            "resident_kib",
            "fill_us",
            "seed_us",
            "accept_us",
        ):
            values = np.array([float(row[key]) for row in rows])
            if values.std() == 0:
                continue
            print(f"  {key:>16}: {np.corrcoef(values, fill)[0, 1]:+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
