"""Compare the two engines on identical states, which is the only fair way.

`rigor.agreement` showed there is no schema the two lower to the same language
and no whitespace setting that makes one - so any measurement that lets each
engine *generate* is partly a measurement of its grammar. That is fine for
"which produces valid documents", where the schema is the referee, and it is
not fine for "which computes a mask faster".

So this drives both through the *same* real documents, token by token, and
times each one's mask at every step. Neither engine chooses anything: the
sequence comes from the corpus, the states are identical by construction, and
what is left is the cost of answering the same question.

The documents are re-serialised without whitespace, which removes the one
difference that survives every configuration.

    python -m engrain_lab.rigor.lockstep --batches 1 32 128 512
"""

from __future__ import annotations

import argparse
import functools
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

RESULTS = Path("results")
OUT = RESULTS / "lockstep.json"


def _fill_rows(matchers, bitmask, rows) -> None:
    """One thread's share of a step, the way vLLM slices it."""
    for row in rows:
        matchers[row].fill_next_token_bitmask(bitmask, row)


def main() -> int:
    import torch
    import xgrammar as xg
    from transformers import AutoTokenizer

    from engrain.internals import Compiler, DeviceGrammar

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 32, 128, 512])
    parser.add_argument("--schemas", type=int, default=64)
    parser.add_argument("--steps", type=int, default=48)
    parser.add_argument(
        "--warmup",
        type=int,
        default=8,
        help="steps to run before timing. Triton compiles on first use and a "
        "device batch's buffers are cold, so the first steps measure neither "
        "engine.",
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--forward",
        action="store_true",
        help="also time an unconstrained decode step at each batch, and report "
        "what share of it each engine's mask is. A ratio between two grammar "
        "engines is not a system result: if theirs is 5% of the step, making "
        "it 3x cheaper saves 3%.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="threads to fill XGrammar's rows with, in chunks of sixteen, "
        "which is what vLLM does above a batch of 128. Their per-row work is "
        "the same either way; this is the difference between counting it and "
        "waiting for it, and serving waits.",
    )
    parser.add_argument(
        "--distinct",
        type=int,
        nargs="+",
        default=None,
        help="how many distinct schemas the batch draws from, cycled to fill "
        "it. One is the case our fill deduplicates and theirs cannot; all of "
        "them is the case that should hurt us. Default: as many as there are.",
    )
    arguments = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(arguments.model)
    vocabulary = []
    for token_id in range(len(tokenizer)):
        piece = tokenizer.convert_ids_to_tokens(token_id)
        try:
            vocabulary.append(tokenizer.convert_tokens_to_string([piece]).encode())
        except Exception:  # noqa: BLE001
            vocabulary.append(b"")

    ours = Compiler(vocabulary)
    info = xg.TokenizerInfo.from_huggingface(tokenizer, vocab_size=len(tokenizer))
    xgc = xg.GrammarCompiler(info)
    instances = json.loads(
        (RESULTS / "jsonschemabench-instances.json").read_text()
    )["instances"]

    # A schema, a grammar from each engine, and a document both accept. All
    # three have to hold or the pair is dropped: a state only one engine can
    # reach is not a state to time.
    pairs = []
    for instance in instances:
        if len(pairs) >= arguments.schemas:
            break
        try:
            document = json.dumps(
                json.loads(instance["text"]), separators=(",", ":")
            )
            grammar = ours.compile_json_schema(instance["schema"])
            compiled = xgc.compile_json_schema(
                instance["schema"], any_whitespace=True, any_order=True
            )
        except Exception:  # noqa: BLE001
            continue
        tokens = tokenizer(document, add_special_tokens=False).input_ids
        # Long enough to outlast the warmup, or one short document in the
        # batch ends the walk for every row before anything is timed.
        if len(tokens) < arguments.warmup + arguments.steps // 2:
            continue
        mine = grammar.matcher(0)
        yours = xg.GrammarMatcher(compiled, terminate_without_stop_token=True)
        if not all(mine.accept_token(t) and yours.accept_token(t) for t in tokens):
            continue
        pairs.append((grammar, compiled, tokens))
    if not pairs:
        print("no schema had a document both engines accept")
        return 1
    print(
        f"{len(pairs)} schemas with a document both engines accept and long "
        f"enough to time"
    )

    forward: dict[int, float] = {}
    if arguments.forward:
        from engrain_lab.rigor.latency import forward_pass_cost

        forward = forward_pass_cost(arguments.model, arguments.batches, 20)
        if not forward:
            print("no model weights, so no denominator")

    report = []
    spreads = arguments.distinct or [len(pairs)]
    print(
        f"\n{'batch':>6} {'distinct':>9} {'engrain':>12} {'xg serial':>12} "
        f"{'xg threaded':>14} {'ratio':>7}"
        + (f" {'our share':>8} {'their share':>9}" if arguments.forward else "")
    )
    for batch, distinct in ((b, d) for b in arguments.batches for d in spreads):
        spread = max(1, min(distinct, len(pairs), batch))
        chosen = [pairs[index % spread] for index in range(batch)]
        pool = DeviceGrammar()
        try:
            assignment = []
            seen: dict[int, int] = {}
            for grammar, _, _ in chosen:
                key = id(grammar)
                if key not in seen:
                    seen[key] = pool.admit(grammar)
                assignment.append(seen[key])
            device = pool.new_batch(batch)
            device.set_grammars(assignment)

            bitmask = xg.allocate_token_bitmask(batch, len(tokenizer))
            ourtimes, theirtimes, theirthreaded = [], [], []
            pool_of_threads = (
                ThreadPoolExecutor(max_workers=arguments.workers)
                if arguments.workers > 1 and batch > 16
                else None
            )
            for _ in range(arguments.repeats):
                mine = [grammar.matcher(0) for grammar, _, _ in chosen]
                yours = [
                    xg.GrammarMatcher(compiled, terminate_without_stop_token=True)
                    for _, compiled, _ in chosen
                ]
                for step in range(arguments.steps):
                    if any(step >= len(tokens) for _, _, tokens in chosen):
                        break
                    timed = step >= arguments.warmup
                    # Ours: seed the device from the host matchers and fill.
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                    device.set_matchers(mine)
                    device.fill_mask()[:batch].to("cpu", non_blocking=False)
                    if timed:
                        ourtimes.append((time.perf_counter() - started) * 1e6)
                    # Theirs: one host call per row, which is their whole
                    # step - serially for the work, and over vLLM's thread
                    # pool for the latency a serving engine actually sees.
                    started = time.perf_counter()
                    for row in range(batch):
                        yours[row].fill_next_token_bitmask(bitmask, row)
                    serial = (time.perf_counter() - started) * 1e6
                    started = time.perf_counter()
                    if pool_of_threads is None:
                        threaded = serial
                    else:
                        chunks = [
                            range(at, min(at + 16, batch))
                            for at in range(0, batch, 16)
                        ]
                        list(
                            pool_of_threads.map(
                                functools.partial(_fill_rows, yours, bitmask),
                                chunks,
                            )
                        )
                        threaded = (time.perf_counter() - started) * 1e6
                    if timed:
                        theirtimes.append(serial)
                        theirthreaded.append(threaded)
                    for row, (_, _, tokens) in enumerate(chosen):
                        token = tokens[step]
                        mine[row].accept_token(token)
                        yours[row].accept_token(token)
            if not ourtimes:
                continue
            if pool_of_threads is not None:
                pool_of_threads.shutdown()
            us = statistics.median(ourtimes)
            them = statistics.median(theirtimes)
            threaded = statistics.median(theirthreaded)
            step = forward.get(batch)
            shares = ""
            if step:
                shares = (
                    f" {us / (step + us) * 100:>7.1f}%"
                    f" {threaded / (step + threaded) * 100:>8.1f}%"
                )
            print(
                f"{batch:>6} {spread:>9} {us:>11.1f}u {them:>11.1f}u "
                f"{threaded:>13.1f}u {us / threaded:>7.2f}{shares}"
            )
            report.append(
                {
                    "batch": batch,
                    "distinct": spread,
                    "engrain_us": us,
                    "xgrammar_serial_us": them,
                    "xgrammar_threaded_us": threaded,
                    "forward_us": forward.get(batch),
                }
            )
        finally:
            del pool
            torch.cuda.empty_cache()

    RESULTS.mkdir(exist_ok=True)
    OUT.write_text(
        json.dumps({"schemas": len(pairs), "rows": report}, indent=2)
    )
    print(f"\nwritten to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
