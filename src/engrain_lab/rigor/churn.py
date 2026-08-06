"""Rows that join and leave between steps, which is the shape nothing else tests.

`engrain_lab.verify.mixed` changes which grammar each sequence is under between
recording a graph and replaying it, but every row is present for the whole run.
Continuous batching does something else: a row is retired mid-flight and the
next request takes its place, at a different document position, possibly under
a different grammar, while every other row keeps going.

`rigor.online` showed that this produces a wrong mask - a row missing twelve
words of bits against the same row computed alone - and this reproduces it away
from vLLM so it can be bisected.

    python -m engrain_lab.rigor.churn --steps 200 --rows 32
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

RESULTS = Path("results")


def main() -> int:
    import torch

    import engrain.internals as E
    from engrain_lab.rigor.harness import load_vocabulary

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--corpus", default=str(RESULTS / "corpus-exact.json"))
    parser.add_argument("--schemas", type=int, default=12)
    parser.add_argument("--rows", type=int, default=32)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--churn", type=float, default=0.15,
                        help="chance per row per step that it is retired and replaced")
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--max-configs", type=int, default=8)
    parser.add_argument("--no-memo", action="store_true")
    parser.add_argument("--no-graph", action="store_true")
    parser.add_argument("--vary-live", action="store_true",
                        help="let the number of live rows change every step while "
                             "the device batch stays full width, which is what a "
                             "scheduler does and what a fixed-width probe cannot see")
    parser.add_argument("--reset-every-step", action="store_true",
                        help="call set_grammars every step instead of only when the "
                             "assignment changes, which is what the vLLM backend "
                             "avoids for speed")
    arguments = parser.parse_args()

    vocabulary = load_vocabulary(arguments.model)
    compiler = E.Compiler(vocabulary)
    instances = json.loads(Path(arguments.corpus).read_text())["instances"]

    grammars: list[Any] = []
    for instance in instances:
        if len(grammars) >= arguments.schemas:
            break
        try:
            grammars.append(compiler.compile_json_schema(instance["schema"]))
        except Exception:  # noqa: BLE001
            continue
    pool = E.DeviceGrammar(max_configs=arguments.max_configs)
    ids = [pool.admit(grammar) for grammar in grammars]
    print(f"{len(grammars)} grammars, {arguments.rows} rows, {arguments.steps} steps")

    batch = pool.new_batch(arguments.rows)
    if arguments.no_memo:
        batch.memo_slots = 0
    rng = random.Random(arguments.seed)

    # Each row holds a live sequence: a grammar and a reference matcher walking
    # a document of its own.
    def fresh(row: int):
        which = rng.randrange(len(grammars))
        return {"which": which, "matcher": grammars[which].matcher(0), "steps": 0}

    live = [fresh(row) for row in range(arguments.rows)]
    assigned: list[int] | None = None
    failures = 0
    churned = 0
    grew = 0

    for step in range(arguments.steps):
        # Retire some rows and let new ones take their place, which is the one
        # thing the six verifications never do.
        for row in range(arguments.rows):
            if step and rng.random() < arguments.churn:
                live[row] = fresh(row)
                churned += 1

        # vLLM's device batch is `max_num_seqs` wide for the whole run, and the
        # number of live requests in a step is whatever the scheduler gave it.
        # So the grammar list is full width and the matcher list is a varying
        # prefix - the shape the fixed-width probe above never produced.
        alive = (
            rng.randint(max(1, arguments.rows // 4), arguments.rows)
            if arguments.vary_live
            else arguments.rows
        )
        here = live[:alive]
        wanted = [ids[entry["which"]] for entry in here]
        wanted = wanted + [wanted[0]] * (arguments.rows - alive)
        if arguments.reset_every_step or wanted != assigned:
            batch.set_grammars(wanted)
            assigned = list(wanted)
        # The batch grows its configuration ceiling mid-flight, which is what
        # the vLLM backend does rather than refusing: a parse that carries more
        # configurations than the batch was built for rebuilds it wider. That
        # rebuild lands in the middle of a run with rows at every document
        # position, and it is the one thing the fixed-width probe above did
        # not do.
        while True:
            try:
                batch.set_matchers([entry["matcher"] for entry in here])
                break
            except E.ConfigurationsExceeded as refusal:
                wider = max(int(refusal.needed), 1)
                pool.max_configs = max(pool.max_configs * 2, 1 << wider.bit_length())
                grew += 1
                batch = pool.new_batch(arguments.rows)
                if arguments.no_memo:
                    batch.memo_slots = 0
                batch.set_grammars(wanted)
                assigned = list(wanted)
        if arguments.no_graph:
            batch.graph = None
        device = batch.fill_mask()[:alive].to("cpu")
        _, flags = batch.problems()
        narrowed = flags[:alive].to("cpu").bool()

        for row, entry in enumerate(here):
            if narrowed[row]:
                continue
            reference = torch.zeros(pool.mask_words, dtype=torch.int32)
            entry["matcher"].fill_bitmask(reference)
            if torch.equal(device[row], reference):
                continue
            failures += 1
            extra = int(((device[row] & ~reference) != 0).sum())
            missing = int(((reference & ~device[row]) != 0).sum())
            print(
                f"step {step} row {row} grammar {wanted[row]} "
                f"(alive {entry['steps']} steps): {extra} words extra, "
                f"{missing} missing"
            )
            if failures >= 5:
                print(
                    f"stopping after {failures}; {churned} replacements, "
                    f"{grew} ceiling growths"
                )
                return 1

        # Advance every row by a token its own mask allows.
        for row, entry in enumerate(live):
            allowed = entry["matcher"].allowed_tokens()
            if not allowed:
                live[row] = fresh(row)
                continue
            entry["matcher"].accept_token(rng.choice(allowed))
            entry["steps"] += 1

    print(
        f"{arguments.steps} steps, {churned} row replacements, "
        f"{grew} ceiling growths, {failures} failures"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
