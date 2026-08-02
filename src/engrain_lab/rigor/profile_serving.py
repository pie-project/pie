"""Where the end-to-end loss goes (q08 follow-up).

End to end we are 0.92 s against XGrammar's 0.85 on 256 requests, on a workload
that favours us at every turn: a 0.6B model, so the forward pass is small and
the grammar's share of a step is as large as it ever gets; three-property
schemas, which is where our per-step measurements win by 5-9x; and batch 256,
which is our best regime. Losing there is worth explaining rather than
averaging away.

The suspicion this measures is that the loss is not in the kernels but in the
interface around them. vLLM's `StructuredOutputBackend` owns token acceptance
and hands the backend a *host* bitmask, so our backend keeps host matchers as
the source of truth, re-seeds the device from them every step, and copies the
mask back. That is XGrammar's whole per-step cost, plus a host-to-device
transfer, plus our kernels, plus a device-to-host transfer.

So this times the five pieces of one serving step separately, at the batch
sizes the end-to-end run used, and reports what each is worth. What is
attributable to the design is `fill`; everything else is the price of speaking
through an interface that predates it.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

INSTANCES = Path("results/jsonschemabench-instances.json")
OUT = Path("results/e2e-profile.json")

SCHEMAS = [
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
            "author": {"type": "string"},
        },
        "required": ["title", "pages", "author"],
        "additionalProperties": False,
    },
    {
        "type": "object",
        "properties": {
            "city": {"type": "string"},
            "population": {"type": "integer"},
            "country": {"type": "string"},
        },
        "required": ["city", "population", "country"],
        "additionalProperties": False,
    },
]


def timed(call, warmup: int, repeats: int, sync) -> float:
    for _ in range(warmup):
        call()
    sync()
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        call()
        sync()
        samples.append((time.perf_counter() - started) * 1e6)
    return statistics.median(sorted(samples))


def main() -> int:
    import torch
    import xgrammar as xg
    from transformers import AutoTokenizer

    from engrain.internals import Compiler, DeviceGrammar

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--batches", type=int, nargs="+", default=[16, 64, 256])
    parser.add_argument("--repeats", type=int, default=40)
    parser.add_argument(
        "--unique",
        action="store_true",
        help="give every row its own schema, drawn from the corpus. The three "
        "built-in schemas share every row's grammar and parse state, which is "
        "exactly what the fill deduplicates - so the shared case is the "
        "flattering one and this is the case to attribute a loss in.",
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

    schemas = SCHEMAS
    if arguments.unique:
        from engrain_lab.rigor.e2e import _agreed_schemas

        schemas = _agreed_schemas()[: max(arguments.batches)]

    ours = Compiler(vocabulary)
    pool = DeviceGrammar()
    compiled = []
    kept = []
    for schema in schemas:
        try:
            compiled.append(ours.compile_json_schema(json.dumps(schema), max_digits=8))
        except Exception:  # noqa: BLE001
            continue
        kept.append(schema)
    schemas = kept
    for grammar in compiled:
        pool.admit(grammar)
    print(f"{len(schemas)} schemas, {pool.resident_bytes() / 2**20:.1f} MiB resident")

    info = xg.TokenizerInfo.from_huggingface(tokenizer, vocab_size=len(tokenizer))
    xgc = xg.GrammarCompiler(info)
    theirs = [xgc.compile_json_schema(json.dumps(s), any_whitespace=True) for s in schemas]

    sync = torch.cuda.synchronize
    seed = json.loads(INSTANCES.read_text())["instances"][0]
    del seed

    print(
        f"{'batch':>6} {'host advance':>13} {'set_matchers':>13} {'fill':>9} "
        f"{'mask to host':>13} {'row copies':>11} | {'was':>10} {'now':>10} "
        f"{'xgrammar':>9}"
    )
    report = []
    for batch in arguments.batches:
        assignment = [index % len(schemas) for index in range(batch)]
        matchers = []
        for index in assignment:
            matcher = compiled[index].matcher(0)
            for byte in b'{"':
                matcher.accept_token(byte)
            matchers.append(matcher)

        device = pool.new_batch(batch)
        device.set_grammars(assignment)
        device.set_matchers(matchers)
        device.fill_mask()
        device.capture()

        bitmask = xg.allocate_token_bitmask(batch, len(tokenizer))

        def advance_hosts(matchers=matchers) -> None:
            for matcher in matchers:
                matcher.rollback(0)

        def seed_device(device=device, matchers=matchers) -> None:
            device.set_matchers(matchers)

        def fill(device=device) -> None:
            device.fill_mask()

        def to_host(device=device, batch=batch) -> None:
            device.fill_mask()[:batch].to("cpu", non_blocking=False)

        held = None

        def rows(device=device, batch=batch, bitmask=bitmask) -> None:
            """The loop the backend used to run: one tensor copy per row."""
            nonlocal held
            if held is None:
                held = device.fill_mask()[:batch].to("cpu", non_blocking=False)
            width = held.shape[1]
            for row in range(batch):
                bitmask[row, :width].copy_(held[row])
                bitmask[row, width:].zero_()

        def rows_vectorised(device=device, batch=batch, bitmask=bitmask) -> None:
            """And what replaced it: one index_copy_ for the whole batch."""
            nonlocal held
            if held is None:
                held = device.fill_mask()[:batch].to("cpu", non_blocking=False)
            width = held.shape[1]
            rowsel = torch.arange(batch, dtype=torch.long)
            bitmask[:, :width].index_copy_(0, rowsel, held)
            if bitmask.shape[1] > width:
                bitmask[:, width:].index_fill_(0, rowsel, 0)

        nothing = lambda: None  # noqa: E731
        host_us = timed(advance_hosts, 4, arguments.repeats, nothing)
        seed_us = timed(seed_device, 4, arguments.repeats, sync)
        fill_us = timed(fill, 4, arguments.repeats, sync)
        copy_us = timed(to_host, 4, arguments.repeats, sync) - fill_us
        rows_us = timed(rows, 4, arguments.repeats, nothing)
        fast_us = timed(rows_vectorised, 4, arguments.repeats, nothing)

        their_matchers = [
            xg.GrammarMatcher(theirs[index], terminate_without_stop_token=True)
            for index in assignment
        ]
        for matcher in their_matchers:
            for byte in b'{"':
                matcher.accept_token(byte)
        driver = xg.BatchGrammarMatcher(max_threads=8)

        def their_fill(
            driver=driver, their_matchers=their_matchers, bitmask=bitmask
        ) -> None:
            driver.batch_fill_next_token_bitmask(their_matchers, bitmask)

        their_us = timed(their_fill, 4, arguments.repeats, nothing)

        total = host_us + seed_us + fill_us + copy_us + rows_us
        report.append(
            {
                "batch": batch,
                "host_advance_us": host_us,
                "set_matchers_us": seed_us,
                "fill_us": fill_us,
                "mask_to_host_us": copy_us,
                "row_copies_us": rows_us,
                "row_copies_vectorised_us": fast_us,
                "ours_total_us": total,
                "ours_total_vectorised_us": total - rows_us + fast_us,
                "xgrammar_us": their_us,
            }
        )
        print(
            f"{batch:>6} {host_us:>12.1f}u {seed_us:>12.1f}u {fill_us:>8.1f}u "
            f"{copy_us:>12.1f}u {rows_us:>10.1f}u | {total:>10.1f}u "
            f"{total - rows_us + fast_us:>10.1f}u {their_us:>8.1f}u",
            flush=True,
        )
        del device
        torch.cuda.empty_cache()

    OUT.write_text(json.dumps({"model": arguments.model, "rows": report}, indent=2))
    print(f"\nwritten to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
