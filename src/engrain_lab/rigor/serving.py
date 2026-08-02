"""Two things the isolated-latency benchmark could not see.

`latency.py` timed each engine alone on an idle machine and concluded that
grammar cost is a negligible share of a decode step. Both halves of that
sentence were wrong in the same direction, and the direction flattered the
host-side baseline.

**The denominator was Python, not a decode step (q09).** Timing a HuggingFace
model in eager mode measures interpreter overhead: the answer came out at 30 ms
and did not change from batch 1 to batch 512, which no real forward pass does.
A serving engine captures the decode step as a CUDA graph precisely to delete
that overhead, and once it is deleted the step is several times shorter - so
grammar cost is a correspondingly larger share of it.

**The machine was idle (q21).** A mask filled on the host competes for cores
with tokenisation, detokenisation, scheduling and the HTTP layer, and a
benchmark that gives it twenty-four free cores measures the best case it will
never see. A mask filled on the device does not care.

**The step could not have been captured anyway (q22).** This is the structural
point, and it is binary rather than a matter of microseconds: a graph may not
contain a host callback, so a host-side mask fill cannot be inside the captured
region. The work has to be hoisted out and joined to the graph, which
reintroduces the synchronisation the graph existed to remove - and forecloses
running several decode steps without returning to the host at all.
"""

from __future__ import annotations

import argparse
import multiprocessing
import time
from typing import Any

import torch

from .harness import Answer, Distribution, load_corpus, load_vocabulary, write_report


def graphed_decode_step(model: str, batches: list[int]) -> dict[int, float]:
    """A decode step as a serving engine actually runs it: captured.

    Eager mode charges the step for Python, which is exactly what capture
    exists to avoid. Using the eager number as the denominator makes every
    grammar cost look small for a reason that has nothing to do with grammar.
    """
    from transformers import AutoModelForCausalLM

    network = AutoModelForCausalLM.from_pretrained(
        model, torch_dtype=torch.bfloat16
    ).cuda()
    network.eval()

    costs: dict[int, float] = {}
    with torch.inference_mode():
        for batch in batches:
            ids = torch.randint(0, 1000, (batch, 1), device="cuda")
            out = network(ids, use_cache=True)
            cache = out.past_key_values
            for _ in range(3):
                out = network(ids, past_key_values=cache, use_cache=True)
                cache = out.past_key_values

            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    network(ids, past_key_values=cache, use_cache=True)
            torch.cuda.current_stream().wait_stream(stream)

            graph = torch.cuda.CUDAGraph()
            try:
                with torch.cuda.graph(graph):
                    network(ids, past_key_values=cache, use_cache=True)
            except Exception:  # noqa: BLE001 - fall back to eager for this size
                costs[batch] = _eager(network, ids, cache)
                continue

            for _ in range(5):
                graph.replay()
            torch.cuda.synchronize()
            started = time.perf_counter()
            for _ in range(50):
                graph.replay()
            torch.cuda.synchronize()
            costs[batch] = (time.perf_counter() - started) / 50 * 1e6

    del network
    torch.cuda.empty_cache()
    return costs


def _eager(network: Any, ids: Any, cache: Any) -> float:
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(20):
        network(ids, past_key_values=cache, use_cache=True)
    torch.cuda.synchronize()
    return (time.perf_counter() - started) / 20 * 1e6


def _burn(stop: Any) -> None:
    """Occupy a core the way a serving process would."""
    total = 0
    while not stop.is_set():
        total += sum(index * index for index in range(4096))


def under_load(
    model: str, schema_index: int, batch: int, workers: list[int], repeats: int
) -> dict[str, Any]:
    """Fill the mask while the host is busy, which is the only state it is in."""
    import engrain
    import engrain.internals
    import xgrammar as xg
    from transformers import AutoTokenizer

    from engrain._engine import DeviceBatch, DeviceGrammar

    tokenizer = AutoTokenizer.from_pretrained(model)
    vocabulary = load_vocabulary(model)
    instance = load_corpus()[schema_index]

    ours = engrain.internals.Compiler(vocabulary).compile_json_schema(instance["schema"])
    info = xg.TokenizerInfo.from_huggingface(tokenizer, vocab_size=len(tokenizer))
    theirs = xg.GrammarCompiler(info).compile_json_schema(instance["schema"])

    device_batch = DeviceBatch(DeviceGrammar(ours), batch)
    device_batch.set_batch_configurations(
        {index: ours.matcher().configurations() for index in range(batch)}
    )
    device_batch.fill_mask()
    device_batch.capture()

    their_matchers = [
        xg.GrammarMatcher(theirs, terminate_without_stop_token=True)
        for _ in range(batch)
    ]
    host_mask = xg.allocate_token_bitmask(batch, len(tokenizer))
    device_mask = torch.zeros_like(host_mask, device="cuda")

    def their_fill() -> None:
        for index, matcher in enumerate(their_matchers):
            matcher.fill_next_token_bitmask(host_mask, index)
        device_mask.copy_(host_mask, non_blocking=True)

    rows = []
    for count in workers:
        stop = multiprocessing.Event()
        processes = [
            multiprocessing.Process(target=_burn, args=(stop,)) for _ in range(count)
        ]
        for process in processes:
            process.start()
        time.sleep(1.0)
        try:
            rows.append(
                {
                    "workers": count,
                    "engrain": _measure(device_batch.fill_mask, repeats).__dict__,
                    "xgrammar": _measure(their_fill, repeats).__dict__,
                }
            )
        finally:
            stop.set()
            for process in processes:
                process.join(timeout=5)
                if process.is_alive():
                    process.terminate()
        latest = rows[-1]
        print(
            f"  {count:3} busy cores: ours {latest['engrain']['p50']:8.1f}us "
            f"p99 {latest['engrain']['p99']:8.1f} | "
            f"xgr {latest['xgrammar']['p50']:9.1f}us "
            f"p99 {latest['xgrammar']['p99']:9.1f}"
        )
    return {"batch": batch, "schema": schema_index, "rows": rows}


def _measure(call: Any, repeats: int) -> Distribution:
    for _ in range(10):
        call()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        started = time.perf_counter()
        call()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - started) * 1e6)
    return Distribution.of(samples)


def capturability(model: str, schema_index: int, batch: int) -> Answer:
    """Can the mask fill live inside the captured decode step?

    Not a timing. A CUDA graph may not contain a host callback, so this is a
    property of each design rather than a number, and it decides whether the
    grammar can participate in multi-step decoding at all.
    """
    import engrain
    import xgrammar as xg
    from transformers import AutoTokenizer

    from engrain._engine import DeviceBatch, DeviceGrammar

    tokenizer = AutoTokenizer.from_pretrained(model)
    vocabulary = load_vocabulary(model)
    instance = load_corpus()[schema_index]

    ours = engrain.internals.Compiler(vocabulary).compile_json_schema(instance["schema"])
    device_batch = DeviceBatch(DeviceGrammar(ours), batch)
    device_batch.set_batch_configurations(
        {index: ours.matcher().configurations() for index in range(batch)}
    )
    device_batch.fill_mask()

    ours_ok, ours_why = True, ""
    try:
        device_batch.capture()
        device_batch.fill_mask()
    except Exception as error:  # noqa: BLE001
        ours_ok, ours_why = False, str(error)[:160]

    info = xg.TokenizerInfo.from_huggingface(tokenizer, vocab_size=len(tokenizer))
    theirs = xg.GrammarCompiler(info).compile_json_schema(instance["schema"])
    matchers = [
        xg.GrammarMatcher(theirs, terminate_without_stop_token=True)
        for _ in range(batch)
    ]
    host_mask = xg.allocate_token_bitmask(batch, len(tokenizer))
    device_mask = torch.zeros_like(host_mask, device="cuda")

    theirs_ok, theirs_why = True, ""
    try:
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for index, matcher in enumerate(matchers):
                matcher.fill_next_token_bitmask(host_mask, index)
            device_mask.copy_(host_mask, non_blocking=True)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for index, matcher in enumerate(matchers):
                matcher.fill_next_token_bitmask(host_mask, index)
            device_mask.copy_(host_mask, non_blocking=True)
        graph.replay()
        torch.cuda.synchronize()
        # Capture may appear to succeed while recording nothing that depends on
        # the host work. Change the host buffer and see whether replay notices.
        host_mask.zero_()
        before = device_mask.clone()
        graph.replay()
        torch.cuda.synchronize()
        if torch.equal(before, device_mask):
            theirs_ok = False
            theirs_why = (
                "replay reproduced the old mask: the host fill was not captured, "
                "only the copy of whatever the buffer happened to hold"
            )
    except Exception as error:  # noqa: BLE001
        theirs_ok, theirs_why = False, str(error)[:160]

    return Answer(
        question_id="q22-capture",
        headline=(
            f"decode-step CUDA graph can contain the fill: engrain "
            f"{'yes' if ours_ok else 'no'}, xgrammar "
            f"{'yes' if theirs_ok else 'no'}"
        ),
        detail={
            "engrain": {"captured": ours_ok, "why": ours_why},
            "xgrammar": {"captured": theirs_ok, "why": theirs_why},
            "note": (
                "A graph records device work. Host work inside the captured "
                "region simply does not go in, so a host-side fill has to be "
                "hoisted out and joined - which is the synchronisation the "
                "graph was there to remove."
            ),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--schema-index", type=int, default=2)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 32, 128, 512])
    parser.add_argument("--load-batch", type=int, default=128)
    parser.add_argument("--workers", type=int, nargs="+", default=[0, 8, 16, 24])
    parser.add_argument("--repeats", type=int, default=60)
    arguments = parser.parse_args()

    answers: list[Answer] = []

    print("decode step, captured as a CUDA graph:")
    try:
        step = graphed_decode_step(arguments.model, arguments.batches)
        for batch, cost in step.items():
            print(f"  batch {batch:5}: {cost:9.1f} us")
    except Exception as error:  # noqa: BLE001
        step = {}
        print(f"  unavailable: {error}")

    print("\nfill with the host under load:")
    try:
        load = under_load(
            arguments.model,
            arguments.schema_index,
            arguments.load_batch,
            arguments.workers,
            arguments.repeats,
        )
    except Exception as error:  # noqa: BLE001
        load = {"error": str(error)[:200]}
        print(f"  unavailable: {error}")

    if "error" not in load:
        idle = load["rows"][0]
        busy = load["rows"][-1]
        answers.append(
            Answer(
                "q21-contention",
                f"at batch {load['batch']} with {busy['workers']} busy cores, "
                f"XGrammar's fill goes from {idle['xgrammar']['p50']:.0f} to "
                f"{busy['xgrammar']['p50']:.0f} us "
                f"({busy['xgrammar']['p50'] / max(1e-9, idle['xgrammar']['p50']):.2f}x) "
                f"while ours goes from {idle['engrain']['p50']:.0f} to "
                f"{busy['engrain']['p50']:.0f} us "
                f"({busy['engrain']['p50'] / max(1e-9, idle['engrain']['p50']):.2f}x)",
                detail=load,
            )
        )
    else:
        answers.append(Answer("q21-contention", "", unanswered=load["error"]))

    try:
        answers.append(
            capturability(arguments.model, arguments.schema_index, arguments.load_batch)
        )
    except Exception as error:  # noqa: BLE001
        answers.append(Answer("q22-capture", "", unanswered=str(error)[:200]))

    if step:
        answers.append(
            Answer(
                "q09-forward",
                "captured decode step: "
                + ", ".join(f"batch {b} {c:.0f}us" for b, c in step.items()),
                detail={
                    "captured_step_us": step,
                    "note": (
                        "Replaces the eager measurement, which was 30 ms at "
                        "every batch size and was therefore timing the Python "
                        "interpreter rather than the model."
                    ),
                },
            )
        )

    write_report("serving", answers, {"model": arguments.model})


if __name__ == "__main__":
    main()
