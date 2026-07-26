"""Can the grammar work hide inside the forward pass?

The per-step benchmark reports a row called "only what cannot overlap" and
charges the advance to it, on the reasoning that an advance follows the token
that was just sampled. That reasoning is wrong. The forward pass follows the
same token, and the two do not depend on each other: a decode step embeds the
token sampled at `t-1`, and so does the parser. The mask is not needed until
the logits exist. So the shape of a step is

    sample(t-1)  ->  forward pass          ->  apply mask  ->  sample(t)
                 ->  advance + fill        ->

with the two middle branches concurrent, which is what a serving engine already
does for XGrammar by running its fill on a worker thread.

The difference is where the work lands. XGrammar's advance and fill are host
work, so they overlap with the forward pass by using a core the GPU is not
using anyway - but they compete with tokenisation, scheduling and the HTTP
layer for that core. Ours is device work, so it overlaps by sharing the
streaming multiprocessors *with the forward pass itself*. Overlapping is
therefore not free for us in the way it is for them: the question is not
whether the kernels run concurrently but whether the step gets longer.

This measures exactly that. A decode step captured as a CUDA graph, a grammar
step captured as another, and then both on separate streams - the honest cost
of the grammar is how much longer the pair takes than the forward pass alone.
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Any

import torch

from .harness import Answer, Distribution, load_corpus, load_vocabulary, write_report


def _graph_of(call: Any) -> torch.cuda.CUDAGraph:
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            call()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    return graph


def _time(call: Any, repeats: int, warmup: int) -> Distribution:
    for _ in range(warmup):
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


def measure(
    model: str, schema_index: int, batch: int, repeats: int, warmup: int
) -> dict[str, Any]:
    import gpugrammar
    import xgrammar as xg
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from ..device_parser import DeviceBatch, DeviceGrammar

    tokenizer = AutoTokenizer.from_pretrained(model)
    vocabulary = load_vocabulary(model)
    instance = load_corpus()[schema_index]
    tokens = tokenizer(instance["text"], add_special_tokens=False)["input_ids"][:16]

    network = AutoModelForCausalLM.from_pretrained(model, dtype=torch.bfloat16).cuda()
    network.eval()

    compiled = gpugrammar.Compiler(vocabulary).compile_json_schema(instance["schema"])
    device_grammar = DeviceGrammar(compiled)
    device_batch = DeviceBatch(device_grammar, batch)
    import random

    rng = random.Random(20260726)
    matchers = []
    for _ in range(batch):
        matcher = compiled.matcher()
        for token in tokens[: rng.randrange(1, max(2, len(tokens)))]:
            if not matcher.accept_token(token):
                break
        matchers.append(matcher)
    device_batch.set_batch_configurations(
        {index: matcher.configurations() for index, matcher in enumerate(matchers)}
    )
    sampled = torch.full((batch,), tokens[0], dtype=torch.int32, device="cuda")
    device_batch.fill_mask()
    device_batch.capture()
    device_batch.advance(sampled)
    device_batch.capture_advance()

    with torch.inference_mode():
        ids = torch.randint(0, 1000, (batch, 1), device="cuda")
        out = network(ids, use_cache=True)
        cache = out.past_key_values
        for _ in range(3):
            out = network(ids, past_key_values=cache, use_cache=True)
            cache = out.past_key_values

        forward_graph = _graph_of(
            lambda: network(ids, past_key_values=cache, use_cache=True)
        )

        # The parser already captured its two halves, and a graph cannot be
        # captured inside another, so they are replayed rather than re-recorded.
        def grammar() -> None:
            device_batch.advance_graph.replay()
            device_batch.graph.replay()

        side = torch.cuda.Stream()

        def serial() -> None:
            forward_graph.replay()
            grammar()

        def overlapped() -> None:
            # Both branches start from the token sampled at t-1 and neither
            # needs the other, so they are issued together; the sampler would
            # join them afterwards.
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                grammar()
            forward_graph.replay()
            torch.cuda.current_stream().wait_stream(side)

        forward_only = _time(forward_graph.replay, repeats, warmup)
        grammar_only = _time(grammar, repeats, warmup)
        both_serial = _time(serial, repeats, warmup)
        both_overlapped = _time(overlapped, repeats, warmup)

    # XGrammar's equivalent: host work on a worker thread, which overlaps by
    # using a core rather than the SMs the forward pass wants.
    info = xg.TokenizerInfo.from_huggingface(tokenizer, vocab_size=len(tokenizer))
    theirs = xg.GrammarCompiler(info).compile_json_schema(instance["schema"])
    # Put them where ours are. A fresh matcher sits at the start state, which
    # has a different mask cost from the middle of a document, and comparing a
    # loaded batch against an empty one measures the arrangement rather than
    # the engines.
    their_rng = random.Random(20260726)
    their_matchers = []
    for _ in range(batch):
        matcher = xg.GrammarMatcher(
            theirs, terminate_without_stop_token=True, max_rollback_tokens=4
        )
        for token in tokens[: their_rng.randrange(1, max(2, len(tokens)))]:
            if not matcher.accept_token(token):
                break
        their_matchers.append(matcher)
    host_mask = xg.allocate_token_bitmask(batch, len(tokenizer))
    device_mask = torch.zeros_like(host_mask, device="cuda")

    def their_grammar() -> None:
        for index, matcher in enumerate(their_matchers):
            if matcher.accept_token(tokens[0]):
                matcher.rollback(1)
            matcher.fill_next_token_bitmask(host_mask, index)
        device_mask.copy_(host_mask, non_blocking=True)

    their_only = _time(their_grammar, max(8, repeats // 4), max(2, warmup // 4))

    def their_overlapped() -> None:
        forward_graph.replay()
        their_grammar()
        torch.cuda.synchronize()

    with torch.inference_mode():
        their_both = _time(their_overlapped, max(8, repeats // 4), max(2, warmup // 4))

    del network
    torch.cuda.empty_cache()

    return {
        "batch": batch,
        "schema": schema_index,
        "forward_us": forward_only.__dict__,
        "gpugrammar_alone_us": grammar_only.__dict__,
        "serial_us": both_serial.__dict__,
        "overlapped_us": both_overlapped.__dict__,
        "xgrammar_alone_us": their_only.__dict__,
        "xgrammar_overlapped_us": their_both.__dict__,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--schema-index", type=int, default=2)
    parser.add_argument("--batches", type=int, nargs="+", default=[32, 128, 512])
    parser.add_argument("--repeats", type=int, default=40)
    parser.add_argument("--warmup", type=int, default=12)
    arguments = parser.parse_args()

    rows = []
    for batch in arguments.batches:
        try:
            row = measure(
                arguments.model,
                arguments.schema_index,
                batch,
                arguments.repeats,
                arguments.warmup,
            )
        except Exception as error:  # noqa: BLE001
            print(f"batch {batch}: {error}")
            continue
        rows.append(row)
        forward = row["forward_us"]["p50"]
        print(
            f"batch {batch:4}: forward {forward:8.0f} us | "
            f"ours alone {row['gpugrammar_alone_us']['p50']:7.0f} "
            f"serial {row['serial_us']['p50']:8.0f} "
            f"overlapped {row['overlapped_us']['p50']:8.0f} "
            f"-> hidden cost {row['overlapped_us']['p50'] - forward:7.0f} us | "
            f"xgr alone {row['xgrammar_alone_us']['p50']:8.0f} "
            f"overlapped {row['xgrammar_overlapped_us']['p50']:8.0f} "
            f"-> {row['xgrammar_overlapped_us']['p50'] - forward:8.0f} us"
        )

    answers = []
    if rows:
        answers.append(
            Answer(
                "q10-overlap",
                "; ".join(
                    f"batch {row['batch']}: ours adds "
                    f"{row['overlapped_us']['p50'] - row['forward_us']['p50']:.0f} us to a "
                    f"{row['forward_us']['p50']:.0f} us step, XGrammar adds "
                    f"{row['xgrammar_overlapped_us']['p50'] - row['forward_us']['p50']:.0f} us"
                    for row in rows
                ),
                detail={
                    "rows": rows,
                    "note": (
                        "The advance does not have to be serialised after the "
                        "forward pass. Both start from the token sampled at "
                        "t-1 and neither needs the other; the mask is wanted "
                        "only once the logits exist. What is measured here is "
                        "how much longer the step becomes with the grammar "
                        "running beside it, which for a device-resident parser "
                        "means sharing the SMs the forward pass is using."
                    ),
                },
            )
        )
    else:
        answers.append(Answer("q10-overlap", "", unanswered="no batch completed"))

    write_report("overlap", answers, {"model": arguments.model})


if __name__ == "__main__":
    main()
