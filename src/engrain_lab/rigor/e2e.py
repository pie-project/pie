"""End to end, through vLLM, against every backend it can dispatch (q08, q19).

The one measurement this project has repeatedly refused to make a claim from,
because the first attempt was too noisy to attribute anything: 7,052 tok/s
against XGrammar's 6,588, with ranges of 3,972-7,104 and 4,644-7,630. Numbers
that overlap that far are not a result, and the paper says so.

So this is built around the reasons that run was noisy.

**A control (q09).** Grammar work is a few percent of a decode step, so the
question is not "which backend is faster" but "how much of the step does each
one cost". An unconstrained run is the denominator, and without it a difference
between two constrained runs cannot be told from run-to-run variance.

**Compare on the makespan, not on tokens per second.** Both arms serve the same
512 requests, so how long that takes is the comparison an operator gets, and it
needs no normalising. Tokens per second does: it is tokens over the makespan,
so a backend whose requests run longer generates more tokens *and* keeps the
batch fuller, and a fuller batch spreads the per-step fixed cost over more rows.
Measured against XGrammar on the exact fragment, the inflation is exactly the
gap in tokens generated - 17.7% more tokens, and a tokens-per-second ratio 17%
better than the makespan ratio. Both are reported; the makespan is the one to
quote.

**One process per backend.** vLLM caches prefixes and compiles graphs on first
use; running two backends in one process measures the second one warm.

**Distributions, not a median (q15).** Serving is judged at the tail, and the
whole reason the earlier number was withdrawn is that its spread was not
reported next to it. p25/p50/p75 and the full range, every time.

**All three baselines (q19).** vLLM dispatches `xgrammar`, `guidance`
(llguidance) and our `engrain` from the same config, so the comparison needs no
integration work - only the honesty to run it.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")

RESULTS = Path("results")

# Three schemas, so a batch is heterogeneous the way a serving batch is:
# requests bring their own. Each is small enough that a 0.6B model can fill it
# and large enough that the parser is doing something.
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

SUBJECTS = ["person", "book", "city"]


def _agreed_schemas(corpus: str) -> list[dict]:
    """Corpus schemas every engine will take, checked here rather than assumed.

    A coverage sweep run against each library directly is not the same set vLLM
    ends up with: its backends compile with their own options, and a schema one
    of them refuses raises out of `generate` and takes the whole arm with it.
    Three arms died that way. So the set is intersected here, with each engine
    asked in the configuration vLLM will use it in, and the count is printed so
    a reader knows what the comparison ran on.
    """
    import llguidance
    import llguidance.hf
    import xgrammar as xg
    from transformers import AutoTokenizer
    from vllm.v1.structured_output.backend_guidance import validate_guidance_grammar
    from vllm.v1.structured_output.backend_xgrammar import (
        has_xgrammar_unsupported_json_features,
    )

    from engrain.internals import Compiler

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
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
    llt = llguidance.hf.from_tokenizer(tokenizer)

    instances = json.loads(Path(corpus).read_text())["instances"]
    kept: list[dict] = []
    refused = {"engrain": 0, "xgrammar": 0, "guidance": 0, "not json": 0}
    for instance in instances:
        text = instance["schema"]
        try:
            schema = json.loads(text)
        except Exception:  # noqa: BLE001
            refused["not json"] += 1
            continue
        try:
            ours.compile_json_schema(text, max_digits=8)
        except Exception:  # noqa: BLE001
            refused["engrain"] += 1
            continue
        # vLLM screens a schema before its backend ever sees it, and its
        # allowlist is stricter than the library's compiler - the library takes
        # every corpus schema and vLLM refuses several. Asking the library was
        # what let three arms die inside `generate`.
        try:
            if has_xgrammar_unsupported_json_features(schema):
                raise ValueError("vLLM refuses this for xgrammar")
            xgc.compile_json_schema(text)
        except Exception:  # noqa: BLE001
            refused["xgrammar"] += 1
            continue
        try:
            from vllm.sampling_params import SamplingParams, StructuredOutputsParams

            validate_guidance_grammar(
                SamplingParams(
                    structured_outputs=StructuredOutputsParams(json=schema)
                ),
                tokenizer=None,
            )
            matcher = llguidance.LLMatcher(
                llt, llguidance.LLMatcher.grammar_from_json_schema(text)
            )
            if matcher.is_error():
                raise ValueError(matcher.get_error()[:60])
        except Exception:  # noqa: BLE001
            refused["guidance"] += 1
            continue
        kept.append(schema)
    print(f"schemas every engine takes: {len(kept)} of {len(instances)} {refused}")
    return kept


def _quantiles(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "min": ordered[0],
        "p25": ordered[max(0, int(0.25 * (len(ordered) - 1)))],
        "p50": statistics.median(ordered),
        "p75": ordered[min(len(ordered) - 1, int(0.75 * (len(ordered) - 1)))],
        "max": ordered[-1],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        default="engrain",
        help="engrain, xgrammar, guidance, or none for the unconstrained control",
    )
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--batches", type=int, nargs="+", default=[16, 64, 256])
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--repeats", type=int, default=12)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument(
        "--unique",
        action="store_true",
        help="give every request its own schema, drawn from the corpus rather "
        "than the three built in. This is the case that should hurt us: our "
        "fill deduplicates rows sharing a grammar and a parse state, and the "
        "buffers are sized by maxima over the whole pool, while a host-side "
        "matcher's per-sequence call is schema-agnostic. It also makes the "
        "compiler part of the measurement, which is where we are weakest.",
    )
    parser.add_argument(
        "--schemas",
        type=int,
        default=None,
        help="how many distinct schemas to draw from, cycling to fill the "
        "batch. 1 is the other extreme from all-distinct: every request under "
        "the same grammar in the same parse state, which is exactly what the "
        "fill deduplicates, so it is the case that should flatter us most. "
        "Under --unique the schemas come from the corpus, so sweeping this is "
        "what finds the count at which sharing stops paying for residency.",
    )
    parser.add_argument("--memory", type=float, default=0.45)
    parser.add_argument(
        "--corpus",
        default=str(RESULTS / "jsonschemabench-instances.json"),
        help="which corpus `--unique` draws from. `results/corpus-exact.json` "
        "is the fragment this engine enforces with nothing left over, so on it "
        "a validity number is not confounded by a widened mask.",
    )
    arguments = parser.parse_args()

    import jsonschema

    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    constrained = arguments.backend != "none"
    settings = (
        {"structured_outputs_config": {"backend": arguments.backend}}
        if constrained
        else {}
    )
    print(f">>> backend: {arguments.backend}", flush=True)
    llm = LLM(
        model=arguments.model,
        max_model_len=1024,
        gpu_memory_utilization=arguments.memory,
        seed=20260802,
        **settings,
    )

    corpus: list[dict] = []
    if arguments.unique:
        # Only schemas every engine compiles, so no arm is measured on a set
        # another could not have run. The list comes from the coverage sweep.
        corpus = _agreed_schemas(arguments.corpus)

    report = {
        "backend": arguments.backend,
        "model": arguments.model,
        "unique": arguments.unique,
        "schemas": arguments.schemas,
        "rows": [],
    }
    for batch in arguments.batches:
        if arguments.unique:
            pool = min(arguments.schemas or len(corpus), len(corpus))
            pool = max(1, pool)
            assigned = [corpus[i % pool] for i in range(batch)]
            prompts = [
                f"Produce one JSON document, number {i}. JSON only."
                for i in range(batch)
            ]
            distinct = min(batch, pool)
        else:
            pool = arguments.schemas or len(SCHEMAS)
            pool = max(1, min(pool, len(SCHEMAS)))
            assigned = [SCHEMAS[i % pool] for i in range(batch)]
            prompts = [
                f"Give a JSON {SUBJECTS[i % pool]} record {i}. JSON only."
                for i in range(batch)
            ]
            distinct = min(batch, pool)
        params = [
            SamplingParams(
                temperature=0.8,
                top_p=0.95,
                max_tokens=arguments.max_tokens,
                seed=20260802 + i,
                **(
                    {"structured_outputs": StructuredOutputsParams(json=schema)}
                    if constrained
                    else {}
                ),
            )
            for i, schema in enumerate(assigned)
        ]

        # Warm-up is not a measurement: vLLM compiles and captures on first use,
        # and a compile in the first sample is what made the earlier run's
        # maximum three times its minimum.
        for _ in range(arguments.warmup):
            llm.generate(prompts, params, use_tqdm=False)

        seconds: list[float] = []
        produced: list[int] = []
        longest: list[int] = []
        capped: list[int] = []
        for _ in range(arguments.repeats):
            started = time.perf_counter()
            outputs = llm.generate(prompts, params, use_tqdm=False)
            seconds.append(time.perf_counter() - started)
            lengths = [len(o.outputs[0].token_ids) for o in outputs]
            produced.append(sum(lengths))
            # The wall clock is the makespan, and the makespan is the longest
            # request, not the average one. Two engines can generate almost the
            # same number of tokens and still run for very different times if
            # one of them has a tail that never stops.
            longest.append(max(lengths))
            capped.append(sum(1 for n in lengths if n >= arguments.max_tokens))

        rates = [
            count / elapsed for count, elapsed in zip(produced, seconds, strict=True)
        ]
        valid = 0
        for output, schema in zip(outputs, assigned, strict=True):
            try:
                document = json.loads(output.outputs[0].text.strip())
            except Exception:  # noqa: BLE001
                continue
            # The schema itself, not a check of the root's `required` keys.
            # That cheaper test cannot see a violation anywhere below the root,
            # so it reported no change from a lowering that measurably enforced
            # more - which is exactly the kind of blindness that makes a
            # measurement worse than none.
            try:
                jsonschema.validate(document, schema)
            except jsonschema.ValidationError:
                continue
            except Exception:  # noqa: BLE001
                # An invalid schema is not the engine's failure to report.
                pass
            valid += 1

        row = {
            "batch": batch,
            "distinct_schemas": distinct,
            # The metric that needs no normalising, and the one to compare on.
            # Tokens per second is tokens over the makespan, so an engine whose
            # requests run longer generates more tokens *and* keeps the batch
            # fuller, and a fuller batch spreads the per-step fixed cost over
            # more rows. Measured against XGrammar on the exact fragment, the
            # inflation is exactly the gap in tokens generated: 17.7% more
            # tokens, 17% better tokens-per-second than the makespan says.
            # Both arms serve the same 512 requests, so how long that takes is
            # the comparison an operator actually gets.
            "makespan_seconds": _quantiles(seconds),
            "seconds": _quantiles(seconds),
            "tokens_per_second": _quantiles(rates),
            "tokens_generated_p50": statistics.median(produced),
            "longest_request_p50": statistics.median(longest),
            "hit_the_cap_p50": statistics.median(capped),
            "valid_last_run": valid,
            "requests": batch,
        }
        report["rows"].append(row)
        rate = row["tokens_per_second"]
        span = row["makespan_seconds"]
        print(
            f"  batch {batch:>4}: {span['p50']:>7.3f} s to serve {batch} "
            f"[p25 {span['p25']:.3f}, p75 {span['p75']:.3f}]  "
            f"{rate['p50']:>8.0f} tok/s "
            f"[p25 {rate['p25']:.0f}, p75 {rate['p75']:.0f}, "
            f"range {rate['min']:.0f}-{rate['max']:.0f}]  "
            f"{row['tokens_generated_p50']} tokens  "
            f"(longest {row['longest_request_p50']}, "
            f"{row['hit_the_cap_p50']} at the cap)  "
            f"{distinct} schemas  {valid}/{batch} valid",
            flush=True,
        )

    RESULTS.mkdir(exist_ok=True)
    tag = "-unique" if arguments.unique else ""
    if "exact" in arguments.corpus:
        tag = f"{tag}-exact"
    if arguments.schemas:
        tag = f"{tag}-{arguments.schemas}schema"
    out = RESULTS / f"e2e-{arguments.backend}{tag}.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
