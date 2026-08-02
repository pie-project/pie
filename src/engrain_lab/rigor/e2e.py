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

**Fixed work per arm.** The same prompts, the same seed, the same `max_tokens`,
and the generated token count is reported next to the rate - because a backend
whose grammar is a relaxation can emit longer documents, generate more tokens,
and post a better tokens-per-second while doing more work for the same request.

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
    parser.add_argument("--memory", type=float, default=0.45)
    arguments = parser.parse_args()

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

    report = {"backend": arguments.backend, "model": arguments.model, "rows": []}
    for batch in arguments.batches:
        assigned = [SCHEMAS[i % len(SCHEMAS)] for i in range(batch)]
        prompts = [
            f"Give a JSON {SUBJECTS[i % len(SCHEMAS)]} record {i}. JSON only."
            for i in range(batch)
        ]
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
        for _ in range(arguments.repeats):
            started = time.perf_counter()
            outputs = llm.generate(prompts, params, use_tqdm=False)
            seconds.append(time.perf_counter() - started)
            produced.append(sum(len(o.outputs[0].token_ids) for o in outputs))

        rates = [
            count / elapsed for count, elapsed in zip(produced, seconds, strict=True)
        ]
        valid = 0
        for output, schema in zip(outputs, assigned, strict=True):
            try:
                document = json.loads(output.outputs[0].text.strip())
            except Exception:  # noqa: BLE001
                continue
            if set(document) == set(schema["required"]):
                valid += 1

        row = {
            "batch": batch,
            "seconds": _quantiles(seconds),
            "tokens_per_second": _quantiles(rates),
            "tokens_generated_p50": statistics.median(produced),
            "valid_last_run": valid,
            "requests": batch,
        }
        report["rows"].append(row)
        rate = row["tokens_per_second"]
        print(
            f"  batch {batch:>4}: {rate['p50']:>8.0f} tok/s "
            f"[p25 {rate['p25']:.0f}, p75 {rate['p75']:.0f}, "
            f"range {rate['min']:.0f}-{rate['max']:.0f}]  "
            f"{row['tokens_generated_p50']} tokens  "
            f"{valid}/{batch} valid",
            flush=True,
        )

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"e2e-{arguments.backend}.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
