"""Requests that arrive over time, which is what serving actually is.

Every other end-to-end measurement here submits 512 requests at once and waits.
That is a throughput experiment, and it hides the two things an operator is
judged on: how long a request waits for its first token, and how evenly the
rest arrive. It also hides the case this engine's admit/release design exists
for --- a batch whose composition changes every step, because requests are
joining and leaving rather than starting and finishing together.

So this drives the engine the way a server is driven: Poisson arrivals at a
target rate, each request carrying its own schema, and per-request time to
first token and inter-token latency recorded rather than a wall clock over the
whole set.

    python -m engrain_lab.rigor.online --backend engrain --qps 8 --requests 200
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import time
from pathlib import Path
from typing import Any

RESULTS = Path("results")


def _quantiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    ordered = sorted(values)
    return {
        "p50": ordered[len(ordered) // 2],
        "p90": ordered[min(len(ordered) - 1, int(0.90 * len(ordered)))],
        "p99": ordered[min(len(ordered) - 1, int(0.99 * len(ordered)))],
        "max": ordered[-1],
    }


async def _one(
    engine: Any,
    prompt: str,
    params: Any,
    request_id: str,
    record: list[dict[str, Any]],
) -> None:
    started = time.perf_counter()
    first: float | None = None
    stamps: list[float] = []
    text = ""
    async for output in engine.generate(prompt, params, request_id):
        now = time.perf_counter()
        if first is None:
            first = now - started
        stamps.append(now)
        text = output.outputs[0].text
    gaps = [b - a for a, b in zip(stamps, stamps[1:], strict=False)]
    record.append(
        {
            "ttft": first if first is not None else float("nan"),
            "itl": gaps,
            "tokens": len(stamps),
            "text": text,
            "total": time.perf_counter() - started,
        }
    )


async def run(arguments: argparse.Namespace) -> dict[str, Any]:
    import jsonschema
    from vllm import SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.sampling_params import StructuredOutputsParams
    from vllm.v1.engine.async_llm import AsyncLLM

    # vLLM refuses a schema its chosen backend cannot take *before* the backend
    # sees it, and one refusal raises out of the engine and takes the whole run
    # with it - so every arm runs on the set all of them accept. Screened once
    # into `corpus-agreed.json` rather than at every arm: doing it per run
    # compiles the whole corpus in three engines before a single request is
    # served, which is minutes of nothing.
    schemas = [
        instance["schema"]
        for instance in json.loads(Path(arguments.corpus).read_text())["instances"]
    ][: arguments.schemas]

    settings: dict[str, Any] = {}
    if arguments.backend != "none":
        settings["structured_outputs_config"] = {"backend": arguments.backend}
    engine_args = AsyncEngineArgs(
        model=arguments.model,
        max_model_len=1024,
        gpu_memory_utilization=arguments.memory,
        seed=20260806,
        disable_log_stats=True,
        **settings,
    )
    engine = AsyncLLM.from_engine_args(engine_args)

    prompts = [
        f"Return one JSON object. Request {i}. Answer with JSON only."
        for i in range(arguments.requests)
    ]
    rng = random.Random(arguments.seed)
    record: list[dict[str, Any]] = []
    assigned = [schemas[i % len(schemas)] for i in range(arguments.requests)]

    async def arrive() -> None:
        tasks = []
        for i, prompt in enumerate(prompts):
            params = SamplingParams(
                temperature=0.8,
                top_p=0.95,
                max_tokens=arguments.max_tokens,
                seed=20260806 + i,
                **(
                    {
                        "structured_outputs": StructuredOutputsParams(
                            json=assigned[i]
                        )
                    }
                    if arguments.backend != "none"
                    else {}
                ),
            )
            tasks.append(
                asyncio.create_task(_one(engine, prompt, params, f"r{i}", record))
            )
            # Poisson arrivals: the gap between two requests is exponential,
            # which is what makes the queue depth vary rather than step.
            await asyncio.sleep(rng.expovariate(arguments.qps))
        await asyncio.gather(*tasks)

    # A cold engine compiles kernels and captures graphs on the first requests,
    # so the warmup is not a measurement.
    warm = [
        asyncio.create_task(
            _one(
                engine,
                prompts[i],
                SamplingParams(
                    temperature=0.8,
                    max_tokens=8,
                    **(
                        {
                            "structured_outputs": StructuredOutputsParams(
                                json=assigned[i]
                            )
                        }
                        if arguments.backend != "none"
                        else {}
                    ),
                ),
                f"w{i}",
                [],
            )
        )
        for i in range(min(8, arguments.requests))
    ]
    await asyncio.gather(*warm)

    started = time.perf_counter()
    await arrive()
    elapsed = time.perf_counter() - started

    valid = 0
    for entry, schema in zip(record, assigned, strict=False):
        try:
            jsonschema.validate(json.loads(entry["text"].strip()), json.loads(schema))
        except Exception:  # noqa: BLE001
            continue
        valid += 1

    every_gap = [gap for entry in record for gap in entry["itl"]]
    report = {
        "backend": arguments.backend,
        "qps_offered": arguments.qps,
        "qps_served": len(record) / elapsed,
        "requests": len(record),
        "seconds": elapsed,
        "ttft_ms": {k: 1000 * v for k, v in _quantiles(
            [entry["ttft"] for entry in record]
        ).items()},
        "itl_ms": {k: 1000 * v for k, v in _quantiles(every_gap).items()},
        "tokens": sum(entry["tokens"] for entry in record),
        "valid": valid,
    }
    engine.shutdown()
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", default="engrain")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--corpus", default=str(RESULTS / "corpus-agreed.json"))
    parser.add_argument("--schemas", type=int, default=409)
    parser.add_argument("--requests", type=int, default=200)
    parser.add_argument("--qps", type=float, default=8.0)
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--memory", type=float, default=0.45)
    parser.add_argument("--seed", type=int, default=5)
    arguments = parser.parse_args()

    report = asyncio.run(run(arguments))
    print(
        f"  {report['backend']:<10} offered {report['qps_offered']:>5.1f} q/s, "
        f"served {report['qps_served']:>5.1f}  "
        f"TTFT p50 {report['ttft_ms'].get('p50', 0):>7.1f} "
        f"p99 {report['ttft_ms'].get('p99', 0):>8.1f} ms  "
        f"ITL p50 {report['itl_ms'].get('p50', 0):>5.1f} "
        f"p99 {report['itl_ms'].get('p99', 0):>6.1f} ms  "
        f"{report['valid']}/{report['requests']} valid"
    )
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"online-{report['backend']}-{int(report['qps_offered'])}qps.json"
    out.write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
