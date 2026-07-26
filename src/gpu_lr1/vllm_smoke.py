"""A/B the gpugrammar backend against stock XGrammar inside vLLM.

Run with `--backend gpugrammar` or `--backend xgrammar`. Both are real vLLM
backend names now that `third_party/vllm` carries the registration, so nothing
is monkeypatched and the engine may stay in its own process.

Validity is the correctness check: every output must parse and satisfy the
schema. Throughput at this batch size is not the headline number - the mask
fill is far from dominant at batch 16 - it only catches a regression of the
wrong order of magnitude.
"""

import argparse
import json
import os
import time

os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")

SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
        "active": {"type": "boolean"},
    },
    "required": ["name", "age", "active"],
    "additionalProperties": False,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", default=os.environ.get("BACKEND", "gpugrammar"))
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--prompts", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="throughput here is noisy enough that one run says little; the "
        "median of several is what to compare",
    )
    arguments = parser.parse_args()

    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    print(f">>> backend: {arguments.backend}")
    llm = LLM(
        model=arguments.model,
        max_model_len=1024,
        gpu_memory_utilization=0.35,
        structured_outputs_config={"backend": arguments.backend},
    )
    params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        max_tokens=arguments.max_tokens,
        structured_outputs=StructuredOutputsParams(json=SCHEMA),
    )
    prompts = [
        f"Give a JSON profile for person {index}. JSON only."
        for index in range(arguments.prompts)
    ]

    rates = []
    for _ in range(arguments.repeats):
        start = time.perf_counter()
        outputs = llm.generate(prompts, params)
        elapsed = time.perf_counter() - start
        rates.append(
            sum(len(output.outputs[0].token_ids) for output in outputs) / elapsed
        )
    rates.sort()
    median = rates[len(rates) // 2]

    valid = 0
    for output in outputs:
        text = output.outputs[0].text.strip()
        try:
            value = json.loads(text)
            assert set(value) == {"name", "age", "active"}
            assert isinstance(value["age"], int)
            assert isinstance(value["active"], bool)
            valid += 1
        except Exception as error:  # noqa: BLE001
            print("INVALID:", repr(text[:120]), error)

    print(
        f"{valid}/{len(outputs)} valid | median {median:.0f} tok/s "
        f"over {len(rates)} runs (min {rates[0]:.0f}, max {rates[-1]:.0f})"
    )
    if os.environ.get("VLLM_GRAMMAR_TIMING"):
        from vllm.v1.structured_output import StructuredOutputManager

        spent = StructuredOutputManager.grammar_seconds
        print(
            f"  time inside the grammar: {spent:.2f}s of "
            f"{sum(1 / rate for rate in rates) * 0 + arguments.repeats:.0f} runs"
        )
    return 0 if valid == len(outputs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
