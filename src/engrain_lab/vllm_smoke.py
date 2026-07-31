"""A/B the engrain backend against stock XGrammar inside vLLM.

Run with `--backend engrain` or `--backend xgrammar`. Both are real vLLM
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

# A second and third schema, so `--mixed` puts the batch under several at once.
# That is what a serving batch looks like - requests bring their own - and it is
# the case the device tables are laid out as one arena for.
SCHEMAS = [
    SCHEMA,
    {
        "type": "object",
        "properties": {"title": {"type": "string"}, "pages": {"type": "integer"}},
        "required": ["title", "pages"],
        "additionalProperties": False,
    },
    {
        "type": "object",
        "properties": {"city": {"type": "string"}, "population": {"type": "integer"}},
        "required": ["city", "population"],
        "additionalProperties": False,
    },
]


def _check(text: str, schema: dict) -> None:
    value = json.loads(text)
    required = set(schema["required"])
    assert set(value) == required, f"{set(value)} != {required}"
    for key, kind in schema["properties"].items():
        expected = {"string": str, "integer": int, "boolean": bool}[kind["type"]]
        assert isinstance(value[key], expected), f"{key} is not {kind['type']}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", default=os.environ.get("BACKEND", "engrain"))
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--prompts", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument(
        "--mixed",
        action="store_true",
        help="put the batch under several schemas at once, which is what a "
        "serving batch looks like and what one arena of tables is for",
    )
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
    schemas = SCHEMAS if arguments.mixed else [SCHEMA]
    assigned = [schemas[index % len(schemas)] for index in range(arguments.prompts)]
    subjects = ["person", "book", "city"]
    prompts = [
        f"Give a JSON {subjects[index % len(schemas)]} record {index}. JSON only."
        for index in range(arguments.prompts)
    ]
    params = [
        SamplingParams(
            temperature=0.8,
            top_p=0.95,
            max_tokens=arguments.max_tokens,
            structured_outputs=StructuredOutputsParams(json=schema),
        )
        for schema in assigned
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
    truncated = 0
    for index, output in enumerate(outputs):
        text = output.outputs[0].text.strip()
        try:
            _check(text, assigned[index])
            valid += 1
        except Exception as error:  # noqa: BLE001
            # A document that ran out of tokens is not an invalid document. The
            # model will happily emit eighty digits of an integer, which the
            # schema allows, so the two have to be counted apart or the test
            # reports a grammar failure for a sampling one.
            if output.outputs[0].finish_reason == "length":
                truncated += 1
            else:
                print("INVALID:", repr(text[:120]), error)

    print(
        f"{valid}/{len(outputs) - truncated} valid "
        f"({truncated} ran out of tokens) | median {median:.0f} tok/s "
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
