from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from gpu_lr1.benchmark import machine_metadata
from gpu_lr1.vocab import Vocabulary
from gpu_lr1.workloads import NamedSchema, benchmark_schemas


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark the CPU XGrammar mask-generation baseline"
    )
    parser.add_argument("--schemas", type=int, default=14)
    parser.add_argument("--vocab-size", type=int, default=32768)
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1, 8, 32, 128, 512],
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="XGrammar batch worker threads; 1 avoids thread-pool overhead",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/xgrammar-baseline.json"),
    )
    args = parser.parse_args()

    try:
        import xgrammar as xgr
    except ImportError as exc:
        raise RuntimeError("install gpu-lr1[baselines] first") from exc

    device = torch.device(args.device)
    schemas = benchmark_schemas(args.schemas)
    vocabulary = Vocabulary.tiktoken("gpt2", args.vocab_size)
    tokenizer_info = xgr.TokenizerInfo(
        list(vocabulary.tokens),
        vocab_type=xgr.VocabType.RAW,
        vocab_size=vocabulary.size,
        stop_token_ids=[vocabulary.eos_token_id],
    )
    compiler = xgr.GrammarCompiler(tokenizer_info)

    compile_started = time.perf_counter()
    compiled = [
        compiler.compile_json_schema(
            item.schema,
            any_whitespace=False,
            separators=(",", ":"),
            strict_mode=True,
            any_order=False,
        )
        for item in schemas
    ]
    compile_seconds = time.perf_counter() - compile_started
    batch_matcher = xgr.BatchGrammarMatcher(max_threads=args.threads)
    byte_token_ids = _single_byte_token_ids(vocabulary)

    records = []
    for mix in ("homogeneous", "mixed4", "mixed_all"):
        for batch_size in args.batch_sizes:
            schema_ids = _schema_ids(mix, batch_size, len(schemas))
            matchers = []
            prefix_lengths = []
            for sequence_id, schema_id in enumerate(schema_ids):
                matcher = xgr.GrammarMatcher(compiled[schema_id])
                sample = canonical_sample_bytes(schemas[schema_id])
                prefix_length = (sequence_id * 17 + schema_id * 7) % len(sample)
                for byte in sample[:prefix_length]:
                    token_id = byte_token_ids[byte]
                    if not matcher.accept_token(token_id):
                        raise AssertionError(
                            f"XGrammar rejected canonical prefix for schema {schema_id}"
                        )
                matchers.append(matcher)
                prefix_lengths.append(prefix_length)

            host_mask = torch.empty(
                xgr.get_bitmask_shape(batch_size, vocabulary.size),
                dtype=xgr.bitmask_dtype,
                pin_memory=True,
            )
            device_mask = torch.empty_like(host_mask, device=device)
            logits_pool = torch.randn(
                (
                    args.warmup + args.iterations,
                    batch_size,
                    vocabulary.size,
                ),
                dtype=torch.float16,
                device=device,
            )
            output_tokens = torch.empty(
                batch_size,
                dtype=torch.int64,
                device=device,
            )
            call_index = 0

            def fill_mask() -> None:
                batch_matcher.batch_fill_next_token_bitmask(matchers, host_mask)

            def full_step() -> torch.Tensor:
                nonlocal call_index
                fill_mask()
                device_mask.copy_(host_mask, non_blocking=True)
                working_logits = logits_pool[call_index]
                call_index += 1
                xgr.apply_token_bitmask_inplace(
                    working_logits,
                    device_mask,
                    vocab_size=vocabulary.size,
                    backend="triton",
                )
                return torch.argmax(
                    working_logits,
                    dim=1,
                    out=output_tokens,
                )

            fill_timing = measure_cpu(
                fill_mask,
                warmup=args.warmup,
                iterations=args.iterations,
            )
            full_timing = measure_cuda_wall(
                full_step,
                warmup=args.warmup,
                iterations=args.iterations,
            )
            records.append(
                {
                    "mix": mix,
                    "batch_size": batch_size,
                    "unique_schemas": len(set(schema_ids)),
                    "prefix_bytes_median": float(np.median(prefix_lengths)),
                    "fill_wall_p50_us": fill_timing["p50_us"],
                    "fill_wall_p95_us": fill_timing["p95_us"],
                    "full_wall_p50_us": full_timing["p50_us"],
                    "full_wall_p95_us": full_timing["p95_us"],
                    "sequences_per_second": float(
                        batch_size / (full_timing["mean_us"] * 1e-6)
                    ),
                }
            )

    payload = {
        "metadata": machine_metadata(device),
        "config": {
            **vars(args),
            "output": str(args.output),
            "vocabulary_name": vocabulary.name,
            "logits_buffering": "rotating buffers across timed iterations",
        },
        "compile_seconds": compile_seconds,
        "benchmarks": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        "mix,batch,schemas,fill_p50_us,fill_p95_us,"
        "full_p50_us,full_p95_us,sequences_per_second"
    )
    for item in records:
        print(
            f"{item['mix']},{item['batch_size']},{item['unique_schemas']},"
            f"{item['fill_wall_p50_us']:.3f},{item['fill_wall_p95_us']:.3f},"
            f"{item['full_wall_p50_us']:.3f},{item['full_wall_p95_us']:.3f},"
            f"{item['sequences_per_second']:.1f}"
        )


def canonical_sample_bytes(named_schema: NamedSchema) -> bytes:
    value = _schema_witness(named_schema.schema, named_schema.schema)
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=False,
    ).encode("ascii")


def _schema_witness(schema: Any, root: Mapping[str, Any]) -> Any:
    if schema is False:
        raise ValueError("false schema has no witness")
    if schema is True:
        return None
    if "$ref" in schema:
        current: Any = root
        for raw_part in schema["$ref"][2:].split("/"):
            part = raw_part.replace("~1", "/").replace("~0", "~")
            current = current[part]
        return _schema_witness(current, root)
    if "const" in schema:
        return schema["const"]
    if "enum" in schema:
        return schema["enum"][0]
    for keyword in ("oneOf", "anyOf"):
        if keyword in schema:
            return _schema_witness(schema[keyword][0], root)

    schema_type = schema.get("type")
    if isinstance(schema_type, list):
        schema_type = schema_type[0]
    if schema_type is None:
        schema_type = "object" if "properties" in schema else "array"
    if schema_type == "object":
        properties = schema.get("properties", {})
        return {
            name: _schema_witness(properties[name], root)
            for name in properties
        }
    if schema_type == "array":
        item_schema = schema.get("items", {})
        count = max(1, int(schema.get("minItems", 0)))
        return [_schema_witness(item_schema, root) for _ in range(count)]
    if schema_type == "string":
        return "x" * max(1, int(schema.get("minLength", 0)))
    if schema_type in ("integer", "number"):
        lower = schema.get("minimum", schema.get("exclusiveMinimum", 0))
        value = math.ceil(float(lower))
        if "exclusiveMinimum" in schema:
            value += 1
        multiple = schema.get("multipleOf", 1)
        if isinstance(multiple, int) and multiple > 1:
            value = math.ceil(value / multiple) * multiple
        return value
    if schema_type == "boolean":
        return False
    if schema_type == "null":
        return None
    raise ValueError(f"cannot build witness for type {schema_type!r}")


def _single_byte_token_ids(vocabulary: Vocabulary) -> dict[int, int]:
    result = {}
    for token_id, token in enumerate(vocabulary.tokens):
        if len(token) == 1:
            result[token[0]] = token_id
    missing = sorted(set(range(256)) - result.keys())
    if missing:
        raise ValueError(f"vocabulary lacks byte tokens: {missing[:8]}")
    return result


def _schema_ids(mix: str, batch_size: int, schema_count: int) -> list[int]:
    if mix == "homogeneous":
        return [0] * batch_size
    active = min(4, schema_count) if mix == "mixed4" else schema_count
    return [index % active for index in range(batch_size)]


def measure_cpu(
    function: Callable[[], None],
    *,
    warmup: int,
    iterations: int,
) -> dict[str, float]:
    for _ in range(warmup):
        function()
    samples = []
    for _ in range(iterations):
        started = time.perf_counter_ns()
        function()
        samples.append((time.perf_counter_ns() - started) / 1_000)
    return _summarize(samples)


def measure_cuda_wall(
    function: Callable[[], torch.Tensor],
    *,
    warmup: int,
    iterations: int,
) -> dict[str, float]:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iterations):
        started = time.perf_counter_ns()
        function()
        torch.cuda.synchronize()
        samples.append((time.perf_counter_ns() - started) / 1_000)
    return _summarize(samples)


def _summarize(samples: list[float]) -> dict[str, float]:
    ordered = sorted(samples)
    p95_index = min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)
    return {
        "p50_us": float(statistics.median(ordered)),
        "p95_us": float(ordered[p95_index]),
        "mean_us": float(statistics.fmean(ordered)),
    }


if __name__ == "__main__":
    main()
