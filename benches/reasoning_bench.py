#!/usr/bin/env python3
"""Benchmark Pie reasoning patterns on GSM8K-format JSONL data.

The reference answer stays in this harness and is never sent to the inferlet.
Official GSM8K records (`question`, `answer` containing `#### N`) and simple
records (`id`, `question`, `answer`) are both accepted.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import re
import statistics
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
INFERLET = "reasoning-benchmark"
PATTERNS = ("direct", "best_of_n", "tree_of_thought", "graph_of_thought")
NUMBER_RE = re.compile(r"[-+]?(?:\d[\d,]*\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


@dataclass(frozen=True)
class Problem:
    id: str
    question: str
    answer: str


@dataclass
class RunResult:
    problem_id: str
    pattern: str
    repetition: int
    correct: bool
    oracle_correct: bool
    correct_candidates: int
    expected_answer: str
    predicted_answer: str | None
    latency_s: float
    output: dict[str, Any] | None
    engine_stats_delta: dict[str, int]
    error: str | None = None


def normalize_number(value: str | int | float | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip().replace(",", "").replace("$", "")
    matches = NUMBER_RE.findall(text)
    if not matches:
        return None
    raw = matches[-1].replace(",", "")
    try:
        number = float(raw)
    except ValueError:
        return None
    if not math.isfinite(number):
        return None
    if number.is_integer():
        return str(int(number))
    return format(number, ".12g")


def reference_answer(raw: Any) -> str:
    text = str(raw)
    if "####" in text:
        text = text.rsplit("####", 1)[1]
    normalized = normalize_number(text)
    if normalized is None:
        raise ValueError(f"reference answer has no numeric value: {raw!r}")
    return normalized


def load_problems(path: Path, limit: int | None) -> list[Problem]:
    problems: list[Problem] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            question = str(record["question"]).strip()
            if not question:
                raise ValueError(f"{path}:{line_number}: empty question")
            problems.append(
                Problem(
                    id=str(record.get("id", f"{path.stem}-{line_number}")),
                    question=question,
                    answer=reference_answer(record["answer"]),
                )
            )
            if limit is not None and len(problems) >= limit:
                break
    if not problems:
        raise ValueError(f"no problems loaded from {path}")
    return problems


def inferlet_paths() -> tuple[Path, Path, str]:
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib

    directory = ROOT / "inferlets" / INFERLET
    candidates = [
        directory / "target" / "wasm32-wasip2" / "release" / "reasoning_benchmark.wasm",
        directory / "target" / "wasm32-wasip2" / "debug" / "reasoning_benchmark.wasm",
    ]
    wasm = next((path for path in candidates if path.exists()), None)
    if wasm is None:
        raise FileNotFoundError(
            "reasoning-benchmark wasm is missing; build it with "
            "`cargo build --manifest-path inferlets/reasoning-benchmark/Cargo.toml "
            "--target wasm32-wasip2 --release`"
        )
    manifest = directory / "Pie.toml"
    package = tomllib.loads(manifest.read_text(encoding="utf-8"))["package"]
    return wasm, manifest, f"{package['name']}@{package['version']}"


def build_config(args: argparse.Namespace):
    from pie.config import (
        AuthConfig,
        Config,
        DriverConfig,
        ModelConfig,
        ServerConfig,
        TelemetryConfig,
    )

    device = [part.strip() for part in args.device.split(",")]
    options: dict[str, Any] = {}
    if args.driver in {"cuda_native", "portable"}:
        options["max_num_kv_pages"] = args.kv_pages
    if args.driver == "portable" and args.portable_n_gpu_layers is not None:
        options["n_gpu_layers"] = args.portable_n_gpu_layers
    return Config(
        server=ServerConfig(port=0, max_concurrent_processes=1),
        auth=AuthConfig(enabled=False),
        telemetry=TelemetryConfig(),
        models=[
            ModelConfig(
                name="default",
                hf_repo=args.model,
                driver=DriverConfig(
                    type=args.driver,
                    device=device,
                    options=options,
                ),
            )
        ],
    )


async def model_stats(client) -> dict[str, int]:
    ok, raw = await client.query("model_status", "")
    if not ok:
        raise RuntimeError(f"model_status query failed: {raw}")
    return {
        key: int(value)
        for key, value in json.loads(raw).items()
        if isinstance(value, (int, float))
    }


def stats_delta(before: dict[str, int], after: dict[str, int]) -> dict[str, int]:
    return {
        key: after.get(key, 0) - before.get(key, 0)
        for key in sorted(set(before) | set(after))
        if key.endswith((".total_batches", ".total_tokens_processed"))
    }


def payload_for(problem: Problem, pattern: str, args: argparse.Namespace) -> dict[str, Any]:
    return {
        "pattern": pattern,
        "question": problem.question,
        "num_candidates": args.num_candidates,
        "beam_width": args.beam_width,
        "max_tokens": args.max_tokens,
        "score_tokens": args.score_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "thinking": args.thinking,
    }


def result_for(
    problem: Problem,
    pattern: str,
    repetition: int,
    latency_s: float,
    output: dict[str, Any] | None,
    engine_stats_delta: dict[str, int],
    error: str | None,
) -> RunResult:
    predicted = normalize_number(output.get("final_answer") if output else None)
    candidate_answers = [
        normalize_number(candidate.get("answer"))
        for candidate in (output or {}).get("candidates", [])
    ]
    correct_candidates = sum(answer == problem.answer for answer in candidate_answers)
    return RunResult(
        problem_id=problem.id,
        pattern=pattern,
        repetition=repetition,
        correct=predicted == problem.answer,
        oracle_correct=correct_candidates > 0,
        correct_candidates=correct_candidates,
        expected_answer=problem.answer,
        predicted_answer=predicted,
        latency_s=latency_s,
        output=output,
        engine_stats_delta=engine_stats_delta,
        error=error,
    )


def print_result(result: RunResult) -> None:
    status = "correct" if result.correct else "wrong"
    if result.error:
        status = "error"
    print(
        f"{result.problem_id:18} {result.pattern:18} rep={result.repetition} "
        f"{status:7} answer={result.predicted_answer!r} "
        f"latency={result.latency_s:.2f}s",
        flush=True,
    )


def parse_pie_run_output(stdout: str) -> dict[str, Any]:
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            output = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(output, dict):
            return output
    raise ValueError("pie run did not emit a JSON object on stdout")


def run_cli(args: argparse.Namespace) -> list[RunResult]:
    if not args.config:
        raise ValueError("--config is required when --execution-mode=cli")

    problems = load_problems(Path(args.dataset), args.max_problems)
    wasm, manifest, _package = inferlet_paths()
    patterns = PATTERNS if args.pattern == "all" else (args.pattern,)
    results: list[RunResult] = []

    pie_bin = Path(args.pie_bin)
    config = Path(args.config)
    for problem in problems:
        for pattern in patterns:
            for repetition in range(args.repetitions):
                payload = payload_for(problem, pattern, args)
                cmd = [
                    str(pie_bin),
                    "run",
                    "--path",
                    str(wasm),
                    "--manifest",
                    str(manifest),
                    "--config",
                    str(config),
                    "--quiet",
                    "--input",
                    json.dumps(payload, separators=(",", ":")),
                ]
                started = time.perf_counter()
                output = None
                error = None
                try:
                    completed = subprocess.run(
                        cmd,
                        cwd=ROOT,
                        check=True,
                        capture_output=True,
                        text=True,
                        timeout=args.timeout,
                    )
                    output = parse_pie_run_output(completed.stdout)
                except Exception as exc:  # keep the full experiment running
                    error = f"{type(exc).__name__}: {exc}"
                latency_s = time.perf_counter() - started
                result = result_for(
                    problem,
                    pattern,
                    repetition,
                    latency_s,
                    output,
                    {},
                    error,
                )
                results.append(result)
                print_result(result)
    return results


async def run_embedded(args: argparse.Namespace) -> list[RunResult]:
    from pie.server import Server
    from pie_client import Event

    problems = load_problems(Path(args.dataset), args.max_problems)
    wasm, manifest, package = inferlet_paths()
    patterns = PATTERNS if args.pattern == "all" else (args.pattern,)
    results: list[RunResult] = []

    async with Server(build_config(args)) as server:
        client = await server.connect()
        await client.install_program(wasm, manifest, force_overwrite=True)

        for problem in problems:
            for pattern in patterns:
                for repetition in range(args.repetitions):
                    payload = payload_for(problem, pattern, args)
                    before = await model_stats(client)
                    started = time.perf_counter()
                    output = None
                    error = None
                    try:
                        process = await client.launch_process(package, input=payload)
                        while True:
                            event, message = await asyncio.wait_for(
                                process.recv(), timeout=args.timeout
                            )
                            if event == Event.Return:
                                output = json.loads(message)
                                break
                            if event == Event.Error:
                                raise RuntimeError(str(message))
                    except Exception as exc:  # keep the full experiment running
                        error = f"{type(exc).__name__}: {exc}"
                    latency_s = time.perf_counter() - started
                    after = await model_stats(client)
                    result = result_for(
                        problem,
                        pattern,
                        repetition,
                        latency_s,
                        output,
                        stats_delta(before, after),
                        error,
                    )
                    results.append(result)
                    print_result(result)
    return results


async def run(args: argparse.Namespace) -> list[RunResult]:
    if args.execution_mode == "cli":
        return run_cli(args)
    return await run_embedded(args)


def summarize(results: list[RunResult]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for pattern in sorted({result.pattern for result in results}):
        rows = [result for result in results if result.pattern == pattern]
        valid = [result for result in rows if result.error is None]
        latencies = [result.latency_s for result in valid]
        generated = [
            int(result.output["stats"]["generated_tokens"])
            for result in valid
            if result.output is not None
        ]
        summary[pattern] = {
            "runs": len(rows),
            "errors": sum(result.error is not None for result in rows),
            "accuracy": (
                sum(result.correct for result in rows) / len(rows) if rows else 0.0
            ),
            "oracle_candidate_accuracy": (
                sum(result.oracle_correct for result in rows) / len(rows)
                if rows
                else 0.0
            ),
            "mean_correct_candidates": (
                statistics.fmean(result.correct_candidates for result in rows)
                if rows
                else 0.0
            ),
            "mean_latency_s": statistics.fmean(latencies) if latencies else None,
            "mean_generated_tokens": statistics.fmean(generated) if generated else None,
        }
    return summary


def write_results(path: Path, args: argparse.Namespace, results: list[RunResult]) -> None:
    artifact = {
        "config": vars(args),
        "summary": summarize(results),
        "runs": [asdict(result) for result in results],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Pie reasoning-pattern benchmark")
    p.add_argument(
        "--execution-mode",
        choices=("embedded", "cli"),
        default="embedded",
        help="Use embedded pie.server or shell out to `pie run`.",
    )
    p.add_argument(
        "--pie-bin",
        default="./target/release/pie",
        help="Pie binary to use when --execution-mode=cli.",
    )
    p.add_argument(
        "--config",
        default=None,
        help="Pie config TOML. Required when --execution-mode=cli.",
    )
    p.add_argument("--dataset", default=str(ROOT / "benches" / "reasoning_smoke.jsonl"))
    p.add_argument("--pattern", choices=("all", *PATTERNS), default="all")
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument(
        "--driver",
        choices=("dev", "dummy", "cuda_native", "portable", "vllm", "sglang"),
        default="dev",
    )
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max-problems", type=int, default=None)
    p.add_argument("--repetitions", type=int, default=1)
    p.add_argument("--num-candidates", type=int, default=4)
    p.add_argument("--beam-width", type=int, default=2)
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--score-tokens", type=int, default=16)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.95)
    thinking = p.add_mutually_exclusive_group()
    thinking.add_argument(
        "--thinking",
        dest="thinking",
        action="store_true",
        help="Allow model thinking blocks.",
    )
    thinking.add_argument(
        "--no-thinking",
        dest="thinking",
        action="store_false",
        help="Use the model/template no-thinking path. This is the default.",
    )
    p.set_defaults(thinking=False)
    p.add_argument("--timeout", type=float, default=300.0)
    p.add_argument("--kv-pages", type=int, default=2048)
    p.add_argument("--portable-n-gpu-layers", type=int, default=None)
    p.add_argument("--json-out", default=str(ROOT / ".tmp" / "reasoning-benchmark.json"))
    return p


def main() -> None:
    args = parser().parse_args()
    results = asyncio.run(run(args))
    summary = summarize(results)
    print("\n" + json.dumps(summary, indent=2))
    write_results(Path(args.json_out), args, results)


if __name__ == "__main__":
    main()
