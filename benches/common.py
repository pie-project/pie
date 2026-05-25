from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import json
import math
import os
import re
import statistics
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
BENCH_SYSTEM = "You are a helpful benchmarking assistant."
BENCH_PROMPT = "Write a short story about a robot."


@dataclass
class RequestResult:
    ok: bool
    latency_s: float
    output_tokens: int
    prompt_tokens: int | None = None
    error: str | None = None


@dataclass
class BenchSummary:
    mode: str
    engine: str
    model: str
    requests: int
    completed: int
    failed: int
    wall_s: float
    output_tokens: int
    prompt_tokens: int | None
    req_per_s: float
    output_tok_per_s: float
    latency_mean_ms: float | None
    latency_p50_ms: float | None
    latency_p95_ms: float | None
    latency_p99_ms: float | None
    config: dict[str, Any]


def percentile(xs: list[float], q: float) -> float | None:
    if not xs:
        return None
    s = sorted(xs)
    k = (len(s) - 1) * q
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return s[int(k)]
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def summarize(
    *,
    mode: str,
    engine: str,
    model: str,
    results: list[RequestResult],
    wall_s: float,
    config: dict[str, Any],
) -> BenchSummary:
    ok = [r for r in results if r.ok]
    lats = [r.latency_s for r in ok if r.latency_s > 0.0]
    output_tokens = sum(r.output_tokens for r in ok)
    prompt_known = [r.prompt_tokens for r in ok if r.prompt_tokens is not None]
    prompt_tokens = sum(prompt_known) if len(prompt_known) == len(ok) else None
    return BenchSummary(
        mode=mode,
        engine=engine,
        model=model,
        requests=len(results),
        completed=len(ok),
        failed=len(results) - len(ok),
        wall_s=wall_s,
        output_tokens=output_tokens,
        prompt_tokens=prompt_tokens,
        req_per_s=(len(ok) / wall_s) if wall_s > 0 else 0.0,
        output_tok_per_s=(output_tokens / wall_s) if wall_s > 0 else 0.0,
        latency_mean_ms=(statistics.fmean(lats) * 1000.0) if lats else None,
        latency_p50_ms=(percentile(lats, 0.50) * 1000.0) if lats else None,
        latency_p95_ms=(percentile(lats, 0.95) * 1000.0) if lats else None,
        latency_p99_ms=(percentile(lats, 0.99) * 1000.0) if lats else None,
        config=config,
    )


def print_summary(s: BenchSummary) -> None:
    print()
    print("-" * 52)
    print(f"mode:              {s.mode}")
    print(f"engine:            {s.engine}")
    print(f"model:             {s.model}")
    print(f"completed:         {s.completed}/{s.requests}  failed={s.failed}")
    print(f"wall:              {s.wall_s:.3f} s")
    print(f"requests/sec:      {s.req_per_s:.2f}")
    print(f"output tokens:     {s.output_tokens}")
    if s.prompt_tokens is not None:
        print(f"prompt tokens:     {s.prompt_tokens}")
    print(f"output tok/sec:    {s.output_tok_per_s:.2f}")
    if s.latency_mean_ms is not None:
        print(f"lat mean/p50:      {s.latency_mean_ms:.1f} / {s.latency_p50_ms:.1f} ms")
        print(f"lat p95/p99:       {s.latency_p95_ms:.1f} / {s.latency_p99_ms:.1f} ms")
    # Speculation counters live in the config blob, sourced from the
    # server's `model_status` query. Only printed when present so
    # baseline runs (no speculation capability) stay quiet.
    spec_keys = (
        "spec attempted",
        "spec hits",
        "spec misses",
        "spec rule skipped",
        "spec budget skipped",
        "spec dropped orphan",
        "spec need pages",
        "spec chain now",
        "spec chain peak",
        "spec longest chain",
        "total batches",
        "cumulative batch latency us",
        "avg batch latency us",
        "last batch latency us",
        "system spec draft tokens proposed",
        "system spec draft tokens accepted",
        "bypass hits",
        "chain submits",
        "chain drops",
        "total requests",
        "max forward requests",
        "batch size hist",
        "runtime launch ack mean ms",
        "runtime launch ack p50 ms",
        "runtime launch ack p95 ms",
        "runtime launch ack max ms",
        "runtime launch ack before start ms",
        "runtime first launch ack ms",
        "runtime all launch ack ms",
        "runtime ready before start ms",
        "runtime all ready ms",
        "runtime first return ms",
        "runtime last return ms",
        "runtime driver cumulative ms",
        "runtime wall minus driver ms",
        "runtime non-driver after launch ms",
        "vllm spec drafts",
        "vllm spec draft tokens",
        "vllm spec accepted tokens",
        "vllm spec accepted per position",
        "vllm spec acceptance rate",
        "vllm spec mean acceptance length",
    )
    if any(k in s.config for k in spec_keys):
        for k in spec_keys:
            if k in s.config:
                print(f"{k+':':<18} {s.config[k]}")
        # Derived efficiency metrics. `hit rate` is the fraction of
        # the inferlet's execute() calls that short-circuited via
        # the chain. `chain yield` is how much of the theoretical
        # max (attempts × depth_cap) we actually captured as hits;
        # 1.0 means every chain entry the driver fired was matched
        # by an inferlet call, lower means some entries got
        # truncated (page boundaries) or orphaned (ctx ended).
        hits = s.config.get("spec hits", 0)
        misses = s.config.get("spec misses", 0)
        attempted = s.config.get("spec attempted", 0)
        depth_cap = s.config.get("max chain depth")
        total_calls = hits + misses + attempted
        if total_calls > 0:
            print(f"{'spec hit rate:':<18} {hits / total_calls:.1%}")
        if depth_cap and attempted > 0:
            theoretical = attempted * depth_cap
            print(f"{'spec chain yield:':<18} {hits / theoretical:.1%}")
    print("-" * 52)


def write_json(path: str | None, summary: BenchSummary, results: list[RequestResult]) -> None:
    if not path:
        return
    out = {"summary": asdict(summary), "requests": [asdict(r) for r in results]}
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2))


def add_mode_subcommands(parser: argparse.ArgumentParser) -> None:
    sub = parser.add_subparsers(dest="mode", required=True)
    latency = sub.add_parser("latency", help="single-request latency")
    add_common_args(latency)
    latency.add_argument("--requests", type=int, default=16)
    latency.set_defaults(num_requests=0, concurrency=1)

    tput = sub.add_parser("tput", help="many-request throughput")
    add_common_args(tput)
    tput.add_argument("--num-requests", type=int, default=512)
    tput.add_argument(
        "--concurrency",
        type=int,
        default=0,
        help="Throughput concurrency cap. 0 means unlimited.",
    )
    tput.set_defaults(requests=0)


def add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", default="Qwen/Qwen2-0.5B")
    p.add_argument("--prompt", default=BENCH_PROMPT)
    p.add_argument("--system", default=BENCH_SYSTEM)
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--ignore-eos", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--unique-prompts", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument(
        "--warmup-max-tokens",
        type=int,
        default=None,
        help="Override max_tokens only for warmup requests.",
    )
    p.add_argument("--json-out", default=None)
    p.add_argument(
        "--cuda-profiler-capture",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Call cudaProfilerStart/Stop around the measured section. Use with "
             "`nsys profile --capture-range=cudaProfilerApi` to exclude setup "
             "and warmup from the trace.",
    )
    p.add_argument("--request-timeout", type=float, default=300.0)
    p.add_argument("--tp-size", type=int, default=1)
    p.add_argument(
        "--cpu-affinity",
        default="auto",
        help="CPU affinity for the benchmark process. Use 'auto' to pin to "
             "the GPU-local CPUs from `nvidia-smi topo -m`, 'none' to disable, "
             "or an explicit list like '80-87,208-215'.",
    )
    p.add_argument("--gpu-mem-util", type=float, default=0.90)
    p.add_argument("--max-model-len", type=int, default=2048)
    p.add_argument("--sglang-attention-backend", default=None)
    p.add_argument("--sglang-sampling-backend", default=None)
    p.add_argument("--sglang-disable-cuda-graph", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--sglang-disable-piecewise-cuda-graph", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument(
        "--wasm-delay-us",
        type=int,
        default=0,
        help="Busy-spin in the bench inferlet's WASM between every "
             "execute() call (microseconds). Simulates per-token "
             "WASM work — useful for measuring the wall-clock benefit "
             "of async chain firing on W>0 workloads. Default 0.",
    )


def make_prompts(args: argparse.Namespace, n: int) -> list[str]:
    if args.unique_prompts:
        return [f"{args.prompt} (Request #{i})" for i in range(n)]
    return [args.prompt for _ in range(n)]


def hf_chat_prompts_and_counts(
    model: str, system: str, prompts: list[str]
) -> tuple[list[str], list[int]]:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model)
    rendered = [
        tok.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": p}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for p in prompts
    ]
    counts = [len(tok.encode(p, add_special_tokens=False)) for p in rendered]
    return rendered, counts


def hf_chat_token_ids_and_counts(
    model: str, system: str, prompts: list[str]
) -> tuple[list[list[int]], list[int]]:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    rendered = [
        tok.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": p}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for p in prompts
    ]
    token_ids = [tok.encode(p, add_special_tokens=False) for p in rendered]
    return token_ids, [len(ids) for ids in token_ids]


def finish(summary: BenchSummary, results: list[RequestResult], json_out: str | None) -> None:
    print_summary(summary)
    write_json(json_out, summary, results)
    if summary.failed:
        print(f"{summary.failed} request(s) failed; inspect JSON output for details.")


def _load_cudart():
    candidates = []
    found = ctypes.util.find_library("cudart")
    if found:
        candidates.append(found)
    candidates.extend(
        [
            "libcudart.so",
            "libcudart.so.12",
            "/usr/local/cuda/lib64/libcudart.so",
            "/usr/local/cuda-12.8/lib64/libcudart.so",
        ]
    )
    last_error: Exception | None = None
    for candidate in candidates:
        try:
            return ctypes.CDLL(candidate)
        except OSError as exc:
            last_error = exc
    raise RuntimeError(f"could not load CUDA runtime: {last_error}")


def cuda_profiler_start(enabled: bool) -> None:
    if not enabled:
        return
    cudart = _load_cudart()
    rc = int(cudart.cudaProfilerStart())
    if rc != 0:
        raise RuntimeError(f"cudaProfilerStart failed with CUDA error {rc}")


def cuda_profiler_stop(enabled: bool) -> None:
    if not enabled:
        return
    cudart = _load_cudart()
    rc = int(cudart.cudaProfilerStop())
    if rc != 0:
        raise RuntimeError(f"cudaProfilerStop failed with CUDA error {rc}")


def _parse_cpu_list(spec: str) -> set[int]:
    cpus: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            cpus.update(range(int(lo), int(hi) + 1))
        else:
            cpus.add(int(part))
    return cpus


def _format_cpu_list(cpus: set[int]) -> str:
    if not cpus:
        return ""
    xs = sorted(cpus)
    ranges: list[str] = []
    start = prev = xs[0]
    for x in xs[1:]:
        if x == prev + 1:
            prev = x
            continue
        ranges.append(f"{start}-{prev}" if start != prev else str(start))
        start = prev = x
    ranges.append(f"{start}-{prev}" if start != prev else str(start))
    return ",".join(ranges)


def gpu_local_cpu_affinity(gpu_ids: list[int]) -> set[int]:
    """Return the union of CPU-affinity masks reported by `nvidia-smi topo -m`."""
    if not gpu_ids:
        return set()
    try:
        topo = subprocess.check_output(
            ["nvidia-smi", "topo", "-m"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return set()
    ansi = re.compile(r"\x1b\[[0-9;]*m")
    lines = [ansi.sub("", line).strip() for line in topo.splitlines() if line.strip()]
    if not lines:
        return set()
    header = lines[0].split()
    gpu_cols = sum(1 for tok in header if re.fullmatch(r"GPU\d+", tok))
    affinity_by_gpu: dict[int, set[int]] = {}
    for line in lines[1:]:
        toks = line.split()
        if len(toks) <= gpu_cols + 1 or not re.fullmatch(r"GPU\d+", toks[0]):
            continue
        affinity_by_gpu[int(toks[0][3:])] = _parse_cpu_list(toks[1 + gpu_cols])
    cpus: set[int] = set()
    for gpu in gpu_ids:
        cpus.update(affinity_by_gpu.get(gpu, set()))
    return cpus


def visible_cuda_devices(tp_size: int) -> list[int]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        ids: list[int] = []
        for item in visible.split(","):
            item = item.strip()
            if item.isdigit():
                ids.append(int(item))
        if ids:
            return ids[: max(1, tp_size)]
    return list(range(max(1, tp_size)))


def maybe_set_cpu_affinity(args: argparse.Namespace, gpu_ids: list[int]) -> str | None:
    mode = getattr(args, "cpu_affinity", "auto")
    if mode in (None, "", "none"):
        return None
    cpus = gpu_local_cpu_affinity(gpu_ids) if mode == "auto" else _parse_cpu_list(mode)
    if not cpus or not hasattr(os, "sched_setaffinity"):
        return None
    os.sched_setaffinity(0, cpus)
    return _format_cpu_list(cpus)
