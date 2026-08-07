from __future__ import annotations

import argparse
import asyncio
import ctypes
import ctypes.util
import json
import math
import os
import random
import re
import statistics
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
BENCH_SYSTEM = "You are a helpful benchmarking assistant."
BENCH_PROMPT = "Write a short story about a robot."


def resolve_local_model(model: str) -> str:
    candidate = Path(model).expanduser()
    if candidate.exists():
        return str(candidate.resolve())
    encoded = f"models--{model.replace('/', '--')}"
    roots = []
    if cache := os.environ.get("HF_HUB_CACHE"):
        roots.append(Path(cache).expanduser())
    if home := os.environ.get("HF_HOME"):
        roots.append(Path(home).expanduser() / "hub")
    roots.extend(
        (
            Path(os.environ.get("XDG_CACHE_HOME", "~/.cache")).expanduser()
            / "huggingface"
            / "hub",
            Path("~/.pie/programs").expanduser(),
        )
    )
    for root in roots:
        repository = root / encoded
        revision = None
        for ref_name in ("main", "master"):
            ref = repository / "refs" / ref_name
            if ref.is_file():
                revision = ref.read_text().strip()
                break
        if revision:
            snapshot = repository / "snapshots" / revision
            if snapshot.is_dir():
                return str(snapshot.resolve())
    raise FileNotFoundError(
        f"no local snapshot for {model!r}; download it before benchmarking"
    )


@dataclass
class RequestResult:
    ok: bool
    latency_s: float
    output_tokens: int
    prompt_tokens: int | None = None
    error: str | None = None
    # Populated only when the bench requests timing (--report-timing):
    # ttft_s is launch-inclusive (client stamp on the engine's first-token
    # signal); intertoken_us are gaps between successive token deliveries.
    ttft_s: float | None = None
    intertoken_us: list[int] | None = None
    # Absolute client-vantage stamps relative to the measured-run epoch.
    # Populated only for measured requests under --report-timing.
    client_send_s: float | None = None
    token_arrival_s: list[float] | None = None
    token_arrival_monotonic_ns: list[int] | None = None
    client_return_s: float | None = None
    process_id: str | None = None
    # pie only: guest prologue step durations in us, ordered
    # [main_pre, tokenize, setup, reserve, build_submit].
    prologue_us: list[int] | None = None
    # SHA-256 over emitted u32 token ids in little-endian order. Pie's
    # multi-process throughput path records this for every measured request.
    output_token_sha256: str | None = None
    # Diagnostic-only raw sequence, populated by --dump-all-token-ids.
    output_token_ids: list[int] | None = None
    # Diagnostic-only decoded sequence, populated by --dump-all-texts.
    output_text: str | None = None


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
    # Timing dimensions (present only when the bench collected them):
    ttft_mean_ms: float | None = None
    ttft_p50_ms: float | None = None
    ttft_p95_ms: float | None = None
    ttft_p99_ms: float | None = None
    # Inter-token gaps pooled across all requests' accepted tokens.
    intertoken_p50_ms: float | None = None
    intertoken_p99_ms: float | None = None
    intertoken_max_ms: float | None = None


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


def arrival_schedule(
    n: int, rate: float, process: str, seed: int
) -> list[float]:
    """Offsets in seconds, from the start of the measured run, at which each
    request should be submitted.

    `rate <= 0` reproduces the historical behaviour: every request is offered
    at t=0 and the engine's admission cap is the only thing pacing them. That
    is a closed-loop saturation test and it cannot distinguish an engine that
    is fast from one that is merely deep, because the offered load adapts to
    whatever the engine can absorb. A finite rate makes the load exogenous, so
    queueing delay shows up as latency instead of hiding in the arrival
    process.

    The schedule is derived from `seed` alone, so every engine in a comparison
    is offered the byte-identical arrival pattern.
    """
    if rate <= 0:
        return [0.0] * n
    rng = random.Random(seed)
    offsets: list[float] = []
    t = 0.0
    for _ in range(n):
        offsets.append(t)
        # Poisson arrivals are the bursty case: the exponential gap
        # distribution produces clumps that a uniform schedule never does, and
        # clumps are what actually exercise admission.
        t += rng.expovariate(rate) if process == "poisson" else 1.0 / rate
    return offsets


class ArrivalPacer:
    """Holds a submission schedule so both harnesses pace identically.

    Also records how far behind schedule each submission actually went out.
    A large lag means the *client* saturated, not the engine, and any latency
    read from that run is measuring asyncio rather than the server.
    """

    def __init__(self, offsets: list[float]) -> None:
        self._offsets = offsets
        self._t0: float | None = None
        self.lag_s: list[float] = []

    @property
    def enabled(self) -> bool:
        return any(o > 0.0 for o in self._offsets)

    def start(self) -> None:
        self._t0 = time.perf_counter()

    async def wait(self, index: int) -> None:
        if self._t0 is None or index >= len(self._offsets):
            return
        due = self._t0 + self._offsets[index]
        delay = due - time.perf_counter()
        if delay > 0:
            await asyncio.sleep(delay)
        self.lag_s.append(max(0.0, time.perf_counter() - due))

    def stats(self) -> dict[str, Any]:
        if not self.lag_s:
            return {}
        return {
            "arrival_lag_p50_ms": (percentile(self.lag_s, 0.50) or 0.0) * 1e3,
            "arrival_lag_p99_ms": (percentile(self.lag_s, 0.99) or 0.0) * 1e3,
            "arrival_lag_max_ms": max(self.lag_s) * 1e3,
            "arrival_span_s": self._offsets[-1] if self._offsets else 0.0,
        }


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
    ttfts = [r.ttft_s for r in ok if getattr(r, "ttft_s", None) is not None]
    gaps_ms = [
        g / 1000.0
        for r in ok
        for g in (getattr(r, "intertoken_us", None) or [])
    ]
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
        ttft_mean_ms=(statistics.fmean(ttfts) * 1000.0) if ttfts else None,
        ttft_p50_ms=(percentile(ttfts, 0.50) * 1000.0) if ttfts else None,
        ttft_p95_ms=(percentile(ttfts, 0.95) * 1000.0) if ttfts else None,
        ttft_p99_ms=(percentile(ttfts, 0.99) * 1000.0) if ttfts else None,
        intertoken_p50_ms=percentile(gaps_ms, 0.50) if gaps_ms else None,
        intertoken_p99_ms=percentile(gaps_ms, 0.99) if gaps_ms else None,
        intertoken_max_ms=max(gaps_ms) if gaps_ms else None,
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
    if s.ttft_mean_ms is not None:
        print(f"ttft mean/p50:     {s.ttft_mean_ms:.1f} / {s.ttft_p50_ms:.1f} ms")
        print(f"ttft p95/p99:      {s.ttft_p95_ms:.1f} / {s.ttft_p99_ms:.1f} ms")
    if s.intertoken_p50_ms is not None:
        print(
            f"intertoken p50/p99/max: {s.intertoken_p50_ms:.2f} / "
            f"{s.intertoken_p99_ms:.2f} / {s.intertoken_max_ms:.2f} ms"
        )
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
        "system spec draft tokens proposed per pos",
        "system spec draft tokens accepted per pos",
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
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--prompt", default=BENCH_PROMPT)
    p.add_argument("--system", default=BENCH_SYSTEM)
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--ignore-eos", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--unique-prompts", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--shared-prefix-words",
        type=int,
        default=0,
        help="Prefix-heavy shape: prepend a deterministic shared block of "
             "roughly this many tokens to every prompt (unique tails kept). "
             "Pair with prefix caching on engines that support it.",
    )
    p.add_argument(
        "--output-spread",
        type=float,
        default=0.0,
        help="Fan per-request output budgets out over "
             "[max_tokens*(1-F), max_tokens*(1+F)] on a deterministic "
             "sawtooth with mean exactly max_tokens. Breaks the lockstep a "
             "uniform shape imposes on a closed-loop client without changing "
             "the total work.",
    )
    p.add_argument(
        "--mixed-phase",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Mixed-phase shape: even-indexed requests are prefill-heavy "
             "(long prompt, --mixed-short-output tokens out), odd-indexed "
             "are decode-heavy (short prompt, --max-tokens out). Run with "
             "prefix caching OFF on both engines.",
    )
    p.add_argument(
        "--mixed-long-prompt-words",
        type=int,
        default=400,
        help="Word count of the long prompt body in --mixed-phase.",
    )
    p.add_argument(
        "--mixed-short-output",
        type=int,
        default=16,
        help="max_tokens for the prefill-heavy half in --mixed-phase.",
    )
    p.add_argument(
        "--arrival-rate",
        type=float,
        default=0.0,
        help="Open-loop offered load in requests/second. 0 (the default) "
             "offers every request at t=0, which measures saturation only. "
             "A finite rate makes the load exogenous so queueing delay shows "
             "up in TTFT instead of being absorbed by the arrival process.",
    )
    p.add_argument(
        "--arrival-process",
        choices=["poisson", "uniform"],
        default="poisson",
        help="Inter-arrival distribution for --arrival-rate.",
    )
    p.add_argument(
        "--arrival-seed",
        type=int,
        default=20260730,
        help="Seeds the arrival schedule. Engines compared against each "
             "other must share this so they see the identical pattern.",
    )
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument(
        "--warmup-seconds",
        type=float,
        default=0.0,
        help="Keep repeating the warmup set until this much wall time has "
             "elapsed, on top of --warmup. Off by default: on this hardware "
             "the run-to-run spread came from CPU pinning, not from a short "
             "warmup, and a longer warmup did not reduce it.",
    )
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
    p.add_argument(
        "--cuda-profiler-delay-s",
        type=float,
        default=0.0,
        help="Delay cudaProfilerStart after measured work begins.",
    )
    p.add_argument(
        "--cuda-profiler-duration-s",
        type=float,
        default=0.0,
        help="Stop a delayed profiler capture after this duration. Zero keeps "
        "the legacy whole-measured-window capture.",
    )
    p.add_argument("--request-timeout", type=float, default=300.0)
    p.add_argument("--tp-size", type=int, default=1)
    p.add_argument(
        "--dp-size",
        type=int,
        default=1,
        help="Data-parallel replicas. Each replica is its own tp-size-wide "
             "shard group, so the run occupies tp_size * dp_size devices. "
             "PIE derives this from the device list (world_size / tp_size); "
             "vLLM takes it as data_parallel_size.",
    )
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
    p.add_argument(
        "--sglang-max-total-tokens",
        type=int,
        default=None,
        help="Pin SGLang's KV pool to this many tokens. The cross-engine "
             "analogue of pie's `--total-pages P` and vLLM's "
             "`--num-gpu-blocks-override P --block-size B`: without it the "
             "pool is sized from --gpu-mem-util and the three engines are "
             "not given the same KV budget.",
    )
    p.add_argument(
        "--sglang-page-size",
        type=int,
        default=None,
        help="SGLang KV page size in tokens. Defaults to SGLang's own "
             "default (1). Set to 16 to match vLLM's --block-size and pie's "
             "page granularity so internal fragmentation is comparable too.",
    )
    p.add_argument(
        "--sglang-cuda-graph-max-bs",
        type=int,
        default=None,
        help="Largest decode batch SGLang captures a CUDA graph for. "
             "SGLang's default is a hardware-tier heuristic keyed on total "
             "GPU memory (32 on an L40S at tp<4), so a decode batch of 128 "
             "runs eager and a small model loses most of its throughput to "
             "launch overhead. vLLM captures up to max_num_seqs, so leaving "
             "this at SGLang's default measures the heuristic, not the "
             "engine. Set it to the concurrency.",
    )
    p.add_argument("--sglang-disable-cuda-graph", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--sglang-disable-piecewise-cuda-graph", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument(
        "--sglang-disable-hybrid-swa-memory",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Size one uniform KV pool instead of SGLang's split "
             "sliding-window/full-attention pools. On a sliding-window model "
             "(Gemma) the hybrid sizer derives a token budget from the cheap "
             "SWA layers and then refuses to start because the full-attention "
             "layers cannot hold that many tokens — SGLang aborts where vLLM "
             "just sizes the pool from free memory and schedules within it. "
             "Set this to get a run at all, and to put both engines on the "
             "same pool policy.",
    )
    p.add_argument(
        "--wasm-delay-us",
        type=int,
        default=0,
        help="Busy-spin in the bench inferlet's WASM between every "
             "execute() call (microseconds). Simulates per-token "
             "WASM work — useful for measuring the wall-clock benefit "
             "of async chain firing on W>0 workloads. Default 0.",
    )
    p.add_argument(
        "--run-ahead-frames",
        type=int,
        default=None,
        help="Decode frames each pie lane keeps queued in the engine. "
             "The wave quorum waits for every lane, so deeper queues hide "
             "guest turnaround. pie only; unset means the engine sizes the "
             "window itself via model.channel-capacity().",
    )


def gpu_clock_state() -> list[dict[str, Any]]:
    """Per-GPU SM clock and utilization, or [] when nvidia-smi is unavailable.

    Recorded on both sides of the measured window so a run that executed at
    a partially-ramped clock is visible in the result instead of silently
    reading low.
    """
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=index,clocks.sm,clocks.max.sm,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=True).stdout
    except Exception:
        return []
    state = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 4:
            continue
        try:
            state.append({
                "index": int(parts[0]),
                "sm_mhz": int(parts[1]),
                "sm_max_mhz": int(parts[2]),
                "util_pct": int(parts[3]),
            })
        except ValueError:
            continue
    return state


async def run_timed_warmup(run_once, seconds: float, label: str = "pie") -> int:
    """Repeat `run_once()` until `seconds` of wall time have elapsed.

    Returns the number of extra passes performed. `run_once` is an async
    callable that issues one full warmup set. Disabled (0) by default.
    """
    if seconds <= 0:
        return 0
    import time as _time
    start = _time.monotonic()
    passes = 0
    while _time.monotonic() - start < seconds:
        await run_once()
        passes += 1
    print(f"[{label}-bench] warmup: {passes} extra pass(es) over "
          f"{_time.monotonic() - start:.1f}s", flush=True)
    return passes


def run_timed_warmup_sync(run_once, seconds: float, label: str = "vllm") -> int:
    """Blocking counterpart of `run_timed_warmup`."""
    if seconds <= 0:
        return 0
    import time as _time
    start = _time.monotonic()
    passes = 0
    while _time.monotonic() - start < seconds:
        run_once()
        passes += 1
    print(f"[{label}-bench] warmup: {passes} extra pass(es) over "
          f"{_time.monotonic() - start:.1f}s", flush=True)
    return passes


def _filler_words(count: int) -> str:
    base = (
        "the quick brown fox jumps over the lazy dog and then walks "
        "back home again to rest before the next long day begins"
    ).split()
    return " ".join(base[i % len(base)] for i in range(count))


OUTPUT_SPREAD_PERIOD = 16


def request_max_tokens(args: argparse.Namespace, i: int) -> int:
    """Per-request output budget: uniform unless --mixed-phase.

    `--output-spread` fans the budget out around `--max-tokens` on a
    deterministic sawtooth whose mean over each period is exactly 1, so the
    total output token count is preserved and the ledger's same-work guard
    still holds. Both engines call this, so they see identical budgets.
    """
    if getattr(args, "mixed_phase", False) and i % 2 == 0:
        return args.mixed_short_output
    spread = getattr(args, "output_spread", 0.0) or 0.0
    if spread > 0.0:
        frac = (i % OUTPUT_SPREAD_PERIOD) / (OUTPUT_SPREAD_PERIOD - 1)
        return max(1, round(args.max_tokens * (1.0 + spread * (2.0 * frac - 1.0))))
    return args.max_tokens


def request_max_tokens_varies(args: argparse.Namespace) -> bool:
    """True when `request_max_tokens` is not constant across requests."""
    if getattr(args, "mixed_phase", False):
        return True
    return (getattr(args, "output_spread", 0.0) or 0.0) > 0.0


def hash_output_tokens(token_ids: list[int]) -> str:
    """SHA-256 over emitted u32 token ids, little-endian.

    Shared by every engine harness so a cross-engine parity check is a
    string comparison and not a re-derivation of the same digest twice.
    """
    import hashlib

    digest = hashlib.sha256()
    for token in token_ids:
        digest.update(int(token).to_bytes(4, "little", signed=False))
    return digest.hexdigest()


def add_output_dump_args(p: argparse.ArgumentParser) -> None:
    """Cross-engine output-inspection flags (parity checking)."""
    p.add_argument("--dump-first-text", action="store_true")
    p.add_argument("--dump-all-token-ids", action="store_true")
    p.add_argument("--dump-all-texts", action="store_true")


def print_first_output(args: argparse.Namespace, text: str | None) -> None:
    if not getattr(args, "dump_first_text", False) or text is None:
        return
    import hashlib

    sha = hashlib.sha256(text.encode()).hexdigest()[:16]
    print(f"\nFIRST REQUEST OUTPUT (sha256[:16]={sha}):")
    print(text)
    print(f"END OUTPUT (chars={len(text)})")


def make_prompts(args: argparse.Namespace, n: int) -> list[str]:
    if getattr(args, "mixed_phase", False):
        long_body = _filler_words(getattr(args, "mixed_long_prompt_words", 400))
        return [
            f"{long_body}. {args.prompt} (Request #{i})"
            if i % 2 == 0
            else f"{args.prompt} (Request #{i})"
            for i in range(n)
        ]
    shared_words = getattr(args, "shared_prefix_words", 0) or 0
    if shared_words > 0:
        # Prefix-heavy shape: every prompt opens with the SAME long
        # instruction block (deterministic filler; ~1 token per word)
        # followed by a short unique tail. Exercises prefix caching on
        # engines that support it.
        base = (
            "the quick brown fox jumps over the lazy dog and then walks "
            "back home again to rest before the next long day begins"
        ).split()
        filler = " ".join(base[i % len(base)] for i in range(shared_words))
        shared = (
            "You will answer questions about the following policy. "
            + filler
            + ". Policy ends. "
        )
        return [f"{shared}{args.prompt} (Request #{i})" for i in range(n)]
    if args.unique_prompts:
        return [f"{args.prompt} (Request #{i})" for i in range(n)]
    return [args.prompt for _ in range(n)]


def _render_chat(tok, system: str, prompts: list[str]) -> list[str]:
    """Chat-render `prompts`, falling back to plain text when the tokenizer
    ships no chat template.

    Some checkpoints (gemma-4-E4B) have no `chat_template`, and
    `apply_chat_template` then raises. PIE tokenizes prompts itself and was
    unaffected, but vLLM went through this path and failed outright, which made
    the model impossible to compare. Both engines get the same rendered string
    either way, so the comparison stays honest.
    """
    if getattr(tok, "chat_template", None):
        return [
            tok.apply_chat_template(
                [{"role": "system", "content": system},
                 {"role": "user", "content": p}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for p in prompts
        ]
    return [f"{system}\n\n{p}" if system else p for p in prompts]


def hf_chat_prompts_and_counts(
    model: str, system: str, prompts: list[str]
) -> tuple[list[str], list[int]]:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    rendered = _render_chat(tok, system, prompts)
    counts = [len(tok.encode(p, add_special_tokens=False)) for p in rendered]
    return rendered, counts


def hf_chat_token_ids_and_counts(
    model: str, system: str, prompts: list[str]
) -> tuple[list[list[int]], list[int]]:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    rendered = _render_chat(tok, system, prompts)
    token_ids = [tok.encode(p, add_special_tokens=False) for p in rendered]
    return token_ids, [len(ids) for ids in token_ids]


def finish(summary: BenchSummary, results: list[RequestResult], json_out: str | None) -> None:
    print_summary(summary)
    write_json(json_out, summary, results)
    if summary.failed:
        print(f"{summary.failed} request(s) failed; inspect JSON output for details.")


def _load_cudart():
    candidates = []
    if override := os.environ.get("CUDART_LIBRARY"):
        candidates.append(override)
    for env_name in ("CUDA_HOME", "CUDA_PATH"):
        if cuda_root := os.environ.get(env_name):
            candidates.extend(
                (
                    str(Path(cuda_root) / "lib64" / "libcudart.so"),
                    str(
                        Path(cuda_root)
                        / "targets"
                        / "x86_64-linux"
                        / "lib"
                        / "libcudart.so"
                    ),
                )
            )
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


def _numa_node_cpus(node: int) -> set[int]:
    try:
        text = Path(
            f"/sys/devices/system/node/node{node}/cpulist").read_text()
    except OSError:
        return set()
    return _parse_cpu_list(text.strip())


def gpu_local_cpu_affinity(gpu_ids: list[int]) -> set[int]:
    """CPUs local to `gpu_ids`, preferring the GPU's whole NUMA node.

    `nvidia-smi topo -m` has a "CPU Affinity" column, but on some hosts it
    reports only a handful of cores (six, on a 64-core-per-node Xeon) while the
    GPU's actual NUMA node holds far more. Pinning the whole benchmark — client,
    engine, tokio workers and every concurrent inferlet — onto that short list
    starves the host-side loop and leaves the GPU idle between batches: measured
    here as a 1.8x drop in median throughput and a 2.1x run-to-run spread versus
    the same run unpinned. So prefer the NUMA node's full CPU list and fall back
    to the topology column only when the node is unknown.
    """
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
    gpu_cols = sum(1 for tok in header if re.fullmatch(r"(?:GPU|NIC)\d+", tok))
    affinity_by_gpu: dict[int, set[int]] = {}
    numa_by_gpu: dict[int, int] = {}
    for line in lines[1:]:
        toks = line.split()
        if len(toks) <= gpu_cols + 1 or not re.fullmatch(r"GPU\d+", toks[0]):
            continue
        gpu = int(toks[0][3:])
        affinity_by_gpu[gpu] = _parse_cpu_list(toks[1 + gpu_cols])
        if len(toks) > gpu_cols + 2:
            try:
                numa_by_gpu[gpu] = int(toks[2 + gpu_cols])
            except ValueError:
                pass
    cpus: set[int] = set()
    for gpu in gpu_ids:
        node = numa_by_gpu.get(gpu)
        node_cpus = _numa_node_cpus(node) if node is not None else set()
        cpus.update(node_cpus or affinity_by_gpu.get(gpu, set()))
    # Never widen past what this process is actually allowed to run on.
    if hasattr(os, "sched_getaffinity"):
        cpus &= set(os.sched_getaffinity(0))
    return cpus


def visible_cuda_devices(tp_size: int, dp_size: int = 1) -> list[int]:
    # A run occupies one device per rank: tp_size ranks per replica,
    # dp_size replicas.
    world = max(1, tp_size) * max(1, dp_size)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        ids: list[int] = []
        for item in visible.split(","):
            item = item.strip()
            if item.isdigit():
                ids.append(int(item))
        if ids:
            return ids[:world]
    return list(range(world))


def maybe_set_cpu_affinity(args: argparse.Namespace, gpu_ids: list[int]) -> str | None:
    mode = getattr(args, "cpu_affinity", "auto")
    if mode in (None, "", "none"):
        return None
    cpus = gpu_local_cpu_affinity(gpu_ids) if mode == "auto" else _parse_cpu_list(mode)
    if not cpus or not hasattr(os, "sched_setaffinity"):
        return None
    os.sched_setaffinity(0, cpus)
    return _format_cpu_list(cpus)
