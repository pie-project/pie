#!/usr/bin/env python3
"""Pie vs vLLM multimodal (image) head-to-head runner.

Runs `pie_mm_bench.py` and `vllm_mm_bench.py` for latency and/or throughput on
the same model + image + workload, sequentially (so the two engines never
contend for the GPU), then prints a side-by-side table.

  python run_mm_compare.py \
      --model Qwen/Qwen3-VL-2B-Instruct --image assets/bench_image.png \
      --max-tokens 128 --latency-requests 16 --tput-requests 128 \
      --concurrency 32 --warmup 4 --out-dir out/mm

Pie runs under the current interpreter; vLLM runs under --vllm-python
(default /root/.venv/vllm/bin/python).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _run(python: str, script: str, mode: str, json_out: Path, extra: list[str]) -> dict | None:
    json_out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [python, str(HERE / script), mode, "--json-out", str(json_out), *extra]
    print(f"\n$ {' '.join(cmd)}", flush=True)
    rc = subprocess.run(cmd, cwd=str(HERE)).returncode
    if rc != 0:
        print(f"!! {script} {mode} exited {rc}", flush=True)
        return None
    try:
        return json.loads(json_out.read_text())["summary"]
    except Exception as e:
        print(f"!! could not read {json_out}: {e}", flush=True)
        return None


def _common(args: argparse.Namespace) -> list[str]:
    return [
        "--model", args.model,
        "--image", args.image,
        "--max-tokens", str(args.max_tokens),
        "--warmup", str(args.warmup),
        "--temperature", str(args.temperature),
        "--top-p", str(args.top_p),
        ("--ignore-eos" if args.ignore_eos else "--no-ignore-eos"),
    ]


def _fmt(x, nd=1):
    return "—" if x is None else f"{x:.{nd}f}"


def _table(rows: list[tuple[str, dict | None]]) -> None:
    print("\n" + "=" * 78)
    hdr = f"{'metric':<26}" + "".join(f"{name:>17}" for name, _ in rows)
    print(hdr)
    print("-" * 78)

    def line(label, key, nd=2, scale=1.0):
        vals = []
        for _, s in rows:
            v = (s.get(key) if s else None)
            vals.append("—" if v is None else f"{v * scale:.{nd}f}")
        print(f"{label:<26}" + "".join(f"{v:>17}" for v in vals))

    line("completed", "completed", 0)
    line("wall (s)", "wall_s", 2)
    line("prompt tokens (total)", "prompt_tokens", 0)
    line("output tokens (total)", "output_tokens", 0)
    line("requests/sec", "req_per_s", 2)
    line("output tok/sec", "output_tok_per_s", 1)
    line("latency mean (ms)", "latency_mean_ms", 1)
    line("latency p50 (ms)", "latency_p50_ms", 1)
    line("latency p95 (ms)", "latency_p95_ms", 1)
    line("latency p99 (ms)", "latency_p99_ms", 1)
    print("=" * 78)


def main() -> None:
    ap = argparse.ArgumentParser(description="Pie vs vLLM multimodal head-to-head")
    ap.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct")
    ap.add_argument("--image", default="assets/bench_image.png")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--latency-requests", type=int, default=16)
    ap.add_argument("--tput-requests", type=int, default=128)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--warmup", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--ignore-eos", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--out-dir", default="out/mm")
    ap.add_argument("--vllm-python", default="/root/.venv/vllm/bin/python")
    ap.add_argument("--pie-python", default=sys.executable)
    ap.add_argument("--skip", choices=["latency", "tput"], action="append", default=[])
    args = ap.parse_args()

    out = Path(args.out_dir)
    common = _common(args)

    if "latency" not in args.skip:
        lat = ["--requests", str(args.latency_requests)]
        # vLLM first (clean GPU), then Pie. Order doesn't matter — sequential.
        v = _run(args.vllm_python, "vllm_mm_bench.py", "latency", out / "vllm_latency.json", common + lat)
        p = _run(args.pie_python, "pie_mm_bench.py", "latency", out / "pie_latency.json", common + lat)
        print("\n### LATENCY (single-stream, concurrency=1)")
        _table([("vLLM", v), ("Pie", p)])

    if "tput" not in args.skip:
        tp = ["--num-requests", str(args.tput_requests), "--concurrency", str(args.concurrency)]
        v = _run(args.vllm_python, "vllm_mm_bench.py", "tput", out / "vllm_tput.json", common + tp)
        p = _run(args.pie_python, "pie_mm_bench.py", "tput", out / "pie_tput.json", common + tp)
        print(f"\n### THROUGHPUT (concurrency={args.concurrency}, n={args.tput_requests})")
        _table([("vLLM", v), ("Pie", p)])


if __name__ == "__main__":
    main()
