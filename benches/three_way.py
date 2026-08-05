#!/usr/bin/env python3
"""Run pie, mlx-lm and llama.cpp over the same shapes and write one table.

Three harnesses already exist and already agree on a client (`common.py`), so
this adds no measurement of its own: it invokes them, collects the JSON each
writes, and reports the fields that decide whether a comparison is honest —
not just tokens per second, but how many requests completed, how much prompt
each engine was actually handed, and how many output tokens it actually
produced.

Those last two are why this exists. The three engines do NOT receive identical
work by default, and the differences run in opposite directions:

* pie's inferlet applies its own chat template, so at the same `--prompt` it
  sees ~58 prompt tokens where the other two see ~34. It does more prefill per
  request, which makes any lead it shows an understatement.
* mlx-lm's server has no `ignore_eos`, so nothing can pin output length. Every
  run here is `--no-ignore-eos` for all three, and the per-engine output-token
  counts are printed so a run where they diverged is visible rather than
  averaged away.
* llama.cpp reads a GGUF, which is not the MLX checkpoint. `Q4_0` is the
  closest in bits per weight, but measured on these four models it still runs
  7-16% larger than the MLX file, and decode is bandwidth-bound — so a decode
  comparison flatters whoever holds the smaller file. The size is reported so
  the reader can normalise.
* llama.cpp divides `--ctx-size` across `--parallel` slots, and its harness
  defaults to 2048. At 16 concurrent requests that is 128 tokens a slot, so a
  400-word prompt does not fit and every request is refused — a whole column of
  zeroes that looks like a crash and is a setting. `--max-model-len` is raised
  here to match the widest shape; on the decode-heavy shapes, where it fit
  either way, doing so moved llama.cpp by 3-5%.

Nothing here decides a winner. It prints what was measured and what was not
controlled; the reading is a human's.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MLX_PY = os.path.expanduser("~/.cache/pie-metal/mlxenv/bin/python")
PIE_PY = "/tmp/pievenv/bin/python"


def run_json(cmd: list[str], env: dict[str, str] | None = None) -> dict | None:
    """Run one harness and hand back the JSON it wrote, or None if it failed."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as out:
        path = out.name
    try:
        proc = subprocess.run(
            cmd + ["--json-out", path],
            cwd=ROOT,
            env={**os.environ, **(env or {})},
            capture_output=True,
            text=True,
            timeout=1800,
        )
        if not os.path.getsize(path):
            sys.stderr.write(
                f"    (no result: rc={proc.returncode})\n"
                + "".join(f"      {l}\n" for l in proc.stdout.splitlines()[-4:])
                + "".join(f"      {l}\n" for l in proc.stderr.splitlines()[-4:])
            )
            return None
        with open(path) as handle:
            return json.load(handle)
    except subprocess.TimeoutExpired:
        sys.stderr.write("    (timed out)\n")
        return None
    finally:
        os.unlink(path)


def digest(doc: dict) -> dict:
    """The fields a comparison stands or falls on, not just the headline."""
    s = doc["summary"]
    ok = [r for r in doc["requests"] if r["ok"]]
    out = [r["output_tokens"] for r in ok] or [0]
    prompt = [r.get("prompt_tokens") or 0 for r in ok] or [0]
    return {
        "tok_s": s["output_tok_per_s"],
        "completed": s["completed"],
        "failed": s["failed"],
        "wall_s": s["wall_s"],
        "lat_ms": s.get("latency_mean_ms"),
        "out_mean": sum(out) / len(out),
        "out_min": min(out),
        "out_max": max(out),
        "prompt_mean": sum(prompt) / len(prompt),
    }


def shape_args(shape: str, n: int, conc: int, max_tokens: int) -> list[str]:
    if shape == "latency":
        return ["latency", "--requests", str(n), "--max-tokens", str(max_tokens)]
    return [
        "tput",
        "--num-requests", str(n),
        "--concurrency", str(conc),
        "--max-tokens", str(max_tokens),
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mlx-model", required=True, help="MLX checkpoint dir (pie and mlx-lm)")
    ap.add_argument("--gguf", required=True, help="GGUF file (llama.cpp)")
    ap.add_argument("--label", required=True)
    ap.add_argument("--inferlet-dir", default=str(ROOT.parent / "tests/inferlets/text-completion-bench"))
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--prompt-words", type=int, default=0,
                    help="Prepend this many words so the per-engine template "
                         "difference is a small share of the prompt.")
    ap.add_argument("--shapes", default="latency:8:1:64,tput:32:8:64,tput:32:32:64",
                    help="Comma list of shape:n:concurrency:max_tokens.")
    ap.add_argument("--engines", default="pie,mlx,llamacpp")
    ap.add_argument("--max-model-len", type=int, default=16384,
                    help="llama.cpp splits this across its parallel slots, so it "
                         "has to cover the widest prompt times the concurrency.")
    ap.add_argument("--out", default=None)
    ap.add_argument("--extra", default="",
                    help="Space-separated flags forwarded to all three harnesses "
                         "verbatim -- `common.py` owns the workload shapes "
                         "(--mixed-phase, --shared-prefix-words, --arrival-rate, "
                         "--output-spread), so they are passed rather than "
                         "reimplemented.")
    args = ap.parse_args()

    prompt = ["--prompt", " ".join(["the quick brown fox jumps over the lazy dog"] *
                                   max(1, args.prompt_words // 9))] if args.prompt_words else []
    common = ["--model", args.mlx_model, "--no-ignore-eos", "--warmup", str(args.warmup)]
    common += args.extra.split() if args.extra else []
    gguf_size = os.path.getsize(args.gguf) / (1 << 30)

    rows = []
    for spec in args.shapes.split(","):
        shape, n, conc, max_tokens = spec.split(":")
        base = shape_args(shape, int(n), int(conc), int(max_tokens)) + common + prompt
        for engine in args.engines.split(","):
            for rep in range(args.repeats):
                print(f"  {args.label} {spec} {engine} rep{rep}", flush=True)
                if engine == "pie":
                    cmd = [PIE_PY, "pie_bench.py"] + base + [
                        "--driver", "metal", "--inferlet-dir", args.inferlet_dir]
                    env = {"PYTHONPATH": f"{ROOT.parent}/client/python/src:"
                                         f"{ROOT.parent}/sdk/python-server/python"}
                elif engine == "mlx":
                    cmd, env = [MLX_PY, "mlx_bench.py"] + base, None
                else:
                    cmd = [MLX_PY, "llamacpp_bench.py"] + base + [
                        "--server-bin", shutil.which("llama-server") or "llama-server",
                        "--gguf-model", args.gguf, "--port", "8099",
                        "--max-model-len", str(args.max_model_len)]
                    env = None
                doc = run_json(cmd, env)
                if doc is None:
                    continue
                rows.append({"model": args.label, "shape": spec, "engine": engine,
                             "rep": rep, "gguf_gib": round(gguf_size, 2), **digest(doc)})

    if args.out:
        with open(args.out, "a") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
    print(f"{'shape':<20}{'engine':<10}{'tok/s':>9}{'ok':>7}{'out':>7}{'prompt':>8}{'wall':>8}")
    for row in rows:
        print(f"{row['shape']:<20}{row['engine']:<10}{row['tok_s']:>9.1f}"
              f"{row['completed']:>4}/{row['completed'] + row['failed']:<2}"
              f"{row['out_mean']:>7.0f}{row['prompt_mean']:>8.0f}{row['wall_s']:>8.2f}")


if __name__ == "__main__":
    main()
