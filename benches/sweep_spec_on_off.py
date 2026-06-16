#!/usr/bin/env python3
"""A/B sweep: speculation ON (speculation_depth=N) vs OFF
(speculation_depth=0).

Each scenario starts its own pie server. Results are appended to a CSV
so a partial run is still useful if interrupted.
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Scenario:
    label: str
    wasm_delay_us: int
    spec: str  # "off" or "on"
    depth: int
    num_requests: int
    max_tokens: int
    warmup: int


SCENARIOS: list[Scenario] = [
    # W=0: saturated GPU, increasing request count.
    Scenario("n1_W0",       0, "off", 1,   1, 256,  0),
    Scenario("n1_W0",       0, "on",  1,   1, 256,  0),
    Scenario("n4_W0",       0, "off", 1,   4, 256,  4),
    Scenario("n4_W0",       0, "on",  1,   4, 256,  4),
    Scenario("n16_W0",      0, "off", 1,  16, 256,  8),
    Scenario("n16_W0",      0, "on",  1,  16, 256,  8),
    Scenario("n64_W0",      0, "off", 1,  64, 256, 16),
    Scenario("n64_W0",      0, "on",  1,  64, 256, 16),
    Scenario("n64_W0",      0, "on",  4,  64, 256, 16),
    Scenario("n128_W0",     0, "off", 1, 128, 256, 16),
    Scenario("n128_W0",     0, "on",  1, 128, 256, 16),
    Scenario("n128_W0",     0, "on",  4, 128, 256, 16),
    # W=10ms: idle GPU between calls, deep chains shine
    Scenario("n1_W10",  10000, "off",  1, 1, 64, 0),
    Scenario("n1_W10",  10000, "on",   1, 1, 64, 0),
    Scenario("n1_W10",  10000, "on",   4, 1, 64, 0),
    Scenario("n1_W10",  10000, "on",  16, 1, 64, 0),
    Scenario("n4_W10",  10000, "off",  1, 4, 64, 4),
    Scenario("n4_W10",  10000, "on",   1, 4, 64, 4),
    Scenario("n4_W10",  10000, "on",   4, 4, 64, 4),
]


def run_scenario(s: Scenario, model: str, iter_idx: int, args: argparse.Namespace) -> dict:
    """Run pie_bench tput for one scenario; return a parsed summary dict."""
    json_out = Path(args.json_dir) / f"{s.label}_{s.spec}_d{s.depth}_i{iter_idx}.json"
    json_out.parent.mkdir(parents=True, exist_ok=True)
    # spec=off → depth=0 (kill switch lives entirely in
    # scheduler.speculation_depth now); spec=on → depth=s.depth.
    cfg_depth = 0 if s.spec == "off" else s.depth
    cmd = [
        "uv", "--project", str(ROOT / "sdk" / "python-server"), "run",
        "python", str(ROOT / "benches" / "pie_bench.py"), "tput",
        "--driver", "cuda_native",
        "--model", model,
        "--device", args.device,
        "--num-requests", str(s.num_requests),
        "--max-tokens", str(s.max_tokens),
        "--warmup", str(s.warmup),
        "--wasm-delay-us", str(s.wasm_delay_us),
        "--speculation-depth", str(cfg_depth),
        "--pie-bin", str(ROOT / "target" / "release" / "pie"),
        "--json-out", str(json_out),
    ]

    print(f"[{s.label}/{s.spec}/d{s.depth}/i{iter_idx}] $ {shlex.join(cmd)}", flush=True)
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    dt = time.perf_counter() - t0
    if proc.returncode != 0:
        print(f"  FAIL rc={proc.returncode} wall={dt:.1f}s", flush=True)
        print("  stderr tail:\n" + "\n".join(proc.stderr.splitlines()[-30:]), flush=True)
        print("  stdout tail:\n" + "\n".join(proc.stdout.splitlines()[-30:]), flush=True)
        return {"status": "fail", "elapsed_s": dt}

    try:
        data = json.loads(json_out.read_text())
        summary = data["summary"]
    except Exception as e:
        print(f"  PARSE-FAIL: {type(e).__name__}: {e}", flush=True)
        return {"status": "fail", "elapsed_s": dt}

    cfg = summary.get("config", {})
    bypass = int(cfg.get("bypass hits", 0) or 0)
    wall = float(summary["wall_s"])
    tps = float(summary["output_tok_per_s"])
    print(f"  ok wall={wall:.3f}s tok/s={tps:.1f} bypass_hits={bypass} (wallclock {dt:.1f}s)", flush=True)
    return {
        "status": "ok",
        "wall_s": wall,
        "tok_per_s": tps,
        "mean_lat_ms": float(summary.get("latency_mean_ms") or 0),
        "p99_lat_ms": float(summary.get("latency_p99_ms") or 0),
        "bypass_hits": bypass,
        "elapsed_s": dt,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="A/B sweep: PIE speculation on vs off")
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out-csv", default=str(ROOT / "benches" / "spec_on_off_sweep.csv"))
    p.add_argument("--json-dir", default="/tmp/benches-spec-on-off")
    p.add_argument("--iters", type=int, default=1)
    p.add_argument("--filter", help="run only scenarios whose label contains this", default=None)
    args = p.parse_args()

    out_csv = Path(args.out_csv)
    write_header = not out_csv.exists()
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    selected = [s for s in SCENARIOS if not args.filter or args.filter in s.label]
    total = len(selected) * args.iters
    print(f"running {total} bench(es) → {out_csv}", flush=True)

    with out_csv.open("a") as f:
        if write_header:
            f.write("label,num_requests,wasm_delay_us,spec,depth,iter,wall_s,tok_per_s,mean_lat_ms,p99_lat_ms,bypass_hits,status\n")
            f.flush()
        for s in selected:
            for it in range(1, args.iters + 1):
                r = run_scenario(s, args.model, it, args)
                f.write(
                    f"{s.label},{s.num_requests},{s.wasm_delay_us},{s.spec},{s.depth},{it},"
                    f"{r.get('wall_s', '')},{r.get('tok_per_s', '')},"
                    f"{r.get('mean_lat_ms', '')},{r.get('p99_lat_ms', '')},"
                    f"{r.get('bypass_hits', '')},{r['status']}\n"
                )
                f.flush()

    print(f"done; results in {out_csv}", flush=True)


if __name__ == "__main__":
    main()
