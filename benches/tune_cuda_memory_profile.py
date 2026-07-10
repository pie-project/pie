#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PROFILE_PATH = Path.home() / ".cache" / "pie" / "cuda_memory_profiles.json"


def model_key(model: str, tp_size: int, device: int) -> dict[str, Any]:
    import torch
    from transformers import AutoConfig

    prop = torch.cuda.get_device_properties(device)
    cfg = AutoConfig.from_pretrained(model, trust_remote_code=False)
    hidden = int(getattr(cfg, "hidden_size"))
    num_heads = int(getattr(cfg, "num_attention_heads"))
    head_dim = int(getattr(cfg, "head_dim", hidden // num_heads))
    return {
        "gpu_name": prop.name,
        "compute_major": prop.major,
        "compute_minor": prop.minor,
        "sm_count": prop.multi_processor_count,
        "tp_size": tp_size,
        "model_type": str(getattr(cfg, "model_type")),
        "hidden_size": hidden,
        "num_hidden_layers": int(getattr(cfg, "num_hidden_layers")),
        "num_attention_heads": num_heads,
        "num_key_value_heads": int(getattr(cfg, "num_key_value_heads", num_heads)),
        "head_dim": head_dim,
    }


def write_profile(path: Path, key: dict[str, Any], plan: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"entries": [{"key": key, "plan": plan}]}, indent=2))


def run_case(args: argparse.Namespace, json_out: Path, extra_env: dict[str, str] | None = None) -> dict[str, Any]:
    cmd = [
        sys.executable,
        str(ROOT / "benches" / "pie_bench.py"),
        "tput",
        "--driver",
        "cuda_native",
        "--device",
        args.device,
        "--model",
        args.model,
        "--num-requests",
        str(args.num_requests),
        "--max-tokens",
        str(args.max_tokens),
        "--warmup",
        str(args.warmup),
        "--warmup-max-tokens",
        str(args.warmup_max_tokens),
        "--request-timeout",
        str(args.request_timeout),
        "--server-startup-timeout",
        str(args.server_startup_timeout),
        "--memory-profile",
        "auto",
        "--json-out",
        str(json_out),
    ]
    if args.tp_size != 1:
        cmd += ["--tp-size", str(args.tp_size)]
    env = os.environ.copy()
    env["PYTHONPATH"] = (
        f"{ROOT / 'sdk/python-server/python'}:"
        f"{ROOT / 'client/python/src'}:"
        f"{env.get('PYTHONPATH', '')}"
    )
    if extra_env:
        env.update(extra_env)
    proc = subprocess.run(cmd, cwd=ROOT, env=env, text=True)
    if proc.returncode != 0:
        return {"ok": False, "returncode": proc.returncode}
    data = json.loads(json_out.read_text())
    summary = data.get("summary", {})
    return {
        "ok": summary.get("failed", 1) == 0,
        "output_tok_per_s": float(summary.get("output_tok_per_s", 0.0)),
        "wall_s": float(summary.get("wall_s", 0.0)),
        "summary": summary,
    }


def candidate_plans(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.candidate:
        out = []
        for raw in args.candidate:
            profile, page, tokens, requests = raw.split(":")
            out.append({
                "policy_profile": profile,
                "kv_page_size": int(page),
                "max_forward_tokens": int(tokens),
                "max_forward_requests": int(requests),
            })
        return out

    tokens = [1024, 2048, 4096, 8192]
    requests = [128, 256, 512]
    profiles = ["balanced", "throughput", "capacity"]
    pages = [16, 32]
    out = []
    for profile in profiles:
        for page in pages:
            for n in tokens:
                for r in requests:
                    if r <= n:
                        out.append({
                            "policy_profile": profile,
                            "kv_page_size": page,
                            "max_forward_tokens": n,
                            "max_forward_requests": r,
                        })
    return out


def main() -> None:
    p = argparse.ArgumentParser(
        description="Profile CUDA memory planner candidates and install the fastest auto profile cache entry."
    )
    p.add_argument("--model", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--tp-size", type=int, default=1)
    p.add_argument("--num-requests", type=int, default=256)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--warmup-max-tokens", type=int, default=4)
    p.add_argument("--request-timeout", type=float, default=2400.0)
    p.add_argument("--server-startup-timeout", type=float, default=700.0)
    p.add_argument("--candidate", action="append", help="profile:page_size:max_tokens:max_requests")
    p.add_argument("--results-out", default=None)
    p.add_argument("--profile-path", default=str(PROFILE_PATH))
    p.add_argument("--keep-existing", action="store_true")
    args = p.parse_args()

    device_id = int(args.device.split(":", 1)[1].split(",", 1)[0]) if ":" in args.device else 0
    key = model_key(args.model, args.tp_size, device_id)
    profile_path = Path(args.profile_path)
    old_profile = profile_path.read_text() if profile_path.exists() else None

    results: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="pie-cuda-tune-") as td:
        tmp = Path(td)
        if profile_path.exists():
            profile_path.unlink()
        baseline = run_case(args, tmp / "baseline.json")
        results.append({"name": "rule", "plan": None, **baseline})

        for i, plan in enumerate(candidate_plans(args)):
            write_profile(profile_path, key, plan)
            result = run_case(args, tmp / f"candidate-{i}.json")
            results.append({"name": f"candidate-{i}", "plan": plan, **result})

    valid = [r for r in results if r.get("ok")]
    best = max(valid, key=lambda r: r.get("output_tok_per_s", 0.0)) if valid else None
    if best and best["plan"] is not None:
        write_profile(profile_path, key, best["plan"])
    elif old_profile is not None and args.keep_existing:
        profile_path.parent.mkdir(parents=True, exist_ok=True)
        profile_path.write_text(old_profile)
    elif profile_path.exists():
        profile_path.unlink()

    if args.results_out:
        Path(args.results_out).write_text(json.dumps({"key": key, "results": results, "best": best}, indent=2))
    print(json.dumps({"best": best, "profile_path": str(profile_path)}, indent=2))


if __name__ == "__main__":
    main()
