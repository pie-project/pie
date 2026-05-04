#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


TOPOLOGIES = [
    {"name": "dp1_tp1", "device": "cuda:0", "tp": 1, "skip": None},
    {
        "name": "dp1_tp2",
        "device": "cuda:0,cuda:1",
        "tp": 2,
        "skip": None,
    },
    {"name": "dp2_tp1", "device": "cuda:0,cuda:1", "tp": 1, "skip": None},
]


def model_info(path: Path) -> dict[str, Any]:
    cfg_path = path / "config.json"
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    weights = list(path.glob("*.safetensors")) + list(path.glob("*.bin"))
    expected = None
    index_path = path / "model.safetensors.index.json"
    if index_path.exists():
        weight_map = json.loads(index_path.read_text()).get("weight_map", {})
        expected = len(set(weight_map.values()))
    size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    complete = (
        cfg_path.exists()
        and ((path / "tokenizer.json").exists() or (path / "tokenizer.model").exists())
        and bool(weights)
        and (expected is None or len(weights) >= expected)
    )
    return {
        "name": path.name,
        "path": str(path),
        "complete": complete,
        "size_gib": size / 1024**3,
        "model_type": cfg.get("model_type"),
        "architectures": cfg.get("architectures") or [],
        "num_weight_files": len(weights),
        "expected_weight_files": expected,
    }


def load_done(path: Path) -> set[tuple[str, str]]:
    done: set[tuple[str, str]] = set()
    if not path.exists():
        return done
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if obj.get("status") in {"pass", "skip"}:
            done.add((obj.get("model", ""), obj.get("topology", "")))
    return done


def append_jsonl(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(obj, sort_keys=True) + "\n")
        f.flush()


def run_case(args: argparse.Namespace, model: dict[str, Any], topo: dict[str, Any]) -> dict[str, Any]:
    out_dir = Path(args.out_dir)
    case_id = f"{model['name']}__{topo['name']}"
    json_out = out_dir / "cases" / f"{case_id}.json"
    log_out = out_dir / "logs" / f"{case_id}.log"
    cmd = [
        "uv",
        "--project",
        str(ROOT / "sdk" / "python-server"),
        "run",
        "python",
        str(ROOT / "benches" / "pie_bench.py"),
        "latency",
        "--driver",
        "cuda_native",
        "--model",
        model["path"],
        "--device",
        topo["device"],
        "--tp-size",
        str(topo["tp"]),
        "--requests",
        str(args.requests),
        "--warmup",
        str(args.warmup),
        "--max-tokens",
        str(args.max_tokens),
        "--max-model-len",
        str(args.max_model_len),
        "--kv-pages",
        str(args.kv_pages),
        "--max-batch-size",
        str(args.max_batch_size),
        "--max-batch-tokens",
        str(args.max_batch_tokens),
        "--default-token-limit",
        str(args.default_token_limit),
        "--default-endowment-pages",
        str(args.default_endowment_pages),
        "--server-startup-timeout",
        str(args.server_startup_timeout),
        "--request-timeout",
        str(args.request_timeout),
        "--pie-bin",
        args.pie_bin,
        "--json-out",
        str(json_out),
    ]
    start = time.perf_counter()
    log_out.parent.mkdir(parents=True, exist_ok=True)
    with log_out.open("w") as log_f:
        proc = subprocess.Popen(
            cmd,
            cwd=ROOT,
            text=True,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            proc.wait(timeout=args.case_timeout)
        except subprocess.TimeoutExpired as exc:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
            log_f.flush()
            exc.output = log_out.read_text() if log_out.exists() else ""
            raise exc
    elapsed = time.perf_counter() - start
    stdout = log_out.read_text()
    result: dict[str, Any] = {
        "model": model["name"],
        "model_path": model["path"],
        "model_type": model["model_type"],
        "architectures": model["architectures"],
        "topology": topo["name"],
        "device": topo["device"],
        "tp_size": topo["tp"],
        "elapsed_s": elapsed,
        "returncode": proc.returncode,
        "json_out": str(json_out),
        "log_out": str(log_out),
    }
    if proc.returncode != 0:
        result["status"] = "fail"
        result["error"] = "\n".join(stdout.splitlines()[-80:])
        return result
    try:
        summary = json.loads(json_out.read_text())["summary"]
    except Exception as e:
        result["status"] = "fail"
        result["error"] = f"missing/unreadable summary JSON: {type(e).__name__}: {e}"
        return result
    result["summary"] = summary
    result["status"] = "pass" if summary.get("failed", 0) == 0 else "fail"
    if result["status"] == "fail":
        result["error"] = f"{summary.get('failed')} request(s) failed"
    return result


def main() -> None:
    p = argparse.ArgumentParser(description="Sweep Pie cuda_native benches smoke over local HF models.")
    p.add_argument("--model-root", default=str(ROOT / "models" / "hf"))
    p.add_argument("--out-dir", default="/tmp/benches-pie-cuda-native-sweep")
    p.add_argument("--pie-bin", default=str(ROOT / "target" / "release" / "pie"))
    p.add_argument("--requests", type=int, default=1)
    p.add_argument("--warmup", type=int, default=0)
    p.add_argument("--max-tokens", type=int, default=1)
    p.add_argument("--max-model-len", type=int, default=512)
    p.add_argument("--kv-pages", type=int, default=128)
    p.add_argument("--max-batch-size", type=int, default=8)
    p.add_argument("--max-batch-tokens", type=int, default=512)
    p.add_argument("--default-token-limit", type=int, default=64)
    p.add_argument("--default-endowment-pages", type=int, default=8)
    p.add_argument("--server-startup-timeout", type=float, default=900.0)
    p.add_argument("--request-timeout", type=float, default=300.0)
    p.add_argument("--case-timeout", type=float, default=1200.0)
    p.add_argument("--case-cooldown", type=float, default=5.0)
    p.add_argument(
        "--topology",
        action="append",
        choices=[t["name"] for t in TOPOLOGIES],
        help="Topology to run; may be repeated. Defaults to all topologies.",
    )
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    summary_path = out_dir / "summary.jsonl"
    models = [model_info(d) for d in sorted(Path(args.model_root).iterdir()) if d.is_dir()]
    selected_topologies = [
        t for t in TOPOLOGIES if args.topology is None or t["name"] in set(args.topology)
    ]
    append_jsonl(out_dir / "models.jsonl", {"models": models})
    done = load_done(summary_path) if args.resume else set()

    for model in sorted(models, key=lambda m: (not m["complete"], m["size_gib"], m["name"])):
        if not model["complete"]:
            obj = {
                "status": "skip",
                "reason": "incomplete local model directory",
                "model": model["name"],
                "model_path": model["path"],
                "topology": "all",
                "model_type": model["model_type"],
                "size_gib": model["size_gib"],
                "num_weight_files": model["num_weight_files"],
                "expected_weight_files": model["expected_weight_files"],
            }
            if (obj["model"], obj["topology"]) not in done:
                print(f"SKIP {model['name']} all incomplete", flush=True)
                append_jsonl(summary_path, obj)
            continue
        for topo in selected_topologies:
            key = (model["name"], topo["name"])
            if args.resume and key in done:
                print(f"DONE {model['name']} {topo['name']}", flush=True)
                continue
            if topo["skip"]:
                print(f"SKIP {model['name']} {topo['name']} {topo['skip']}", flush=True)
                append_jsonl(
                    summary_path,
                    {
                        "status": "skip",
                        "reason": topo["skip"],
                        "model": model["name"],
                        "model_path": model["path"],
                        "topology": topo["name"],
                        "device": topo["device"],
                        "tp_size": topo["tp"],
                        "model_type": model["model_type"],
                    },
                )
                continue
            print(
                f"RUN  {model['name']} {topo['name']} size={model['size_gib']:.2f}GiB",
                flush=True,
            )
            try:
                result = run_case(args, model, topo)
            except subprocess.TimeoutExpired as e:
                result = {
                    "status": "fail",
                    "model": model["name"],
                    "model_path": model["path"],
                    "model_type": model["model_type"],
                    "architectures": model["architectures"],
                    "topology": topo["name"],
                    "device": topo["device"],
                    "tp_size": topo["tp"],
                    "elapsed_s": args.case_timeout,
                    "error": f"case timeout after {args.case_timeout}s",
                    "partial_output": (e.stdout or "")[-8000:],
                }
            append_jsonl(summary_path, result)
            if result["status"] == "pass":
                s = result["summary"]
                print(
                    f"PASS {model['name']} {topo['name']} "
                    f"out_tok={s['output_tokens']} wall={s['wall_s']:.3f}s",
                    flush=True,
                )
            else:
                print(f"FAIL {model['name']} {topo['name']}: {result.get('error','')}", flush=True)
            if args.case_cooldown > 0:
                time.sleep(args.case_cooldown)


if __name__ == "__main__":
    main()
