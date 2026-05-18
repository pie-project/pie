#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent


def run_cmd(
    cmd: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None,
    log_path: Path,
    timeout_s: float | None,
) -> dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = dt.datetime.now(dt.UTC)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(cmd) + "\n\n")
        log.flush()
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout_s,
        )
    ended = dt.datetime.now(dt.UTC)
    return {
        "cmd": cmd,
        "log": str(log_path),
        "returncode": proc.returncode,
        "started_at": started.isoformat(),
        "ended_at": ended.isoformat(),
        "wall_s": (ended - started).total_seconds(),
    }


def bench_env(
    base: dict[str, str],
    *,
    plan_dump: Path | None = None,
    pie_driver: str = "cuda_native",
    loader_planner: str = "rust",
) -> dict[str, str]:
    env = base.copy()
    paths = [
        str(ROOT / "sdk" / "python-server" / "python"),
        str(ROOT / "client" / "python" / "src"),
    ]
    if env.get("PYTHONPATH"):
        paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(paths)
    if loader_planner != "cpp":
        if pie_driver == "portable":
            env["PIE_PORTABLE_LOADER_PLANNER"] = loader_planner
            if plan_dump is not None:
                env["PIE_PORTABLE_RUST_LAYOUT_PLAN_DUMP"] = str(plan_dump)
        else:
            env["PIE_CUDA_LOADER_PLANNER"] = loader_planner
            if plan_dump is not None:
                env["PIE_CUDA_RUST_LAYOUT_PLAN_DUMP"] = str(plan_dump)
    elif plan_dump is not None and pie_driver == "cuda_native":
        env["PIE_CUDA_LAYOUT_PLAN_DUMP"] = str(plan_dump)
    return env


def parse_pie_loader_log(log_path: Path) -> dict[str, Any]:
    if not log_path.exists():
        return {}
    text = log_path.read_text(encoding="utf-8", errors="replace")
    out: dict[str, Any] = {}
    loaded = re.search(
        r"loaded\s+\d+\s+tensors\s+\((\d+)\s+MiB.*?\)\s+in\s+(\d+)\s+ms",
        text,
    )
    if loaded:
        out["load_weight_mib"] = int(loaded.group(1))
        out["load_time_ms"] = int(loaded.group(2))
    memory = re.search(
        r"load memory high-water: planned_peak~(\d+) MiB, "
        r"planned_temp<=(\d+) MiB, actual_cuda_delta~(\d+) MiB, "
        r"free (\d+) -> min (\d+) -> (\d+) MiB across (\d+) samples",
        text,
    )
    if memory:
        out.update(
            {
                "planned_peak_mib": int(memory.group(1)),
                "planned_temp_mib": int(memory.group(2)),
                "actual_cuda_delta_mib": int(memory.group(3)),
                "cuda_free_before_mib": int(memory.group(4)),
                "cuda_min_free_mib": int(memory.group(5)),
                "cuda_free_after_mib": int(memory.group(6)),
                "cuda_memory_samples": int(memory.group(7)),
            }
        )
    storage = re.search(r"storage compiler: (.*)", text)
    if storage:
        out["storage_summary"] = storage.group(1).strip()
    source = re.search(r"checkpoint byte source: (\w+) \(policy=(\w+)\)", text)
    if source:
        out["checkpoint_byte_source"] = source.group(1)
        out["checkpoint_io_policy"] = source.group(2)
    return out


def load_bench_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_plan_dump(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if "storage" not in data and "instruction_count" in data:
        instr_kinds = data.get("instruction_kinds", {})
        tile_kinds = data.get("tile_map_kinds", {})
        memory = dict(data.get("memory", {}))
        memory.setdefault("extent_write_count", instr_kinds.get("ExtentWrite"))
        memory.setdefault("tile_map_count", instr_kinds.get("TileMap"))
        return {
            "layout_summary": data.get("summary"),
            "storage_summary": data.get("summary"),
            "algebra_expr_count": None,
            "algebra_binding_count": data.get("cpp_tensor_count"),
            "algebra_expr_kinds": {},
            "storage_instr_count": data.get("instruction_count"),
            "storage_instr_kinds": instr_kinds,
            "storage_transform_kinds": tile_kinds,
            "storage_memory": memory,
            "optimizer": data.get("optimizer", {}),
        }
    algebra = data.get("algebra", {})
    storage = data.get("storage", {})
    expr_kinds: dict[str, int] = {}
    for expr in algebra.get("exprs", []):
        kind = str(expr.get("kind", ""))
        expr_kinds[kind] = expr_kinds.get(kind, 0) + 1
    instr_kinds: dict[str, int] = {}
    transform_kinds: dict[str, int] = {}
    for instr in storage.get("schedule", []):
        kind = str(instr.get("kind", ""))
        transform = str(instr.get("transform_kind", ""))
        instr_kinds[kind] = instr_kinds.get(kind, 0) + 1
        if transform and transform != "None":
            transform_kinds[transform] = transform_kinds.get(transform, 0) + 1
    return {
        "layout_summary": data.get("summary"),
        "storage_summary": storage.get("summary"),
        "algebra_expr_count": len(algebra.get("exprs", [])),
        "algebra_binding_count": len(algebra.get("bindings", [])),
        "algebra_expr_kinds": expr_kinds,
        "storage_instr_count": len(storage.get("schedule", [])),
        "storage_instr_kinds": instr_kinds,
        "storage_transform_kinds": transform_kinds,
        "storage_memory": storage.get("memory", {}),
        "optimizer": data.get("optimizer", storage.get("optimizer", {})),
    }


def bench_cmd(args: argparse.Namespace, engine: str, mode: str, model: str, json_out: Path) -> list[str]:
    common = [
        "--model",
        model,
        "--max-tokens",
        str(args.max_tokens),
        "--warmup",
        str(args.warmup),
        "--tp-size",
        str(args.tp_size),
        "--gpu-mem-util",
        str(args.gpu_mem_util),
        "--max-model-len",
        str(args.max_model_len),
        "--json-out",
        str(json_out),
    ]
    if mode == "latency":
        mode_args = ["latency", "--requests", str(args.requests)]
    else:
        mode_args = [
            "tput",
            "--num-requests",
            str(args.num_requests),
            "--concurrency",
            str(args.concurrency),
        ]

    if engine == "pie":
        cmd = [
            sys.executable,
            str(ROOT / "benches" / "pie_bench.py"),
            *mode_args,
            *common,
            "--driver",
            args.pie_driver,
            "--device",
            args.device,
            "--pie-bin",
            args.pie_bin,
            "--server-startup-timeout",
            str(args.server_startup_timeout),
        ]
        if args.checkpoint_io and args.pie_driver == "cuda_native":
            cmd += ["--checkpoint-io", args.checkpoint_io]
        if args.mxfp4_moe and args.pie_driver == "cuda_native":
            cmd += ["--mxfp4-moe", args.mxfp4_moe]
        return cmd
    if engine == "vllm":
        cmd = [
            sys.executable,
            str(ROOT / "benches" / "vllm_bench.py"),
            *mode_args,
            *common,
        ]
        if args.vllm_enforce_eager:
            cmd.append("--enforce-eager")
        return cmd
    if engine == "sglang":
        cmd = [
            sys.executable,
            str(ROOT / "benches" / "sglang_bench.py"),
            *mode_args,
            *common,
        ]
        if args.sglang_disable_cuda_graph:
            cmd.append("--sglang-disable-cuda-graph")
        if args.sglang_disable_piecewise_cuda_graph:
            cmd.append("--sglang-disable-piecewise-cuda-graph")
        if args.sglang_attention_backend:
            cmd += ["--sglang-attention-backend", args.sglang_attention_backend]
        if args.sglang_sampling_backend:
            cmd += ["--sglang-sampling-backend", args.sglang_sampling_backend]
        return cmd
    raise ValueError(f"unknown engine: {engine}")


def write_markdown(path: Path, evidence: dict[str, Any]) -> None:
    lines = [
        "# Loader Evidence",
        "",
        f"Generated: `{evidence['generated_at']}`",
        "",
        "## Tests",
        "",
    ]
    for test in evidence["tests"]:
        status = "PASS" if test["returncode"] == 0 else "FAIL"
        lines.append(f"- `{status}` `{Path(test['log']).name}` in {test['wall_s']:.1f}s")
    lines += ["", "## Benchmarks", ""]
    for run in evidence["benchmarks"]:
        status = "PASS" if run["returncode"] == 0 else "FAIL"
        summary = run.get("bench_json", {}).get("summary", {})
        metric = ""
        if summary:
            metric = (
                f": {summary.get('output_tok_per_s', 0.0):.2f} tok/s, "
                f"p50={summary.get('latency_p50_ms')} ms"
            )
        loader = run.get("pie_loader", {})
        if loader:
            metric += (
                f", load={loader.get('load_time_ms')} ms, "
                f"actual_cuda_delta={loader.get('actual_cuda_delta_mib')} MiB"
            )
        plan = run.get("plan_dump_summary", {})
        if plan:
            memory = plan.get("storage_memory", {})
            metric += (
                f", instr={plan.get('storage_instr_count')}, "
                f"extent_writes={memory.get('extent_write_count')}, "
                f"tile_maps={memory.get('tile_map_count')}"
            )
        lines.append(
            f"- `{status}` `{run['engine']}` `{run['mode']}` `{run['model']}`{metric}"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regenerate loader tests and Pie/vLLM/SGLang evidence."
    )
    parser.add_argument("--model", action="append", default=None)
    parser.add_argument("--engines", default="pie,vllm,sglang")
    parser.add_argument("--modes", default="latency,tput")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--skip-benches", action="store_true")
    parser.add_argument(
        "--cuda-build-dir",
        default=os.environ.get("PIE_CUDA_BUILD_DIR"),
        help="CMake build directory containing CUDA loader tests.",
    )
    parser.add_argument("--timeout-s", type=float, default=3600.0)
    parser.add_argument("--requests", type=int, default=8)
    parser.add_argument("--num-requests", type=int, default=64)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-mem-util", type=float, default=0.80)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--checkpoint-io", choices=["auto", "mmap", "gds"], default="auto")
    parser.add_argument(
        "--pie-driver",
        choices=["cuda_native", "portable"],
        default="cuda_native",
        help="Pie backend used for Pie evidence runs.",
    )
    parser.add_argument(
        "--loader-planner",
        choices=["cpp", "rust", "dual"],
        default="rust",
        help="Set PIE_*_LOADER_PLANNER for Pie evidence runs.",
    )
    parser.add_argument(
        "--mxfp4-moe",
        choices=["auto", "routed_dequant", "packed", "bf16", "dequant", "eager_bf16", "native"],
        default="auto",
    )
    parser.add_argument("--pie-bin", default=str(ROOT / "target" / "release" / "pie"))
    parser.add_argument("--server-startup-timeout", type=float, default=600.0)
    parser.add_argument("--vllm-enforce-eager", action="store_true")
    parser.add_argument(
        "--sglang-disable-cuda-graph",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--sglang-disable-piecewise-cuda-graph",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--sglang-attention-backend", default="triton")
    parser.add_argument("--sglang-sampling-backend", default="pytorch")
    parser.add_argument(
        "--flashinfer-cuda-arch-list",
        default=os.environ.get("FLASHINFER_CUDA_ARCH_LIST", "12.0a"),
        help="Compatibility override for FlashInfer JIT on Blackwell/CUDA 12.8 hosts.",
    )
    args = parser.parse_args()

    models = args.model or ["Qwen/Qwen3-32B"]
    engines = [e.strip() for e in args.engines.split(",") if e.strip()]
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.output_dir) if args.output_dir else ROOT / ".tmp" / "loader_evidence" / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    cuda_build_dir = (
        Path(args.cuda_build_dir)
        if args.cuda_build_dir
        else (
            ROOT / "driver" / "cuda" / "build"
            if (ROOT / "driver" / "cuda" / "build").exists()
            else Path("/tmp/pie-cuda-loader-build")
        )
    )

    evidence: dict[str, Any] = {
        "generated_at": dt.datetime.now(dt.UTC).isoformat(),
        "root": str(ROOT),
        "cuda_build_dir": str(cuda_build_dir),
        "models": models,
        "tests": [],
        "benchmarks": [],
    }

    test_cmd = [
        "ctest",
        "--test-dir",
        str(cuda_build_dir),
        "-R",
        "layout_plan|loader_golden",
        "--output-on-failure",
    ]
    evidence["tests"].append(
        run_cmd(
            test_cmd,
            cwd=ROOT,
            env=os.environ.copy(),
            log_path=out_dir / "tests.log",
            timeout_s=args.timeout_s,
        )
    )

    if not args.skip_benches:
        for model in models:
            model_tag = re.sub(r"[^A-Za-z0-9_.-]+", "_", model)
            for engine in engines:
                for mode in modes:
                    json_out = out_dir / f"{engine}-{mode}-{model_tag}.json"
                    log_out = out_dir / f"{engine}-{mode}-{model_tag}.log"
                    plan_dump = out_dir / f"{engine}-{mode}-{model_tag}.plan.json"
                    env = bench_env(
                        os.environ,
                        plan_dump=plan_dump if engine == "pie" else None,
                        pie_driver=args.pie_driver,
                        loader_planner=args.loader_planner,
                    )
                    pie_server_log = out_dir / f"{engine}-{mode}-{model_tag}.server.log"
                    if engine == "pie":
                        env["PIE_BENCH_SERVER_LOG"] = str(pie_server_log)
                    if engine == "sglang":
                        env.setdefault(
                            "FLASHINFER_CUDA_ARCH_LIST",
                            args.flashinfer_cuda_arch_list,
                        )
                        env.setdefault("SGLANG_DISABLE_PDL", "1")
                    cmd = bench_cmd(args, engine, mode, model, json_out)
                    run = run_cmd(
                        cmd,
                        cwd=ROOT,
                        env=env,
                        log_path=log_out,
                        timeout_s=args.timeout_s,
                    )
                    run.update(
                        {
                            "engine": engine,
                            "mode": mode,
                            "model": model,
                            "json": str(json_out),
                            "bench_json": load_bench_json(json_out),
                        }
                    )
                    if engine == "pie":
                        run["plan_dump"] = str(plan_dump)
                        run["plan_dump_summary"] = parse_plan_dump(plan_dump)
                        run["server_log"] = str(pie_server_log)
                        run["pie_loader"] = parse_pie_loader_log(pie_server_log)
                    evidence["benchmarks"].append(run)

    evidence_path = out_dir / "evidence.json"
    evidence_path.write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    write_markdown(out_dir / "README.md", evidence)
    print(f"wrote {evidence_path}")
    failed = [
        item for item in [*evidence["tests"], *evidence["benchmarks"]]
        if item["returncode"] != 0
    ]
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
