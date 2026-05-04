"""Shared test infrastructure for per-inferlet E2E tests.

Provides:
  - `run_inferlet()` to install + launch + collect output from an inferlet.
  - `run_tests()` entrypoint that spins up a Pie server once and runs caller-
    supplied test coroutines against it.
  - Standard CLI options (--model, --device, --dummy, --timeout, --verbose).

Each ``test_<name>.py`` file defines one or more async test functions and a
``tests()`` list, then calls ``run_tests(tests())`` from its ``__main__`` block.

Usage from project root::

    uv run python tests/inferlets/test_watermarking.py --dummy
    uv run python tests/inferlets/test_watermarking.py --model Qwen/Qwen3-0.6B
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
import tomllib
from pathlib import Path
from typing import Callable, Coroutine

from pie_client import Event

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent.parent
INFERLETS_DIR = ROOT / "inferlets"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def make_parser(description: str = "Inferlet E2E Test") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.addoption = parser.add_argument  # convenience alias
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B", help="HuggingFace model ID")
    parser.add_argument("--device", default="cuda:0", help="Device(s), comma-separated")
    parser.add_argument("--timeout", type=int, default=120, help="Timeout per inferlet (seconds)")
    parser.add_argument("--verbose", action="store_true", help="Show stdout on failure")
    parser.add_argument("--driver", default="dev", choices=["dev", "vllm", "sglang", "dummy", "cuda_native", "portable"],
                        help="Inference driver: 'dev' (pie_driver_dev), 'vllm' (pie_driver_vllm), 'sglang' (pie_driver_sglang), 'dummy' (pie_driver_dummy), 'cuda_native' (pie_driver_cuda_native), 'portable' (pie_driver_portable)")
    parser.add_argument("--vllm-attention-backend", default=None,
                        help="vLLM attention backend (FLASH_ATTN / FLASHINFER / TRITON_ATTN / FLEX_ATTENTION). Default: vllm auto-picks")
    parser.add_argument("--sglang-attention-backend", default="triton",
                        help="SGLang attention backend (triton / flashinfer / flex_attention / fa3). Default: triton (cleanest custom-mask support)")
    parser.add_argument("--cpu-mem-gb", type=int, default=0,
                        help="Pinned host KV pool size in GiB. 0 = swap disabled. "
                             "Native and sglang both honor this; vllm doesn't yet.")
    parser.add_argument("--spec-ngram", action="store_true",
                        help="Enable driver-supplied NGRAM speculative-decoding drafts "
                             "(sglang and vllm drivers).")
    parser.add_argument("--spec-num-drafts", type=int, default=4,
                        help="Number of NGRAM draft tokens proposed per iteration.")
    parser.add_argument("--output-dir", default=None,
                        help="If set, write each test's captured inferlet output to "
                             "<dir>/<test-name>.txt (one file per test, multiple "
                             "run_inferlet calls concatenated with separators).")
    parser.add_argument("--portable-n-gpu-layers", type=int, default=None,
                        help="(--driver portable only) Override n_gpu_layers; "
                             "-1 = all layers on GPU, 0 = CPU only, N = first N.")
    return parser


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Per-test scratchpad: each `run_inferlet` call appends `(inferlet, output)`.
# `_run` clears the list before each test and dumps it after, so multiple
# calls within one test land in one file in order.
_captured: list[tuple[str, str]] = []


def _dump_captured(path: Path, captured: list[tuple[str, str]]) -> None:
    n = len(captured)
    with open(path, "w") as f:
        for i, (inf, out) in enumerate(captured):
            if i > 0:
                f.write("\n")
            f.write(f"=== {inf} (call {i + 1}/{n}) ===\n")
            f.write(out)
            if not out.endswith("\n"):
                f.write("\n")


def _clear_wasmtime_cache():
    """Remove the on-disk wasmtime module cache.

    After WASM binaries are recompiled, stale cached compiled modules may
    have incompatible WIT type orderings. Clearing the cache forces
    wasmtime to recompile from the current .wasm files.
    """
    import shutil
    cache_dir = Path.home() / ".cache" / "wasmtime"
    if cache_dir.exists():
        shutil.rmtree(cache_dir, ignore_errors=True)


async def run_inferlet(
    client,
    name: str,
    extra_args: dict | list | None = None,
    *,
    timeout: int = 120,
) -> str:
    """Install a WASM inferlet, launch it, and collect its output.

    Returns the concatenated stdout on success.
    Raises ``RuntimeError`` on error or timeout, ``FileNotFoundError`` if the
    WASM binary or manifest is missing.
    """
    if extra_args is None:
        extra_args = []
    wasm_name = name.replace("-", "_")
    # Rust: cargo emits to target/wasm32-wasip2/{release,debug}/<name>.wasm
    # JS (bakery build) / Python (componentize-py): flat target/<name>.wasm
    inferlet_dir = INFERLETS_DIR / name
    candidates = [
        inferlet_dir / "target" / "wasm32-wasip2" / "release" / f"{wasm_name}.wasm",
        inferlet_dir / "target" / "wasm32-wasip2" / "debug" / f"{wasm_name}.wasm",
        inferlet_dir / "target" / f"{wasm_name}.wasm",
    ]
    wasm_path = next((p for p in candidates if p.exists()), None)
    manifest_path = inferlet_dir / "Pie.toml"

    if wasm_path is None:
        raise FileNotFoundError(
            f"No WASM binary for {name} (tried: {', '.join(str(p) for p in candidates)})"
        )
    if not manifest_path.exists():
        raise FileNotFoundError(f"No Pie.toml at {manifest_path}")

    manifest = tomllib.loads(manifest_path.read_text())
    pkg_name = manifest["package"]["name"]
    version = manifest["package"]["version"]
    inferlet_id = f"{pkg_name}@{version}"

    await client.install_program(wasm_path, manifest_path, force_overwrite=True)
    process = await client.launch_process(inferlet_id, input=extra_args)

    output_parts: list[str] = []
    start = time.time()
    try:
        while True:
            if time.time() - start > timeout:
                raise RuntimeError("TIMEOUT")
            event, msg = await asyncio.wait_for(process.recv(), timeout=timeout)
            if event in (Event.Stdout, Event.Message):
                # Message = session.send() from inside the inferlet (JS/Python SDKs
                # emit their output there rather than to stdout).
                output_parts.append(msg)
            elif event == Event.Return:
                output_parts.append(msg)
                output = "".join(output_parts)
                _captured.append((name, output))
                return output
            elif event == Event.Error:
                raise RuntimeError(msg)
    except asyncio.TimeoutError:
        raise RuntimeError("TIMEOUT")


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

# A test is an async callable (client, args) -> None that raises on failure.
TestFn = Callable[..., Coroutine]


async def _run(tests: list[TestFn], args: argparse.Namespace) -> int:
    from pie.server import Server
    from pie.config import (
        Config, ModelConfig, ServerConfig, AuthConfig, TelemetryConfig,
        DriverConfig,
    )

    device = [d.strip() for d in args.device.split(",")] if "," in args.device else args.device
    if isinstance(device, str):
        device = [device]

    # Clear stale wasmtime module cache to avoid linker mismatches
    # between recompiled WASM components and cached compiled modules.
    _clear_wasmtime_cache()

    print(f"Model:  {args.model}")
    print(f"Device: {device}")
    print(f"Driver: {args.driver}")
    print()

    # Build the [model.driver.options] subsection content.
    driver_subsection: dict = {}
    if args.driver == "vllm" and args.vllm_attention_backend is not None:
        driver_subsection["attention_backend"] = args.vllm_attention_backend
    if args.driver == "sglang":
        driver_subsection["attention_backend"] = args.sglang_attention_backend
    if args.cpu_mem_gb > 0 and args.driver in ("dev", "sglang", "dummy"):
        driver_subsection["cpu_mem_budget_in_gb"] = args.cpu_mem_gb
    if args.driver in ("sglang", "vllm") and args.spec_ngram:
        driver_subsection["spec_ngram_enabled"] = True
        driver_subsection["spec_ngram_num_drafts"] = args.spec_num_drafts
    if args.driver == "portable" and args.portable_n_gpu_layers is not None:
        driver_subsection["n_gpu_layers"] = args.portable_n_gpu_layers

    cfg = Config(
        server=ServerConfig(port=0),
        auth=AuthConfig(enabled=False),
        telemetry=TelemetryConfig(),
        models=[
            ModelConfig(
                name="default",
                hf_repo=args.model,
                driver=DriverConfig(
                    type=args.driver,
                    device=device,
                    options=driver_subsection,
                ),
            ),
        ],
    )
    out_dir = Path(args.output_dir) if args.output_dir else None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    async with Server(cfg) as server:
        client = await server.connect()
        results: list[tuple[str, str, str]] = []

        for test_fn in tests:
            name = test_fn.__name__.removeprefix("test_").replace("_", "-")
            print(f"🔄 {name:30s} ", end="", flush=True)
            start = time.time()
            _captured.clear()

            try:
                await test_fn(client, args)
                elapsed = time.time() - start
                print(f"✅ ({elapsed:.1f}s)")
                results.append((name, "PASS", ""))
            except FileNotFoundError as e:
                elapsed = time.time() - start
                print(f"⏭️  ({elapsed:.1f}s) SKIPPED")
                results.append((name, "SKIP", str(e)))
            except Exception as e:
                elapsed = time.time() - start
                detail = str(e)[:300]
                print(f"❌ ({elapsed:.1f}s)")
                print(f"   {detail}")
                if args.verbose and hasattr(e, "output"):
                    for line in e.output.splitlines()[:20]:
                        print(f"   | {line}")
                results.append((name, "FAIL", detail))

            if out_dir is not None and _captured:
                _dump_captured(out_dir / f"{name}.txt", _captured)

        # Summary
        print(f"\n{'─' * 70}")
        print(f"{'Inferlet':30s} {'Status':10s} {'Detail'}")
        print(f"{'─' * 70}")
        for name, status, detail in results:
            icon = {"PASS": "✅", "FAIL": "❌", "SKIP": "⏭️"}.get(status, "?")
            print(f"{name:30s} {icon} {status:6s}  {detail[:50]}")
        print(f"{'─' * 70}")

        passed = sum(1 for _, s, _ in results if s == "PASS")
        total = sum(1 for _, s, _ in results if s != "SKIP")
        print(f"\n{passed}/{total} passed")
        return 0 if passed >= total else 1


def run_tests(tests: list[TestFn], description: str = "Inferlet E2E Test") -> None:
    """Parse CLI args, start server, run tests, exit."""
    parser = make_parser(description)
    args = parser.parse_args()
    try:
        rc = asyncio.run(_run(tests, args))
    except KeyboardInterrupt:
        print("\nTests interrupted.")
        rc = 1
    sys.exit(rc)
