"""Shared test infrastructure for per-inferlet E2E tests.

Provides:
  - `run_inferlet()` to install + launch + collect output from an inferlet.
  - `run_tests()` entrypoint that spins up a Pie server once and runs caller-
    supplied test coroutines against it.
  - Standard CLI options (--model, --device, --dummy, --timeout, --verbose).

Each ``test_<name>.py`` file defines one or more async test functions and a
``tests()`` list, then calls ``run_tests(tests())`` from its ``__main__`` block.

Usage from project root::

    uv run python tests/inferlets/test_curated.py --dummy
    uv run python tests/inferlets/test_curated.py --model Qwen/Qwen3-0.6B
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

INFERLETS_DIR = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def make_parser(description: str = "Inferlet E2E Test") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.addoption = parser.add_argument  # convenience alias
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B", help="HuggingFace model ID")
    parser.add_argument("--device", default=None,
                        help="Device(s), comma-separated. Default: 'metal:0' for --driver metal, "
                             "'gpu:0' for --driver wgpu or vulkan, else 'cuda:0'")
    parser.add_argument("--timeout", type=int, default=120, help="Timeout per inferlet (seconds)")
    # A SMALL number here is a stress rather than a tuning knob: `driver-vulkan`
    # and `driver-wgpu` both open their KV pool at 1024 pages, which almost no
    # curated inferlet ever fills, so the pool's growth path is barely entered
    # by a default run. Two real defects have been found in that path, both by
    # this sweep and both only because thirty-nine programs share one server.
    # `--kv-pages 8` puts every request in it.
    parser.add_argument("--kv-pages", type=int, default=None,
                        help="KV pages the backend opens with (default: the backend's own)")
    parser.add_argument("--verbose", action="store_true", help="Show stdout on failure")
    driver_group = parser.add_mutually_exclusive_group()
    # `wgpu` and `vulkan` are the two pure-Rust backends. They were missing
    # here for as long as the embedded wheel could not host them -- it pinned
    # `worker/driver-cuda` -- so naming one produced a server with no driver.
    # The wheel takes a feature now (`maturin build --no-default-features
    # --features driver-wgpu`), and a build that did not select the named
    # backend fails at boot saying so, which is a better answer than a choice
    # list that pretends the option does not exist.
    driver_group.add_argument("--driver", default="dev", choices=["dev", "vllm", "sglang", "tensorrt_llm", "dummy", "cuda_native", "metal", "vulkan", "wgpu"],
                              help="Inference driver: 'dev', 'vllm', 'sglang', 'tensorrt_llm', 'dummy', 'cuda_native', 'metal', 'vulkan' or 'wgpu'")
    driver_group.add_argument("--dummy", action="store_true",
                              help="Alias for --driver dummy")
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


def _build_guests():
    """Build every curated guest from source before anything is installed.

    The harness installs prebuilt `.wasm` files. Without this, a guest whose
    SOURCE was fixed is still tested in its OLD shape, and the failure is
    read as the SERVER's -- which is exactly what happened to
    `prefix-tree-kv-cache`, whose one-pipeline-per-leaf fix sat unbuilt while
    the run kept failing with `pipeline is closed` and the blame went to the
    driver for two sessions.

    Skipped when `PIE_INFERLETS_NO_BUILD` is set, for runs against artifacts
    that were built elsewhere (a cross-compiled or vendored guest).
    """
    import os
    import subprocess

    if os.environ.get("PIE_INFERLETS_NO_BUILD"):
        print("Guests: not built (PIE_INFERLETS_NO_BUILD)")
        return
    result = subprocess.run(
        ["cargo", "build", "--workspace", "--target", "wasm32-wasip2"],
        cwd=INFERLETS_DIR,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        # Loud, and NOT fatal: a workspace that cannot build here may still
        # have usable artifacts, and a stale run says more than no run.
        print("Guests: BUILD FAILED -- testing whatever artifacts exist")
        print(result.stderr.strip()[-2000:])
        return
    print("Guests: built from source")


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
    # Rust workspace artifacts live under this directory's target/. Keep the member
    # paths as fallbacks for inferlets built outside the curated workspace.
    # JS (bakery build) / Python (componentize-py): flat target/<name>.wasm
    inferlet_dir = INFERLETS_DIR / name
    candidates = [
        INFERLETS_DIR / "target" / "wasm32-wasip2" / "release" / f"{wasm_name}.wasm",
        INFERLETS_DIR / "target" / "wasm32-wasip2" / "debug" / f"{wasm_name}.wasm",
        inferlet_dir / "target" / "wasm32-wasip2" / "release" / f"{wasm_name}.wasm",
        inferlet_dir / "target" / "wasm32-wasip2" / "debug" / f"{wasm_name}.wasm",
        inferlet_dir / "target" / f"{wasm_name}.wasm",
    ]
    # The NEWEST of the ones that exist, not the first.
    #
    # The list is a preference order and was read as one, which is wrong here
    # for a reason that cost real time: `_build_guests` above builds DEBUG --
    # no `--release` -- while this list prefers release. So a release artifact
    # built once, by hand or by an older harness, shadows every rebuild this
    # harness does afterwards, silently and for good. The failure mode is a
    # source edit that appears to have no effect, which is indistinguishable
    # from a correct program, and it is how a mutation test came back "the
    # mutation survived" against a guest that had never been rebuilt.
    #
    # Newest-wins keeps every path working -- a vendored or cross-built guest
    # is still found, and a deliberate release build is still preferred right
    # after it is made -- while making the thing just built the thing that
    # runs.
    present = [p for p in candidates if p.exists()]
    wasm_path = max(present, key=lambda p: p.stat().st_mtime, default=None)
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
        Config, ModelConfig, ServerConfig, TelemetryConfig,
        DriverConfig,
    )

    raw_device = args.device
    if raw_device is None:
        # `gpu:0` for the two portable backends: neither reads it as a
        # selector -- wgpu asks the platform for an adapter and vulkan
        # enumerates -- but `device` is required of every driver, so it has to
        # say something and `cuda:0` would be a lie about the hardware.
        raw_device = {
            "metal": "metal:0",
            "wgpu": "gpu:0",
            "vulkan": "gpu:0",
        }.get(args.driver, "cuda:0")
    device = [d.strip() for d in raw_device.split(",")] if "," in raw_device else raw_device
    if isinstance(device, str):
        device = [device]

    # Clear stale wasmtime module cache to avoid linker mismatches
    # between recompiled WASM components and cached compiled modules.
    _build_guests()
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

    cfg = Config(
        server=ServerConfig(port=0),
        telemetry=TelemetryConfig(),
        model=ModelConfig(
            name="default",
            hf_repo=args.model,
            kv_pages=args.kv_pages,
            driver=DriverConfig(
                type=args.driver,
                device=device,
                options=driver_subsection,
            ),
        ),
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
    if args.dummy:
        args.driver = "dummy"
    try:
        rc = asyncio.run(_run(tests, args))
    except KeyboardInterrupt:
        print("\nTests interrupted.")
        rc = 1
    sys.exit(rc)
