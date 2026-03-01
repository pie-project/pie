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
    parser.add_argument("--dummy", action="store_true", help="Use dummy mode (no GPU)")
    parser.add_argument("--timeout", type=int, default=120, help="Timeout per inferlet (seconds)")
    parser.add_argument("--verbose", action="store_true", help="Show stdout on failure")
    return parser


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def run_inferlet(
    client,
    name: str,
    extra_args: dict | None = None,
    *,
    timeout: int = 120,
) -> str:
    """Install a WASM inferlet, launch it, and collect its output.

    Returns the concatenated stdout on success.
    Raises ``RuntimeError`` on error or timeout, ``FileNotFoundError`` if the
    WASM binary or manifest is missing.
    """
    extra_args = extra_args or {}
    wasm_name = name.replace("-", "_")
    wasm_path = INFERLETS_DIR / name / "target" / "wasm32-wasip2" / "release" / f"{wasm_name}.wasm"
    manifest_path = INFERLETS_DIR / name / "Pie.toml"

    if not wasm_path.exists():
        raise FileNotFoundError(f"No WASM binary at {wasm_path}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"No Pie.toml at {manifest_path}")

    manifest = tomllib.loads(manifest_path.read_text())
    pkg_name = manifest["package"]["name"]
    version = manifest["package"]["version"]
    inferlet_id = f"{pkg_name}@{version}"

    await client.install_program(wasm_path, manifest_path)
    process = await client.launch_process(inferlet_id, input=extra_args)

    output_parts: list[str] = []
    start = time.time()
    try:
        while True:
            if time.time() - start > timeout:
                raise RuntimeError("TIMEOUT")
            event, msg = await asyncio.wait_for(process.recv(), timeout=timeout)
            if event == Event.Stdout:
                output_parts.append(msg)
            elif event == Event.Return:
                output_parts.append(msg)
                return "".join(output_parts)
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
    from pie.config import Config, ModelConfig, AuthConfig

    device = [d.strip() for d in args.device.split(",")] if "," in args.device else args.device

    print(f"Model:  {args.model}")
    print(f"Device: {device}")
    print(f"Dummy:  {args.dummy}")
    print()

    cfg = Config(
        port=0,
        auth=AuthConfig(enabled=False),
        models=[ModelConfig(
            hf_repo=args.model,
            device=[device] if isinstance(device, str) else device,
            dummy_mode=args.dummy,
        )],
    )
    async with Server(cfg) as server:
        client = await server.connect()
        results: list[tuple[str, str, str]] = []

        for test_fn in tests:
            name = test_fn.__name__.removeprefix("test_").replace("_", "-")
            print(f"🔄 {name:30s} ", end="", flush=True)
            start = time.time()

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
