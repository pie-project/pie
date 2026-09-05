"""Shared test infrastructure for per-inferlet E2E tests.

Provides:
  - `run_inferlet()` to install + launch + collect output from an inferlet.
  - `run_tests()` entrypoint that spins up a Pie server once and runs caller-
    supplied test coroutines against it.
  - Standard CLI options (--model, --device, --engine, --timeout, --verbose).

Each ``test_<name>.py`` file defines one or more async test functions and a
``tests()`` list, then calls ``run_tests(tests())`` from its ``__main__`` block.

Usage from project root::

    uv run python tests/inferlets/test_curated.py
    uv run python tests/inferlets/test_curated.py --engine vulkan \
        --model mlx-community/Qwen3-0.6B-4bit
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
    # A store name, a HuggingFace repo id, or a path to a `.zt` artifact --
    # whatever `[model] model` accepts, since that is where this lands.
    #
    # THE DEFAULT DOES NOT BOOT THE SHADER BACKENDS. `Qwen/Qwen3-0.6B` is an
    # unquantised release, and `metal`, `vulkan` and `wgpu` all load through
    # the MLX contract, which binds every projection on its affine-U4 path and
    # refuses a checkpoint carrying no `.scales`. The refusal is clear about
    # what to do -- a pre-quantised repo (`mlx-community/*-4bit`) or a `.zt`
    # built with `--quant int4` -- but it arrives at server construction, so a
    # `--engine vulkan` run with the default stops before the first inferlet
    # rather than reporting thirty-nine failures. The CUDA engines take the
    # unquantised release as it is, which is why this default is what it is.
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B", help="HuggingFace model ID")
    # WHICH ROW OF THAT CHECKPOINT (`[model] sku`). A vision artifact fits its
    # family's text row and its own, and the load identifies the cheap one
    # first -- deliberately, because a two-unit load stands the fold down. A
    # suite that wants the tower names the row.
    parser.add_argument("--sku", default=None,
                        help="Catalog SKU to serve (default: identify one from the checkpoint)")
    parser.add_argument("--max-total-pages", type=int, default=None,
                        help="[model.engine.options] total_pages: cap the KV page pool "
                             "(a dsv4 page carries every layer's index-key and compressor "
                             "caches, so the derived pool can outrun the device)")
    parser.add_argument("--engine-option", action="append", default=[],
                        metavar="KEY=VALUE",
                        help="one `[model.engine.options]` entry, repeatable (e.g. "
                             "max_forward_tokens=64); integers and floats are typed, the "
                             "rest is text")
    parser.add_argument("--device-weight-budget", default=None,
                        help="[model] device_weight_budget, e.g. 20GiB: what a streamed-expert "
                             "artifact (dsv4) needs to load on a box that cannot hold it whole")
    parser.add_argument("--device", default=None,
                        help="Device(s), comma-separated. Default: 'metal:0' for --engine metal, "
                             "'gpu:0' for --engine wgpu or vulkan, else 'cuda:0'")
    parser.add_argument("--timeout", type=int, default=120, help="Timeout per inferlet (seconds)")
    # A SMALL number here is a stress rather than a tuning knob: `engine-vulkan`
    # and `engine-wgpu` both open their KV pool at 1024 pages, which almost no
    # curated inferlet ever fills, so the pool's growth path is barely entered
    # by a default run. Two real defects have been found in that path, both by
    # this sweep and both only because thirty-nine programs share one server.
    # `--kv-pages 8` puts every request in it.
    parser.add_argument("--kv-pages", type=int, default=None,
                        help="KV pages the backend opens with (default: the backend's own)")
    parser.add_argument("--verbose", action="store_true", help="Show stdout on failure")
    # THE FOUR THE CONFIG ACCEPTS, AND NOT ONE MORE.
    #
    # This list is `worker::config::EngineKind`, whose serde is
    # `rename_all = "snake_case"` over four variants. A name outside it does
    # not degrade or fall back -- `Server(cfg)` raises before the first
    # inferlet runs:
    #
    #     unknown variant `dev`, expected one of `cuda_native`, `metal`,
    #     `vulkan`, `wgpu`
    #
    # It offered nine for a while, five of which were the pre-rewrite Python
    # engine's out-of-process engines (`dev`, `vllm`, `sglang`,
    # `tensorrt_llm`, `dummy`). Those went with that engine, and no Python
    # engine tree is left in this repository to host them. `dev` was the
    # DEFAULT and `--dummy` the invocation this module's own docstring gave,
    # so the harness's two most-typed commands both died at the door, in a
    # message about TOML.
    #
    # `wgpu` and `vulkan` really do work: the wheel takes a feature
    # (`maturin build --no-default-features --features engine-vulkan`), and a
    # build that did not select the named backend fails at boot saying so,
    # which is a better answer than a choice list that pretends the option
    # does not exist.
    parser.add_argument("--engine", default="cuda_native",
                        choices=["cuda_native", "metal", "vulkan", "wgpu"],
                        help="Embedded engine: 'cuda_native', 'metal', 'vulkan' or 'wgpu'")
    parser.add_argument("--cpu-mem-gb", type=int, default=0,
                        help="Pinned host KV pool size in GiB. 0 = swap disabled. "
                             "Only cuda_native serves a host swap pool.")
    parser.add_argument("--spec-ngram", action="store_true",
                        help="Enable engine-supplied NGRAM speculative-decoding drafts.")
    parser.add_argument("--spec-num-drafts", type=int, default=4,
                        help="Number of NGRAM draft tokens proposed per iteration.")
    # **THE RECORDING POLICY, AS A HARNESS KNOB** (`[engine] graphs`).
    #
    # A suite whose plan bakes a CONDITIONAL region cannot be served by the
    # cuda shell's recorded path — `cudaGraphSetConditional` wants an rdc +
    # cudadevrt link stage this crate does not have — and the MTP draft head is
    # the catalog's one conditional (`model-compiler`'s
    # `which_skus_get_a_conditional`: "the MTP head and nothing else"). Eager
    # is slow and correct, so a gate about a draft head can ask for it and say
    # in its own header that it did.
    parser.add_argument("--graphs", default=None, choices=["on", "off", "shaped"],
                        help="engine graph-recording policy ([engine] graphs)")
    parser.add_argument("--output-dir", default=None,
                        help="If set, write each test's captured inferlet output to "
                             "<dir>/<test-name>.txt (one file per test, multiple "
                             "run_inferlet calls concatenated with separators).")
    # Attach to a `pie serve` that is already up instead of booting the
    # embedded engine: the model is then whatever that server was started
    # with, and every other option about the engine is ignored. What it buys
    # is iteration speed -- a boot is minutes, a test is seconds -- and a
    # harness that needs no `pie-server` wheel.
    parser.add_argument("--attach", default=None, metavar="URI",
                        help="Run against a live `pie serve` at URI (e.g. ws://127.0.0.1:8080) "
                             "instead of booting the embedded engine")
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
    engine for two sessions.

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
    # Attached: no embedded engine, no `pie-server` wheel needed.
    if args.attach:
        from pie_client import PieClient

        _build_guests()
        print(f"Server: {args.attach} (attached; its model is whatever it was started with)")
        print()
        async with PieClient(args.attach) as client:
            await client.authenticate("default")
            return await _run_tests_on(client, tests, args)

    from pie.server import Server
    from pie.config import (
        Config, ModelConfig, ServerConfig, TelemetryConfig,
        EngineConfig,
    )

    raw_device = args.device
    if raw_device is None:
        # `gpu:0` for the two portable backends: neither reads it as a
        # selector -- wgpu asks the platform for an adapter and vulkan
        # enumerates -- but `device` is required of every engine, so it has to
        # say something and `cuda:0` would be a lie about the hardware.
        raw_device = {
            "metal": "metal:0",
            "wgpu": "gpu:0",
            "vulkan": "gpu:0",
        }.get(args.engine, "cuda:0")
    device = [d.strip() for d in raw_device.split(",")] if "," in raw_device else raw_device
    if isinstance(device, str):
        device = [device]

    # Clear stale wasmtime module cache to avoid linker mismatches
    # between recompiled WASM components and cached compiled modules.
    _build_guests()
    _clear_wasmtime_cache()

    print(f"Model:  {args.model}")
    print(f"Device: {device}")
    print(f"Engine: {args.engine}")
    print()

    # Build the [model.engine.options] subsection content.
    engine_subsection: dict = {}
    if args.cpu_mem_gb > 0 and args.engine == "cuda_native":
        engine_subsection["cpu_mem_budget_in_gb"] = args.cpu_mem_gb
    if args.spec_ngram:
        engine_subsection["spec_ngram_enabled"] = True
        engine_subsection["spec_ngram_num_drafts"] = args.spec_num_drafts
    if args.graphs is not None:
        engine_subsection["graphs"] = args.graphs
    if args.max_total_pages is not None:
        engine_subsection["total_pages"] = args.max_total_pages
    for entry in args.engine_option:
        key, _, value = entry.partition("=")
        typed: object = value
        for cast in (int, float):
            try:
                typed = cast(value)
                break
            except ValueError:
                continue
        engine_subsection[key.strip()] = typed

    cfg = Config(
        server=ServerConfig(port=0),
        telemetry=TelemetryConfig(),
        model=ModelConfig(
            name="default",
            hf_repo=args.model,
            sku=args.sku,
            device_weight_budget=args.device_weight_budget,
            # `kv_pages` retired from the schema (6d3189654): the worker
            # refuses the key by name, so the flag is not forwarded.
            engine=EngineConfig(
                type=args.engine,
                device=device,
                options=engine_subsection,
            ),
        ),
    )
    async with Server(cfg) as server:
        client = await server.connect()
        return await _run_tests_on(client, tests, args)


async def _run_tests_on(client, tests: list[TestFn], args: argparse.Namespace) -> int:
    """Run `tests` against a connected client; the per-test loop and summary."""
    out_dir = Path(args.output_dir) if args.output_dir else None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
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


#: Engine capabilities that NO engine in this repository advertises.
#:
#: `EtaCaps` (`crates/runtime/src/model.rs`) declares `has_kv_envelopes`,
#: `has_attn_score` and `has_attn_page_mask`. A suite whose inferlets bind a
#: name listed here cannot pass anywhere, on any backend, on any checkpoint --
#: not a bug in the suite, but the repo-side regression floor for a feature
#: that is partly built. Running one without knowing costs a model boot and
#: produces a screen of identical bind refusals, which is exactly the shape of
#: output that teaches nothing.
#:
#: **`attn_score` CAME OFF THIS LIST** (`.wiki/alto/attn-score.md` §4, wave
#: S1). The CUDA shell carves an observability slab, the capture arm writes a
#: per-key rectangle into it as the graph runs, and the epilogue binds it --
#: so `engine-cuda`'s `profile()` answers `shell.observes_scores()` rather
#: than a literal `false`, and a program reading the intrinsic at
#: `Stage::Epilogue` binds and fires. What did NOT come off is the other two:
#: `envelope_dot` wants a second-party page-envelope kernel that no shell
#: ships (a separate design, `attn-score.md` §3's closing line), and
#: `attn_page_mask` wants a shell that CONSUMES the sink -- and the audit
#: (`.wiki/alto/attn-score.md` section 7) found nobody knocking: every
#: enforcing program here, `trackb-h2o` and `trackb-snapkv` included, goes
#: through the mask door (`KvGeometry.mask` -> `Port::AttnMask`), which both
#: shells advertise. The sink's sole caller is `quest`, behind
#: `envelope_dot`; on Metal its legal stages and the plane's one boundary
#: do not even intersect. The entry below stays because the WORD is still
#: honestly false -- not because anything stops at bind any more.
UNADVERTISED = {
    "envelope_dot": "has_kv_envelopes",
    "attn_page_mask": "has_attn_page_mask",
}


def run_tests(
    tests: list[TestFn],
    description: str = "Inferlet E2E Test",
    requires: tuple[str, ...] = (),
) -> None:
    """Parse CLI args, start server, run tests, exit.

    `requires` names the engine-advertised kernels and intrinsics this suite's
    inferlets bind. Any entry in `UNADVERTISED` makes the suite skip before the
    server starts, because no backend here can serve it.
    """
    parser = make_parser(description)
    args = parser.parse_args()
    blocked = [name for name in requires if name in UNADVERTISED]
    if blocked:
        bits = ", ".join(f"`{UNADVERTISED[name]}` (for `{name}`)" for name in blocked)
        print(f"\n=== {description}")
        print(
            f"SKIPPED: this suite binds {', '.join(f'`{n}`' for n in blocked)}, and "
            f"no engine in this repository advertises {bits}.\n"
            "Every such field is a literal `false` in engine-cuda, engine-vulkan, "
            "engine-metal and the two engine-side adapters, so the suite would boot "
            "a model and then fail every case at bind with the same message.\n"
            "See `conftest.UNADVERTISED`; when the capability lands, "
            "`crates/engine/tests/nothing_advertises_the_attention_taps.rs` "
            "fails and points back here."
        )
        sys.exit(0)
    try:
        rc = asyncio.run(_run(tests, args))
    except KeyboardInterrupt:
        print("\nTests interrupted.")
        rc = 1
    sys.exit(rc)
