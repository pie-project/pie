---
name: pie
description: How to build, run, test and debug Pie
---

# Architecture (one-screen)

Pie is **one engine library** (`server/src/lib.rs`, crate
`pie-standalone`) with **two front-ends**:

* **CLI binary** (`server/src/main.rs` → `pie`). Used by humans:
  `pie serve`, `pie run`, `pie driver dev install`, etc.
* **`pie-server` Python wheel** (`sdk/python-server/`). Used inside
  Python scripts: `async with pie.server.Server(cfg) as server:`.
  Mirrors the legacy `pie-server` API so existing tests
  (`tests/inferlets/`, `benches/`, `sdk/demo/`) keep working.

Both link the same `pie_standalone::serve::start_engine` — single
source of truth for the boot path; can never drift.

**Drivers** come in two flavors:

* **Embedded** (compiled into `pie` by Cargo feature, run on a thread):
  `driver/portable/`, `driver/cuda/`, `driver/dummy/`.
* **Subprocess** (Python wheels, run via `python -m pie_driver_<flavor>`):
  `driver/dev/`, `driver/vllm/`, `driver/sglang/`. The standalone
  supervises them as child processes (`PR_SET_PDEATHSIG` → script
  exit kills launcher → no orphans). They share the shmem fast path
  and ipc-channel cold path with embedded drivers.

`sdk/python-rpc/` — `pie-rpc` wheel. Tiny pyo3 wheel that exposes
`RpcServer` to the Python driver launchers (cold-path channel back to
the Rust runtime).

User-facing TOML discriminator: `[model.driver].type ∈ {portable,
cuda_native, dummy, dev, vllm, sglang}` — the dispatch happens in
`server/src/serve/topology.rs::resolve_flavor`.

# Setup

## Just the binary (covers portable / cuda_native / dummy)

```bash
git clone https://github.com/pie-project/pie.git && cd pie
cargo install --path server                                   # portable + dummy (default features)
cargo install --path server --features driver-cuda            # add embedded CUDA driver
# or `cargo build` for in-tree iteration: ./target/debug/pie ...
```

## Add a Python driver venv (only if using dev / vllm / sglang)

```bash
pie driver dev install ~/.pie/venvs/dev          # prints the uv recipe
pie driver dev install ~/.pie/venvs/dev --run    # actually executes it
pie driver dev set venv ~/.pie/venvs/dev         # persist as default
```

`pie` resolves which Python to invoke via this chain (highest wins):
`[model.driver.options].venv|python` → `$PIE_PYTHON` →
`$VIRTUAL_ENV/bin/python` → `~/.pie/drivers.toml [driver.<type>]` →
`[python]` → `which python3`. `pie driver <type> show` prints the
resolved path and which step matched.

# Iteration loops

| Edit | Rebuild |
|------|---------|
| `runtime/`, `server/`, embedded drivers | `cargo build -p pie-standalone` |
| `driver/<py-flavor>/src/...` (Python) | none — editable install picks it up |
| `sdk/python-rpc/src/lib.rs` (pyo3) | `uv pip install --python <venv>/bin/python -e sdk/python-rpc/` |
| `sdk/python-server/src/lib.rs` (pyo3) or `python/pie/*.py` | `uv pip install --python <venv>/bin/python -e sdk/python-server/` |

WIT files: `sdk/rust/inferlet/wit/` (SDK side) and `runtime/wit/`
(runtime side) must stay in sync.

# Running

## Via the CLI

```bash
pie config init                        # first-time: writes ~/.pie/config.toml
pie doctor                             # platform + GPU + driver readiness
pie serve --config ~/.pie/config.toml  # long-running engine
pie run text-completion -- --prompt "Hello"   # one-shot, spawns its own engine
```

## Embedded inside a Python script

```python
import asyncio
from pie.server import Server
from pie.config import Config, ServerConfig, AuthConfig, ModelConfig, DriverConfig

cfg = Config(
    server=ServerConfig(port=0),
    auth=AuthConfig(enabled=False),
    models=[ModelConfig(
        name="default", hf_repo="Qwen/Qwen3-0.6B",
        driver=DriverConfig(type="dummy", device=["cpu"]),
    )],
)

async def main():
    async with Server(cfg) as server:
        client = await server.connect()
        # ... use the PieClient ...

asyncio.run(main())
```

`tests/inferlets/test_*.py --driver dummy` and `benches/tput.py --driver
dummy` both use this surface; same `pie.server.Server` + `pie.config.*`
shape the legacy wheel had.

# Building Inferlets (WASM components)

```bash
cd inferlets/text-completion
cargo build --release --target wasm32-wasip2
bakery inferlet publish                # publish to registry
```

# Testing

```bash
cargo test -p pie-standalone --features driver-dummy    # engine lib (69 tests)
cargo test -p pie                                        # runtime
cargo test -p pie-rpc                                    # python-rpc smoke
cargo build -p pie-server-py                             # python-server wheel build

# Python-side e2e via the embed flow (no GPU needed):
~/.pie/venvs/<v>/bin/python tests/inferlets/test_text_completion.py --driver dummy
~/.pie/venvs/<v>/bin/python benches/tput.py --driver dummy --num-requests 32 \
    --max-tokens 32 --default-token-limit 256

cd sdk/rust/inferlet         && cargo check --target wasm32-wasip2
cd inferlets/text-completion && cargo check --target wasm32-wasip2
```

# Diagnostics

| Command | Use |
|---------|-----|
| `pie doctor` | One-screen environment readiness (platform, GPUs, compiled-in flavors, venv resolution) |
| `pie driver list` | All known driver types + which ones the binary has compiled in |
| `pie driver <type> doctor` | Resolve venv + run import probe (subprocess); GPU/feature probe (embedded) |
| `pie driver <type> show` | Print resolved python path + precedence step |
| `pie driver <type> exec -- <cmd>` | Run a command under the resolved venv (`pip list`, etc.) |
| `pie check <toml>` | Validate a config TOML without booting |
| `pie smoke [--rpc] [--flavor <name>]` | FFI / RpcServer smoke test |

# Key directories

| Path | Purpose |
|------|---------|
| `server/` | `pie-standalone` engine library + `pie` CLI binary (lib + bin) |
| `runtime/` | Rust runtime (wasmtime, scheduler, RpcServer). `rlib`-only. |
| `runtime/wit/` | Runtime-side WIT |
| `sdk/rust/inferlet/` | Rust SDK for inferlets |
| `sdk/rust/inferlet/wit/` | SDK-side WIT (must match runtime/wit/) |
| `sdk/python-rpc/` | `pie-rpc` wheel — Python `RpcServer` for driver launchers |
| `sdk/python-server/` | `pie-server` wheel — embed `pie.server.Server` in Python |
| `inferlets/` | Standard inferlets (text-completion, etc.) |
| `driver/{portable,cuda,dummy}/` | Embedded drivers (linked into `pie` by Cargo feature) |
| `driver/dev/` | Reference Python driver wheel `pie-driver-dev`. Houses model implementations under `pie_driver_dev/model/` and ships `pie_kernels`. |
| `driver/vllm/`, `driver/sglang/` | Python driver wheels delegating to vLLM / SGLang. Both depend on `pie-driver-dev` for shared worker scaffolding. |
| `~/.pie/config.toml` | User-facing serve config |
| `~/.pie/drivers.toml` | Per-driver venv defaults (managed by `pie driver <type> set`) |

# Where things live

| Concern | File |
|---|---|
| Engine boot path (single source of truth) | `server/src/serve.rs::start_engine` |
| TOML schema validation | `server/src/config.rs` |
| Embedded vs subprocess routing | `server/src/serve/topology.rs::resolve_flavor` |
| Embedded supervisor (C++/Rust thread) | `server/src/embedded_driver.rs` |
| Subprocess supervisor (Python child) | `server/src/subprocess_driver.rs` |
| Venv resolution chain | `server/src/python_resolve.rs` |
| CLI dispatch | `server/src/cli.rs` |
| pyo3 boot wrapper for Python embed | `sdk/python-server/src/lib.rs` |
| `pie.server.Server` async ctx mgr | `sdk/python-server/python/pie/server.py` |
| `pie.config.*` dataclasses | `sdk/python-server/python/pie/config.py` |
| Per-driver Python launcher entry | `driver/<flavor>/src/pie_driver_<flavor>/__main__.py` (thin shim) |
| Shared launcher lifecycle | `driver/dev/src/pie_driver_dev/_launcher.py` |
