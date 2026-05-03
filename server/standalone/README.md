# server/standalone

All-Rust Pie server. Sibling to `server/torch` (the existing Python
`pie-server` package, currently still rooted at `pie/`). Statically
links one of the native drivers (`driver/portable` or `driver/cuda`)
in-process so the resulting `pie` binary is a single deployable file
with no Python or CUDA-shared-library dependencies at runtime.

> **Status:** Architecture verified end-to-end as of M4 — the binary
> builds and the FFI link to either driver works. Hello-world e2e
> against a real model has not yet been validated; cuda config-TOML
> parsing is the last remaining piece (see [Known gaps](#known-gaps)).

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ pie (single binary, this crate)                             │
│                                                             │
│  ┌───────────────┐   intra-process   ┌───────────────────┐  │
│  │ runtime/      │ ◄───────────────► │ embedded driver   │  │
│  │ (Rust crate)  │   shmem + RPC     │  (thread, links   │  │
│  │  - websocket  │                   │   driver/cuda or  │  │
│  │  - scheduler  │                   │   driver/portable │  │
│  │  - wasmtime   │                   │   STATIC)         │  │
│  └───────────────┘                   └───────────────────┘  │
└─────────────────────────────────────────────────────────────┘
        ▲                                       │
        │ websocket                             │ optional aux IPC
        │                                       │ (page-copy)
   PieClient                                    ▼
                                          (in-process today;
                                           hosted by the same
                                           binary)
```

Same wire protocol the Python wrapper drives, just collapsed into one
process. The runtime crate has no idea whether it's talking to a
subprocess or a thread.

## Build

```bash
# Default: portable driver (CPU + ggml's optional GPU backends).
cargo build -p pie-standalone --release

# CUDA driver (statically linked cudart/cublas/cublasLt; GPU at runtime
# only — no CUDA toolkit required at runtime).
cargo build -p pie-standalone --release \
    --no-default-features --features driver-cuda
```

Both produce `target/release/pie`. Mutual exclusion is enforced — you
get exactly one driver flavor per binary; ship two binaries if you
need both.

### Build prerequisites

| | `driver-portable` | `driver-cuda` |
|---|---|---|
| C++ compiler | C++20 (gcc/clang) | C++20 |
| CMake | ≥3.23 | ≥3.23 |
| CUDA toolkit | not required | required at build (static-linked, not at runtime) |
| First-build time | ~2–5 min (ggml) | ~10–30 min (CUDA kernels) |

## Run

```bash
# Boot the runtime + embedded driver. Reads ~/.pie/config.toml shape;
# rejects torch driver types (`native`, `vllm`, `sglang`) with a clear
# error pointing you at server/torch.
pie --config /etc/pie/dev.toml

# Validate a config without booting.
pie --check /etc/pie/dev.toml

# FFI smoke (drives the embedded driver entry with --help).
pie --smoke

# Verify the runtime crate is callable without pyo3.
pie --smoke-rpc
```

Config schema mirrors `pie.config` (the Python `server/torch` schema).
Two restrictions:

1. `driver.type` must be `"portable"` or `"cuda_native"`. The torch
   drivers (`native`, `vllm`, `sglang`) need a Python interpreter to
   host torch in-process and aren't supported here.
2. `model.hf_repo` must currently be a local snapshot directory. HF
   download support is post-MVP — for now, pre-download with
   `huggingface-cli download <repo> --local-dir <dir>` and point
   `hf_repo` at `<dir>`.

Example minimum portable config:

```toml
[server]
host = "127.0.0.1"
port = 8080

[auth]
enabled = false

[[model]]
name = "default"
hf_repo = "/var/cache/hf/Qwen3-0.6B"

[model.driver]
type = "portable"
device = ["cpu"]

[model.driver.options]
kv_page_size = 32
max_num_kv_pages = 1024
```

## Capability matrix vs `server/torch`

| | `server/torch` (Python) | `server/standalone` (Rust) |
|---|---|---|
| Distribution shape | pip wheel + venv | single binary |
| Python at runtime | required | not required |
| Drivers | `native`, `vllm`, `sglang` (torch-hosted), `cuda_native`, `portable` | `cuda_native`, `portable` only |
| HF download | yes (huggingface_hub) | not yet — point at local snapshot |
| `pie monitor` TUI | yes | no |
| `pie new` / `pie build` (inferlet authoring) | yes | no — use `server/torch` for authoring |
| `pie serve` (run a model) | yes | yes |

The native-driver paths (`cuda_native`, `portable`) work in both —
this crate is the right choice for production deployments where you
want a single binary, no Python, and no CUDA-runtime-toolkit
dependency. `server/torch` remains the right choice for interactive
development, multi-driver workloads (especially vLLM/SGLang), and
inferlet authoring.

## Known gaps

These are all implementable from where the architecture sits today;
they were intentionally deferred to land the architecture first.

- **CUDA config TOML schema.** `embedded_driver::write_startup_toml`
  emits the portable driver's TOML shape (`n_gpu_layers`, `n_ctx`,
  …). The cuda driver expects different keys (`model.device`,
  `model.dtype`, …) — adding a `write_cuda_startup_toml` + a
  `CudaDriverOptions` config struct mirrors what `pie_driver_portable`
  already has.
- **Aux-IPC client.** `rpc_loop.rs` returns clean errors for the
  page-copy / `load_adapter` cold-path methods. Wiring them to the
  driver's aux IPC unix socket lifts that cap. Not on the critical
  path for hello-world (those RPCs only fire when `swap_pool_size > 0`).
- **HF download.** Currently `hf_repo` must be a local directory. A
  Rust-side huggingface-hub fetcher would lift this — but it's
  optional; pre-downloading via the python tool works today.
- **macOS / Metal.** `build.rs` panics on non-Linux for the cuda
  flavor, and the portable flavor's OpenMP linkage on macOS is a
  no-op stub.
- **CI matrix.** The repo's existing `.github/workflows` covers
  releases (cargo / npm / pypi). Build verification for both
  `pie-standalone` flavors should be added once a CI runner with a
  CUDA toolkit is available.

## Layout

```
server/standalone/
├── Cargo.toml          # `pie-standalone` crate metadata, feature flags
├── build.rs            # cmake-rs invocation per driver flavor + link flags
├── src/
│   ├── main.rs         # CLI dispatch (--config / --check / --smoke[-rpc])
│   ├── config.rs       # User-facing TOML schema (mirror of pie.config)
│   ├── driver_ffi.rs   # Feature-gated FFI extern declarations
│   ├── embedded_driver.rs  # Driver thread spawn, caps callback bridge
│   ├── rpc_loop.rs     # Cold-path RPC dispatch loop
│   ├── bootstrap_translate.rs  # config::Config → pie::bootstrap::Config
│   └── serve.rs        # Tokio orchestrator: bootstrap, ctrl-c
└── README.md           # this file
```

`pie::bootstrap::bootstrap()` (in `runtime/`) does the heavy lifting
once `serve.rs` hands it a `Config`. The runtime crate's `python`
feature is default-on for the maturin build (`server/torch`); we
disable it here via `default-features = false` so pyo3 stays out of
the dep graph entirely.
