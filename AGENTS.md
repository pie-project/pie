# Pie Project Structure

This document provides a concise overview of the key directories and components in the Pie project for Agents to understand the architecture.

## `runtime`
**The Rust runtime.**
*   **Path**: `runtime/`
*   **Language**: Rust
*   **Description**: Core engine — WASM (Wasmtime) + scheduler + networking. Consumed as an `rlib` by `server` and `sdk/python-rpc/`.
*   **Key Dependencies**: `wasmtime`, `tokio`, `ipc-channel`. After Phase 8, runtime no longer ships pyo3 bindings — the legacy `pie._runtime` maturin module is gone; pyo3 lives only in `sdk/python-rpc/`.

## `server`
**`pie` CLI — the Rust-only standalone server.**
*   **Path**: `server/`
*   **Language**: Rust
*   **Description**: Single-binary entrypoint. Boots embedded drivers (portable / cuda_native / dummy) as static libs and Python drivers (dev / vllm / sglang) as subprocesses; mirrors the legacy `pie-server` Python CLI surface.
*   **Subcommands**: `serve`, `run`, `model`, `config`, `auth`, `new`, `build`, `doctor`, `check`, `smoke`, `driver`.
*   `pie driver <type> {install|doctor|set|unset|show|exec}` — manages per-driver venvs in `~/.pie/drivers.toml` (BYO-venv; standalone never installs wheels itself, just prints the `uv venv` + `uv pip install` recipe).

## `driver/`
**Inference drivers — embedded (Rust/C++) + subprocess (Python).**
*   `driver/portable/` — C++ ggml driver (CMake static lib, linked into pie-standalone).
*   `driver/cuda/` — C++ CUDA driver (CMake static lib).
*   `driver/dummy/` — Rust dummy driver (rlib).
*   `driver/dev/` (wheel `pie-driver-dev`) — reference Python driver. Houses model implementations under `pie_driver_dev/model/` (llama3, qwen2/3, gemma2/3/4, mistral3, olmo3, gpt_oss) and ships `pie_kernels` (Metal + CUDA kernel dispatch) since dev is its only consumer.
*   `driver/vllm/` (wheel `pie-driver-vllm`), `driver/sglang/` (wheel `pie-driver-sglang`) — Python driver wheels delegating to vLLM / SGLang. Reuse `pie_driver_dev`'s worker scaffolding (`run_worker`, `calculate_topology`, `batching`, `telemetry`).
*   Each Python driver has a `__main__.py` launcher that the standalone's `SubprocessDriver` invokes via `python -m pie_driver_<flavor>`.

## `sdk/python-rpc`
**`pie-rpc` wheel — Python bindings for `pie::device::RpcServer`.**
*   **Path**: `sdk/python-rpc/`
*   **Description**: Tiny pyo3 wheel exposing `RpcServer` to Python drivers (cold-path channel back to the Rust runtime). Pairs with the shmem fast path each driver mounts at `/pie_shmem_g{group_id}` for `fire_batch`.

## `sdk`
**SDK for Writing Inferlets.**
*   **Path**: `sdk/`
*   **Description**: Contains libraries and tools for developers to write "Inferlets" (WASM programs that run on Pie).
*   **Subdirectories**:
    *   `rust/`, `python/`, `javascript/`: Client SDKs and Inferlet APIs.

### `sdk/tools/bakery`
**Inferlet Toolchain (`bakery`).**
*   **Path**: `sdk/tools/bakery/`
*   **Language**: Python (`pie-bakery`)
*   **Description**: The CLI tool for developing Inferlets.
    *   `bakery create`: Scaffolds new projects.
    *   `bakery build`: Compiles source (Rust/JS) to WASM.
    *   `bakery publish`: Publishes inferlets to the registry.

## `client`
**Client Libraries.**
*   **Path**: `client/`
*   **Description**: Contains client-side libraries for connecting to a serving Pie instance.