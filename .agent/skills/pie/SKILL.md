---
name: pie
description: How to run, test and debug Pie
---

# Setup

```bash
# First time / clean install
uv sync --extra cu128
```

```bash
# After modifying Rust runtime code
touch pyproject.toml && uv sync --extra cu128
```

Python edits take effect immediately (editable install).

# Building Inferlets

Inferlets compile to WASM. Target: `wasm32-wasip2`.

```bash
# Build a single inferlet
cd std/text-completion
cargo build --release --target wasm32-wasip2

# Publish to registry
bakery inferlet publish
```

WIT definitions live in `sdk/rust/inferlet/wit/` (SDK side) and `runtime/wit/` (runtime side). Keep them in sync.

# Running

```bash
# One-shot inferlet (from registry)
uv run pie run text-completion -- --prompt "Hello world"

# One-shot inferlet (local wasm + manifest)
uv run pie run --path target/wasm32-wasip2/release/text_completion.wasm \
               --manifest Pie.toml \
               -- --prompt "Hello world"

# Long-running server
uv run pie serve
```

Config: `~/.pie/config.toml` (edit directly or via `pie config`).

# Testing

```bash
# Runtime unit tests
cd runtime && cargo test

# SDK check (WASM target)
cd sdk/rust/inferlet && cargo check --target wasm32-wasip2

# Specific inferlet check
cd std/text-completion && cargo check --target wasm32-wasip2
```

# Key Directories

| Path | Purpose |
|------|---------|
| `runtime/` | Rust runtime (wasmtime, linker, bootstrap, PyO3 bindings) |
| `runtime/wit/` | Runtime-side WIT interfaces |
| `sdk/rust/inferlet/` | Rust SDK for writing inferlets |
| `sdk/rust/inferlet/wit/` | SDK-side WIT interfaces (must match runtime) |
| `std/` | Standard inferlets (text-completion, etc.) |
| `pie/src/pie_cli/` | Python CLI (`pie run`, `pie serve`) |
| `pie/src/pie_backend/` | Python GPU backend (model loading, inference engine) |
| `pie/pyproject.toml` | Build config (maturin + uv) |
