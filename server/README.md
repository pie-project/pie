# pie-server

Standalone Pie server and CLI. The `pie` binary hosts the Rust runtime,
the always-available embedded `dummy` driver, optional embedded native
drivers (`portable`, `cuda_native`), and subprocess Python drivers
(`dev`, `vllm`, `sglang`) behind the same config and websocket surface.

## Build

```bash
# Default: portable driver.
cargo build -p pie-server --release

# CUDA + portable + always-linked dummy in one binary.
CUDACXX=/usr/local/cuda/bin/nvcc cargo build -p pie-server --release \
  --no-default-features --features driver-cuda,driver-portable
```

The resulting CLI is `target/release/pie`. `driver-portable` and
`driver-cuda` are optional Cargo features; `dummy` is always linked.
Runtime dispatch follows each `[[model]].driver.type`.

## Driver Types

Embedded drivers run inside the standalone process:

- `portable`: ggml/llama.cpp-backed portable driver.
- `cuda_native`: Pie CUDA driver.
- `dummy`: random-token plumbing driver for smoke tests, always available.

Subprocess drivers run `python -m pie_driver_<type>` with an interpreter
resolved by `pie driver <type> ...`:

- `dev`: reference PyTorch driver.
- `vllm`: vLLM-backed driver.
- `sglang`: SGLang-backed driver.

Use `pie driver list`, `pie driver <type> install`, `pie driver <type> set
venv <path>`, and `pie driver <type> doctor` to manage subprocess driver
environments.

## Config

`pie serve` reads `~/.pie/config.toml` by default. `hf_repo` accepts either
a local HuggingFace snapshot directory or an `owner/name` repo ID. Repo IDs
resolve through the default HuggingFace cache and download on cache miss.

Minimal CUDA config:

```toml
[server]
host = "127.0.0.1"
port = 8080

[auth]
enabled = true

[[model]]
name = "default"
hf_repo = "Qwen/Qwen3-0.6B"

[model.driver]
type = "cuda_native"
device = ["cuda:0"]
tensor_parallel_size = 1

[model.driver.options]
max_num_kv_pages = 1024
```

For subprocess drivers, `venv` or `python` may be set under
`[model.driver.options]` as a per-model override. Those keys are consumed
by the standalone and are not forwarded to the Python launcher.

## Common Commands

```bash
pie config init
pie config show
pie check ~/.pie/config.toml
pie check ~/.pie/config.toml --debug
pie doctor
pie serve --config ~/.pie/config.toml
pie serve --monitor
pie run text-completion@0.1.0 --input '{"prompt":"Paris is","max_tokens":16}'
pie run text-completion@0.1.0 --stdout --input '{"prompt":"Paris is","max_tokens":16}'
pie model list
pie model download Qwen/Qwen3-0.6B
```

`pie run` starts a one-shot in-process engine and shuts it down after
the inferlet completes. By default it prints the raw final return value;
use `--stdout` to also pass through stdout emitted by the inferlet as it
arrives. During the current testing phase, server verbose startup logging
is enabled by default.

## Layout

```text
server/src/cli/              CLI subcommands
server/src/config.rs         User-facing TOML schema and validation
server/src/serve.rs          Engine startup and shutdown orchestration
server/src/embedded_driver.rs Embedded driver supervision and startup TOML
server/src/subprocess_driver.rs Python driver supervision and handshake
server/src/rpc_loop.rs       Embedded-driver cold-path RPC dispatcher
```
