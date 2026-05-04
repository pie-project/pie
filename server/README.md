# server

Rust crate for the `pie` CLI and standalone engine.

`pie` hosts the runtime, embedded drivers (`portable`, `cuda_native`,
`dummy`), and Python subprocess drivers (`dev`, `vllm`, `sglang`) behind the
same config and WebSocket API.

## Build

```bash
cargo build -p pie-server --release
```

Include the embedded CUDA driver:

```bash
CUDACXX=/usr/local/cuda/bin/nvcc \
  cargo build -p pie-server --release --features driver-portable,driver-cuda
```

The binary lands at `target/release/pie`.

## Useful commands

```bash
pie config init
pie doctor
pie driver list
pie serve --config ~/.pie/config.toml
pie run text-completion --prompt "Paris is"
pie model download Qwen/Qwen3-0.6B
```

`pie run` starts a one-shot engine and exits after the inferlet returns.
`pie serve` starts a long-lived WebSocket server.

## Files

```text
server/src/cli/                 subcommands
server/src/config.rs            user-facing TOML schema
server/src/serve.rs             engine startup/shutdown
server/src/embedded_driver.rs   embedded driver startup
server/src/subprocess_driver.rs Python driver supervision
server/src/rpc_loop.rs          embedded-driver RPC loop
```
