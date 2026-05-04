<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://pie-project.org/img/pie-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="https://pie-project.org/img/pie-light.svg">
    <img alt="Pie: Programmable serving system for emerging LLM applications"
         src="https://pie-project.org/img/pie-light.svg"
         width="30%">
    <p></p>
  </picture>

  [![CI](https://github.com/pie-project/pie/actions/workflows/ci.yml/badge.svg)](https://github.com/pie-project/pie/actions/workflows/ci.yml)
  [![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
</div>

A programmable serving system for custom inference logic,
stateful agents, and serving-side optimization.

Pie accepts *inferlets* - small programs in Rust, Python, or TypeScript that compile to WebAssembly and run inside the engine with direct access to the KV cache, token stream, and forward pass.

> **Note**
>
> Pie is pre-release software under active development. It is best suited
> for testing and research right now.

- **[Documentation](https://pie-project.org/)**

- **[Paper (SOSP '25)](https://ingim.org/papers/gim2025pie.pdf)**
## Quick Start

Pie is a standalone binary (~14 MB on macOS) with no Python/PyTorch dependencies.

```bash
curl -fsSL https://pie-project.org/install.sh | bash
```

Then configure and run:

```bash
pie config init
pie run text-completion --prompt "The capital of France is"
```

## Project Layout

| Directory | Description |
|---|---|
| `runtime/` | Wasm-based inferlet runtime (host-side) |
| `server/` | `pie` CLI and standalone engine |
| `inferlets/` | Example inferlets (agents, decoding strategies, tools, …) |
| `sdk/` | Inferlet SDKs (Rust · Python · JavaScript) |
| `client/` | Client libraries (Rust · Python · JavaScript) |
| `driver/` | Model backends (portable / CUDA / vLLM / SGLang) |
| `website/` | [pie-project.org](https://pie-project.org) docs site |

## Getting Help

Questions and bug reports are welcome on
[GitHub Issues](https://github.com/pie-project/pie/issues) and
[GitHub Discussions](https://github.com/pie-project/pie/discussions).

## License

[Apache License 2.0](LICENSE)
