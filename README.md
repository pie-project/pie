<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://pie-project.org/img/pie-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="https://pie-project.org/img/pie-light.svg">
    <img alt="Pie: Programmable serving system for emerging LLM applications"
         src="https://pie-project.org/img/pie-light.svg"
         width="30%">
    <p></p>
  </picture>

[Website] | [Guide] | [Reference] | [Paper (SOSP'25)]
</div>

[Website]: https://pie-project.org/
[Guide]: https://pie-project.org/docs/guide/install
[Reference]: https://pie-project.org/docs/reference/sdk-rust
[Paper (SOSP'25)]: https://ingim.org/papers/gim2025pie.pdf

A programmable serving system for custom inference logic,
stateful agents, and serving-side optimization.



> **Note**
> Pie is pre-release software under active development. It is best suited
> for testing and research right now.


## What is Pie?

Today's LLM serving engines (e.g., vLLM, SGLang, TensorRT-LLM) are black boxes: prompt in, tokens out. But AI agents are a different kind of workload. They branch, call tools, retry, and coordinate long-running workflows, and forcing them through a monolithic token-generation pipeline leads to wasted round trips, KV cache thrashing, and engine patches for every new decoding trick.

Pie is a programmable serving system. It runs small user-supplied WebAssembly programs, called *inferlets*, directly next to the model. Inferlets have direct access to the KV cache and forward pass, so agent loops, tool calls, custom samplers, and cache policies can be customized and optimized per-application without modifying the engine.

## Quick Start

Pie is a standalone binary, no Python needed.

For macOS and Linux:
```bash
curl -fsSL https://pie-project.org/install.sh | bash
```

For Windows, follow the [installation guide](https://pie-project.org/docs/guide/install).


Then configure and run:

```bash
pie config init
pie run text-completion -- --prompt "The capital of France is"
```

## Project Layout

| Directory | Description |
|---|---|
| `worker/` | The `pie` CLI and standalone engine — the invariant entry point |
| `runtime/` | Inferlet runtime |
| `controller/` | Cluster-coordination control plane (pairing · roles · health) |
| `driver/transport/` | Worker↔worker P2P KV-tensor data plane |
| `driver/` | Backend drivers (portable · CUDA) + runtime↔driver IPC |
| `protocol/` | Shared wire schema (`schema` · `schema-derive` · `schema-bindgen`) — the dependency floor |
| `inferlets/` | Example inferlets |
| `sdk/` | Inferlet SDKs (Rust · Python · JavaScript) |
| `client/` | Client libraries (Rust · Python · JavaScript) |
| `website/` | [pie-project.org](https://pie-project.org) docs site |

## Getting Help

Questions and bug reports are welcome on
[GitHub Issues](https://github.com/pie-project/pie/issues) and
[GitHub Discussions](https://github.com/pie-project/pie/discussions).

## License

[Apache License 2.0](LICENSE)
