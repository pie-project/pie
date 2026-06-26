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
| `interface/` | Boundary contract crates (`ids` · `driver` · `controller` · `worker` · `client` · `inferlet`) — the dependency floor |
| `inferlets/` | Example inferlets |
| `sdk/` | Inferlet SDKs (Rust · Python · JavaScript) |
| `client/` | Client libraries (Rust · Python · JavaScript) |
| `website/` | [pie-project.org](https://pie-project.org) docs site |

## Building inferlets (wasip3 toolchain)

Inferlets compile to the `wasm32-wasip3` component target (WASI 0.3 /
Component-Model async). The pinned toolchain lives in `rust-toolchain.toml`
(a dated nightly — `wasm32-wasip3` is Tier-3 and needs `-Zbuild-std`). Run the
one-time setup once after cloning (or after a toolchain bump):

```bash
./scripts/setup-wasip3.sh
```

It installs the `wasm32-wasip2` std as a wasi-libc donor (`wasm32-wasip3` ships
no bundled libc) and puts the `wasip3-link.sh` linker wrapper on your `PATH`.
Set `WASI_SDK_PATH` to a pinned wasi-sdk to link against its wasi-sysroot libc
instead of the donor. Inferlet guest crates pin a wit-bindgen version distinct
from the one std's wasip3 bindings bundle (avoids a duplicate `cabi_realloc`
symbol under `-Zbuild-std`).

## Getting Help

Questions and bug reports are welcome on
[GitHub Issues](https://github.com/pie-project/pie/issues) and
[GitHub Discussions](https://github.com/pie-project/pie/discussions).

## License

[Apache License 2.0](LICENSE)
