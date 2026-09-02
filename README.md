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
pie model import Qwen/Qwen3.5-0.8B
pie serve
```

The model has to be one this build ships a forward for — a checkpoint is
matched against the catalog's import contracts at load and refused by name when
none fits. `pie model list` prints the SKU beside every snapshot it can see, and
`Qwen/Qwen3.5-0.8B` is the smallest one; it is also what `pie config init`
writes into `[model] model`.

`pie serve` boots the engine and holds the terminal. From another shell, submit an
inferlet to it with the Python client (`pip install pie-client`):

```bash
pie-client submit text-completion -- --prompt "The capital of France is"
```

`pie run` is the same round trip without a server: it boots a one-shot engine,
runs one inferlet, prints what it produced, and exits — which is what to reach
for when iterating on a local build.

```bash
pie run --path ./target/wasm32-wasip2/debug/text_completion.wasm \
        --manifest ./Pie.toml -- --prompt "The capital of France is"
```

### Backends

Two engines serve today, each a compile-time feature:

```bash
cargo build --release -p pie --bin pie --features cuda    # NVIDIA, Linux/Windows
cargo build --release -p pie --bin pie --features metal   # Apple silicon, macOS
```

`metal` is Apple-only at the crate level: a non-Apple build with the flag on
links no Metal device half, and a config naming that engine is told so.

A binary built with no engine feature has nothing true to put in `[engine]`,
so `pie config init` says so instead of writing a config that will not parse.

## Project Layout

| Directory | Description |
|---|---|
| `src/` | The `pie` CLI and the three role daemons — the invariant entry point |
| `crates/runtime/` | Inferlet runtime |
| `crates/eta-*/` | ETA (Embedded Tensor Algebra) toolchain: authoring eDSL → ETA → planning → CUDA/Metal codegen (+ the reference interpreter) |
| `crates/model*/` | What a model is: the catalog, the authoring eDSL and its traced IR, the forward compiler, the checkpoint loader |
| `crates/controller/` | Cluster-coordination control plane (pairing · roles · health) |
| `crates/transport/` | Worker↔worker P2P KV-tensor data plane |
| `crates/engine*/` | Backend engines: the CUDA and Metal engines + the shared execution-shell substrate |
| `crates/*-api` | Boundary contracts (`client` · `controller` · `worker` · `engine`) — the dependency floor |
| `tests/inferlets/` | Curated inferlet E2E fixtures |
| `sdk/inferlet/` | SDKs for programs that run ON pie (Python · JavaScript · tools) |
| `sdk/client/` | SDKs for programs that CALL pie (Python · JavaScript) |

Every Rust crate lives under `crates/`; the repo root is the workspace and the
`pie` package both. The [pie-project.org](https://pie-project.org) docs site
has its own repo, [pie-project/website](https://github.com/pie-project/website).

## Building inferlets

Inferlets compile to the `wasm32-wasip2` component target. Install the target
once after cloning:

```bash
rustup target add wasm32-wasip2
```

Build an inferlet with:

```bash
cargo build --target wasm32-wasip2
```

## Getting Help

Questions and bug reports are welcome on
[GitHub Issues](https://github.com/pie-project/pie/issues) and
[GitHub Discussions](https://github.com/pie-project/pie/discussions).

## Acknowledgements

The constrained-decoding engine in `crates/grammar` is a Rust rewrite derived
in part from [XGrammar](https://github.com/mlc-ai/xgrammar), licensed under
Apache License 2.0. See [NOTICE](NOTICE) for attribution.

## License

[Apache License 2.0](LICENSE)
