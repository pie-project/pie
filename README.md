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
pie model import Qwen/Qwen3-0.6B
pie serve
```

`pie serve` boots the engine and holds the terminal. From another shell, submit an
inferlet to it with the Python client (`pip install pie-client`):

```bash
pie-client submit text-completion -- --prompt "The capital of France is"
```

### Running on Vulkan

The Vulkan driver is not in the released binary — it is a compile-time feature,
and a binary without it refuses `type = "vulkan"` at boot rather than falling
back to something slower. Build one that has it:

```bash
cargo build --release -p pie --bin pie --features driver-vulkan
```

It needs two things the CUDA path does not. First, an **artifact rather than a
snapshot**: this driver reads its declared quantization out of the artifact's
embedded config, so the checkpoint is converted once, offline.

```bash
pie model build /path/to/checkpoint --quant int4 --backend vulkan --out qwen3-vk.zt
```

The source is a snapshot directory, a repo ID in the local HF cache, or an
existing `.zt`. `--backend` is not cosmetic: CUDA binds fused q/k/v banks under
HuggingFace names while Metal and Vulkan bind in-place projections under MLX
names, and an artifact materialized for one is not what the other's bind path
reads.

Second, the **compiled shaders**. `crates/kernels-vulkan`'s build script emits
SPIR-V into the cargo out-dir, and the driver is told where rather than
guessing — it consumes modules, it does not produce them:

```bash
ls target/release/build/kernels-vulkan-*/out/spv
```

Then replace `[driver]` in `$PIE_HOME/config.toml` — the block `pie config
init` writes is the dummy driver's, and its `vocab_size`/`arch_name` are keys
the Vulkan driver rejects by name at boot:

```toml
[model]
name = "qwen3"
model = "/path/to/qwen3-vk.zt"

[driver]
type = "vulkan"
device = ["vulkan:0"]
kernels = "/abs/path/to/target/release/build/kernels-vulkan-<hash>/out/spv"
kv_pages = 256
activation_dtype = "bfloat16"
```

`name` is required by the config and is a LABEL: the driver reads the
architecture out of the artifact's embedded config, so it does not select
anything. `kernels` must be absolute. `device` is stated because the config requires it
and ignored because the driver takes the first Vulkan device the loader
reports. `kv_pages` sizes the KV pool directly: unlike CUDA there is no
`gpu_mem_utilization` here, because this driver is told how many pages to hold
rather than deriving them from a fraction of the card.

`pie serve` then boots as usual, and `pie run` will answer without one:

```bash
pie run chat-completion -- --prompt "The capital of France is" --max_tokens 16
```

## Project Layout

| Directory | Description |
|---|---|
| `src/` | The `pie` CLI and the three role daemons — the invariant entry point |
| `crates/engine/` | Inferlet runtime |
| `crates/tensor-*/` | Tensor-program toolchain: authoring eDSL → PTIR → planning → CUDA/Metal codegen (+ the reference interpreter) |
| `crates/model*/` | What a model is: the registries, the generations, the checkpoint loader, the forward compiler |
| `crates/controller/` | Cluster-coordination control plane (pairing · roles · health) |
| `crates/transport/` | Worker↔worker P2P KV-tensor data plane |
| `crates/driver*/` | Backend drivers (CUDA · Metal · Vulkan · WGPU) + the shared execution-shell substrate |
| `crates/*-api`, `crates/*-abi` | Boundary contracts (`client` · `controller` · `worker` · `inferlet` · `driver`) — the dependency floor |
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
