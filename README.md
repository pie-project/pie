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

Pie runs small user-supplied WebAssembly programs, called *inferlets*, directly
next to the model. Inferlets have direct access to the KV cache and forward
pass, so agent loops, tool calls, custom samplers, and cache policies are
customized per application without modifying the engine.

> **Note**
> Pie is pre-release software under active development.

## Quick Start

Pie is a standalone binary, no Python needed.

```bash
curl -fsSL https://pie-project.org/install.sh | bash        # Linux, macOS, WSL
irm https://pie-project.org/install.ps1 | iex               # Windows PowerShell
```

```bash
pie config init
pie model import Qwen/Qwen3.5-0.8B
pie serve
```

The installer also places the Python and JavaScript language components
(`pie-language-<language>.tar.gz`, one release asset per language) under
`~/.pie/languages`; `pie language list` shows them, and `pie language install
<file>` adds one from a downloaded asset or a locally built `.wasm`.

A checkpoint is matched against the catalog's import contracts at load and
refused by name when none fits; `pie model list` prints the SKU beside every
snapshot it can see.

`pie serve` holds the terminal. From another shell, submit an inferlet with the
Python client (`pip install pie-client`):

```bash
pie-client submit text-completion -- --prompt "The capital of France is"
```

`pie run` is the same round trip without a server:

```bash
pie run --path ./target/wasm32-wasip2/debug/text_completion.wasm \
        --manifest ./Pie.toml -- --prompt "The capital of France is"
```

### Compatible APIs

`pie serve` also speaks the APIs existing clients already use, on the same
port, with no API key:

| Route | Client |
|---|---|
| `POST /v1/chat/completions`, `/v1/completions`, `/v1/responses`, `GET /v1/models` | `openai` |
| `POST /v1/messages` | `anthropic` |
| `POST /v1beta/models/{model}:generateContent`, `:streamGenerateContent?alt=sse` | `google-genai` |

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="unused")
print(client.chat.completions.create(model="default", messages=[{"role": "user", "content": "Hi"}]).choices[0].message.content)
```

The same server embeds in a process: `pie-server` on PyPI (`python/server`)
and `@pie-project/server` on npm (`javascript/server`) boot what `pie serve`
boots and hand back the address, each carrying every engine its platform
supports (`engine.type` picks one at boot); `@pie-project/browser` is pie
compiled for the browser.

Each API is served by a built-in inferlet (`crates/builtins/inferlets/compat-openai`,
`compat-anthropic`, `compat-gemini`) built into the `pie` binary; `pie doctor`
lists them, and a newer version installed with `pie inferlet install` takes
over. Tool calling, JSON schema output, streaming and reasoning
(`reasoning_content`, `thinking` blocks, thought parts) are supported; images
are not yet.

### Backends

Four engines serve today, each a compile-time feature:

```bash
cargo build --release -p pie --bin pie --features cuda    # NVIDIA, Linux/Windows
cargo build --release -p pie --bin pie --features metal   # Apple silicon, macOS
cargo build --release -p pie --bin pie --features vulkan  # any Vulkan 1.2 device
cargo build --release -p pie --bin pie --features wgpu    # WebGPU: Vulkan, Metal or D3D12
```

`vulkan` compiles Slang to SPIR-V at build time and wants `slangc` on `PATH`
(or `PIE_SLANGC`); `wgpu` needs no shader toolchain.

## Building inferlets

Inferlets compile to the `wasm32-wasip2` component target:

```bash
rustup target add wasm32-wasip2
cargo build --target wasm32-wasip2
```

## Getting Help

[GitHub Issues](https://github.com/pie-project/pie/issues) and
[GitHub Discussions](https://github.com/pie-project/pie/discussions).

## License

[Apache License 2.0](LICENSE). Third-party attributions: [NOTICE](NOTICE).
