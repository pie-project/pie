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
next to the model. An inferlet drives the forward pass and owns its KV cache,
so agent loops, tool calls, custom samplers and cache policies are written per
application, in Python, JavaScript or Rust, without modifying the engine.

> **Note**
> Pie is pre-release software under active development.

## Install

Pie is one binary plus a language component per scripting language, all on the
[latest release](https://github.com/pie-project/pie/releases/latest). Pick the
archive for your platform and GPU (`cuda`, `metal`, `vulkan` or `wgpu`):

```bash
tar -xzf pie-x86_64-linux-cuda13.0.tar.gz
install pie ~/.local/bin/                             # anywhere on PATH
pie language install pie-language-python.tar.gz       # Python inferlets
pie language install pie-language-javascript.tar.gz   # JavaScript inferlets
```

The installer does the same four steps, choosing the archive for you:

```bash
curl -fsSL https://pie-project.org/install.sh | bash        # Linux, macOS, WSL
irm https://pie-project.org/install.ps1 | iex               # Windows PowerShell
```

Then a model:

```bash
pie config init
pie model import Qwen/Qwen3.5-0.8B
```

## Write an inferlet

`Pie.toml` names the program and its language; `main.py` is the program. This
one is greedy text completion, one forward pass per token:

```toml
[package]
name = "quickstart"
version = "0.1.0"

[runtime]
language = "python"
core = "^0.2.0"
```

```python
from inferlet import chat, model
from inferlet.eta import ForwardPass, Pipeline, WorkingSet, intrinsics, reduce_argmax


async def main(input: dict) -> dict:
    tokens = chat.prefix() + model.encode(input.get("prompt", "The capital of France is"))
    max_tokens = int(input.get("max_tokens", 8))
    ws = WorkingSet(tokens=len(tokens) + max_tokens)  # KV pages (and recurrent state on a hybrid model)
    pipe = Pipeline()

    done = 0  # tokens already in the KV cache
    for _ in range(max_tokens):
        fwd = ForwardPass()
        fwd.embed(tokens[done:])                        # the new tokens
        fwd.bind_state(ws, ws.geometry(done, len(tokens)))
        out = fwd.epilogue(lambda: reduce_argmax(intrinsics.logits()))  # runs on the device
        pipe.submit(fwd)
        done = len(tokens)
        tokens.append(await out)
    pipe.close()

    return {"text": model.decode(tokens[-max_tokens:])}
```

Every line stands for device state. `embed` seeds the token channel,
`ws.geometry` derives the KV geometry of the span, and the `epilogue` is
traced once and runs on the GPU after each forward pass; the host only sees
the token it publishes. The same program in JavaScript is
[`examples/quickstart-js`](examples/quickstart-js); Rust inferlets use the
[`inferlet`](https://crates.io/crates/inferlet) crate and compile to
`wasm32-wasip2`. [`examples/`](examples) goes on from here: chunked prefill
with a device loop-carried decode, beam search, speculative decoding,
constrained decoding, KV-cache sharing, diffusion.

## Run it

```bash
pie run main.py -- --prompt "The capital of France is"
```

```
{"text": " Paris, and the capital of the United States is Washington,"}
```

`pie run` boots a one-shot engine for the program. To keep one running, `pie
serve` holds the terminal, and the Python client (`pip install pie-client`)
submits the same file from another shell:

```bash
pie-client submit --path main.py --manifest Pie.toml -- --prompt "The capital of France is"
```

## Compatible APIs

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

Each API is itself a built-in inferlet, with tool calling, JSON-schema output,
streaming and reasoning. The same server embeds in a process as
[`pie-server`](https://pypi.org/project/pie-server/) on PyPI and
[`@pie-project/server`](https://www.npmjs.com/package/@pie-project/server) on
npm; under a bundler the npm package is pie compiled for the browser, on
WebGPU.

## Building from source

Four engines, each a compile-time feature:

```bash
cargo build --release -p pie --bin pie --features cuda    # NVIDIA, Linux/Windows
cargo build --release -p pie --bin pie --features metal   # Apple silicon, macOS
cargo build --release -p pie --bin pie --features vulkan  # any Vulkan 1.2 device (slangc on PATH)
cargo build --release -p pie --bin pie --features wgpu    # WebGPU: Vulkan, Metal or D3D12
```

## Getting Help

[GitHub Issues](https://github.com/pie-project/pie/issues) and
[GitHub Discussions](https://github.com/pie-project/pie/discussions).

## License

[Apache License 2.0](LICENSE). Third-party attributions: [NOTICE](NOTICE).
