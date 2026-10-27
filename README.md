<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset=".github/assets/pie-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset=".github/assets/pie-light.svg">
    <img alt="Pie: Programmable inference engine"
         src=".github/assets/pie-light.svg"
         width="30%">
    <p></p>
  </picture>

[Website] | [Docs] | [Playground] | [Benchmarks] | [Paper (SOSP'25)]
</div>

[Website]: https://pie-project.org/
[Docs]: https://pie-project.org/docs
[Playground]: https://pie-project.org/playground
[Benchmarks]: https://benchmarks.pie-project.org/
[Paper (SOSP'25)]: https://ingim.org/papers/gim2025pie.pdf

Pie is a programmable inference engine that serves programs, not prompts. It
runs small user-supplied WebAssembly programs, called *inferlets*, directly
next to the model. An inferlet drives the forward pass and owns its state, so
agent loops, speculative decoding, custom samplers, and KV-cache policies are
written per application—in Python, JavaScript, or Rust—without modifying the
engine.

Models themselves are programs written in [Poem](models), not hardcoded in
Rust, so a single runtime serves language, multimodal, diffusion, image, and
video models. The entire system ships as one ~60 MB binary with no Python or
PyTorch runtime dependency, running on NVIDIA and AMD GPUs, Apple silicon,
Google TPUs, Vulkan, and in the browser on WebGPU.

## Install

```bash
curl -fsSL https://pie-project.org/install.sh | bash        # Linux, macOS, WSL
irm https://pie-project.org/install.ps1 | iex               # Windows PowerShell
```

Then initialize a default configuration and import a model:

```bash
pie config init
pie model import Qwen/Qwen3.5-0.8B
```

See the [install guide](https://pie-project.org/docs/install) for manual
archives and backend flavors (`cuda`, `metal`, `vulkan`, `wgpu`), or open the
[Playground] to run Pie in a browser tab with no install.

## Write an inferlet

An inferlet is one file; its extension selects the language. This one is greedy
text completion ([`examples/quickstart-py`](examples/quickstart-py)), one
forward pass per token:

```python
from inferlet import chat, model
from inferlet.eta import ForwardPass, Pipeline, StateGeometry, WorkingSet, intrinsics, reduce_argmax


async def main(input: dict) -> dict:
    tokens = chat.prefix() + model.encode(input.get("prompt", "The capital of France is"))
    max_tokens = int(input.get("max_tokens", 8))
    ws = WorkingSet()  # the sequence's state, whatever the model keeps
    pipe = Pipeline()

    done = 0  # tokens already in the working set
    for _ in range(max_tokens):
        fwd = ForwardPass()
        fwd.embed(tokens[done:])                        # the new tokens
        fwd.state([ws], StateGeometry.folding())        # appended, and final
        out = fwd.epilogue(lambda: reduce_argmax(intrinsics.output(0, "logits")))  # runs on the device
        pipe.submit(fwd)
        done = len(tokens)
        tokens.append(await out)
    pipe.close()

    return {"text": model.decode(tokens[-max_tokens:])}
```

Every line describes device work. `embed` feeds the new tokens,
`StateGeometry.folding()` appends them to the working set, and `epilogue` is
traced once and compiled into a GPU kernel that runs after each forward pass;
the host only sees the token it publishes.

The same program in JavaScript is [`examples/quickstart-js`](examples/quickstart-js).
Rust inferlets use the [`inferlet`](https://crates.io/crates/inferlet) crate
and compile to `wasm32-wasip3` ([`examples/rust-toolchain.toml`](examples/rust-toolchain.toml)).
[`examples/`](examples) goes on from here: device loop-carried decode, beam
search, speculative decoding, KV eviction and sharing, constrained decoding,
custom samplers, watermarking, and media generation.

## Run & serve

`pie run` runs an inferlet in a one-shot engine:

```bash
pie run main.py -- --prompt "The capital of France is"
# {"text": " Paris, and the capital of the United States is Washington,"}
```

`pie serve` keeps the engine running. You can submit inferlets to it with
`pie-client` (`pip install pie-client`), or connect existing OpenAI, Anthropic,
and Gemini clients to the same port with no API key—each compatibility API is
itself a built-in inferlet:

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="unused")
print(client.chat.completions.create(model="default", messages=[{"role": "user", "content": "Hi"}]).choices[0].message.content)
```

The same server embeds in a process via [`pie-server`](https://pypi.org/project/pie-server/)
on PyPI and [`@pie-project/server`](https://www.npmjs.com/package/@pie-project/server)
on npm (which also bundles for the browser on WebGPU).

## Building from source

Pick a backend feature (`cuda`, `metal`, `vulkan`, `wgpu`, `rocm`, or `xla`):

```bash
cargo build --release -p pie --bin pie --features metal
```

## Community & License

Join [Discord](https://discord.gg/R62Udyzum), open an
[Issue](https://github.com/pie-project/pie/issues), or start a
[Discussion](https://github.com/pie-project/pie/discussions).

Licensed under the [Apache License 2.0](LICENSE) ([NOTICE](NOTICE)).
