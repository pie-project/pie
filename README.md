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

Pie runs WebAssembly programs—[*inferlets*](examples)—directly next to the
model. An inferlet drives the forward pass, owns its state, and compiles custom
epilogues to GPU kernels, so decoding, sampling, caching, and agent loops are
written per application in Python, JavaScript, or Rust.

Models are programs too, written in [Poem](models) across language, vision,
image, and video. Everything ships as one ~60 MB binary with no Python or
PyTorch runtime, running on CUDA, ROCm, Metal, Vulkan, XLA (TPUs), and WebGPU.

## Install

```bash
curl -fsSL https://pie-project.org/install.sh | bash        # Linux, macOS, WSL
irm https://pie-project.org/install.ps1 | iex               # Windows PowerShell
```

Import a model from Hugging Face:

```bash
pie config init
pie model import Qwen/Qwen3.5-0.8B
```

See the [install guide](https://pie-project.org/docs/install) for manual
archives and GPU flavors, or run Pie in your browser on the [Playground].

## Write an inferlet

Greedy text completion in Python ([`examples/quickstart-py`](examples/quickstart-py)),
one forward pass per token (also in [JavaScript](examples/quickstart-js) and
[Rust](crates/inferlet)):

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

`epilogue` is traced once and compiled into a GPU kernel; the host only sees
the token it publishes. More in [`examples/`](examples):
[device loop-carried decode](examples/text-completion),
[speculative decoding](examples/mtp-speculative-decoding),
[beam search](examples/beam-search),
[KV eviction](examples/snapkv-eviction) and
[prefix sharing](examples/prefix-tree-kv-cache),
[constrained decoding](examples/json-schema-constrained-decoding),
[custom samplers](examples/top-a-sampling),
[watermarking](examples/synthid-tournament-sampling), and
[media generation](examples/text-to-media).

## Run & serve

Run an inferlet one-shot, or submit it to a running `pie serve` with
[`pie-client`](https://pypi.org/project/pie-client/):

```bash
pie run main.py -- --prompt "The capital of France is"
pie-client submit --path main.py -- --prompt "The capital of France is"
```

`pie serve` also speaks the
[OpenAI](crates/builtins/inferlets/compat-openai),
[Anthropic](crates/builtins/inferlets/compat-anthropic), and
[Gemini](crates/builtins/inferlets/compat-gemini) APIs on the same port as
[built-in inferlets](crates/builtins/inferlets):

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="unused")
print(client.chat.completions.create(model="default", messages=[{"role": "user", "content": "Hi"}]).choices[0].message.content)
```

To embed in a process, use [`pie-server`](https://pypi.org/project/pie-server/)
(Python) or [`@pie-project/server`](https://www.npmjs.com/package/@pie-project/server)
(Node and WebGPU browsers).

## Build from source

```bash
cargo build --release -p pie --bin pie --features metal   # or: cuda, vulkan, wgpu, rocm, xla
```

## Community & License

[Discord](https://discord.gg/R62Udyzum) ·
[Issues](https://github.com/pie-project/pie/issues) ·
[Discussions](https://github.com/pie-project/pie/discussions) ·
[Contributing](CONTRIBUTING.md)

[Apache License 2.0](LICENSE) ([NOTICE](NOTICE)).
