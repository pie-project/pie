# inferlet Python SDK

Python API for writing Pie inferlets.

```python
from inferlet import Context, Model, Sampler, runtime

async def main(input: dict) -> str:
    model = Model.load(runtime.models()[0])
    ctx = Context(model)

    ctx.system("You are helpful.").user(input["prompt"])

    return await ctx.generate(
        Sampler.top_p(0.6, 0.95),
        max_tokens=256,
    ).collect_text()
```

## Main pieces

- `Context`: owns KV-cache state and chat/raw token buffers.
- `Forward`: runs one explicit forward pass with samplers, probes, masks, and
  manual page control.
- `Generator`: multi-step generation loop with stop conditions, constraints,
  speculation, adapters, and JSON collection.
- `chat`, `reasoning`, `tools`: optional decoders and helpers for model-native
  formats.
- `runtime`, `session`, `messaging`, `mcp`: host services exposed to inferlets.

## Build notes

Python inferlets are packaged as Wasm components. Pure-Python dependencies can
be bundled; native extensions such as `numpy`, `orjson`, `msgspec`, or
`pydantic_core` cannot be loaded in the Wasm runtime.

Build through Bakery:

```bash
pie build ./my-python-inferlet -o out.wasm
```
