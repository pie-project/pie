# inferlet (Python)

Python SDK for writing Pie inferlets.

```python
from inferlet import chat, model
from inferlet.eta import ForwardPass, Pipeline, StateGeometry, WorkingSet, intrinsics, reduce_argmax


async def main(input: dict) -> dict:
    tokens = chat.prefix() + model.encode(input.get("prompt", "The capital of France is"))
    max_tokens = int(input.get("max_tokens", 8))
    ws = WorkingSet()
    pipe = Pipeline()

    done = 0
    for _ in range(max_tokens):
        fwd = ForwardPass()
        fwd.embed(tokens[done:])
        fwd.state([ws], StateGeometry.folding())
        out = fwd.epilogue(lambda: reduce_argmax(intrinsics.output(0, "logits")))
        pipe.submit(fwd)
        done = len(tokens)
        tokens.append(await out)
    pipe.close()

    return {"text": model.decode(tokens[-max_tokens:])}
```

## Modules

- `inferlet.eta`: tensor eDSL (`Tensor`, `Channel`, `WorkingSet`, `StateGeometry`, `ForwardPass`, `Pipeline`); emits trace bytes identical to the Rust [`inferlet`](../../../crates/inferlet) crate.
- `inferlet.model` / `inferlet.tokenizer`: model metadata and tokenization.
- `inferlet.grammar` / `inferlet.mask`: JSON Schema, regex, and EBNF constraints.
- `inferlet.chat` / `inferlet.reasoning` / `inferlet.tools`: chat templates, thinking blocks, and tool calls.
- `inferlet.media`: multimodal image, video, and audio spans.
- `inferlet.session`: client streaming and messaging.

## Run & test

```bash
pie run main.py
PYTHONPATH=src python -m pytest tests
```
