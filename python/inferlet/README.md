# inferlet (Python)

The Python library for writing Pie inferlets — the small programs that run next
to the model.

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
        fwd = ForwardPass()                             # the model's pass kind
        fwd.embed(tokens[done:])                        # the new tokens
        fwd.bind_state(ws, ws.geometry(done, len(tokens)))
        out = fwd.epilogue(lambda: reduce_argmax(intrinsics.logits()))  # traced once, runs on the device
        pipe.submit(fwd)
        done = len(tokens)
        tokens.append(await out)
    pipe.close()

    return {"text": model.decode(tokens[-max_tokens:])}
```

That is `examples/quickstart-py`. Every line above stands for device state:
`embed(list)` seeds the token and row channels, `ws.geometry(start, end)`
derives the six KV-geometry channels of one contiguous span (edit its fields
for anything else), and an `epilogue` that returns a tensor publishes it on a
channel the host `await`s. The explicit spellings — `Channel.from_(...,
dtype.u32)`, `KvGeometry(...)`, `ch.put(tensor)` inside the stage — are the
same objects and stay available; `examples/text-completion-py` uses them for
chunked prefill and a device loop-carried decode.

## What is here

| Module | Interface | Notes |
|---|---|---|
| `inferlet.eta` | `forward*`, `channel`, `working-set`, `pipeline` | The ETA authoring surface: `Tensor` + the op set, `Channel`, `WorkingSet`, `RsWorkingSet`, `ForwardPass`, `Pipeline` (`submit`, `run_ahead`), `prefill_chunks`, and `eta.diffusion` (the diffusion pass's `Mode` plus `entropy_bound_accept` / `stable_and_confident` / `linear_temperature`). A port of the Rust `eta-dsl`/`eta-ir` crates and `inferlet::eta`; the container bytes it emits are **byte-identical** to the Rust `inferlet` crate's for the same program (`tests/test_eta_goldens.py` pins them), so a Python and a Rust inferlet share the host's program cache. |
| `inferlet.model` / `inferlet.tokenizer` | `model`, `tokenizer` | The bound model's facts; `model` re-exports the tokenizer functions. |
| `inferlet.grammar` / `inferlet.mask` | `grammar` | JSON-Schema / regex / EBNF constraints and the packed-bitmask helpers a `masked_argmax` epilogue reads. |
| `inferlet.chat` / `inferlet.reasoning` / `inferlet.tools` | `chat`, `reasoning`, `tools` | The host's chat template, thinking-block and tool-call decoders. |
| `inferlet.media` | `media` | Image / video / audio spans for multimodal models. |
| `inferlet.session` | `session` | Client communication. |

Spelling differences from Rust: `Channel.from_([...], dtype.u32)` needs a
dtype for an integer sequence (the way a Rust literal needs a suffix);
scalars in arithmetic take the partner tensor's dtype; integer division is
`x // y` (ETA `div` truncates — `/` emits the same op, but `//` says so);
`<`/`<=`/`>`/`>=` compare elementwise, while `==` stays Python identity —
elementwise equality is `eq(a, b)`.


## Running

A Python inferlet is its source: there is no build. `pie run main.py` (or
`pie inferlet install main.py`, or `client.install_program(main_py, pie_toml)`)
hands the file to the server, which runs it under the Python language
component -- one wasm that bundles CPython, this package and the standard
library, built once by `language/build.sh` with stock
`componentize-py >= 0.25` against `crates/inferlet/wit`, and installed at
`~/.pie/languages/python.wasm`. The world is component-model async, so
`main` may be `async` and host reads are awaited (`await ch.take_host()`).

From a client, a function can go straight up: decorate it with
`pie_client.inferlet` and hand it to `PieClient.run` (see
`python/client`).

## Tests

```
PYTHONPATH=src python -m pytest tests
```

The stub `wit_world` in `tests/conftest.py` covers the unit tests; the
end-to-end twins live in `examples/*-py` in the pie repo
(`tests/examples/test_twins.py --attach ws://127.0.0.1:8080`).
