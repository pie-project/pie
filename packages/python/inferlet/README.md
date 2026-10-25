# inferlet (Python)

The Python library for writing Pie inferlets — the small programs that run next
to the model.

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
        out = fwd.epilogue(lambda: reduce_argmax(intrinsics.logits()))  # traced once, runs on the device
        pipe.submit(fwd)
        done = len(tokens)
        tokens.append(await out)
    pipe.close()

    return {"text": model.decode(tokens[-max_tokens:])}
```

That is `examples/quickstart-py`. Every line above stands for device state:
`embed(list)` seeds the token and row channels, `state([ws], ...)` says where
the rows land (by default after what the working set holds) and what freezes,
and an `epilogue` that returns a tensor publishes it on a channel the host
`await`s. The explicit spellings — `Channel.from_(..., dtype.u32)`,
`StateGeometry(slots=..., fold=...)`, `ch.put(tensor)` inside the stage — are the
same objects and stay available; `examples/text-completion-py` uses them for
a one-pass prefill and a device loop-carried decode.

A model receives two things per fire: **inputs** (device channels) and
**switches** (a value from a finite set that picks the model's path). One
`ForwardPass` holds every lane of one computation (a CFG denoise step's text
and image lanes of both branches, say); `switch(lane, name, value)`,
`input(lane, name, ch)`, `media(lane, spans)` and `group(lane, id)` address a
lane by index, `mask(rows)` is pass-wide, and epilogue intrinsics take the lane
(`velocity(lane, …)`, `hidden(lane, …)`, `pixels(lane, …)`). `model.switches()`
lists a model's switches and `model.lane(switches)` says what a lane with those
values binds: its inputs, whether it binds a working set (`state`), whether it
embeds tokens, and its readout. Generative latents are the VAE's own
`[t, h, w, C]`; patching and positions live in the model. A causal VAE
decodes frame by frame into a working set, one latent frame per slot, and the
runtime's `fresh` switch (cursor 0) picks its first-frame branch. A `prologue`
runs before the forward, an `epilogue` after it.

## What is here

| Module | Interface | Notes |
|---|---|---|
| `inferlet.eta` | `forward`, `channel`, `working-set`, `pipeline` | The ETA authoring surface: `Tensor` + the op set, `Channel`, `WorkingSet`, `StateGeometry`, `ForwardPass`, `Pipeline` (`submit`, `run_ahead`), `readout_spans`, and `eta.diffusion` (a block-diffusion text's samplers `entropy_bound_accept` / `stable_and_confident` / `linear_temperature`). A port of the Rust `eta-dsl`/`eta-ir` crates and `inferlet::eta`; the container bytes it emits are **byte-identical** to the Rust `inferlet` crate's for the same program (`tests/test_eta_goldens.py` pins them), so a Python and a Rust inferlet share the host's program cache. |
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
`packages/python/client`).

## Tests

```
PYTHONPATH=src python -m pytest tests
```

The stub `wit_world` in `tests/conftest.py` covers the unit tests; the
end-to-end twins live in `examples/*-py` in the pie repo
(`tests/examples/test_twins.py --attach ws://127.0.0.1:8080`).
