# inferlet (Python)

The Python SDK for writing Pie inferlets — the small programs that run next
to the model.

```python
from inferlet import model
from inferlet.eta import *

async def main(input: dict) -> dict:
    prompt = model.encode(input.get("prompt", "The capital of France is"))
    n = len(prompt)
    ws = WorkingSet()
    ws.reserve(max(-(-(n + 9) // kv_page_size()), 1))

    tokens = Channel.from_(prompt, dtype.i32)
    indptr_ch = Channel.from_([0, n], dtype.u32)
    kv_len = Channel.from_([n], dtype.u32)
    tok_out = Channel([1], dtype.i32)
    # ...pages / page_indptr / w_slot / w_off / positions as in
    # tests/inferlets/text-completion-py/main.py

    fwd = ForwardPass()                    # picks the model's pass kind
    fwd.embed(tokens, indptr_ch)
    fwd.bind_state(ws, KvGeometry(kv_len=kv_len, ...))

    @fwd.epilogue                          # traced once, runs on the device
    def _():
        tok_out.put(reshape(reduce_argmax(intrinsics.logits()), [1]))

    pipe = Pipeline()
    fwd.submit(pipe)
    first = await tok_out.take_scalar()
    pipe.close()
    return {"text": model.decode([first])}
```

## What is here

| Module | Interface | Notes |
|---|---|---|
| `inferlet.eta` | `forward*`, `channel`, `working-set`, `pipeline` | The ETA authoring surface: `Tensor` + the op set, `Channel`, `WorkingSet`, `RsWorkingSet`, `ForwardPass`, `Pipeline`, `run_ahead`, `prefill_chunks`. A port of the Rust `eta-dsl`/`eta-ir` crates and `inferlet::eta`; the container bytes it emits are **byte-identical** to the Rust SDK's for the same program (`tests/test_eta_goldens.py` pins them), so a Python and a Rust inferlet share the host's program cache. |
| `inferlet.model` / `inferlet.tokenizer` | `model`, `tokenizer` | The bound model's facts; `model` re-exports the tokenizer functions. |
| `inferlet.grammar` / `inferlet.mask` | `grammar` | JSON-Schema / regex / EBNF constraints and the packed-bitmask helpers a `masked_argmax` epilogue reads. |
| `inferlet.chat` / `inferlet.reasoning` / `inferlet.tools` | `chat`, `reasoning`, `tools` | The host's chat template, thinking-block and tool-call decoders. |
| `inferlet.media` | `media` | Image / video / audio spans for multimodal models. |
| `inferlet.session` | `session` | Client communication. |

Spelling differences from Rust: `Channel.from_([...], dtype.u32)` needs a
dtype for an integer sequence (the way a Rust literal needs a suffix);
scalars in arithmetic take the partner tensor's dtype; integer division is
`x // y` (ETA `div` truncates — `/` emits the same op, but `//` says so);
comparisons are `lt(a, b)`, `gt(a, b)`, ….


## Building

`bakery build <dir> -o out.wasm` (`sdk/inferlet/tools/bakery`) componentizes a
directory with `main.py` + `Pie.toml` using stock `componentize-py >= 0.25`
against `crates/inferlet/wit`. The world is component-model async, so
`main` may be `async` and host reads are awaited (`await ch.take_host()`).

## Tests

```
PYTHONPATH=src python -m pytest tests
```

The stub `wit_world` in `tests/conftest.py` covers the unit tests; the
end-to-end twins live in `tests/inferlets/*-py` in the pie repo
(`tests/inferlets/test_twins.py --attach ws://127.0.0.1:8080`).
