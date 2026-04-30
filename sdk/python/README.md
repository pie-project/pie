# `inferlet` — Python SDK for Pie

Build Pie inferlets (WASM programs that run inside the Pie runtime) from
Python. Uses `componentize-py` under the hood; the surface here is what
your `main.py` actually imports.

## At a glance

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

## Three-layer surface

The SDK is layered. Each layer is usable independently — pick the one
that fits your control budget.

### `Context`

Stateful wrapper over the host's KV cache. Buffers tokens via chat
fillers (`system / user / assistant / cue / seal`) or raw `append`,
drains via `flush()` or by handing the buffer to a `Forward` /
`Generator`. Lifecycle: `Context(model) / fork() / save(name) /
open(model, name) / take(model, name) / release()`.

```python
ctx = Context(model)
ctx.system("...").user("...").cue()
with Context(model) as ctx:
    ...  # auto-released on exit
```

### `Forward` — single forward-pass primitive

`ctx.forward()` returns a `Forward` builder with auto page management.
Attach inputs, samplers, probes, masks, then `await fwd.execute()`:

```python
from inferlet import Sampler, Distribution, Logits

fwd = ctx.forward()
fwd.input(prompt_tokens)
h_token = fwd.sample([len(prompt_tokens) - 1], Sampler.argmax())
h_logits = fwd.probe(0, Logits())
out = await fwd.execute()

token = out.token(h_token)
logits = out.logits(h_logits)
```

`Forward` covers prefill, scoring, custom decode loops, and anywhere the
generator loop is too high-level. Page reservation, position derivation,
and post-execute commit happen automatically. Page-level rollback (e.g.
for speculation) goes through `ctx.truncate(n)`.

### `Sampler` vs probes

* **`Sampler`** produces a token: `argmax()`, `top_p(t, p)`,
  `top_k(t, k)`, `min_p(t, p)`, `top_k_top_p(t, k, p)`,
  `multinomial(t, draws)`.
* **Probes** read the distribution: `Logits()`, `Distribution(t, k)`,
  `Logprob(token)`, `Logprobs(tokens)`, `Entropy()`. Each is a frozen
  dataclass that doubles as both spec and runtime marker.

Both attach to the same forward-pass slot, so one `Forward` can sample
at some positions and probe at others.

### `Generator` — multi-step state machine

`ctx.generate(sampler, ...)` returns a `Generator`. Configure with
kwargs (Pythonic) or chain methods. Iterate with `async for`:

```python
# Common case: one-shot
text = await ctx.generate(Sampler.argmax(), max_tokens=256).collect_text()

# Per-step (custom sampling, e.g. watermarking)
g = ctx.generate(Sampler.argmax(), max_tokens=256, auto_flush=False)
async for step in g:
    step.clear_sampler()
    h = step.probe(0, Distribution(temperature=1.0, k=0))
    out = await step.execute()
    chosen = my_watermark.sample(out.distribution(h))
    g.accept([chosen])
```

Auto-flush is on by default — `generate(...)` appends `cue()` and uses
chat-template stop tokens automatically. Pass `auto_flush=False` to
inspect the buffer before generation.

### Decoders — independent, with `Idle`

```python
from inferlet import chat, reasoning

chat_dec = chat.Decoder(model)
think    = reasoning.Decoder(model)

async for step in g:
    out = await step.execute()
    match chat_dec.feed(out.tokens):
        case chat.Event.Delta(text=t): print(t, end="")
        case chat.Event.Done(text=full): break
        case _: pass
    match think.feed(out.tokens):
        case reasoning.Event.Delta(text=t): print(t, file=sys.stderr, end="")
        case _: pass
```

Per `feed()`, exactly one event fires. `Event.Idle` is the no-op signal
when the batch didn't cross a boundary worth surfacing.

### `tools` — opt-in tool helpers

Hand-rolling the agent loop is the default. For models with a native
tool-call template:

```python
from inferlet import GrammarConstraint, tools

ctx.append(tools.equip_prefix(model, schemas))
ctx.user("...").cue()

g = ctx.generate(Sampler.argmax())
matcher = tools.native_matcher(model, schemas)
if matcher is not None:
    g = g.constrain(GrammarConstraint(matcher))

tool_dec = tools.Decoder(model)
async for step in g:
    out = await step.execute()
    match tool_dec.feed(out.tokens):
        case tools.Event.Call(name=n, args=a):
            result = await run_tool(n, a)
            ctx.append(tools.answer_prefix(model, n, result))
            ctx.cue()
            break
        case _: pass
```

### Schema as Protocol

`Schema` is a `runtime_checkable` Protocol — any class with a
`build_constraint(model)` method satisfies it. Built-in implementors are
frozen dataclasses; user code plugs in by adding the method:

```python
from inferlet import GrammarConstraint, Grammar, Schema

class MyLark:
    def __init__(self, source): self.source = source
    def build_constraint(self, model):
        g = compile_lark_to_pie(self.source)
        return GrammarConstraint.from_grammar(g, model)

assert isinstance(MyLark("..."), Schema)  # duck-typed

await ctx.generate(Sampler.argmax(), constrain=MyLark(grammar)).collect_text()
```

### `collect_json` — schema string and (optional) custom validator

```python
# Untyped — returns dict / list / primitive
data = await g.collect_json(schema=schema_str)

# Bring-your-own validator — gets the generated text post-grammar.
city = await ctx.generate(Sampler.argmax()).collect_json(
    schema=schema_str,
    parse=my_validator,
)
```

> **Native extensions don't load in WASM.** `componentize-py` bundles
> pure-Python packages but cannot load native (Rust/C) extensions today,
> so pydantic v2 (`pydantic_core`), msgspec, orjson, numpy, etc. abort
> the inferlet at instantiation. For typed output, use
> ``schema=schema_str`` with a pure-Python validator passed via
> ``parse=``.

### Idle — RAII via `with`

```python
with ctx.idle():
    result = await http_get(url)
# bid restored on exit
```

Drops the bid to zero for the duration; restores the truthful generation
bid on context-manager exit. Use across external waits (HTTP, tool
calls, anything off-GPU). On uncontended devices it's a no-op cost-wise.

## Module layout

```
inferlet/
    __init__.py     — top-level: Context, Model, Sampler, runtime, Schema, etc.
    context.py      — Context class
    forward.py      — Forward primitive + SampleHandle / ProbeHandle / Output
    generation.py   — Generator + GenStep
    sample.py       — Sampler + probe dataclasses
    chat.py         — chat fillers + Decoder + Event
    reasoning.py    — Decoder + Event
    tools.py        — equip_prefix / answer_prefix / native_matcher / Decoder / Event
    grammar.py      — Schema Protocol + JsonSchema/AnyJson/Regex/Ebnf + Grammar/Matcher/GrammarConstraint
    spec.py         — Speculator Protocol
    scheduling.py   — market accessors (balance / rent / dividend / latency / price)
    model.py        — Model + Tokenizer
    adapter.py      — Adapter
    runtime.py / messaging.py / session.py / mcp.py / zo.py — host services
```

See [`sdk/CONSTRAINED_DECODING.md`](../CONSTRAINED_DECODING.md) for the
constrained-decoding details and the SDK divergence between Rust /
Python / JS.

## Differences from the Rust SDK (intentional)

| Concept | Rust | Python |
| --- | --- | --- |
| Async terminator | `.execute().await?` | `await fwd.execute()` |
| Generator iteration | `while let Some(step) = gen.next()? { … }` | `async for step in gen` |
| RAII bid yielding | `let _idle = ctx.idle();` | `with ctx.idle():` |
| Schema | trait + structs | Protocol + dataclasses |
| Probe spec | marker structs | frozen dataclasses |
| Decoder events | enum + match | dataclasses + match |
| Auto-cue / auto-flush | explicit | on by default (Python convention) |
| Typed JSON | `collect_json::<T>()` via `schemars` | `collect_json(schema=..., parse=...)` |
| Constructors | `Sampler::TopP { … }` or `top_p(t, p)` | `Sampler.top_p(t, p)` |

The conceptual layers — `Forward` → `Generator` → independent decoders;
`Sampler` vs probe; opt-in tools; extensible Schema — are identical
across both languages.
