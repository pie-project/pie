# Constrained Decoding & Generation Behavior

This document covers behavior shared across the Rust, Python, and
JavaScript SDKs, plus the places where they diverge intentionally.

---

## Auto-flush + auto-cue (Python and JS only)

In Python and JS, calling `generate()` implicitly calls `cue()` before
returning the `Generator`. The generator's first step drains the buffer
through a forward pass — pending fills from `system()` / `user()` /
`assistant()` land in the KV cache and the model is cued to begin
generating, all on the first iteration.

**Python**: pass `auto_flush=False` to `generate(...)` to disable.

**JS**: pass `autoFlush: false` in the options object to `generate(...)`.

**Rust**: no auto-flush. Call `ctx.cue()` before `ctx.generate(...)` —
the first step drains the buffer:

```rust
ctx.user("hello").cue();
let text = ctx.generate(Sampler::Argmax).collect_text().await?;
```

This asymmetry tracks language conventions: Python and JS lean on
convenient defaults; Rust prefers explicit control flow.

> Don't follow `cue()` with an explicit `flush().await?` before
> `generate()` — flush will drain the cue tokens, leaving the
> generator's first forward pass with zero input. Either flush *or* let
> the generator do it.

---

## Pick a sampler

All three SDKs separate **`Sampler`** (token-producing) from **`Probe`**
(distribution-access). Picking is a Sampler call; reading the
distribution is a Probe call. They share a forward-pass slot but produce
different `Output` shapes.

| Use case | Rust | Python | JS / TS |
| --- | --- | --- | --- |
| Grammar / regex / JSON-schema constrained | `Sampler::Argmax` | `Sampler.argmax()` | `Sampler.argmax()` |
| Free-form text | `Sampler::TopP { temperature: 0.6, p: 0.95 }` | `Sampler.top_p(0.6, 0.95)` | `Sampler.topP(0.6, 0.95)` |
| Distribution-shape experimentation | `Sampler::TopK { … }`, `MinP { … }`, `TopKTopP { … }` | `Sampler.top_k(t, k)`, `min_p(t, p)`, `top_k_top_p(t, k, p)` | `Sampler.topK(t, k)`, `minP(t, p)`, `topKTopP(t, k, p)` |
| Read distribution | `Probe::Logits` / `Distribution { … }` / `Logprob(t)` / `Logprobs(ts)` / `Entropy` | `Logits()` / `Distribution(t, k)` / `Logprob(t)` / `Logprobs(ts)` / `Entropy()` | same as Python |

For grammar-constrained generation, argmax is almost always the right
choice — most masked positions have only a handful of valid tokens and
stochastic sampling rarely helps.

---

## `Schema` vs `Constraint` vs custom impl

| You want… | Rust | Python | JS / TS |
| --- | --- | --- | --- |
| JSON conforming to a schema string | `Schema::JsonSchema(s)` | `JsonSchema(s)` | `jsonSchema(s)` |
| Any valid JSON | `Schema::Json` | `AnyJson()` | `anyJson()` |
| Strings matching a regex | `Schema::Regex(p)` | `Regex(p)` | `regex(p)` |
| Custom EBNF grammar | `Schema::Ebnf(g)` | `Ebnf(g)` | `ebnf(g)` |
| Reuse a precompiled grammar across contexts | `Schema::Grammar(&g)` | implement the `Schema` Protocol | implement the `Schema` interface |
| Banned tokens, learned constraints, anything stateful that isn't a grammar | implement `Constraint` directly | implement `Constraint` directly | implement `Constraint` directly |

Schemas are declarative; the SDK compiles them into a stateful matcher
and drives it per generated token. Custom `Constraint` impls do the
driving themselves.

In all three SDKs, attach via `Generator::constrain(c)` (a `Constraint`
impl or a `Schema`). Multiple calls compose by AND-ing the per-step
BRLE masks.

### Typed JSON

| SDK | API | Validator |
| --- | --- | --- |
| Rust | `Generator::collect_json::<T>()` | `schemars::JsonSchema` derive on `T` |
| Python | `await g.collect_json(schema=schema_str)` returns `dict` / `list` / primitive | — |
| Python (typed) | `await g.collect_json(schema=schema_str, parse=my_validator)` | bring your own pure-Python validator |
| JS / TS | `await g.collectJson({ schema })` returns `unknown` | — |
| JS / TS (typed) | `await g.collectJson<T>({ schema, parse: MyZod.parse })` | bring your own (Zod, arktype, …) |

> **Native extensions don't load in WASM.** `componentize-py` bundles
> pure-Python packages but cannot load native (Rust/C) extensions today
> — pydantic v2 (`pydantic_core`), msgspec, orjson, and similar abort
> the inferlet at instantiation. For typed Python output, use
> `schema=` + a pure-Python validator passed via `parse=`.

In Rust, `collect_json::<T>()` adds a constraint built from `T`'s
schema. If you've already attached a constraint, the masks compose —
there's no runtime mutual-exclusion check.

---

## Composition

All three SDKs compose constraints by repeating `constrain(...)` —
masks are AND-ed across all attached constraints per step:

```rust
// Rust
ctx.generate(Sampler::Argmax)
    .constrain(Schema::Ebnf(grammar))?
    .constrain(BannedTokens::new(...))
    .collect_text().await?;
```

```python
# Python
await ctx.generate(
    Sampler.argmax(),
    constrain=[Ebnf(grammar), BannedTokens()],
).collect_text()
```

```typescript
// JS / TS
await ctx
  .generate(Sampler.argmax(), { constrain: [ebnf(grammar), bannedTokens] })
  .collectText();
```

Python and JS also accept a `logitMask` / `logit_mask` option for a
static BRLE mask applied every step — composes with `constrain` like
any other constraint. (Rust doesn't expose a separate knob; wrap the
static mask in a `Constraint` impl whose `step()` returns the same
BRLE every time.)

---

## Per-step control

All three SDKs expose a per-step handle for cases where the high-level
`collect_*` sugar is too coarse:

```rust
let mut g = ctx.generate(Sampler::Argmax).max_tokens(256);
while let Some(step) = g.next()? {
    let out = step.execute().await?;
    // out.tokens, plus probe handles registered earlier
    if my_stop_condition(&out.tokens) { break; }
}
```

```python
async for step in ctx.generate(Sampler.argmax(), max_tokens=256):
    out = await step.execute()
    if my_stop_condition(out.tokens): break
```

```typescript
for await (const step of ctx.generate(Sampler.argmax(), { maxTokens: 256 })) {
  const out = await step.execute();
  if (myStopCondition(out.tokens)) break;
}
```

`step` is a `GenStep` — a builder for the upcoming forward pass,
pre-populated with the generator's pending fills, configured sampler,
constraint mask, and any speculator drafts. Tweak before `execute()`:

- `step.probe(idx, probe)` — attach an extra probe for this iteration.
- `step.clear_sampler()` / `step.clearSampler()` — drop the
  auto-attached sampler. The execute call returns probe results with
  `tokens` empty; sample yourself off the distribution, then call
  `g.accept([chosen])` to register the pick (commits to KV, advances
  constraints, applies stop / max-tokens).

This is the path for watermarking and similar custom-sampling flows.

---

## Tool use

The `tools` module is opt-in across all three SDKs. None of the agent
inferlets in `inferlets/` use it — they hand-roll their own format. If
you want the model's native format:

```rust
use inferlet::{Context, GrammarConstraint, Sampler, tools};

let prefix = tools::equip_prefix(&model, &tool_schemas)?;
ctx.append(&prefix);
ctx.user("...").cue();

let mut g = ctx.generate(Sampler::Argmax);
if let Some(matcher) = tools::native_matcher(&model, &tool_schemas) {
    g = g.constrain(GrammarConstraint::new(matcher));
}

let mut decoder = tools::Decoder::new(&model);
while let Some(step) = g.next()? {
    let out = step.execute().await?;
    if let Event::Call { name, args } = decoder.feed(&out.tokens) {
        let result = run_tool(&name, &args).await;
        ctx.append(&tools::answer_prefix(&model, &name, &result));
        ctx.cue();
        break;
    }
}
```

Python and JS expose the same shape — see each SDK's `tools` submodule.

---

## Structured returns

`session.send` and the inferlet's `main` return value both accept
structured values across all three SDKs — the framework
JSON-serializes for you.

| SDK | `session.send` | `main` return |
| --- | --- | --- |
| Rust | always `&str` (byte-level WIT API) | `T: Serialize` via `#[inferlet::main]` |
| Python | `str \| dict \| list \| dataclass \| ...` | same — bakery wrapper auto-stringifies |
| JS / TS | `string \| object \| number \| boolean \| ...` | same |

So for the typed-JSON path you don't have to `JSON.stringify` (or
`json.dumps`) manually:

```typescript
const person = await ctx
  .generate(Sampler.argmax(), { maxTokens })
  .collectJson<Person>({ schema: PERSON_SCHEMA });
session.send(`Hello ${person.name}!`);  // string interpolation
return person;                          // auto-stringified
```

```python
person = await ctx.generate(
    Sampler.argmax(), max_tokens=max_tokens,
).collect_json(schema=PERSON_SCHEMA)
session.send(person)   # dict → JSON
return person          # same
```
