# `inferlet` — Rust SDK for Pie

Build inferlets (WASM programs that run inside Pie's runtime) with a
typed Rust API for forward passes, generation, decoding, and grammar
constraints.

## At a glance

```rust
use inferlet::{Context, Result, model::Model, runtime, sample::Sampler};

#[inferlet::main]
async fn main(_: ()) -> Result<String> {
    let model = Model::load(runtime::models().first().ok_or("no models")?)?;
    let mut ctx = Context::new(&model)?;

    ctx.system("You are helpful.")
        .user("What's 2 + 2?")
        .cue();

    let text = ctx
        .generate(Sampler::Argmax)
        .max_tokens(64)
        .collect_text()
        .await?;

    Ok(text)
}
```

## Architecture

The SDK is layered. Each layer is usable independently — pick the one
that fits your control budget.

### `Context`

Stateful wrapper over the host's KV-cache resource. Buffers tokens via
`system / user / assistant / cue / seal / append`, drains via `flush()`
or by handing the buffer to a `Forward` / `Generator`. Lifecycle:
`new / open / take / fork / save / snapshot / destroy / suspend`.
Market knobs: `bid / yield_bid / bid_horizon`.

### `Forward` — single forward pass

`ctx.forward()` returns a builder for one forward pass with auto page
management. Attach inputs, samplers, probes, masks, then `.execute()`:

```rust
let mut pass = ctx.forward();
pass.input(&prompt_tokens);
let h = pass.sample(&[input.len() as u32 - 1], Sampler::Argmax);
let logit_h = pass.probe(0, sample::Logits);
let out = pass.execute().await?;

let token = out.token(h).unwrap();
let logits = out.logits(logit_h).unwrap();
```

`Forward` covers prefill, scoring, custom decode loops, and anywhere the
generator loop is too high-level. Page reservation, position derivation,
and post-execute commit happen automatically. Page-level rollback (e.g.
for speculation) goes through `ctx.truncate(n)`.

### `Sampler` vs `Probe`

- `Sampler` produces a token: `Argmax`, `TopP { temperature, p }`,
  `TopK { … }`, `MinP { … }`, `TopKTopP { … }`, `Multinomial { … }`.
- `Probe` reads the distribution: `Logits`, `Distribution { temperature, k }`,
  `Logprob(t)`, `Logprobs(ts)`, `Entropy`. Each is a marker struct that
  doubles as the spec, with a phantom-typed `ProbeHandle<P>` for typed
  output access.

Both attach to the same forward-pass slot, so one `Forward` can sample at
some positions and probe at others.

### `Generator` — multi-step state machine

`ctx.generate(sampler)` returns a `Generator`. Configure with builder
methods (`max_tokens`, `stop`, `constrain`, `constrain_with`,
`speculator`, `system_speculation`, `adapter`, `zo_seed`, `horizon`,
`probe_each_step`), then either consume with `collect_*` sugar or drive
per-step.

```rust
// One-shot
ctx.generate(Sampler::Argmax)
    .max_tokens(256)
    .constrain_with(Schema::JsonSchema(SCHEMA))?
    .collect_text().await?;

// Per-step (custom sampling)
let mut g = ctx.generate(Sampler::Argmax).max_tokens(256);
while let Some(mut step) = g.next()? {
    step.clear_sampler();
    let d = step.probe(0, sample::Distribution { temperature: 1.0, k: 0 });
    let out = step.execute().await?;
    let chosen = my_watermark.sample(out.distribution(d).unwrap());
    g.accept(&[chosen]);
}
```

### `chat`, `reasoning` — independent decoders

`chat::Decoder` and `reasoning::Decoder` each wrap one host decoder
resource. Compose by hand — feed each step's tokens to whichever
decoders matter and interleave their events:

```rust
let mut chat_dec  = chat::Decoder::new(&model);
let mut think_dec = reasoning::Decoder::new(&model);

while let Some(step) = g.next()? {
    let out = step.execute().await?;
    match think_dec.feed(&out.tokens)? {
        reasoning::Event::Delta(s) => eprint!("{s}"),
        _ => {}
    }
    match chat_dec.feed(&out.tokens)? {
        chat::Event::Delta(s) => print!("{s}"),
        chat::Event::Done(_) => break,
        _ => {}
    }
}
```

### `tools` — opt-in tool helpers

Hand-rolling agent loops is the default — see `inferlets/agent-react`
and `inferlets/agent-codeact`. For models with a native tool-call
template, `inferlet::tools` exposes `equip_prefix`, `answer_prefix`,
`native_grammar`, `native_matcher`, `parse_call`, and a streaming
`Decoder`. None of these are required.

### `spec` — speculative decoding

`Speculator` is a single trait. Use `Generator::system_speculation()` to
let the host drive drafts (the runtime returns next-iter draft chains
in the forward-pass output channel), or `Generator::speculator(s)` to
plug in a custom drafter.

### `constrain` — grammar / mask constraints

`Schema` (declarative: `JsonSchema`, `Json`, `Regex`, `Ebnf`, `Grammar`)
or any `Constrain` impl. Multiple `.constrain(...)` /
`.constrain_with(...)` calls compose by AND-ing per-step BRLE masks.
`GrammarConstraint` is the lower-level handle for callers that want to
keep a constraint instance around.

## Crate layout

```
inferlet
├── prelude              — common imports
├── Context, Model
├── runtime, scheduling, messaging
├── forward              — Forward primitive (single forward pass)
├── generation           — Generator + GenStep
├── sample               — Sampler, Probe (+ markers)
├── chat                 — chat fillers + Decoder + Event
├── reasoning            — Decoder + Event
├── tools                — opt-in tool helpers + Decoder + Event
├── spec                 — Speculator
├── Schema (trait), JsonSchema / AnyJson / Regex / Ebnf / Grammar
├── Constrain, GrammarConstraint, Matcher
└── inference            — raw WIT re-exports (escape hatch)
```

See [`CONSTRAINED_DECODING.md`](../../CONSTRAINED_DECODING.md) for
constrained-decoding details and the SDK divergence between Rust /
Python / JS.

## Examples

The full set of inferlets lives in `inferlets/`. Notable examples for
the SDK surface:

| Example | Demonstrates |
| --- | --- |
| `text-completion` | One-shot chat with reasoning fan-out |
| `text-completion-spec` | Host-driven speculative decoding |
| `cacheback-decoding` | Custom speculator + draft context |
| `constrained-decoding` | EBNF-constrained generation |
| `json-schema-validation` | JSON-schema-constrained generation |
| `template-generation` | Schema → struct → template render |
| `output-validation` | Score multiple candidates via `Probe::Distribution` |
| `watermarking` | User-sampling via `clear_sampler` + `gen.accept` |
| `raw-logits-demo` | Sampler vs probe overhead via `Forward` |
| `sampler-suite` | All sampler / probe shapes in one forward pass |
| `attention-sink` | Per-step attention masks via `Forward::attention_mask` |
| `windowed-attention` | Sliding-window attention masks |
| `tree-of-thought` | Forked contexts + concurrent generation |
| `best-of-n` | Forked contexts + diversity ranking |
| `agent-react` | Hand-rolled ReAct loop |
| `agent-codeact` | Hand-rolled CodeAct loop |
