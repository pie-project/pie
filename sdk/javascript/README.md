# `inferlet` — JavaScript / TypeScript SDK for Pie

Build Pie inferlets (WASM programs that run inside the Pie runtime) from
TypeScript or JavaScript. Bundled to a component via `jco`; the surface
here is what your `index.ts` actually imports.

## At a glance

```typescript
import { Context, Model, Sampler, runtime, session } from 'inferlet';

export async function main(input: { prompt: string }) {
  const model = Model.load(runtime.models()[0]);
  using ctx = new Context(model);

  ctx.system('You are helpful.').user(input.prompt);

  return await ctx
    .generate(Sampler.topP(0.6, 0.95), { maxTokens: 256 })
    .collectText();
}
```

## Three-layer surface

The SDK is layered. Each layer is usable independently — pick the one
that fits your control budget.

### `Context`

Stateful wrapper over the host's KV cache. Buffers tokens via chat
fillers (`system / user / assistant / cue / seal`) or raw `append`,
drains via `flush()` or by handing the buffer to a `Forward` /
`Generator`. Lifecycle: `new Context(model) / fork() / save(name) /
Context.open(model, name) / Context.take(model, name) / release()`.

```typescript
const ctx = new Context(model);
ctx.system('...').user('...').cue();

// Auto-released via TC39 `using`:
{
  using ctx = new Context(model);
  // ... ctx[Symbol.dispose]() called on scope exit
}
```

### `Forward` — single forward-pass primitive

`ctx.forward()` returns a `Forward` builder with auto page management.
Attach inputs, samplers, probes, masks, then `await fwd.execute()`:

```typescript
import { Sampler, Distribution, Logits } from 'inferlet';

const fwd = ctx.forward();
fwd.input(promptTokens);
const hToken  = fwd.sample([promptTokens.length - 1], Sampler.argmax());
const hLogits = fwd.probe(0, Logits());
const out = await fwd.execute();

const token  = out.token(hToken);
const logits = out.logits(hLogits);
```

`Forward` covers prefill, scoring, custom decode loops, and anywhere the
generator loop is too high-level. Page reservation, position derivation,
and post-execute commit happen automatically. Page-level rollback (e.g.
for speculation) goes through `ctx.truncate(n)`.

### `Sampler` vs probes

* **`Sampler`** produces a token: `argmax()`, `topP(t, p)`, `topK(t, k)`,
  `minP(t, p)`, `topKTopP(t, k, p)`, `multinomial(t, draws)`.
* **Probes** read the distribution: `Logits()`, `Distribution(t, k)`,
  `Logprob(token)`, `Logprobs(tokens)`, `Entropy()`. Each is a tagged
  interface that doubles as both spec and runtime marker; the typed
  handle returned by `forward.probe(...)` selects the matching
  `output.*` accessor.

Both attach to the same forward-pass slot, so one `Forward` can sample
at some positions and probe at others.

### `Generator` — multi-step state machine

`ctx.generate(sampler, options)` returns a `Generator`. Configure with
the options object (idiomatic for TS) or chain methods. Iterate with
`for await`:

```typescript
// Common case: one-shot
const text = await ctx
  .generate(Sampler.argmax(), { maxTokens: 256 })
  .collectText();

// Per-step (custom sampling, e.g. watermarking)
const g = ctx.generate(Sampler.argmax(), {
  maxTokens: 256,
  autoFlush: false,
});
for await (const step of g) {
  const h = step.clearSampler().probe(0, Distribution(1.0, 0));
  const out = await step.execute();
  const chosen = myWatermark.sample(out.distribution(h)!);
  g.accept([chosen]);
}
```

Auto-flush is on by default — `generate(...)` appends `cue()` and uses
chat-template stop tokens automatically. Pass `autoFlush: false` to
inspect the buffer before generation.

### Decoders — independent, with `Idle`

```typescript
import { chat, reasoning } from 'inferlet';

const chatDec  = new chat.Decoder(model);
const thinkDec = new reasoning.Decoder(model);

for await (const step of g) {
  const out = await step.execute();

  const c = chatDec.feed(out.tokens);
  if (c.type === 'delta') process.stdout.write(c.text);
  else if (c.type === 'done') break;

  const r = thinkDec.feed(out.tokens);
  if (r.type === 'delta') process.stderr.write(r.text);
}
```

Per `feed()`, exactly one event fires. `event.type === 'idle'` is the
no-op signal when the batch didn't cross a boundary worth surfacing.

### `tools` — opt-in tool helpers

Hand-rolling the agent loop is the default. For models with a native
tool-call template:

```typescript
import { tools, GrammarConstraint } from 'inferlet';

ctx.append(tools.equipPrefix(model, schemas));
ctx.user('...').cue();

let g = ctx.generate(Sampler.argmax(), { maxTokens: 256 });
const matcher = tools.nativeMatcher(model, schemas);
if (matcher !== undefined) g = g.constrain(new GrammarConstraint(matcher));

const toolDec = new tools.Decoder(model);
for await (const step of g) {
  const out = await step.execute();
  const ev = toolDec.feed(out.tokens);
  if (ev.type === 'call') {
    const result = await runTool(ev.name, ev.args);
    ctx.append(tools.answerPrefix(model, ev.name, result));
    ctx.cue();
    break;
  }
}
```

### Schema as duck-typed interface

`Schema` is a TypeScript interface — any object with a
`buildConstraint(model)` method satisfies it. Built-in implementors are
the factory functions `jsonSchema(...)`, `anyJson()`, `regex(...)`,
`ebnf(...)`; user code plugs in by adding the method to its own class:

```typescript
import { GrammarConstraint, Grammar, type Schema } from 'inferlet';

class MyLark implements Schema {
  constructor(public readonly source: string) {}
  buildConstraint(model: Model) {
    const g = compileLarkToPie(this.source);
    return GrammarConstraint.fromGrammar(g, model);
  }
}

await ctx
  .generate(Sampler.argmax(), { constrain: new MyLark(grammar) })
  .collectText();
```

### `collectJson` — schema string, or custom validator (e.g. Zod)

```typescript
// Untyped — returns unknown
const data = await g.collectJson({ schema: schemaStr });

// Typed via TS generic
const data = await g.collectJson<{ name: string; age: number }>({
  schema: schemaStr,
});

// Bring-your-own validator (Zod, arktype, ...):
import { z } from 'zod';
import { zodToJsonSchema } from 'zod-to-json-schema';

const Person = z.object({ name: z.string(), age: z.number() });
const person = await ctx.generate(Sampler.argmax(), { maxTokens: 512 })
  .collectJson({
    schema: JSON.stringify(zodToJsonSchema(Person)),
    parse: Person.parse,
  });
// person: z.infer<typeof Person>
```

### Idle — RAII via TC39 `using`

```typescript
{
  using _ = ctx.idle();
  const result = await fetch(url);
}
// bid restored when `_` goes out of scope
```

Drops the bid to zero for the duration; restores the truthful generation
bid on `Symbol.dispose`. Use across external waits (HTTP, tool calls,
anything off-GPU). On uncontended devices it's a no-op cost-wise.

## Module layout

```
sdk/javascript/src/
    index.ts        — top-level exports
    context.ts      — Context class
    forward.ts      — Forward primitive + SampleHandle / ProbeHandle / Output
    generation.ts   — Generator + GenStep
    sample.ts       — Sampler + Probe constructors
    chat.ts         — chat fillers + Decoder + Event
    reasoning.ts    — Decoder + Event
    tools.ts        — equipPrefix / answerPrefix / nativeMatcher / Decoder / Event
    grammar.ts      — Schema interface + jsonSchema/anyJson/regex/ebnf + Grammar/Matcher/GrammarConstraint
    spec.ts         — Speculator interface
    scheduling.ts   — market accessors (balance / rent / dividend / latency / price)
    model.ts        — Model + Tokenizer
    adapter.ts      — Adapter
    runtime.ts / messaging.ts / session.ts / mcp.ts / zo.ts — host services
    bindings/       — auto-generated WIT type declarations
```

See [`sdk/CONSTRAINED_DECODING.md`](../CONSTRAINED_DECODING.md) for the
constrained-decoding details and the SDK divergence between Rust /
Python / JS.

## Differences from the Rust SDK (intentional)

| Concept | Rust | TypeScript |
| --- | --- | --- |
| Async terminator | `.execute().await?` | `await fwd.execute()` |
| Generator iteration | `while let Some(step) = gen.next()? { … }` | `for await (const step of gen)` |
| RAII bid yielding | `let _idle = ctx.idle();` | `using _ = ctx.idle();` |
| Schema | trait + structs | interface + factory functions |
| Probe spec | marker structs | tagged interfaces (constructors + types) |
| Decoder events | enum + match | discriminated union (`event.type`) |
| Auto-cue / auto-flush | explicit | on by default (`autoFlush: false` to disable) |
| Typed JSON | `collect_json::<T>()` via `schemars` | `collectJson<T>({ schema, parse })` |
| Constructors | `Sampler::TopP { … }` | `Sampler.topP(t, p)` |

The conceptual layers — `Forward` → `Generator` → independent decoders;
`Sampler` vs probe; opt-in tools; extensible Schema — are identical
across all three languages.

## Development

```bash
cd sdk/javascript

# Install dependencies
npm install

# Type-check
npx tsc --noEmit

# Regenerate WIT bindings (after runtime WIT changes)
npm run generate-bindings

# Build
npm run build
```
