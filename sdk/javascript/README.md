# inferlet

JavaScript/TypeScript SDK for writing Pie inferlets.

## Quick Start

```js
import { Model, Context, Sampler, session, runtime } from 'inferlet';

const model = Model.load(runtime.models()[0]);
const ctx = Context.create(model);

ctx.system('You are a helpful assistant.');
ctx.user('Hello, world!');

const text = ctx.generateText({
  sampler: Sampler.topP(0.6, 0.95),
  maxTokens: 256,
});

session.send(text);
```

## Examples

Examples live under `sdk/examples/javascript/`:

| Example | Demonstrates |
|---------|-------------|
| **hello-world** | Streaming with `on().run()` callback pattern |
| **text-completion** | One-shot `generateText()` convenience |
| **beam-search** | Forked contexts, `stream.text()`, `using` auto-dispose |

Build and run an example:

```bash
# Build to WASM
bakery build sdk/examples/javascript/hello-world/ -o hello-world.wasm

# Run with dummy model
pie run --dummy --path hello-world.wasm --manifest sdk/examples/javascript/hello-world/Pie.toml
```

## API Reference

Import everything from `'inferlet'`:

```typescript
import {
  Model, Tokenizer,         // Model loading & tokenization
  Context,                   // Generation context (KV cache)
  TokenStream, EventStream,  // Streaming iterators
  Sampler,                   // Sampling strategies
  ForwardPass,               // Low-level forward pass
  Grammar, Matcher,          // Structured generation
  Adapter,                   // LoRA adapters
  session,                   // Client communication
  runtime,                   // Runtime info & spawning
  messaging,                 // Pub/sub messaging
  mcp,                       // MCP tool server client
} from 'inferlet';
```

### Context

The main entry point for chat-based generation:

```js
const ctx = Context.create(model);       // auto-named
const ctx = Context.create(model, 'my-ctx');  // explicit name

// Chat formatting (pie:instruct/chat)
ctx.system('You are helpful.');
ctx.user('Hello!');
ctx.assistant('Hi there!');   // history replay
ctx.cue();                    // start generation header
ctx.seal();                   // insert stop token

// Text-level buffer fill
ctx.fill('arbitrary text');          // encodes via tokenizer
ctx.fillTokens(new Uint32Array([1, 2, 3]));  // raw tokens

// Forking
const fork = ctx.fork();              // auto-named
const fork = ctx.fork('beam-0');      // explicit name

// Cleanup
ctx.destroy();          // manual
using _ = ctx;          // auto-dispose via TC39 `using`
```

### Generation

Three ways to generate:

```js
// 1. One-shot (simplest)
const text = ctx.generateText({
  sampler: Sampler.topP(0.6, 0.95),
  maxTokens: 256,
});

// 2. Streaming with callbacks
ctx.generate({
  sampler: Sampler.topP(0.6, 0.95),
  maxTokens: 256,
  decode: {},
})
.on('text', text => session.send(text))
.on('tool-call', (name, args) => { /* ... */ })
.run();

// 3. Iterator (advanced)
for (const event of ctx.generate({ ..., decode: {} })) {
  if (event.type === 'text') session.send(event.text);
}

// 4. Raw tokens (lowest level)
for (const tokens of ctx.generate({ sampler, maxTokens: 128 })) {
  // tokens: Uint32Array
}
const allText = stream.text();      // collect decoded text
const allTokens = stream.tokens();  // collect token batches
```

Stop tokens default to the model's stop tokens when omitted.

### Sampler

```js
Sampler.greedy()                          // temperature=0, top-k=1
Sampler.topP(temperature, topP)           // nucleus sampling
Sampler.topK(temperature, topK)           // top-k sampling
Sampler.minP(temperature, minP)           // min-p sampling
Sampler.multinomial(temperature, topK)    // multinomial
Sampler.topKTopP(temperature, topK, topP) // combined
Sampler.embedding()                       // embedding extraction
Sampler.dist(temperature, topK)           // full distribution
```

### Event Types

When using `decode: {}` in generate options, events have a `type` discriminant:

| `event.type` | Fields | Description |
|--------------|--------|-------------|
| `'text'` | `text` | Generated text chunk |
| `'thinking'` | `text` | Reasoning/thinking chunk |
| `'thinking-done'` | `text` | Reasoning complete (full text) |
| `'tool-call-start'` | — | Tool call detected |
| `'tool-call'` | `name`, `arguments` | Complete tool call |
| `'done'` | `text` | Generation finished (full text) |

Enable reasoning/tool-use detection via decoder options:

```js
ctx.generate({
  sampler, maxTokens: 256,
  decode: { reasoning: true, toolUse: true },
})
```

### Tool Use

```js
ctx.equipTools([toolSchemaJson1, toolSchemaJson2]);
// ... generate and detect tool calls ...
ctx.answerTool('get_weather', '{"temp": 72}');
```

### Structured Generation

```js
const grammar = Grammar.fromJsonSchema('{"type":"object",...}');
const matcher = new Matcher(grammar, model.tokenizer());

// Use in generation loop for constrained decoding
const mask = matcher.nextTokenLogitMask();
// pass mask as logitMask option to generate()
```

### Session & Runtime

```js
session.send('Hello');              // send text to client
const msg = session.receive();      // receive text from client
session.sendFile(data);             // send binary
const file = session.receiveFile(); // receive binary

runtime.version();                  // runtime version
runtime.models();                   // available model names
runtime.username();                 // invoking user
runtime.spawn('pkg@1.0.0', args);   // spawn another inferlet
```

### Messaging

```js
messaging.push('topic', 'message');       // queue push
const msg = messaging.pull('topic');      // queue pull (blocking)
messaging.broadcast('topic', 'message');  // broadcast

const sub = messaging.subscribe('topic');
for (const msg of sub) {                  // infinite iterator
  // process msg
  if (done) break;
}
sub.unsubscribe();
```

## Project Structure

```
sdk/javascript/
├── src/
│   ├── index.ts          # Public API exports
│   ├── context.ts        # Context, TokenStream, EventStream, Decoder
│   ├── model.ts          # Model, Tokenizer
│   ├── sampler.ts        # Sampler constructors
│   ├── forward.ts        # ForwardPass (low-level)
│   ├── grammar.ts        # Grammar, Matcher
│   ├── adapter.ts        # LoRA Adapter
│   ├── session.ts        # Client communication
│   ├── runtime.ts        # Runtime info
│   ├── messaging.ts      # Pub/sub messaging
│   ├── mcp.ts            # MCP client
│   ├── zo.ts             # Zeroth-order optimization
│   ├── _async.ts         # Internal WASI future utilities
│   └── bindings/         # Auto-generated WIT type declarations
├── package.json
├── tsconfig.json
└── README.md
```

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
