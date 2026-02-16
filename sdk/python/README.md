# inferlet

Python SDK for writing Pie inferlets.

## Quick Start

```python
from inferlet import Model, Context, Sampler, session, runtime

model = Model.load(runtime.models()[0])
ctx = Context(model)

ctx.system("You are a helpful assistant.")
ctx.user("Hello, world!")

text = ctx.generate_text(Sampler.top_p())
session.send(text)
```

## Examples

Examples live under `sdk/examples/python/`:

| Example | Demonstrates |
|---------|-------------|
| **hello-world** | One-shot `generate_text()` |
| **text-completion** | Streaming with `Event` pattern matching |
| **beam-search** | Forked contexts, `generate_text()` per beam |

Build and run an example:

```bash
# Build to WASM
bakery build sdk/examples/python/hello-world/ -o hello-world.wasm

# Run with dummy model
pie run --dummy --path hello-world.wasm --manifest sdk/examples/python/hello-world/Pie.toml
```

## API Reference

Import everything from `inferlet`:

```python
from inferlet import (
    Model, Tokenizer,         # Model loading & tokenization
    Context,                  # Generation context (KV cache)
    TokenStream, EventStream, # Streaming iterators
    Sampler,                  # Sampling strategies
    Event, Decoder,           # Event types & unified decoder
    ForwardPass,              # Low-level forward pass
    Grammar, Matcher,         # Structured generation
    Adapter,                  # LoRA adapters
    session,                  # Client communication
    runtime,                  # Runtime info & spawning
    messaging,                # Pub/sub messaging
    mcp,                      # MCP tool server client
    zo,                       # Zeroth-order optimization
)
```

### Context

The main entry point for chat-based generation:

```python
ctx = Context(model)                     # auto-named
ctx = Context(model, name="my-ctx")      # explicit name

# Chat formatting (pie:instruct/chat)
ctx.system("You are helpful.")
ctx.user("Hello!")
ctx.assistant("Hi there!")    # history replay
ctx.cue()                     # start generation header
ctx.seal()                    # insert stop token

# Text-level buffer fill
ctx.fill("arbitrary text")          # encodes via tokenizer
ctx.fill_tokens([1, 2, 3])         # raw tokens

# Forking
fork = ctx.fork()                   # auto-named
fork = ctx.fork("beam-0")          # explicit name

# Cleanup
ctx.release()             # manual
with Context(model) as ctx:  # auto-release via context manager
    ...
```

### Generation

Three ways to generate:

```python
# 1. One-shot (simplest)
text = ctx.generate_text(Sampler.top_p(), max_tokens=256)

# 2. Streaming events
for event in ctx.generate(Sampler.top_p(), decode=True):
    match event:
        case Event.Text(text=t):
            session.send(t)
        case Event.Done():
            break

# 3. Raw tokens (lowest level)
for tokens in ctx.generate(Sampler.top_p(), max_tokens=128, auto_flush=False):
    # tokens: list[int]
    pass
stream = ctx.generate(Sampler.top_p())
all_text = stream.text()        # collect decoded text
all_tokens = stream.tokens()    # collect token IDs
```

`generate()` auto-calls `cue()` + `flush()` by default. Pass `auto_flush=False` to skip.

### Sampler

```python
Sampler.greedy()                                    # deterministic
Sampler.top_p(temperature=0.6, top_p=0.95)          # nucleus sampling
Sampler.top_k(temperature=0.6, top_k=50)            # top-k sampling
Sampler.min_p(temperature=0.6, min_p=0.1)           # min-p sampling
Sampler.multinomial(temperature=1.0, seed=0)        # multinomial
Sampler.top_k_top_p(temperature=0.6, top_k=50, top_p=0.95)  # combined
Sampler.embedding()                                  # embedding extraction
Sampler.dist(temperature=1.0, seed=0)               # full distribution
```

### Event Types

When using `decode=True` in generate, events are dataclasses:

| Event class | Fields | Description |
|-------------|--------|-------------|
| `Event.Text` | `text` | Generated text chunk |
| `Event.Thinking` | `text` | Reasoning/thinking chunk |
| `Event.ThinkingDone` | `text` | Reasoning complete (full text) |
| `Event.ToolCallStart` | — | Tool call detected |
| `Event.ToolCall` | `name`, `arguments` | Complete tool call |
| `Event.Done` | `text` | Generation finished (full text) |

Enable reasoning/tool-use detection:

```python
for event in ctx.generate(Sampler.top_p(), decode=True, reasoning=True, tool_use=True):
    ...
```

### Tool Use

```python
ctx.equip_tools([tool_schema_json_1, tool_schema_json_2])
# ... generate and detect tool calls ...
ctx.answer_tool("get_weather", '{"temp": 72}')
```

### Structured Generation

```python
grammar = Grammar.from_json_schema('{"type":"object",...}')
matcher = Matcher(grammar, model.tokenizer())

# Use in generation loop for constrained decoding
mask = matcher.next_token_logit_mask()
# pass mask as logit_mask to generate()
```

### Session & Runtime

```python
session.send("Hello")               # send text to client
msg = session.receive()             # receive text (blocking)
session.send_file(data)             # send binary
file = session.receive_file()       # receive binary

runtime.version()                   # runtime version
runtime.models()                    # available model names
runtime.username()                  # invoking user
runtime.spawn("pkg@1.0.0", args)    # spawn another inferlet
```

### Messaging

```python
messaging.push("topic", "message")        # queue push
msg = messaging.pull("topic")             # queue pull (blocking)
messaging.broadcast("topic", "message")   # broadcast

with messaging.subscribe("topic") as sub:
    for msg in sub:                       # infinite iterator
        # process msg
        break
```

## Project Structure

```
sdk/python/
├── src/inferlet/
│   ├── __init__.py       # Public API exports
│   ├── context.py        # Context, TokenStream, EventStream, Decoder, Event
│   ├── model.py          # Model, Tokenizer
│   ├── sampler.py        # Sampler presets
│   ├── forward.py        # ForwardPass (low-level)
│   ├── grammar.py        # Grammar, Matcher
│   ├── adapter.py        # LoRA Adapter
│   ├── session.py        # Client communication
│   ├── runtime.py        # Runtime info
│   ├── messaging.py      # Pub/sub messaging
│   ├── mcp.py            # MCP client
│   ├── zo.py             # Zeroth-order optimization
│   └── _async.py         # Internal WASI future utilities
├── tests/
├── scripts/
├── pyproject.toml
└── README.md
```

## Development

```bash
cd sdk/python

# Create virtualenv and install
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
uv pip install -e ../tools/bakery

# Run tests
pytest tests/

# Regenerate WIT bindings (after runtime WIT changes)
python scripts/generate_bindings.py
```
