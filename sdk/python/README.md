# inferlet

Python SDK for writing Pie inferlets.

## Setup (One-Time)

From the pie repository root:

```bash
cd sdk/python
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
uv pip install -e ../tools/bakery

# Verify
componentize-py --version
bakery --help
```

## Building Python Inferlets

Activate the venv, then use Bakery to build inferlets:

```bash
# If needed, activate Python venv (e.g., sdk/python/.venv)
# source .venv/bin/activate

# Build inferlet
bakery build "$PWD/<input>" -o "$PWD/<output.wasm>"
```

### Example

```bash
# From pie root
bakery build "$PWD/sdk/examples/python/text-completion" \
    -o "$PWD/text-completion.wasm"
```

## Run (Requires Pie Engine)

When a Pie engine is running, submit the built inferlet:

```bash
pie-cli submit text-completion.wasm -- --prompt "What is Python?"
```

## Writing an Inferlet

Create `main.py`:

```python
from inferlet import Context, get_auto_model, get_arguments, send

args = get_arguments()
prompt = args.get("prompt", "Hello!")

model = get_auto_model()

with Context(model) as ctx:
    ctx.system("You are a helpful assistant.")
    ctx.user(prompt)

    for token in ctx.generate(stream=True):
        send(token, streaming=True)
```

### Beam Search Example

```python
from inferlet import Context, get_auto_model, get_arguments, set_return

args = get_arguments()
prompt = args.get("prompt", "Hello!")

model = get_auto_model()

with Context(model) as ctx:
    ctx.system("You are a helpful assistant.")
    ctx.user(prompt)

    # Generate with beam search for higher quality output
    result = ctx.generate_with_beam(beam_size=4, max_tokens=256)
    set_return(result)
```

## API Reference

### Runtime
- `get_version()` - Get Pie runtime version
- `get_instance_id()` - Get unique instance ID
- `get_arguments()` - Get CLI arguments as dict
- `set_return(value)` - Set return value

### Messaging
- `send(message, streaming=False)` - Send output
- `receive()` - Receive input
- `broadcast(topic, message)` - Broadcast to topic

### Model
- `get_auto_model()` - Get default model
- `get_model(service_id)` - Get specific model
- `get_all_models()` - List available models

### Context
- `system(content)` - Add system message
- `user(content)` - Add user message
- `assistant(content)` - Add assistant message
- `generate(...)` - Generate text (supports streaming)
- `generate_with_beam(beam_size, max_tokens, stop)` - Generate with beam search

## Limitations

Python inferlets run in WASM. These are **not available**:
- Network libraries (requests, httpx)
- Native extensions (numpy, pandas)
- Threading/multiprocessing
- File system (limited)

Check compatibility:

```bash
# With venv activated
python scripts/validate_imports.py <your-app>/
```
