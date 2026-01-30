# Text Completion Example

A simple text completion Python inferlet.

## Setup (One-Time)

From the pie repository root:

```bash
cd sdk/python
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
uv pip install -e ../tools/bakery
```

## Build

From the pie repository root (with venv activated):

```bash
# Build
bakery build "$PWD/sdk/examples/python/text-completion" \
	-o "$PWD/text-completion.wasm"
```

## Run

Requires a running Pie engine:

```bash
# Submit inferlet (arguments go after --)
pie-cli submit text-completion.wasm -- --prompt "What is Python?"
```

## Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `prompt` | "Hello, world!" | User prompt |
| `max_tokens` | 256 | Maximum tokens to generate |
| `system` | "You are a helpful assistant." | System prompt |
