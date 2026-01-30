# Beam Search Example

Demonstrates beam search decoding for higher-quality text generation.

## What is Beam Search?

Beam search is a decoding strategy that maintains multiple candidate sequences (beams) at each step, selecting the most likely overall sequences rather than greedily choosing the best token at each position.

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
bakery build "$PWD/sdk/examples/python/beam-search" \
    -o "$PWD/beam-search.wasm"
```

## Run

Requires a running Pie engine:

```bash
# Submit with default settings (beam_size=4)
pie-cli submit beam-search.wasm -- --prompt "What is 2 + 2?"

# Submit with custom beam size
pie-cli submit beam-search.wasm -- \
    --prompt "What is 2 + 2?" \
    --beam_size 8 \
    --max_tokens 256
```

## Arguments

- `prompt`: The input prompt (default: "Explain the LLM decoding process ELI5.")
- `beam_size`: Number of candidate sequences to maintain (default: 4)
- `max_tokens`: Maximum tokens to generate (default: 128)
- `system`: System prompt (default: "You are a helpful, respectful and honest assistant.")
