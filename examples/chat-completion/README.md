# chat-completion

Reference chat completion harness with ETA prefill, device-carried decode, and
in-graph top-p plus temperature sampling. This is not an algorithm inferlet.

## Purpose

This crate is the canonical chat path for the test workspace. It templates a
system and user message, pre-fills the prompt once, then streams sampled tokens
through the chat decoder while the next token remains device-carried into the
following pass.

Other inferlets use it as a reference for output-equivalence checks, and
`tests/examples/test_curated.py` asserts on its content — the continuation of
"The capital of France is" must mention France or Paris — not just liveness.

## What it validates

The harness validates the production chat skeleton:

- chat-template tokenization of system, user, and assistant cue;
- one prefill pass followed by a 1-wide decode loop;
- nucleus sampling at positive temperature and exact greedy decoding at
  `temperature = 0`;
- host-side streaming and stop-token handling.

## Parameters

| Name | Type | Default | Meaning |
| --- | --- | --- | --- |
| `prompt` | string | — | User prompt to complete |
| `max_tokens` | int | `256` | Maximum number of tokens to generate |
| `system` | string | `"You are a helpful, respectful and honest assistant."` | System message for the assistant |
| `temperature` | float | `0.6` | Sampling temperature; `0.0` is exact greedy decoding |
| `top_p` | float | `0.95` | Nucleus sampling threshold, `0 < top_p <= 1` |

## Implementation notes

The prefill epilogue samples the first token and mirrors it to the host. The
decode pass then carries the token and RNG state through device channels; where
each token's row lands is the runtime's. The host drains tokens only for
streaming text and stop checks; it does not feed the sampled token back into the
next pass.

## Run

```bash
cargo build --release --target wasm32-wasip2
python tests/examples/run_all.py --engine cuda_native --model <model-path>
```

Details are in the `//!` header of [`src/lib.rs`](src/lib.rs).
