# chat-completion

Reference chat completion harness with PTIR prefill, device-carried decode, and
in-graph top-p plus temperature sampling. This is not an algorithm inferlet.

## Purpose

This crate is the canonical chat path for the test workspace. It templates a
system and user message, pre-fills the prompt once, then streams sampled tokens
through the chat decoder while the next token remains device-carried into the
following pass.

Other inferlets use it as a reference for output-equivalence checks. It is also
covered by `tests/gpu/tests/cuda_chat_completion_e2e.rs`, which asserts on output
content — the continuation of "The capital of France is" must contain "Paris" —
not just liveness. That ignored e2e was the test that exposed a silent decode
KV-corruption bug affecting six inferlets; see
[`10-implementation-faithfulness-audit.md`](../../../inference-time-algorithms/10-implementation-faithfulness-audit.md).

## What it validates

The harness validates the production chat skeleton:

- chat-template tokenization of system, user, and assistant cue;
- one N-wide prefill fire followed by a 1-wide decode loop;
- page-CSR geometry that tracks the true KV length, not the reserved pool;
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
decode pass then carries token, position, KV length, mask, page CSR and RNG state
through device channels. The host drains tokens only for streaming text and stop
checks; it does not feed the sampled token back into the next pass.

## Run

```bash
cargo build --release --target wasm32-wasip2
python tests/inferlets/run_all.py --driver cuda_native --model <model-path>
```

Details are in the `//!` header of [`src/lib.rs`](src/lib.rs).
