# cacheback-speculative-decoding

Cache-based speculative decoding: draft a short continuation by looking for a
matching suffix in the committed token history, then let the target model verify
the whole draft before any token becomes visible.

## Source

Ma, Gim and Zhong, ***Cacheback: Speculative Decoding With Nothing But Cache***
(EMNLP) — <https://arxiv.org/abs/2511.21699>. Implements the cache-drafting
idea, but as a single in-context prompt-lookup cache rather than the full
dual-table candidate tree.

**Faithfulness: Partial.** The verifier is faithful greedy speculative decoding;
the drafter is deliberately reduced to one longest previous suffix match.

## What it does

The drafter searches the already committed token sequence for the longest suffix
that appeared earlier. If it finds one, the tokens that followed the previous
occurrence become the draft. A target-model verification pass reads logits for
each draft position plus one bonus position, accepts the matching prefix, and
commits the target correction at the first mismatch.

Speculation is therefore a latency optimization only. With `draft_length = 0`
the draft is always empty, the same verifier reduces to one-row greedy decoding,
and the output token sequence should be identical.

## The rule

```
suffix = longest committed[-w:] that occurred earlier, w <= max_ngram
draft  = tokens after that earlier occurrence, capped at draft_length
target = greedy target tokens over committed + draft
keep   = matching draft prefix
emit   = keep + first target correction, or keep + bonus if all draft tokens match
```

## Parameters

| Name | Type | Default | Meaning |
| --- | --- | --- | --- |
| `prompt` | string | `"Repeat this pattern: red green blue, red green blue, red green"` | Prompt to send to the model |
| `max_tokens` | int | `256` | Number of generated tokens |
| `draft_length` | int | `4` | Maximum number of cache tokens to draft per verification step; `0` is the greedy control |
| `max_ngram` | int | `8` | Longest committed suffix width to search for in the token history |

## Implementation notes

Each verification window rebuilds the target KV from the committed sequence plus
the draft. That is slower than rolling back speculative KV, but it makes the
correctness property explicit: rejected draft state cannot leak into later
steps. The returned JSON includes the exact generated token ids plus
`verification_steps`, `drafted`, `accepted`, and `acceptance_rate`, so the
`draft_length = 0` control can compare token sequences directly.

## Run

```bash
cargo build --release --target wasm32-wasip2
python tests/inferlets/run_all.py --engine cuda_native --model <model-path>
```

Details are in the `//!` header of [`src/lib.rs`](src/lib.rs).
