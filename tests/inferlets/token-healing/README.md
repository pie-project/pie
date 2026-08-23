# token-healing

Backs the prompt off its trailing partial token and re-expands it under a
prefix mask, so a prompt that ends mid-token does not poison the first
generated token.

## Source

No paper. Reference implementations:
[guidance-ai/guidance](https://github.com/guidance-ai/guidance), its
[llguidance](https://github.com/guidance-ai/llguidance) backend, and llama.cpp's
`--token-healing`.

**Faithfulness: Exact.** See
`inference-time-algorithms/10-implementation-faithfulness-audit.md`.

## What it does

Tokenizers are greedy, so a prompt ending in `"The capital of Fra"` gets
tokenized with `Fra` as a *complete* token. But the model has almost never seen
`Fra` as a standalone token followed by `nce` — in training, `France` was one
token. The prompt has been pushed off-distribution by its own boundary, and the
model's next-token distribution is measurably worse.

Token healing rolls the last `backoff` token(s) off the prompt, then constrains
the next step to only those tokens whose *bytes start with* the removed text.
The model re-chooses the boundary itself, and the emitted text is byte-identical
to the prompt — nothing is lost, only re-segmented.

## The rule

```
1. drop the last `backoff` tokens, remembering their bytes as `fragment`
2. prefill the shortened prompt
3. mask the next step to {x : bytes(x) starts with fragment}
4. sample from the masked distribution and continue normally
```

The output must reproduce the original prompt bytes exactly; the reported
`prompt_preserved` asserts this.

## Parameters

| Name | Type | Default | Meaning |
| --- | --- | --- | --- |
| `prompt` | string | *(a prompt ending mid-word)* | Prompt whose final token is likely a split fragment |
| `heal` | bool | `true` | Enable healing; `false` reproduces the unhealed baseline |
| `backoff` | int | `1` | Trailing tokens to roll back and re-expand, `1..4` |
| `max_tokens` | int | `32` | Number of generated tokens |

`heal=false` runs the identical code path without the first mask, so the two
settings are a clean A/B on exactly one thing.

## Cost

**2.48 ms/token, 0.75× the [`naive-baseline`](../naive-baseline) control** on
an L40S with Qwen3-0.6B. Faster than the baseline because it decodes greedily
via `reduce_argmax` and never materializes a noise tensor — the healing mask
itself is a one-time cost on the first step only, not a per-token one.

## Run

```bash
cargo build --release --target wasm32-wasip2
python tests/inferlets/run_all.py --driver cuda_native --model <model-path>
```

Details are in the `//!` header of [`src/lib.rs`](src/lib.rs).
