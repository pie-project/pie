# dry-repetition-penalty

DRY — "Don't Repeat Yourself". Penalizes repeated *sequences* rather than
repeated *tokens*, with a charge that grows exponentially in the length of the
repeat.

## Source

p-e-w, ***DRY*** — first shipped in oobabooga/text-generation-webui, since
adopted by [llama.cpp](https://github.com/ggml-org/llama.cpp)
(`src/llama-sampler.cpp`) and KoboldCpp.

**Faithfulness: Faithful with two bounded deviations** (below). See
`inference-time-algorithms/10-implementation-faithfulness-audit.md`.

## What it does

A plain repetition penalty cannot tell the difference between the word "the"
appearing for the fortieth time — which is fine — and a whole clause being
replayed verbatim, which is not. DRY separates them. It looks at the suffix of
the text generated so far, finds every earlier place that suffix also occurred,
and charges the token that *continued* it there. Short repeats are free;
the price of extending a repeat compounds, so continuing an 8-gram becomes
unaffordable long before the model gets there.

## The rule

For each candidate token `x`, let `L(x)` be the length of the longest suffix
match that `x` would extend:

```
penalty(x) = multiplier · base^(L(x) − allowed_length)   if L(x) >= allowed_length
             0                                           otherwise
logit(x)  -= penalty(x)
```

## Parameters

| Name | Type | Default | Meaning |
| --- | --- | --- | --- |
| `prompt` | string | *(a short paragraph request)* | Prompt to send to the model |
| `multiplier` | float | `0.8` | Penalty scale; `0` disables DRY |
| `base` | float | `1.75` | Exponential growth per token of repeat length |
| `allowed_length` | int | `2` | Repeats shorter than this are free |
| `max_ngram` | int | `8` | Longest repeat the scan can see, `<= 16` |
| `temperature` | float | `1.0` | Temperature applied after the penalty |
| `max_tokens` | int | `32` | Number of generated tokens |
| `seed` | int | — | RNG key for the Gumbel-max draw |

## Deviations from the reference

Two, both deliberate:

- **No sequence breakers.** The reference resets matching at configured strings
  (`\n`, `:`, quotes). Those are strings, and mapping them to token IDs is
  tokenizer-dependent, so they are omitted.
- **`max_ngram` caps the match length.** The penalty therefore saturates at
  `multiplier · base^(max_ngram − allowed_length)` instead of growing without
  bound. At the defaults that ceiling is already ~29 logits, far past the point
  where the token is unreachable, so the cap is not observable in practice.

## Cost

**7.08 ms/token, 2.15× the [`naive-baseline`](../naive-baseline) control** on
an L40S with Qwen3-0.6B. The `max_ngram`-deep match is unrolled, so the scan
depth is what you pay for — this is the most expensive of the single-pass
penalties, and the scatters are the reason `max_ngram` is capped at 16.

## Run

```bash
cargo build --release --target wasm32-wasip2
python tests/inferlets/run_all.py --driver cuda_native --model <model-path>
```

Details are in the `//!` header of [`src/lib.rs`](src/lib.rs).
