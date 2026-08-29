# greenlist-watermarking

A logit-bias watermark: each previous token chooses a pseudorandom greenlist,
and the next token is nudged toward that list before sampling.

## Source

Kirchenbauer et al., ***A Watermark for Large Language Models*** (ICML 2023) —
<https://arxiv.org/abs/2301.10226>. Implements the KGW previous-token keyed
greenlist bias.

**Faithfulness: Faithful with one implementation-level simplification.** The
logit rule matches KGW, but the greenlist is built with Rust's `DefaultHasher`
rather than the paper's keyed PRF/detector setup.

## What it does

The vocabulary is split into green and red tokens using a hash of the previous
output token and the candidate token. Green tokens get a positive logit offset
`delta`, so sampling is slightly more likely to choose them. Over many tokens the
excess green rate is detectable, while any single step still looks like ordinary
biased sampling.

The watermark is deliberately distribution-shifting. Unlike distortion-free
schemes that key the sampling noise, KGW changes the next-token distribution by
adding a bias. `gamma` controls the expected greenlist fraction; `delta` controls
how strongly the model is pushed toward it.

## The rule

```
green_t(x) = hash(previous_token, x) <= gamma · MAX_HASH
bias_t(x)  = delta if green_t(x) else 0

token_t = gumbel_max(logits_t + bias_t, rng_t)
```

## Parameters

| Name | Type | Default | Meaning |
| --- | --- | --- | --- |
| `prompt` | string | `"Explain language-model decoding in simple terms."` | User prompt to send to the model |
| `max_tokens` | int | `256` | Maximum number of generated tokens |
| `gamma` | float | `0.5` | Target fraction of the vocabulary assigned to the greenlist |
| `delta` | float | `2.0` | Logit bonus added to green tokens |

## Implementation notes

The greenlist mask is host-derived from the immediately preceding output token,
so the decode loop is structurally **depth-1**. It cannot submit
`DEFAULT_RUNAHEAD_DEPTH` fires ahead: fire `k + 1` needs the token produced by
fire `k` before the host can publish its green mask. This is one of the depth-1
cases called out in
`inference-time-algorithms/10-implementation-faithfulness-audit.md`.

The PTIR epilogue itself is small: take the host-provided boolean mask, select a
`delta` or zero bias for every vocabulary entry, add it to `intrinsics::logits()`,
and draw with `gumbel_max`.

## Run

```bash
cargo build --release --target wasm32-wasip2
python tests/inferlets/run_all.py --engine cuda_native --model <model-path>
```
