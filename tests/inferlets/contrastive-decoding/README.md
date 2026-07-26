# contrastive-decoding

Contrastive decoding chooses tokens the full-context expert likes more than a
short-window amateur, while filtering out tokens the expert itself finds
implausible.

## Source

Li et al., ***Contrastive Decoding: Open-ended Text Generation as Optimization***
(ACL) — <https://arxiv.org/abs/2210.15097>. Implements the contrastive score and
plausibility constraint from §3.

**Faithfulness: Structural — same-model, short-window amateur.** The paper uses
separate expert and amateur models; this inferlet demonstrates the mechanism
with one model run under two attention contexts.

## What it does

A plain sampler only asks whether the main model likes a token. Contrastive
decoding asks a sharper question: does the expert like it more than an amateur
would? Generic, bland, or degenerate continuations tend to be easy for both
models, so subtracting the amateur log-probability pushes them down.

The plausibility filter matters. Without it, a token could win merely because the
amateur hates it, even if the expert also thinks it is nonsense. This inferlet
therefore first keeps only tokens whose expert probability is at least `alpha`
times the expert maximum, then maximizes the contrastive score inside that set.

## The rule

```
expert(x)    = log p_expert(x | full context)
amateur(x)   = log p_amateur(x | short context)
plausible(x) = expert(x) >= max_y expert(y) + log(alpha)
score(x)     = expert(x) - lambda · amateur(x)
pick         = argmax score(x) over plausible tokens
```

`lambda = 0` recovers expert-only argmax under the plausibility filter.

## Parameters

| Name | Type | Default | Meaning |
| --- | --- | --- | --- |
| `prompt` | string | `"Explain why the sky appears blue."` | User prompt wrapped in the chat template |
| `max_tokens` | int | `256` | Maximum number of generated tokens |
| `amateur_window` | int | `8` | Attention window used by the same-model amateur |
| `lambda` | float | `0.5` | Weight applied to the amateur log-probability |
| `alpha` | float | `0.1` | Expert plausibility threshold as a fraction of the max probability |

## Implementation notes

The amateur and expert use separate KV working sets. The amateur attends only a
sliding `amateur_window`; the expert attends the complete context. The amateur
logits are passed through a host `Writer` channel into the expert epilogue, where
the contrastive argmax runs in PTIR.

This loop is structurally depth-1, as noted in
[`10-implementation-faithfulness-audit.md`](../../../inference-time-algorithms/10-implementation-faithfulness-audit.md):
the amateur input for step `k + 1` is the expert's just-selected output token from
step `k`, so the decode loop cannot run ahead. That is an algorithmic dependency,
not a missing optimization.

## Run

```bash
cargo build --release --target wasm32-wasip2
python tests/inferlets/run_all.py --driver cuda_native --model <model-path>
```

Details are in the `//!` header of [`src/lib.rs`](src/lib.rs).
