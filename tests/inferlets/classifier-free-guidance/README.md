# classifier-free-guidance

Classifier-free guidance for text: run the model on a conditional and an
unconditional prompt in the same frame, then extrapolate *away* from the
unconditional one.

## Source

Sanchez, Fan, Spangher, Levi, Ammanamanchi and Biderman, ***Stay on topic with
Classifier-Free Guidance*** — <https://arxiv.org/abs/2306.17806>. Implements
Eq. 7.

**Faithfulness: Exact (equivalent form).** See
[`10-implementation-faithfulness-audit.md`](../../../inference-time-algorithms/10-implementation-faithfulness-audit.md).

## What it does

Borrowed from diffusion models. Two forward passes run per step — one on the
real prompt, one on a negative or empty prompt — and the difference between
them is amplified. What the conditional prompt adds over the unconditional one
gets pushed further in that direction, which makes the model stay on topic and
follow instructions more literally than plain sampling does.

## The rule

```
logits = logits_uncond + γ · (logits_cond − logits_uncond)
```

`γ = 1` recovers plain conditional sampling exactly.

The paper writes this over log-probabilities while this works in logits. The
two differ by the per-stream `logsumexp` constant, which is uniform over the
vocabulary — it shifts every entry of the blended vector by the same amount and
cancels in the softmax.

## Parameters

| Name | Type | Default | Meaning |
| --- | --- | --- | --- |
| `prompt` | string | *(a conditioning prompt)* | Conditioning prompt |
| `negative_prompt` | string | *(empty)* | Prompt to extrapolate away from; empty means unconditional |
| `guidance` | float | `1.5` | Guidance strength; `1.0` is plain conditional sampling |
| `temperature` | float | `1.0` | Temperature applied after guidance |
| `max_tokens` | int | `32` | Number of generated tokens |
| `seed` | int | — | RNG key for the Gumbel-max draw |

## Cost

**13.87 ms/token, 4.21× the [`naive-baseline`](../naive-baseline) control** on
an L40S with Qwen3-0.6B. The 4× is two compounding effects, not one:

- **Two forward passes** by construction, which is 2× on its own.
- **Loss of run-ahead.** The next input depends on the *combined* output of
  both passes, so neither pass may be pipelined against the next step. That is
  the other 2×.

Both are inherent to contrastive decoding, not artifacts of this
implementation. [`context-aware-decoding`](../context-aware-decoding) pays the
same 4× for the same reason.

## Run

```bash
cargo build --release --target wasm32-wasip2
python tests/inferlets/run_all.py --driver cuda_native --model <model-path>
```

The suite includes an **identity control**: at `guidance = 1.0` the rule
provably reduces to plain conditional sampling, so the reported `mean_kl` must
be exactly `0.0000`. It is.

Details are in the `//!` header of [`src/lib.rs`](src/lib.rs).
