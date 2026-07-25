# gumbel-watermark

Aaronson's distortion-free watermark: key the sampler's randomness on the
preceding context instead of on a counter, so the text carries a signal that
only the key holder can read — and the output distribution is unchanged.

## Source

Aaronson's rule, as stated and analysed in Kuditipudi, Thickstun, Hashimoto and
Liang, ***Robust Distortion-free Watermarks for Language Models*** —
<https://arxiv.org/abs/2307.15593>, §2–3.

The greenlist scheme this is contrasted with is Kirchenbauer et al. —
<https://arxiv.org/abs/2301.10226> — implemented separately in
[`greenlist-watermarking`](../greenlist-watermarking).

**Faithfulness: Exact (equivalent form)**, with one bounded deviation (below).
See
[`10-implementation-faithfulness-audit.md`](../../../inference-time-algorithms/10-implementation-faithfulness-audit.md).

## What it does

Greenlist watermarking biases the logits, so it provably changes what the model
writes. This one does not touch the logits at all. It changes only *where the
sampler's noise comes from*: the Gumbel noise is drawn from a stream keyed by
the secret and the last `context_width` tokens.

Because the noise is still marginally correct Gumbel noise, the output
distribution is **exactly** the untouched one — the watermark is distortion
free. But a detector who knows the secret can recompute the same noise and see
that the chosen tokens line up with it far better than chance.

## The rule

Sampling, as an exponential race:

```
ξ(x)   = keyed uniform from (secret, last context_width tokens)
choose  argmax_x ξ(x)^(1/p(x))     ==     argmax_x (log p(x) + G_keyed(x))
```

Detection, over `n` generated tokens:

```
score_t = -log(1 - ξ(chosen_t))          # Exp(1) under the null
z       = (mean_t score_t − 1) · √n
```

An unkeyed null is scored alongside on the same text, so every run reports its
own control.

## Parameters

| Name | Type | Default | Meaning |
| --- | --- | --- | --- |
| `prompt` | string | *(a short paragraph request)* | Prompt to send to the model |
| `secret` | int | — | Watermark secret; the detector needs this and nothing else |
| `context_width` | int | `4` | Preceding tokens seeding each step, `1..8` |
| `watermark` | bool | `true` | `false` free-runs the RNG, producing unwatermarked text |
| `temperature` | float | `1.0` | Temperature applied before sampling |
| `max_tokens` | int | `32` | Number of generated tokens |
| `seed` | int | — | RNG key used only when watermarking is off |

## Deviations from the reference

One. The per-token score is floored at `1e-7`. Without it, a strongly favoured
token gives `ξ ≈ 1` and the score diverges to `+∞`, destroying the statistic.
The floor biases the null mean low by that amount, which is far below the
detection threshold.

## Cost

**2.78 ms/token, 0.84× the [`naive-baseline`](../naive-baseline) control** on
an L40S with Qwen3-0.6B. The watermark is not merely cheap, it measures
*faster* than the baseline — and not because it does less work.

This inferlet contains its own controlled A/B: with `watermark=true` it samples
via `reduce_argmax(add(scaled, gumbel(...)))`, and with `watermark=false` it
uses the fused `gumbel_max` op. Same inferlet, same channels, one op different:

| Sampling op | ms/token |
| --- | --- |
| `gumbel()` + `reduce_argmax` (decomposed) | 2.85 |
| `gumbel_max()` (fused) | 3.60 |

The fused op is **27 % slower**, so the watermark's true marginal cost is zero
and the apparent speedup is an artifact of the faster code path.

## Run

```bash
cargo build --release --target wasm32-wasip2
python tests/inferlets/run_all.py --driver cuda_native --model <model-path>
```

The smoke test is deliberately structural: at the handful of tokens a smoke
test can afford, the score mean has too much variance to separate from its
null. Detection power versus `n` is measured in the audit instead, where the
null mean converges to the theoretical `E[Exp(1)] = 1`.

Details are in the `//!` header of [`src/lib.rs`](src/lib.rs).
