# mirostat-v2-sampling

Adaptive sampling that targets a chosen per-token surprise: it tightens or
loosens the keep set after every token so the observed surprise stays near
`tau`.

## Source

Basu et al., ***Mirostat: A Neural Text Decoding Algorithm that Directly
Controls Perplexity*** (ICLR) — <https://arxiv.org/abs/2007.14966>. Implements
Mirostat v2's feedback update.

**Faithfulness: Faithful with one safety floor.** The control law is the v2
update; the keep mask is optionally ORed with a non-empty floor so invalid `mu`
states cannot produce an empty sampler.

## What it does

Top-k and top-p use a fixed truncation rule, so the entropy of the sampled text
is whatever falls out of the model distribution. Mirostat closes that loop. It
keeps tokens whose surprise is below the current control value `mu`, samples from
that set, measures the selected token's surprise, and moves `mu` in the opposite
direction of the error from `tau`.

If the sampled token was too surprising, `mu` decreases and the next keep set is
stricter. If it was too predictable, `mu` increases and the next keep set opens
up. The result is not a one-shot truncation heuristic but a per-token feedback
controller.

## The rule

```
keep(x)    = -log p(x) <= mu
x_t        = sample from keep with Gumbel-max
surprise_t = -log p(x_t)
mu         = mu - learning_rate · (surprise_t - tau)
```

With the default `floor = "argmax"`, the argmax token is always kept. With
`floor = "rank"`, the top `k_min` tokens are always kept. With `floor = "plain"`,
the raw Mirostat mask is used.

## Parameters

| Name | Type | Default | Meaning |
| --- | --- | --- | --- |
| `prompt` | string | *(a short paragraph request)* | Prompt to send to the model |
| `tau` | float | `3.0` | Target surprise in nats |
| `learning_rate` | float | `0.6` | Feedback step size for updating `mu` |
| `max_tokens` | int | `64` | Number of generated tokens |
| `k_min` | int | `8` | Minimum rank floor when `floor = "rank"` |
| `floor` | string | `"argmax"` | Non-empty keep-set mode: `argmax`, `rank`, or `plain` |
| `mu0` | float or null | — | Optional initial control value; default is `ln(vocab) + 1` |

## Implementation notes

`mu` is a loop-carried channel. The prefill fire spends the first token budget,
returns the sampled token and surprise to the host, and the decode loop then
advances both token state and `mu` on device. The reported `mean_surprise`,
`tail_mean_surprise`, and `final_mu` are observability for the controller rather
than extra decoding inputs.

The floor is a bounded deviation from the paper's plain threshold rule. It is
there to make the sampler total over all finite `mu` values; `floor = "plain"`
selects the degenerate raw rule if that is what you want to test.

## Run

```bash
cargo build --release --target wasm32-wasip2
python tests/inferlets/run_all.py --engine cuda_native --model <model-path>
```

Details are in the `//!` header of [`src/lib.rs`](src/lib.rs).
