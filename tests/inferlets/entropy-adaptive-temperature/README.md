# entropy-adaptive-temperature

EDT — derives the sampling temperature at each step from that step's own
entropy, instead of holding it fixed across the whole generation.

## Source

Zhang, Bao and Huang, ***EDT: Improving Large Language Models' Generation by
Entropy-based Dynamic Temperature Sampling*** —
<https://arxiv.org/abs/2403.14541>. Implements Eq. 7.

**Faithfulness: Exact.** See
`inference-time-algorithms/10-implementation-faithfulness-audit.md`.

## What it does

A fixed temperature is a single compromise applied to two opposite situations.
Where the model is confident, sampling hot invents errors; where it is
genuinely uncertain, sampling cold collapses into repetition. EDT reads the
entropy of the current distribution and moves the temperature the *right* way:
high entropy pushes the temperature down toward determinism, low entropy lets
it rise toward `t0`.

## The rule

```
H = -Σ p(x) log p(x)
T = t0 · n^(θ/H)          with 0 < n < 1
```

Because `0 < n < 1`, the exponent `θ/H` shrinks as `H` grows, so `T` *falls*
with rising entropy and is bounded above by `t0`.

> **Note.** Several secondary summaries state this rule as
> `T = t0 + θ·(H / log K)`, which inverts the behaviour — temperature rising
> with entropy. Equation 7 of the paper is the multiplicative form above, and
> that is what this implements.

## Parameters

| Name | Type | Default | Meaning |
| --- | --- | --- | --- |
| `prompt` | string | *(a short paragraph request)* | Prompt to send to the model |
| `t0` | float | `1.0` | Upper bound on temperature |
| `theta` | float | `0.1` | Sensitivity of temperature to entropy |
| `n` | float | `0.8` | Base of the entropy response, `0 < n < 1` |
| `min_temperature` | float | `0.05` | Floor on the derived temperature |
| `max_tokens` | int | `32` | Number of generated tokens |
| `seed` | int | — | RNG key for the Gumbel-max draw |

The reported `mean_temperature` equal to `t0` would mean the entropy signal is
saturated and the knob is doing nothing.

## Cost

**3.53 ms/token, 1.07× the [`naive-baseline`](../naive-baseline) control** on
an L40S with Qwen3-0.6B — one entropy reduction and a scalar power. This is the
cheapest algorithm in the set; adaptivity here is close to free.

## Run

```bash
cargo build --release --target wasm32-wasip2
python tests/inferlets/run_all.py --engine cuda_native --model <model-path>
```

Details are in the `//!` header of [`src/lib.rs`](src/lib.rs).
