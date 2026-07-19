# locally-typical-sampling

Truncation sampling that keeps the tokens whose information content sits
*closest to the distribution's own entropy*, rather than the tokens that are
simply most probable.

## Source

Meister, Pimentel, Wiher and Cotterell, ***Locally Typical Sampling*** (TACL) —
<https://arxiv.org/abs/2202.00666>. Implements §3, Eq. 6.

**Faithfulness: Exact.** See the per-line reduction in
[`10-implementation-faithfulness-audit.md`](../../../inference-time-algorithms/10-implementation-faithfulness-audit.md).

## What it does

Top-p keeps a prefix of the *probability* ranking, which always keeps the
argmax and so systematically over-samples the head. Typical sampling ranks by
`|log p(x) + H|` instead — the absolute deviation of a token's surprisal from
the mean surprisal — and keeps the smallest-deviation tokens until their mass
reaches `mass`. A token that is *too predictable* is discarded just as readily
as one that is too surprising, which is the whole point: natural language sits
near the entropy, not at the mode.

## The rule

```
H       = -Σ p(x) log p(x)                  # entropy of the step
score(x)= |log p(x) + H|                    # typicality deviation
keep    = smallest-score prefix whose cumulative p first reaches `mass`
```

The mask is never empty: the most typical token has exclusive prefix mass
`0 < mass`, so it always survives the comparison.

## Parameters

| Name | Type | Default | Meaning |
| --- | --- | --- | --- |
| `prompt` | string | *(a short paragraph request)* | Prompt to send to the model |
| `mass` | float | `0.95` | Typical mass to retain, `0 < mass <= 1` |
| `k_max` | int | `128` | Candidate pool bound applied before typicality |
| `temperature` | float | `1.0` | Temperature applied before typicality |
| `max_tokens` | int | `32` | Number of generated tokens |
| `seed` | int | `0x7ce1` | RNG key for the Gumbel-max draw |

`k_max` bounds an otherwise full-vocabulary sort. The reported `mean_mass`
tells you whether it is binding: if it sits well below `mass`, raise `k_max`.

## Cost

**17.71 ms/token, 5.37× the [`naive-baseline`](../naive-baseline) control** on
an L40S with Qwen3-0.6B. Almost all of it is `top_k` over a 262144-entry
vocabulary, not the typicality maths — `tail-free-sampling` computes a
completely different statistic and costs the same. `k_max` is the dial that
matters.

## Run

```bash
cargo build --release --target wasm32-wasip2
python tests/inferlets/run_all.py --driver cuda_native --model <model-path>
```

The deep explanation — including why the deviation is computed with
`max(d, -d)` and how the `k_max` approximation is made observable rather than
silent — is the `//!` header of [`src/lib.rs`](src/lib.rs).
