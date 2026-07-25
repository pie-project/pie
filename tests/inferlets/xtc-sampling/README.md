# xtc-sampling

Exclude Top Choices — a sampler that removes the *most* likely tokens, on
purpose, to push the model off its most predictable continuations.

## Source

p-e-w, ***Exclude Top Choices (XTC)*** — oobabooga/text-generation-webui
[PR #6335](https://github.com/oobabooga/text-generation-webui/pull/6335),
merged 2024-09-28. Also in llama.cpp `src/llama-sampler.cpp`.

**Faithfulness: Faithful with one bounded deviation** (below). See
[`10-implementation-faithfulness-audit.md`](../../../inference-time-algorithms/10-implementation-faithfulness-audit.md).

## What it does

Every other truncation sampler here removes the *tail*. XTC removes the
**head**. When more than one token clears `threshold`, all of them except the
least likely such token are dropped, so the model is forced onto its second
thoughts. Because that would wreck coherence if done at every step, a Bernoulli
gate fires it only a `probability` fraction of the time.

The asymmetry is deliberate: the *least* likely above-threshold token always
survives, so the candidate set is never empty and the step never degenerates.

## The rule

```
if Bernoulli(probability):
    T = {x : p(x) >= threshold}
    if |T| > 1:  drop all of T except argmin_{x∈T} p(x)
```

A min-p floor is applied *before* XTC, matching the PR's guidance to place XTC
after all truncation samplers.

## Parameters

| Name | Type | Default | Meaning |
| --- | --- | --- | --- |
| `prompt` | string | *(a short paragraph request)* | Prompt to send to the model |
| `threshold` | float | `0.1` | Probability above which a token counts as a top choice |
| `min_p` | float | `0.02` | Min-p floor applied before XTC; `0` disables |
| `probability` | float | `0.5` | Chance of firing at each step, `0..1` |
| `temperature` | float | `1.0` | Temperature applied before the rule |
| `max_tokens` | int | `32` | Number of generated tokens |
| `seed` | int | — | RNG key for the gate and the Gumbel-max draw |

The `min_p 0.02` + DRY `0.8` pairing is the PR's own recommendation.

## Deviations from the reference

One. The reference aborts XTC when the top choice is a newline or EOS token, to
avoid mangling structure. That guard needs a tokenizer-specific token-ID table
which this inferlet does not carry, so it is omitted.

## Cost

**4.48 ms/token, 1.36× the [`naive-baseline`](../naive-baseline) control** on
an L40S with Qwen3-0.6B — a threshold scan plus the Bernoulli gate.

## Run

```bash
cargo build --release --target wasm32-wasip2
python tests/inferlets/run_all.py --driver cuda_native --model <model-path>
```

The tests pin both gate endpoints (`fire_rate` exactly `0.0` and `1.0`).
Details are in the `//!` header of [`src/lib.rs`](src/lib.rs).
