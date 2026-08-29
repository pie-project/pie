# eta-epsilon-sampling

Two truncation rules that cut the tail at an absolute probability floor, with
the floor either fixed (epsilon) or adapted to the step's entropy (eta).

## Source

Hewitt, Manning and Liang, ***Truncation Sampling as Language Model
Desmoothing*** (Findings of EMNLP) —
<https://arxiv.org/abs/2210.15191>. Implements §3.

**Faithfulness: Exact.** See
`inference-time-algorithms/10-implementation-faithfulness-audit.md`.

## What it does

The paper's framing is that a neural LM is a *smoothed* version of the true
distribution, and truncation is desmoothing. That reframing predicts the floor
should depend on how peaked the step already is: a confident step deserves an
aggressive floor, an uncertain one deserves a lenient floor. Epsilon sampling
uses a constant floor; eta sampling takes the smaller of the constant and a
term that decays with entropy.

## The rule

```
epsilon:  keep {x : p(x) >= ε}
eta:      keep {x : p(x) >= min(ε, √ε · exp(-H))}
```

where `H` is the entropy of the step in nats. Both are re-normalized after
truncation. The mask is protected so that an over-aggressive `epsilon`
degenerates to greedy decoding rather than to an empty candidate set.

## Parameters

| Name | Type | Default | Meaning |
| --- | --- | --- | --- |
| `prompt` | string | *(a short paragraph request)* | Prompt to send to the model |
| `mode` | string | `eta` | Truncation mode: `eta` or `epsilon` |
| `epsilon` | float | `0.0003` | Probability floor |
| `temperature` | float | `1.0` | Temperature applied before truncation |
| `max_tokens` | int | `32` | Number of generated tokens |
| `seed` | int | — | RNG key for the Gumbel-max draw |

## Cost

**4.31 ms/token, 1.31× the [`naive-baseline`](../naive-baseline) control** on
an L40S with Qwen3-0.6B. One entropy reduction plus two elementwise passes; no
sort, which is why this is an order of magnitude cheaper than the rank-based
truncations.

## Run

```bash
cargo build --release --target wasm32-wasip2
python tests/inferlets/run_all.py --engine cuda_native --model <model-path>
```

Details are in the `//!` header of [`src/lib.rs`](src/lib.rs).
