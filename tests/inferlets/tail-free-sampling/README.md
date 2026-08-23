# tail-free-sampling

Truncation that finds the tail by looking at the *curvature* of the sorted
probability curve, cutting where the curve goes flat.

## Source

Trenton Bricken, ***Tail Free Sampling*** (2019) —
<https://github.com/TrentBrick/TailFreeSampling>. Reference implementations:
llama.cpp `src/llama-sampler.cpp` and oobabooga/text-generation-webui
`modules/sampler_hijack.py`.

**Faithfulness: Exact (equivalent form).** See
`inference-time-algorithms/10-implementation-faithfulness-audit.md`.

## What it does

Top-k needs you to guess a count, top-p needs you to guess a mass, and the
right answer for both moves step to step. TFS instead asks where the sorted
probability curve stops *dropping* and starts crawling — the second derivative
is large across the head and near zero across the tail. Normalizing the
absolute second difference into a distribution and keeping the first `z` of its
mass locates that elbow without a fixed budget.

## The rule

```
p        = sort(softmax(logits), descending)
d1(i)    = p(i+1) - p(i)                     # first difference
d2(i)    = |d1(i+1) - d1(i)|                 # absolute second difference
d2̂       = d2 / Σ d2                         # normalized into a distribution
keep     = shortest prefix of d2̂ whose cumulative mass first reaches z
```

## Parameters

| Name | Type | Default | Meaning |
| --- | --- | --- | --- |
| `prompt` | string | *(a short paragraph request)* | Prompt to send to the model |
| `z` | float | `0.95` | Fraction of total curvature to retain, `0 < z <= 1` |
| `temperature` | float | `1.0` | Temperature applied before truncation |
| `k_max` | int | `128` | Candidate-set bound for the curvature scan |
| `max_tokens` | int | `32` | Number of generated tokens |
| `seed` | int | — | RNG key for the Gumbel-max draw |

## Deviations from the reference

Two, both deliberate:

- **Rank convention.** This keeps ranks `0..j`, matching oobabooga. The
  original gist keeps `0..j+1`; it also indexes *unsorted* logits with a
  *sorted* rank, which is a plain bug that no downstream implementation copied.
- **`d2` length.** `d2` is kept at length `k_max` with clamped edges instead of
  shrinking by two, so the vector shape is static. The clamped edges contribute
  zero curvature, so the selection is unchanged.

## Cost

**5.20 ms/token, 1.49× the [`naive-baseline`](../naive-baseline) control** on an
L40S with Qwen3-0.6B. The residual is the region break that `top_k` and
`cum_sum` force as schedule barriers, not the curvature arithmetic. It read
17.47 ms (5.30×) until the `top_k` kernel was changed from a per-pick row rescan
to a radix select; cost is now effectively flat in `k_max`.

## Run

```bash
cargo build --release --target wasm32-wasip2
python tests/inferlets/run_all.py --driver cuda_native --model <model-path>
```

Details are in the `//!` header of [`src/lib.rs`](src/lib.rs).
