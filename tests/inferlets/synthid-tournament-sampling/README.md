# synthid-tournament-sampling

SynthID-Text: layered keyed g-functions reweight the sampling distribution, and
the mean of those g-values over the generated text is the detector.

## Source

Dathathri et al., ***Scalable watermarking for identifying large language model
outputs***, *Nature* **634**, 818–823 (2024) —
<https://doi.org/10.1038/s41586-024-08025-4>.

**Faithfulness: Exact (equivalent form).** See
`inference-time-algorithms/10-implementation-faithfulness-audit.md`.

## What it does

For each step, the secret and the preceding `context_width` tokens seed `depth`
independent pseudorandom `{0,1}` functions over the vocabulary. The paper
describes a knockout tournament: sample `2^depth` candidates, and at each layer
pair them up and keep the one with the higher `g`. Tokens that happen to score
well under the key are systematically favoured, without any single step being
forced.

Detection is then just the mean `g` over the generated tokens, which sits at
`0.5` under the null and above it under the key.

## The rule

Per layer `ℓ`, this implements the tournament's exact per-round win
probability in closed form:

```
m_ℓ    = E_{p_ℓ}[g_ℓ]
p_{ℓ+1}(w) = p_ℓ(w) · (1 + g_ℓ(w) − m_ℓ)
```

This is not an approximation of the tournament — it *is* its per-round win
probability, and it composes across layers because the round-`ℓ` winners are
i.i.d. draws from `p_ℓ`. The derivation is in the audit.

Detection over `n` tokens:

```
z = (mean_t g(chosen_t) − 0.5) · √n
```

## Parameters

| Name | Type | Default | Meaning |
| --- | --- | --- | --- |
| `prompt` | string | *(a short paragraph request)* | Prompt to send to the model |
| `secret` | int | — | Watermark secret; the detector needs this and nothing else |
| `context_width` | int | `4` | Preceding tokens seeding each step (`ngram_len − 1`), `1..8` |
| `depth` | int | `9` | Tournament layers, `1..16` |
| `history_size` | int | `64` | Ring buffer of recent context hashes, `1..1024` |
| `watermark` | bool | `true` | `false` skips the tournament, producing unwatermarked text |
| `temperature` | float | `1.0` | Temperature applied before sampling |
| `max_tokens` | int | `32` | Number of generated tokens |
| `seed` | int | — | RNG key for the sampler draw |

`depth` is the direct cost dial — it is the number of reweighting rounds run
per token.

## Cost

**33.55 ms/token, 10.2× the [`naive-baseline`](../naive-baseline) control** on
an L40S with Qwen3-0.6B at the default `depth = 9`. That is nine full knockout
rounds per token, and it is the most expensive algorithm in this set.

Two caveats on that number. It is roughly linear in `depth`, so lowering `depth`
buys back most of it — at `depth = 3` a 160-token run is a stable 816–864 ms.
And at the default `depth = 9` this inferlet is **bimodal**: the same inputs
return either ~1.3 s or ~5–6 s for a 160-token budget, and consecutive calls in
one process alternate between the two. The figure above is the slow mode and
should be read as an upper bound; the fast mode's slope is ≈2.4 ms/token.

The response is **bit-identical** in both modes (same text, same `z_score`), so
this is purely a latency effect. It is host-bound — the GPU is ~25 % utilised in
both modes — and it is not explained by run-ahead depth, ring capacity, GPU
contention, or compiler nondeterminism, all of which were tested. See the "A10
is bimodal" section of
`inference-time-algorithms/10-implementation-faithfulness-audit.md`.

Separately: the **first** call against a plan shape it has not seen before pays
a 12–31 s NVRTC compile, cached thereafter under `$PIE_HOME/cache/cubins`. All
figures here are warm.

## Run

```bash
cargo build --release --target wasm32-wasip2
python tests/inferlets/run_all.py --engine cuda_native --model <model-path>
```

As with [`gumbel-watermark`](../gumbel-watermark), the smoke test is structural
rather than a detection assertion: an observed 8-token run gave `mean_score`
0.5 against a null of 0.53, which is exactly the variance you expect at that
`n`. Detection power versus `n` is measured in the audit, where the null mean
converges to the theoretical `0.5`.

Details are in the `//!` header of [`src/lib.rs`](src/lib.rs).
