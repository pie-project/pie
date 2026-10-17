# tova-attention

**TOVA — Token Omission Via Attention** (Oren et al., 2024,
[arXiv:2401.06104](https://arxiv.org/abs/2401.06104)).

TOVA is the simplest KV-eviction policy there is: at every decoding step keep
the `cache_size` KV positions the *current* query attended to most, and drop
the rest. No accumulated history (that is H2O), no observation window (that is
SnapKV) — just the attention distribution of the most recent token.

This inferlet is the **observability half**: it computes TOVA's decision
quantity against the live KV cache and drains the scores to the host so the
keep-set can be checked. It does not mask the evicted positions, so it produces
**bit-identical output to `naive-baseline`** — any divergence is a bug in the
tap.

## What it demonstrates

`intrinsics::attn_score(planes)` at the **epilogue** returns
`[planes, intrinsics::attn_score_kv_max()]` f32 — every exported attention
layer and every query head of this fire, at once, as a device tensor captured
by the attention kernel as it ran. Rows are `layer * heads + head`
(**layer-major**), so declaring fewer planes than the load exports reads a
prefix of the layers. Per row, slots `< kv_len` hold that head's attention
probability (the live prefix sums to 1) and the rest are exactly `0.0`.

The backend folds the last `min(32, qo_len)` query rows into the row it
publishes; this program taps only 1-row decode fires, so the row is the current
query's own distribution — what TOVA ranks.

The fold is in-graph, at the epilogue, on the device:

```rust
let rect   = intrinsics::attn_score(planes);          // [planes, kv_max]
let folded = &reduce_sum(&transpose(&rect)) / heads;  // [kv_max], mass = layers
```

`planes = layers * heads` is **declared**: the program states it and the
backend refuses a claim larger than the load exports. The defaults fit
`Qwen/Qwen3.5-0.8B`, whose hybrid stack puts attention on 6 of its 24 layers,
with 8 query heads. `kv_max` in the report is the program's page-aligned window
into the published row; a request that would outgrow the row is refused.

## Requirements

Needs the `has_attn_score` model capability.

## Reading the report

`score_mass` must equal `layers_observed` (`planes / heads`): the mean over
heads is one distribution per layer. That makes the drained row
self-validating — it fails if the capture is mis-normalized, mis-strided, or
truncated. `trace` pairs each fire's declared `kv_len` with the live length
observed in the row; they must agree on every fire.

## Deviations from the paper

1. **Heads are folded by the program.** TOVA ranks per head; one keep-set
   serves a request, so the program takes the mean over heads.
2. **Layers are folded by the program.** TOVA keeps a cache per layer; this sums
   the per-layer rows — the layer-uniform variant the paper evaluates.
3. **Selection is observed, not enforced.**

## Parameters

| name | type | default | meaning |
|---|---|---|---|
| `prompt` | string | "The capital of France is" | prompt |
| `temperature` | float | 1.0 | applied before the Gumbel-max draw |
| `max_tokens` | int | 32 | tokens to generate |
| `seed` | int | — | RNG key |
| `cache_size` | int | 16 | KV positions TOVA would keep |
| `layers` | int | 6 | exported attention layers to claim |
| `heads` | int | 8 | query heads per exported layer |
