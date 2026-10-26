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

`intrinsics::output(0, "attn.scores")` at the **epilogue** returns the
model's `attn.scores` output for lane 0: `[layers, heads * kv_ceiling]` f32,
one row a layer the model reads scores out of, each a `kv_ceiling` run a
query head — a device tensor its `attn.score_capture` op wrote. Per run, slots
`< kv_len` hold that head's attention probability (the live prefix sums to 1)
and the rest are exactly `0.0`.

The op folds the lane's last `observe` query rows (32 in the in-tree models)
into the row; this program taps only 1-row decode fires, so the row is the
current query's own distribution — what TOVA ranks.

The fold is in-graph, at the epilogue, on the device:

```rust
let rect   = reshape(intrinsics::output(0, "attn.scores"), [planes, kv_ceiling]);
let folded = &reduce_sum(&transpose(&rect)) / heads;  // [kv_ceiling], mass = layers
```

`layers` and the row's width come from the model's output declaration
(`model::lane_of`); the program states `heads`, which splits the width into
`kv_ceiling`. The default fits `Qwen/Qwen3.5-0.8B` (8 query heads). `kv_max` in
the report is the program's page-aligned window into the row; a request that
would outgrow the row is refused.

## Requirements

Needs a model that reads out `attn.scores`.

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
| `heads` | int | 8 | query heads per exported layer |
