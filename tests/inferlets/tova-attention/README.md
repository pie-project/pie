# tova-attention

**TOVA — Token Omission Via Attention** (Oren et al., 2024,
[arXiv:2401.06104](https://arxiv.org/abs/2401.06104)).

TOVA is the simplest KV-eviction policy there is, and that is the point: at
every decoding step keep the `cache_size` KV positions the *current* query
attended to most, and drop the rest. No accumulated history (that is H2O), no
observation window (that is SnapKV) — just the attention distribution of the
most recent token.

This inferlet is the **observability half**, in the same sense as
`quest-attention`: it runs TOVA's exact decision quantity on real hardware,
against the live KV cache, and drains the scores to the host so the keep-set can
be checked. It does not yet mask the evicted positions out of the attention
kernel, so it produces **bit-identical output to `naive-baseline`** — which is
what makes it testable. Any divergence is a bug in the tap.

## What it demonstrates

`intrinsics::attn_score(planes)` at the **epilogue** returns
`[planes, intrinsics::attn_score_kv_max()]` f32 — every exported attention
layer and every query head of this fire, at once, as a device tensor. The graph
wrote it (the attention capture arm accumulates per-key mass as it runs) and the
epilogue reads it; there is no per-layer stage, no mid-forward tap, and no host
in the loop. `Stage::OnAttn` does not admit the intrinsic at all: a program that
reads it there is refused at bind.

Row `layer * heads + head` — **layer-major, head-minor**, so declaring fewer
planes than the load exports reads a prefix of the LAYERS rather than a stripe
of the heads. Slot semantics per row:

- `i < kv_len` — that (layer, head)'s attention probability at that position,
  averaged over the observation window's query rows; the live prefix sums to 1.
- `kv_len <= i < attn_score_kv_max()` — exactly `0.0`, rewritten every fire (so
  never a stale tail). A position that does not exist received no attention, so
  it sorts to the bottom of every eviction ranking without needing a sentinel.
  (Contrast Quest, whose unbounded criticality bounds need `+inf`/`-inf`.)

The **observation window** is the backend's statute: the last `min(32, qo_len)`
query rows of the request. This program taps only 1-row decode fires, so the
window is one row — the current query's own distribution, which is exactly what
TOVA ranks.

The **fold is in-graph, at the epilogue, on the device** — only the decision
crosses to the host:

```rust
let rect   = intrinsics::attn_score(planes);          // [planes, kv_max]
let folded = &reduce_sum(&transpose(&rect)) / heads;  // [kv_max], mass = layers
```

`transpose` puts the planes on the last axis so `reduce_sum` (which reduces the
last axis) sums down them, and `/ heads` turns that plane-sum into "mean over
heads, then sum over layers" in one pass, because
`Σ_l (1/H) Σ_h row = (1/H) Σ_planes row`.

`planes = layers * heads` is **declared, not derived** — the same contract
`intrinsics::hidden(width)` carries. The plane count is not in the model profile
and the SDK has no host call for it, so the program states it and the backend
**refuses** a claim larger than the load exports, by name. The defaults are
`Qwen/Qwen3.5-0.8B`: `Model::d0_8b` is `layers: 24, attn_every: 4`, so the
hybrid SKU puts attention on 6 of its 24 layers (the other 18 are GDN and export
nothing), with `q_heads: 8` → **6 layers × 8 heads = 48 planes**.

The row's WIDTH is not the program's to declare: a slab pitch cannot be a
per-program number, so `attn_score_kv_max()` publishes the one that was carved.
`kv_max` in the report is the program's own page geometry
(`max_pages * page_size`), the prefix of that row it reads; a request whose
geometry would outgrow the published ceiling is refused up front rather than
truncated.

## Requirements

Needs the `has_attn_score` model capability. Without it the program is rejected
at bind rather than reading an unwritten buffer.

## Reading the report

`score_mass` must equal `layers_observed`: the mean over heads is one
distribution per layer, so the layer-sum is a row of mass exactly the layer
count. That makes the drained row **self-validating** — it fails if the capture
is mis-normalized, mis-strided, or truncated.

`layers_observed` is now **derived from the declared shape** (`planes / heads`)
rather than counted by a device channel: with the rectangle arriving whole there
is no per-layer tap left to count.

`trace` pairs each fire's declared `kv_len` with the live length actually
observed in the row. They must agree on every fire. See §9.3 of
`inference-time-algorithms/12-attention-observability-design.md` for why this is
checked per fire rather than once at the end.

## Deviations from the paper

1. **Heads are folded by the program.** TOVA ranks per head; the paged layout
   carries one page list per request, so a per-head keep-set has no
   representable consumer. The rectangle is per-head — observability wants it
   that way — so the program takes the mean itself, in-graph.
   `quest-attention` documents the same collapse.
2. **Layers are folded by the program.** TOVA keeps a cache per layer; this sums
   the per-layer rows and ranks the sum — the layer-uniform variant the paper
   itself evaluates, and monotone-equivalent to the mean.
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
