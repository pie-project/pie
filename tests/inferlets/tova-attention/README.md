# tova-attention

**TOVA — Token Omission Via Attention** (Oren et al., 2024,
[arXiv:2401.06104](https://arxiv.org/abs/2401.06104)).

TOVA is the simplest KV-eviction policy there is, and that is the point: at
every decoding step keep the `cache_size` KV positions the *current* query
attended to most, and drop the rest. No accumulated history (that is H2O), no
observation window (that is SnapKV) — just the attention distribution of the
most recent token.

This inferlet is the **observability half**, in the same sense as
`quest-attention`: it runs TOVA's exact decision quantity on real hardware, per
layer, against the live KV cache, and drains the scores to the host so the
keep-set can be checked. It does not yet mask the evicted positions out of the
attention kernel, so it produces **bit-identical output to `naive-baseline`** —
which is what makes it testable. Any divergence is a bug in the tap.

## What it demonstrates

`intrinsics::attn_score(kv_max)` at the `on_attn` stage returns `[kv_max]` f32:
the attention probability the request's most recent query token assigned to each
live KV position, averaged over query heads. These are the probabilities the
decode kernel itself computed — captured inside the attention kernel through a
FlashInfer attention variant, not recomputed — so they cannot drift from the
attention the model actually performed.

Slot semantics:

- `i < kv_len` — the mean attention probability at that position; the live
  prefix sums to 1.
- `kv_len <= i < kv_max` — exactly `0.0`. A position that does not exist
  received no attention, so it sorts to the bottom of every eviction ranking
  without needing a sentinel. (Contrast Quest, whose unbounded criticality
  bounds need `+inf`/`-inf`.)

`kv_max` is the program's own static ceiling, mirroring `envelope_dot`'s
`p_max`: an inferlet cannot know the runtime KV length, so it declares a bound
and the backend **refuses** a longer request rather than truncating.

## Requirements

Needs the `has_attn_score` model capability: llama-like family, `tp == 1`,
native bf16 non-HND pages, no sliding window, and decode on the plain paged
path. Without it the program is rejected at bind rather than reading an
unwritten buffer.

## Reading the report

`score_mass` must equal `layers_observed`: each layer contributes a
distribution, so the folded row sums to the layer count. That makes the drained
row **self-validating** — it fails if the capture is mis-normalized,
mis-strided, or truncated.

`trace` pairs each fire's declared `kv_len` with the live length actually
observed in the row. They must agree on every fire. See §9.3 of
`inference-time-algorithms/12-attention-observability-design.md` for why this is
checked per fire rather than once at the end.

## Deviations from the paper

1. **Heads are folded by the backend.** TOVA ranks per head; the paged layout
   carries one page list per request, so a per-head keep-set has no
   representable consumer. `quest-attention` documents the same collapse.
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
