# quest-attention

Query-aware sparse attention: score every KV page by an upper bound on how much
attention it could receive, then attend only to the top pages.

## Source

Tang et al., ***Quest: Query-Aware Sparsity for Efficient Long-Context LLM
Inference*** (ICML 2024) — <https://arxiv.org/abs/2406.10774>.

**Faithfulness: Criticality scoring is exact; page selection is observed, not
yet enforced.** The per-page criticality bound this inferlet computes is Quest's
own, evaluated on the real KV cache at every layer of every decode step. The
selection it derives is reported rather than applied — attention still reads the
full page list — so this demonstrates the scoring rule and its ranking quality,
not the speedup.

## What it does

Quest's observation is that the pages a decode step actually needs depend on the
*current query*, and that you can decide which pages those are without reading
them. Each page keeps an element-wise **envelope** of its keys — a per-dimension
`min` and `max` over the page's live tokens. For a query `q`, the largest dot
product any key in that page could produce is bounded above by

```
upper(q, page) = Σ_d max(q_d · min_d, q_d · max_d)
```

because each coordinate's contribution is maximised at one end of its interval.
That bound costs `O(head_dim)` per page instead of `O(page_size · head_dim)`,
and it is what Quest ranks pages by.

This inferlet installs a tap at every attention projection, computes the bound
for all pages against that layer's post-RoPE query, and reduces it over layers.
At the end of each decode step it reports the per-page scores and the page set
Quest would have kept.

## The rule

```
score[page] = max over layers of ( max over kv_heads of
                Σ_{qh in group} Σ_d max(q·env_min, q·env_max) )

keep = top-(budget-1) pages by score, plus the in-flight last page
```

The last page is force-kept because it holds the tokens being written right now;
its envelope is deliberately `+inf` ("always keep") until the fire that fills it
completes.

## Deviations from the paper

Three, all recorded here because each one changes what the numbers mean.

**1. The score is a union over heads, not per-head.** Quest selects top-K pages
independently for each attention head. This engine's paged KV has *one* page
list per request, and the custom-mask offset (`qo_idx * kv_len + kv_idx`) has no
head index, so a per-head selection has nowhere to live. The kernel therefore
takes the max over KV heads: a page is kept if *any* head wants it. Quality is
at least Quest's (no head loses a page it wanted); the speedup is at most
Quest's (the kept set is a union, so it is larger).

**2. In-flight pages score `+inf`, not a real bound.** A page whose tokens are
still being appended by the current fire has no settled envelope. Rather than
score it from a partially written page, the kernel pins it. `pages_pinned`
in the report is exactly this count and should always be 1 for a single
request.

**3. Selection is reported, not enforced.** See the faithfulness note above.

## Operator opt-in

The envelopes cost `2 / page_size` of the KV cache — 12.5% at `page_size = 16` —
and they must be allocated *with* the pages, because the KV pool is sized to
consume the device and because a page written before its envelope existed would
keep the empty seed and score `+inf` forever. So they are an explicit opt-in:

```
PIE_CUDA_KV_ENVELOPES=1
```

Without it the engine does not advertise `has_kv_envelopes`, and this inferlet
fails at **bind** — `backend does not provide this kernel/sink` — rather than at
its first fire. Envelope maintenance itself rides the existing KV append and is
free within measurement noise; the cost of the tap is the per-layer scoring.

The capability additionally requires a native BF16 NHD page layout, a post-RoPE
query (which excludes the MLA and gated-delta families), and `tp_size == 1`
(envelopes are per-rank but the page list is shared).

## How the layer loop closes

An inferlet cannot ask the model how many layers it has, so it cannot publish
one score vector per layer into a fixed-capacity host channel. Instead the fold
happens on-device: the per-layer tap takes a `[p_max]` f32 accumulator, folds
`max(previous, envelope_dot(...))` into it, and puts it back. The epilogue —
which fires exactly once per fire — drains the accumulator, publishes it to the
host, and re-seeds it with `-inf`. `layers_observed` counts the firings and is
the proof the tap ran on every layer.

## Inputs

| Name | Default | Meaning |
|---|---|---|
| `prompt` | — | The prompt. Should be long enough to fill several pages. |
| `max_tokens` | 8 | Tokens to decode. |
| `page_budget` | 4 | Pages Quest would keep, including the in-flight page. |

## Report

| Field | Meaning |
|---|---|
| `layers_observed` | Tap firings in the last step. Must equal the model's layer count. |
| `page_scores` | Per-page criticality, as strings so `inf`/`-inf`/`nan` survive JSON. |
| `pages_finite` / `pages_pinned` / `pages_absent` / `pages_nan` | Score breakdown. `pages_nan` is always a bug. |
| `kept_pages` | The page set Quest would have attended to. |
