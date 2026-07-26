# attention-sink

Streaming attention that keeps the first few tokens visible forever while the
rest of the context rolls through a fixed recent-token window.

## Source

Xiao et al., ***Efficient Streaming Language Models with Attention Sinks***
(NeurIPS 2024) — <https://arxiv.org/abs/2309.17453>. Implements the
StreamingLLM sink-plus-window cache policy.

**Faithfulness: Structural — demonstrates the mask, not bounded-memory eviction.**
It applies the paper's positional visibility rule, but masked KV pages remain
allocated in this example.

## What it does

A pure sliding window forgets the beginning of the prompt. StreamingLLM's
observation is that the first few tokens act as attention sinks: later tokens
continue to route probability mass through them even when their semantic content
is not special. Keeping those positions restores stability while still bounding
what each decode step may attend to.

This inferlet pre-fills the prompt normally, then decodes with a custom attention
mask. A query may see keys that are either in the first `sink_size` positions or
within the last `window_size` positions. Everything else is hidden from
attention, although its backing KV pages are left in the working set.

## The rule

```
visible(q, k) = k <= q and (k < sink_size or k + window_size > q)

logits = model(tokens; attention_mask = visible)
token  = argmax(logits)
```

## Parameters

| Name | Type | Default | Meaning |
| --- | --- | --- | --- |
| `prompt` | string | `"Tell me a long story about a cat."` | User prompt to send to the model |
| `max_tokens` | int | `512` | Maximum number of generated tokens |
| `sink_size` | int | `4` | Number of initial positions that remain visible forever |
| `window_size` | int | `64` | Number of recent positions visible outside the sink |

## Implementation notes

This is in the suite because it selects KV entries by **position**, which PTIR's
geometry exposes. The attention-score eviction family — H2O, SnapKV, TOVA,
Quest, RetrievalAttention — is not implementable today because `IntrinsicId`
exposes logits, embeddings and geometry but no attention weights.

The page CSR has to track the logical KV length exactly. The CUDA driver derives
`kv_len = (page_count - 1) * page_size + last_page_len` from the page CSR, not
from the `KvLen` port, so the inferlet uses `page_count = ceil(kv_len / page_size)`
instead of declaring the whole page pool. Over-declaring pages makes attention
read uninitialised KV and produces fluent garbage; this is contract 3 in
[`11-ptir-limits.md`](../../../inference-time-algorithms/11-ptir-limits.md).

## Run

```bash
cargo build --release --target wasm32-wasip2
python tests/inferlets/run_all.py --driver cuda_native --model <model-path>
```
