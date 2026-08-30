# sliding-window-attention

Windowed decoding that lets each new token attend only to the recent past,
rather than to the whole accumulated KV cache.

## Source

Beltagy, Peters and Cohan, ***Longformer: The Long-Document Transformer*** —
<https://arxiv.org/abs/2004.05150>. Implements §3's sliding-window attention
pattern, the same windowed decoding later adopted at inference time by Mistral 7B
(<https://arxiv.org/abs/2310.06825>) and by StreamingLLM's recent-token window
(<https://arxiv.org/abs/2309.17453>, and see the sibling
[`attention-sink`](../attention-sink) inferlet). FlashInfer
(<https://arxiv.org/abs/2501.01005>) is the kernel substrate that makes this
variant cheap to execute, not its source.

**Faithfulness: Structural — demonstrates the mask, not physical cache eviction.**
The attended span is exactly windowed by position, but old KV pages stay
allocated and are only masked out.

## What it does

Full causal decoding grows the attended prefix by one token at every step. A
sliding window keeps latency and memory pressure predictable by discarding the
oldest positions from the attention pattern: each query sees the current prefix
only through its last `window_size` keys.

The important distinction is that the rule is positional, not score-based. There
is no attempt to keep whichever old tokens look important to the current query;
anything outside the window is hidden uniformly. That makes the mechanism simple
enough to express as ETA geometry and a dense boolean attention mask.

## The rule

```
visible(q, k) = k <= q and k + window_size > q

logits = model(tokens; attention_mask = visible)
token  = argmax(logits)
```

## Parameters

| Name | Type | Default | Meaning |
| --- | --- | --- | --- |
| `prompt` | string | `"Tell me a long story about a cat."` | User prompt to send to the model |
| `max_tokens` | int | `512` | Maximum number of generated tokens |
| `window_size` | int | `64` | Number of recent positions visible to each query |

## Implementation notes

Like `attention-sink`, this inferlet is possible because it selects KV by
**position**, which ETA's geometry exposes. KV policies that select by attention
score are outside the current surface area: `IntrinsicId` exposes logits,
embeddings and geometry, but no attention weights.

This is also one of the places where the KV page CSR contract matters most. The
page CSR is the wire's source of truth for the attended span; the engine computes
`kv_len = (page_count - 1) * page_size + last_page_len`. The correct idiom is
`page_count = ceil(kv_len / page_size)`. Declaring `[0, pool_pages]` would make
the lane attend through uninitialised cache cells, and the failure mode is
plausible text rather than an error.

## Run

```bash
cargo build --release --target wasm32-wasip2
python tests/inferlets/run_all.py --engine cuda_native --model <model-path>
```
