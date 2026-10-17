# attention-sink

Streaming attention that keeps the first few tokens visible forever while the
rest of the context rolls through a fixed recent-token window.

## Source

Xiao et al., ***Efficient Streaming Language Models with Attention Sinks***
(NeurIPS 2024) — <https://arxiv.org/abs/2309.17453>. Implements the
StreamingLLM sink-plus-window cache policy.

**Faithfulness: Structural — demonstrates the mask, not bounded-memory eviction.**
It applies the paper's positional visibility rule, but masked KV remains
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
attention, although its KV is left in the working set.

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
| `sink_size` | int | `4` | Number of initial positions that remain visible forever; `0` is a plain sliding window |
| `window_size` | int | `64` | Number of recent positions visible outside the sink |

## Output

One JSON object, the shape the other KV-policy inferlets (and
`scripts/bench/pie_bench.py`) read: `text`, `count` (generated tokens),
`prompt_len`, `sink_size`, `window_size`, `kv_len` (KV positions live at the
last fire) and `visible_kv` (keys the last fire's query could see, at most
`sink_size + window_size`).

## Implementation notes

This policy selects KV entries by **position**, which a mask row states
directly. The score-based family — [`trackb-h2o`](../trackb-h2o),
[`trackb-snapkv`](../trackb-snapkv), [`tova-attention`](../tova-attention) —
ranks them by `intrinsics::attn_score` instead.

Every pass appends and folds what it wrote, and the mask is the whole policy. A mask row is a bool per slot of the
sequence; the decode epilogue rebuilds the next row on the device
(`sink_window_mask`) beside the token it feeds back.

## Run

```bash
cargo build --release --target wasm32-wasip2
python tests/examples/run_all.py --engine cuda_native --model <model-path>
```
