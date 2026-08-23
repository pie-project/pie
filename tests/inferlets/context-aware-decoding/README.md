# context-aware-decoding

Amplifies a retrieved document against the model's own parametric prior, so the
answer is grounded in the context rather than in what the model already
believes.

## Source

Shi, Han, Lewis, Tsvetkov, Zettlemoyer and Yih, ***Trusting Your Evidence:
Hallucinate Less with Context-aware Decoding*** —
<https://arxiv.org/abs/2305.14739>. Implements §2.2.

**Faithfulness: Exact (equivalent form).** See
`inference-time-algorithms/10-implementation-faithfulness-audit.md`.

## What it does

In retrieval-augmented generation the model is given a document and asked to
answer from it — and then frequently ignores it, answering from memorised
parametric knowledge instead. CAD makes that failure mode directly
addressable: it runs the query *with* the context and *without* it, and
amplifies the difference. Whatever the document contributes is scaled up;
whatever the model would have said anyway is scaled down.

Structurally this is the same contrastive shape as
[`classifier-free-guidance`](../classifier-free-guidance), but the two streams
differ in *what they condition on* rather than in guidance strength.

## The rule

```
logits = (1 + α) · logits_with_context − α · logits_without_context
```

`α = 0` recovers plain context-conditioned decoding exactly.

As with CFG, the paper writes this over log-probabilities and this works in
logits; the difference is a per-stream constant that cancels in the softmax.

## Parameters

| Name | Type | Default | Meaning |
| --- | --- | --- | --- |
| `context` | string | *(a short document)* | Retrieved document the answer must be grounded in |
| `query` | string | *(a question)* | Question asked of the context |
| `alpha` | float | `0.5` | Context amplification; `0.0` is plain context-conditioned decoding |
| `max_tokens` | int | `32` | Number of generated tokens |

## Cost

**13.12 ms/token, 3.98× the [`naive-baseline`](../naive-baseline) control** on
an L40S with Qwen3-0.6B — two forward passes, and neither can be pipelined
because the next input depends on their combined output. See the
[CFG README](../classifier-free-guidance/README.md#cost) for the breakdown;
it is inherent to contrastive decoding.

## Run

```bash
cargo build --release --target wasm32-wasip2
python tests/inferlets/run_all.py --driver cuda_native --model <model-path>
```

The suite includes an **identity control**: at `alpha = 0.0` the rule provably
reduces to plain context-conditioned decoding, so the reported `mean_kl` must
be exactly `0.0000`. It is.

Details are in the `//!` header of [`src/lib.rs`](src/lib.rs).
