# mtp-speculative-decoding

Speculative decoding with a model's native multi-token-prediction heads: draft
several future tokens from the MTP heads, verify them with the main logits, and
commit only the accepted prefix.

## Source

Gloeckle, Idrissi, Rozière, Lopez-Paz and Synnaeve, ***Better & Faster Large
Language Models via Multi-token Prediction*** (ICML) —
<https://arxiv.org/abs/2404.19737>. Implements the inference-time
self-speculative use of MTP heads described in §3.2.

**Faithfulness: Structurally faithful, unverified end-to-end.** The loop uses
`mtp_logits(k)` as the draft source and verifies against target logits, but this
environment's `Qwen/Qwen3-0.6B` model has no MTP head, so hardware correctness is
not verified.

## What it does

One prefill pass emits the normal next token and `k` MTP draft tokens. Each later
round embeds a fixed `k + 1` window: the pending correct token followed by the
previous round's drafts. The main logits verify the drafts in order; the accepted
prefix is committed, and the first rejected position supplies the correction for
the next round.

The pass always has the same shape. Rejected draft KV cells are left above the
advanced logical length and overwritten by the next fire, while committed tokens
are returned as `-1`-padded windows for the host to unpad.

## The rule

```
drafts = argmax(MTP heads 1..k)
truth  = argmax(main logits at each draft position)
m      = length of matching draft prefix
emit   = window[0 .. m]              # pending token plus accepted drafts
next   = truth[m] + fresh MTP drafts # correction/bonus plus new drafts
```

Only `m + 1` tokens advance the loop-carried KV length.

## Parameters

| Name | Type | Default | Meaning |
| --- | --- | --- | --- |
| `prompt` | string | `"The quick brown fox jumps over"` | Prompt to send to the model |
| `max_tokens` | int | `64` | Number of generated tokens |
| `k` | int | `4` | Number of MTP draft tokens per round; must be between `1` and `32` |

## Implementation notes

The inferlet depends on `intrinsics::mtp_logits(k)`. It cannot be exercised with
ordinary next-token-only checkpoints; on the available `Qwen/Qwen3-0.6B` model it
is therefore **untested end-to-end**. Treat the current code as a PTIR expression
of the algorithm and a wiring test for an MTP-capable model, not as a measured
green path.

## Run

```bash
cargo build --release --target wasm32-wasip2
python tests/inferlets/run_all.py --driver cuda_native --model <model-path>
```

Details are in the `//!` header of [`src/lib.rs`](src/lib.rs).
