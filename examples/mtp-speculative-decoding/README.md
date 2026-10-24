# mtp-speculative-decoding

Speculative decoding with a model's native multi-token-prediction heads: draft
several future tokens from the MTP heads, verify them with the main logits, and
commit only the accepted prefix.

## Source

Gloeckle, Idrissi, Rozière, Lopez-Paz and Synnaeve, ***Better & Faster Large
Language Models via Multi-token Prediction*** (ICML) —
<https://arxiv.org/abs/2404.19737>. Implements the inference-time
self-speculative use of MTP heads described in §3.2.

**Faithfulness: Structurally faithful.** The loop drafts from the head's chain
(`mtp_drafts`) and verifies against the trunk's argmax, so its text equals greedy
decoding whatever the head proposes. `tests/examples/test_eagle.py` runs it
end-to-end on a drafting artifact.

## What it does

One prefill pass emits the normal next token and `k` draft tokens. Each later
round embeds a fixed `k + 1` window: the pending correct token followed by the
head's drafts. The main logits verify the drafts in order; the accepted prefix is
committed, and the first rejected position supplies the correction for the next
round. All of it runs in the epilogue; the host only drains committed tokens.

The pass always has the same shape. Rejected rows' slots are written again by the
next fire, which folds the run the last one accepted, and committed tokens are
returned as `-1`-padded windows for the host to unpad.

## The rule

```
drafts = the head's chain at the accepted row (mtp_drafts)
truth  = argmax(main logits at each draft position)
m      = length of matching draft prefix
emit   = window[0 .. m]              # pending token plus accepted drafts
next   = truth[m] + fresh MTP drafts # correction/bonus plus new drafts
```

Only `m + 1` tokens advance the committed length.

## Parameters

| Name | Type | Default | Meaning |
| --- | --- | --- | --- |
| `prompt` | string | `"The quick brown fox jumps over"` | Prompt to send to the model |
| `max_tokens` | int | `64` | Number of generated tokens |
| `k` | int | `4` | Draft tokens per round, capped at the model's `mtp_depth`; `0` decodes greedily through the same loop |

`pad`, `trace` and `margin` are diagnostics; see `src/lib.rs`.

## Implementation notes

Drafting needs a load with a draft head (`mtp_depth > 0`). On a plain checkpoint
`k` caps at `0`, and the loop is sequential greedy decoding through the same
geometry, fire and commit arithmetic.

## Run

```bash
cargo build --release --target wasm32-wasip3
python tests/examples/run_all.py --engine cuda --model <model-path>
```

Details are in the `//!` header of [`src/lib.rs`](src/lib.rs).
