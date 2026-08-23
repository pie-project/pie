# beam-search

Fixed-width beam search: keep the best partial continuations by cumulative log
probability instead of committing to the single locally best token at each step.

## Source

Wiseman and Rush, ***Sequence-to-Sequence Learning as Beam-Search Optimization***
(EMNLP) — <https://arxiv.org/abs/1606.02960>. Implements the standard
left-to-right beam-search recurrence used by neural sequence decoders.

**Faithfulness: Exact for fixed-width cumulative-logprob beam search.** The
inferlet's width-1 identity check is documented in
`inference-time-algorithms/10-implementation-faithfulness-audit.md`.

## What it does

Greedy decoding chooses the best next token and never revisits that choice. Beam
search keeps `beams` live hypotheses. At each step it scores every beam-token
extension by parent cumulative score plus the token log-probability, selects the
best `beams` extensions globally, and carries their parent pointers forward.

The useful distinction is that lower-ranked beams are allowed to take tokens
that are not the local argmax if their total path score is still competitive.
That gives the search a chance to recover when the greedy first choice leads to
a worse continuation, while still bounding the work by a fixed beam width.

## The rule

```
score_0(beam 0) = 0
score_0(other beams) = -∞

for each step t:
  cand_score(parent, x) = score_t(parent) + log p(x | parent prefix)
  keep = top `beams` entries over all parent × token candidates
  prefix_{t+1}(lane) = prefix_t(parent(lane)) + token(lane)
  score_{t+1}(lane) = cand_score(parent(lane), token(lane))
```

At `beams = 1`, this reduces to greedy decoding because `log_softmax` and adding
a scalar score preserve the logits argmax.

## Parameters

| Name | Type | Default | Meaning |
| --- | --- | --- | --- |
| `max_tokens` | int | `16` | Number of generated tokens |
| `beams` | int | `2` | Beam width; `1` degenerates to greedy decoding |

There is deliberately no `prompt` parameter. This inferlet starts from the fixed
Qwen BOS token.

## Implementation notes

The KV cache is a shared fixed pool. Forking is logical: each survivor inherits
its parent's boolean attention mask, appends one new pool cell, and updates a
parent pointer emitted by the epilogue. Dead cells are not compacted, so
`max_tokens` is bounded by the fixed pool capacity.

The suite includes the beam identity as a real check. With `beams = 1`,
`beam-search` produced **0 mismatches** against greedy decoding over 16 steps.
At width 4 it produced 46 mismatches and a better cumulative log-probability
(-13.68 versus -19.80 for width 1), demonstrating both the identity at width 1
and genuine search behaviour at larger widths.

## Run

```bash
cargo build --release --target wasm32-wasip2
python tests/inferlets/run_all.py --driver cuda_native --model <model-path>
```

Details are in the `//!` header of [`src/lib.rs`](src/lib.rs).
