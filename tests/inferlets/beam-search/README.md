# beam-search

Fixed-width beam search: keep the best partial continuations by cumulative log
probability instead of committing to the single locally best token at each step.

## Source

Wiseman and Rush, ***Sequence-to-Sequence Learning as Beam-Search Optimization***
(EMNLP) — <https://arxiv.org/abs/1606.02960>. Implements the standard
left-to-right beam-search recurrence used by neural sequence decoders.

**Faithfulness: Exact for fixed-width cumulative-logprob beam search.** The
inferlet's width-1 identity check is documented in
[`10-implementation-faithfulness-audit.md`](../../../inference-time-algorithms/10-implementation-faithfulness-audit.md).

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
| `mask` | bool | `true` | Bind the ancestry mask to the attention port; `false` is the deliberately-broken control described below |

There is deliberately no `prompt` parameter. This inferlet starts from the fixed
Qwen BOS token.

### `mask = false` is a control, not a mode

The whole program is identical in both arms — the epilogue still evolves the
mask — and only the attention port's binding changes. With the mask unbound
every beam attends the entire filled pool instead of its own ancestry, so the
search is wrong by construction.

It exists because a driver that ignores the mask cannot be detected any other
way. `ModelCapabilities` carries no `supports_custom_mask` flag, and a model
family whose forward never reads `custom_mask_d` does not fail — it returns
fluent, plausible, meaningless output at a perfectly ordinary speed. Differencing
the two arms by the returned beam's token digest is what distinguishes "the mask
worked" from "the mask was dropped on the floor".

Note the difference is vacuous at `beams = 1`: a single beam's ancestry is the
whole filled span, so masked and unmasked attention cover the same cells and the
two arms agree. The control only carries signal at `beams >= 2`.

## Output

Three lines: the decoded best hypothesis, then

```
[beam] width=<B> steps=<N> best_score=<f> greedy_mismatches=<n>
[beam] mask=<bool> kv_cells_occupied_peak=<n> returned_tokens=<id,id,…>
```

`returned_tokens` carries the best beam's raw token ids so a consumer can digest
them directly; re-tokenizing the decoded text is not equivalent, because
detokenize→tokenize is not the identity.

`kv_cells_occupied_peak` is the shared pool's occupancy, `1 + width * (steps - 1)`.
The write descriptors are loop-carried — epilogue *j* publishes what fire *j+1*
consumes — and fire 0 uses the seeded descriptors, where every lane writes the
one shared BOS cell at position 0. So `steps` fires reach a highest flat position
of `(steps - 1) * width`, and the last epilogue's `fill` is published to a fire
that never runs. It is **derived** from the width and the number of steps that
actually completed, not read back from the device: `fill` is loop-carried and
never drained, and adding a host round-trip per step to observe it would perturb
the decode timing this figure accompanies.

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
