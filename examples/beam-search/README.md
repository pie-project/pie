# beam-search

Fixed-width beam search: keep the best partial continuations by cumulative log
probability instead of committing to the single locally best token at each step.

## Source

Wiseman and Rush, ***Sequence-to-Sequence Learning as Beam-Search Optimization***
(EMNLP) — <https://arxiv.org/abs/1606.02960>. Implements the standard
left-to-right beam-search recurrence used by neural sequence decoders.

**Faithfulness: Exact for fixed-width cumulative-logprob beam search.**

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
| `prompt` | string | a short request | User message, rendered with the model's chat template |
| `system` | string | a helpful-assistant prompt | System message |
| `no_think` | bool | `false` | Turn a thinking model's thinking off |
| `prompt_tokens` | int list | none | The prompt as token ids, in place of the templated `prompt` |

The search runs `max_tokens` steps; a beam that emits end of sequence is not
retired, so width 1 is greedy decoding of the same prompt to the same length.

## Implementation notes

The prompt is prefilled once, and that fire's epilogue picks the first `beams`
tokens (the top of the prompt's next-token distribution). The beams then share
one working set, a pool repeated for every lane and reserved for the whole
search (the prompt plus one slot per beam per step). Each lane reads the pool
up to its own slot, where every ancestor stands, and its mask hides the other
beams' cells. Forking is logical:
each survivor inherits its parent's boolean attention mask, appends one new pool
slot chosen on the device, and updates a parent pointer emitted by the epilogue.
Dead cells are masked, not compacted. On a model with recurrent layers the same
program runs: each beam's recurrent layers replay the slots its mask admits, its
ancestry, before its own token.

The inferlet checks the beam identity itself: at `beams = 1` every step's pick
must equal the raw-logit argmax, and the run fails otherwise. Wider beams pick
non-argmax continuations for their lower lanes and reach better cumulative
log-probabilities.

## Run

```bash
cargo build --release --target wasm32-wasip3
python tests/examples/run_all.py --engine cuda --model <model-path>
```

Details are in the `//!` header of [`src/lib.rs`](src/lib.rs).
