# constrained-speculative-decoding

Grammar-constrained decoding and speculative decoding, running together on one
token stream — and a proof that the combination changes nothing but the number
of forward passes.

## Source

Two independent lines of work, composed:

- **Constrained decoding.** Willard and Louf, ***Efficient Guided Generation for
  Large Language Models*** — <https://arxiv.org/abs/2307.09702>. The
  JSON-Schema path is the same one used by
  [`json-schema-constrained-decoding`](../json-schema-constrained-decoding).
- **Speculative decoding.** Leviathan, Kalman and Matias, ***Fast Inference from
  Transformers via Speculative Decoding*** — <https://arxiv.org/abs/2211.17192>,
  with the drafter replaced by prompt lookup as in CacheBack /
  [`cacheback-speculative-decoding`](../cacheback-speculative-decoding).

There is no single paper for the composition. What this inferlet demonstrates is
the *interface* the composition needs, which is where most implementations get
stuck.

## Why the composition is hard

Constrained decoding is **positional**: the set of legal tokens at position `t`
depends on every token accepted before `t`. Speculative decoding proposes tokens
for positions `t … t+k` and verifies all of them in **one** forward pass. So the
verifier needs `k+1` different masks inside a single pass, which means the
grammar has to be advanced over the draft *before* anyone knows whether the
draft is correct — and then wound back to whatever was actually accepted.

A grammar matcher that only offers `accept` and `reset` cannot do this. The
rollback has to be emulated by resetting and replaying the entire accepted
prefix on every rejection, which costs more than speculation saves. That is why
these two features are usually offered but not offered *together*.

PTIR's matcher resource therefore exposes two extra operations, added for this
composition (`interface/inferlet/grammar.wit`):

| Operation | Role here |
| --- | --- |
| `fork()` | Take a reference copy of the grammar state before the speculative walk |
| `rollback(n)` | Undo the walk once the verifier has ruled |
| `rollback-capacity()` | The retained history bound, so deep drafts fail loudly instead of silently |

## What it does

Each step:

1. **Draft.** Longest-suffix prompt lookup over the committed history proposes
   up to `draft_length` tokens.
2. **Walk the grammar.** The matcher is advanced over the proposal, recording
   one allowed-token mask per position. A proposal token the grammar forbids is
   dropped on the spot — it could never have been accepted, so it does not earn
   a readout row. This pruning is reported as `grammar_pruned` and is a saving
   unique to the composition.
3. **Rewind and check.** The walk is rolled back in full and the restored mask
   is compared against the `fork()` taken in step 2. A rollback that silently
   failed to restore state would corrupt the constraint rather than error, so
   the invariant is asserted every step, not assumed.
4. **Verify.** One target forward over `committed ++ draft` reads out `k+1` rows
   and takes a **grammar-masked** argmax at each, using that row's mask.
5. **Accept.** A draft token survives only where it equals the target's own
   masked argmax. On the first mismatch the target's token is taken instead and
   the rest of the draft is discarded.

The KV working set is rebuilt per verification window, so rejected draft state
can never leak into a later step.

### Constraint satisfaction is structural

The correction token emitted on a rejection is a masked argmax, so it is legal
by construction. A rejection cannot push the grammar into an invalid state, and
the implementation needs no post-hoc repair step. The final output is
additionally parsed as JSON before being returned.

## The correctness property, and how to test it

Verification is greedy and masked at every row, so a draft token is kept only
when it is what the target model would have produced anyway under the same mask.
Speculation is therefore a pure latency optimization *on top of* constrained
decoding: it must change how many forward passes run and **nothing else**.

`draft_length = 0` makes that testable without a second inferlet. The drafter
returns empty, `verify` collapses to a one-row masked readout, and the loop
degenerates to sequential constrained greedy decoding — through the same prompt,
the same schema and the same `verify()` call. Running `0` against `4` is a
controlled A/B in which the emitted token sequence must be **identical**.

`tests/inferlets/test_curated.py::test_constrained_speculative_decoding` asserts
exactly that, and additionally asserts that speculation actually fired, was
rejected at least once (otherwise the reject path is never exercised), and ran
strictly fewer forward passes.

## Parameters

| Name | Type | Default | Meaning |
| --- | --- | --- | --- |
| `prompt` | string | *(a profile request)* | Description of the JSON value to generate |
| `schema` | string | *(a 3-field object)* | JSON Schema the output must satisfy |
| `max_tokens` | int | `256` | Generated-token budget |
| `draft_length` | int | `4` | Tokens proposed per verification; `0` disables speculation |
| `max_ngram` | int | `8` | Longest suffix the prompt-lookup drafter will match |

## Reported fields

`verification_steps` is the count of target forwards. At `draft_length = 0` it
equals `count`; below `count` means speculation is paying off. `grammar_pruned`
counts draft tokens the grammar rejected before verification. `rollback_checks`
counts how many times the fork/rollback invariant was checked and held.

## Run

```bash
cargo build --release --target wasm32-wasip2
python tests/inferlets/run_all.py --engine cuda_native --model <model-path>
```

Details are in the `//!` header of [`src/lib.rs`](src/lib.rs).
