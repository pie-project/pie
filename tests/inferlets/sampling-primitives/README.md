# sampling-primitives

Regression test for the generated ETA sampling primitives. This is not an
algorithm inferlet; it exists because one un-fused sampler path used to hang the
GPU.

## Purpose

`pivot_threshold(cummass_le(...))` originally only worked when the compiler could
recognize the full `LibraryOp::NucleusSample` pattern. If that exact match broke
— for example because a stray `broadcast` appeared between `mask_apply` and
`reduce_argmax` — the compiler fell back to a generated reference path whose
`cummass_le` arm was a **thread-0-only O(len³) selection sort**. At a
151936-token vocabulary that path did not produce a wrong answer; it never
terminated, leaving the GPU pegged.

The fix ported the block-cooperative selection loop into the generated path.
This crate pins that fix by publishing the nucleus keep-mask as a standalone
value in a shape that **cannot** match the library nucleus pattern, forcing the
generated `pivot_threshold` path. See "A GPU hang in the generated sampler path"
in
`inference-time-algorithms/10-implementation-faithfulness-audit.md`.

## What it validates

One forward pass emits the greedy token, raw logits, probabilities,
log-probabilities, entropy, and a top-p keep-mask. The host then checks that:

- token, logits and probabilities agree on the argmax;
- probability and log-probability agree for the selected token;
- entropy is finite and in range;
- the keep-mask is boolean, non-empty, vocabulary-sized, and a descending
  probability prefix;
- `cummass_le(0.9)` keeps the minimal prefix whose mass reaches the threshold.

## The rule

```
logprobs = log_softmax(logits)
probs    = exp(logprobs)
entropy  = -Σ probs * logprobs
keep     = pivot_threshold(probs, cummass_le(0.9))
token    = argmax(logits)
```

The keep-mask is the point of the test: it is not fed into a Gumbel-max tail, so
the library-fused nucleus sampler cannot claim it.

## Parameters

None. The input struct has no fields; the prompt is fixed to `"The capital of
France is"` and `top_p` is fixed to `0.9`.

## Implementation notes

The assertions avoid depending on exact floating-point summation order. They
check the nucleus contract by comparing kept and dropped probabilities and by
verifying that the kept mass reaches `top_p` while the mass without the last kept
token does not.

## Run

```bash
cargo build --release --target wasm32-wasip2
python tests/inferlets/run_all.py --engine cuda_native --model <model-path>
```

Details are in the `//!` header of [`src/lib.rs`](src/lib.rs).
