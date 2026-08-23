# asap-grammar-aligned-decoding

ASAp — removes the distribution distortion that hard grammar masking
introduces, so constrained output is not just *valid* but correctly
*distributed*.

## Source

Park, Wang, Berg-Kirkpatrick, Polikarpova and D'Antoni, ***Grammar-Aligned
Decoding*** (NeurIPS 2024) — <https://arxiv.org/abs/2405.21047>. Implements
Eq. 3–4 and Algorithm 1.

**Faithfulness: Exact (equivalent form).** See
`inference-time-algorithms/10-implementation-faithfulness-audit.md`.

## What it does

Standard grammar-constrained decoding (GCD) masks out every token the grammar
forbids and renormalizes. That guarantees validity but it is *not* the model's
distribution conditioned on validity — it is greedy locally and biased
globally, because a token that looks good now may lead only to dead ends the
mask has not reached yet.

ASAp fixes this by iterating. Each round records how much probability mass each
prefix actually reached, and the next round uses those recorded
*approximations* to discount prefixes that turned out to be traps. The
approximation is provably non-decreasing in the round index and converges to
the true grammar-conditioned distribution.

Round 1 is exactly plain GCD, so the round trace shows you what the correction
is worth.

## The rule

```
round 1:      p₁(x | prefix) = GCD mask, renormalized
round k+1:    reweight each prefix by the mass its subtree was observed to
              reach in rounds 1..k, then renormalize
```

The reported `root_alpha_trace` is the sequence of root approximation masses,
and `monotone` asserts the paper's guarantee that it never decreases.

## Parameters

| Name | Type | Default | Meaning |
| --- | --- | --- | --- |
| `prompt` | string | *(a JSON request)* | Prompt to send to the model |
| `schema` | string | *(a small object schema)* | JSON Schema the output must satisfy |
| `rounds` | int | `6` | ASAp refinement rounds, `1..16`; round 1 is plain GCD |
| `max_tokens` | int | `64` | Token budget per round |
| `seed` | int | — | Sampling RNG key |

## Deviation from the reference

None in the maths. The paper's `EXPAND` walks a shared trie in place; this
rebuilds the path each round from a stored `Vec<Vec<u32>>` of approximations.
That is the same recurrence with different storage, and it is an exact
representation rather than a truncation.

## Cost

**Not benchmarked.** The two-point regression used for every other inferlet
here measures a marginal per-token cost, and ASAp's multi-round shape does not
fit that model — it regenerates the whole sequence `rounds` times. As a rough
guide, expect roughly `rounds ×` a constrained single-pass decode.

## Run

```bash
cargo build --release --target wasm32-wasip2
python tests/inferlets/run_all.py --driver cuda_native --model <model-path>
```

The test asserts `monotone`, which is the paper's actual guarantee — a
regression there means the trie bookkeeping is wrong, which no output-validity
check would catch.

Details are in the `//!` header of [`src/lib.rs`](src/lib.rs).
