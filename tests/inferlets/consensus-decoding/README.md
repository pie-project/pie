# consensus-decoding

Generate several candidate answers from the same prompt, compare their final
answers pairwise, and return the response whose answer is most central to the
set.

## Source

Wang et al., ***Self-Consistency Improves Chain of Thought Reasoning in Language
Models*** (ICLR 2023) — <https://arxiv.org/abs/2203.11171> — is the algorithm
being implemented: sample several reasoning chains independently, then take the
answer they agree on. Because the answers here are free-form strings rather than
a closed label set, the agreement step is the similarity-based generalisation of
Chen et al., ***Universal Self-Consistency for Large Language Model Generation***
— <https://arxiv.org/abs/2311.17311> — which is in turn exactly the
sample-and-rerank decision rule that Bertsch et al., ***It's MBR All the Way
Down*** — <https://arxiv.org/abs/2310.01387> — identifies as minimum Bayes risk
with a string-similarity utility.

**Faithfulness: Structural — consensus reranking, not a learned verifier.** It
uses parallel top-p samples and host-side normalized Levenshtein centrality as a
simple MBR-style utility over extracted answers.

## What it does

Self-consistency-style decoding spends extra inference compute on diversity: it
samples multiple reasoning traces instead of trusting one. This inferlet asks the
model for step-by-step reasoning and a final answer, then compares only the text
after the last `Final Answer:` marker.

Instead of plurality voting over exact strings, it computes pairwise normalized
Levenshtein similarity and selects the candidate with the highest mean similarity
to the others. That makes near-equivalent free-form answers reinforce one another
without requiring a verifier model.

## The rule

```
for c in 1..B:
  y_c = top-p sample(prompt, temperature, top_p)
  a_c = text after the last "Final Answer:" marker, or the full response

utility(i, j) = normalized_levenshtein(a_i, a_j)
centrality(i) = mean_{j != i} utility(i, j)
pick          = argmax_i centrality(i)
```

With one candidate, centrality is defined as `1.0` and the lone response wins.

## Parameters

| Name | Type | Default | Meaning |
| --- | --- | --- | --- |
| `question` | string | `"What is 17 * 24 + 13?"` | User question wrapped in the reasoning system prompt |
| `num_candidates` | int | `5` | Number of parallel candidate responses |
| `max_tokens` | int | `1024` | Maximum generated tokens per candidate |
| `temperature` | float | `0.9` | Sampling temperature, clamped to at least `1e-4` |
| `top_p` | float | `0.95` | Nucleus mass, clamped to `[0, 1]` |

## Implementation notes

The shared prompt prefix is prefilled once. Candidate lanes share those KV cells,
and the decode loop uses per-lane masks so each continuation attends the shared
prefix plus its own appended cells. The top-p sampling in the decode loop is
really `[B, vocab]`, with independent per-lane Gumbel noise.

The prefill deliberately samples a single `[1, vocab]` read-out row and starts
all `B` candidates from that same first token. Broadcasting the prefill row to
`[B, vocab]` would trigger the nucleus-sampler scratch-elision bug documented in
`inference-time-algorithms/11-ptir-limits.md`
under "Four unchecked contracts that fail silently", contract 4. The verified
behaviour is that at `temperature = 1.2` candidates diverge by the second
sentence; at `0.6` they are identical because the nucleus keep-set is a single
token.

## Run

```bash
cargo build --release --target wasm32-wasip2
python tests/inferlets/run_all.py --driver cuda_native --model <model-path>
```

Details are in the `//!` header of [`src/lib.rs`](src/lib.rs).
