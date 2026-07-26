# Inference-time algorithms

A literature survey of inference-time algorithms — decoding, sampling, search,
speculation, KV/attention manipulation, constrained generation, activation
steering — read through one question:

> **Which of these need engine-level control, and which of those does Pie
> uniquely give you?**

Anchor paper: Welleck, Bertsch, Finlayson, Schoelkopf, Xie, Neubig, Kulikov,
Harchaoui, *From Decoding to Meta-Generation: Inference-time Algorithms for
Large Language Models*, [arXiv:2406.16838](https://arxiv.org/abs/2406.16838),
TMLR 2024.

Pie itself: [arXiv:2510.24051](https://arxiv.org/abs/2510.24051), SOSP'25.

## Contents

| File | What's in it |
|---|---|
| [`00-pie-capability-map.md`](./00-pie-capability-map.md) | What Pie actually exposes to inferlets, grounded in this repo, and the eight expressiveness axes the rest of the docs cite |
| [`01-survey-taxonomy-and-successors.md`](./01-survey-taxonomy-and-successors.md) | The anchor survey's full taxonomy + later surveys (2024-2026) |
| [`02-token-level-decoding-sampling.md`](./02-token-level-decoding-sampling.md) | Truncation samplers, contrastive/multi-distribution decoding, repetition control, MAP/MBR, watermarking, adaptive temperature |
| [`03-search-and-meta-generation.md`](./03-search-and-meta-generation.md) | Best-of-N, self-consistency, ToT/GoT, MCTS, PRM/value-guided search, test-time scaling, refinement loops, multi-agent |
| [`04-speculative-decoding-and-efficiency.md`](./04-speculative-decoding-and-efficiency.md) | Speculative decoding families, self-drafting, tree verification, MTP, early exit, cascades |
| [`05-kv-cache-and-attention.md`](./05-kv-cache-and-attention.md) | Attention sinks, KV eviction/compression, query-aware sparse attention, prefix/non-prefix cache reuse, long-context restructuring |
| [`06-constrained-decoding-and-steering.md`](./06-constrained-decoding-and-steering.md) | Grammar/JSON/regex constrained decoding and its distortion problem; activation steering, ITI/CAA/RepE, SAE steering, uncertainty control |
| [`07-programmable-serving-and-emerging.md`](./07-programmable-serving-and-emerging.md) | vLLM/SGLang/Orca/LMQL/Guidance/Parrot and where their expressiveness stops; diffusion-LM decoding, latent reasoning, adaptive compute |
| [`08-pie-uniqueness-matrix.md`](./08-pie-uniqueness-matrix.md) | **Synthesis** — the three tiers, the eight axes, and the highest-leverage open projects |
| [`candidates.md`](./candidates.md) | The build queue: every algorithm scored for PTIR feasibility, with implementation status |
| [`09-ptir-unbuilt-algorithms.md`](./09-ptir-unbuilt-algorithms.md) | What PTIR still cannot express, and the exact engine primitive each blocked algorithm is waiting on |
| [`10-implementation-faithfulness-audit.md`](./10-implementation-faithfulness-audit.md) | **Every built inferlet checked line by line against its paper's published equation** |
| [`11-ptir-limits.md`](./11-ptir-limits.md) | Where PTIR strains, measured from thirty built inferlets: fusion, performance, expressiveness |

**Start with `08`** if you want the conclusion, `00` if you want the mechanism,
`01` if you want the map of the field.

## Citation integrity

These reports were produced by parallel research agents, so every arXiv
citation was mechanically re-checked afterwards: `verify_citations.py` parses
each `(claimed title, arXiv id)` pair, fetches the registered title from the
arXiv API, and reports mismatches, non-existent IDs, and cases where a link
label disagrees with its URL.

```bash
python3 verify_citations.py *.md
```

Current status — **289 citation entries across 252 unique papers, 0 unresolved**:

| File | Citations | Notes |
|---|---|---|
| `01-taxonomy.md` | 10 | clean |
| `02-token-level-decoding-sampling.md` | 41 | 1 real error found and fixed (invalid id `2019.09751` → `1904.09751`) |
| `03-search-and-meta-generation.md` | 54 | clean |
| `04-speculative-and-efficiency.md` | 42 | 2 titles corrected (DeepSeek-V3 report; SmartSpec→TurboSpec rename) |
| `05-kv-cache-and-attention.md` | 65 | 2 resolved (CacheBlend was conflated with a different paper; RetrievalAttention located) |
| `06-constrained-decoding-and-steering.md` | 46 | clean |
| `07-programmable-serving-and-emerging.md` | 31 | clean |

Residual diffs reported by the tool are section-heading nicknames (e.g. "vLLM —
PagedAttention" vs the registered title) and are documented in each file's
*Citation audit* section. Re-run the script before quoting anything; arXiv
titles do get revised.

## Scope and caveats

- This is a **literature survey**, not a design doc or a roadmap. The "open
  projects" list in `08` is a set of candidates, not commitments.
- Claims of the form "not expressible on vLLM/SGLang" always mean *without
  patching engine internals*, and reflect those systems as described in the
  papers surveyed in `07`. Both move fast — re-check before citing.
- Coverage is deliberately broad rather than deep. Each entry is a pointer with
  enough context to decide whether to read the paper.
