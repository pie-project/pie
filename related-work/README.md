# Related work

This directory surveys constrained-decoding research relevant to a
GPU-resident, heterogeneous-batch JSON Schema engine. The snapshot date is
**2026-07-24**.

The survey separates three claims that are often conflated:

1. the model forward pass runs on a GPU;
2. a packed grammar mask is applied to logits on a GPU;
3. grammar state, parser-stack transitions, mask construction, and sampling
   remain device-resident without a per-token CPU dependency.

Only the third meaning is called **GPU-native grammar execution** here.

## Main conclusions

- The established production architecture is still **CPU parser or matcher,
  optional host-to-device mask transfer, GPU mask application**. XGrammar,
  XGrammar-2, llguidance, and their vLLM/SGLang/TensorRT-LLM integrations use
  this split, although scheduling and overlap can make its end-to-end overhead
  small.
- [Pre3](https://aclanthology.org/2025.acl-long.551/) is the closest explicit
  LR(1) predecessor. It converts LR(1) transition graphs into a DPDA with
  prefix-conditioned edges, but its published contribution is an algorithmic
  parser optimization rather than a verified GPU-resident runtime.
- [PSC](https://conf.researchr.org/details/issta-2026/issta-2026-research-papers/81/Efficient-Grammar-Constrained-Decoding-via-Parser-Stack-Classification)
  is the closest parser-stack classification result. Its public abstract says
  that one stack classification produces the complete vocabulary mask with
  complexity independent of vocabulary size. The public material available at
  this snapshot does not state CPU/GPU placement.
- [Gram2Token](https://icml.cc/virtual/2026/poster/62392) is the closest direct
  prior art to the high-level gpu-lr1 idea. It preprocesses deterministic
  byte-level grammar execution into token categories and transition tables,
  then performs category lookup, masking, and state update on the GPU. Its
  abstract reports a 1.38x geometric-mean and 1.85x maximum throughput
  improvement under schema-diverse continuous batching. The declared code URL
  was not publicly reachable at this snapshot.
- gpu-lr1 now has three execution modes. The JSON Schema backend compiles a
  canonical, acyclic subset into a tokenizer-aware byte DFA. The LR backend
  compiles arbitrary deterministic canonical LR(1) grammars into sparse
  ACTION/GOTO tables and executes stack-dependent reductions on GPU over grammar
  terminals. A bounded compiler additionally expands reachable LR stacks into
  finite configurations and supports real tokenizer bytes for byte-terminal
  grammars; it exposes state-explosion rather than solving unbounded recursion.
- The defensible project contribution is therefore not "the first
  grammar-constrained decoder" or "the first GPU-aware structured-output
  system." It is a measured design for **GPU-resident automaton/parser stepping
  and sparse token or terminal selection over heterogeneous batches**, with
  explicit table, stack, and memory trade-offs.

## Documents

- [`papers.md`](papers.md): chronological, annotated paper and artifact survey.
- [`comparison.md`](comparison.md): architecture and device-placement matrix.
- [`evaluation.md`](evaluation.md): publication-grade evaluation checklist.
- [`references.bib`](references.bib): BibTeX for the cited academic work.

## Short timeline

| Year | Work | Primary contribution |
|---:|---|---|
| 2021 | [PICARD](https://aclanthology.org/2021.emnlp-main.779/) | Incremental parser guard for text-to-SQL decoding. |
| 2022 | [Synchromesh](https://arxiv.org/abs/2201.11227) | Constrained semantic decoding using an external completion engine. |
| 2023 | [Efficient Guided Generation](https://arxiv.org/abs/2307.09702) | Regex/FSM vocabulary index, the original Outlines formulation. |
| 2023 | [Grammar-Constrained Decoding for Structured NLP](https://aclanthology.org/2023.emnlp-main.674/) | General CFG and input-dependent grammar constraints. |
| 2024 | [SynCode](https://openreview.net/forum?id=HiUZtgAPoH) | Offline DFA mask store with soundness/completeness analysis. |
| 2024 | [DOMINO](https://proceedings.mlr.press/v235/beurer-kellner24a.html) | Fully subword-aligned constraints and opportunistic/speculative execution. |
| 2024 | [Automata-based Constraints](https://arxiv.org/abs/2407.08103) | Automata-theoretic, provably correct token/grammar alignment. |
| 2024 | [Grammar-Aligned Decoding](https://arxiv.org/abs/2405.21047) | Distribution-faithful sampling under grammar constraints. |
| 2024 | [SGLang](https://arxiv.org/abs/2312.07104) | Compressed-FSM jump-forward execution inside a serving runtime. |
| 2025 | [XGrammar](https://arxiv.org/abs/2411.15100) | PDA, adaptive token-mask cache, and persistent parser stacks. |
| 2025 | [JSONSchemaBench](https://arxiv.org/abs/2501.10868) | Real-schema coverage, correctness, and efficiency benchmark. |
| 2025 | [Flexible and Efficient GCD](https://proceedings.mlr.press/v267/park25l.html) | Faster tokenizer/CFG alignment preprocessing. |
| 2025 | [CRANE](https://proceedings.mlr.press/v267/banerjee25a.html) | Preserve reasoning by constraining only appropriate output regions. |
| 2025 | [Pre3](https://aclanthology.org/2025.acl-long.551/) | LR(1)-to-DPDA construction with prefix-conditioned edges. |
| 2026 | [XGrammar-2](https://arxiv.org/abs/2601.04426) | Dynamic structures, cross-grammar caching, and repetition compression. |
| 2026 | [PSC](https://conf.researchr.org/details/issta-2026/issta-2026-research-papers/81/Efficient-Grammar-Constrained-Decoding-via-Parser-Stack-Classification) | One parser-stack classifier for the complete vocabulary mask. |
| 2026 | [Gram2Token](https://icml.cc/virtual/2026/poster/62392) | GPU-native token categories and grammar transition tables. |
| 2026 | [GRID](https://arxiv.org/abs/2607.11951) | LALR(1) viable-prefix oracle for policy-constrained SQL. |

## Evidence policy

Each entry distinguishes among:

- **paper-verified**: supported by a proceedings paper or full preprint;
- **source-verified**: supported by public implementation or official API docs;
- **abstract-only**: only a conference abstract was publicly available;
- **engineering artifact**: repository, documentation, or blog without a
  peer-reviewed paper;
- **reported**: a number is quoted with its original comparison boundary rather
  than treated as a universal speedup.

No PDFs are vendored in this repository. Canonical proceedings, arXiv,
OpenReview, documentation, and source links are used instead.
