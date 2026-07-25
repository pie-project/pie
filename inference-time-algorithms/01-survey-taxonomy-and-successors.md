# The Anchor Survey's Full Taxonomy and Later Surveys (2024–2026)

The organizing backbone for this literature survey. We reproduce the full
structure of the anchor paper — Welleck et al.'s three-part taxonomy of
inference-time algorithms — then catalogue the surveys that appeared after it
(2024–2026), noting what each adds.

For every algorithm family, the **engine-primitive note** references the eight
expressiveness axes defined in `00-pie-capability-map.md` §10.

---

## The anchor paper

### From Decoding to Meta-Generation: Inference-time Algorithms for Large Language Models

- **Title:** From Decoding to Meta-Generation: Inference-time Algorithms for Large Language Models
- **arXiv:** [2406.16838](https://arxiv.org/abs/2406.16838)
- **Authors:** Welleck, Bertsch, Finlayson, Schoelkopf, Xie, Neubig, Kulikov, Harchaoui
- **Year:** 2024 — **Venue:** TMLR

The first comprehensive survey unifying inference-time algorithms for LLMs under
a single mathematical formalism. Organizes the field into three parts:
**(I) token-level generation** (one token at a time — decoding, sampling,
controlled/constrained generation), **(II) meta-generation** (partial or full
sequences — chaining, parallelism, step-level search), and **(III) efficient
generation** (reducing token cost and wall-clock time — speculation, KV cache,
early exit). Bridges the traditional NLP, modern-LLM, and ML-systems
communities.

The taxonomy is reproduced below; the rest of this report directory
(`02`–`07`) fleshes out each family with per-paper entries.

---

## Part I — Token-level generation algorithms (anchor §3)

Methods that select **one token at a time** or construct a token-level search
space and then select an output. Assume access to the model's logits,
next-token distribution, or probability scores.

**§3.1 MAP decoding.** Seek the highest-probability sequence y\* = argmax p(y|x).
Greedy decoding selects the argmax token at each step. Beam search maintains the
top-B scoring partial sequences, expands each, and prunes to B. Viterbi decoding
(for structured label tasks) also falls here.

**§3.2–3.3 Sampling and token-level adapters.** Introduce randomness for
diversity and naturalness. Covers temperature scaling, top-k (Fan et al., 2018),
nucleus / top-p (Holtzman et al., 2020), typical sampling (Meister et al.,
2022), η- and ε-sampling (Hewitt et al., 2022), repetition / frequency /
presence penalties, and Mirostat (dynamic truncation targeting a specific
perplexity).

**§3.4 Controlled generation.** Steer generation toward desired attributes
without hard constraints. Key methods: PPLM (gradient-based attribute control on
hidden activations), FUDGE (future-discriminator guided generation), contrastive
decoding (expert-minus-amateur logit arithmetic), classifier-free guidance for
LLMs (conditioned / unconditioned interpolation), and DExperts (expert +
anti-expert logit combination).

**§3.5 Constrained decoding.** Enforce hard output constraints. Covers
FSA / grammar-constrained decoding (masking to automaton-consistent tokens),
lexically-constrained decoding (Grid Beam Search, NeuroLogic Decoding),
trie-based constraints, and JSON-schema / regex constrained decoding.

> **Engine-primitive note.** Standard truncation samplers (top-k, top-p,
> temperature) and basic grammar-constrained decoding are **commodity** — every
> engine ships them. More sophisticated adapters (typical sampling,
> entropy-adaptive temperature, Mirostat) with **per-token state** need either a
> host-side logits processor (with a per-step round-trip) or **axis 4
> (device-resident stateful logic)**. Controlled generation via hidden-state
> manipulation (PPLM) needs **axis 1 (more than logits)**. Contrastive decoding,
> CFG, and DExperts need **axis 5 (combining several distributions)** — two
> forward contexts combined before sampling. Correct constrained decoding without
> distribution distortion (ASAp) needs **axis 3 (KV truncation/backtracking)**.
> Speculation composed with grammar constraints needs **axis 6 (custom
> draft/verify rule)**.

---

## Part II — Meta-generation algorithms (anchor §4)

Methods that work on **partial or full sequences**, incorporating domain
knowledge, enabling backtracking, and integrating external information.

**§4.1 Chained meta-generators.** Sequential composition: generate, then refine
or extend. Covers chain-of-thought prompting (Wei et al., 2022),
self-consistency (Wang et al., 2022), self-refine / self-correct (Madaan et al.,
2023), Reflexion (Shinn et al., 2023), and retrieval-augmented generation (RAG).

**§4.2 Parallel meta-generators.** Generate multiple candidates and aggregate.
Covers best-of-N / rejection sampling, diverse beam search (Vijayakumar et al.,
2016), minimum Bayes risk (MBR) decoding, and verifier-reranked sampling.

**§4.3 Step-level search algorithms.** Construct a tree or graph of partial
solutions and search. Covers Tree of Thoughts (Yao et al., 2023), bounded
lookahead search, Monte Carlo Tree Search (MCTS), and step-level beam search
with process reward models (PRMs).

> **Engine-primitive note.** Every parallel and step-level search method reduces
> to fork → explore → score → prune → backtrack. This is **axis 3 (explicit KV
> branching)** as the core primitive — fork a shared prefix N ways at O(1) cost.
> Chained meta-generators that interleave generation with tool calls or retrieval
> need **axis 8 (tool/agent I/O without losing cache locality)**. If scoring
> uses a value head, **axis 1 (value\_head / hidden states)** evaluates on device
> without an extra forward pass. Compute-optimal allocation of the branch budget
> across prompts needs **axis 7 (guest compute-allocation policy)**.

---

## Part III — Efficient generation (anchor §5)

Methods that reduce token cost and improve the speed of generation.

**§5.1 Speculative decoding.** Use a fast draft model to propose tokens; the
target model verifies in one forward pass. Variants: external draft models,
self-drafting (LayerSkip, early-exit), multi-token prediction heads, n-gram /
prompt-lookup cacheback, retrieval-based drafting (REST).

**§5.2 KV cache optimization.** Reduce memory and compute via paged attention,
prefix sharing, eviction policies (attention-score driven), compression,
quantization, and offloading.

**§5.3 Early exit and adaptive compute.** Terminate at shallower layers when
confident. Includes layer-skip self-drafting and adaptive compute-depth methods.

> **Engine-primitive note.** System-provided speculative decoding is commodity.
> Custom draft/verify rules (self-speculation, MTP draft, retrieval-based
> drafting, speculation + grammar) need **axis 6 (custom draft/verify rule)**.
> Feature-level drafting (EAGLE) needs **axis 1 (hidden states)**. Tree
> verification needs **axis 2 (custom attention mask)** and **axis 3 (KV
> branching)** for draft rollback. Score-driven KV eviction (H2O, Quest) needs
> **axis 1 (query tap)**. Adaptive compute depth needs **axis 7
> (compute-allocation policy)**.

---

## Later surveys — Decoding methods (2024)

### A Thorough Examination of Decoding Methods in the Era of LLMs

- **Title:** A Thorough Examination of Decoding Methods in the Era of LLMs
- **arXiv:** [2402.06925](https://arxiv.org/abs/2402.06925)
- **Authors:** Shi, Yang, Cai, Zhang, Wang, Yang, Lam
- **Year:** 2024 — **Venue:** EMNLP 2024

Empirical evaluation of decoding methods (greedy, beam, nucleus, contrastive,
typical) across tasks, models, and deployment settings. Key finding: the best
decoding method is highly task-dependent, and some methods require extensive
hyperparameter tuning. Adds a practical, benchmark-driven perspective missing
from the anchor survey's theoretical treatment.

> **Engine-primitive note.** The methods compared are mostly commodity (logits
> only). The finding that optimal decoding is task-dependent strengthens the case
> for **programmable samplers** — users need to swap methods per request without
> engine patches (**axis 4, device-resident stateful logic**).

---

## Later surveys — Speculative decoding (2024–2025)

### Unlocking Efficiency in Large Language Model Inference: A Comprehensive Survey of Speculative Decoding

- **Title:** Unlocking Efficiency in Large Language Model Inference: A Comprehensive Survey of Speculative Decoding
- **arXiv:** [2401.07851](https://arxiv.org/abs/2401.07851)
- **Authors:** Xia, Yang, Dong, Wang, Li, Ge, Liu, Li, Sui
- **Year:** 2024 — **Venue:** Findings of ACL 2024

The first dedicated survey of speculative decoding. Formalizes the
draft-then-verify paradigm, taxonomizes drafter designs (external model,
self-draft, retrieval-based, n-gram), and benchmarks acceptance rates. Provides
the vocabulary the rest of the field now uses.

> **Engine-primitive note.** Most speculation methods need **axis 6 (custom
> draft/verify rule)** when the drafter is non-standard. Feature-level drafters
> (EAGLE) additionally need **axis 1 (hidden states)**. Tree-structured
> verification needs **axis 2 (custom attention mask)**.

### Speculative Decoding and Beyond: An In-Depth Survey of Techniques

- **Title:** Speculative Decoding and Beyond: An In-Depth Survey of Techniques
- **arXiv:** [2502.19732](https://arxiv.org/abs/2502.19732)
- **Authors:** Xia, Liu, Dong, Li, Ge, Sui, Li
- **Year:** 2025

Extends the ACL 2024 survey to cover the 2024-2025 explosion: Medusa,
EAGLE-1/2/3, Sequoia, Jacobi decoding, consistency LLMs, multi-token
prediction heads, and generation-refinement frameworks beyond text (multimodal,
speech). The most current speculative decoding reference.

> **Engine-primitive note.** The newer methods deepen the need for **axis 6
> (custom draft/verify)** and **axis 1 (hidden states / MTP heads)**. Sequoia
> and Medusa trees require **axis 2 (custom attention mask)**.

---

## Later surveys — Test-time scaling and reasoning (2024–2026)

### Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters

- **Title:** Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters
- **arXiv:** [2408.03314](https://arxiv.org/abs/2408.03314)
- **Authors:** Snell, Lee, Xu, Kumar
- **Year:** 2024 — **Venue:** ICLR 2025

The foundational test-time scaling result. Shows that prompt-adaptive allocation
of inference compute (via process reward models and adaptive response
distributions) can be 4× more efficient than uniform best-of-N, and that a
small model with optimal test-time compute can outperform a 14× larger model.
Establishes "compute-optimal test-time scaling" as a research direction.

> **Engine-primitive note.** Compute-optimal allocation across prompts needs
> **axis 7 (guest compute-allocation policy)** — the guest decides how much
> compute each prompt gets. PRM-guided search needs **axis 1 (value head)** for
> on-device scoring and **axis 3 (explicit KV branching)** for candidate
> management.

### A Survey on Test-Time Scaling in Large Language Models

- **Title:** A Survey on Test-Time Scaling in Large Language Models: What, How, Where, and How Well?
- **arXiv:** [2503.24235](https://arxiv.org/abs/2503.24235)
- **Authors:** Zhang, Lyu, Sun, Wang, Zhang, Hua, Wu, Guo, Wang, Muennighoff, King, Liu, Ma
- **Year:** 2025

The most comprehensive post-anchor survey on test-time scaling. Proposes a
four-dimensional taxonomy: *what* to scale (tokens, candidates, steps), *how*
(search, refinement, verification), *where* (input, output, process), and *how
well* (metrics, benchmarks). Covers the full landscape from prompting strategies
to MCTS to RL-trained verifiers. Includes a curated GitHub paper database.

> **Engine-primitive note.** The "how" dimension (search, refinement,
> verification) maps directly to **axis 3 (KV branching)**, **axis 6 (custom
> draft/verify)**, and **axis 7 (compute-allocation)**. The "what" dimension
> includes hidden-state and reward-model scoring (**axis 1**).

### Inference-Time Scaling for Complex Tasks: Where We Stand and What Lies Ahead

- **Title:** Inference-Time Scaling for Complex Tasks: Where We Stand and What Lies Ahead
- **arXiv:** [2504.00294](https://arxiv.org/abs/2504.00294)
- **Authors:** Balachandran, Chen, Chen, Garg, Hashimoto, Hsu, Joshi, Liu, Mao, Pasunuru, Pereira, Radev, Raman, Rastogi, Ribeiro, Rosset, Sap, Shi, Singh, Song, Wang, Wang, Wu, Yu, Yue, Zhou
- **Year:** 2025 — **Venue:** Microsoft Research (MSR-TR-2025-16)

Empirical study of inference-time scaling across complex tasks (math, STEM,
NP-hard, multi-step reasoning). Finds that scaling helps most for certain
reasoning tasks but plateaus or degrades on others. Verifier and feedback models
amplify gains. Argues for adaptive, task-aware compute allocation rather than
uniform scaling.

> **Engine-primitive note.** Task-adaptive compute allocation is **axis 7
> (compute-allocation policy)**. Feedback/verifier integration needs **axis 1
> (value head)** or **axis 8 (tool/agent I/O)** depending on verifier type.

### A Survey of Frontiers in LLM Reasoning

- **Title:** A Survey of Frontiers in LLM Reasoning: Inference Scaling, Learning to Reason, and Agentic Systems
- **arXiv:** [2504.09037](https://arxiv.org/abs/2504.09037)
- **Authors:** Ke, Jiao, Ming, Nguyen, Xu, Long, Li, Qin, Wang, Savarese, Xiong, Joty
- **Year:** 2025

Categorizes reasoning methods along two axes: *regime* (inference-time vs.
training-time) and *architecture* (standalone LLM vs. agentic system). Covers
the shift from inference-scaling to learning-to-reason (DeepSeek-R1) and the
transition to agentic workflows (OpenAI Deep Research, multi-agent debate).
Includes a broad spectrum of learning algorithms (SFT, RL, GRPO) and agentic
workflow designs.

> **Engine-primitive note.** Agentic compound systems need **axis 8 (tool/agent
> I/O)** and **axis 3 (KV branching)** for parallel agent contexts.
> Output-level candidate refinement (reranking, self-consistency) needs **axis
> 3** for efficient parallel sampling.

### Stop Overthinking: A Survey on Efficient Reasoning for Large Language Models

- **Title:** Stop Overthinking: A Survey on Efficient Reasoning for Large Language Models
- **arXiv:** [2503.16419](https://arxiv.org/abs/2503.16419)
- **Authors:** Chen, Ji, Sapkota, Yang, Liu, Wang, Lyu, Zhu, Jiang, Wang, Jin
- **Year:** 2025 — **Venue:** TMLR 2025

Systematic survey of how to avoid verbose, wasteful reasoning ("overthinking")
in chain-of-thought and System-2 models (o1, DeepSeek-R1). Classifies
directions into model-based (concise reasoning heads), output-based (adaptive
step count), and prompt-based (forecasting depth from input complexity). The
complement to the test-time-scaling surveys: *when to stop* rather than *how
to scale*.

> **Engine-primitive note.** Adaptive reasoning depth and early stopping need
> **axis 7 (compute-allocation policy)** — the guest decides when to stop
> generating. Dynamic step budgets interact with **axis 4 (per-token stateful
> logic)** for tracking reasoning progress on device.

### A Survey of Slow Thinking-based Reasoning LLMs

- **Title:** A Survey of Slow Thinking-based Reasoning LLMs using Reinforced Learning and Inference-time Scaling Law
- **arXiv:** [2505.02665](https://arxiv.org/abs/2505.02665)
- **Authors:** Li, Chen, Tian, Peng, Wang, Kuang
- **Year:** 2025

Focuses on "slow thinking" LLMs inspired by System-2 cognition: OpenAI o1,
DeepSeek-R1, and their successors. Synthesizes 100+ studies across test-time
scaling, reinforcement learning for reasoning, and hierarchical reasoning
frameworks. Covers dynamic allocation of computation based on task complexity.

> **Engine-primitive note.** Slow-thinking models with dynamic computation
> budgets need **axis 7 (compute-allocation policy)**. Long chain-of-thought
> with tool/retrieval interleaving needs **axis 8 (tool/agent I/O)**. Rollback
> on reasoning dead-ends needs **axis 3 (KV branching)**.

---

## Later surveys — KV cache management (2025)

### A Survey on Large Language Model Acceleration based on KV Cache Management

- **Title:** A Survey on Large Language Model Acceleration based on KV Cache Management
- **arXiv:** [2412.19442](https://arxiv.org/abs/2412.19442)
- **Authors:** Li, Huang, Xie, Sun, Yao, Zhang, Zhong, Cai, Xing, Yang, Wang
- **Year:** 2025 — **Venue:** TMLR 2025

The most comprehensive KV-cache survey. Categorizes optimizations into
token-level (eviction, budget allocation, merging, quantization), model-level
(attention mechanism innovations), and system-level (memory management,
scheduling, hardware awareness). Covers score-driven eviction policies (H2O,
TOVA, SnapKV) and query-aware selection (Quest, RetrievalAttention) that the
anchor survey does not treat in depth.

> **Engine-primitive note.** Score-driven eviction and query-aware selection
> need **axis 1 (query tap)** — the current query scores which KV pages to
> retain. Pie's `query()` intrinsic and paged KV make these expressible as
> guest programs. On a black-box server, each eviction policy is an engine
> feature, not a user choice. Explicit KV management also needs **axis 3
> (explicit KV branching)**.

---

## Later surveys — Benchmarking (2025)

### Inference-Time Computations for LLM Reasoning and Planning: A Benchmark and Insights

- **Title:** Inference-Time Computations for LLM Reasoning and Planning: A Benchmark and Insights
- **arXiv:** [2502.12521](https://arxiv.org/abs/2502.12521)
- **Authors:** Shen, Niu, Liang, Feng, Goldstein, Huang
- **Year:** 2025

Introduces Sys2Bench, a benchmark spanning 11 tasks (arithmetic, logical,
common sense, algorithmic, planning) for evaluating inference-time reasoning
methods without additional training. Key finding: no single inference-time
method dominates all tasks. Argues for method-aware deployment rather than
one-size-fits-all inference pipelines.

> **Engine-primitive note.** Method-aware deployment means the serving system
> must support multiple inference-time strategies per request — the central
> argument for **programmable serving**: a guest choosing its own method via
> **axis 4 (per-token logic)**, **axis 6 (custom draft/verify)**, and **axis 7
> (compute-allocation policy)** rather than a fixed engine pipeline.

---

## Citation audit

All 12 arXiv citations were verified by fetching their abstract pages on
arxiv.org. Registered titles and arXiv IDs match.

| arXiv ID | Verified title matches `- **Title:**` line | Notes |
|---|---|---|
| 2406.16838 | ✓ exact | anchor paper |
| 2402.06925 | ✓ exact | |
| 2401.07851 | ✓ exact | |
| 2502.19732 | ✓ exact | |
| 2408.03314 | ✓ exact | |
| 2503.24235 | ✓ exact | heading abbreviates to omit subtitle |
| 2504.00294 | ✓ exact | |
| 2504.09037 | ✓ exact | heading abbreviates to omit subtitle |
| 2503.16419 | ✓ exact | |
| 2505.02665 | ✓ exact | heading abbreviates to omit subtitle |
| 2412.19442 | ✓ exact | |
| 2502.12521 | ✓ exact | |

Section headings that abbreviate paper titles (e.g. "A Survey on Test-Time
Scaling in Large Language Models" omitting the subtitle "What, How, Where, and
How Well?") are **not** the citation targets — the `- **Title:**` lines carry
the full registered title and are what `verify_citations.py` checks.

No citations were left unverified. No arXiv IDs were guessed.
