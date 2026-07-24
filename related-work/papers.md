# Annotated literature survey

This survey focuses on techniques that affect at least one of these design
questions:

- how a grammar is represented during generation;
- how tokenizer tokens are aligned with grammar terminals or bytes;
- whether parser state and mask construction run on CPU or GPU;
- how recursion, rollback, speculative decoding, and heterogeneous schemas are
  handled;
- how correctness, coverage, compile cost, and per-token cost are evaluated.

The entries are chronological within each theme. Performance numbers retain the
scope and baseline stated by the source.

## 1. Incremental guards and external controllers

### PICARD: Parsing Incrementally for Constrained Auto-Regressive Decoding

**Scholak, Schucher, Bahdanau. EMNLP 2021.**
[Paper](https://aclanthology.org/2021.emnlp-main.779/) ·
[Code](https://github.com/ServiceNow/picard)

- **Contribution:** runs an incremental SQL parser during decoding and rejects
  beam candidates that cannot lead to a valid query.
- **Representation:** an incremental, schema-aware parser rather than a
  precomputed vocabulary mask for every parser state.
- **Tokenizer alignment:** candidates are detokenized and checked by the parser;
  it does not build a tokenizer/grammar cross-product.
- **Runtime placement:** the parser is a separate CPU-side Haskell service while
  the model executes on an accelerator.
- **Important boundary:** PICARD checks a limited set of beam candidates rather
  than exhaustively masking the full vocabulary.
- **Relevance:** establishes the incremental-parser guard pattern and shows the
  value of semantic constraints such as table and column validity, but its IPC
  and candidate-check architecture does not target high-batch GPU mask
  generation.

### Synchromesh: Reliable Code Generation from Pre-trained Language Models

**Poesia et al. ICLR 2022.**
[Paper](https://arxiv.org/abs/2201.11227) ·
[Code](https://github.com/kanishkg/synchromesh)

- **Contribution:** introduces constrained semantic decoding using an external
  completion engine that can enforce syntax, scoping, and typing constraints
  without finetuning.
- **Representation:** a programmatic completion oracle, not a fixed automaton
  table.
- **Runtime placement:** CPU-side controller logic around model inference.
- **Relevance:** broadens constrained decoding from syntax to semantic program
  validity. It is an ancestor of controller APIs such as AICI, but not a
  device-resident grammar engine.

### Grammar-Constrained Decoding for Structured NLP Tasks without Finetuning

**Geng, Josifoski, Peyrard, West. EMNLP 2023.**
[Paper](https://aclanthology.org/2023.emnlp-main.674/) ·
[Code](https://github.com/epfl-dlab/GCD)

- **Contribution:** applies general CFGs and input-dependent grammars to
  structured NLP tasks without task-specific model training.
- **Representation:** token-level grammar plus an incremental parser.
- **Tokenizer alignment:** explicitly identifies a foundational problem: BPE
  tokenization is ambiguous and does not naturally align with grammar
  terminals. The paper's conversion is tokenizer-specific and motivates the
  later automata-alignment literature.
- **Runtime placement:** Python/CPU parser in the reference implementation.
- **Relevance:** input-dependent grammars are an early analogue of
  per-request JSON Schemas. They make compile latency and cross-request reuse
  first-class system concerns.

## 2. Automata indexes and tokenizer alignment

### Efficient Guided Generation for Large Language Models

**Willard and Louf. arXiv 2023.**
[Paper](https://arxiv.org/abs/2307.09702) ·
[Outlines](https://github.com/dottxt-ai/outlines)

- **Contribution:** compiles a regular-language constraint into an FSM and
  builds an index from FSM state to acceptable tokenizer tokens.
- **Representation:** `state -> allowed vocabulary subset`.
- **Tokenizer alignment:** walks each vocabulary token through the automaton
  from relevant states during index construction.
- **Runtime placement:** the original index construction and lookup are
  CPU-side; the model and final logit operations may run on GPU.
- **Strength:** after preprocessing, online lookup is effectively constant time.
- **Limitation:** table size and compile cost grow with grammar and vocabulary
  complexity. Later work also identifies token-boundary and schema-coverage
  pitfalls in regex/FSM implementations.
- **Relation to gpu-lr1:** this is important prior art for precomputed
  `state -> token-set` tables. gpu-lr1's distinct focus is heterogeneous
  global-state packing and fused device-side token selection plus next-state
  update.

### SynCode: LLM Generation with Grammar Augmentation

**Ugare, Suresh, Kang, Misailovic, Singh. TMLR 2025; first released 2024.**
[Paper](https://openreview.net/forum?id=HiUZtgAPoH) ·
[arXiv](https://arxiv.org/abs/2403.01632) ·
[Code](https://github.com/structuredllm/syncode)

- **Contribution:** constructs an offline DFA mask store for terminals of a CFG
  and uses it to retain valid tokens while filtering invalid ones.
- **Representation:** terminal DFAs plus lookup tables parameterized by live
  parser/lexer conditions.
- **Tokenizer alignment:** tests whether a tokenizer token is a complete or
  partial match for terminal sequences, addressing tokens that cross terminal
  boundaries.
- **Correctness:** explicitly states soundness and conditional completeness
  properties.
- **Runtime placement:** reference grammar processing and table lookup are
  CPU-side; no GPU parser kernel is described.
- **Trade-off:** moves substantial work offline. This is structurally close to
  gpu-lr1's DFA/tokenizer cross-product and reinforces the need to report cold
  compile time and persistent-cache behavior.

### Guiding LLMs The Right Way: Fast, Non-Invasive Constrained Generation

**Beurer-Kellner, Fischer, Vechev. ICML 2024.**
[Paper](https://proceedings.mlr.press/v235/beurer-kellner24a.html) ·
[Code](https://github.com/eth-sri/domino)

- **Contribution:** DOMINO aligns constraints with the model's subword
  vocabulary so that constrained decoding does not accidentally remove valid,
  high-probability tokenizations.
- **Representation:** vocabulary-aligned subterminal trees built during
  preprocessing.
- **Runtime strategy:** combines precomputation with speculative decoding and
  opportunistic masking; expensive constraint work is avoided when proposals
  are already valid.
- **Reported result:** virtually no overhead and, in some workloads, almost 2x
  speedup over unconstrained decoding.
- **Runtime placement:** the paper does not present a GPU-resident parser or
  automaton kernel. Speculation overlaps constraint work with accelerator
  inference.
- **Relevance:** tokenizer alignment affects both correctness and model quality,
  not only speed. gpu-lr1's token cross-product must be tested across tokenizer
  families rather than only one truncated GPT-2 vocabulary.

### Automata-based Constraints for Language Model Decoding

**Koo, Liu, He. COLM 2024.**
[Paper](https://arxiv.org/abs/2407.08103) ·
[Google Research page](https://research.google/pubs/automata-based-constraints-for-language-model-decoding/)

- **Contribution:** gives an automata-theoretic solution to ambiguous,
  misaligned tokenization for regular languages and extends it to deterministic
  context-free languages.
- **Representation:** compositions and closed-form constructions over
  automata/transducers instead of bespoke token filters.
- **Strength:** provides a correctness-oriented formal basis for compiling
  character- or byte-level languages into token-level constraints.
- **Reported result:** the paper reports roughly 7,000x faster constraint
  compilation than the specific prior construction it evaluates.
- **Runtime placement:** designed to lower into model-independent calculations
  over logits; it is not presented as a GPU-resident parser-stack engine.
- **Relation to gpu-lr1:** the DFA/tokenizer cross-product should be described as
  an instance of this broader automata-composition lineage, not as a new
  tokenizer-alignment concept.

### Flexible and Efficient Grammar-Constrained Decoding

**Park, Zhou, D'Antoni. ICML 2025.**
[Paper](https://proceedings.mlr.press/v267/park25l.html) ·
[Code](https://github.com/large-loris-models/greatgramma)

- **Contribution:** jointly analyzes the tokenizer vocabulary and CFG terminals
  to avoid preprocessing token/terminal combinations that cannot occur.
- **Representation:** a lexer/parser-aware token mapping for general CFGs.
- **Reported result:** 17.71x faster offline preprocessing than the evaluated
  prior approaches while preserving state-of-the-art online mask efficiency.
- **Runtime placement:** the contribution is an offline/online algorithmic
  improvement; the paper does not claim a GPU-native parser hot path.
- **Relevance:** directly targets gpu-lr1's comparatively expensive
  tokenizer-cross-product compilation. Any production compiler should compare
  against its realizability pruning and cache strategy.

## 3. Distribution and reasoning quality

### Grammar-Aligned Decoding

**Park, Wang, Berg-Kirkpatrick, Polikarpova, D'Antoni. NeurIPS 2024.**
[Paper](https://arxiv.org/abs/2405.21047) ·
[Code](https://github.com/ebmoon/transformers-GAD)

- **Problem:** standard hard masking guarantees grammaticality but generally
  distorts the language model's conditional distribution over valid strings.
- **Method:** adaptive sampling with approximate expected futures (ASAp)
  estimates future grammaticality and converges toward distribution-faithful
  sampling.
- **Trade-off:** the method adds substantial statistical computation and can
  converge slowly.
- **Relevance:** gpu-lr1 currently benchmarks greedy argmax. A future top-k,
  top-p, or temperature sampler must state whether it implements ordinary hard
  masking or attempts distribution-faithful grammar alignment.

### CRANE: Reasoning with Constrained LLM Generation

**Banerjee, Suresh, Ugare, Misailovic, Singh. ICML 2025.**
[Paper](https://proceedings.mlr.press/v267/banerjee25a.html) ·
[Code](https://github.com/uiuc-focal-lab/CRANE)

- **Problem:** applying a restrictive final-answer grammar to an entire
  reasoning trace can remove the model's ability to express useful intermediate
  reasoning.
- **Method:** augments or switches grammar regions so reasoning can remain
  flexible while the final structured answer is constrained.
- **Reported result:** up to a 10 percentage-point accuracy improvement over
  evaluated constrained and unconstrained strategies.
- **Relevance:** a serving API should support explicit constrained regions or
  structural tags. It should not assume that every output token belongs to the
  JSON payload.

### The Hidden Cost of Structured Generation in LLMs: Draft-Conditioned Constrained Decoding

**Reddy, Walker, Ide, Bedi. arXiv 2026.**
[Paper](https://arxiv.org/abs/2603.03305)

- **Problem:** describes a semantic or projection cost caused by standard
  mask-and-renormalize structured generation.
- **Method:** draft-conditioned constrained decoding separates an unconstrained
  draft from constrained realization.
- **Relevance:** complements Grammar-Aligned Decoding and CRANE. Kernel speed
  alone does not establish that constrained outputs preserve task quality or
  the model's intended distribution.

## 4. Serving systems and parser engines

### SGLang: Efficient Execution of Structured Language Model Programs

**Zheng et al. NeurIPS 2024.**
[Paper](https://arxiv.org/abs/2312.07104) ·
[Code](https://github.com/sgl-project/sglang) ·
[Compressed-FSM note](https://www.lmsys.org/blog/2024-02-05-compressed-fsm/)

- **Contribution relevant here:** jump-forward decoding compresses deterministic
  FSM paths and inserts known strings with an extend/prefill operation instead
  of spending one model step per forced token.
- **Tokenizer handling:** re-tokenizes after a jump-forward to reduce boundary
  artifacts.
- **Reported result:** the original note reports up to 2x lower latency and
  2.5x higher throughput than the evaluated Guidance/Outlines baselines.
- **Runtime placement:** the serving runtime and model are GPU-oriented, but
  grammar-state calculation is not presented as a GPU-resident LR/PDA engine.
- **Relevance:** forced-path compression is orthogonal to gpu-lr1's faster
  branching-state selection and should be part of a production integration.

### XGrammar: Flexible and Efficient Structured Generation Engine for LLMs

**Dong et al. MLSys 2025.**
[Paper](https://arxiv.org/abs/2411.15100) ·
[Code](https://github.com/mlc-ai/xgrammar) ·
[API documentation](https://xgrammar.mlc.ai/docs/api/python/grammar_matcher.html)

- **Contribution:** compiles general CFGs into a PDA-like representation,
  separates context-independent from context-dependent tokens, caches common
  masks, and uses persistent parser stacks for branching and rollback.
- **Representation:** byte-level per-rule automata connected by stack
  push/return transitions; current releases also use Earley-style parsing for
  dynamic grammar support.
- **Runtime placement, source-verified:** `fill_next_token_bitmask` requires a
  CPU bitmask. CUDA/Triton kernels apply that packed bitmask to logits; they do
  not perform parser-stack transitions.
- **Batching:** per-request matchers can be processed in a batch API and CPU
  threads; masks are then applied to selected rows on GPU.
- **Speculation:** supports rollback, fork, and draft-tree traversal.
- **Reported result:** up to 100x lower mask-generation latency than evaluated
  predecessors and near-zero serving overhead in configurations where CPU
  grammar work is effectively overlapped with GPU inference.
- **Relation to gpu-lr1:** it is the most important production baseline and has
  much broader grammar coverage. gpu-lr1 instead demonstrates a narrower path
  in which automaton state and next-state updates stay on device.

### Pre3: Enabling Deterministic Pushdown Automata for Faster Structured LLM Generation

**Chen et al. ACL 2025.**
[Paper](https://aclanthology.org/2025.acl-long.551/) ·
[arXiv](https://arxiv.org/abs/2506.03887) ·
[LightLLM project](https://github.com/ModelTC/LightLLM)

- **Contribution:** transforms LR(1) transition graphs into a DPDA using
  prefix-conditioned edges. Stack-prefix conditions make transitions
  deterministic and remove runtime path exploration and backtracking.
- **Representation:** a real stack-dependent DPDA, including precomputed
  reduction behavior and parallel checks of stack-prefix conditions.
- **Reported result:** up to 40% TPOT reduction and 36% throughput improvement
  over the evaluated baseline.
- **Runtime placement:** the paper focuses on deterministic parser construction
  and parallel transition verification. A production GPU-resident DPDA
  implementation is not established by the public artifact reviewed for this
  survey.
- **Artifact note:** LightLLM's README announces the paper, but a search of its
  current main branch did not locate a merged Pre3 parser implementation.
- **Relation to gpu-lr1:** gpu-lr1 now executes ordinary canonical LR(1)
  reduce/goto closures over a terminal stream. Pre3 remains the closest formal
  predecessor for compiling tokenizer-level, prefix-conditioned transitions
  that reduce online stack exploration. It directly addresses the stack-exposed
  `goto` dependency that invalidates a universal stack-independent
  `M_next[state, token]`.

### XGrammar-2: Efficient Dynamic Structured Generation Engine for Agentic LLMs

**Li et al. ACM CAIS 2026 / arXiv 2026.**
[Paper](https://arxiv.org/abs/2601.04426) ·
[Code](https://github.com/mlc-ai/xgrammar) ·
[Release note](https://blog.mlc.ai/2026/05/04/xgrammar-2-fast-customizable-structured-generation)

- **Contribution:** adds structural-tag dispatch, cross-grammar substructure
  caching, repetition-state compression, adaptive/JIT mask caching, batching,
  and speculative-decoding support.
- **Reported result:** more than 6x faster compilation than prior structured
  generation engines and near-zero end-to-end serving overhead.
- **Runtime placement:** improves the CPU compiler/matcher/cache architecture;
  the public GPU kernels remain packed-mask application kernels.
- **Relevance:** cross-grammar reuse is directly important for workloads with
  many related tool schemas. gpu-lr1's flat global-state packing addresses a
  different phase: execution after compilation.

### Efficient Grammar-Constrained Decoding via Parser Stack Classification

**Yongmin Li, Yihong Dong, Jia Li, Ge Li. ISSTA 2026, forthcoming.**
[Conference abstract](https://conf.researchr.org/details/issta-2026/issta-2026-research-papers/81/Efficient-Grammar-Constrained-Decoding-via-Parser-Stack-Classification)

- **Evidence level:** abstract-only at this snapshot; no public artifact was
  located.
- **Contribution:** PSC combines the acceptance conditions of all vocabulary
  tokens into one classifier over the parser stack during preprocessing.
- **Online algorithm:** checks the parser stack once per decoding step to
  construct the full vocabulary mask, with complexity independent of
  vocabulary size.
- **Reported result:** up to 700x faster mask computation for complex
  programming-language grammars and up to 30x for schema-conformant JSON;
  end-to-end throughput approaches unconstrained decoding.
- **Unknowns:** the public abstract does not specify the parser family, CPU/GPU
  placement, batching strategy, classifier representation, memory growth, or
  comparison versions.
- **Relation to gpu-lr1:** PSC may subsume much of the same "precompute enough
  parser context to make online work table-like" insight for recursive
  grammars. It must be treated as important prior art once the full paper is
  available.

### Gram2Token: Enabling Run-time GPU-Native Grammar-Constrained Decoding for LLMs

**Hua, Su, Tang, Yao, Zhu. ICML 2026.**
[Conference page](https://icml.cc/virtual/2026/poster/62392)

- **Evidence level:** the technical abstract embedded in the conference page;
  the page also contains a shorter lay summary. The full implementation was not
  publicly reachable during this audit.
- **Contribution:** preprocesses deterministic byte-level grammar execution
  into token-level transitions. A token trie groups vocabulary tokens with
  identical transition outcomes across grammar states, producing compact
  validity masks and transition tables.
- **Online algorithm:** category lookup, masking, and grammar-state update on
  GPU instead of parser-style byte traversal and CPU-controlled mask
  construction.
- **Reported result:** 1.38x geometric-mean throughput improvement over the
  strongest evaluated baseline and a maximum 1.85x speedup across four model
  families under schema-diverse continuous batching.
- **Trade-off:** additional preprocessing and time-to-first-token overhead;
  benefits improve with grammar reuse, longer outputs, and larger batches.
- **Artifact status:** the abstract declares
  `github.com/Paradozile/Gram2Token`, but that URL returned 404 and no public
  repository was found on 2026-07-24.
- **Relation to gpu-lr1:** this is the closest high-level prior art. Distinctive
  gpu-lr1 evidence is its public table construction, CSR/bitset density
  crossover, flat heterogeneous state namespace, memory measurements, and
  reproducible Triton kernels. A novelty claim must compare the detailed
  representations once Gram2Token's paper or code becomes available.

### GRID: Grammar-Railed Decoding for Enterprise SQL Generation

**Arjmandi. arXiv 2026.**
[Paper](https://arxiv.org/abs/2607.11951)

- **Contribution:** keys exact next-token masks on
  `(lexer scan state, LALR(1) parser stack)` and uses the incremental parser as a
  viable-prefix oracle. It also compiles role and schema policy into the SQL
  language.
- **Representation:** LALR(1) parser stack, token trie, and
  context-independent/context-dependent token split.
- **Runtime placement:** mask generation is implemented with CPU/Rust kernels
  and overlapped with GPU model execution.
- **Reported result:** 3.6-6.7 microsecond median mask time on the evaluated
  tokenizers, plus SQL execution-accuracy and auditability results.
- **Limitations:** SQL-specific policy focus, LALR(1)-language boundary, and no
  public implementation located at this snapshot.
- **Relevance:** demonstrates that highly optimized CPU parser execution can be
  very competitive, so gpu-lr1 must compare wall-clock and end-to-end serving
  results rather than CUDA kernel time alone.

## 5. Different decoding regimes

### Lookahead-then-Verify: Reliable Constrained Decoding for Diffusion LLMs under CFGs

**Zhang et al. arXiv 2026.**
[Paper](https://arxiv.org/abs/2602.00612)

- **Contribution:** LAVE adapts CFG-constrained decoding to diffusion language
  models, which predict distributions at multiple positions in parallel rather
  than generating strictly left-to-right.
- **Method:** uses lookahead distributions to verify that proposed intermediate
  states remain completable.
- **Relevance:** not a direct autoregressive-engine baseline, but it shows that
  grammar execution and state representation must be redesigned for the
  underlying decoding algorithm rather than assumed universal.

## 6. Correctness and benchmark literature

### JSONSchemaBench: A Rigorous Benchmark of Structured Outputs for Language Models

**Geng et al. arXiv 2025; ES-FoMo III at ICML 2025.**
[Paper](https://arxiv.org/abs/2501.10868) ·
[Code and data](https://github.com/guidance-ai/jsonschemabench)

- **Corpus:** roughly ten thousand real-world schemas across function-calling,
  GitHub, Kubernetes, JSON Schema Store, and other sources, plus the official
  JSON Schema Test Suite.
- **Correctness vocabulary:** distinguishes declared coverage, empirical
  coverage, true coverage, compliance rate, over-constraint, and
  under-constraint.
- **Efficiency metrics:** grammar compilation time, time to first token, and
  time per output token.
- **Key finding:** accepting a schema does not imply exact schema semantics.
  Engines differ substantially in unsupported keywords, silent
  under-constraint, compile failures, and tail latency.
- **Limitation for gpu-lr1:** the published end-to-end comparison is mainly
  batch 1 and mixes serving backends. Its corpus and correctness taxonomy are
  more reusable than its absolute latency numbers.

### MaskBench

**Engineering benchmark within JSONSchemaBench.**
[Benchmark](https://github.com/guidance-ai/jsonschemabench/tree/main/maskbench)

- **Purpose:** isolates grammar compilation and token-mask computation without
  running an LLM.
- **Data:** more than 11,000 schemas in the current result table, with valid and
  invalid instances totaling millions of tokens.
- **Method:** single-thread engine execution, CPU-only mask generation, and
  percentile reporting for time to first mask and time between masks.
- **Important result:** median performance alone is misleading. XGrammar is
  extremely fast on simple rows but has much larger p99/p99.9 tails than
  llguidance in the published run.
- **Relevance:** gpu-lr1 should reproduce MaskBench-style p50-p99.9 wall-clock
  distributions on GPU and should report compile errors, invalid acceptance,
  crashes, OOMs, and timeouts alongside speed.

## 7. Engineering artifacts without a standalone peer-reviewed paper

### llguidance

[Repository](https://github.com/guidance-ai/llguidance) ·
[Technical note](https://guidance-ai.github.io/llguidance/llg-go-brrr)

- Rust CPU engine using a symbolic-regex lexer, token trie, vocabulary slicing,
  and a general Earley parser for the residual hard cases.
- Integrated into llama.cpp, vLLM, SGLang, TensorRT-LLM-related tooling, and
  other serving stacks.
- One of the strongest CPU baselines, especially in MaskBench tail latency.

### AICI

[Repository](https://github.com/microsoft/aici)

- Runs sandboxed Wasm controllers on CPU while the GPU performs model
  generation.
- Supports more general per-token control than grammar masking, including
  dynamic prompt editing and parallel generation coordination.
- Useful systems precedent for overlap and multi-tenant controller isolation;
  llguidance is the maintained specialization for grammar constraints.

### JSONformer and lm-format-enforcer

[JSONformer](https://github.com/1rgs/jsonformer) ·
[lm-format-enforcer](https://github.com/noamgat/lm-format-enforcer)

- JSONformer inserts structural tokens programmatically and asks the model only
  for value content. It is a useful restricted baseline but is not a general
  grammar automaton.
- lm-format-enforcer incrementally checks token prefixes against JSON
  Schema/regex constraints through a CPU callback.
- Both illustrate practical alternatives to a compiled LR/PDA engine, but
  neither addresses GPU-resident heterogeneous parser state.

### vLLM, SGLang, and TensorRT-LLM integrations

- These serving engines provide production scheduling, batching, and GPU
  kernels around XGrammar, llguidance, Outlines, or similar backends.
- Integration should not be confused with GPU-native grammar execution:
  the parser may still run on CPU while only packed-mask application and model
  sampling run on GPU.

## 8. Additional adjacent work

- [GRAMMAR-LLM](https://aclanthology.org/2025.findings-acl.177/) studies
  grammar-constrained natural-language generation rather than the GPU execution
  problem.
- [Using Grammar Masking to Ensure Syntactic Validity in LLM-based Modeling
  Tasks](https://arxiv.org/abs/2407.06146) applies CFG masking to
  model-driven-software-engineering outputs.
- [When Grammar Guides the Attack](https://arxiv.org/abs/2503.24191) analyzes
  control-plane security risks introduced by structured-output constraints.
- Proprietary structured-output implementations from model API providers are
  operationally important but cannot be used to establish parser architecture,
  GPU placement, or reproducible performance without public technical details.
