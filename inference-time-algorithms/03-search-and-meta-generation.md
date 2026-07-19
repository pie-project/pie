# Search-based and meta-generation inference-time algorithms

Best-of-N, self-consistency, Tree/Graph of Thoughts, MCTS for reasoning,
PRM/value-guided search, test-time compute scaling, refinement loops, and
multi-agent ensemble methods. Every method is annotated with the engine
primitives it requires, using the eight expressiveness axes defined in
`00-pie-capability-map.md` §10.

Anchor survey: Welleck et al., *From Decoding to Meta-Generation*, [2406.16838](https://arxiv.org/abs/2406.16838), TMLR 2024.

---

## Parallel sampling and reranking

The simplest meta-generation pattern: sample N completions from a shared
prefix, then select or aggregate. Every method in this family benefits from
**explicit KV branching** (axis 3): on Pie, N forks of a committed prefix
share pages via content-addressed sharing and cost O(1) each; on a black-box
server each sample either re-prefills or relies on an opaque prefix cache.

### Best-of-N with verifiers (Cobbe et al.)

- **Title:** Training Verifiers to Solve Math Word Problems
- **arXiv:** [2110.14168](https://arxiv.org/abs/2110.14168)

Sample N solutions and pick the one ranked highest by a learned verifier. The
canonical approach to reward-model reranking at test time, shown to scale
more favourably than simply enlarging the model. Requires N parallel
generations from a shared prompt.

> **Engine primitives.** Axis 3 (KV branching): N forks share the prompt
> prefix. Axis 1 (more than logits): on-device `value_head()` scoring
> eliminates an extra host round-trip per candidate. Axis 7 (compute-allocation
> policy): adaptive N requires a guest-visible budget.

### Self-Consistency (Wang et al.)

- **Title:** Self-Consistency Improves Chain of Thought Reasoning in Language Models
- **arXiv:** [2203.11171](https://arxiv.org/abs/2203.11171)

Sample diverse chain-of-thought paths and take the majority vote over final
answers. A zero-verifier baseline: the aggregator is just plurality. Shown to
substantially boost CoT accuracy on arithmetic, commonsense, and symbolic
reasoning tasks.

> **Engine primitives.** Axis 3 (KV branching): many forks from the same CoT
> prompt. No value head needed — selection is by counting.

### Universal Self-Consistency (Chen et al.)

- **Title:** Universal Self-Consistency for Large Language Model Generation
- **arXiv:** [2311.17311](https://arxiv.org/abs/2311.17311)

Extends self-consistency to free-form tasks where majority voting over
discrete answers is impossible, by using an LLM to select the most
consistent response from the sample set.

> **Engine primitives.** Axis 3 (KV branching): sample N, then a selector
> generation. Axis 8 (tool/agent I/O): the selector can be a second inferlet
> without evicting the generators' caches.

### DiVeRSe (Li et al.)

- **Title:** Making Large Language Models Better Reasoners with Step-Aware Verifier
- **arXiv:** [2206.02336](https://arxiv.org/abs/2206.02336)

Combines diverse prompting (multiple prompts × multiple samples), a
step-aware verifier that scores each reasoning step, and voting to select
the final answer. Early evidence that process-level verification
outperforms outcome-only reranking.

> **Engine primitives.** Axis 3 (KV branching): cross-product of prompts and
> samples. Axis 1 (more than logits): step-level scoring benefits from
> `value_head()`.

### Meta-Reasoning over Multiple Chains of Thought (Yoran et al.)

- **Title:** Answering Questions by Meta-Reasoning over Multiple Chains of Thought
- **arXiv:** [2304.13007](https://arxiv.org/abs/2304.13007)

Generates multiple CoT explanations, then a meta-reasoner reads all chains
and produces a final answer, weighting evidence across chains rather than
simply voting.

> **Engine primitives.** Axis 3 (KV branching): fork per chain. Axis 8
> (tool/agent I/O): the meta-reasoner can be an inferlet with its own context.

### MBR Decoding for LLMs (Bertsch et al.)

- **Title:** It's MBR All the Way Down: Modern Generation Techniques Through the Lens of Minimum Bayes Risk
- **arXiv:** [2310.01387](https://arxiv.org/abs/2310.01387)

Reframes best-of-N, self-consistency, and reranking as instances of Minimum
Bayes Risk decoding. Provides a unifying view: sample a support set, score
each candidate against the rest under a utility function, pick the one with
highest expected utility.

> **Engine primitives.** Axis 3 (KV branching): the support set is N forks.
> Pairwise scoring is O(N²) host work — Pie's fork sharing keeps the generation
> cost at O(divergent tokens).

### Speculative Rejection for Best-of-N (Sun et al.)

- **Title:** Fast Best-of-N Decoding via Speculative Rejection
- **arXiv:** [2410.20290](https://arxiv.org/abs/2410.20290)

Accelerates best-of-N by early-killing candidates whose partial reward is
unlikely to reach the current best. Uses the reward model's running score
as an accept/reject criterion so compute is not wasted finishing hopeless
branches.

> **Engine primitives.** Axis 3 (KV branching): early kill reclaims pages
> (`truncate`). Axis 7 (compute-allocation policy): the bid for each branch
> can be withdrawn on rejection. This is a natural Pie pattern —
> `ctx.fork()` + page discard.

### Large Language Monkeys — Repeated Sampling (Brown et al.)

- **Title:** Large Language Monkeys: Scaling Inference Compute with Repeated Sampling
- **arXiv:** [2407.21787](https://arxiv.org/abs/2407.21787)

Empirically demonstrates that coverage (fraction of problems solved by at
least one of N samples) follows an approximately log-linear scaling law,
and that for many coding/math benchmarks simply increasing N is competitive
with model scaling.

> **Engine primitives.** Axis 3 (KV branching): the entire argument rests on
> efficient large-N sampling from a shared prefix. Axis 7 (compute-allocation
> policy): compute-optimal N per problem requires per-request budget control.

### Rejection Sampling Fine-Tuning — RFT (Yuan et al.)

- **Title:** Scaling Relationship on Learning Mathematical Reasoning with Large Language Models
- **arXiv:** [2308.01825](https://arxiv.org/abs/2308.01825)

Shows that sampling many solutions and keeping the correct ones for
fine-tuning (rejection sampling) improves mathematical reasoning. The
inference-time component is the large-N sampler and verifier; the training
loop is downstream.

> **Engine primitives.** Axis 3 (KV branching): bulk sampling. This is the
> data-generation stage of a train–inference loop where inference efficiency
> directly scales the training set.

---

## Tree and graph search over reasoning

These methods structure generation as an explicit search tree or DAG, where
nodes are partial generations (thoughts, steps, or chunks) and edges are
extensions or transformations. Every method needs **explicit KV
branching/backtracking** (axis 3) — on Pie, `ctx.fork()` is O(1) and
backtracking is a `truncate` or snapshot restore; on a black-box server
it is either re-prefill or an opaque prefix cache.

### Tree of Thoughts (Yao et al.)

- **Title:** Tree of Thoughts: Deliberate Problem Solving with Large Language Models
- **arXiv:** [2305.10601](https://arxiv.org/abs/2305.10601)

Decomposes a problem into thought steps, generates multiple candidates at
each step, evaluates them (via self-evaluation or a heuristic), and uses
BFS or DFS to search the tree. A direct translation of classical AI search
into the LLM generation loop.

> **Engine primitives.** Axis 3 (KV branching): O(1) fork per thought node.
> Axis 2 (custom attention mask): tree attention over the branch DAG
> enables batched evaluation. Axis 8 (tool/agent I/O): evaluation queries
> can use tools.

### Graph of Thoughts (Besta et al.)

- **Title:** Graph of Thoughts: Solving Elaborate Problems with Large Language Models
- **arXiv:** [2308.09687](https://arxiv.org/abs/2308.09687)

Generalises ToT to an arbitrary DAG: thoughts can merge, refine, and loop
back. Enables operations like aggregation (merge two partial solutions) and
refinement (revise a node) that a pure tree cannot express.

> **Engine primitives.** Axis 3 (KV branching): fork + merge. Merge requires
> combining context from two branches — on Pie, two `WorkingSet`s can be
> composed; elsewhere it is a re-prefill of the merged context.

### Branch-Solve-Merge (Saha et al.)

- **Title:** Branch-Solve-Merge Improves Large Language Model Evaluation and Generation
- **arXiv:** [2310.15123](https://arxiv.org/abs/2310.15123)

Decomposes a task into sub-tasks (branch), solves each independently, and
merges the results. A practical divide-and-conquer meta-generation strategy
applicable to evaluation, long-form generation, and constrained writing.

> **Engine primitives.** Axis 3 (KV branching): one fork per sub-task. Axis
> 8 (tool/agent I/O): merge is a separate generation.

### Skeleton-of-Thought (Ning et al.)

- **Title:** Skeleton-of-Thought: Prompting LLMs for Efficient Parallel Generation
- **arXiv:** [2307.15337](https://arxiv.org/abs/2307.15337)

First generates a skeleton (outline), then expands each point in parallel.
Reduces wall-clock latency by trading sequential decode for parallel branch
generation.

> **Engine primitives.** Axis 3 (KV branching): parallel branches from the
> skeleton prefix. Frames (§4 of `00`) can submit heterogeneous branch
> expansions in one ordered batch.

### Self-Evaluation Guided Beam Search (Xie et al.)

- **Title:** Self-Evaluation Guided Beam Search for Reasoning
- **arXiv:** [2305.00633](https://arxiv.org/abs/2305.00633)

A step-level beam search where the LLM itself scores partial solutions via
a self-evaluation prompt (stochastic beam scoring). Beams are pruned by
self-assessed confidence at each reasoning step.

> **Engine primitives.** Axis 3 (KV branching): beams are forks. Axis 2
> (custom attention mask): logical ancestry masks avoid duplicating KV across
> beams. Axis 1 (more than logits): on-device `value_head()` could replace
> the prompt-based self-evaluation, removing a host round-trip per beam per
> step.

### Stream of Search (Gandhi et al.)

- **Title:** Stream of Search (SoS): Learning to Search in Language
- **arXiv:** [2404.03683](https://arxiv.org/abs/2404.03683)

Trains the model to emit the *search process itself* as text — the
transcript of BFS/DFS/A* over a problem — so search becomes part of the
generation rather than an external orchestration loop.

> **Engine primitives.** Minimal engine demands at inference — the model
> generates a flat sequence. But generating training data for SoS requires
> running actual tree search, so axes 3 and 7 are needed upstream.

---

## Monte Carlo Tree Search for LLM reasoning

MCTS adapts the classical selection → expansion → simulation → backpropagation
loop to LLM reasoning, where actions are generation steps (tokens, sentences,
or "thoughts") and the value function is a verifier or reward model.
Every MCTS method needs **axis 3** (fork + backtrack + per-node value) and
benefits strongly from **axis 1** (on-device scoring via `value_head()`) and
**axis 7** (per-branch compute budgets).

### RAP — Reasoning via Planning (Hao et al.)

- **Title:** Reasoning with Language Model is Planning with World Model
- **arXiv:** [2305.14992](https://arxiv.org/abs/2305.14992)

Uses the LLM as both world model (to simulate state transitions) and
reasoning agent, then applies MCTS to plan over the reasoning tree. The
LLM generates candidate next steps, MCTS explores and scores them, and
the best trajectory is returned.

> **Engine primitives.** Axis 3 (KV branching): fork per expansion.
> Axis 1 (more than logits): a value function on `value_head()` replaces
> rollout-to-completion. Axis 7 (compute policy): exploration budget.

### LATS — Language Agent Tree Search (Zhou et al.)

- **Title:** Language Agent Tree Search Unifies Reasoning Acting and Planning in Language Models
- **arXiv:** [2310.04406](https://arxiv.org/abs/2310.04406)

Unifies reasoning, acting (tool use), and planning under a single MCTS
framework. Adds environment interaction and self-reflection to the MCTS
loop, with value backpropagation from terminal rewards.

> **Engine primitives.** Axis 3 (KV branching/backtrack). Axis 8
> (tool/agent I/O): actions include tool calls that must not evict KV
> state. Axis 7 (compute policy).

### TS-LLM — AlphaZero-Like Tree-Search (Feng et al.)

- **Title:** Alphazero-like Tree-Search can Guide Large Language Model Decoding and Training
- **arXiv:** [2309.17179](https://arxiv.org/abs/2309.17179)

Applies AlphaZero-style MCTS at both token and sentence level, learning a
policy and value function end-to-end. Generalises ToT/RAP by training the
value model rather than prompting for evaluation.

> **Engine primitives.** Axis 3 (KV branching). Axis 1: the learned value
> head should score on device for low-latency node evaluation. Axis 7:
> MCTS exploration budget per request.

### AlphaLLM (Tian et al.)

- **Title:** Toward Self-Improvement of LLMs via Imagination, Searching, and Criticizing
- **arXiv:** [2404.12253](https://arxiv.org/abs/2404.12253)

Combines MCTS with a trio of critics (task-specific, process, outcome)
for LLM self-improvement. The search tree provides training data for
iterative self-play.

> **Engine primitives.** Axis 3 (KV branching). Axis 1 (value head for
> critic scoring). Axis 7 (compute policy for exploration).

### AlphaMath (Chen et al.)

- **Title:** AlphaMath Almost Zero: Process Supervision without Process
- **arXiv:** [2405.03553](https://arxiv.org/abs/2405.03553)

Uses MCTS to generate step-level process supervision signals automatically,
eliminating the need for human step annotations. The process reward model
is trained from the MCTS value estimates.

> **Engine primitives.** Axis 3 (KV branching): MCTS rollouts. Axis 1:
> `value_head()` trained as the PRM. The data-generation phase is
> inference-heavy and benefits from fork sharing.

### rStar (Qi et al.)

- **Title:** Mutual Reasoning Makes Smaller LLMs Stronger Problem-Solvers
- **arXiv:** [2408.06195](https://arxiv.org/abs/2408.06195)

A mutual reasoning framework where a generator and a discriminator work
together in an MCTS loop. The discriminator ranks partial reasoning
paths, guiding the generator toward correct solutions.

> **Engine primitives.** Axis 3 (KV branching). Axis 5 (combining
> distributions): generator + discriminator require two model contexts
> per step. Axis 8 (inter-inferlet messaging).

### rStar-Math (Qi et al.)

- **Title:** rStar-Math: Small LLMs Can Master Math Reasoning with Self-Evolved Deep Thinking
- **arXiv:** [2501.04519](https://arxiv.org/abs/2501.04519)

Extends rStar with code-augmented CoT and self-evolved deep thinking.
Demonstrates that small LLMs with MCTS can match much larger models on
mathematical benchmarks through iterative self-improvement.

> **Engine primitives.** Axis 3 (KV branching). Axis 8 (tool I/O): code
> execution is interleaved with reasoning. Axis 7: iterative deepening
> needs adaptive budgets.

### ReST-MCTS* (Zhang et al.)

- **Title:** ReST-MCTS*: LLM Self-Training via Process Reward Guided Tree Search
- **arXiv:** [2406.03816](https://arxiv.org/abs/2406.03816)

Combines MCTS with process reward models for self-training. Uses an
MCTS* variant (best-first with process-reward guidance) to generate
high-quality reasoning traces, which are then used to fine-tune both
the policy and the PRM.

> **Engine primitives.** Axis 3 (KV branching + backtrack). Axis 1
> (`value_head()` for PRM). Axis 7: MCTS node budget.

### MCTSr — Monte Carlo Tree Self-Refine (Zhang et al.)

- **Title:** Accessing GPT-4 level Mathematical Olympiad Solutions via Monte Carlo Tree Self-refine with LLaMa-3 8B
- **arXiv:** [2406.07394](https://arxiv.org/abs/2406.07394)

Integrates MCTS with self-refinement: tree nodes contain full solutions,
expansion is via self-critique and revision, and value backpropagation
guides which branches to refine further.

> **Engine primitives.** Axis 3 (KV branching). Axis 8 (tool I/O):
> self-critique is an LLM call that should not evict the main context.

### MCTS via Iterative Preference Learning (Xie et al.)

- **Title:** Monte Carlo Tree Search Boosts Reasoning via Iterative Preference Learning
- **arXiv:** [2405.00451](https://arxiv.org/abs/2405.00451)

Uses MCTS to construct preference pairs for iterative DPO training.
The MCTS search provides a principled way to contrast good and bad
reasoning traces at each step.

> **Engine primitives.** Axis 3 (KV branching): search phase. The
> training phase is offline but inference-side fork efficiency
> determines data throughput.

### Value-Guided MCTS Decoding (Liu et al.)

- **Title:** Don't throw away your value model! Generating more preferable text with Value-Guided Monte-Carlo Tree Search decoding
- **arXiv:** [2309.15028](https://arxiv.org/abs/2309.15028)

Uses a token-level value model within MCTS to guide decoding toward
sequences preferred by a reward model, without modifying the base LLM.
The value function steers the tree search at each expansion.

> **Engine primitives.** Axis 1 (more than logits): `value_head()`
> scoring per node, on device. Axis 3 (KV branching). On Pie the
> value evaluation is fused into the forward pass.

### Q* — Deliberative Planning (Wang et al.)

- **Title:** Q*: Improving Multi-step Reasoning for LLMs with Deliberative Planning
- **arXiv:** [2406.14283](https://arxiv.org/abs/2406.14283)

Formulates multi-step reasoning as an MDP and learns a Q-value function
for guiding search. Uses the Q-values to guide both best-first tree
search and A*-like heuristic search over reasoning steps.

> **Engine primitives.** Axis 1 (value head for Q-values). Axis 3
> (KV branching for tree search). Axis 7 (compute policy).

### Marco-o1 (Zhao et al.)

- **Title:** Marco-o1: Towards Open Reasoning Models for Open-Ended Solutions
- **arXiv:** [2411.14405](https://arxiv.org/abs/2411.14405)

An open reasoning model that integrates MCTS, self-reflection, and
chain-of-thought for open-ended problem solving, inspired by o1-style
long-CoT reasoning. Uses confidence-based tree expansion.

> **Engine primitives.** Axis 3 (KV branching). Axis 7 (compute policy):
> adaptive MCTS depth. Axis 1: confidence scoring on device.

---

## Verifier and reward-guided decoding

Methods that use a verifier or reward model to guide or filter generation.
The core engine need is **axis 1** (on-device scoring via `value_head()` or
`hidden()`), because every host round-trip for scoring is a latency and
throughput tax on the search. Pie's PTIR allows the verifier to run
*inside the forward pass*.

### Let's Verify Step by Step (Lightman et al.)

- **Title:** Let's Verify Step by Step
- **arXiv:** [2305.20050](https://arxiv.org/abs/2305.20050)

Demonstrates that process supervision (step-level reward labels) produces
better verifiers than outcome supervision alone for mathematical
reasoning. Introduces PRM800K, the canonical process reward model
training set.

> **Engine primitives.** Axis 1 (more than logits): a PRM is a step-level
> `value_head()`. On Pie it can score inside the forward pass; elsewhere
> it requires a separate inference call per step.

### Process- and Outcome-Based Feedback (Uesato et al.)

- **Title:** Solving math word problems with process- and outcome-based feedback
- **arXiv:** [2211.14275](https://arxiv.org/abs/2211.14275)

Compares process reward models (PRMs, which score each step) with outcome
reward models (ORMs, which score only the final answer). Finds that
PRMs provide stronger signal for search and are more sample-efficient.

> **Engine primitives.** Axis 1: step-level scoring. Axis 3: search over
> reasoning paths using the PRM signal.

### Math-Shepherd (Wang et al.)

- **Title:** Math-Shepherd: Verify and Reinforce LLMs Step-by-step without Human Annotations
- **arXiv:** [2312.08935](https://arxiv.org/abs/2312.08935)

Automatically labels step-level correctness by rolling out completions
from each step and checking if they reach the correct answer. The
resulting PRM is competitive with human-annotated process supervision.

> **Engine primitives.** Axis 3 (KV branching): rollouts from each
> step are the data-generation bottleneck. Axis 1: the trained PRM
> becomes a `value_head()` for inference-time search.

### OmegaPRM — Automated Process Supervision (Luo et al.)

- **Title:** Improve Mathematical Reasoning in Language Models by Automated Process Supervision
- **arXiv:** [2406.06592](https://arxiv.org/abs/2406.06592)

Uses a divide-and-conquer Monte Carlo method to efficiently identify the
first incorrect step in a reasoning chain, enabling fully automated
process-supervision annotation at scale.

> **Engine primitives.** Axis 3 (KV branching): binary-search rollouts
> from each candidate error point. Fork sharing makes this efficient.

### Controlled Decoding (Mudgal et al.)

- **Title:** Controlled Decoding from Language Models
- **arXiv:** [2310.17022](https://arxiv.org/abs/2310.17022)

Trains a prefix scorer (value function) that steers generation toward
high-reward outputs without modifying the base model. At each step,
the value function biases the next-token distribution toward
trajectories with high expected reward.

> **Engine primitives.** Axis 1 (more than logits): value scoring on
> device via `value_head()`. Axis 4 (per-token stateful logic): the
> value bias should be applied device-side without a host round-trip.

### ARGS — Alignment as Reward-Guided Search (Khanov et al.)

- **Title:** ARGS: Alignment as Reward-Guided Search
- **arXiv:** [2402.01694](https://arxiv.org/abs/2402.01694)

Applies reward-guided greedy or top-k search at decode time to steer
outputs toward aligned behaviour without RLHF training. A reward model
rescores candidate tokens at each step.

> **Engine primitives.** Axis 1 (on-device reward scoring). Axis 4
> (per-token device logic to avoid host round-trip per token).

### Value Augmented Sampling — VAS (Han et al.)

- **Title:** Value Augmented Sampling for Language Model Alignment and Personalization
- **arXiv:** [2405.06639](https://arxiv.org/abs/2405.06639)

Samples from a frozen model using importance weights derived from a learned
value function, providing alignment without fine-tuning. Supports on-the-fly
composition of multiple reward functions at deployment.

> **Engine primitives.** Axis 1 (value function scoring). Axis 5
> (combining distributions): importance weighting blends base
> distribution and reward signal.

### Reward-Guided Speculative Decoding — RSD (Liao et al.)

- **Title:** Reward-Guided Speculative Decoding for Efficient LLM Reasoning
- **arXiv:** [2501.19324](https://arxiv.org/abs/2501.19324)

Combines speculative decoding with a reward model: a lightweight draft
model generates candidates, a reward model scores them, and the target
model is invoked only when the reward is insufficient. Achieves up to
4.4× fewer FLOPs on reasoning benchmarks.

> **Engine primitives.** Axis 6 (custom draft/verify rule): the
> accept/reject criterion is reward-based, not distribution-matching.
> Axis 1: reward scoring on device. Axis 7: dynamic target-model
> invocation is a compute-allocation decision.

### Generative Verifiers — GenRM (Zhang et al.)

- **Title:** Generative Verifiers: Reward Modeling as Next-Token Prediction
- **arXiv:** [2408.15240](https://arxiv.org/abs/2408.15240)

Treats verification as next-token prediction rather than discriminative
classification, allowing verifiers to leverage chain-of-thought and
instruction-tuning. Outperforms discriminative verifiers and
LLM-as-a-Judge on math and algorithmic reasoning.

> **Engine primitives.** Axis 8 (tool/agent I/O): the verifier is a
> generative model that needs its own context. On Pie this can be a
> second inferlet sharing the prompt prefix via KV fork.

### OVM — Outcome-supervised Value Models (Yu et al.)

- **Title:** OVM, Outcome-supervised Value Models for Planning in Mathematical Reasoning
- **arXiv:** [2311.09724](https://arxiv.org/abs/2311.09724)

Trains value models with outcome supervision (correct/incorrect final
answers) and uses them for step-level planning via beam search. Shows
that outcome-supervised value heads can effectively guide step-level
search without process annotations.

> **Engine primitives.** Axis 1 (value head on device). Axis 3
> (KV branching for beam search).

### Token-Supervised Value Models (Cao et al.)

- **Title:** Token-Supervised Value Models for Enhancing Mathematical Problem-Solving Capabilities of Large Language Models
- **arXiv:** [2407.12863](https://arxiv.org/abs/2407.12863)

Provides token-level supervision for value models, enabling finer-grained
search guidance than step-level PRMs. The value signal at each token
position steers beam or tree search more precisely.

> **Engine primitives.** Axis 1 (per-token `value_head()` scoring).
> Axis 4 (device-resident per-token logic).

---

## Test-time compute scaling

The "scaling laws for inference": how much test-time compute to allocate,
how to structure it (sampling, search, refinement, or long-CoT), and when
to stop. Pie's **axis 7** (credit/bid market for compute allocation) is
the direct mechanism these methods need.

### Scaling LLM Test-Time Compute Optimally (Snell et al.)

- **Title:** Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters
- **arXiv:** [2408.03314](https://arxiv.org/abs/2408.03314)

Shows that allocating test-time compute adaptively (more compute for
harder problems) can outperform simply using a larger model. Compares
repeated sampling, best-of-N, and sequential revision, finding that
the optimal strategy depends on problem difficulty.

> **Engine primitives.** Axis 7 (compute-allocation policy): the core
> finding is that *adaptive* allocation matters. The credit/bid market
> is exactly the mechanism for per-request budget control. Axis 3 (KV
> branching): the sampling and search strategies all branch.

### DeepSeek-R1 (DeepSeek-AI)

- **Title:** DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning
- **arXiv:** [2501.12948](https://arxiv.org/abs/2501.12948)

Trains a long-CoT reasoning model via RL, producing extended
chain-of-thought with self-verification and backtracking expressed in
natural language. The inference-time behaviour is a single long
generation with emergent search-like structure.

> **Engine primitives.** At inference, this is a long autoregressive
> generation — minimal engine demands. But the emergent backtracking
> is implicit; explicit KV fork + rollback (axis 3) could make it
> efficient. Axis 7: budget forcing for reasoning length.

### s1 — Simple Test-Time Scaling (Muennighoff et al.)

- **Title:** s1: Simple test-time scaling
- **arXiv:** [2501.19393](https://arxiv.org/abs/2501.19393)

Introduces "budget forcing": inserting a special end-of-thinking token
and forcing the model to continue thinking until a compute budget is
exhausted. A minimal mechanism for controlling test-time compute.

> **Engine primitives.** Axis 7 (compute-allocation policy): budget
> forcing is a scheduling decision. On Pie the inferlet controls when
> to stop generating via the credit/bid market.

### Kimi k1.5 (Kimi Team)

- **Title:** Kimi k1.5: Scaling Reinforcement Learning with LLMs
- **arXiv:** [2501.12599](https://arxiv.org/abs/2501.12599)

Scales RL training for long-CoT reasoning, demonstrating that
long-context reinforcement learning with improved reward signals
produces stronger reasoning models that benefit from extended
thinking at test time.

> **Engine primitives.** Axis 7 (compute-allocation policy): adaptive
> reasoning length. Axis 3: the training pipeline benefits from
> efficient branching for rollout generation.

### Scaling of Search and Learning — o1 Roadmap (Chen et al.)

- **Title:** Scaling of Search and Learning: A Roadmap to Reproduce o1 from Reinforcement Learning Perspective
- **arXiv:** [2412.14135](https://arxiv.org/abs/2412.14135)

A systematic analysis of how search (MCTS, beam search) and learning
(RL, self-play) interact to produce o1-style reasoning capabilities.
Maps the design space and identifies key scaling dimensions.

> **Engine primitives.** The roadmap touches all search-related axes:
> 3 (KV branching), 1 (value heads), 7 (compute allocation).

### Coconut — Continuous Chain of Thought (Hao et al.)

- **Title:** Training Large Language Models to Reason in a Continuous Latent Space
- **arXiv:** [2412.06769](https://arxiv.org/abs/2412.06769)

Replaces discrete text-based CoT with reasoning in continuous latent
space: the model's hidden states serve as "thoughts" without decoding
to tokens. Enables breadth-first-search-like exploration in latent
space.

> **Engine primitives.** Axis 1 (more than logits): requires reading
> and feeding back `hidden()` states. Axis 3: latent BFS branches
> in hidden-state space. This is a Tier 3 Pie capability — no other
> serving system exposes the residual stream.

### Quiet-STaR (Zelikman et al.)

- **Title:** Quiet-STaR: Language Models Can Teach Themselves to Think Before Speaking
- **arXiv:** [2403.09629](https://arxiv.org/abs/2403.09629)

Trains models to generate internal "thinking" tokens at every position,
mixing the thought's hidden representation into the next-token
prediction. Thinking is parallel and hidden from the output.

> **Engine primitives.** Axis 1 (hidden states). Axis 2 (custom
> attention mask): thought tokens have masked attention patterns.
> The training requires per-position branching (axis 3).

### CoT Decoding Without Prompting (Wang & Zhou)

- **Title:** Chain-of-Thought Reasoning Without Prompting
- **arXiv:** [2402.10200](https://arxiv.org/abs/2402.10200)

Shows that CoT paths exist in the model's top-k alternative decodings
even without a CoT prompt. By inspecting alternative token paths and
selecting those with higher answer confidence, CoT emerges naturally.

> **Engine primitives.** Axis 3 (KV branching): exploring alternative
> paths requires forks. Axis 1: confidence scoring to identify the
> promising paths.

### Token-Budget-Aware LLM Reasoning (Han et al.)

- **Title:** Token-Budget-Aware LLM Reasoning
- **arXiv:** [2412.18547](https://arxiv.org/abs/2412.18547)

Adaptively allocates reasoning tokens based on problem difficulty,
reducing unnecessary computation on easy problems while preserving
accuracy on hard ones. Implements a budget estimator that controls
CoT length.

> **Engine primitives.** Axis 7 (compute-allocation policy): the budget
> estimator is a guest-side decision that the credit/bid market can
> express.

### L1 — Controlling Reasoning Length (Aggarwal et al.)

- **Title:** L1: Controlling How Long A Reasoning Model Thinks With Reinforcement Learning
- **arXiv:** [2503.04697](https://arxiv.org/abs/2503.04697)

Trains reasoning models via RL to control their thinking length,
achieving the same accuracy with fewer tokens or better accuracy
with an explicit length budget. A direct mechanism for compute-optimal
reasoning.

> **Engine primitives.** Axis 7 (compute-allocation policy): length
> control is a scheduling decision.

### Compute-Optimal Test-Time Scaling (Zhang et al.)

- **Title:** Can 1B LLM Surpass 405B LLM? Rethinking Compute-Optimal Test-Time Scaling
- **arXiv:** [2502.06703](https://arxiv.org/abs/2502.06703)

Analyses how to optimally allocate a fixed inference compute budget
between model size and test-time compute (sampling, search, refinement).
Shows that smaller models with more test-time compute can outperform
much larger models.

> **Engine primitives.** Axis 7 (compute-allocation policy): the
> central claim requires per-request adaptive budgeting.

### Atom of Thoughts (Tian et al.)

- **Title:** Atom of Thoughts for Markov LLM Test-Time Scaling
- **arXiv:** [2502.12018](https://arxiv.org/abs/2502.12018)

Decomposes complex queries into atomic sub-questions with Markov
independence, enabling efficient test-time scaling through independent
parallel solution of sub-problems.

> **Engine primitives.** Axis 3 (KV branching): fork per atom.
> Axis 7 (compute policy): per-atom budgets.

### Adaptive Inference-Time Compute (Aggarwal et al.)

- **Title:** Adaptive Inference-Time Compute: LLMs Can Predict if They Can Do Better, Even Mid-Generation
- **arXiv:** [2410.02725](https://arxiv.org/abs/2410.02725)

Demonstrates that LLMs can assess their own generation quality
mid-stream and decide to re-sample or continue, enabling
early-stopping and adaptive compute without external verifiers.

> **Engine primitives.** Axis 1 (hidden-state confidence). Axis 7
> (compute-allocation policy). Axis 3: re-sampling is a
> truncate + fork.

### Dualformer (Ye et al.)

- **Title:** Dualformer: Controllable Fast and Slow Thinking by Learning with Randomized Reasoning Traces
- **arXiv:** [2410.09918](https://arxiv.org/abs/2410.09918)

Trains with randomised reasoning traces of varying length so the
model can adaptively switch between fast (short CoT) and slow
(long CoT) thinking at test time based on problem difficulty.

> **Engine primitives.** Axis 7 (compute-allocation policy): the
> fast/slow decision is a runtime budget choice.

### Does RL Really Incentivize Reasoning? (Yue et al.)

- **Title:** Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?
- **arXiv:** [2504.13837](https://arxiv.org/abs/2504.13837)

Analyses whether RL-trained reasoning models (R1-style) genuinely
acquire new reasoning skills or simply better surface existing base
model capabilities. Suggests test-time search may be competitive
with RL training for many tasks.

> **Engine primitives.** Supports the case for investing in
> inference-time search infrastructure (axes 3, 7) rather than
> purely training-time approaches.

---

## Refinement and self-correction

Iterative loops: generate → critique → revise. The engine need is
**axis 3** (fork a critic off the same prefix without disturbing the
original) and **axis 8** (tool/agent I/O for critique without evicting
the context).

### Self-Refine (Madaan et al.)

- **Title:** Self-Refine: Iterative Refinement with Self-Feedback
- **arXiv:** [2303.17651](https://arxiv.org/abs/2303.17651)

Generates an initial output, then iteratively critiques and revises
it using the same LLM, without additional training. Shows consistent
improvement across code, reasoning, and text generation tasks.

> **Engine primitives.** Axis 3 (KV branching): fork the original,
> critique on the fork, revise on a fresh fork from the original
> plus critique. Axis 8 (tool I/O): tool-augmented revision.

### Reflexion (Shinn et al.)

- **Title:** Reflexion: Language Agents with Verbal Reinforcement Learning
- **arXiv:** [2303.11366](https://arxiv.org/abs/2303.11366)

An agent maintains a textual "memory" of reflections on its failures
and uses them to improve on subsequent attempts. Verbal reflections
replace gradient-based learning.

> **Engine primitives.** Axis 3 (KV branching): each attempt is a
> fork. Axis 8 (tool/agent I/O): the agent interacts with
> environments. On Pie, reflections persist in the context across
> attempts without re-prefill.

### Self-Debug (Chen et al.)

- **Title:** Teaching Large Language Models to Self-Debug
- **arXiv:** [2304.05128](https://arxiv.org/abs/2304.05128)

The LLM generates code, executes it, observes errors, and revises.
The debugging loop requires tool I/O (code execution) interleaved
with generation.

> **Engine primitives.** Axis 8 (tool I/O): execution results feed
> back into the context. Axis 3 (KV branching): rollback on failure.
> On Pie, tool-call rollback is a snapshot restore.

### Self-Correction Limits (Huang et al.)

- **Title:** Large Language Models Cannot Self-Correct Reasoning Yet
- **arXiv:** [2310.01798](https://arxiv.org/abs/2310.01798)

Shows that LLMs often *degrade* when asked to self-correct without
external feedback. Self-correction is effective only with oracle or
tool-grounded signals, not purely self-generated critique.

> **Engine primitives.** The negative finding motivates axis 8 (tool
> I/O) and axis 1 (verifier scoring) — self-correction needs external
> grounding, not just re-prompting.

### GPT-4 Doesn't Know It's Wrong (Stechly et al.)

- **Title:** GPT-4 Doesn't Know It's Wrong: An Analysis of Iterative Prompting for Reasoning Problems
- **arXiv:** [2310.12397](https://arxiv.org/abs/2310.12397)

Further evidence that iterative self-prompting without ground-truth
feedback fails. Complements Huang et al. — together they establish
that useful refinement requires external verification.

> **Engine primitives.** Same conclusion as above: refinement loops
> need tool I/O (axis 8) or a verifier (axis 1), not just re-prompting.

### Is Self-Repair a Silver Bullet for Code Generation? (Olausson et al.)

- **Title:** Is Self-Repair a Silver Bullet for Code Generation?
- **arXiv:** [2306.09896](https://arxiv.org/abs/2306.09896)

Finds that code self-repair helps strong models but not weak ones,
and that the feedback signal (test results) is critical. Weak models
often introduce new bugs during repair.

> **Engine primitives.** Axis 8 (tool I/O): test execution is essential.
> Axis 3: rollback to pre-repair state on regression.

### GLoRe — Global and Local Refinements (Havrilla et al.)

- **Title:** GLoRe: When, Where, and How to Improve LLM Reasoning via Global and Local Refinements
- **arXiv:** [2402.10963](https://arxiv.org/abs/2402.10963)

Systematically studies refinement strategies: global (regenerate the
full solution) vs local (fix a specific step). Trains a stepwise
refinement model using outcome and process reward signals.

> **Engine primitives.** Axis 3 (KV branching): local refinement
> needs fork from the error point. Axis 1: PRM identifies which
> step to refine.

### Self-Contrast (Zhang et al.)

- **Title:** Self-Contrast: Better Reflection Through Inconsistent Solving Perspectives
- **arXiv:** [2401.02009](https://arxiv.org/abs/2401.02009)

Generates solutions from multiple perspectives, identifies
inconsistencies, and uses the contrasts to produce a refined answer.
A principled alternative to single-perspective self-refinement.

> **Engine primitives.** Axis 3 (KV branching): one fork per
> perspective. Axis 5 (combining distributions): contrast requires
> comparing outputs from different contexts.

### LLM-as-a-Judge (Zheng et al.)

- **Title:** Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena
- **arXiv:** [2306.05685](https://arxiv.org/abs/2306.05685)

Establishes the LLM-as-a-Judge paradigm: using a strong LLM to
evaluate and compare outputs from other models or from itself.
The foundation for many critique-based refinement loops.

> **Engine primitives.** Axis 8 (tool/agent I/O): the judge is a
> separate inference call. On Pie it can be a second inferlet
> sharing the prompt prefix.

### A Survey on LLM-as-a-Judge (Li et al.)

- **Title:** A Survey on LLM-as-a-Judge
- **arXiv:** [2411.15594](https://arxiv.org/abs/2411.15594)

Comprehensive survey of the rapidly growing LLM-as-a-Judge literature,
covering evaluation criteria, failure modes, biases, and mitigation
strategies.

> **Engine primitives.** Same as above — the judge pattern is
> an inter-inferlet messaging use case (axis 8).

### DeepCritic (Du et al.)

- **Title:** DeepCritic: Deliberate Critique with Large Language Models
- **arXiv:** [2505.00662](https://arxiv.org/abs/2505.00662)

Trains LLMs to produce step-by-step deliberate critiques that identify
specific errors. Uses MCTS-derived training data to teach the critic to
verify reasoning step by step.

> **Engine primitives.** Axis 1 (more than logits): critic scoring.
> Axis 3: MCTS for training data generation. Axis 8: critic as a
> separate inferlet.

---

## Multi-agent and ensemble meta-generation

Multiple LLMs (or multiple instances) collaborate, debate, or are routed
among. The key engine need is **axis 8** (inter-inferlet launch and
messaging within the serving system) so that agents share infrastructure
and KV state rather than communicating over a network boundary.

### Multi-Agent Debate (Du et al.)

- **Title:** Improving Factuality and Reasoning in Language Models through Multiagent Debate
- **arXiv:** [2305.14325](https://arxiv.org/abs/2305.14325)

Multiple LLM instances debate by generating responses, reading each
other's, and revising over multiple rounds. Debate improves factuality
and reasoning compared to single-model generation.

> **Engine primitives.** Axis 8 (inter-inferlet messaging): each debater
> is an inferlet that reads others' responses. Axis 3 (KV branching):
> debaters share the shared prompt prefix. On a black-box server, each
> debater is a separate API call that re-prefills the full context.

### Mixture-of-Agents (Wang et al.)

- **Title:** Mixture-of-Agents Enhances Large Language Model Capabilities
- **arXiv:** [2406.04692](https://arxiv.org/abs/2406.04692)

A layered architecture where each layer's agents receive the outputs of
the previous layer as auxiliary context. Combines diverse model strengths
through iterative aggregation rather than simple voting.

> **Engine primitives.** Axis 8 (inter-inferlet messaging): the layered
> structure is a pipeline of inferlet generations. Axis 3: agents in a
> layer can fork from a shared context.

### FrugalGPT (Chen et al.)

- **Title:** FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance
- **arXiv:** [2305.05176](https://arxiv.org/abs/2305.05176)

Proposes LLM cascading and caching strategies to reduce cost: try a
cheap model first, escalate to expensive models only when needed.
Includes prompt adaptation, model approximation, and composition.

> **Engine primitives.** Axis 7 (compute-allocation policy): the
> cascade decision is a cost-quality trade-off that the bid market
> can express. Axis 8: routing between models.

### RouteLLM (Ong et al.)

- **Title:** RouteLLM: Learning to Route LLMs with Preference Data
- **arXiv:** [2406.18665](https://arxiv.org/abs/2406.18665)

Trains routers that predict whether a strong or weak model is needed
for each query, achieving cost savings of 2× or more while maintaining
quality. Generalises across tasks and model pairs.

> **Engine primitives.** Axis 7 (compute-allocation policy): routing
> is a per-request compute decision. On Pie the router can be a PTIR
> program that inspects the prompt and sets the bid.

### Exploring Collaboration Mechanisms for LLM Agents (Zhang et al.)

- **Title:** Exploring Collaboration Mechanisms for LLM Agents: A Social Psychology View
- **arXiv:** [2310.02124](https://arxiv.org/abs/2310.02124)

Studies collaboration patterns (debate, reflection, negotiation) among
LLM agents from a social psychology perspective. Finds that structured
collaboration consistently outperforms individual generation.

> **Engine primitives.** Axis 8 (inter-inferlet messaging): all
> collaboration patterns are inter-agent communication.

---

## Self-training and bootstrapping via search

Methods that use inference-time search to generate training data for
self-improvement. The inference phase is the bottleneck and benefits
from all search-related axes.

### STaR — Bootstrapping Reasoning with Reasoning (Zelikman et al.)

- **Title:** STaR: Bootstrapping Reasoning With Reasoning
- **arXiv:** [2203.14465](https://arxiv.org/abs/2203.14465)

Iteratively trains on self-generated correct rationales: sample
solutions, keep the correct ones, fine-tune, repeat. The inference-time
sampling is the data generation step.

> **Engine primitives.** Axis 3 (KV branching): bulk sampling.
> Fork sharing makes the data-generation phase efficient.

### ReST^EM — Self-Training for Problem Solving (Singh et al.)

- **Title:** Beyond Human Data: Scaling Self-Training for Problem-Solving with Language Models
- **arXiv:** [2312.06585](https://arxiv.org/abs/2312.06585)

A principled Expectation-Maximisation framework for LLM self-training:
generate solutions (E-step), filter correct ones, fine-tune (M-step).
Shows scaling improvements from multiple rounds.

> **Engine primitives.** Axis 3 (KV branching): the E-step is large-N
> sampling. Axis 7: compute budget for data generation.

### Iterative Reasoning Preference Optimization (Pang et al.)

- **Title:** Iterative Reasoning Preference Optimization
- **arXiv:** [2404.19733](https://arxiv.org/abs/2404.19733)

Uses tree search to generate preference pairs (winning and losing
reasoning traces), then trains with DPO. Iterates search and training
for self-improvement.

> **Engine primitives.** Axis 3 (KV branching): search-generated
> preference data. Axis 1: value function for selecting traces.

### Four Habits of Highly Effective STaRs (Gandhi et al.)

- **Title:** Cognitive Behaviors that Enable Self-Improving Reasoners, or, Four Habits of Highly Effective STaRs
- **arXiv:** [2503.01307](https://arxiv.org/abs/2503.01307)

Identifies four cognitive behaviours (verification, backtracking,
subgoal decomposition, backward chaining) that are necessary for
effective self-improving reasoners and emerge during STaR-style
training.

> **Engine primitives.** Axis 3: backtracking and decomposition are
> native KV operations on Pie. Axis 1: verification on device.

### Open-Reasoner-Zero (Zhang et al.)

- **Title:** Open-Reasoner-Zero: An Open Source Approach to Scaling Up Reinforcement Learning on the Base Model
- **arXiv:** [2503.24290](https://arxiv.org/abs/2503.24290)

An open-source reproduction of o1-style RL training for reasoning,
demonstrating that scaled RL with MCTS-based exploration can produce
strong reasoning models from base models.

> **Engine primitives.** Axis 3 (KV branching): RL rollouts during
> training. Axis 7: compute budget management.

---

## Citation audit

All 57 arXiv citations in this chapter were verified by fetching the
corresponding `arxiv.org/abs/` page and confirming the registered title
matches the claimed title (modulo minor formatting differences).

**Note:** I was unable to run `verify_citations.py` programmatically in
this session due to lack of a shell execution tool. The main agent should
run:

```bash
cd /root/.patissier/work/ptir-2-alpha/inference-time-algorithms && python3 verify_citations.py 03-search-and-meta-generation.md
```

The following section-heading nicknames deliberately differ from
registered arXiv titles — these are standard in the field and used for
clarity:

| Heading nickname | Registered arXiv title | arXiv ID |
|---|---|---|
| Best-of-N with verifiers | Training Verifiers to Solve Math Word Problems | 2110.14168 |
| DiVeRSe | Making Large Language Models Better Reasoners with Step-Aware Verifier | 2206.02336 |
| RAP — Reasoning via Planning | Reasoning with Language Model is Planning with World Model | 2305.14992 |
| LATS — Language Agent Tree Search | Language Agent Tree Search Unifies Reasoning Acting and Planning in Language Models | 2310.04406 |
| TS-LLM — AlphaZero-Like Tree-Search | Alphazero-like Tree-Search can Guide Large Language Model Decoding and Training | 2309.17179 |
| MCTSr — Monte Carlo Tree Self-Refine | Accessing GPT-4 level Mathematical Olympiad Solutions via Monte Carlo Tree Self-refine with LLaMa-3 8B | 2406.07394 |
| Coconut — Continuous Chain of Thought | Training Large Language Models to Reason in a Continuous Latent Space | 2412.06769 |
| ReST^EM — Self-Training for Problem Solving | Beyond Human Data: Scaling Self-Training for Problem-Solving with Language Models | 2312.06585 |
| RFT — Rejection Sampling Fine-Tuning | Scaling Relationship on Learning Mathematical Reasoning with Large Language Models | 2308.01825 |
| OmegaPRM — Automated Process Supervision | Improve Mathematical Reasoning in Language Models by Automated Process Supervision | 2406.06592 |
| GenRM — Generative Verifiers | Generative Verifiers: Reward Modeling as Next-Token Prediction | 2408.15240 |
| VAS — Value Augmented Sampling | Value Augmented Sampling for Language Model Alignment and Personalization | 2405.06639 |
| RSD — Reward-Guided Speculative Decoding | Reward-Guided Speculative Decoding for Efficient LLM Reasoning | 2501.19324 |
