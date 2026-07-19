# Speculative decoding families, self-drafting, tree verification, MTP, early exit, cascades

A survey of speculative decoding and inference-efficiency methods, read through the
Pie expressiveness lens. Every entry notes which of the eight engine-primitive axes
from `00-pie-capability-map.md` §10 the method requires — and therefore whether it is
user-space on Pie or requires an engine patch elsewhere.

Companion surveys for context:
Xia et al., [2401.07851](https://arxiv.org/abs/2401.07851) (ACL 2024 Findings);
Zhang et al., [2404.14897](https://arxiv.org/abs/2404.14897).

---

## Foundational speculative decoding

### Speculative Decoding (Leviathan et al.)

- **Title:** Fast Inference from Transformers via Speculative Decoding
- **arXiv:** [2211.17192](https://arxiv.org/abs/2211.17192)

A small draft model proposes γ tokens; the target model scores them in a single
batched forward pass. A rejection-sampling acceptance scheme guarantees the output
distribution is identical to the target's — the key "lossless" property. Expected
tokens per step is 1/(1−α) where α is per-token acceptance rate.

**Engine primitive:** Axis 6 (custom draft/verify rule). The accept/reject loop
must access both draft and target distributions and perform KV rollback on rejection
(axis 3). On vLLM/SGLang the draft/verify schedule is engine-internal; on Pie it is
a user-written inferlet with working-vs-committed page rollback.

### Speculative Sampling (Chen et al. / DeepMind)

- **Title:** Accelerating Large Language Model Decoding with Speculative Sampling
- **arXiv:** [2302.01318](https://arxiv.org/abs/2302.01318)

Independent co-discovery of speculative decoding. Frames the acceptance as modified
rejection sampling from q(x) toward p(x), with a residual distribution correction
step. Demonstrates the technique at scale on Chinchilla 70B with a 4B draft model.

**Engine primitive:** Same as above — axis 6 (custom draft/verify) and axis 3
(KV rollback on rejection).

### Blockwise Parallel Decoding (Stern et al.)

- **Title:** Blockwise Parallel Decoding for Deep Autoregressive Models
- **arXiv:** [1811.03115](https://arxiv.org/abs/1811.03115)

Early precursor: k auxiliary prediction heads are added on top of the base model to
predict the next k tokens simultaneously. Verification is greedy — accept the longest
prefix that matches the base model. No separate draft model needed; the heads are
cheap feedforward layers.

**Engine primitive:** Axis 1 (more than logits — needs intermediate hidden states to
feed auxiliary heads) and axis 6 (custom verify rule). Heads must be fused into the
forward pass.

---

## Self-drafting / no-separate-draft-model methods

### Medusa

- **Title:** Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads
- **arXiv:** [2401.10774](https://arxiv.org/abs/2401.10774)

Adds k extra LM heads to the base model, each predicting position +i. Candidates
are assembled into a tree and verified with tree attention in one forward pass.
Medusa-1 freezes the backbone; Medusa-2 jointly fine-tunes heads and backbone.
Achieves 2.2–3.6× speedup.

**Engine primitive:** Axis 2 (custom tree attention mask for multi-candidate
verification) and axis 1 (hidden-state access for the heads). On Pie the tree mask
is a guest-bound channel; elsewhere it is an engine patch.

### EAGLE — Speculative Sampling Requires Rethinking Feature Uncertainty

- **Title:** EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty
- **arXiv:** [2401.15077](https://arxiv.org/abs/2401.15077)

Drafts at the hidden-state (feature) level rather than token level: a lightweight
autoregressive head consumes the target model's second-to-last-layer features to
predict the next feature, then projects to logits. This captures feature-level
uncertainty that token-level heads miss, yielding higher acceptance rates than Medusa.

**Engine primitive:** Axis 1 (hidden states — requires `hidden()` tap) and axis 6
(custom draft/verify). The feature-level draft head needs direct access to the
residual stream mid-forward, unavailable on black-box servers.

### EAGLE-2 — Dynamic Draft Trees

- **Title:** EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees
- **arXiv:** [2406.16858](https://arxiv.org/abs/2406.16858)

Extends EAGLE with context-aware dynamic draft trees: at each step the draft tree
topology is adjusted based on the confidence distribution of the draft head, allocating
more branches to uncertain positions. Achieves ~1.4× over EAGLE-1.

**Engine primitive:** Axis 1 + axis 2 (dynamic tree mask constructed per step) +
axis 6. The per-step tree topology change requires a programmable mask channel.

### EAGLE-3 — Scaling up via Training-Time Test

- **Title:** EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test
- **arXiv:** [2503.01840](https://arxiv.org/abs/2503.01840)

Abandons autoregressive feature prediction; instead uses multi-layer feature fusion
and direct token prediction with a "training-time test" technique that mimics
inference during training. Fully leverages data scaling, achieving up to 6.5× speedup
over vanilla decoding and ~1.4× over EAGLE-2.

**Engine primitive:** Axis 1 (multi-layer hidden-state access) + axis 2 (tree mask)
+ axis 6.

### Hydra — Sequentially-Dependent Draft Heads

- **Title:** Hydra: Sequentially-Dependent Draft Heads for Medusa Decoding
- **arXiv:** [2402.05109](https://arxiv.org/abs/2402.05109)

Improves Medusa by making heads sequentially dependent: each head conditions on the
token predicted by the previous head, improving inter-position coherence and
acceptance rates. 1.31× speedup over Medusa, 2.70× over standard decoding.

**Engine primitive:** Axis 1 + axis 2 (tree mask for verification) + axis 6.

### Clover — Regressive Lightweight Speculative Decoding

- **Title:** Clover: Regressive Lightweight Speculative Decoding with Sequential Knowledge
- **arXiv:** [2405.00263](https://arxiv.org/abs/2405.00263)

Uses a lightweight RNN-based draft head with a "Regressive Connection" to inject
sequential knowledge into parallel prediction. An Augmenting Block aligns hidden
states for speculative generation, improving hit rate over Medusa-style independent
heads.

**Engine primitive:** Axis 1 (hidden-state access) + axis 6 (custom draft/verify).

### Clover-2

- **Title:** Clover-2: Accurate Inference for Regressive Lightweight Speculative Decoding
- **arXiv:** [2408.00264](https://arxiv.org/abs/2408.00264)

Extends Clover with architectural refinements and knowledge distillation from the
target model, improving RNN-based drafting accuracy to match transformer-based drafts
while retaining computational efficiency.

**Engine primitive:** Same as Clover — axis 1 + axis 6.

### Recurrent Drafter (ReDrafter)

- **Title:** Recurrent Drafter for Fast Speculative Decoding in Large Language Models
- **arXiv:** [2403.09919](https://arxiv.org/abs/2403.09919)

Lightweight RNN drafter conditioned on the target LLM's hidden states, combined with
dynamic tree attention over beam-search candidates. Knowledge distillation trains the
RNN to align with the target. Up to 2.8× speedup on H100.

**Engine primitive:** Axis 1 (hidden-state conditioning) + axis 2 (dynamic tree
attention mask) + axis 6.

### Speculative Streaming

- **Title:** Speculative Streaming: Fast LLM Inference without Auxiliary Models
- **arXiv:** [2402.11131](https://arxiv.org/abs/2402.11131)

Fine-tunes the target model itself to predict future n-grams via multi-stream
attention, eliminating the need for a separate draft model. Up to 10,000× fewer extra
parameters than Medusa. Achieves 1.8–3.1× speedup across diverse generation tasks.

**Engine primitive:** Axis 1 (shared hidden states between streams) + axis 6. The
multi-stream attention pattern requires axis 2 (custom mask).

### Lookahead Decoding

- **Title:** Break the Sequential Dependency of LLM Inference Using Lookahead Decoding
- **arXiv:** [2402.02057](https://arxiv.org/abs/2402.02057)

Casts autoregressive generation as a Jacobi fixed-point iteration: all future
positions are initialized and iteratively refined in parallel until convergence. A
lookahead branch generates n-gram candidates while a verification branch confirms
them. Training-free; works with any off-the-shelf model.

**Engine primitive:** Axis 2 (custom attention mask for the Jacobi window) + axis 3
(KV management for speculative positions) + axis 6.

### Jacobi Decoding (Santilli et al.)

- **Title:** Accelerating Transformer Inference for Translation via Parallel Decoding
- **arXiv:** [2305.10427](https://arxiv.org/abs/2305.10427)

Original formulation of Jacobi/Gauss-Seidel fixed-point iteration for parallel
autoregressive decoding. Applied primarily to machine translation; demonstrates that
standard AR models can be decoded in parallel without retraining by treating
generation as a system of equations.

**Engine primitive:** Axis 2 (custom mask for parallel positions) + axis 6.

### CLLMs — Consistency Large Language Models

- **Title:** CLLMs: Consistency Large Language Models
- **arXiv:** [2403.00835](https://arxiv.org/abs/2403.00835)

Trains LLMs for Jacobi-decoding consistency: the model learns to predict the
fixed-point output regardless of the initial state, so Jacobi iteration converges in
far fewer steps. 2.4–3.4× speedup over standard AR decoding with no auxiliary model
and no extra parameters.

**Engine primitive:** Axis 6 (the Jacobi iteration is the decode loop) + axis 2.

### LayerSkip — Early Exit and Self-Speculative Decoding

- **Title:** LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding
- **arXiv:** [2404.16710](https://arxiv.org/abs/2404.16710)

Trains with layer dropout and an early-exit loss so that early layers produce usable
logits. At inference, early layers draft tokens (self-speculation) and the full model
verifies. No separate draft model — the draft is the model's own early layers.

**Engine primitive:** Axis 1 (per-layer logits via early exit, needs `layer()` /
early-exit hidden readout) + axis 6 (self-speculative verify). On Pie, `layer()` is
wired; on vLLM/SGLang, early exit requires engine modification.

### Draft & Verify — Self-Speculative Decoding

- **Title:** Draft & Verify: Lossless Large Language Model Acceleration via Self-Speculative Decoding
- **arXiv:** [2309.08168](https://arxiv.org/abs/2309.08168)

Skips intermediate layers during drafting (layer-level self-speculation), then verifies
with the full model. Introduces Bayesian optimization to select which layers to skip.
No auxiliary model needed.

**Engine primitive:** Axis 1 (layer-skip control, needs per-layer execution control)
+ axis 6.

### Kangaroo — Double Early Exiting

- **Title:** Kangaroo: Lossless Self-Speculative Decoding via Double Early Exiting
- **arXiv:** [2404.18911](https://arxiv.org/abs/2404.18911)

Trains a lightweight adapter on a fixed early-exit point to produce draft tokens.
A second early-exit criterion decides when to stop drafting. Achieves 1.7× speedup
on Vicuna-7B with a single fixed shallow sub-network as the drafter.

**Engine primitive:** Axis 1 (early-exit hidden states) + axis 6.

---

## Retrieval / n-gram / cache-based drafting

### Prompt Lookup Decoding

No arXiv paper; released as open-source code by Saxena (2023,
[github.com/apoorvumang/prompt-lookup-decoding](https://github.com/apoorvumang/prompt-lookup-decoding)).

Generates draft candidates by n-gram string matching against the input prompt.
When the current suffix matches a substring in the prompt, the subsequent tokens are
used as the draft. Zero overhead, no model, extremely effective for input-grounded
tasks (summarization, code editing). Implemented in Pie as the `cacheback` inferlet.

**Engine primitive:** Axis 6 (custom draft rule) + axis 3 (KV rollback on rejection).
Pure user-space on Pie.

### REST — Retrieval-Based Speculative Decoding

- **Title:** REST: Retrieval-Based Speculative Decoding
- **arXiv:** [2311.08252](https://arxiv.org/abs/2311.08252)

Replaces the draft model with a retrieval datastore: at each step, retrieve the
longest suffix match from a token-level datastore and use the continuation as the
draft. Effective when the output distribution overlaps a large corpus. Up to 2.36×
speedup.

**Engine primitive:** Axis 6 (custom draft source — retrieval instead of a model) +
axis 8 (I/O interleaved with generation, for datastore access without losing cache).

### LLMA — Inference with Reference

- **Title:** Inference with Reference: Lossless Acceleration of Large Language Models
- **arXiv:** [2304.04487](https://arxiv.org/abs/2304.04487)

When a reference document is available (e.g., a document to summarize), LLMA copies
spans from the reference as draft tokens and verifies them. Achieves lossless
acceleration by exploiting the high overlap between reference and output in grounded
tasks.

**Engine primitive:** Axis 6 (copy-based draft rule) + axis 3 (rollback on mismatch).

### SuffixDecoding

- **Title:** SuffixDecoding: Extreme Speculative Decoding for Emerging AI Applications
- **arXiv:** [2411.04975](https://arxiv.org/abs/2411.04975)

Builds efficient suffix trees from past outputs to cache and query long token
sequences. Adaptively adjusts draft length based on match quality. Particularly
effective for agentic/repetitive workloads (SWE-Bench, text-to-SQL). Up to 5.3×
speedup.

**Engine primitive:** Axis 6 + axis 7 (adaptive draft length policy based on
suffix-tree match confidence).

### Token Recycling

- **Title:** Turning Trash into Treasure: Accelerating Inference of Large Language Models with Token Recycling
- **arXiv:** [2408.08696](https://arxiv.org/abs/2408.08696)

Reuses rejected candidate tokens from previous verification steps. Stores candidates
in a compact adjacency matrix (<2 MB); constructs a draft tree via BFS from the
matrix, verified with tree attention. Training-free, ~2× speedup.

**Engine primitive:** Axis 2 (tree attention for verification) + axis 4 (per-token
state: the adjacency matrix is stateful, ideally device-resident) + axis 6.

---

## Tree-structured drafting and verification

### SpecInfer — Tree-based Speculative Inference

- **Title:** SpecInfer: Accelerating Generative Large Language Model Serving with Tree-based Speculative Inference and Verification
- **arXiv:** [2305.09781](https://arxiv.org/abs/2305.09781)

Uses multiple small draft models (a boost-tuned ensemble) to propose a tree of
candidates, verified in a single target forward pass with tree attention. Introduces
the "token tree" abstraction and tree-attention verification kernel.

**Engine primitive:** Axis 2 (tree attention mask — the core innovation) + axis 3
(KV for divergent branches; shared prefix pages) + axis 6. Tree attention masks are
not guest-expressible on vLLM/SGLang; on Pie they are a mask-channel binding.

### Sequoia — Hardware-aware Speculative Decoding

- **Title:** Sequoia: Scalable, Robust, and Hardware-aware Speculative Decoding
- **arXiv:** [2402.12374](https://arxiv.org/abs/2402.12374)

Optimizes the draft-tree topology for a given hardware target: uses a dynamic
programming algorithm to select the tree structure maximizing expected accepted tokens
per unit wall-clock time. Robust across temperatures and models.

**Engine primitive:** Axis 2 (optimized tree mask) + axis 6 + axis 7 (hardware-aware
draft-tree selection is a compute-allocation policy).

### Staged Speculative Decoding

- **Title:** Accelerating LLM Inference with Staged Speculative Decoding
- **arXiv:** [2308.04623](https://arxiv.org/abs/2308.04623)

Extends speculative decoding with tree-structured speculation and multi-stage
verification. After the first round of speculation, a second round reuses verified
tokens. Achieves 3.16× latency reduction for batch inference with GPT-2-L.

**Engine primitive:** Axis 2 (tree mask) + axis 6 (multi-stage verify).

### Multi-Candidate Speculative Decoding

- **Title:** Multi-Candidate Speculative Decoding
- **arXiv:** [2401.06706](https://arxiv.org/abs/2401.06706)

Extends the standard single-sequence draft to multiple candidate sequences verified
in parallel. Proposes topology-aware causal masks for efficient batched verification
of diverse candidates, with target-model initialization of draft sequences.

**Engine primitive:** Axis 2 (multi-candidate attention mask) + axis 3 (KV for
multiple branches) + axis 6.

### SpecTr — Optimal Transport Acceptance

- **Title:** SpecTr: Fast Speculative Decoding via Optimal Transport
- **arXiv:** [2310.15141](https://arxiv.org/abs/2310.15141)

Recasts multi-candidate draft acceptance as a discrete optimal transport problem.
Derives a (1−1/e)-optimal acceptance algorithm with near-linear-in-vocabulary
computation. 2.13× wall-clock speedup (1.37× over standard speculative decoding).

**Engine primitive:** Axis 6 (the OT-based acceptance rule replaces standard
rejection sampling — must be fused into the device verify step on Pie via PTIR).

---

## Multi-token prediction

### Better & Faster LLMs via Multi-token Prediction (Gloeckle et al.)

- **Title:** Better & Faster Large Language Models via Multi-token Prediction
- **arXiv:** [2404.19737](https://arxiv.org/abs/2404.19737)

Trains the model with k auxiliary prediction heads, each predicting token +i. At
inference the extra heads serve as a native draft source — no separate model needed.
Also improves training signal, especially for code. Foundational for MTP-based
speculation.

**Engine primitive:** Axis 1 (MTP heads require hidden-state access; Pie exposes
`mtp_logits(k)` natively) + axis 6 (the MTP draft/verify loop is a custom rule).
Pie already ships `mtp-specdecode`, `mtp-native-verify`, and `mtp-grammar` inferlets.

### DeepSeek-V3 MTP Modules

- **Title:** DeepSeek-V3 Technical Report
- **arXiv:** [2412.19437](https://arxiv.org/abs/2412.19437)

671B-parameter MoE model (37B active) trained with a multi-token prediction objective.
MTP modules are integrated into the architecture, enabling native speculative decoding
from the MTP heads at inference with no auxiliary model.

**Engine primitive:** Axis 1 (`mtp_logits(k)`) + axis 6. Pie's `mtp_logits` intrinsic
directly taps these heads.

---

## Serving-system-level speculation

### Online Speculative Decoding (OSD)

- **Title:** Online Speculative Decoding
- **arXiv:** [2310.07177](https://arxiv.org/abs/2310.07177)

Continuously updates the draft model during serving via online distillation from
observed queries. Adapts the draft to the actual query distribution, improving
acceptance rate by 0.1–0.65 and latency by 1.42–2.17×.

**Engine primitive:** Axis 7 (adaptive draft policy — the system needs a per-request
or per-epoch draft-update policy) + axis 6.

### SpecDec++ — Adaptive Candidate Lengths

- **Title:** SpecDec++: Boosting Speculative Decoding via Adaptive Candidate Lengths
- **arXiv:** [2405.19715](https://arxiv.org/abs/2405.19715)

Models draft-length selection as an MDP; the optimal policy has a threshold form.
Augments the draft model with a trained acceptance-prediction head that estimates
rejection probability per step. Stops drafting when the probability exceeds the
threshold.

**Engine primitive:** Axis 7 (adaptive draft length is a compute-allocation policy)
+ axis 6 (the acceptance head must be fused into the draft step).

### TurboSpec — Closed-loop Speculation Control

- **Title:** TurboSpec: Closed-loop Speculation Control System for Optimizing LLM Serving Goodput
- **arXiv:** [2406.14066](https://arxiv.org/abs/2406.14066)

Feedback-driven control system that dynamically tunes speculation parameters (draft
length, batch composition) based on real-time serving conditions. Maximizes "goodput"
(rate of successfully generated tokens) across varying loads and hardware.

**Engine primitive:** Axis 7 (guest compute-allocation policy). Requires the serving
system to expose speculation knobs to a controller — on Pie this is the credit/bid
market.

### MagicDec — Latency-Throughput Tradeoff for Long Context

- **Title:** MagicDec: Breaking the Latency-Throughput Tradeoff for Long Context Generation with Speculative Decoding
- **arXiv:** [2408.11049](https://arxiv.org/abs/2408.11049)

Shows that speculative decoding recovers its advantage in high-throughput batched
settings when sequences are long (KV-cache-heavy, memory-bound). Uses sparse KV
for the draft model to keep drafting fast even with long contexts.

**Engine primitive:** Axis 7 (adaptive policy: speculation is beneficial only when
memory-bound) + axis 3 (KV management for draft vs. target).

### BASS — Batched Attention-optimized Speculative Sampling

- **Title:** BASS: Batched Attention-optimized Speculative Sampling
- **arXiv:** [2404.15778](https://arxiv.org/abs/2404.15778)

Addresses the "ragged tensor" problem in batched speculative decoding, where
sequences in the same batch accept different numbers of tokens. Proposes optimized
batched attention for speculative sampling. Notes diminishing returns at very large
batch sizes.

**Engine primitive:** Axis 6 + axis 7 (batch-aware speculation policy).

### Dynamic Speculation Lookahead

- **Title:** Dynamic Speculation Lookahead Accelerates Speculative Decoding of Large Language Models
- **arXiv:** [2405.04304](https://arxiv.org/abs/2405.04304)

Dynamically adjusts the speculation window length based on runtime statistics of
acceptance rates. Longer windows when acceptance is high, shorter when it drops.

**Engine primitive:** Axis 7 (adaptive draft length) + axis 6.

### DistillSpec — Knowledge Distillation for Drafting

- **Title:** DistillSpec: Improving Speculative Decoding via Knowledge Distillation
- **arXiv:** [2310.08461](https://arxiv.org/abs/2310.08461)

Uses knowledge distillation (on-policy, with custom divergence functions) to train
the draft model to better approximate the target, improving acceptance rates by
10–45%. Shows that distill-for-performance then distill-for-SD yields best results.

**Engine primitive:** Axis 6 (the improved draft still needs the standard verify
loop). Training-time method; inference is standard speculative decoding.

### PEARL — Parallel Speculative Decoding with Adaptive Draft Length

- **Title:** PEARL: Parallel Speculative Decoding with Adaptive Draft Length
- **arXiv:** [2408.11850](https://arxiv.org/abs/2408.11850)

Introduces pre-verify (verify first draft token during drafting) and post-verify
(continue drafting during verification) to overlap draft and verify phases. Achieves
adaptive draft length and up to 4.43× speedup.

**Engine primitive:** Axis 6 + axis 7 (adaptive length). The overlapped
draft/verify requires fine-grained scheduling control.

### Ouroboros — Phrase-by-Phrase Drafting

- **Title:** Ouroboros: Generating Longer Drafts Phrase by Phrase for Faster Speculative Decoding
- **arXiv:** [2402.13720](https://arxiv.org/abs/2402.13720)

Generates drafts phrase-by-phrase (not token-by-token), using a candidate pool
informed by past verification outcomes. Training-free. Up to 2.8× over standard
speculative decoding.

**Engine primitive:** Axis 6 (custom phrase-level draft rule) + axis 3 (KV for
speculative phrases).

### Decoding Speculative Decoding

- **Title:** Decoding Speculative Decoding
- **arXiv:** [2402.01528](https://arxiv.org/abs/2402.01528)

Empirical analysis showing that draft model *latency*, not language modeling quality,
is the primary determinant of speculative decoding speedup. Explores hardware-efficient
draft architectures optimized for acceptance throughput.

**Engine primitive:** Axis 6 + axis 7 (design-space exploration for draft model
selection is a system-level policy).

### SPEED — Speculative Pipelined Execution

- **Title:** SPEED: Speculative Pipelined Execution for Efficient Decoding
- **arXiv:** [2310.12072](https://arxiv.org/abs/2310.12072)

Exploits parameter-shared decoders to speculatively execute multiple tokens in
parallel by amortizing memory operations. Early predictions from early layers feed
subsequent positions, pipelining compute and memory access.

**Engine primitive:** Axis 1 (early-layer predictions) + axis 6. Requires
parameter-shared architectures.

### Faster Cascades via Speculative Decoding

- **Title:** Faster Cascades via Speculative Decoding
- **arXiv:** [2405.19261](https://arxiv.org/abs/2405.19261)

Combines model cascades (use small model by default, defer to large on "hard" inputs)
with speculative decoding: the cascade's deferral rule is implemented via speculative
execution. Derives the optimal deferral rule and achieves better cost-quality
tradeoffs than either method alone.

**Engine primitive:** Axis 6 + axis 7 (the deferral/cascade policy is a
compute-allocation decision).

### Minions — Aggregated Speculative Execution

- **Title:** Minions: Accelerating Large Language Model Inference with Aggregated Speculative Execution
- **arXiv:** [2402.15678](https://arxiv.org/abs/2402.15678)

Uses *multiple* small speculative models (SSMs) in parallel with majority-voted
aggregation. Adaptive speculation length and pipelined execution between SSMs and the
target model. Disaggregated architecture for hardware efficiency.

**Engine primitive:** Axis 5 (combining several distributions — multiple draft models
fused) + axis 6 + axis 7 (adaptive length policy).

---

## Constrained + speculative composition

### DOMINO — Grammar-Constrained Speculative Decoding

- **Title:** Guiding LLMs The Right Way: Fast, Non-Invasive Constrained Generation
- **arXiv:** [2403.06988](https://arxiv.org/abs/2403.06988)

Shows that grammar constraints and speculative decoding can be composed efficiently
via subword-aligned constraint checking and pre-computation. Achieves near-zero
overhead (sometimes 2× speedup) versus unconstrained decoding.

**Engine primitive:** Axis 6 (speculation composed with constraints). Pie ships
`mtp-grammar` as an inferlet demonstrating exactly this composition — speculation +
grammar masks in a single pass.

---

## Early exit, cascades, and dynamic depth

### CALM — Confident Adaptive Language Modeling

- **Title:** Confident Adaptive Language Modeling
- **arXiv:** [2207.07061](https://arxiv.org/abs/2207.07061)

Confidence-based early exit: "easy" tokens (high confidence at an early layer) skip
remaining layers, while "hard" tokens use the full stack. Up to 3× speedup with
minimal quality loss. Requires per-token per-layer confidence estimation.

**Engine primitive:** Axis 1 (per-layer hidden states / early-exit logits, needs
`layer()` + early readout) + axis 4 (per-token confidence state).

### Big Little Decoder (BiLD)

- **Title:** Speculative Decoding with Big Little Decoder
- **arXiv:** [2302.07863](https://arxiv.org/abs/2302.07863)

Pairs a small AR model (for fast decoding) with a large model that intervenes
selectively. Introduces a fallback policy (when to call the big model) and a rollback
policy (correct small model mistakes). Up to 2.12× speedup.

**Engine primitive:** Axis 5 (two models in lockstep) + axis 3 (rollback on
correction) + axis 6.

### Mixture-of-Depths

- **Title:** Mixture-of-Depths: Dynamically allocating compute in transformer-based language models
- **arXiv:** [2404.02258](https://arxiv.org/abs/2404.02258)

Each layer decides which tokens to process via top-k routing; non-selected tokens
skip self-attention and MLP. Compute budget per forward pass is fixed but dynamically
allocated. Up to 50% faster inference with matched performance.

**Engine primitive:** Axis 1 (per-layer routing decisions, needs layer-level control)
+ axis 4 (routing state).

### Deja Vu — Contextual Sparsity

- **Title:** Deja Vu: Contextual Sparsity for Efficient LLMs at Inference Time
- **arXiv:** [2310.17157](https://arxiv.org/abs/2310.17157)

Discovers that for each input, a small input-dependent subset of attention heads and
MLP parameters produces nearly the same output as the full model. A low-cost predictor
identifies the sparse set per step. >2× latency reduction on OPT-175B. No retraining.

**Engine primitive:** Axis 1 (needs activation-level inspection to predict sparse
sets) + axis 4 (predictor state). The sparse-set prediction must run inside the
forward pass — requires engine integration on black-box servers.

### PowerInfer — Activation Sparsity on Consumer GPUs

- **Title:** PowerInfer: Fast Large Language Model Serving with a Consumer-grade GPU
- **arXiv:** [2312.12456](https://arxiv.org/abs/2312.12456)

Exploits the power-law distribution in neuron activations: "hot" neurons are
GPU-resident, "cold" neurons computed on CPU. Achieves 11.69× over llama.cpp on a
single RTX 4090. Relies on ReLU-family activation sparsity (>90% in FFN layers).

**Engine primitive:** Axis 1 (neuron-level activation sparsity prediction) + custom
scheduling. A system-level optimization that Pie's frame model could support via
heterogeneous slot submission.

---

## Parallel decoding for diffusion language models

### MDLM — Masked Diffusion Language Models

- **Title:** Simple and Effective Masked Diffusion Language Models
- **arXiv:** [2406.07524](https://arxiv.org/abs/2406.07524)

Treats language modeling as masked discrete diffusion. A simplified Rao-Blackwellized
objective turns diffusion training into a mixture of masked LM losses.
Semi-autoregressive decoding enables 25–30× faster generation than prior diffusion
models. Competitive perplexity with autoregressive models.

**Engine primitive:** Axis 2 (arbitrary remasking/unmasking attention patterns) +
axis 3 (non-sequential KV construction). Pie's mask channel + `readout()` indices
are the exact primitives required.

### SEDD — Score Entropy Discrete Diffusion

- **Title:** Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution
- **arXiv:** [2310.16834](https://arxiv.org/abs/2310.16834)

Introduces "score entropy" loss extending score matching to discrete domains.
Matches or surpasses GPT-2 in perplexity with up to 32× less compute. Enables
flexible generation including arbitrary infilling.

**Engine primitive:** Axis 2 (arbitrary attention masks for diffusion schedules) +
axis 6 (the denoising decode loop is a custom rule).

### LLaDA — Large Language Diffusion with Masking

- **Title:** Large Language Diffusion Models
- **arXiv:** [2502.09992](https://arxiv.org/abs/2502.09992)

Scales masked diffusion to 8B parameters. Forward process masks tokens progressively;
reverse process predicts all masked tokens in parallel. Matches LLaMA3-8B on
benchmarks. Excels at reversal reasoning tasks. Full bidirectional context at every
step.

**Engine primitive:** Axis 2 (arbitrary masks per diffusion step, non-left-to-right)
+ custom `readout()` indices. Pie's mask channel is the enabling primitive.

---

## Surveys

### Comprehensive Survey of Speculative Decoding (Xia et al.)

- **Title:** Unlocking Efficiency in Large Language Model Inference: A Comprehensive Survey of Speculative Decoding
- **arXiv:** [2401.07851](https://arxiv.org/abs/2401.07851)

First unified formal definition and taxonomy of speculative decoding. Introduces
Spec-Bench for standardized comparison. Covers drafter architectures, verification
strategies, and deployment considerations.

### Beyond the Speculative Game (Zhang et al.)

- **Title:** Beyond the Speculative Game: A Survey of Speculative Execution in Large Language Models
- **arXiv:** [2404.14897](https://arxiv.org/abs/2404.14897)

Broader survey covering speculative execution from an execution/systems perspective.
Includes hardware considerations and deployment tradeoffs.

---

## Citation audit

All arXiv IDs above were individually verified by fetching the corresponding
`arxiv.org/abs/` page. The following notes apply:

1. **Prompt Lookup Decoding** has no arXiv paper — it is a GitHub-only release by
   Saxena (2023). No arXiv link is provided.
2. **TurboSpec** (2406.14066): the paper was originally titled with "SmartSpec" in
   some references; the registered arXiv title is "TurboSpec: Closed-loop Speculation
   Control System for Optimizing LLM Serving Goodput".
3. **DeepSeek-V3 Technical Report** (2412.19437): section heading uses "DeepSeek-V3
   MTP Modules" as a nickname; registered title is "DeepSeek-V3 Technical Report".
4. **DOMINO** (2403.06988): section heading uses "DOMINO" as nickname; registered
   title is "Guiding LLMs The Right Way: Fast, Non-Invasive Constrained Generation".
5. **SEDD** (2310.16834): section heading uses "SEDD" as nickname; registered title is
   "Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution".
6. **LLaDA** (2502.09992): section heading uses "LLaDA" as nickname; registered title
   is "Large Language Diffusion Models".
7. **LLMA** (2304.04487): section heading uses "LLMA" as nickname; registered title is
   "Inference with Reference: Lossless Acceleration of Large Language Models".
