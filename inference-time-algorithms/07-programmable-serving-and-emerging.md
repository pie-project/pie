# Programmable Serving, Programming Models, and Emerging Inference-Time Algorithms

Where existing serving systems and language layers draw their expressiveness
boundary, and which emerging 2025-2026 algorithm families fundamentally need
engine-level control. Each entry notes the engine primitives required using the
eight axes defined in `00-pie-capability-map.md` §10.

---

## Core serving engines

### vLLM — PagedAttention

- **Title:** Efficient Memory Management for Large Language Model Serving with PagedAttention
- **arXiv:** [2309.06180](https://arxiv.org/abs/2309.06180)

Kwon et al., 2023, SOSP. Introduces PagedAttention, which manages KV cache in
non-contiguous virtual-memory-style pages, eliminating fragmentation and
enabling near-zero-waste memory sharing of prefixes across requests. Supports
host-side `LogitsProcessor` callbacks, engine-provided speculative decoding,
and automatic prefix caching. The guest-visible surface is a sampling-parameter
struct plus an optional per-step Python logit transform.

**Expressiveness boundary.** No access to hidden states or per-layer
activations (axis 1). No user-supplied attention mask (axis 2). KV sharing is
internal — no explicit fork/snapshot/rollback API (axis 3). LogitsProcessor
runs host-side with a Python round-trip per token (axis 4). No multi-context
logit combination in one pass (axis 5). Speculative decoding methods are
engine-provided, not user-written (axis 6). No guest-visible scheduling market
(axis 7). Tool calls are external; resumption relies on opaque prefix cache
(axis 8).

### SGLang — RadixAttention and structured generation frontend

- **Title:** SGLang: Efficient Execution of Structured Language Model Programs
- **arXiv:** [2312.07104](https://arxiv.org/abs/2312.07104)

Zheng et al., 2024, ICLR. Adds a frontend language for multi-call LLM programs
with `fork`, `join`, `select`, and `gen` primitives, backed by RadixAttention —
a radix-tree-indexed KV cache that automatically shares prefixes across calls.
Compressed finite-state machine constrained decoding runs fused with sampling.
Significantly higher throughput than vLLM for multi-turn programs.

**Expressiveness boundary.** Same as vLLM on axes 1, 2, 4, 5, 6. Frontend
`fork` exposes branching (partial axis 3), but the operation is language-level,
not a direct KV API — no snapshot, no explicit rollback, no page-level control.
No guest scheduling market (axis 7). No in-engine tool I/O (axis 8).

### Orca — iteration-level scheduling

Yu et al., 2022, OSDI. Introduced iteration-level (continuous) batching:
the scheduler reassigns the batch after every decode step, so finished requests
leave immediately and new arrivals join without waiting for a batch boundary.
Up to 36.9× throughput over FasterTransformer. Every subsequent system (vLLM,
SGLang, TensorRT-LLM) adopted this idea.

**Expressiveness boundary.** The scheduler is internal; the user submits a
request and receives tokens. No programmable hooks of any kind.

### DeepSpeed-FastGen

- **Title:** DeepSpeed-FastGen: High-throughput Text Generation for LLMs via MII and DeepSpeed-Inference
- **arXiv:** [2401.08671](https://arxiv.org/abs/2401.08671)

Holmes et al., 2024. Combines Dynamic SplitFuse (chunking long prompts and
fusing them with decode micro-batches) with continuous batching for high
throughput. Part of the DeepSpeed-Inference / MII stack.

**Expressiveness boundary.** Same as vLLM on all eight axes. The user
interface is a request with sampling parameters; the scheduling and fusion
logic is internal.

### Sarathi-Serve — chunked prefills

- **Title:** Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve
- **arXiv:** [2403.02310](https://arxiv.org/abs/2403.02310)

Agrawal et al., 2024, OSDI. Splits prefills into fixed-size chunks that
co-execute with ongoing decodes in a "stall-free" schedule, bounding
time-between-tokens and eliminating generation stalls. 2.6–6.9× serving
capacity at the same tail latency.

**Expressiveness boundary.** Infrastructure-level; no guest programmability
surface.

### DistServe — disaggregated prefill and decode

- **Title:** DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving
- **arXiv:** [2401.09670](https://arxiv.org/abs/2401.09670)

Zhong et al., 2024, OSDI. Runs prefill and decode on separate GPU pools,
optimizing each independently for its bottleneck (compute vs. memory
bandwidth). Up to 7.4× higher request rate or 12.6× tighter SLOs than
co-located designs. KV transfer overhead < 0.1% of total latency.

**Expressiveness boundary.** Cluster scheduling, not guest-programmable.

### Splitwise — phase splitting

- **Title:** Splitwise: Efficient generative LLM inference using phase splitting
- **arXiv:** [2311.18677](https://arxiv.org/abs/2311.18677)

Patel et al., 2024, ISCA. Same disaggregated-prefill-decode idea as DistServe,
targeting heterogeneous hardware. 1.4× throughput at 20% lower cost. Adopted
by production systems (Microsoft, Moonshot).

**Expressiveness boundary.** Cluster scheduling, not guest-programmable.

---

## Language and framework layers

### LMQL — prompting as programming

- **Title:** Prompting Is Programming: A Query Language for Large Language Models
- **arXiv:** [2212.06094](https://arxiv.org/abs/2212.06094)

Beurer-Kellner et al., 2023, PLDI. A query language that combines prompt
templates with Python control flow and typed constraints. Constraint
enforcement is host-side: LMQL masks logits via the API before each token,
paying a round-trip per step.

**Expressiveness boundary.** Constraints run host-side (axis 4). No access to
hidden states (axis 1), attention mask (axis 2), KV operations (axis 3), or
speculation (axis 6). Orchestration layer only.

### Outlines — efficient guided generation

- **Title:** Efficient Guided Generation for Large Language Models
- **arXiv:** [2307.09702](https://arxiv.org/abs/2307.09702)

Willard & Louf, 2023. Compiles regular expressions and context-free grammars
into finite-state machines indexing the tokenizer vocabulary, enabling
efficient constrained decoding with precomputed token masks. Widely adopted as
a backend in vLLM and other engines.

**Expressiveness boundary.** Logit masking only; does not touch hidden states,
KV, attention, or scheduling.

### XGrammar — structured generation engine

- **Title:** XGrammar: Flexible and Efficient Structured Generation Engine for Large Language Models
- **arXiv:** [2411.15100](https://arxiv.org/abs/2411.15100)

Dong et al., 2024. Separates context-independent tokens (pre-cacheable) from
context-dependent tokens (~1%), achieving up to 100× speedup over prior
grammar-based constrained decoding. Default backend for vLLM, SGLang,
TensorRT-LLM, and MLC-LLM.

**Expressiveness boundary.** Grammar enforcement only; a mask computation
library, not a serving programmability surface.

### Guidance — interleaved generation and control

Microsoft, 2023 (open-source library, no standalone arXiv paper). Lets users
interleave Python control flow with constrained LLM generation in a single
prompt program, enforcing JSON-schema, regex, and CFG constraints token by
token. Parsing engine rewritten in Rust (llguidance) for speed.

**Expressiveness boundary.** Host-side constraint enforcement via logit
masking. Same limits as LMQL: no hidden states, no KV control, no device-side
state.

### DSPy — compiling declarative LM programs

- **Title:** DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines
- **arXiv:** [2310.03714](https://arxiv.org/abs/2310.03714)

Khattab et al., 2024. Defines LM calls as typed *signatures*, compiles
multi-step pipelines with automatic prompt optimization (teleprompters), and
self-improves via bootstrapped demonstrations. Operates at the API level with
no engine hooks.

**Expressiveness boundary.** Pure orchestration layer. All eight axes are
opaque; DSPy treats the LM as a black-box function.

### Parrot — semantic variables for LLM applications

- **Title:** Parrot: Efficient Serving of LLM-based Applications with Semantic Variable
- **arXiv:** [2405.19888](https://arxiv.org/abs/2405.19888)

Lin et al., 2024, OSDI. Introduces *semantic variables* — typed placeholders
for LLM outputs that capture data and control dependencies across an
application's call graph. The serving engine (ParrotServe) sees the DAG and
co-optimizes scheduling, prefix sharing, and batching across dependent calls.

**Expressiveness boundary.** Application-aware scheduling but no per-step
programmability. The engine exploits inter-call structure (partial axis 7) but
the guest cannot write custom sampling, attention, or KV logic.

### Teola / Ayo — end-to-end LLM-application optimization

- **Title:** Teola: Towards End-to-End Optimization of LLM-based Applications
- **arXiv:** [2407.00326](https://arxiv.org/abs/2407.00326)

Tan et al., 2025, ASPLOS (published as "Ayo"). Decomposes LLM applications
into fine-grained *task primitives* and builds a primitive-level dataflow graph,
exposing cross-module parallelism and pipelining. Up to 2.09× speedup.

**Expressiveness boundary.** Orchestration-level optimization; no per-token or
per-step engine hooks.

---

## Serving for agents and multi-call workloads

### InferCept — intercept support for augmented LLM inference

- **Title:** InferCept: Efficient Intercept Support for Augmented Large Language Model Inference
- **arXiv:** [2402.01869](https://arxiv.org/abs/2402.01869)

Abhyankar et al., 2024. First serving system designed for "augmented LLMs" that
pause generation for tool calls. Applies a min-waste interception principle to
decide per-pause whether to retain, discard, or swap KV state. 1.6–2× higher
throughput than discarding context on every tool call.

**Engine primitive.** Directly addresses axis 8 (tool/agent I/O interleaved
with generation), but the policy is engine-internal — the guest cannot choose
the retain/discard strategy.

### Preble — distributed prompt scheduling

- **Title:** Preble: Efficient Distributed Prompt Scheduling for LLM Serving
- **arXiv:** [2407.00023](https://arxiv.org/abs/2407.00023)

2024. Lifts prefix caching from per-GPU to cluster-level: a hierarchical
scheduler clusters requests by shared prefix and co-locates them, maximizing KV
reuse. 1.5–14.5× lower average latency vs. single-GPU prefix caching.

**Engine primitive.** Prefix sharing (related to axis 3) but as an automatic
cluster policy, not a guest-exposed operation.

### Mooncake — KVCache-centric disaggregated architecture

- **Title:** Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving
- **arXiv:** [2407.00079](https://arxiv.org/abs/2407.00079)

Qin et al., 2024. Makes the KV cache a first-class disaggregated storage
layer, exploiting CPU/DRAM/SSD across the cluster. KV-centric scheduler
maximizes throughput under latency SLOs with a prediction-based early rejection
policy. Up to 525% throughput increase in simulated long-context settings;
powers the Kimi service in production.

**Engine primitive.** Cache infrastructure, not guest-programmable.

### Autellix — serving LLM agents as general programs

- **Title:** Autellix: An Efficient Serving Engine for LLM Agents as General Programs
- **arXiv:** [2502.13965](https://arxiv.org/abs/2502.13965)

Luo et al., 2025. Treats entire agentic programs (not individual calls) as the
scheduling unit. Intercepts all LLM calls from a program and augments the
scheduler with program-level context (wait times, dependencies), implementing
PLAS and ATLAS scheduling. 4–15× higher throughput at the same latency vs.
vLLM/SGLang.

**Engine primitive.** Program-aware scheduling (axis 7 from the engine side),
but the guest program is a black-box Python script — it cannot bid or influence
the policy directly.

### Continuum — multi-turn agent scheduling with KV cache TTL

- **Title:** Continuum: Efficient and Robust Multi-Turn LLM Agent Scheduling with KV Cache Time-to-Live
- **arXiv:** [2511.02230](https://arxiv.org/abs/2511.02230)

2025. Assigns per-request TTL-based KV retention policies to selectively pin
cache in GPU memory for agentic workflows, dramatically reducing recomputation
during tool-use pauses. 8× improvement in average job completion for SWE-Bench
agent workflows.

**Engine primitive.** Addresses axis 8 (tool/agent I/O without cache eviction),
but as an engine-internal policy.

---

## GPU-side programmability and Pie

### Pie — a programmable serving system

- **Title:** Pie: A Programmable Serving System for Emerging LLM Applications
- **arXiv:** [2510.24051](https://arxiv.org/abs/2510.24051)

Gim et al., 2025, SOSP. User-supplied WebAssembly programs (*inferlets*) run
next to the model with direct access to the forward pass via a guest-authored
device-resident tensor IR (PTIR). Intrinsics expose `logits`, `hidden`,
`query`, `value_head`, `layer`, `mtp_logits`. GPU-resident channels eliminate
host round-trips. Arbitrary custom attention masks, explicit O(1) KV fork /
snapshot / truncate, programmable constrained decoding, custom speculators,
inter-inferlet messaging, tool calling / MCP, and a credit/bid scheduling
market. Pie is the only system in this survey that exposes all eight axes to
the guest.

**Engine primitive.** All eight axes are guest-programmable: (1) more than
logits per step, (2) custom attention mask, (3) explicit KV
branching/backtracking, (4) per-token stateful logic without host round-trip,
(5) combining several distributions, (6) custom draft/verify rule, (7) its own
compute-allocation policy, (8) tool/agent I/O interleaved with generation.

---

## Expressiveness boundary summary

| Axis | vLLM | SGLang | LMQL/Guidance/DSPy | Parrot/Autellix | Pie |
|---|---|---|---|---|---|
| 1. More than logits | ✗ | ✗ | ✗ | ✗ | ✓ `hidden`/`query`/`layer`/`value_head` |
| 2. Custom attention mask | ✗ | ✗ | ✗ | ✗ | ✓ mask channel |
| 3. Explicit KV branch/backtrack | ✗ (internal) | partial (frontend `fork`) | ✗ | ✗ | ✓ `fork`/snapshot/truncate |
| 4. Device-side per-token state | ✗ | ✗ | ✗ | ✗ | ✓ channels |
| 5. Multi-distribution combine | ✗ | ✗ | ✗ | ✗ | ✓ heterogeneous frame |
| 6. Custom draft/verify | ✗ | ✗ | ✗ | ✗ | ✓ user speculators |
| 7. Guest compute-allocation | ✗ | ✗ | ✗ | engine-side only | ✓ credit/bid market |
| 8. Tool I/O without eviction | ✗ | ✗ | ✗ | engine policy | ✓ HTTP/MCP/launch in-engine |

---

## Diffusion / any-order LLMs and their decoding algorithms

### LLaDA — large language diffusion models

- **Title:** Large Language Diffusion Models
- **arXiv:** [2502.09992](https://arxiv.org/abs/2502.09992)

Nie et al., 2025. Treats language modelling as a masked diffusion process:
the forward process randomly masks tokens, and a reverse process reconstructs
all masked tokens simultaneously using bidirectional context. Scaled to 8B
parameters, competitive with LLaMA 3 8B. Demonstrates that key LLM abilities
(ICL, instruction following) do not depend on autoregressive left-to-right
generation.

**Engine primitive.** Decoding requires arbitrary attention masks (axis 2) for
remasking schedules and non-left-to-right readout positions. A black-box
autoregressive server cannot execute these decoding schedules at all.

### MDLM — masked discrete diffusion

- **Title:** Simple and Effective Masked Diffusion Language Models
- **arXiv:** [2406.07524](https://arxiv.org/abs/2406.07524)

Sahoo et al., 2024. Shows that masked diffusion with a simplified
Rao-Blackwellized objective achieves SOTA among diffusion LMs, closing the gap
with autoregressive perplexity. Supports both ancestral and semi-autoregressive
generation.

**Engine primitive.** Same as LLaDA: arbitrary masks (axis 2), parallel
unmasking readout.

### Generalized masked diffusion

- **Title:** Simplified and Generalized Masked Diffusion for Discrete Data
- **arXiv:** [2406.04329](https://arxiv.org/abs/2406.04329)

Shi et al., 2024, NeurIPS. Unifies masked diffusion approaches under a
continuous-time variational objective equivalent to weighted cross-entropy
losses. Introduces state-dependent masking schedules, achieving SOTA on both
image and language benchmarks.

**Engine primitive.** Arbitrary masks (axis 2), custom masking schedules.

### Dream — diffusion large language models

- **Title:** Dream 7B: Diffusion Large Language Models
- **arXiv:** [2508.15487](https://arxiv.org/abs/2508.15487)

2025. Scales discrete diffusion to 7B via AR-based initialization and
context-adaptive token-level noise rescheduling. Outperforms prior diffusion
LMs and matches AR LLMs on general language, math, and code. Supports
arbitrary-order generation, sequence infilling, and tunable quality-speed
tradeoffs.

**Engine primitive.** Arbitrary attention masks (axis 2), non-autoregressive
readout. Confidence-based decoding schedules may benefit from device-side
stateful logic (axis 4).

### Mercury — ultra-fast diffusion LMs

- **Title:** Mercury: Ultra-Fast Language Models Based on Diffusion
- **arXiv:** [2506.17298](https://arxiv.org/abs/2506.17298)

Inception Labs, 2025. First commercial-grade diffusion LLM. Generates text by
denoising blocks of tokens in parallel, achieving 1,109 tokens/sec on a single
H100 — up to 10× faster than speed-optimized autoregressive models while
maintaining comparable code quality.

**Engine primitive.** Block-parallel denoising requires custom attention masks
(axis 2) and non-standard readout patterns incompatible with left-to-right
serving assumptions.

### Diffusion Forcing — bridging next-token and full-sequence diffusion

- **Title:** Diffusion Forcing: Next-token Prediction Meets Full-Sequence Diffusion
- **arXiv:** [2407.01392](https://arxiv.org/abs/2407.01392)

Chen et al., 2024. Trains a causal model to denoise tokens with independent
per-token noise levels, unifying next-token prediction with full-sequence
diffusion. Enables variable-horizon, guided generation and stable long-sequence
rollout. Applicable to planning, video, and language.

**Engine primitive.** Independent per-token noise levels require per-position
mask control (axis 2) and on-device state for the noise schedule (axis 4).

### Self speculative decoding for diffusion LLMs

- **Title:** Self Speculative Decoding for Diffusion Large Language Models
- **arXiv:** [2510.04147](https://arxiv.org/abs/2510.04147)

2025. A lossless acceleration method that uses the diffusion LLM itself to
draft and verify multi-token completions in one forward pass — no auxiliary
model needed. Up to 3.46× speedup on LLaDA and Dream while matching stepwise
output.

**Engine primitive.** Requires a custom draft/verify rule (axis 6) and
arbitrary attention masks (axis 2). Cannot be expressed through a standard
speculative decoding API.

---

## Latent and continuous-space reasoning at inference

### Coconut — chain of continuous thought

- **Title:** Training Large Language Models to Reason in a Continuous Latent Space
- **arXiv:** [2412.06769](https://arxiv.org/abs/2412.06769)

Hao et al., 2024, Meta FAIR. Instead of generating chain-of-thought tokens,
the model reasons by feeding the last hidden state directly back as the next
input embedding, iterating in continuous latent space. Encodes breadth-first
search over alternatives. Outperforms language-based CoT on tasks requiring
substantial search.

**Engine primitive.** Requires reading the hidden state (axis 1) and feeding
it back as input without decoding to tokens — fundamentally impossible on a
black-box text-in/text-out server. Also needs per-step stateful logic (axis 4)
for the latent iteration loop.

### Pause tokens — extra compute before committing

- **Title:** Think before you speak: Training Language Models With Pause Tokens
- **arXiv:** [2310.02226](https://arxiv.org/abs/2310.02226)

Goyal et al., 2024, ICLR. Appends learnable dummy "pause" tokens to the input,
giving the model extra transformer steps before producing the next real token.
+18% EM on SQuAD, +8% on CommonsenseQA for a 1B model. Requires pause tokens
in both pretraining and finetuning.

**Engine primitive.** Serving pause tokens requires custom attention masks
(axis 2) to hide pause outputs and non-standard readout (only the token after
the last pause matters). A standard serving API can approximate this with
prompt padding, but the correct mask requires engine support.

### Quiet-STaR — self-taught internal reasoning

- **Title:** Quiet-STaR: Language Models Can Teach Themselves to Think Before Speaking
- **arXiv:** [2403.09629](https://arxiv.org/abs/2403.09629)

Zelikman et al., 2024. Generalizes STaR to arbitrary text: the model generates
internal rationales ("thinking tokens") at every position using parallel
sampling, trains to reinforce rationales that improve next-token prediction.
Improvements on GSM8K and CommonsenseQA without task-specific finetuning.

**Engine primitive.** Parallel internal-thought generation requires
custom attention masks (axis 2) for thought boundaries and per-token
branching state (axis 4). Production serving would also need thought tokens
hidden from the output stream.

### Recurrent-depth transformers — latent test-time compute

- **Title:** Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach
- **arXiv:** [2502.05171](https://arxiv.org/abs/2502.05171)

Geiping et al., 2025. Re-applies a single transformer block multiple times at
inference, decoupling effective depth from parameter count. Adaptive compute:
harder examples get more iterations. Trained with randomized unrolling. The
Huginn 3.5B model demonstrates competitive reasoning with much larger
non-looped models.

**Engine primitive.** Requires reading the hidden state after each loop
iteration and feeding it back (axis 1), plus an adaptive halting criterion
as per-token device-side state (axis 4). The serving engine must support
variable-depth forward passes — not expressible as a fixed pipeline.

### Looped transformers — theoretical foundations

- **Title:** Reasoning with Latent Thoughts: On the Power of Looped Transformers
- **arXiv:** [2502.17416](https://arxiv.org/abs/2502.17416)

Saunshi et al., 2025. Shows theoretically that a shallow model looped multiple
times can match the representational power of a deep unlooped model on many
reasoning tasks. Effective depth via loops matters more than parameter count for
algorithmic and multi-hop tasks.

**Engine primitive.** Same as recurrent-depth: axes 1 and 4 for latent-space
iteration with adaptive halting.

---

## Adaptive compute allocation at inference

### Compute-optimal test-time scaling

- **Title:** Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters
- **arXiv:** [2408.03314](https://arxiv.org/abs/2408.03314)

Snell et al., 2024. Shows that adaptively allocating more test-time compute to
harder prompts — via verifier reward models or adaptive answer distributions —
yields > 4× efficiency over best-of-N. A smaller model with compute-optimal
test-time scaling can outperform a 14× larger model. The key insight: the
compute policy must vary per prompt, not per model.

**Engine primitive.** Implementing a compute-optimal policy requires a
guest-visible compute-allocation mechanism (axis 7). On a fixed-budget server,
the client cannot bid for more compute on harder queries.

### s1 — budget forcing

- **Title:** s1: Simple test-time scaling
- **arXiv:** [2501.19393](https://arxiv.org/abs/2501.19393)

Muennighoff et al., 2025. Finetunes on 1,000 reasoning traces, then applies
*budget forcing* at inference: appending "Wait" tokens when the model tries to
terminate early, or truncating when the budget is spent. s1-32B exceeds
o1-preview on competition math by up to 27%.

**Engine primitive.** Budget forcing needs the serving engine to inject or
suppress tokens based on a per-request policy (axis 7) and to track reasoning
state across steps (axis 4). On a black-box API this requires fragile prompt
manipulation.

### RouteLLM — difficulty-aware routing

- **Title:** RouteLLM: Learning to Route LLMs with Preference Data
- **arXiv:** [2406.18665](https://arxiv.org/abs/2406.18665)

Ong et al., 2024, ICLR 2025. A lightweight router trained on human preference
data dynamically selects between a strong and a weak LLM per query. > 2×
cost reduction with ~95% of GPT-4 quality. Routing is external to the
serving engine.

**Engine primitive.** Routing itself is a client-side policy, but
*compute-optimal* routing that also controls speculation depth and branch
budgets per query requires axis 7 (guest compute-allocation policy).

### LayerSkip — early exit and self-speculative decoding

- **Title:** LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding
- **arXiv:** [2404.16710](https://arxiv.org/abs/2404.16710)

Elhoushi et al., 2024, Meta. Trains with increasing layer dropout so early
layers make strong predictions, then uses early layers for draft and later
layers for verify — self-speculative decoding with no separate draft model.
Up to 2.16× speedup.

**Engine primitive.** Requires per-layer readout (axis 1 via `layer()`) and a
custom draft/verify rule (axis 6) where the draft and verify share the same
model at different depths.

### Dynamic early exit in reasoning models

- **Title:** Dynamic Early Exit in Reasoning Models
- **arXiv:** [2504.15895](https://arxiv.org/abs/2504.15895)

2025. Monitors reasoning chains for "thought switch" signals and terminates
generation when confidence is high, reducing chain length by 30-40% while
sometimes improving accuracy. Addresses the inefficiency of fixed-length
reasoning in slow-thinking models.

**Engine primitive.** Early exit based on hidden-state confidence requires
axis 1 (access to hidden states) and axis 7 (compute-allocation policy to
actually terminate early rather than generate to a fixed budget).

---

## Long-horizon agent inference

### Mixture-of-Agents

- **Title:** Mixture-of-Agents Enhances Large Language Model Capabilities
- **arXiv:** [2406.04692](https://arxiv.org/abs/2406.04692)

Wang et al., 2024. A layered architecture where multiple LLM agents each
receive outputs from all agents in the previous layer, iteratively refining
responses. Open-source MoA outperforms GPT-4o on AlpacaEval 2.0 (65.1% vs.
57.5%). Inference-time only, no retraining.

**Engine primitive.** Multi-agent layers require multiple concurrent contexts
(axis 5) and inter-agent messaging. Running this inside the serving system
(axis 8) avoids evicting each agent's KV cache between rounds. Pie's
inter-inferlet launch and messaging are a direct fit.

### ACON — context compression for long-horizon agents

- **Title:** ACON: Optimizing Context Compression for Long-horizon LLM Agents
- **arXiv:** [2510.00615](https://arxiv.org/abs/2510.00615)

Kang et al., 2025, Microsoft. Compresses both observation and history for LLM
agents using iteratively refined compression guidelines, cutting peak token
usage by 26–54%. Smaller LMs with ACON see up to 46% performance boost by
reducing context distraction.

**Engine primitive.** Context compaction as an inference-time algorithm needs
explicit KV truncation and re-prefill (axis 3) plus tool/agent I/O that
preserves cache locality (axis 8). On a black-box server, compression is a
client-side prompt rewrite with no KV-level control.

### LLMLingua — prompt compression

- **Title:** LLMLingua: Compressing Prompts for Accelerated Inference of Large Language Models
- **arXiv:** [2310.05736](https://arxiv.org/abs/2310.05736)

Jiang et al., 2023, EMNLP. Coarse-to-fine compression using a budget
controller and token-level iterative compression, achieving up to 20×
compression with negligible performance loss. Extended by LongLLMLingua (2024)
for long-context settings.

**Engine primitive.** Currently runs as a client-side prompt rewrite. Making
it engine-internal — compacting KV pages rather than rewriting text — would
require axis 3 (explicit KV truncation) and axis 1 (hidden-state-driven
importance scoring).

### STILL-2 — slow-thinking reasoning with self-improvement

- **Title:** Imitate, Explore, and Self-Improve: A Reproduction Report on Slow-thinking Reasoning Systems
- **arXiv:** [2412.09413](https://arxiv.org/abs/2412.09413)

2024. Open reproduction of o1-style slow-thinking: finetune on long reasoning
traces, explore multiple rollouts, self-improve from high-quality trajectories.
Competitive with industry solutions on Olympiad-level math.

**Engine primitive.** Multiple rollouts from a shared prefix need O(1) KV fork
(axis 3). Adaptive compute budgets per rollout need axis 7.
Confidence-based early stopping across rollouts needs hidden-state access
(axis 1) or at minimum device-side entropy tracking (axis 4).

---

## Citation audit

All 39 arXiv citations were verified by fetching the corresponding
`arxiv.org/abs/` page and confirming the registered title. The following
section-heading nicknames deliberately differ from the registered arXiv title:

| Heading nickname | Registered arXiv title | arXiv ID |
|---|---|---|
| vLLM — PagedAttention | Efficient Memory Management for Large Language Model Serving with PagedAttention | 2309.06180 |
| SGLang — RadixAttention and structured generation frontend | SGLang: Efficient Execution of Structured Language Model Programs | 2312.07104 |
| LMQL — prompting as programming | Prompting Is Programming: A Query Language for Large Language Models | 2212.06094 |
| Generalized masked diffusion | Simplified and Generalized Masked Diffusion for Discrete Data | 2406.04329 |
| LLaDA — large language diffusion models | Large Language Diffusion Models | 2502.09992 |
| MDLM — masked discrete diffusion | Simple and Effective Masked Diffusion Language Models | 2406.07524 |
| Dream — diffusion large language models | Dream 7B: Diffusion Large Language Models | 2508.15487 |
| Mercury — ultra-fast diffusion LMs | Mercury: Ultra-Fast Language Models Based on Diffusion | 2506.17298 |
| Coconut — chain of continuous thought | Training Large Language Models to Reason in a Continuous Latent Space | 2412.06769 |
| Compute-optimal test-time scaling | Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters | 2408.03314 |
| s1 — budget forcing | s1: Simple test-time scaling | 2501.19393 |
| Looped transformers — theoretical foundations | Reasoning with Latent Thoughts: On the Power of Looped Transformers | 2502.17416 |
| STILL-2 — slow-thinking reasoning with self-improvement | Imitate, Explore, and Self-Improve: A Reproduction Report on Slow-thinking Reasoning Systems | 2412.09413 |
| Teola / Ayo — end-to-end LLM-application optimization | Teola: Towards End-to-End Optimization of LLM-based Applications | 2407.00326 |

**Systems without arXiv papers** (mentioned inline, no verifiable ID):
Orca (Yu et al., OSDI 2022), TensorRT-LLM (NVIDIA, commercial),
Guidance (Microsoft, open-source library), LangChain (orchestration framework).
