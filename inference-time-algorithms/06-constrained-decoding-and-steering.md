# Constrained decoding, controlled generation, and activation steering

Grammar/JSON/regex constrained decoding and its distortion problem; semantic and
soft-control methods (FUDGE, DExperts, classifier-free guidance); activation
steering, ITI/CAA/RepE, SAE-based feature steering, concept erasure; layer-level
interventions (DoLa, early exit); uncertainty and hallucination control at decode
time.

The eight expressiveness axes referenced throughout are defined in
`00-pie-capability-map.md` §10.

---

## A. Constrained / structured / controlled generation

### A.1 Grammar-constrained and structured decoding

#### PICARD — Parsing Incrementally for Constrained Auto-Regressive Decoding

- **Title:** PICARD: Parsing Incrementally for Constrained Auto-Regressive Decoding from Language Models
- **arXiv:** [2109.05093](https://arxiv.org/abs/2109.05093)

Scholak et al., 2021, EMNLP 2021. Runs an incremental parser alongside beam search to reject tokens that would make the partial output unparseable (applied to SQL). The parser state advances token-by-token; invalid continuations are masked from the beam.

**Engine primitive:** Per-token logit mask from an incremental parser (**axis 4: per-token stateful logic**). The parser state is a host-side round-trip per step on black-box servers; Pie's grammar engine + late masking moves it on-device.

---

#### Synchromesh — Reliable Code Generation from Pre-trained Language Models

- **Title:** Synchromesh: Reliable code generation from pre-trained language models
- **arXiv:** [2201.11227](https://arxiv.org/abs/2201.11227)

Poesia et al., 2022, ICLR 2022. Introduces Target-Constrained Decoding (TCD): a completion engine provides the set of valid next tokens given the partial program, and beam search is restricted to that set. Also contributes Completion-Augmented Prompting.

**Engine primitive:** Per-token logit mask driven by a completion engine (**axis 4**). Identical structural needs to PICARD.

---

#### LMQL — Prompting Is Programming

- **Title:** Prompting Is Programming: A Query Language for Large Language Models
- **arXiv:** [2212.06094](https://arxiv.org/abs/2212.06094)

Beurer-Kellner et al., 2023, PLDI 2023. A query language that compiles high-level type/format/length constraints into token-level masks applied during decoding. Supports scripted control flow, branching, and nested constraints around generation calls.

**Engine primitive:** Per-token mask + host-side control flow (**axis 4**). Benefits from **axis 3 (KV branching)** for its scripted branches.

---

#### GCD — Grammar-Constrained Decoding for Structured NLP Tasks

- **Title:** Grammar-Constrained Decoding for Structured NLP Tasks without Finetuning
- **arXiv:** [2305.13971](https://arxiv.org/abs/2305.13971)

Geng et al., 2023, NeurIPS 2023. Formalises grammar-constrained decoding for arbitrary CFGs: at each step the incremental parser computes the set of grammar-valid token continuations and masks out the rest. Demonstrates broad applicability to code, semantic parsing, and math.

**Engine primitive:** Per-token logit mask from incremental CFG parser (**axis 4**). Pie-native via `grammar.wit` + `mask_apply`.

---

#### Outlines — Efficient Guided Generation for LLMs

- **Title:** Efficient Guided Generation for Large Language Models
- **arXiv:** [2307.09702](https://arxiv.org/abs/2307.09702)

Willard & Louf, 2023. Compiles a regex into a finite-state machine (FSM) whose states are indexed by token: at each step the FSM state determines the valid token set, and a bitmask is applied. Faster than re-parsing at every step because the FSM transitions are pre-computed.

**Engine primitive:** Per-token logit mask from pre-compiled FSM (**axis 4**). A host-side logit processor on vLLM/SGLang; Pie can apply the mask on-device via late masking.

---

#### SGLang — Efficient Execution of Structured Language Model Programs

- **Title:** SGLang: Efficient Execution of Structured Language Model Programs
- **arXiv:** [2312.07104](https://arxiv.org/abs/2312.07104)

Zheng et al., 2023. Introduces RadixAttention for automatic KV prefix sharing and a compressed FSM for constrained decoding that co-executes with the engine's continuous batching. The FSM state transitions are pushed into the scheduler to avoid per-token round-trips.

**Engine primitive:** Per-token logit mask + prefix cache reuse (**axes 4, 3**). SGLang provides prefix sharing but not explicit fork/snapshot; Pie exposes both plus guest-composable mask intersection.

---

#### Automata-Based Constraints for Language Model Decoding

- **Title:** Automata-based constraints for language model decoding
- **arXiv:** [2407.08103](https://arxiv.org/abs/2407.08103)

2024. Unifies regex, CFG, and pushdown-automaton constraints into a single automata-theoretic framework for constrained decoding. Provides formal correctness guarantees for the incremental token-mask computation across different automaton classes.

**Engine primitive:** Per-token logit mask (**axis 4**). The framework generalises the mask-computation primitive already present in Pie's grammar engine.

---

#### XGrammar — Flexible and Efficient Structured Generation Engine

- **Title:** XGrammar: Flexible and Efficient Structured Generation Engine for Large Language Models
- **arXiv:** [2411.15100](https://arxiv.org/abs/2411.15100)

Dong et al., 2024. A high-performance grammar engine that pre-computes context-independent token sets at compile time and handles context-dependent tokens via an adaptive expansion algorithm. Achieves up to 100× speedup over Outlines. Pie's grammar engine is a Rust rewrite derived from XGrammar.

**Engine primitive:** Per-token logit mask with pre-compiled context-independent sets (**axis 4**). This is exactly the primitive Pie's `runtime/grammar/` implements natively with late masking composed with speculation.

---

#### DOMINO — Guiding LLMs The Right Way

- **Title:** Guiding LLMs The Right Way: Fast, Non-Invasive Constrained Generation
- **arXiv:** [2403.06988](https://arxiv.org/abs/2403.06988)

Beurer-Kellner et al., 2024, ICML 2024. Addresses the **token-alignment problem**: subword tokenisation interacts poorly with character-level constraints, causing grammar masks to mis-handle prefix-splitting. DOMINO introduces minimally invasive constrained decoding with token healing to avoid degradation.

**Engine primitive:** Per-token logit mask with tokeniser-alignment correction (**axis 4**). The token-healing step requires look-back into recently emitted tokens — benefits from **axis 4 (device-side state)** to avoid extra round-trips.

---

#### Flexible and Efficient Grammar-Constrained Decoding

- **Title:** Flexible and Efficient Grammar-Constrained Decoding
- **arXiv:** [2502.05111](https://arxiv.org/abs/2502.05111)

2025. Proposes a more flexible GCD framework that separates the constraint specification from the mask computation, allowing mixed constraint types (regex + CFG + semantic) to be composed. Improves efficiency via lazy evaluation of grammar states.

**Engine primitive:** Per-token composable masks (**axis 4**). Directly matches Pie's `and_into` mask composition primitive for intersecting independent constraints.

---

#### JSONSchemaBench — Structured Outputs for Language Models

- **Title:** JSONSchemaBench: A Rigorous Benchmark of Structured Outputs for Language Models
- **arXiv:** [2501.10868](https://arxiv.org/abs/2501.10868)

Geng et al., 2025. A benchmark of 10,000+ real-world JSON schemas testing constrained decoding frameworks (Guidance, Outlines, llama.cpp, XGrammar, OpenAI). Finds that prompt-only approaches fail 5–15% of outputs; grammar-constrained decoding achieves 100% schema compliance. Quantifies coverage gaps across engines.

**Engine primitive:** JSON-schema constrained decoding (**axis 4**). Pie ships `json-schema-constrained-decoding` as a native inferlet.

---

### A.2 Correctness and distortion in constrained decoding

#### ASAp — Grammar-Aligned Decoding

- **Title:** Grammar-Aligned Decoding
- **arXiv:** [2405.21047](https://arxiv.org/abs/2405.21047)

Park et al., 2024. Shows that hard-masking provably **distorts** the model's distribution: the output is grammatical but its likelihoods deviate from what the unconstrained model would assign, conditional on validity. Proposes Adaptive Sampling with Approximate Expected Futures (ASAp) to correct via resampling when a constraint dead-ends.

**Engine primitive:** Constraint state + KV truncation + resampling (**axes 3, 4**). Needs rollback after dead-ends — Pie's `truncate` + `snapshot` + grammar matcher provide exactly this. A black-box server cannot backtrack KV state at all.

---

#### Let Me Speak Freely — Impact of Format Restrictions

- **Title:** Let Me Speak Freely? A Study on the Impact of Format Restrictions on Performance of Large Language Models
- **arXiv:** [2408.02442](https://arxiv.org/abs/2408.02442)

Tam et al., 2024, EMNLP 2024. Empirically demonstrates that enforcing structured formats (JSON, XML) via constrained decoding degrades reasoning performance, with stricter formats causing greater loss. The NL-to-format two-pass approach mitigates the degradation.

**Engine primitive:** The two-pass mitigation benefits from **axis 3 (KV fork)** — generate reasoning in free-form, then fork and constrain only the formatting pass.

---

#### AdapTrack — Constrained Decoding without Distorting Output Intent

- **Title:** AdapTrack: Constrained Decoding without Distorting LLM's Output Intent
- **arXiv:** [2510.17376](https://arxiv.org/abs/2510.17376)

2025. Addresses the distortion problem with adaptive backtracking: when masking pushes the model too far from its intended distribution, the decoder backtracks and retries. Achieves constraint compliance without semantic drift.

**Engine primitive:** Per-token constraint state + backtracking (**axes 3, 4**). Backtracking requires KV truncation — Pie-native via `truncate`; fundamentally blocked on servers without rollback.

---

#### CDSL — Constrained Decoding with Speculative Lookaheads

- **Title:** Constrained Decoding with Speculative Lookaheads
- **arXiv:** [2412.10418](https://arxiv.org/abs/2412.10418)

2024. Composes constrained decoding with speculative decoding: a small draft model proposes grammar-compliant continuations verified by the target model. Achieves 2–12× speedup over standard constrained decoding while preserving output quality.

**Engine primitive:** Constrained mask + custom draft/verify rule (**axes 4, 6**). Pie ships `mtp-grammar` demonstrating exactly this composition of speculation with grammar constraints in a single pass.

---

### A.3 Semantic and soft control (non-mask methods)

#### PPLM — Plug and Play Language Models

- **Title:** Plug and Play Language Models: A Simple Approach to Controlled Text Generation
- **arXiv:** [1912.02164](https://arxiv.org/abs/1912.02164)

Dathathri et al., 2020, ICLR 2020. Steers generation by computing gradients of a small attribute classifier with respect to the LM's hidden activations, then perturbing the activations at each step. No fine-tuning of the LM.

**Engine primitive:** Gradient access w.r.t. hidden activations at each step (**axis 1: more than logits**). Requires **writing** perturbed activations back into the forward pass — needs a per-layer write port not yet in Pie's ABI. Fundamentally blocked on black-box servers.

---

#### NeuroLogic Decoding

- **Title:** NeuroLogic Decoding: (Un)supervised Neural Text Generation with Predicate Logic Constraints
- **arXiv:** [2010.12884](https://arxiv.org/abs/2010.12884)

Lu et al., 2021, NAACL 2021. Encodes lexical constraints as predicate-logic formulas and enforces them during beam search via a satisfaction score added to the beam score. Tokens that advance constraint satisfaction are rewarded.

**Engine primitive:** Modified beam scoring with constraint satisfaction (**axis 4**). Benefits from **axis 3 (KV branching)** for efficient beam management.

---

#### NeuroLogic A*esque Decoding

- **Title:** NeuroLogic A*esque Decoding: Constrained Text Generation with Lookahead Heuristics
- **arXiv:** [2112.08726](https://arxiv.org/abs/2112.08726)

Lu et al., 2022, NAACL 2022. Extends NeuroLogic with A*-style lookahead: a future-cost heuristic estimates the probability of satisfying remaining constraints from each partial hypothesis, enabling better beam allocation.

**Engine primitive:** Lookahead heuristic requires speculative forward passes per beam (**axes 3, 4, 7**). Benefits from KV fork for lookahead branches and compute-allocation policy for beam budget.

---

#### GeDi — Generative Discriminator Guided Sequence Generation

- **Title:** GeDi: Generative Discriminator Guided Sequence Generation
- **arXiv:** [2009.06367](https://arxiv.org/abs/2009.06367)

Krause et al., 2021, EMNLP Findings 2021. Uses a small generative model as a Bayesian discriminator: at each step the guide model's class-conditional probabilities re-weight the base model's logits. The combination biases generation toward a target attribute.

**Engine primitive:** Two forward contexts combined per step (**axis 5: combining distributions**). Pie can bind both into one frame and combine in PTIR; elsewhere requires two servers and a host loop.

---

#### FUDGE — Future Discriminators for Controlled Text Generation

- **Title:** FUDGE: Controlled Text Generation With Future Discriminators
- **arXiv:** [2104.05218](https://arxiv.org/abs/2104.05218)

Yang & Klein, 2021, NAACL 2021. Trains a lightweight future-discriminator that predicts, for each candidate next token, the probability that the completed sequence will satisfy a constraint. The discriminator's scores multiply the base model's probabilities at decode time.

**Engine primitive:** Auxiliary classifier logits combined with base logits (**axis 5**). The discriminator is a separate model whose outputs are combined per step.

---

#### DExperts — Decoding-Time Controlled Text Generation

- **Title:** DExperts: Decoding-Time Controlled Text Generation with Experts and Anti-Experts
- **arXiv:** [2105.03023](https://arxiv.org/abs/2105.03023)

Liu et al., 2021, ACL 2021. Combines an expert LM (fine-tuned on desired text), an anti-expert (fine-tuned on undesired text), and the base LM via logit arithmetic: `logits = base + α(expert − anti_expert)`. No base model modification needed.

**Engine primitive:** Three forward contexts combined per step (**axis 5**). Pie can bind all three into one frame with PTIR logit arithmetic. On a black-box server: three separate inference calls plus host-side combination.

---

#### Contrastive Decoding — Open-ended Text Generation as Optimization

- **Title:** Contrastive Decoding: Open-ended Text Generation as Optimization
- **arXiv:** [2210.15097](https://arxiv.org/abs/2210.15097)

Li et al., 2022, ACL 2023. Contrasts an expert (large) and amateur (small) model: tokens strongly preferred by the expert over the amateur are upweighted. Reduces degeneration without sacrificing diversity.

**Engine primitive:** Two model distributions combined per step (**axis 5**). The existing Pie `contrastive-decoding` inferlet uses same-model two-context contrast; the original requires expert + amateur models in one frame.

---

#### Classifier-Free Guidance for LLMs

- **Title:** Stay on topic with Classifier-Free Guidance
- **arXiv:** [2306.17806](https://arxiv.org/abs/2306.17806)

Sanchez et al., 2023. Adapts classifier-free guidance from diffusion models to autoregressive LLMs: `logits = uncond + γ(cond − uncond)` where cond and uncond are the prompted and empty-prompt forward passes. Amplifies the effect of the prompt.

**Engine primitive:** Two forward contexts (prompted, unprompted) combined per step (**axis 5**). Pie can bind both into one frame and combine in PTIR.

---

#### Context-Aware Decoding (CAD) — Hallucinate Less

- **Title:** Trusting Your Evidence: Hallucinate Less with Context-aware Decoding
- **arXiv:** [2305.14739](https://arxiv.org/abs/2305.14739)

Shi et al., 2023. Contrasts with-context vs. without-context logits to amplify the contribution of the retrieved context, reducing hallucination. The anti-hallucination specialisation of classifier-free guidance.

**Engine primitive:** Two forward contexts per step (**axis 5**). Same frame pattern as CFG — Pie-native.

---

#### IPA — Inference-Time Policy Adapters

- **Title:** Inference-Time Policy Adapters: Tailoring Extreme-Scale LMs without Fine-tuning
- **arXiv:** [2305.15065](https://arxiv.org/abs/2305.15065)

Lu et al., 2023, EMNLP 2023. Trains a small policy adapter whose logits are added to the frozen base LM at decode time, steering generation toward a reward without fine-tuning the base. The adapter acts as a lightweight expert whose signal is combined per step.

**Engine primitive:** Two distributions combined per step (**axis 5**). The adapter is a second model run in the same frame.

---

#### Controlled Decoding from Language Models

- **Title:** Controlled Decoding from Language Models
- **arXiv:** [2310.17022](https://arxiv.org/abs/2310.17022)

Mudgal et al., 2024, ICML 2024. Trains a prefix scorer (value function) that estimates future reward at each decoding step. The value function's gradient w.r.t. the token distribution re-weights the base model's logits for KL-constrained controlled generation.

**Engine primitive:** Value function evaluation per step (**axes 1, 5**). Benefits from `value_head()` if the scorer is co-located in the model; otherwise a second forward context (**axis 5**).

---

#### Proxy Tuning — Tuning Language Models by Proxy

- **Title:** Tuning Language Models by Proxy
- **arXiv:** [2401.08565](https://arxiv.org/abs/2401.08565)

Liu et al., 2024, COLM 2024. Tunes a small model (expert) and uses the logit difference between tuned and untuned small models as an offset applied to the untunable large model's logits at decode time. Closes 88% of the fine-tuning gap on Llama2-70B.

**Engine primitive:** Three distributions combined per step (**axis 5**). Same logit-arithmetic pattern as DExperts.

---

#### Emulated Fine-Tuning (EFT)

- **Title:** An Emulator for Fine-Tuning Large Language Models using Small Language Models
- **arXiv:** [2310.12962](https://arxiv.org/abs/2310.12962)

Mitchell et al., 2023, ICLR 2024. Decomposes the tuning effect into a pre-training component (large model) and a fine-tuning component (small model), then ensembles them at decode time via logit arithmetic. Enables test-time adjustment of behavioural traits.

**Engine primitive:** Two or more distributions combined per step (**axis 5**). Pie-native via multi-context frames.

---

#### RAIN — Language Models Can Align Themselves without Finetuning

- **Title:** RAIN: Your Language Models Can Align Themselves without Finetuning
- **arXiv:** [2309.07124](https://arxiv.org/abs/2309.07124)

Li et al., 2023, NeurIPS 2023. At each step, the model generates candidate continuations, self-evaluates them for alignment, then backtracks and regenerates from the best branch. Achieves alignment without any external reward model.

**Engine primitive:** Branching + backtracking + self-evaluation (**axes 3, 4, 7**). Requires KV fork for branching, truncate for backtracking, and compute-allocation for branch budgets.

---

#### Diffusion of Thoughts — Chain-of-Thought in Diffusion Language Models

- **Title:** Diffusion of Thoughts: Chain-of-Thought Reasoning in Diffusion Language Models
- **arXiv:** [2402.07754](https://arxiv.org/abs/2402.07754)

Ye et al., 2024. Applies diffusion-model generation to chain-of-thought reasoning: the entire reasoning trace is generated via iterative denoising with a non-left-to-right schedule. Remasking and parallel unmasking replace autoregressive token emission.

**Engine primitive:** Arbitrary attention mask + non-left-to-right readout (**axis 2: custom attention mask**). Needs the mask channel and `readout()` indices, both Pie first-class primitives.

---

### A.4 Structured output at scale and composition

Structured output in production (JSON-mode, function calling, tool-call schema
enforcement) is the industrial application of the grammar-constrained decoding
stack described in §A.1. The key engineering challenges are:

1. **Composing constraints with speculation** — the constraint mask must be
   applied inside the draft/verify loop, not after it. Pie ships `mtp-grammar`
   demonstrating this composition (**axes 4, 6**).
2. **Composing constraints with reasoning traces** — chain-of-thought followed
   by a constrained formatting pass. Requires free-form generation, then
   fork/snapshot, then constrained regeneration of the structured suffix
   (**axes 3, 4**).
3. **Multi-constraint intersection** — Pie's `and_into` composes independent
   grammar masks into a single pass rather than serialising them.

Cross-reference: `04-speculative-and-efficiency.md` covers speculation mechanics;
`07-programmable-serving-and-emerging.md` covers serving-system integration.

---

## B. Activation- and representation-level inference-time interventions

### B.1 Activation steering and representation engineering

#### Extracting Latent Steering Vectors

- **Title:** Extracting Latent Steering Vectors from Pretrained Language Models
- **arXiv:** [2205.05124](https://arxiv.org/abs/2205.05124)

Subramani et al., 2022, ACL Findings 2022. Discovers that pretrained LMs contain latent steering vectors: directions in activation space that, when added to the hidden state, reliably shift generation toward a target sentence. Found via optimisation over the activation space.

**Engine primitive:** Read + write hidden states mid-forward (**axis 1: more than logits**). Requires per-layer write access. Not expressible on any black-box server.

---

#### ITI — Inference-Time Intervention

- **Title:** Inference-Time Intervention: Eliciting Truthful Answers from a Language Model
- **arXiv:** [2306.03341](https://arxiv.org/abs/2306.03341)

Li et al., 2023, NeurIPS 2023. Identifies a small set of attention heads whose activations are linearly correlated with truthfulness, then shifts their activations along the "truth direction" at each decode step. No weight modification — the intervention is purely at inference.

**Engine primitive:** Per-layer, per-head activation write (**axis 1**). Needs a per-layer write port into the residual stream. Pie's `hidden()` is a read tap at the epilogue; ITI needs a **write** port that does not yet exist in the ABI (see `09` §C5). Fundamentally blocked on black-box servers.

---

#### ActAdd — Activation Engineering

- **Title:** Steering Language Models With Activation Engineering
- **arXiv:** [2308.10248](https://arxiv.org/abs/2308.10248)

Turner et al., 2023, ICLR 2025. Adds a "steering vector" (difference of mean activations between contrastive prompt pairs) to the residual stream at a chosen layer during generation. Simple, training-free, and effective for sentiment, topic, and safety control.

**Engine primitive:** Residual-stream write at a specific layer (**axis 1**). Same ABI requirement as ITI — per-layer write port. Not expressible as a black-box API call.

---

#### RepE — Representation Engineering

- **Title:** Representation Engineering: A Top-Down Approach to AI Transparency
- **arXiv:** [2310.01405](https://arxiv.org/abs/2310.01405)

Zou et al., 2023. Identifies "representation reading" directions (linear probes) for concepts like honesty, fairness, and harmlessness in the residual stream, then applies "representation control" by adding or subtracting these directions at inference. A systematic framework unifying prior steering approaches.

**Engine primitive:** Per-layer activation read + write (**axis 1**). Pie's `hidden()` enables reading; writing requires the unbuilt per-layer write port.

---

#### CAA — Contrastive Activation Addition

- **Title:** Steering Llama 2 via Contrastive Activation Addition
- **arXiv:** [2312.06681](https://arxiv.org/abs/2312.06681)

Rimsky et al., 2023. Scales ActAdd to Llama 2: extracts steering vectors via contrastive pairs, adds them at multiple layers, and evaluates across sycophancy, power-seeking, and other behavioural dimensions. Provides detailed analysis of which layers are most effective.

**Engine primitive:** Multi-layer residual-stream write (**axis 1**). Same requirements as ActAdd/RepE.

---

#### Refusal Direction Ablation

- **Title:** Refusal in Language Models Is Mediated by a Single Direction
- **arXiv:** [2406.11717](https://arxiv.org/abs/2406.11717)

Arditi et al., 2024. Demonstrates that a single direction in residual-stream space mediates refusal behaviour. Ablating (projecting out) this direction at inference removes refusal without fine-tuning, while adding it induces refusal on benign inputs.

**Engine primitive:** Residual-stream projection at each layer (**axis 1**). A `hidden()` read + directional projection + write-back. Same per-layer write requirement.

---

#### Task Arithmetic

- **Title:** Editing Models with Task Arithmetic
- **arXiv:** [2212.04089](https://arxiv.org/abs/2212.04089)

Ilharco et al., 2023, ICLR 2023. Defines "task vectors" as the weight difference between a fine-tuned and a pre-trained model; these can be added, negated, or combined to edit model behaviour. While primarily a weight-space method, the concept applies at inference when vectors are added to activations.

**Engine primitive:** When applied at inference as activation offsets: residual-stream write (**axis 1**). When applied to weights: no engine primitive needed, but inference-time flexibility is lost.

---

#### Personalized Steering via Bi-directional Preference Optimisation

- **Title:** Personalized Steering of Large Language Models: Versatile Steering Vectors Through Bi-directional Preference Optimization
- **arXiv:** [2406.00045](https://arxiv.org/abs/2406.00045)

2024. Trains personalised steering vectors via DPO-style preference optimisation in activation space. Each user's preference profile maps to a steering direction applied at inference. Enables per-user behavioural customisation without per-user fine-tuning.

**Engine primitive:** Per-request activation steering (**axis 1**). A natural fit for Pie's per-request inferlet model, once the write port exists.

---

### B.2 Layer-level interventions at decode time

#### DoLa — Decoding by Contrasting Layers

- **Title:** DoLa: Decoding by Contrasting Layers Improves Factuality in Large Language Models
- **arXiv:** [2309.03883](https://arxiv.org/abs/2309.03883)

Chuang et al., 2024, ICLR 2024. Contrasts the output distribution at a mature (late) layer against a premature (early) layer, amplifying the "factual" signal that emerges in later layers. Improves factuality on TruthfulQA without fine-tuning.

**Engine primitive:** Per-layer logits via `layer()` + LM-head readout at two layers (**axis 1**). Pie's `layer()` intrinsic is wired on CUDA but unused; needs a per-layer LM-head readout on top. A black-box server cannot access intermediate-layer logits at all.

---

#### LayerSkip — Early Exit and Self-Speculative Decoding

- **Title:** LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding
- **arXiv:** [2404.16710](https://arxiv.org/abs/2404.16710)

Elhoushi et al., 2024. Trains early-exit classifiers at intermediate layers; at inference the model exits early for easy tokens and runs full depth for hard ones. Also uses early layers as a draft model for self-speculative decoding.

**Engine primitive:** Per-layer readout + custom draft/verify rule (**axes 1, 6**). Pie's `layer()` + custom speculator support make this user-space; elsewhere it is an engine patch.

---

#### ShortGPT — Layer Redundancy in LLMs

- **Title:** ShortGPT: Layers in Large Language Models are More Redundant Than You Expect
- **arXiv:** [2403.03853](https://arxiv.org/abs/2403.03853)

Men et al., 2024. Analyses layer importance via Block Influence (BI) scores computed from hidden-state similarity across layers. Finds that removing up to 25% of layers has minimal quality impact — an empirical basis for dynamic-depth decoding.

**Engine primitive:** Per-layer hidden-state access for BI computation (**axis 1**). Dynamic layer skipping at inference needs `layer()` to condition the skip decision.

---

#### Activation Patching — Locating and Editing Factual Associations

- **Title:** Locating and Editing Factual Associations in GPT
- **arXiv:** [2202.05262](https://arxiv.org/abs/2202.05262)

Meng et al., 2022, NeurIPS 2022. Introduces causal tracing (activation patching): run the model twice, patch activations from one run into the other at specific layers, and observe which patches restore the correct output. Localises factual knowledge to specific MLP modules.

**Engine primitive:** Per-layer activation read + write (**axis 1**). The patching intervention is the same structural operation as steering — adding/replacing activations mid-forward.

---

#### Attention Satisfies — Constraint-Satisfaction Lens on Factual Errors

- **Title:** Attention Satisfies: A Constraint-Satisfaction Lens on Factual Errors of Language Models
- **arXiv:** [2309.15098](https://arxiv.org/abs/2309.15098)

Chuang et al., 2023. Analyses factual errors through attention patterns: when the model's attention over relevant context tokens is low, it hallucinates. Proposes SAT-Probe to predict hallucination from attention distributions and intervene.

**Engine primitive:** Attention-score access via `query()` for diagnosis (**axis 1**). Intervention requires attention modification — benefits from **axis 2 (custom attention mask)**.

---

### B.3 Sparse-autoencoder-based feature steering

#### SAE Features — Sparse Autoencoders Find Interpretable Features

- **Title:** Sparse Autoencoders Find Highly Interpretable Features in Language Models
- **arXiv:** [2309.08600](https://arxiv.org/abs/2309.08600)

Cunningham et al., 2023. Trains sparse autoencoders on LM residual-stream activations and finds that individual SAE features correspond to interpretable concepts. Establishes the foundation for feature-level steering: if features are interpretable, they can be selectively amplified or suppressed.

**Engine primitive:** Read hidden states for SAE encode/decode (**axis 1**). Steering requires writing modified activations back — per-layer write port.

---

#### Scaling Monosemanticity — Extracting Features from Claude 3 Sonnet

- **Title:** Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet
- **arXiv:** [2605.29358](https://arxiv.org/abs/2605.29358)

Templeton et al., 2024 (Anthropic). Scales SAE training to production models (Claude 3 Sonnet), extracting millions of interpretable features including safety-relevant concepts (deception, sycophancy, power-seeking). Demonstrates causal steering: amplifying or suppressing individual features changes model behaviour.

**Engine primitive:** Read + write residual stream through SAE (**axis 1**). At inference, the SAE encode → feature manipulation → decode pipeline runs per-token on the residual stream.

---

#### SAE Steering of Refusal

- **Title:** Steering Language Model Refusal with Sparse Autoencoders
- **arXiv:** [2411.11296](https://arxiv.org/abs/2411.11296)

O'Brien et al., 2024. Amplifies SAE features linked to refusal to increase safety robustness. Finds that SAE-based steering causes systematic performance drops on unrelated tasks — refusal features are entangled with core capabilities.

**Engine primitive:** SAE feature read + write per step (**axis 1**). Same pipeline as Scaling Monosemanticity, with the added finding that capability/safety entanglement limits naive steering.

---

#### SAE-TS — Targeted SAE Steering

- **Title:** Improving Steering Vectors by Targeting Sparse Autoencoder Features
- **arXiv:** [2411.02193](https://arxiv.org/abs/2411.02193)

Chalnev et al., 2024. Uses SAEs to diagnose and minimise side effects of steering: measures the causal effect of steering interventions on SAE features and selectively compensates. Achieves better trade-offs between desired behavioural change and capability preservation.

**Engine primitive:** Fine-grained read + write of residual stream (**axis 1**). The diagnostic step uses SAE encoding; the compensation step modifies the steering vector.

---

#### Sparse Feature Circuits

- **Title:** Sparse Feature Circuits: Discovering and Editing Interpretable Causal Graphs in Language Models
- **arXiv:** [2403.19647](https://arxiv.org/abs/2403.19647)

Marks et al., 2024. Maps causal pathways between SAE features to build interpretable "circuits" explaining model behaviour. These circuits can be intervened upon at inference to suppress or redirect specific behaviours.

**Engine primitive:** Per-layer SAE-level read + write (**axis 1**). The circuit intervention is a structured version of activation steering at SAE-feature granularity.

---

#### LEACE — Perfect Linear Concept Erasure

- **Title:** LEACE: Perfect linear concept erasure in closed form
- **arXiv:** [2306.03819](https://arxiv.org/abs/2306.03819)

Belrose et al., 2023, NeurIPS 2023. Computes in closed form an affine projection that provably erases a specified linear concept from activations. Applied at inference, the projection removes information about a protected attribute (gender, race) without retraining.

**Engine primitive:** Per-layer activation projection (**axis 1**). The projection is a matrix multiply on `hidden()` — Pie-native for the read step; writing the projected result back needs the write port.

---

### B.4 Watermarking at the logit/hidden-state level (brief)

Covered in depth in `02-token-level-decoding-sampling.md`. The key interaction
with this chapter's themes:

#### KGW — A Watermark for Large Language Models

- **Title:** A Watermark for Large Language Models
- **arXiv:** [2301.10226](https://arxiv.org/abs/2301.10226)

Kirchenbauer et al., 2023, ICML 2023. Partitions the vocabulary into green/red lists keyed by the preceding token; a logit bias toward green-list tokens embeds a statistically detectable watermark. Distribution-shifting by design.

**Engine primitive:** Per-token keyed logit bias (**axis 4**). Pie ships `greenlist-watermarking` as an inferlet.

---

#### Distortion-Free Watermarking

- **Title:** Robust Distortion-free Watermarks for Language Models
- **arXiv:** [2307.15593](https://arxiv.org/abs/2307.15593)

Kuditipudi et al., 2023. Keys the Gumbel-max sampling noise off `hash(secret, context)` instead of the RNG counter, producing a statistically detectable watermark without shifting the output distribution. Provably preserves the model's distribution.

**Engine primitive:** Keyed Gumbel-max sampling (**axis 4**). The Pie CUDA driver already runs keyed Gumbel-max sampling with `[key, ctr]` state — this is unusually close to free.

---

### B.5 Uncertainty, hallucination control, and confidence-gated decoding

#### Semantic Entropy — Uncertainty Estimation in Natural Language Generation

- **Title:** Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in Natural Language Generation
- **arXiv:** [2302.09664](https://arxiv.org/abs/2302.09664)

Kuhn et al., 2023, ICLR 2023. Estimates uncertainty by sampling multiple completions, clustering them by semantic equivalence (via entailment), and computing entropy over the clusters. High semantic entropy predicts hallucination and enables abstention.

**Engine primitive:** Multiple forward passes sharing a prefix (**axis 3: KV branching**) for sampling, plus hidden-state access for clustering (**axis 1**). Benefits from `ctx.fork()` for efficient multi-sample generation.

---

#### Self-Consistency — Improving Chain of Thought Reasoning

- **Title:** Self-Consistency Improves Chain of Thought Reasoning in Language Models
- **arXiv:** [2203.11171](https://arxiv.org/abs/2203.11171)

Wang et al., 2022, ICLR 2023. Samples multiple reasoning paths from the same prompt and selects the answer via majority vote. A simple but powerful form of uncertainty-aware decoding.

**Engine primitive:** KV fork for parallel sampling (**axis 3**). Pie's `ctx.fork()` makes the N branches share committed pages — O(1) per fork.

---

#### Closing the Curious Case of Neural Text Degeneration

- **Title:** Closing the Curious Case of Neural Text Degeneration
- **arXiv:** [2310.01693](https://arxiv.org/abs/2310.01693)

Finlayson et al., 2024, TMLR. Provides a theoretical analysis connecting softmax temperature to the entropy of generated text, and shows that entropy-adaptive temperature schedules can prevent degeneration. Bridges the gap between truncation sampling and entropy-based adaptive decoding.

**Engine primitive:** Per-token entropy computation + temperature adjustment (**axis 4**). Matches Pie's `entropycheck` pattern — needs a device-advanced channel to act on the measured entropy.

---

#### Circuit Breakers — Improving Alignment and Robustness

- **Title:** Improving Alignment and Robustness with Circuit Breakers
- **arXiv:** [2406.04313](https://arxiv.org/abs/2406.04313)

Zou et al., 2024. Trains "circuit breakers" that monitor internal representations at inference time and interrupt generation when harmful patterns are detected. The breaker acts on hidden states rather than output tokens, catching harmful intent before it manifests.

**Engine primitive:** Per-step hidden-state monitoring + generation interrupt (**axis 1**). Needs `hidden()` read access at minimum; the interrupt is a host-side decision but the monitoring should be on-device for latency.

---

#### Latent Adversarial Training for Robustness

- **Title:** Latent Adversarial Training Improves Robustness to Persistent Harmful Behaviors in LLMs
- **arXiv:** [2407.15549](https://arxiv.org/abs/2407.15549)

Casper et al., 2024. Trains models to be robust against adversarial perturbations in activation space. At inference, this means the model's hidden representations are hardened against steering attacks, complementing the circuit-breaker approach.

**Engine primitive:** Hidden-state access for adversarial robustness evaluation (**axis 1**). Inference-time application is the defense side of activation steering.

---

### B.6 Recent developments (2025–2026)

The field is accelerating along three axes:

1. **Steering at scale** — SAE-based methods are moving from toy models to
   production (Scaling Monosemanticity; SAE-TS). The open problem is
   capability/safety entanglement: naive feature amplification degrades
   performance (O'Brien et al., 2411.11296).

2. **Combining constrained decoding with steering** — no published work
   composes grammar masks with activation steering in a single pass, but the
   Pie programming model (mask `and_into` + PTIR over `hidden()`) provides
   exactly the right primitives for this composition once the write port ships.

3. **Distortion-aware constrained decoding** — ASAp (2405.21047), AdapTrack
   (2510.17376), and CDSL (2412.10418) address the known distortion problem
   with increasingly practical solutions. All require backtracking (**axis 3**).

---

## Citation audit

All arXiv IDs in this chapter were verified by fetching the corresponding arXiv
abstract pages. The following deliberate heading-nickname deviations exist:

| Heading nickname | Registered arXiv title |
|---|---|
| Outlines — Efficient Guided Generation for LLMs | Efficient Guided Generation for Large Language Models |
| GCD — Grammar-Constrained Decoding for Structured NLP Tasks | Grammar-Constrained Decoding for Structured NLP Tasks without Finetuning |
| LMQL — Prompting Is Programming | Prompting Is Programming: A Query Language for Large Language Models |
| ActAdd — Activation Engineering | Steering Language Models With Activation Engineering |
| CAA — Contrastive Activation Addition | Steering Llama 2 via Contrastive Activation Addition |
| CAD — Hallucinate Less | Trusting Your Evidence: Hallucinate Less with Context-aware Decoding |
| KGW — A Watermark for Large Language Models | A Watermark for Large Language Models |
| EFT | An Emulator for Fine-Tuning Large Language Models using Small Language Models |
| CDSL — Constrained Decoding with Speculative Lookaheads | Constrained Decoding with Speculative Lookaheads |

Papers referenced without arXiv IDs:
- **Guidance** (Microsoft, 2023) — open-source library, no canonical arXiv paper.
- **llama.cpp GBNF** — grammar support in llama.cpp, no arXiv paper.
- **SynthID-Text** (Dathathri et al., *Nature* 2024) — published in Nature, no arXiv preprint.
- **Logit lens** (nostalgebraist, 2020) — blog post, no arXiv paper.

All `[ID](URL)` link labels match their URL targets.
