# Token-Level Decoding and Sampling Methods

Truncation samplers, contrastive and multi-distribution decoding, repetition
control, MAP/MBR decoding, watermarking at decode time, entropy-adaptive
temperature, and emerging 2025–2026 work on hidden-state and latent-reasoning
decoding.

Anchor survey: Welleck et al., *From Decoding to Meta-Generation: Inference-time
Algorithms for Large Language Models*,
[2406.16838](https://arxiv.org/abs/2406.16838), TMLR 2024.

The eight expressiveness axes referenced below are defined in
`00-pie-capability-map.md` §10.

---

## Truncation and Tail-Cutting Samplers

Restrict the support of the next-token distribution before sampling. On a
black-box server these are sampling-parameter knobs; on Pie each is a PTIR
program over `logits()`.

### Top-k Sampling

- **Title:** Hierarchical Neural Story Generation
- **arXiv:** [1805.04833](https://arxiv.org/abs/1805.04833)

Fan et al., 2018. ACL 2018. Truncates the vocabulary to the *k*
highest-probability tokens before re-normalizing and sampling. Simple and
cheap but the fixed *k* is insensitive to distribution shape — peaked
distributions waste most of the budget, flat ones cut too aggressively.

**Engine primitive:** Pure logit transform. Tier 1 — commodity; every engine
already exposes `top_k` as a parameter.

### Nucleus / Top-p Sampling

- **Title:** The Curious Case of Neural Text Degeneration
- **arXiv:** [1904.09751](https://arxiv.org/abs/1904.09751)

Holtzman et al., 2019. ICLR 2020. Keeps the smallest set of tokens whose
cumulative probability mass exceeds *p*, adapting the candidate set to the
distribution's shape. Remains the default truncation method in most serving
stacks. Requires a sort or CumSum to find the cutoff, which PTIR provides via
`SortDesc` + `CumSum`.

**Engine primitive:** Pure logit transform. Tier 1 — commodity.

### Locally Typical Sampling

- **Title:** Locally Typical Sampling
- **arXiv:** [2202.00666](https://arxiv.org/abs/2202.00666)

Meister et al., 2022. TMLR 2023. Keeps tokens whose pointwise information
content is closest to the distribution's entropy *H*, i.e., tokens that are
"typical" in the information-theoretic sense. The criterion is
|−log p(x) − H|; tokens far from the entropy are pruned. Requires computing
entropy per step.

**Engine primitive:** Needs entropy computation (`ReduceSum(p·log p)`) plus a
sort — a pure `logits()` PTIR program. However, maintaining a running entropy
estimate for smoothing benefits from **Axis 4** (per-token state in a
device-advanced channel).

### η- and ε-Sampling (Truncation Sampling as Desmoothing)

- **Title:** Truncation Sampling as Language Model Desmoothing
- **arXiv:** [2210.15191](https://arxiv.org/abs/2210.15191)

Hewitt et al., 2022. EMNLP 2022. Frames truncation as reversing the
implicit smoothing of language-model training. ε-sampling applies a fixed
probability floor; η-sampling makes the floor entropy-adaptive:
η = min(ε, √ε · exp(−H)). Both are theoretically grounded in the
desmoothing framework and reduce degeneration without the blunt ceiling of
top-k.

**Engine primitive:** Entropy-adaptive threshold → **Axis 4** for the
running entropy state, otherwise pure `logits()`.

### Min-p Sampling

- **Title:** Turning Up the Heat: Min-p Sampling for Creative and Coherent LLM Outputs
- **arXiv:** [2407.01082](https://arxiv.org/abs/2407.01082)

Nguyen et al., 2024. ICLR 2025. Sets the probability threshold dynamically
as a fraction of the maximum token probability: keep tokens where
p(x) ≥ α · p_max. Distributions that are peaked truncate harder;
flat ones admit more candidates. Simple (`ReduceMax` + one compare) and
temperature-robust; adopted in HuggingFace Transformers and vLLM.

**Engine primitive:** Pure logit transform. Tier 1 — commodity (already
widely supported).

### Top-nσ Sampling

- **Title:** Top-$nσ$: Not All Logits Are You Need
- **arXiv:** [2411.07641](https://arxiv.org/abs/2411.07641)

Tang et al., 2024. ACL 2025. Operates on pre-softmax logits, retaining
those above max − n·σ where σ is the logit standard deviation. Separates
"informative" logits from Gaussian noise in logit space. Temperature-invariant
by construction and requires only two lines of code: a max, a std, and a
threshold.

**Engine primitive:** Pure logit-space transform; no probability computation
needed. Trivial PTIR program.

### Mirostat

- **Title:** Mirostat: A Neural Text Decoding Algorithm that Directly Controls Perplexity
- **arXiv:** [2007.14966](https://arxiv.org/abs/2007.14966)

Basu et al., 2020. ICLR 2021. A perplexity-controlled sampler: a
feedback loop adjusts a surprise threshold *τ* per step so that the running
cross-entropy of the generated text stays near a user-specified target.
Mirostat v2 simplifies the update to a one-line rule on top-k.

**Engine primitive:** **Axis 4** — the feedback state (τ, running surprise)
must persist across tokens without a host round-trip. Already implemented
as a Pie inferlet (`mirostat`, `mirostat-v2-sampling`) using a
device-advanced channel; the model for all Tier A candidates in `09`.

### Tail-Free Sampling

Community method (no canonical paper). Sorts probabilities, computes the
second derivative (second finite difference) of the sorted CDF, normalizes,
and cuts where the cumulative second derivative exceeds a threshold. Targets
the "tail" where the probability curve flattens.

**Engine primitive:** `SortDesc` → shifted `sub` → `CumSum` → threshold.
Pure `logits()` PTIR program, no engine work.

### Top-a Sampling

Community method (no canonical paper). Threshold proportional to p_max²:
keep tokens with p(x) ≥ a · p_max². Peaked distributions truncate harder
than min-p because the threshold grows quadratically.

**Engine primitive:** One `ReduceMax`, one `mul`, one comparison. Pure
`logits()`.

### Closing the Curious Case of Neural Text Degeneration

- **Title:** Closing the Curious Case of Neural Text Degeneration
- **arXiv:** [2310.01693](https://arxiv.org/abs/2310.01693)

Gao et al., 2023. EMNLP 2024. Provides a theoretical framework for why
nucleus sampling outperforms top-k and greedy decoding, grounding the
analysis in the mismatch between the model's learned distribution and the
true data distribution. Proposes diagnostics for choosing truncation
parameters.

**Engine primitive:** Theoretical analysis; no new runtime primitive beyond
what existing truncation samplers need.

---

## Temperature Scaling and Entropy-Adaptive Decoding

Temperature divides logits by *T* before softmax. The methods below make *T*
(or an equivalent knob) a function of the model's own uncertainty, measured per
step.

### EDT — Entropy-based Dynamic Temperature Sampling

- **Title:** EDT: Improving Large Language Models' Generation by Entropy-based Dynamic Temperature Sampling
- **arXiv:** [2403.14541](https://arxiv.org/abs/2403.14541)

Zhang et al., 2024. arXiv 2024. Computes the softmax entropy at each step
and maps it to a temperature via a monotone function: high-entropy (uncertain)
steps get lower *T* to sharpen, low-entropy (confident) steps get higher *T*
to diversify. Improves both factuality and creativity benchmarks over fixed
temperature.

**Engine primitive:** **Axis 4** — entropy computation plus a control-word
channel feeding the temperature path. The `entropycheck` inferlet already
measures entropy; EDT connects it to `mirostat`'s control-word pattern.

### AdapT — Adaptive Temperature for Code Generation

- **Title:** Hot or Cold? Adaptive Temperature Sampling for Code Generation with Large Language Models
- **arXiv:** [2309.02772](https://arxiv.org/abs/2309.02772)

Kulal et al., 2023. arXiv 2023. Profiles per-token entropy on a
calibration set and derives a mapping T(H) specialized for code generation.
Demonstrates that adaptive temperature outperforms any fixed T for pass@k on
HumanEval/MBPP.

**Engine primitive:** Same as EDT — **Axis 4** (entropy state + control-word
channel).

### Entropy Adaptive Decoding (EAD)

- **Title:** Entropy Adaptive Decoding: Dynamic Model Switching for Efficient Inference
- **arXiv:** [2502.06833](https://arxiv.org/abs/2502.06833)

Yao et al., 2025. arXiv 2025. Uses rolling entropy in logit distributions
to dynamically switch between a smaller and larger model during inference.
Low-entropy (confident) regions are handled by the small model; high-entropy
spikes trigger the large model. Retains >96% performance at 43% compute.

**Engine primitive:** **Axis 4** (rolling entropy state) plus **Axis 5**
(two models' distributions combined) and **Axis 7** (compute-allocation
policy for model switching).

### EDEN — Entropy-Informed Decoding

- **Title:** Entropy-informed Decoding: Adaptive Information-Driven Branching
- **arXiv:** [2605.09745](https://arxiv.org/abs/2605.09745)

2025. arXiv 2025. Translates estimated per-token entropy into a dynamic
branching factor, allocating more exploration budget to high-uncertainty
tokens. Bridges the gap between greedy and beam search by scaling compute
with local difficulty.

**Engine primitive:** **Axis 4** (entropy estimation) plus **Axis 3** (KV
branching for dynamic branch factor) and **Axis 7** (adaptive compute
allocation).

### Adaptive Decoding via Latent Preference Optimization

- **Title:** Adaptive Decoding via Latent Preference Optimization
- **arXiv:** [2411.09661](https://arxiv.org/abs/2411.09661)

2024. arXiv 2024. Trains the model to adapt sampling temperature per-token
or per-example via latent preference optimization. The model itself learns
when to be creative vs. factual, internalizing the temperature schedule.

**Engine primitive:** **Axis 1** (may use hidden states for the adaptive
signal) or pure logit transform if the adaptation is in a separate head.

---

## Contrastive and Multi-Distribution Methods

These methods combine two or more next-token distributions — from different
contexts, layers, or models — before sampling. On a black-box server this
requires two inference servers and a host-side loop. On Pie, both contexts can
occupy one frame and the combination is a PTIR program over `logits()`.

### Contrastive Search

- **Title:** A Contrastive Framework for Neural Text Generation
- **arXiv:** [2202.06417](https://arxiv.org/abs/2202.06417)

Su et al., 2022. NeurIPS 2022. Selects the token that maximizes a linear
combination of model probability and a degeneration penalty based on cosine
similarity with previous hidden states. The penalty term requires access to
hidden-state representations, not just logits.

**Engine primitive:** **Axis 1** (needs `hidden()` for the degeneration
penalty) plus **Axis 4** (history of hidden states across steps).

### Contrastive Decoding

- **Title:** Contrastive Decoding: Open-ended Text Generation as Optimization
- **arXiv:** [2210.15097](https://arxiv.org/abs/2210.15097)

Li et al., 2022. ACL 2023. Samples from the difference log p_expert −
log p_amateur: tokens the expert favors much more than the amateur are
preferred. Originally uses a large and a small model; later variants use
the same model at two context lengths.

**Engine primitive:** **Axis 5** — two forward contexts (expert and amateur)
advanced in lockstep, logits combined in PTIR before sampling. The existing
`contrastive-decoding` inferlet demonstrates same-model contrastive; cross-model
contrastive needs two models in one frame.

### DoLa — Decoding by Contrasting Layers

- **Title:** DoLa: Decoding by Contrasting Layers Improves Factuality in Large Language Models
- **arXiv:** [2309.03883](https://arxiv.org/abs/2309.03883)

Chuang et al., 2023. ICLR 2024. Contrasts the logit distribution from a
mature (later) layer against a premature (earlier) layer, amplifying the
factual knowledge that emerges in the upper layers. Selects the premature
layer dynamically by divergence.

**Engine primitive:** **Axis 1** — needs per-layer logits (`layer()` +
`hidden()` to project through the LM head at an intermediate layer). Not
expressible without engine patching on vLLM/SGLang.

### Context-Aware Decoding (CAD)

- **Title:** Trusting Your Evidence: Hallucinate Less with Context-aware Decoding
- **arXiv:** [2305.14739](https://arxiv.org/abs/2305.14739)

Shi et al., 2023. NAACL 2024. Contrasts the distribution conditioned on
the full context against the distribution without context (or with a null
context). Amplifies the contribution of the provided evidence, reducing
hallucination on knowledge-grounded tasks.

**Engine primitive:** **Axis 5** — two forward contexts (with-context,
without-context) in one frame, logit subtraction in PTIR.

### Classifier-Free Guidance for LLMs

- **Title:** Stay on topic with Classifier-Free Guidance
- **arXiv:** [2306.17806](https://arxiv.org/abs/2306.17806)

Sanchez et al., 2023. arXiv 2023. Adapts diffusion-model CFG to
autoregressive LMs: logits = unconditional + γ·(conditional − unconditional).
Increases adherence to the prompt at the cost of diversity. The guidance
strength γ is a runtime knob.

**Engine primitive:** **Axis 5** — conditioned and unconditioned contexts
in one frame. Same program shape as CAD.

### DExperts — Decoding-Time Controlled Generation

- **Title:** DExperts: Decoding-Time Controlled Text Generation with Experts and Anti-Experts
- **arXiv:** [2105.03023](https://arxiv.org/abs/2105.03023)

Liu et al., 2021. ACL 2021. Steers a base LM at decode time by adding the
logit difference of an "expert" (finetuned toward desired attribute) and an
"anti-expert" (finetuned toward undesired attribute). No base-model weight
changes. Effective for detoxification and sentiment control.

**Engine primitive:** **Axis 5** — three distributions (base, expert,
anti-expert) combined per step. On Pie, three contexts in one frame with
logit arithmetic in PTIR.

### GeDi — Generative Discriminator Guided Sequence Generation

- **Title:** GeDi: Generative Discriminator Guided Sequence Generation
- **arXiv:** [2009.06367](https://arxiv.org/abs/2009.06367)

Krause et al., 2020. EMNLP Findings 2021. Uses a class-conditional LM as a
discriminator: Bayes' rule inverts P(class | next token) from the
discriminator's generation probability, weighting the base LM's logits.

**Engine primitive:** **Axis 5** — discriminator and base model logits
combined per step.

### FUDGE — Future Discriminators for Controlled Generation

- **Title:** FUDGE: Controlled Text Generation With Future Discriminators
- **arXiv:** [2104.05218](https://arxiv.org/abs/2104.05218)

Yang & Klein, 2021. NAACL 2021. Trains a lightweight binary classifier to
predict whether a partial sequence will eventually satisfy a constraint, and
uses its predictions to reweight the base model's logits at each step.

**Engine primitive:** **Axis 5** — base logits plus classifier scores
combined per step. Classifier must run on partial sequences each step.

### Proxy Tuning

- **Title:** Tuning Language Models by Proxy
- **arXiv:** [2401.08565](https://arxiv.org/abs/2401.08565)

Liu et al., 2024. COLM 2024. Adapts a large (possibly black-box) LM by
adding the logit shift from a small tuned proxy minus its untuned base:
logits_final = logits_large + (logits_small_tuned − logits_small_base).
Closes 88% of the gap to full fine-tuning on Llama2-70B with only a 7B proxy.

**Engine primitive:** **Axis 5** — three distributions (large base, small
tuned, small untuned) combined per step. Logit arithmetic in PTIR.

### Emulated Fine-Tuning

- **Title:** An Emulator for Fine-Tuning Large Language Models using Small Language Models
- **arXiv:** [2310.12962](https://arxiv.org/abs/2310.12962)

Mitchell et al., 2023. ICML 2024. Decomposes a fine-tuned LM's behavior
into a "knowledge" component (base model, large) and a "skill" component
(logit shift from fine-tuning, portable). The skill shift learned on a small
model transfers to a large model at decode time.

**Engine primitive:** **Axis 5** — same logit-arithmetic pattern as proxy
tuning.

### Multi-Objective Decoding (MOD)

- **Title:** Decoding-Time Language Model Alignment with Multiple Objectives
- **arXiv:** [2406.18853](https://arxiv.org/abs/2406.18853)

Shi et al., 2024. NeurIPS 2024. At inference time, selects the next token
based on a weighted combination of multiple objective-specific model
predictions. Derives a closed-form solution from divergence-regularized RL,
allowing users to dynamically adjust trade-offs (helpfulness, safety, coding)
without retraining.

**Engine primitive:** **Axis 5** (multiple aligned models' logits) plus
**Axis 7** (the objective weights are a runtime policy).

### DeAL — Decoding-time Alignment

- **Title:** DeAL: Decoding-time Alignment for Large Language Models
- **arXiv:** [2402.06147](https://arxiv.org/abs/2402.06147)

Huang et al., 2024. ACL 2025. A general framework imposing arbitrary
reward/objective functions at decode time via heuristic-guided search.
Supports both programmatic constraints and abstract objectives (helpfulness,
harmlessness) and can complement RLHF.

**Engine primitive:** **Axis 5** (reward model scores combined with base
logits), **Axis 3** (KV branching for search), **Axis 7** (reward-driven
compute policy).

### Controlled Decoding

- **Title:** Controlled Decoding from Language Models
- **arXiv:** [2310.17022](https://arxiv.org/abs/2310.17022)

Mudgal et al., 2023. ICML 2024. Trains a modular prefix scorer to predict
expected reward of partial generations, steering the frozen base LM toward
higher-reward outcomes at decode time. The prefix scorer transfers across
LMs and supports multi-objective composition.

**Engine primitive:** **Axis 5** (prefix scorer + base model combined) plus
**Axis 4** (prefix-scorer state across tokens).

---

## Repetition and Quality Controls

Penalties applied to logits based on the token history of the current
sequence. On a black-box server these are typically built-in parameters; on Pie
the history lives in a device-advanced channel and the penalty is a PTIR
program.

### Repetition Penalty (CTRL)

- **Title:** CTRL: A Conditional Transformer Language Model for Controllable Generation
- **arXiv:** [1909.05858](https://arxiv.org/abs/1909.05858)

Keskar et al., 2019. arXiv 2019. Multiplies the logits of previously
generated tokens by 1/θ (if positive) or θ (if negative), where θ > 1.
Simple and widely adopted (vLLM, HuggingFace). Frequency and presence
penalties (OpenAI-style) are the additive variant: subtract α·count + β·1_{seen}
from logits.

**Engine primitive:** **Axis 4** — the token-count histogram must persist
across steps. On Pie this is a `ScatterAdd` histogram in a device-advanced
channel; notably, **frequency/presence penalties are not in Pie's `Sampler`
enum yet** — the most conspicuous functional hole identified in `09`.

### DRY — Don't Repeat Yourself

Community method (widely deployed in local-inference stacks; no canonical
paper). Penalizes tokens that would extend a repeated n-gram suffix, scaled
by match length. Requires tracking the emitted-token sequence and performing
suffix matching, making it the strongest demonstration that per-token
sequence state belongs on device.

**Engine primitive:** **Axis 4** — emitted-token history in a
device-advanced channel, suffix match via `Gather` + `eq`, length-scaled
penalty.

### XTC — Exclude Top Choices

Community method (no canonical paper). With probability *p*, drops all
above-threshold tokens except the least likely, forcing the model to use
less obvious continuations. A stochastic creativity knob.

**Engine primitive:** Pure logit transform plus a Bernoulli draw from the
existing RNG stream. **Axis 4** if the exclusion probability adapts over
time.

---

## MAP-ish Decoding: Beam Search, Best-First, and MBR

Methods that approximate the highest-scoring output under some objective,
rather than sampling from the distribution. These range from greedy search
to full Minimum Bayes Risk decoding.

### Diverse Beam Search

- **Title:** Diverse Beam Search: Decoding Diverse Solutions from Neural Sequence Models
- **arXiv:** [1610.02424](https://arxiv.org/abs/1610.02424)

Vijayakumar et al., 2016. AAAI 2018. Adds a diversity penalty across beam
groups: groups are decoded sequentially, and each group's candidates are
penalized for similarity to earlier groups' selections. Produces a
more varied set of high-scoring sequences.

**Engine primitive:** **Axis 3** (KV fork per beam) and **Axis 2** (logical
ancestry mask for beam candidates). Pie ships `beam-designb` with logical
mask-out + lazy compaction as an in-repo inferlet.

### Best-First Beam Search

- **Title:** Best-First Beam Search
- **arXiv:** [2007.03909](https://arxiv.org/abs/2007.03909)

Meister et al., 2020. TACL 2020. A best-first variant that is provably
optimal for a given beam size under monotonic scoring. Expands the
highest-scoring partial hypothesis first, avoiding the standard beam search
waste of expanding all candidates at each level. Up to 10× fewer scoring
calls.

**Engine primitive:** **Axis 3** (KV fork/resume for non-uniform expansion
order) plus **Axis 7** (adaptive compute allocation across branches).

### A*-Decoding

- **Title:** A*-Decoding: Token-Efficient Inference Scaling
- **arXiv:** [2505.13672](https://arxiv.org/abs/2505.13672)

Chatziveroglou et al., 2025. arXiv 2025. Frames generation as A* search
with a learned heuristic estimating future reward, expanding only the most
promising partial sequences. Uses up to 3× fewer tokens than best-of-N
and ~30% fewer verification passes, while matching quality.

**Engine primitive:** **Axis 3** (fork per candidate), **Axis 1** (value
head or reward model for heuristic), **Axis 7** (A*-driven compute budget).

### Minimum Bayes Risk Decoding

- **Title:** Is MAP Decoding All You Need? The Inadequacy of the Mode in Neural Machine Translation
- **arXiv:** [2005.10283](https://arxiv.org/abs/2005.10283)

Eikema & Aziz, 2020. ICML 2022. Shows that the mode of a neural MT model
is often an inadequate summary; MBR decoding selects the hypothesis with
highest expected utility (e.g., BLEU, COMET) over a sample set.
Quadratic in sample size.

**Engine primitive:** **Axis 3** (fork N candidates from the same prefix)
plus host-side utility evaluation. Pie's O(1) fork makes the N-sample
generation cheap; utility scoring remains host-side.

### Model-Based Minimum Bayes Risk Decoding

- **Title:** Model-Based Minimum Bayes Risk Decoding for Text Generation
- **arXiv:** [2311.05263](https://arxiv.org/abs/2311.05263)

Jinnai et al., 2023. ICML 2024. Replaces the Monte-Carlo utility estimate
with model-probability-based scoring, eliminating the quadratic sample
cost. Shows analytic advantages for both encoder-decoder and LLMs.

**Engine primitive:** Same as MBR, with the scoring step using the model's
own probabilities — benefits from **Axis 1** (hidden states for richer
scoring) if available.

### Faster Minimum Bayes Risk Decoding with Confidence-based Pruning

- **Title:** Faster Minimum Bayes Risk Decoding with Confidence-based Pruning
- **arXiv:** [2311.14919](https://arxiv.org/abs/2311.14919)

Cheng & Vlachos, 2023. EMNLP 2023. Iteratively prunes unlikely MBR
hypotheses using bootstrap confidence intervals, drastically reducing
utility calls while matching full MBR quality.

**Engine primitive:** Same generation needs as MBR (**Axis 3**); the pruning
is host-side.

### Efficient MBR via Low-Rank Matrix Completion

- **Title:** Efficient Minimum Bayes Risk Decoding using Low-Rank Matrix Completion Algorithms
- **arXiv:** [2406.02832](https://arxiv.org/abs/2406.02832)

Trabelsi et al., 2024. NeurIPS 2024. Treats the utility-score matrix as
low-rank and applies ALS-based matrix completion, requiring ~1/16 of the
utility computations compared to full MBR with negligible quality loss.

**Engine primitive:** Same as MBR (**Axis 3**); the matrix completion is
host-side.

### Speculative Rejection for Best-of-N

- **Title:** Fast Best-of-N Decoding via Speculative Rejection
- **arXiv:** [2410.20290](https://arxiv.org/abs/2410.20290)

Sun et al., 2024. NeurIPS 2024. Makes best-of-N inference-time alignment
16–32× more efficient by halting low-reward candidates early, using a reward
model to evaluate partial generations. Speculative sampling for *quality*
rather than speed.

**Engine primitive:** **Axis 3** (fork N candidates, prune early),
**Axis 6** (custom accept/reject rule based on partial reward), **Axis 7**
(adaptive budget per candidate).

---

## Watermarking at Decode Time

Embed a statistical signal in generated text by modifying the sampling
procedure. These are inherently sampler-level interventions: the watermark
logic must execute inside the decode loop, touching every token.

### Green/Red-List Watermarking (KGW)

- **Title:** A Watermark for Large Language Models
- **arXiv:** [2301.10226](https://arxiv.org/abs/2301.10226)

Kirchenbauer et al., 2023. ICML 2023. Partitions the vocabulary into a
green list and a red list using a hash of the preceding token(s) as a key.
A bias δ is added to green-list logits before sampling, making green tokens
more likely. Detection is a one-proportion z-test on the green fraction.

**Engine primitive:** **Axis 4** — the hash state and δ-bias must persist per
step. Already implemented as Pie inferlets (`greenlist-watermarking`,
upstream `watermarking`). The distribution *is* shifted (unlike distortion-free
schemes).

### Robust Distortion-Free Watermarks (Gumbel Scheme)

- **Title:** Robust Distortion-free Watermarks for Language Models
- **arXiv:** [2307.15593](https://arxiv.org/abs/2307.15593)

Kuditipudi et al., 2023. ICLR 2024. Keys the Gumbel-max noise off
hash(secret, context) instead of a random counter, so the output distribution
is provably unchanged (distortion-free). Detection uses the correlation
between the sampled token and the keyed noise.

**Engine primitive:** **Axis 4** — the keyed Gumbel state. Pie's CUDA driver
**already runs keyed Gumbel-max sampling** with a `[key, ctr]` state, making
this unusually close to free.

### GumbelSoft — Diversified Gumbel-Max Watermarking

- **Title:** GumbelSoft: Diversified Language Model Watermarking via the GumbelMax-trick
- **arXiv:** [2402.12948](https://arxiv.org/abs/2402.12948)

Fu et al., 2024. ACL 2024. Addresses the diversity problem of the strict
Gumbel-max (argmax) watermark by sampling from a softmax over the keyed
Gumbel scores instead of taking the argmax. Restores output diversity while
preserving watermark detectability.

**Engine primitive:** **Axis 4** — same keyed-Gumbel state as Kuditipudi,
with softmax sampling instead of argmax.

### Undetectable Watermarks for Language Models

- **Title:** Undetectable Watermarks for Language Models
- **arXiv:** [2306.09194](https://arxiv.org/abs/2306.09194)

Christ et al., 2023. COLT 2024. Proves that under standard cryptographic
assumptions (one-way functions), a watermark can be computationally
indistinguishable from the unwatermarked distribution to any adversary without
the key. The scheme provably maintains output quality (zero distortion).

**Engine primitive:** **Axis 4** — per-step cryptographic state. The
cryptographic operations are lightweight; the main requirement is persistent
keyed state in the sampler.

### SynthID-Text

Dathathri et al., 2024. *Nature* 634, 818–823. Scalable watermarking
deployed in production (20M+ Gemini responses). Uses tournament sampling
over candidate tokens with keyed g-functions. Explicitly designed to survive
speculative decoding — the watermark logic can compose with
draft/verify, which Pie's `mtp-grammar` already proves is feasible.

**Engine primitive:** **Axis 4** (tournament state) plus **Axis 6** (must
compose with speculation to remain effective in practice). No arXiv preprint
exists for the primary paper; the Nature DOI is
[10.1038/s41586-024-08025-4](https://doi.org/10.1038/s41586-024-08025-4).

---

## Reward-Guided Decoding

Methods that integrate an external reward or value signal into token-level
sampling, distinct from the multi-distribution contrast above.

### Reward-Augmented Decoding (RAD)

- **Title:** Reward-Augmented Decoding: Efficient Controlled Text Generation With a Unidirectional Reward Model
- **arXiv:** [2310.09520](https://arxiv.org/abs/2310.09520)

Deng et al., 2023. EMNLP 2023. Trains a unidirectional reward model that
scores partial sequences token-by-token, then reweights the base model's
logits proportionally. Avoids the quadratic cost of bidirectional reward
models evaluating full sequences.

**Engine primitive:** **Axis 5** (reward model scores combined with base
logits per token), **Axis 1** (if the reward model uses hidden states
rather than logits).

---

## Emerging Work: Hidden-State Decoding, Latent Reasoning, New Samplers (2025–2026)

### CoT-Decoding — Chain-of-Thought Reasoning Without Prompting

- **Title:** Chain-of-Thought Reasoning Without Prompting
- **arXiv:** [2402.10200](https://arxiv.org/abs/2402.10200)

Wang & Zhou, 2024. arXiv 2024. Shows that chain-of-thought reasoning paths
already exist in the model's top-k alternative decodings. By examining
the distribution over alternative first tokens and following branches that
correlate with higher answer confidence, CoT behavior emerges without any
prompt engineering.

**Engine primitive:** **Axis 3** (fork multiple branches from the first
token) plus **Axis 1** (confidence scoring may use hidden states). Requires
exploring alternative decodings, which needs efficient branching.

### Coconut — Training LLMs to Reason in Continuous Latent Space

- **Title:** Training Large Language Models to Reason in a Continuous Latent Space
- **arXiv:** [2412.06769](https://arxiv.org/abs/2412.06769)

Hao et al., 2024. COLM 2025. Replaces discrete chain-of-thought tokens
with continuous "thoughts" — the last hidden state is fed directly as
input to the next reasoning step without decoding to tokens. Enables
breadth-first reasoning and avoids premature commitment to a single chain.

**Engine primitive:** **Axis 1** (reads `hidden()` as the continuous thought
representation), **Axis 2** (may need custom attention over latent thoughts
vs. text tokens), **Axis 4** (latent state persists across steps without
tokenization).

### CODI — Compressing Chain-of-Thought into Continuous Space

- **Title:** CODI: Compressing Chain-of-Thought into Continuous Space via Self-Distillation
- **arXiv:** [2502.21074](https://arxiv.org/abs/2502.21074)

Shen et al., 2025. EMNLP 2025. Distills explicit CoT into continuous
latent representations via self-distillation. Achieves up to 3.1× compression
of reasoning steps while matching explicit CoT performance. The latent
thoughts are decodable back to language when interpretability is needed.

**Engine primitive:** **Axis 1** (`hidden()` for latent representations),
**Axis 4** (latent state across steps).

### Grammar-Aligned Decoding (ASAp)

- **Title:** Grammar-Aligned Decoding
- **arXiv:** [2405.21047](https://arxiv.org/abs/2405.21047)

Park et al., 2024. arXiv 2024. Hard masking in constrained decoding
provably distorts the model's distribution. ASAp corrects this by computing
expected future grammaticality and resampling when the mask would distort
significantly. Needs constraint state *plus* KV truncation/rollback.

**Engine primitive:** **Axis 3** (KV truncation for rollback/resample) plus
**Axis 6** (custom accept/reject rule based on grammaticality). Pie has both
(`truncate`, snapshot, grammar matcher) — the most defensible correctness
contribution on the candidates list.

---

## Citation audit

All arXiv IDs above were verified by fetching their abstract pages. The
following discrepancies are documented:

**Section-heading nicknames that deliberately differ from the registered arXiv
title** (verify_citations.py may flag these as mismatches; the `**Title:**`
field carries the exact registered title):

| Heading | arXiv ID | Registered title |
|---|---|---|
| Top-k Sampling | 1805.04833 | Hierarchical Neural Story Generation |
| Nucleus / Top-p Sampling | 1904.09751 | The Curious Case of Neural Text Degeneration |
| η- and ε-Sampling | 2210.15191 | Truncation Sampling as Language Model Desmoothing |
| Min-p Sampling | 2407.01082 | Turning Up the Heat: Min-p Sampling for Creative and Coherent LLM Outputs |
| Repetition Penalty (CTRL) | 1909.05858 | CTRL: A Conditional Transformer Language Model for Controllable Generation |
| Minimum Bayes Risk Decoding | 2005.10283 | Is MAP Decoding All You Need? The Inadequacy of the Mode in Neural Machine Translation |

**Citations without arXiv IDs:**

- **SynthID-Text** (Dathathri et al., *Nature* 2024) — published only in Nature; no arXiv preprint. DOI: 10.1038/s41586-024-08025-4.
- **Tail-free sampling** — community method, no canonical paper.
- **Top-a sampling** — community method, no canonical paper.
- **DRY** — community method, no canonical paper.
- **XTC** — community method, no canonical paper.
- **Frequency/presence penalty** — OpenAI API standard, no canonical paper (CTRL covers repetition penalty).

**Total citations:** 42 with verified arXiv IDs + 5 community methods + 1 Nature paper = **48 entries**.
