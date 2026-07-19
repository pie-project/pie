# Attention Sinks, KV Eviction/Compression, Query-Aware Sparse Attention, Prefix/Non-Prefix Cache Reuse, Long-Context Restructuring

Inference-time algorithms that manipulate the KV cache and attention mask — which keys/values to keep, share, evict, compress, or attend to. Each entry notes which of the eight engine-primitive axes from `00-pie-capability-map.md` the method requires.

Axis shorthand used throughout:
- **Axis 1** — more than logits per step (hidden states, queries, per-layer activations)
- **Axis 2** — custom attention mask
- **Axis 3** — explicit KV branching / backtracking (fork, snapshot, truncate)
- **Axis 4** — per-token stateful logic without host round-trip
- **Axis 5** — combining several distributions
- **Axis 6** — custom draft/verify rule
- **Axis 7** — guest compute-allocation policy
- **Axis 8** — tool/agent I/O interleaved with generation

---

## Attention sinks and streaming methods

### Efficient Streaming Language Models with Attention Sinks

- **arXiv:** [2309.17453](https://arxiv.org/abs/2309.17453)

Xiao et al., 2023, NeurIPS 2024. Observes that the first few tokens ("attention sinks") receive disproportionate attention mass regardless of semantic content. StreamingLLM retains a fixed set of sink tokens plus a sliding window of recent tokens, enabling infinite-length streaming inference with bounded KV memory. Requires no fine-tuning; the sink set is determined at deployment.

**Engine primitive:** Axis 2 (custom attention mask) — the sink-plus-window pattern is a non-standard mask. Also benefits from Axis 4 (per-token state) to maintain the sink set bookkeeping without host round-trips. Pie ships an `attention-sink` inferlet. vLLM/SGLang lack user-defined masks; StreamingLLM must be a built-in mode.

### LM-Infinite: Zero-Shot Extreme Length Generalization for Large Language Models

- **arXiv:** [2308.16137](https://arxiv.org/abs/2308.16137)

Han et al., 2023. Introduces a Λ-shaped attention mask: attend to distant "landmark" tokens plus a local window, and cap positional encodings beyond the training length. Achieves length generalization to 200k+ tokens without fine-tuning. The position-capping trick is complementary to attention sinks.

**Engine primitive:** Axis 2 (custom mask — the Λ shape) plus position-modification capability (RoPE capping at inference). On a black-box server, both the mask pattern and the position rewrite require engine patches.

---

## KV eviction and sparsification

### H₂O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models

- **Title:** H$_2$O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models
- **arXiv:** [2306.14048](https://arxiv.org/abs/2306.14048)

Zhang et al., 2023, NeurIPS 2023. Identifies "heavy hitter" tokens that accumulate high attention scores across decoding steps, then retains only these plus recent tokens in a fixed-budget KV cache. Eviction decisions are driven by cumulative attention scores computed during the forward pass. Demonstrates that a small fraction of tokens dominate attention mass.

**Engine primitive:** Axis 1 (needs `query()` to compute attention scores on-device for eviction decisions) + Axis 2 (modified attention mask after eviction). On Pie, this maps to a PTIR program using `query()` — currently wired in CUDA but unused (see `09`, Tier B1). Impossible to implement as a user policy on vLLM/SGLang.

### Scissorhands: Exploiting the Persistence of Importance Hypothesis for LLM KV Cache Compression at Test Time

- **arXiv:** [2305.17118](https://arxiv.org/abs/2305.17118)

Liu et al., 2023, NeurIPS 2023. Shows that tokens important in one step tend to remain important in future steps ("persistence of importance"). Exploits this by identifying pivotal tokens from a single attention computation and retaining only those, achieving cache compression with minimal quality loss.

**Engine primitive:** Axis 1 (attention-score-driven eviction) + Axis 2 (post-eviction mask). Same requirements as H₂O — needs the attention tap.

### SnapKV: LLM Knows What You are Looking for Before Generation

- **arXiv:** [2404.14469](https://arxiv.org/abs/2404.14469)

Li et al., 2024. Uses an "observation window" at the end of the prompt to compute attention patterns, then selects per-head important KV positions for the entire generation. One-shot selection at prefill time rather than per-step eviction. Achieves aggressive compression with strong performance on long-context benchmarks.

**Engine primitive:** Axis 1 (observation-window attention scores from `query()` at prefill time) + Axis 2 (sparse mask during decode). Pie Tier B2 (see `09`). Integrated into some vLLM builds as a fixed policy, not user-programmable.

### PyramidKV: Dynamic KV Cache Compression based on Pyramidal Information Funneling

- **arXiv:** [2406.02069](https://arxiv.org/abs/2406.02069)

Cai et al., 2024. Observes that different layers need different cache budgets — lower layers attend broadly (need more cache), upper layers attend sparsely (need less). Allocates cache budget in a pyramid shape across layers: more KV entries for lower layers, fewer for upper layers. Improves over uniform-budget methods like SnapKV.

**Engine primitive:** Axis 1 (per-layer attention analysis) + Axis 2 (layer-varying sparse mask). Requires engine-level per-layer cache management not available as a user API on vLLM/SGLang.

### PyramidInfer: Pyramid KV Cache Compression for High-throughput LLM Inference

- **arXiv:** [2405.12532](https://arxiv.org/abs/2405.12532)

Yang et al., 2024. Compresses the KV cache layer by layer during prefill, retaining fewer keys at higher layers based on attention pivots. Achieves up to 54% GPU memory reduction with negligible accuracy loss. Focuses on throughput improvement via reduced memory footprint.

**Engine primitive:** Axis 1 (per-layer pivot detection from attention scores) + Axis 2 (progressive sparse mask).

### Transformers are Multi-State RNNs

- **arXiv:** [2401.06104](https://arxiv.org/abs/2401.06104)

Oren et al., 2024, EMNLP 2024. Introduces TOVA (Token Omission Via Attention): a training-free KV eviction policy that retains only the tokens with highest attention scores from the most recent query. Conceptually reframes transformers as multi-state RNNs with a bounded hidden state. Achieves strong results at 1/8 cache size.

**Engine primitive:** Axis 1 (single-step attention score as eviction signal from `query()`) — Pie Tier B3. On vLLM/SGLang this requires an engine patch.

### Model Tells You What to Discard: Adaptive KV Cache Compression for LLMs

- **arXiv:** [2310.01801](https://arxiv.org/abs/2310.01801)

Ge et al., 2023, ICML 2024. Known as FastGen. Profiles attention patterns to classify heads as locality, punctuation-focus, frequency-based, or other types, then applies per-head compression policies (windowed, special-token-only, etc.). The profiling step runs once; decode uses the assigned policy per head.

**Engine primitive:** Axis 1 (attention profiling) + Axis 2 (per-head heterogeneous masks). Requires engine-level head-specific cache management.

### Keyformer: KV Cache Reduction through Key Tokens Selection for Efficient Generative Inference

- **arXiv:** [2403.09054](https://arxiv.org/abs/2403.09054)

Adnan et al., 2024. Uses a Gumbel-softmax sampling over attention scores to select key tokens, preserving differentiability. Keeps a fixed-budget cache of the most attended-to tokens plus a window. Avoids the hard top-k selection of H₂O for a smoother approximation.

**Engine primitive:** Axis 1 (attention-score-driven selection) + Axis 2 (post-selection mask).

### Ada-KV: Optimizing KV Cache Eviction by Adaptive Budget Allocation for Efficient LLM Inference

- **arXiv:** [2407.11550](https://arxiv.org/abs/2407.11550)

Feng et al., 2024, NeurIPS 2024. First method to adaptively allocate KV cache eviction budgets per attention head. Proves that redistributing budget from focused heads (which can operate with less cache) to dispersed heads (which need more) yields a lower upper bound on attention output loss. Plug-and-play on top of existing eviction policies.

**Engine primitive:** Axis 1 (per-head attention concentration analysis) + Axis 2 (head-adaptive mask). On Pie, composable with any Tier B eviction policy.

### ThinK: Thinner Key Cache by Query-Driven Pruning

- **arXiv:** [2407.21018](https://arxiv.org/abs/2407.21018)

Xu et al., 2024. Observes that only a small fraction of key *channels* (dimensions) matter for a given query. Prunes the least important channels from cached keys on a per-query basis, reducing the key cache size by up to 20% without pruning value entries. Complementary to token-level eviction.

**Engine primitive:** Axis 1 (query-driven channel importance from `query()`) — operates at the channel/dimension level rather than token level.

### LazyLLM: Dynamic Token Pruning for Efficient Long Context LLM Inference

- **arXiv:** [2407.14057](https://arxiv.org/abs/2407.14057)

Fu et al., 2024. Dynamically prunes tokens during both prefill and decode based on attention scores, computing each layer only over the tokens that layer actually attends to. Tokens pruned at one layer can be revived at a later layer. Training-free, applicable to existing models.

**Engine primitive:** Axis 1 (per-layer attention-based token selection) + Axis 2 (layer-varying token masks). Requires per-layer control flow not exposed to users on black-box servers.

### ALISA: Accelerating Large Language Model Inference via Sparsity-Aware KV Caching

- **arXiv:** [2403.17312](https://arxiv.org/abs/2403.17312)

Zhao et al., 2024. Combines token-level sparsity with a span-level strategy that groups tokens into spans and evicts whole spans, reducing management overhead. Uses an importance metric combining recency and attention frequency. Implements a three-phase strategy: prefill compression, span-based decode eviction, and KV cache offloading.

**Engine primitive:** Axis 1 (attention frequency tracking) + Axis 2 (span-level mask).

### RazorAttention: Efficient KV Cache Compression Through Retrieval Heads

- **arXiv:** [2407.15891](https://arxiv.org/abs/2407.15891)

Tang et al., 2024, ICLR 2025. Identifies a small set of "retrieval heads" that attend globally across the full context; all other heads attend locally. Applies full KV cache only to retrieval heads and a sliding window to the rest. Adds a "compensation token" summarizing dropped remote context. Training-free, 70%+ cache reduction.

**Engine primitive:** Axis 1 (head-type classification via attention analysis) + Axis 2 (head-specific mask). Related to DuoAttention but with the compensation mechanism.

### D2O: Dynamic Discriminative Operations for Efficient Long-Context Inference of Large Language Models

- **arXiv:** [2406.13035](https://arxiv.org/abs/2406.13035)

Xu et al., 2024, ICLR 2025. Dynamically and discriminatively manages KV cache at both layer and token levels. Allocates more cache budget to layers with higher global attention density. At the token level, uses a compensation mechanism that merges evicted tokens back in via similarity, rather than losing them permanently.

**Engine primitive:** Axis 1 (layer-level attention density analysis + token-level merge decisions) + Axis 2 (dynamic layer-varying mask).

### DuoAttention: Efficient Long-Context LLM Inference with Retrieval and Streaming Heads

- **arXiv:** [2410.10819](https://arxiv.org/abs/2410.10819)

Xiao et al., 2024. Classifies attention heads into "retrieval heads" (need full context) and "streaming heads" (need only recent tokens + sinks). Applies full KV to retrieval heads and constant-length KV to streaming heads, identified through an optimization procedure. 2.55× memory reduction for MHA models.

**Engine primitive:** Axis 1 (head classification) + Axis 2 (head-heterogeneous mask). Pie can express this naturally via per-head mask construction in the inferlet.

### Not All Heads Matter: A Head-Level KV Cache Compression Method with Integrated Retrieval and Reasoning

- **arXiv:** [2410.19258](https://arxiv.org/abs/2410.19258)

Fu et al., 2024, ICLR 2025. HeadKV: allocates KV cache budgets at the head level based on each head's importance for both retrieval and reasoning tasks. The HeadKV-R2 variant uses a contextual reasoning ability estimation per head. Retains as little as 1.5% of KV cache with 97% of full-cache performance.

**Engine primitive:** Axis 1 (head importance estimation) + Axis 2 (head-level budget allocation).

### CItruS: Chunked Instruction-aware State Eviction for Long Sequence Modeling

- **arXiv:** [2406.12018](https://arxiv.org/abs/2406.12018)

Xia et al., 2024. Instruction-aware eviction: compresses the KV cache by scoring tokens based on their relevance to the instruction/query, not just attention mass. Processes the context in chunks, evicting irrelevant tokens per chunk. Particularly effective for tasks where the question comes after a long context.

**Engine primitive:** Axis 1 (instruction-aware scoring via attention) + Axis 2 (per-chunk sparse mask).

### A Simple and Effective L₂ Norm-Based Strategy for KV Cache Compression

- **Title:** A Simple and Effective $L_2$ Norm-Based Strategy for KV Cache Compression
- **arXiv:** [2406.11430](https://arxiv.org/abs/2406.11430)

Devoto et al., 2024. Shows that key vectors with anomalously low L₂ norms disproportionately attract attention mass (the "attention sink" phenomenon arises from norm outliers). Proposes retaining keys with lowest norms as a simple, score-free proxy for importance — no attention computation needed for eviction.

**Engine primitive:** Minimal — only needs access to key norms, potentially implementable with Axis 4 (stateful tracking). Does not require the attention tap, making it unusually lightweight.

### ClusterKV: Manipulating LLM KV Cache in Semantic Space for Recallable Compression

- **arXiv:** [2412.03213](https://arxiv.org/abs/2412.03213)

Liu et al., 2024. Clusters KV entries by semantic similarity rather than position, enabling *recallable* compression: evicted tokens can be recalled when a new query is semantically close to them. Operates in semantic space using key embeddings. Achieves strong results with only 1–2k KV slots on 32k contexts.

**Engine primitive:** Axis 1 (query-driven recall from clusters) + Axis 3 (recallable eviction requires page management). Pie's content-addressed pages are a natural fit for semantic clusters.

### Retrieval Head Mechanistically Explains Long-Context Factuality

- **arXiv:** [2404.15574](https://arxiv.org/abs/2404.15574)

Wu et al., 2024. Mechanistic interpretability work showing that a small fraction of attention heads (retrieval heads) are responsible for long-context factual recall. Provides the theoretical grounding for RazorAttention and DuoAttention's head classification. Demonstrates that these heads are universal across models.

**Engine primitive:** Foundational analysis — informs which heads need full KV in head-aware eviction schemes (Axis 1 + 2).

### SampleAttention: Near-Lossless Acceleration of Long Context LLM Inference with Adaptive Structured Sparse Attention

- **arXiv:** [2406.15486](https://arxiv.org/abs/2406.15486)

Tang et al., 2024. Uses a two-stage approach: a lightweight "sample" of attention scores to identify important KV positions, then full attention only on those positions. Combines head-specific patterns (some heads attend locally, others globally) with adaptive sampling. Near-lossless at 5× decode speedup.

**Engine primitive:** Axis 1 (query-driven sampling) + Axis 2 (adaptive sparse mask).

### Deja Vu: Contextual Sparsity for Efficient LLMs at Inference Time

- **arXiv:** [2310.17157](https://arxiv.org/abs/2310.17157)

Liu et al., 2023, ICML 2023. Discovers that LLM activations exhibit contextual sparsity: for a given input, only a small fraction of attention heads and MLP neurons are important. Predicts the sparse set using a small MLP predictor trained offline. Achieves near-exact inference with ~75% sparsity.

**Engine primitive:** Axis 1 (head/neuron selection per input) + Axis 2 (head-level masking). The MLP predictor could run as a PTIR program on Pie using `query()`.

---

## KV cache merging

### Model Tells You Where to Merge: Adaptive KV Cache Merging for LLMs on Long-Context Tasks

- **arXiv:** [2407.08454](https://arxiv.org/abs/2407.08454)

Wang et al., 2024. KVMerger: identifies merging sets of similar KV entries using cosine similarity, then merges them with Gaussian kernel weighting. Unlike eviction (which loses information), merging preserves a weighted summary. Outperforms H₂O and eviction-based methods at 35–50% cache budgets.

**Engine primitive:** Axis 1 (attention-driven similarity analysis) + Axis 4 (maintaining merge state across steps). Merging requires page-level operations (Axis 3) to restructure the KV cache.

---

## Query-aware and retrieval-style sparse attention

### Quest: Query-Aware Sparsity for Efficient Long-Context LLM Inference

- **arXiv:** [2406.10774](https://arxiv.org/abs/2406.10774)

Tang et al., 2024. Each KV page maintains min/max key statistics. At decode time, the current query scores pages via these statistics; only top-scoring pages are loaded into attention. A natural fit for page-based KV stores. 7× decode speedup on long sequences.

**Engine primitive:** Axis 1 (`query()` is essential — the entire method is query-driven page selection) + Axis 2 (page-level sparse mask). Pie's KV is already paged and the attention mask is already a guest-bound channel, making Quest the most natural first user of `query()` (see `09`, B4).

### MInference 1.0: Accelerating Pre-filling for Long-Context LLMs via Dynamic Sparse Attention

- **arXiv:** [2407.02490](https://arxiv.org/abs/2407.02490)

Jiang et al., 2024. Identifies three dominant sparse patterns in long-context prefill: A-shape (attention sinks + diagonal), vertical-slash, and block-sparse. Assigns each head its dominant pattern and applies a pattern-specific sparse kernel during prefill. Up to 10× prefill speedup on 1M-token inputs.

**Engine primitive:** Axis 2 (per-head pattern-specific masks). The pattern assignment is static per model; the mask application requires custom kernels or a programmable mask channel.

### InfLLM: Training-Free Long-Context Extrapolation for LLMs with an Efficient Context Memory

- **arXiv:** [2402.04617](https://arxiv.org/abs/2402.04617)

Xiao et al., 2024, NeurIPS 2024. Offloads distant KV blocks to CPU memory, then selectively retrieves the most relevant blocks per query token. Combines a sliding window for local attention with block-level retrieval for distant context. Enables LLMs pretrained on 2k tokens to handle 1M+ token contexts.

**Engine primitive:** Axis 1 (`query()` to score blocks for retrieval) + Axis 2 (selective block mask) + Axis 3 (page-level offload and retrieval — maps to Pie's page operations). This is effectively Quest with CPU offloading.

### RetrievalAttention: Accelerating Long-Context LLM Inference via Vector Retrieval

- **arXiv:** [2409.10516](https://arxiv.org/abs/2409.10516)

Liu et al., 2024. Builds an approximate nearest-neighbor (ANN) index over cached keys and retrieves the most relevant keys per query using vector search (GPU for local, CPU for remote). Reduces decode from O(n) to O(√n) in context length. Addresses the out-of-distribution gap between key and query distributions.

**Engine primitive:** Axis 1 (`query()` drives the ANN lookup) + Axis 2 (retrieved-subset mask). Pie Tier B5 (see `09`).

### Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention

- **arXiv:** [2502.11089](https://arxiv.org/abs/2502.11089)

Yuan et al., 2025 (DeepSeek). NSA: trains the model with a built-in sparse attention mechanism that combines compressed global attention, selected block-sparse attention, and sliding window attention. Hardware-aligned block structure for efficient GPU execution. Unlike post-hoc methods, the model learns which blocks to attend to.

**Engine primitive:** Axis 2 (the sparse pattern is model-intrinsic but requires a mask that encodes the block selection). When applied at inference on a pretrained NSA model, the serving system must support the compound mask structure.

### MoBA: Mixture of Block Attention for Long-Context LLMs

- **arXiv:** [2502.13189](https://arxiv.org/abs/2502.13189)

Lu et al., 2025 (Moonshot AI). Applies the mixture-of-experts principle to attention: partitions context into blocks, and each query token routes to the most relevant blocks via a learned gating mechanism. Achieves 6.5× prefill speedup on 1M-token sequences. Can smoothly transition between sparse and full attention.

**Engine primitive:** Axis 2 (dynamic block-selection mask). The gating mechanism could run as a PTIR program on Pie using `query()` (Axis 1).

### SparQ Attention: Bandwidth-Efficient LLM Inference

- **arXiv:** [2312.04985](https://arxiv.org/abs/2312.04985)

Ribar et al., 2023. Reduces memory bandwidth during attention by first computing approximate attention scores using only the top-r components of the query (cheap inner product), then fetching full KV entries only for the top-k scoring positions. Achieves up to 8× bandwidth reduction with minimal quality loss.

**Engine primitive:** Axis 1 (`query()` decomposition and approximate scoring). The two-pass approach requires control over the attention computation itself.

---

## KV compression and quantization

### KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache

- **arXiv:** [2402.02750](https://arxiv.org/abs/2402.02750)

Liu et al., 2024. Asymmetric quantization: keys are quantized per-channel (because key magnitudes vary across channels) and values per-token (because value magnitudes vary across tokens). Achieves near-lossless 2-bit KV quantization without any calibration data or fine-tuning. Simple and deployment-ready.

**Engine primitive:** None beyond standard serving — this is a runtime optimization. Currently built into vLLM. On Pie, could be a system-level optimization beneath the inferlet layer.

### KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization

- **arXiv:** [2401.18079](https://arxiv.org/abs/2401.18079)

Hooper et al., 2024. Achieves extreme KV compression (down to 3-bit) via per-channel quantization of keys with pre-RoPE treatment, non-uniform quantization codebooks, and per-vector dense-and-sparse decomposition for outliers. Enables 10M context lengths on a single GPU.

**Engine primitive:** Runtime optimization. The pre-RoPE key quantization requires engine-level integration.

### GEAR: An Efficient KV Cache Compression Recipe for Near-Lossless Generative Inference of LLM

- **arXiv:** [2403.05527](https://arxiv.org/abs/2403.05527)

Kang et al., 2024. Three-component compression: ultra-low-precision quantization for the bulk, low-rank matrix approximation for quantization residuals, and a sparse matrix for individual outliers. Achieves near-lossless 4-bit compression; outperforms uniform quantization at 2-bit.

**Engine primitive:** Runtime optimization. The low-rank + sparse residual structure requires custom kernel support.

### No Token Left Behind: Reliable KV Cache Compression via Importance-Aware Mixed Precision Quantization

- **arXiv:** [2402.18096](https://arxiv.org/abs/2402.18096)

Duanmu et al., 2024. Mixed-precision KV quantization: assigns higher bit-width to important tokens (identified by attention scores) and lower to the rest. Prevents the quality collapse that uniform low-bit quantization causes on outlier tokens.

**Engine primitive:** Axis 1 (attention-score-driven importance classification) for the mixed-precision assignment. Once assigned, execution is a runtime optimization.

### ZipCache: Accurate and Efficient KV Cache Quantization with Salient Token Identification

- **arXiv:** [2405.14256](https://arxiv.org/abs/2405.14256)

He et al., 2024. Identifies salient tokens (attention sinks and high-attention positions) and stores them at higher precision while aggressively quantizing the rest. Combines channel-level and token-level mixed precision. Normalizes KV entries before quantization to reduce range.

**Engine primitive:** Axis 1 (saliency detection) for precision assignment; otherwise a runtime optimization.

### SKVQ: Sliding-window Key and Value Cache Quantization for Large Language Models

- **arXiv:** [2405.06219](https://arxiv.org/abs/2405.06219)

Duanmu et al., 2024. Quantizes the full KV cache to very low precision (e.g., 2-bit) while keeping a sliding window of recent tokens at full precision. Leverages the observation that recent tokens dominate attention and need high fidelity. Combines naturally with attention-sink methods.

**Engine primitive:** Axis 2 (window-aware quantization boundary). The sliding window of high-precision tokens mirrors the StreamingLLM mask pattern.

### Dynamic Memory Compression: Retrofitting LLMs for Accelerated Inference

- **arXiv:** [2403.09636](https://arxiv.org/abs/2403.09636)

Nawrot et al., 2024, ICML 2024. Learns to merge consecutive KV entries within each attention head by training a small gating module. At inference, the gate decides whether to append a new KV entry or merge it into the previous one, achieving up to 7× compression. Requires a short fine-tuning phase.

**Engine primitive:** Axis 1 (the gating decision is attention-derived) + Axis 4 (per-step merge state). The gating module could be a PTIR program.

### SubGen: Token Generation in Sublinear Time and Memory

- **arXiv:** [2402.06082](https://arxiv.org/abs/2402.06082)

Zandieh et al., 2024. Uses online clustering of key embeddings and ℓ₂-sampling of value entries to compress the KV cache with provable error bounds. Achieves sublinear memory and time complexity per token. Theoretical guarantees distinguish it from heuristic methods.

**Engine primitive:** Runtime optimization with streaming clustering state (Axis 4). Requires control over the attention computation.

---

## Cross-layer KV sharing and structural compression

### You Only Cache Once: Decoder-Decoder Architectures for Language Models

- **arXiv:** [2405.05254](https://arxiv.org/abs/2405.05254)

Sun et al., 2024. YOCO: an architectural change where only the first decoder stack produces KV entries; the second decoder stack cross-attends to the first stack's KV cache. Halves KV memory at the architecture level. Applicable at inference when serving YOCO-pretrained models.

**Engine primitive:** At inference, requires the serving system to support cross-decoder attention (a non-standard attention topology). Axis 2 (the cross-attention pattern) + engine support for the dual-decoder topology.

### Reducing Transformer Key-Value Cache Size with Cross-Layer Attention

- **arXiv:** [2405.12981](https://arxiv.org/abs/2405.12981)

Brandon et al., 2024. CLA (Cross-Layer Attention): shares KV entries across adjacent layers — layer l reuses the KV computed at layer l−1 (or l−2). Halves or quarters the KV memory with minimal quality loss when trained from scratch. At inference, the serving system stores one KV entry per sharing group.

**Engine primitive:** Engine must support cross-layer KV sharing in its cache layout. Not expressible as a user policy on any current serving system.

### DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model

- **arXiv:** [2405.04434](https://arxiv.org/abs/2405.04434)

Bi et al., 2024. Introduces Multi-head Latent Attention (MLA): instead of caching full key and value matrices, caches a compressed latent vector and reconstructs K/V on-the-fly via low-rank projection. Reduces per-token KV cache to ~5% of standard GQA. The compression is a model architecture choice; inference requires the engine to support the reconstruct-on-decode pattern.

**Engine primitive:** The serving system must support MLA's compressed KV format and on-the-fly reconstruction. This is an architectural integration, not a user-policy decision.

### MiniCache: KV Cache Compression in Depth Dimension for Large Language Models

- **arXiv:** [2405.14366](https://arxiv.org/abs/2405.14366)

Liu et al., 2024. Observes that KV representations across adjacent layers are highly similar in deep models. Merges KV entries across the depth dimension (across layers), storing shared representations plus small per-layer residuals. Compresses cache in the layer dimension rather than the token dimension.

**Engine primitive:** Engine must support cross-layer KV merging. Requires custom cache layout.

### Layer-Condensed KV Cache for Efficient Inference of Large Language Models

- **arXiv:** [2405.10637](https://arxiv.org/abs/2405.10637)

Wu et al., 2024, ACL 2024. Computes KV pairs only at selected layers and has other layers reuse them. Trains the model with this pattern so it learns to compensate. At inference, only the selected layers contribute new KV entries, reducing cache by up to 75%.

**Engine primitive:** Engine-level cross-layer cache sharing. Requires training with the pattern.

### Effectively Compress KV Heads for LLM

- **arXiv:** [2406.07056](https://arxiv.org/abs/2406.07056)

Yuan et al., 2024. Analyzes which GQA (Grouped Query Attention) heads can be further shared or merged without quality loss. Proposes head-pruning and inter-head merging strategies to reduce the number of distinct KV heads below the GQA group count. Training-free compression.

**Engine primitive:** Engine-level head merging. Requires modifying the KV cache head layout.

---

## Prefix and prompt cache reuse

### Efficient Memory Management for Large Language Model Serving with PagedAttention

- **arXiv:** [2309.06180](https://arxiv.org/abs/2309.06180)

Kwon et al., 2023, SOSP 2023. vLLM and PagedAttention: borrows virtual-memory paging for KV cache, allowing non-contiguous physical storage of logical KV sequences. Enables automatic prefix caching (when two requests share a prefix, their KV pages are shared) and near-zero memory waste. The foundational systems work for modern LLM serving.

**Engine primitive:** Pie builds on the same paging insight but goes further: content-addressed pages, explicit user-visible fork/snapshot/truncate, and page-level guest control. vLLM's prefix caching is automatic and opaque; Pie's is explicit and addressable (Axis 3).

### SGLang: Efficient Execution of Structured Language Model Programs

- **arXiv:** [2312.07104](https://arxiv.org/abs/2312.07104)

Zheng et al., 2023. RadixAttention: organizes the KV cache as a radix tree keyed by token sequences, enabling automatic sharing of any common prefix across requests (not just the longest prefix match). The tree is managed by the runtime; users express sharing implicitly through their program structure.

**Engine primitive:** The radix tree is automatic (Tier 1 — commodity). Not user-programmable; users cannot control which branches to retain or evict. Pie's content-addressed pages give explicit control (Axis 3).

### Prompt Cache: Modular Attention Reuse for Low-Latency Inference

- **arXiv:** [2311.04934](https://arxiv.org/abs/2311.04934)

Gim et al., 2023. Pre-computes KV caches for reusable prompt modules (system prompts, document schemas, few-shot examples) and reuses them across requests. Requires attention positional encoding to be compatible with non-contiguous reuse (position remapping). Achieves 8× latency reduction for modular prompts.

**Engine primitive:** Requires position manipulation at inference (RoPE remapping for non-prefix reuse) plus content-addressed cache storage. Pie's snapshot save/open plus position control in the mask channel are the relevant primitives (Axis 2 + 3).

### CacheBlend: Fast Large Language Model Serving for RAG with Cached Knowledge Fusion

- **arXiv:** [2405.16444](https://arxiv.org/abs/2405.16444)

Yao et al., 2024, EuroSys 2025. Addresses non-prefix KV reuse for RAG: when cached KV chunks appear at arbitrary positions in a new prompt, naïve reuse ignores cross-chunk attention dependencies. CacheBlend selectively recomputes a small subset of tokens to recover inter-chunk attention, achieving near-full-prefill quality at 2–3× lower TTFT.

**Engine primitive:** Axis 3 (non-prefix cache composition requires explicit page management and selective recomputation) plus position handling. This is impossible on prefix-only caching systems. Pie's content-addressed pages and explicit working-set operations are a direct fit.

### CacheGen: KV Cache Compression and Streaming for Fast Large Language Model Serving

- **arXiv:** [2310.07240](https://arxiv.org/abs/2310.07240)

Liu et al., 2023, SIGCOMM 2024. Compresses KV caches for network streaming between prefill and decode nodes. Uses a custom codec that exploits the statistical structure of KV entries (delta encoding, quantization-aware compression) to achieve high compression ratios. Enables KV cache transfer across disaggregated systems.

**Engine primitive:** A systems optimization for disaggregated serving. Relevant to KV offloading and streaming architectures.

### Compute Or Load KV Cache? Why Not Both?

- **arXiv:** [2410.03065](https://arxiv.org/abs/2410.03065)

Hu et al., 2024. Analyzes the trade-off between recomputing KV from scratch vs loading cached KV from storage, and proposes a hybrid: compute KV for some layers while loading for others, overlapping the two. Optimizes the partition to minimize end-to-end latency.

**Engine primitive:** Engine-level scheduling that interleaves compute and I/O per layer. Requires system-level control over the KV pipeline.

---

## Disaggregated prefill/decode and KV offloading

### DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving

- **arXiv:** [2401.09670](https://arxiv.org/abs/2401.09670)

Zhong et al., 2024, OSDI 2024. Separates prefill (compute-bound) and decode (memory-bound) onto different GPU pools, transferring KV caches between them. Optimizes the prefill/decode ratio for goodput. The architectural choice directly affects cache management: KV must be serializable and transferable.

**Engine primitive:** Systems-level disaggregation. Pie's page-based KV model simplifies transfer (pages are self-contained). Axis 7 (compute allocation policy) becomes relevant when the inferlet can influence prefill/decode placement.

### Splitwise: Efficient generative LLM inference using phase splitting

- **arXiv:** [2311.18677](https://arxiv.org/abs/2311.18677)

Patel et al., 2023, ISCA 2024. Independently proposes prefill/decode disaggregation. Assigns prompt computation to high-compute GPUs and token generation to cost-effective memory-optimized hardware. Demonstrates 1.4× throughput at 20% lower cost.

**Engine primitive:** Systems-level. Same KV transfer requirements as DistServe.

### Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving

- **arXiv:** [2407.00079](https://arxiv.org/abs/2407.00079)

Qin et al., 2024. Designs the entire serving architecture around the KV cache as the central data structure, with disaggregated prefill, decode, and caching tiers. KV caches are stored in a distributed cache pool (DRAM + SSD) and transferred to decode nodes on demand. The architecture makes caching a first-class citizen.

**Engine primitive:** Strongly aligned with Pie's philosophy of KV as a programmable data structure (Axis 3). Mooncake's cache pool maps conceptually to Pie's content-addressed page store.

### FlexGen: High-Throughput Generative Inference of Large Language Models with a Single GPU

- **arXiv:** [2303.06865](https://arxiv.org/abs/2303.06865)

Sheng et al., 2023, ICML 2023. Offloads model weights, KV cache, and activations across GPU, CPU, and disk with an LP-based scheduling policy to maximize throughput. The KV offloading strategy is an algorithmic choice: which layers' KV to keep on GPU vs CPU vs disk. Pioneered the KV offloading paradigm.

**Engine primitive:** Axis 7 (the offloading schedule is a compute/memory allocation policy). Pie's page model could expose offloading as a guest-visible operation.

### DéjàVu: KV-cache Streaming for Fast, Fault-tolerant Generative LLM Serving

- **Title:** DéjàVu: KV-cache Streaming for Fast, Fault-tolerant Generative LLM Serving
- **arXiv:** [2403.01876](https://arxiv.org/abs/2403.01876)

Strati et al., 2024. Streams KV caches from prefill nodes to decode nodes using a GPU-initiated, CPU-assisted transfer protocol. Adds fault tolerance by maintaining KV replicas. The streaming protocol co-designs with the paging structure for efficiency.

**Engine primitive:** Systems-level KV streaming. Compatible with page-based KV architectures.

---

## Long-context inference algorithms

### Ring Attention with Blockwise Transformers for Near-Infinite Context

- **arXiv:** [2310.01889](https://arxiv.org/abs/2310.01889)

Liu et al., 2023, ICLR 2024. Distributes the attention computation across multiple hosts in a ring topology: each host processes a block of keys/values, and queries circulate around the ring. Enables arbitrary context lengths limited only by the number of hosts, with memory per host constant.

**Engine primitive:** Systems-level distributed attention. The blockwise decomposition is transparent to the model. On Pie, this would be an engine-internal optimization.

### Landmark Attention: Random-Access Infinite Context Length for Transformers

- **arXiv:** [2305.16300](https://arxiv.org/abs/2305.16300)

Mohtashami & Jaggi, 2023, NeurIPS 2023. Inserts special "landmark" tokens to represent each block of the input. The model attends to landmark tokens to select relevant blocks, then attends within those blocks. Preserves random-access attention (unlike sliding-window-only methods) while bounding memory.

**Engine primitive:** Axis 2 (the two-level attention pattern requires a custom mask) + Axis 3 (block-level retrieval maps to page operations). The landmark tokens must be injected into the KV cache.

### Training-Free Long-Context Scaling of Large Language Models

- **arXiv:** [2402.17463](https://arxiv.org/abs/2402.17463)

Chen et al., 2024, ICML 2024. Dual Chunk Attention (DCA): splits the input into chunks and decomposes attention into intra-chunk (local) and inter-chunk (global) components with separate position encodings. Extends the effective context window to 100k+ tokens without fine-tuning. Integrates with FlashAttention and vLLM.

**Engine primitive:** Axis 2 (the dual intra/inter-chunk mask structure) plus position manipulation (separate RoPE for each component). On a black-box server, both the mask and position rewrite require engine patches.

### Leave No Context Behind: Efficient Infinite Context Transformers with Infini-attention

- **arXiv:** [2404.07143](https://arxiv.org/abs/2404.07143)

Munkhdalai et al., 2024. Augments standard attention with a compressive memory that summarizes past context into a fixed-size state using linear attention (like a recurrent update). Each attention layer has both local dot-product attention and a long-range compressed memory, gated together. Enables infinite context with bounded memory.

**Engine primitive:** Axis 2 (the local/memory gating mask) plus recurrent-state working sets (Pie's `RsWorkingSet` for hybrid models). The compressive memory update is a per-step stateful operation (Axis 4).

### LLM Maybe LongLM: Self-Extend LLM Context Window Without Tuning

- **arXiv:** [2401.01325](https://arxiv.org/abs/2401.01325)

Jin et al., 2024, ICML 2024. Self-Extend: at inference, maps far-away token positions to nearby positions using a group-based remapping, while keeping local positions unchanged. Decomposes attention into a local window (normal positions) and a group-level global attention (remapped positions). Training-free.

**Engine primitive:** Position modification at inference (RoPE remapping) + Axis 2 (the local/global dual mask). On a black-box server, position remapping requires engine patches.

### YaRN: Efficient Context Window Extension of Large Language Models

- **arXiv:** [2309.00071](https://arxiv.org/abs/2309.00071)

Peng et al., 2023, ICLR 2024. "Yet another RoPE extensioN method": selectively interpolates RoPE frequencies by dimension (low/mid/high frequency bands get different treatment) and adds an attention temperature correction. 10× more data-efficient than previous context extension methods. The dominant context extension technique in practice (used in Llama 3.1, Qwen 3, Mistral).

**Engine primitive:** Position/RoPE modification at inference. The frequency-band interpolation is applied to the rotary embeddings during the forward pass. On most serving systems this is a configuration parameter, not a user-programmable feature.

### Extending Context Window of Large Language Models via Positional Interpolation

- **arXiv:** [2306.15595](https://arxiv.org/abs/2306.15595)

Chen et al., 2023, Meta AI. The original positional interpolation method: linearly downscales position indices so that the extended context maps into the trained position range. Requires minimal fine-tuning. Foundational technique that YaRN, NTK-aware scaling, and Self-Extend build upon.

**Engine primitive:** Position modification at inference. The simplest form of RoPE scaling — a single scaling factor applied to all positions.

### Extending Context Window of Large Language Models from a Distributional Perspective

- **arXiv:** [2410.01490](https://arxiv.org/abs/2410.01490)

Wu et al., 2024. Analyzes context extension from an attention-score distribution perspective. Shows that existing methods (PI, YaRN) can cause distribution shift in attention patterns and proposes corrections. Provides theoretical grounding for why some scaling methods work better than others.

**Engine primitive:** Position modification analysis. Informs the design of RoPE scaling strategies.

### MemGPT: Towards LLMs as Operating Systems

- **arXiv:** [2310.08560](https://arxiv.org/abs/2310.08560)

Packer et al., 2023, NeurIPS 2023. Treats the LLM context window as "main memory" and implements virtual memory management with page-in/page-out between the context and external storage. The LLM itself decides when to evict or retrieve context pages via function calls. An agentic approach to long-context management.

**Engine primitive:** Axis 3 (explicit context page management) + Axis 8 (tool calls interleaved with generation for page operations). On a black-box server, the LLM cannot manage its own KV cache. On Pie, the inferlet can explicitly manage pages and call storage.

---

## Context compression and editing

### LLMLingua: Compressing Prompts for Accelerated Inference of Large Language Models

- **arXiv:** [2310.05736](https://arxiv.org/abs/2310.05736)

Jiang et al., 2023, EMNLP 2023. Coarse-to-fine prompt compression: a budget controller identifies salient prompt segments, then an iterative token-level algorithm removes redundant tokens using a small model's perplexity as a proxy for importance. Up to 20× compression with minimal quality loss.

**Engine primitive:** Preprocessing step — runs before the main forward pass. Can be implemented as a host-side program or, on Pie, as an inferlet that compresses context before main generation.

### LongLLMLingua: Accelerating and Enhancing LLMs in Long Context Scenarios via Prompt Compression

- **arXiv:** [2310.06839](https://arxiv.org/abs/2310.06839)

Jiang et al., 2023, ACL 2024. Extends LLMLingua for long-context scenarios with question-aware compression: ranks document chunks by relevance to the question, then compresses within chunks. Adds a reordering strategy to place important content closer to the question (mitigating the "lost in the middle" problem).

**Engine primitive:** Same as LLMLingua — preprocessing. The question-aware ranking could use the `query()` tap (Axis 1) for on-device scoring.

### LLMLingua-2: Data Distillation for Efficient and Faithful Task-Agnostic Prompt Compression

- **arXiv:** [2403.12968](https://arxiv.org/abs/2403.12968)

Pan et al., 2024, ACL 2024. Distills the compression capability into a small BERT-like model trained on data annotated by GPT-4 for token-level importance. Faster than LLMLingua (avoids autoregressive perplexity computation) and more faithful (trained explicitly for compression quality).

**Engine primitive:** Preprocessing. The compressed output is fed to the main model's standard forward pass.

### Learning to Compress Prompts with Gist Tokens

- **arXiv:** [2304.08467](https://arxiv.org/abs/2304.08467)

Mu et al., 2023, NeurIPS 2023. Trains a model to compress any prompt into a small set of "gist tokens" whose KV representations can be cached and reused. Gisting is learned by modifying the attention mask during instruction tuning: after the gist tokens, subsequent tokens cannot attend before them. Up to 26× compression.

**Engine primitive:** Axis 2 (the gist-token attention mask pattern is non-standard) + Axis 3 (caching and reusing gist-token KV representations as pages). The gist KV entries are a form of learned prefix cache.

### Adapting Language Models to Compress Contexts

- **arXiv:** [2305.14788](https://arxiv.org/abs/2305.14788)

Chevalier et al., 2023, EMNLP 2023. AutoCompressor: recursively compresses long contexts by processing them in segments, accumulating "summary vectors" that are prepended to the next segment as soft prompts. Enables a 30k context to be compressed into 50 summary tokens. Requires fine-tuning the model to use summary vectors.

**Engine primitive:** Axis 2 (attention mask must incorporate summary vectors) + Axis 3 (summary vectors are cached KV entries). The recursive segment processing requires multi-step orchestration.

### Compressing Context to Enhance Inference Efficiency of Large Language Models

- **arXiv:** [2310.06201](https://arxiv.org/abs/2310.06201)

Li et al., 2023, EMNLP 2023. Selective Context: uses self-information (negative log probability under a small LM) to score each token/sentence, then drops low-information content. Achieves 50% context reduction with 36% memory savings and minimal quality loss. The simplest context compression method.

**Engine primitive:** Preprocessing. The self-information scoring could run as a PTIR program on Pie (Axis 1 with `logits()`), making it an on-device context compression step.

---

## Benchmarks and surveys

### SCBench: A KV Cache-Centric Analysis of Long-Context Methods

- **arXiv:** [2412.10319](https://arxiv.org/abs/2412.10319)

He et al., 2024. Benchmarks KV cache methods across the cache lifecycle: generation (full attention, sparse attention), compression (eviction, quantization, merging), retrieval (offloading, selection), and loading (prefix caching). Tests methods under shared-context and multi-turn scenarios where methods often degrade.

**Engine primitive:** Benchmark. Notable finding: most eviction methods degrade significantly under multi-turn and shared-context scenarios — exactly the scenarios where Pie's explicit fork/snapshot (Axis 3) provides the most benefit.

### KV Cache Compression, But What Must We Give in Return? A Comprehensive Benchmark of Long Context Capable Approaches

- **arXiv:** [2407.01527](https://arxiv.org/abs/2407.01527)

Yang et al., 2024. Systematically evaluates eviction (H₂O, SnapKV, TOVA), quantization (KIVI), and architectural approaches across 7 benchmarks and 8 models. Finds that eviction methods degrade faster on retrieval-heavy tasks than on reasoning tasks, and that head-aware methods significantly outperform uniform ones.

**Engine primitive:** Benchmark. Informs the design of eviction policies.

### A Survey on Large Language Model Acceleration based on KV Cache Management

- **arXiv:** [2412.19442](https://arxiv.org/abs/2412.19442)

Gao et al., 2024, TMLR 2025. Comprehensive survey covering eviction/pruning, merging, selection, compression, quantization, and low-rank decomposition approaches to KV cache management. Provides a taxonomy and links to code for recent work.

**Engine primitive:** Survey. Covers the full landscape of methods discussed in this chapter.

### FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving

- **arXiv:** [2501.01005](https://arxiv.org/abs/2501.01005)

Ye et al., 2025, MLSys 2025. A kernel library for composable attention with support for variable-length sequences, page-based KV cache, and custom attention variants (sliding window, sinks, tree attention). Provides the systems substrate on which many methods in this chapter can be efficiently executed.

**Engine primitive:** Systems infrastructure. FlashInfer's composable kernels are complementary to Pie's programmable mask channel — Pie provides the user-facing API, FlashInfer (or similar) provides the kernel.

---

## Citation audit

All arXiv IDs above were verified by fetching the corresponding abstract page from `arxiv.org`. The following notes document deliberate heading-vs-title differences and any caveats:

1. **2401.06104** — arXiv title is "Transformers are Multi-State RNNs"; TOVA is the method name introduced within the paper. Heading uses the arXiv title.
2. **2310.01801** — arXiv title is "Model Tells You What to Discard: Adaptive KV Cache Compression for LLMs"; commonly known as FastGen.
3. **2410.19258** — arXiv title is "Not All Heads Matter: A Head-Level KV Cache Compression Method with Integrated Retrieval and Reasoning"; commonly known as HeadKV.
4. **2402.17463** — arXiv title is "Training-Free Long-Context Scaling of Large Language Models"; the method is commonly known as Dual Chunk Attention / ChunkLlama.
5. **2309.06180** — arXiv title is "Efficient Memory Management for Large Language Model Serving with PagedAttention"; commonly known as vLLM.
6. **2312.07104** — arXiv title is "SGLang: Efficient Execution of Structured Language Model Programs"; RadixAttention is the KV sharing mechanism within SGLang.
7. **2405.04434** — arXiv title is "DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model"; Multi-head Latent Attention (MLA) is described within.
8. **2310.06201** — arXiv title is "Compressing Context to Enhance Inference Efficiency of Large Language Models"; commonly known as Selective Context.
9. **CaM (Cache Merging for Memory-efficient LLMs Inference)** — published at ICML 2024 (PMLR 235:58840-58850) without an arXiv preprint. No arXiv link is provided; cited only in the text discussion of merging methods.
