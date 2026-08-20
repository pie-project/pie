//! `llama_like`'s per-backend binding facts.
//!
//! The SHAPE lives in `../spec.rs`. What a deployment BOUND -- a fused bank,
//! a kernel's padded head dim, a TP width -- is per-backend and known only
//! when that backend's aspect is compiled, so it stays here.

use serde::{Deserialize, Serialize};
use model_dsl::WeightRepr;
pub use model_ir::facts::{NormPlacement, QkNorm};

/// The shape, re-exported so a declaration reaches its facts from one place.
pub use super::super::spec::LlamaLikeFacts;

/// CUDA backend facts for a LOWERED llama_like trace
/// (`family::llama_like_cuda`; north-star-dsl.md).
///
/// All load-time: env defaults, kernel-support predicates over the head
/// geometry, what the deployment's binding materialized. The driver VALIDATES
/// at boot that its own derivation agrees rather than choosing.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LlamaLikeCudaFacts {
    /// XQA decode eligibility: `PIE_CUDA_XQA_DECODE` (default on) &&
    /// `xqa_decode_bf16_supported(heads, head_dim_kernel, page_size,
    /// window)` && all-full-attention && native-bf16 cache && !HND layout
    /// (context.cpp:1419-1425, llama_like.cpp:693-701).
    pub xqa_decode: bool,
    /// The fused decode-QKV epilogue is live: `decode_fused_post_enabled`
    /// (env, default on) && native-bf16 cache && unpadded head_dim && no qkv
    /// bias (declared_forward.cpp:465-479). The trace-time terms live on
    /// [`LlamaLikeFacts`] and the declaration checks both.
    pub decode_fused_post: bool,
    /// The workspace carries a rope table (`ws.rope_table` non-empty), so
    /// the fused arm's first layer states [`model_ir::trace::OpKind::RopeTableBuild`];
    /// without it the fused kernel derives cos/sin from theta in-kernel
    /// and no table launch exists.
    pub rope_table: bool,
    /// FlashInfer's decode kernel set lacks this model's GQA ratio
    /// (`!flashinfer_decode_supports_gqa`, context.cpp:1413-1414): decode fires
    /// fall back to [`model_ir::trace::AttnKernel::PrefillDequantDecode`]. XQA,
    /// when eligible, overrides this (context.cpp:1427).
    pub force_prefill_path: bool,
    /// The attention kernels run at a padded `head_dim_kernel` wider than the
    /// logical head dim (Phi-3-mini: 96 -> 128). The generated form stages
    /// zero-padded q/k/v copies around the KV write, overrides the softmax
    /// scale to `1/sqrt(d)`, and strips the attention output.
    #[serde(default)]
    pub head_dim_padded: bool,
    /// The head width the ATTENTION kernels run at (Phi-3-mini: 128 for a
    /// logical 96), or 0 for a deployment that runs at the logical one.
    ///
    /// [`Self::head_dim_padded`] is exactly `head_dim_kernel != 0`; the WIDTH is
    /// what a statement needs (`cuda::pad_head_dim` gives `heads * head_dim_kernel`).
    #[serde(default)]
    pub head_dim_kernel: u32,
    /// The checkpoint materialised a packed gate‖up bank
    /// (`w.layers[l].gate_up_proj_fused != nullptr`), so the MLP's packed GEMM
    /// lands in one buffer and the activation is the CHUNKED swiglu over it;
    /// without it the projection writes two buffers and takes the pair form.
    #[serde(default)]
    pub gate_up_fused: bool,
    /// How this deployment STORES its linear projections
    /// ([`model_dsl::WeightRepr`]).
    ///
    /// ONE repr for the whole deployment: a checkpoint quantizes uniformly and
    /// the build gate refuses a mixed binding by name.
    #[serde(default)]
    pub proj_repr: WeightRepr,
    /// How many ranks this deployment shards its layers across
    /// (`LlamaLikeForwardCfg::tp_size`), or 0/1 for a single GPU.
    ///
    /// Narrows the projection widths and decides whether the recombining
    /// launches (`dist::all_reduce_bf16` and its two friends) exist at all.
    #[serde(default)]
    pub tp_size: u32,
    /// The SLIDING WINDOW each layer attends over, `-1` for none: a config's
    /// `sliding_window`, or its per-layer list where the architecture
    /// alternates (OLMo-3, Mistral).
    ///
    /// Empty means every layer is `-1`, which is why texts read the accessor
    /// ([`Self::window_left_at`]). The per-FIRE override (`runtime_window_left`)
    /// is NOT this: it is a runtime input and wants a guard predicate.
    #[serde(default)]
    pub window_left: Vec<i32>,
    /// Rows below which an all-reduce takes the NVLink P2P kernel instead of
    /// NCCL, or 0 for a deployment that always takes NCCL.
    ///
    /// ZERO also covers the deployment that registered no P2P buffers, since
    /// the kernel reads only registered memory.
    #[serde(default)]
    pub all_reduce_p2p_max_rows: u32,
}

/// WHICH gated activation a deployment takes. See
/// [`LlamaLikeMetalFacts::activation`].
///
/// UNVERIFIED (2026-08-05): the Metal driver cannot build on the box we have,
/// so every field of [`LlamaLikeMetalFacts`] is read off SOURCE, not measured.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub enum Activation {
    /// `silu_mul` — `silu(gate) * up`, every llama-like deployment's.
    #[default]
    SiluMul,
    /// `gptoss_swiglu` — the gate clamped ABOVE only, the linear branch
    /// clamped both ways and carrying a `+1`.
    SwiGlu {
        /// The clamp.
        limit: f32,
        /// The sigmoid's slope. NOT from a config: it is part of the
        /// activation the way `silu`'s sigmoid is.
        alpha: f32,
    },
    /// `geglu_tanh` — gemma's, and the gelu is the TANH approximation.
    Geglu,
}

/// The routed GEMM's tile when a deployment file does not name one.
///
/// See [`LlamaLikeMetalFacts::moe_tile`]. A free function because
/// `#[serde(default = ...)]` takes a path and a const cannot be one.
fn default_moe_tile() -> Option<(u32, u32)> {
    Some(crate::shared::llama_like::project::ROUTED_QMM_TILE)
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LlamaLikeMetalFacts {
    /// The projection GEMV folds the block residual in its epilogue
    /// (`affine_qmv_fast_residual`, `Dispatch::fuse_residual`, `PIE_FUSE_RESIDUAL`),
    /// so a `beta_one` matmul states one launch, not a projection plus an add.
    pub fuse_residual_gemv: bool,
    /// The M>1 lane addresses the KV cache through a page table
    /// (`sdpa_paged_decode` + `kv_append_paged`) rather than the M=1
    /// contiguous pair (`sdpa_vector_decode` + `kv_append`).
    pub paged_multi_batch: bool,
    /// The M>1 projections take MLX's steel quantized GEMM
    /// (`affine_qmm_t`) instead of the GEMV — the driver's
    /// `kQmmMinBatch` gate, as a load-time fact of the deployment.
    pub qmm_multi_batch: bool,
    /// How this deployment's projections are STORED.
    ///
    /// `LlamaLikeFacts::shape()` answers `Bf16` because the semantic facts carry
    /// no backend. An affine kernel reads THREE tensors, and `MatW::scale_names`
    /// is what makes the statement say so.
    #[serde(default)]
    pub proj_repr: model_dsl::WeightRepr,
    /// Bits per packed weight element -- 4 or 8.
    ///
    /// The affine entrypoints are instantiated over `(group size x bit width)`;
    /// `proj_repr` carries the group and `WeightRepr::Scaled` has nowhere to put
    /// a bit width. A wrong axis point reads the wrong bytes.
    #[serde(default)]
    pub affine_bits: u32,
    /// How this deployment stores its EXPERT BANKS, when that is not how it
    /// stores its dense projections. `None` is "the same as `proj_repr`".
    ///
    /// `mlx-community/gpt-oss-20b-MXFP4-Q4` leaves the expert banks OUT of its
    /// `quantization` block, so they take the top-level default, `mxfp4` group
    /// **32**. Reading them with the dense format put every scale at the wrong
    /// offset: 909,207 NaNs from the first routed projection of layer 0.
    #[serde(default)]
    pub moe_repr: Option<model_dsl::WeightRepr>,
    /// [`Self::affine_bits`] for the expert banks; see [`Self::moe_repr`].
    #[serde(default)]
    pub moe_bits: u32,
    /// How this deployment stores its ROUTER GATE, when that is not how it
    /// stores its dense projections; `None` is "the same as `proj_repr`".
    ///
    /// `mlx_lm` publishes the gate WIDER than the stack it routes -- 8 bits
    /// inside a 4-bit model. Getting it wrong is quiet: cosine 0.84 against the
    /// reference logits with not one NaN to notice it by.
    #[serde(default)]
    pub router_repr: Option<model_dsl::WeightRepr>,
    /// [`Self::affine_bits`] for the router gate; see [`Self::router_repr`].
    #[serde(default)]
    pub router_bits: u32,
    /// The GEMM's `(row tile, column tile)`, as the entrypoint spells them.
    ///
    /// `affine_qmm_t` is instantiated over `(group x bits x bm x bn)`. Load-time
    /// because `qmm_bm` picks the widest rung at or under `n`, and a deployment
    /// that always fires the same window always picks the same one. `(0, 0)`
    /// spells no tile: right for a GEMV-only deployment, wrong loudly for a GEMM one.
    #[serde(default)]
    pub qmm_tile: (u32, u32),
    /// Does this build's tiled GEMM tolerate a row count its tile does not
    /// divide? [`MetalBinding::qmm_partial_rows`].
    ///
    /// It decides ONE thing: whether the projections' guard reads
    /// `TokensGT(tile - 1)` or `TokensMultipleOf(tile)`. The second is the
    /// safe reading and the expensive one -- it refuses thirty-one row counts
    /// in thirty-two, so a prompt whose length the tile does not divide runs
    /// its whole prefill on the matvec at about 2.34x the cost.
    ///
    /// [`MetalBinding::qmm_partial_rows`]: crate::catalog::MetalBinding::qmm_partial_rows
    pub qmm_partial_rows: bool,
    /// The ROUTED GEMM's tile, which is not the dense one.
    ///
    /// A separate fact because it decides a second thing the dense tile does
    /// not: the sort's block. `affine_qmm_t_routed` reads one expert id per
    /// tile and applies that bank to all `bm` rows, so every expert's run in
    /// the sorted stack has to be padded out to `bm` — and the padding is
    /// work. At `bm = 32` a mixture routing eight tokens to a hundred and
    /// twenty-eight experts pads eight live rows into 32, which is why the
    /// DECODE lane takes the matvec instead and this fact never applies to
    /// it.
    ///
    /// `(32, 32)` by default — see
    /// [`ROUTED_QMM_TILE`](crate::shared::llama_like::project::ROUTED_QMM_TILE)
    /// for the measurement that picked the row tile.
    ///
    /// **`None` takes the matvec arm in a prefill too**, which is not a
    /// tuning choice but an opt-out: a family whose batched arm is known to
    /// misbehave says so here rather than in the text. `qwen_3_5` is the one
    /// that does; see its `default_moe_tile` for what it hits.
    ///
    /// Serde-defaulted rather than required so that no existing deployment
    /// file changes meaning.
    #[serde(default = "default_moe_tile")]
    pub moe_tile: Option<(u32, u32)>,
    /// The dense GEMM stages its activation to `half` and multiplies there.
    ///
    /// A property of the DEVICE and of the checkpoint's codec, stated as a
    /// load-time fact for the same reason [`Self::qmm_tile`] is: the text
    /// that names an entrypoint has to know which one, and neither the
    /// family nor the fire can answer this.
    ///
    /// # What it buys
    ///
    /// An Apple GPU below family 9 has no `bfloat` matrix unit. MLX's steel
    /// GEMM asks for `simdgroup_matrix<bfloat>` and gets an emulation;
    /// asking for `simdgroup_matrix<half>` gets the instruction. The
    /// checkpoint and the model ABI stay `bfloat` either way — only the
    /// threadgroup tiles and the accumulator change — so this is a KERNEL
    /// choice and not a numeric one, which is why it can be a flag rather
    /// than a second dtype in the trace.
    ///
    /// The C++ driver this backend replaced calls it *"the largest single
    /// win in this driver — about 40% on the GEMM"* and the Rust rewrite
    /// carried the shaders, the driver arms and the tuning table across
    /// without ever naming the entrypoint. This fact is the naming.
    ///
    /// # What it costs
    ///
    /// One [`cast_qmm_input`](model_dsl::metal::cast_qmm_input) dispatch per
    /// activation SOURCE — not per projection: q/k/v share one staged
    /// activation and gate/up share another, which is the same grouping the
    /// C++ driver spells `llama_fp16_cast_before`.
    ///
    /// # When it is false
    ///
    /// `affine_qmm_t_fp16_precast` is instantiated at `gs = 64, b = 4`
    /// ALONE (`qmm_t.metal`'s `instantiate_qmm_t_fp16_precast` takes a tile
    /// and nothing else), so any other codec must leave this off or name a
    /// symbol that does not resolve. A family-9 device should leave it off
    /// too — there the `bfloat` unit is real and the staging pass is pure
    /// cost — which is what `Tuning::fp16_qmm` says on the driver side.
    #[serde(default)]
    pub qmm_fp16_precast: bool,
    /// The ROUTED GEMM runs its tiles and its MMA in half.
    ///
    /// `qmm_fp16_precast`'s sibling and not its consequence, for one reason
    /// that has nothing to do with arithmetic width: a mixture's next layer
    /// reads this layer's output through a TOP-K, and a top-k is a comparison.
    /// Two experts whose logits differ in the last bits can swap under a
    /// rounding change, and then the layer after runs different weights --
    /// not a tolerance away from the reference, a different model. llama's
    /// did, recorded in `llama_numerics_test`, and that is why llama sets this
    /// false at the very codec gemma-4 sets it true.
    ///
    /// The DENSE projections have no such cliff, which is why they are not
    /// gated on this: their output is added and normalized, never compared.
    ///
    /// Needs `gs = 64, b = 4` regardless -- `affine_qmm_t_routed_fp16` is
    /// stamped at that point alone -- and needs the batched arm to be running
    /// at all, since the matvec has no half form and needs none: a one-row
    /// GEMV is bandwidth and not matrix issue.
    ///
    /// Worth 47.9% of a gemma-4-26b-a4b prefill by the kernel's own header,
    /// which is the largest single term that checkpoint has.
    #[serde(default)]
    pub routed_qmm_fp16: bool,
    /// The deployment bound ONE packed `gate‖up` bank.
    ///
    /// Normally **false** on Metal: `compile_load_plan` authors with
    /// `Projections::InPlace`, and `mlp/gated.metal::silu_mul` takes gate and up
    /// as TWO buffers, so a text stating one packed value binds the OUTPUT where
    /// `up` belongs and leaves the output unbound -- a fire that runs.
    #[serde(default)]
    pub gate_up_fused: bool,
    /// `rms_norm_eps`, the epsilon every norm of this deployment carries.
    ///
    /// Taken by the shader as a field of `RmsParams`. A norm handed zero
    /// divides by the root mean square alone, which for a near-zero row is inf.
    #[serde(default)]
    pub rms_eps: f32,
    /// This deployment can LAUNCH `norm::add_bias` -- so the text may state the
    /// Qwen-2 family's q/k/v projection biases.
    ///
    /// A capability, not an architecture fact: whether the biases EXIST is
    /// [`LlamaLikeFacts::qkv_bias`]. Defaulted FALSE, and the default is
    /// load-bearing: a deployment that has not said it can launch the kernel
    /// gets the text it got before.
    #[serde(default)]
    pub add_bias: bool,
    /// This deployment can LAUNCH `norm::rms_rope` -- so the text may state
    /// the per-head q/k norm and its rotation as ONE dispatch instead of two.
    ///
    /// A capability, not an architecture fact, and the same shape
    /// [`LlamaLikeMetalFacts::add_bias`] has: defaulted FALSE, so a
    /// deployment that has not said it can launch the kernel gets exactly the
    /// text it got before. It is defaulted off for a harder reason than
    /// add_bias's, too -- only `driver-vulkan` has the kernel. The symbol
    /// resolves on the Metal side through a census entry with no `.metal`
    /// body behind it, so a Metal deployment that set this would plan a text
    /// it cannot fire.
    ///
    /// The text gates on three more things beside this one, and each is a
    /// correctness condition rather than a preference. See the call site.
    #[serde(default)]
    pub fused_qk_rope: bool,
    /// `rope_theta`, the rotary base.
    ///
    /// Stated rather than defaulted: a reader that only knows the flat key finds
    /// nothing on a config that nests it, and the rotation compounds wrong.
    #[serde(default)]
    pub rope_theta: f32,
    /// The rotary base the SLIDING layers take, or zero for a stack whose
    /// layers all share [`Self::rope_theta`].
    ///
    /// gemma-4 states both -- 1e6 full, 1e4 sliding. Keyed off
    /// [`Self::window_left_at`] rather than a second per-layer list.
    #[serde(default)]
    pub rope_theta_sliding: f32,
    /// The per-head width the FULL-attention layers use, or zero for a stack
    /// whose layers all share one.
    ///
    /// gemma-4's `global_head_dim`: on the 31b, layer 0's `q_norm` is `[256]`
    /// and layer 5's is `[512]`.
    #[serde(default)]
    pub global_head_dim: u32,
    /// The key/value head count the FULL-attention layers use, or zero for
    /// one shape everywhere. See [`Self::global_head_dim`].
    #[serde(default)]
    pub global_kv_heads: u32,
    /// What fraction of each FULL-attention head the rotation covers, or zero
    /// for a deployment that rotates the whole head.
    ///
    /// gemma-4's `partial_rotary_factor: 0.25`. It reaches the grid rather than
    /// the kernel -- `Rule::Rope` launches half of it.
    #[serde(default)]
    pub full_partial_rotary: f32,
    /// Whether the FULL-attention layers take V from the K projection.
    ///
    /// PER LAYER, and measured: `mlx-community/gemma-4-26b-a4b-it-4bit` ships no
    /// `v_proj` for layers 5, 11, 17, 23 and 29. It also reorders the two norms
    /// there: V reads the projection K's norm is about to overwrite, so V goes
    /// first. A bool because `window_left_at(l) < 0` states which layers apply.
    #[serde(default)]
    pub v_from_k: bool,
    /// Whether the mixture sits BESIDE the dense MLP rather than replacing it.
    ///
    /// gemma4's. Both branches read the post-attention residual and their
    /// results are added — SEVEN norms round one block, not five: four the
    /// dense rows also have, plus an output norm per leg and the routed
    /// leg's own input norm. Every other deployment this text serves runs
    /// one FFN or the other, which is why this is a fact and not the shape
    /// of the walk.
    #[serde(default)]
    pub dense_beside_moe: bool,
    /// Whether the router NORMS its input before projecting, at its own
    /// scale.
    ///
    /// gemma-4 publishes `router.scale`, `[hidden]`, and applies
    /// `rms_norm(x, scale * hidden**-0.5)` to the post-attention stream —
    /// not to either leg's normed value. Every other routed deployment this
    /// text serves projects the same value its experts read, so a text that
    /// assumed one input for both was right until this row.
    #[serde(default)]
    pub router_input_norm: bool,
    /// Whether the router has a learned per-expert gain,
    /// `router.per_expert_scale` `[n_experts]`.
    ///
    /// Multiplies the weights AFTER the top-k softmax; see
    /// `moe/route.metal::router_topk_scaled`.
    #[serde(default)]
    pub router_expert_scale: bool,
    /// Whether the router renormalizes over the SELECTED experts.
    ///
    /// HF's `norm_topk_prob`. True softmaxes the k chosen logits; false
    /// softmaxes over ALL experts and then selects, scaling the routed FFN's
    /// whole contribution down. qwen3-moe ships true, qwen2-moe false -- and it
    /// is a WORD of `RouterParams`, so an unstated one leaves `moe/route.metal`
    /// reading the next dispatch's staged scalars.
    #[serde(default)]
    pub norm_topk_prob: bool,
    /// Whether each layer scales the stream by a learned SCALAR.
    ///
    /// gemma's: one number per layer, read from a buffer rather than stated,
    /// because which layer is running is the fire's and not the text's.
    #[serde(default)]
    pub per_layer_scalar: bool,
    /// What this deployment multiplies its GATHERED EMBEDDINGS by, or zero for
    /// a deployment that scales them not at all.
    ///
    /// gemma's `sqrt(hidden)`, and not the same question as "does this carry a
    /// second embedding table": gemma-4-31b has no PLE, is still a gemma, and
    /// unscaled its gather's widest value was 0.058 against MLX's ~70x that.
    #[serde(default)]
    pub embed_scale: f32,
    /// The SOFTMAX TEMPERATURE, or zero for the `1/sqrt(head_dim)` default.
    ///
    /// The number a reader is likeliest to believe is a constant. The three
    /// families this crate serves state three different things:
    ///
    /// | family | scale |
    /// |---|---|
    /// | llama-3 | `head_dim ** -0.5` |
    /// | gemma-3 | `query_pre_attn_scalar ** -0.5`, and the scalar is stated |
    /// | gemma-4 | **`1.0`** |
    ///
    /// gemma-4 normalizes Q and K to unit RMS before attending, so the division
    /// has already happened.
    #[serde(default)]
    pub attn_scale: f32,
    /// Whether V is RMS-normed, per head, before it reaches the KV pool.
    ///
    /// A NORM WITH NO WEIGHT (MLX's `RMSNormNoScale`), so no tensor probe can
    /// answer it. Runs on every layer that PROJECTS KV; its axis is `head_dim`.
    ///
    /// ORDER MATTERS ON A K-EQ-V LAYER: V must be read BEFORE `k_norm`
    /// overwrites it.
    #[serde(default)]
    pub v_norm: bool,
    /// gemma's PER-LAYER EMBEDDING width, or zero for a deployment with none.
    ///
    /// A SIDE NETWORK: a second embedding table gathered once per step,
    /// projected, normed and joined into `[n_layers, ple_dim]`.
    #[serde(default)]
    pub per_layer_emb_dim: u32,
    /// Layers at the END of the stack that SHARE their KV with an earlier one.
    ///
    /// A shared layer rotates its own Q and reads the pages its source wrote:
    /// no k/v projection, no k/v norm, no append.
    #[serde(default)]
    pub kv_shared_layers: u32,
    /// The readout's SOFTCAP — `cap * tanh(x / cap)` — or zero for none.
    /// Zero names nothing rather than passing a cap so large it does nothing,
    /// which would be a kernel run per fire to compute the identity.
    #[serde(default)]
    pub logit_softcap: f32,
    /// Whether every layer carries an attention SINK: a per-head learned logit
    /// that joins the softmax without a value behind it, so a sinked attention
    /// normalizes over one more term than it sums. gpt-oss's.
    #[serde(default)]
    pub attn_sinks: bool,
    /// WHICH gated activation this deployment takes. Defaults to `silu_mul`.
    ///
    /// Three symbols, not one with flags: gpt-oss clamps the gate ABOVE only,
    /// clamps the linear branch both ways and adds one to it; gemma's gelu is
    /// the TANH approximation. Dropping any of that runs and is wrong.
    #[serde(default)]
    pub activation: Activation,
    /// Whether this deployment's rotary frequencies come from a TABLE.
    ///
    /// True for a config that rescales its ladder -- llama-3's `rope_scaling`
    /// with `rope_type: llama3`, YaRN, anything that is not a plain geometric
    /// series in a base, none of which a `rope_theta` can express. The table
    /// itself is the DRIVER's, answered as
    /// `Source::Named(<keys::RopeFrequencies as keys::Fact>::KEY)`.
    #[serde(default)]
    pub rope_freq_table: bool,
    /// Whether the rotation's exponent and pairing are taken over the WHOLE
    /// head rather than over the rotated slice.
    ///
    /// A partial rotary has two readings and the checkpoint picks one. The
    /// ordinary one -- qwen3's, qwen3.6's -- rotates the leading `rotary`
    /// channels as a head of that size: pair `i` with `i + rotary/2`, and let
    /// the exponent run `0..1` across the slice. gemma-4's `rope_type:
    /// proportional` rotates the leading `rotary` channels IN PLACE in a head
    /// of the full width: pair `i` with `i + head_dim/2`, and let the exponent
    /// run across the head so the slice only ever sees its first quarter. At
    /// `head_dim: 512, rotary: 128` the channels that move are `[0,63]` and
    /// `[256,319]`, and every frequency is four ladder steps off the other
    /// reading's.
    ///
    /// Two symbols and not a parameter, because the two arithmetics are
    /// written out separately in `rope/neox.metal` on purpose: folding them
    /// into one body moved gemma-4's recorded continuation, the compiler's
    /// contraction having changed under the same source-level formula.
    /// `neox_prop_mb` is the batched one and reduces exactly to the geometric
    /// form when the rotary covers the head, which is what a gemma-4 sliding
    /// layer does -- so this is a fact of the MODEL and not of the layer.
    #[serde(default)]
    pub rope_proportional: bool,
    /// The SLIDING WINDOW each layer attends over, `-1` for none; empty means
    /// every layer is `-1`. Read through [`Self::window_left_at`].
    ///
    /// The same fact [`LlamaLikeCudaFacts::window_left`] carries, stated here
    /// rather than shared because the two halves are independently
    /// deserialized.
    #[serde(default)]
    pub window_left: Vec<i32>,
}

impl LlamaLikeMetalFacts {
    /// gpt-oss-20b's Metal facts. A SYNTHETIC fixture like `synthetic`.
    #[must_use]
    pub fn gpt_oss_20b() -> Self {
        Self {
            // YaRN's OTHER number, which the rope table cannot carry:
            // `1.3466^2 / 8`, against the `0.125` a derived `1/sqrt(64)` gives.
            // A 1.81x error in the softmax temperature does not fault; it
            // sharpens every distribution in the stack.
            attn_scale: 0.226_657_55,
            attn_sinks: true,
            // `swiglu_limit: 7.0`, and alpha is the activation's own constant.
            activation: Activation::SwiGlu {
                limit: 7.0,
                alpha: 1.702,
            },
            // `rope_theta: 150000` with a YaRN-rescaled ladder over it
            // (factor 32, beta_fast 32, beta_slow 1, `truncate: false`), which
            // a theta alone cannot express, so the driver derives the table at
            // load and answers it as `keys::RopeFrequencies`.
            rope_theta: 150_000.0,
            rope_freq_table: true,
            rms_eps: 1e-5,
            // `sliding_window: 128`, ALTERNATING: every other layer attends
            // the window and the rest attend everything.
            window_left: (0..24).map(|l| if l % 2 == 0 { 128 } else { -1 }).collect(),
            // The EXPERT BANKS' own encoding, which is not the projections'.
            // `mlx-community/gpt-oss-20b-MXFP4-Q4` leaves them out of its
            // `quantization` block, so they take the top-level default: mxfp4,
            // group 32, 4 bits.
            moe_repr: Some(model_dsl::WeightRepr::Mxfp4Marlin),
            moe_bits: 4,
            // The SECOND affine point, measured on this row's own `config.json`:
            // 98 dense tensors at group 64 / 4 bits and 24 `mlp.router` gates at
            // group 64 / EIGHT.
            router_repr: Some(model_dsl::WeightRepr::Scaled {
                layout: model_dsl::ScaleLayout::PerGroup,
                group: 64,
                axis: 0,
                zero_point: true,
            }),
            router_bits: 8,
            ..Self::synthetic()
        }
    }

    /// The three gemma facts that ARE facts, on an otherwise llama-like
    /// deployment.
    ///
    /// The PLE and the KV sharing are here too, which makes this the fixture a
    /// gemma4 text reads. What it is NOT is a measurement: the widths are
    /// plausible rather than any published config's.
    #[must_use]
    pub fn gemma_like() -> Self {
        Self {
            activation: Activation::Geglu,
            // The generation's rotation, and the one fact here that decides a
            // SYMBOL rather than an argument: `neox_prop_mb_bfloat16` against
            // `neox_mb_bfloat16`. A gemma text that read this fixture with the
            // field at its default would trace the rotation every shipped
            // gemma-4 row gets wrong.
            rope_proportional: true,
            logit_softcap: 30.0,
            per_layer_emb_dim: 256,
            kv_shared_layers: 4,
            dense_beside_moe: true,
            router_input_norm: true,
            router_expert_scale: true,
            // Every shipped gemma-4 norms its V, and no other family here
            // does -- so a fixture that left this false made the one text
            // that could exercise the branch fire the branch beside it.
            v_norm: true,
            // ONE, which every shipped gemma-4 row states and no other
            // family here does. Zero is not a smaller temperature, it is
            // the SENTINEL that makes `llama_like_metal` derive
            // `1/sqrt(head_dim)` -- and gemma-4's per-head `q_norm` and
            // `k_norm` have already divided by that, so a derived scale
            // divides twice. A fixture left at zero traced the one text
            // that could exercise the stated scale with the derived one.
            attn_scale: 1.0,
            // `sqrt(1024)`, the `hidden` both call sites pair this with. Zero is
            // a BRANCH: `llama_like_metal` reads `embed_scale > 0.0` and emits
            // `embed_gather` instead.
            embed_scale: 32.0,
            // ONE ENTRY PER LAYER OF THE STACK THIS IS PAIRED WITH, which is
            // `LlamaLikeFacts::qwen3_0_6b()`'s twenty-eight -- the same
            // pairing `embed_scale` above is computed from.
            //
            // It stated twenty-four, and a short list is not a shorter
            // statement: `window_left_at` CLAMPS to the last entry, and the
            // last of a `l % 6 == 5` run of twenty-four is layer 23, whose
            // value is `-1`. So layers 24 through 27 read as full attention,
            // and the fixture described a stack whose full layers are
            // {5, 11, 17, 23, 24, 25, 26, 27} -- not one every six, not one
            // every anything. `batch::geometry` REFUSES an irregular schedule
            // by name, so this was a fixture no deployment could be, and it
            // took the metal text with it: four layers rotated at the global
            // theta, rotated a quarter of a head they do not have, and named
            // `sdpa_paged_*_d_256` for a shape the stack states at 128.
            //
            // Twenty-eight ends the run at layer 27, which is `512` under the
            // same rule, so the clamp is unreachable and the full layers are
            // the four the period names.
            window_left: (0..28).map(|l| if l % 6 == 5 { -1 } else { 512 }).collect(),
            // gemma-4 states TWO attention geometries and every shipped row
            // moves all three of these off the default. A fixture that left
            // them there described one shape for the whole stack, which is
            // every family here EXCEPT the one it is named for -- so the
            // gemma texts never emitted the second geometry and never
            // rotated a fraction.
            //
            // Doubling and halving what the paired `LlamaLikeFacts` states
            // (`qwen3_0_6b`: `head_dim` 128, `kv_heads` 8), because this
            // fixture is not a measurement; what has to be true is that the
            // two geometries DIFFER, the way 31b's 256/16 differs from its
            // 512/4.
            global_head_dim: 256,
            global_kv_heads: 4,
            // A quarter, which is what all three shipped rows publish. Zero is
            // not a smaller fraction, it is "rotate every channel".
            full_partial_rotary: 0.25,
            rope_theta: 1_000_000.0,
            // The SLIDING layers' base. gemma states both, and this fixture
            // slides twenty of its twenty-four layers.
            rope_theta_sliding: 10_000.0,
            // The one fixture that stages the ROUTED tiles. gemma-4's mixture
            // is affine g64/b4, which is where `affine_qmm_t_routed_fp16` is
            // stamped, and its top-k held where llama's moved. gpt-oss reads
            // this through `..synthetic()` at false and correctly: its routed
            // bank is MXFP4, whose kernel is `mxfp4_qmm_t_routed_bias` and
            // stages its tiles already.
            routed_qmm_fp16: true,
            ..Self::synthetic()
        }
    }

    /// This layer's window, `-1` for all of it. See [`Self::window_left`].
    pub fn window_left_at(&self, l: u32) -> i32 {
        model_ir::facts::window_left_at(&self.window_left, l)
    }

    /// This layer's rotary base. See [`Self::rope_theta_sliding`].
    pub fn rope_theta_at(&self, l: u32) -> f32 {
        if self.rope_theta_sliding > 0.0 && self.window_left_at(l) >= 0 {
            self.rope_theta_sliding
        } else {
            self.rope_theta
        }
    }

    /// Whether layer `l` attends the WHOLE context.
    ///
    /// The one question the two per-layer-type facts below are keyed on, so
    /// they cannot disagree with each other or with the window list.
    pub fn is_full_attention(&self, l: u32) -> bool {
        self.window_left_at(l) < 0
    }

    /// This layer's per-head width, given the deployment's usual one. Zero
    /// means one shape for the whole stack.
    ///
    /// gemma-4 states TWO: on `gemma-4-31b-it-4bit`, layer 0 (sliding) has
    /// `q_norm [256]` and `q_proj [8192, …]` = 32x256, while layer 5 (full) has
    /// `q_norm [512]` and `q_proj [16384, …]` = 32x512.
    pub fn head_dim_at(&self, l: u32, sliding: u32) -> u32 {
        if self.global_head_dim > 0 && self.is_full_attention(l) {
            self.global_head_dim
        } else {
            sliding
        }
    }

    /// This layer's key/value head count. See [`Self::head_dim_at`] — the
    /// 31b's full layers carry 4 where its sliding ones carry 16
    /// (`k_proj [2048, …]` = 4x512 against `[4096, …]` = 16x256).
    pub fn kv_heads_at(&self, l: u32, sliding: u32) -> u32 {
        if self.global_kv_heads > 0 && self.is_full_attention(l) {
            self.global_kv_heads
        } else {
            sliding
        }
    }

    /// How many of this layer's channels the rotation covers: the whole head
    /// everywhere but gemma-4's full-attention layers. Rounded DOWN to an even
    /// number because the rotation pairs channels — `Rule::Rope` launches
    /// `rotary/2` and an odd width would leave a lane without its partner.
    pub fn rotary_dim_at(&self, l: u32, head_dim: u32) -> u32 {
        let dim = self.head_dim_at(l, head_dim);
        if self.full_partial_rotary <= 0.0 || !self.is_full_attention(l) {
            return dim;
        }
        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            clippy::cast_precision_loss
        )]
        let want = (f64::from(dim) * f64::from(self.full_partial_rotary)) as u32;
        (want.min(dim) / 2 * 2).max(2)
    }

    /// A SYNTHETIC fixture, not a measurement — see the struct comment.
    /// These are the driver's own defaults as its source reads them.
    pub fn synthetic() -> Self {
        Self {
            qmm_partial_rows: false,
            fuse_residual_gemv: true,
            paged_multi_batch: true,
            qmm_multi_batch: true,
            // TRUE, read off `driver-metal`'s source like the three above:
            // `lowering::dispatch::derived` is the `Source::OutWidth` arm its
            // binder used to lack.
            add_bias: true,
            // FALSE, and it is the one capability in this fixture that is not
            // read off `driver-metal`'s source but off the absence of a file:
            // there is no Metal `rms_rope`.
            fused_qk_rope: false,
            // The value every routed row in this catalog publishes, and the
            // opposite of `Qwen3MoeConfig`'s class default.
            norm_topk_prob: true,
            // No Metal deployment publishes a fused bank; see the field. The
            // symbols this text names are affine, so the deployment is a
            // quantized checkpoint, and MLX's `.scales`/`.biases` pair beside
            // the packed weight is a zero-point layout.
            proj_repr: model_dsl::WeightRepr::Scaled {
                layout: model_dsl::ScaleLayout::PerGroup,
                group: 64,
                axis: 0,
                zero_point: true,
            },
            // g64/b4, which is the one codec `affine_qmm_t_fp16_precast` is
            // stamped at -- so the synthetic deployment takes the staged
            // path, as every MLX checkpoint in this catalog does.
            qmm_fp16_precast: true,
            // FALSE for the generic row, which is llama's answer and not a
            // placeholder: this family's routed checkpoint reordered a
            // next-layer top-k under half rounding. `gemma_like` overrides
            // it, and that pair is what keeps this predicate out of `EXCUSED`.
            routed_qmm_fp16: false,
            affine_bits: 4,
            // Uniform: the synthetic fixture's banks are the dense format.
            moe_repr: None,
            moe_bits: 0,
            // `None` is "the same as the dense projections": a fixture that
            // states a SECOND affine point states a checkpoint fact.
            router_repr: None,
            router_bits: 0,
            // What `project::QMM_TILE` states; `bn = 32` is the only column
            // tile the residual variant is instantiated at. Written out so a
            // fixture and a projection can be compared.
            qmm_tile: (32, 32),
            moe_tile: default_moe_tile(),
            // `Projections::InPlace` is what `compile_load_plan` authors with,
            // and the join declines under it.
            gate_up_fused: false,
            // qwen3's `rms_norm_eps`.
            rms_eps: 1e-6,
            // qwen3's `rope_theta`. The shader raises TWO to a base, so the
            // statement hands it `log2(theta)`; handing theta rotates by a
            // frequency ladder that is wrong from the second channel on.
            rope_theta: 1_000_000.0,
            // qwen3 states ONE base for every layer, which is what zero means.
            rope_theta_sliding: 0.0,
            // And ONE attention shape for every layer, likewise, rotating
            // the whole head.
            global_head_dim: 0,
            global_kv_heads: 0,
            full_partial_rotary: 0.0,
            // qwen3's mixture replaces its dense MLP, it has no per-layer
            // embeddings or scalar, shares no KV, and projects its own V.
            v_from_k: false,
            dense_beside_moe: false,
            router_input_norm: false,
            router_expert_scale: false,
            per_layer_scalar: false,
            embed_scale: 0.0,
            attn_scale: 0.0,
            v_norm: false,
            per_layer_emb_dim: 0,
            kv_shared_layers: 0,
            // qwen3 caps no logits, has no attention sinks, and takes the
            // plain gated activation.
            logit_softcap: 0.0,
            attn_sinks: false,
            activation: Activation::SiluMul,
            // qwen3's ladder is a plain geometric series in `rope_theta`.
            rope_freq_table: false,
            // And it rotates the whole head, where the two readings of a
            // partial rotary coincide anyway.
            rope_proportional: false,
            // qwen3 attends over the whole context at every layer.
            window_left: Vec::new(),
        }
    }
}

impl LlamaLikeCudaFacts {
    /// The window layer `l` attends over, `-1` for none.
    ///
    /// A short list is not an error: a deployment whose config carries ONE
    /// `sliding_window` states a one-element list and every layer reads it.
    pub fn window_left_at(&self, l: u32) -> i32 {
        model_ir::facts::window_left_at(&self.window_left, l)
    }

    /// Qwen3-0.6B on L40S, default env -- MEASURED 2026-08-02 against the
    /// driver's own derivation via the rung-3 digest print:
    /// `.../qk1/fq1/te0/xqa0/dfp1/rt1/fpp0`.
    ///
    /// The deployment's BINDING unties the lm_head (`w.lm_head != w.embed`,
    /// live `te0`) even though the checkpoint's config ties it;
    /// [`LlamaLikeFacts::qwen3_0_6b`] keeps the config-level fact for the
    /// semantic goldens and emission against the live deployment overrides it.
    pub fn qwen3_0_6b_l40s() -> Self {
        Self {
            xqa_decode: false,
            decode_fused_post: true,
            rope_table: true,
            force_prefill_path: false,
            head_dim_padded: false,
            head_dim_kernel: 0,
            // The loader's `dense_fused_projection_joins` contract packs BF16
            // dense groups and declines quantized ones, so a plain BF16
            // deployment carries the bank. VERIFIED LIVE: the digest refuses
            // the deployment if this fixture and the binding disagree.
            gate_up_fused: true,
            // Dense, by the same contract line: a group it packs is a BF16
            // one, so a deployment carrying the bank cannot be quantized.
            proj_repr: WeightRepr::Bf16,
            // One rank: the L40S fixture is a single GPU.
            tp_size: 1,
            // qwen3-0.6B attends over the whole prefix.
            window_left: Vec::new(),
            // Single GPU: no collective, so no threshold.
            all_reduce_p2p_max_rows: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every branch this text takes, taken BOTH WAYS by some fixture -- or named
    /// here with the reason it cannot be.
    ///
    /// Read through `serde` so the field list is the struct's rather than a copy:
    /// a boolean added tomorrow is asked about the day it is added. `EXCUSED` is
    /// held EXACTLY, so an entry that stops being needed fails as loudly as a
    /// branch that goes dark.
    #[test]
    fn every_metal_predicate_is_stated_more_than_one_way_or_excused() {
        use std::collections::{BTreeMap, BTreeSet};

        // `(field, why)`, and the why has to be a fact about the crate
        // rather than an intention.
        const EXCUSED: &[(&str, &str)] = &[
            // Refused at trace time by `llama_like_metal` itself: `silu_mul`
            // takes gate and up as two buffers, no Metal kernel splits a packed
            // bank into them, and `compile_load_plan` authors with
            // `Projections::InPlace`. A fixture stating `true` would assert.
            ("gate_up_fused", "asserted false in the text"),
            // Stated inline by `driver-metal`'s texts rather than by a
            // constructor here; gemma-4-31b is the shipped row behind it, kept
            // out of the real-weights rig only by its seventeen gigabytes.
            ("per_layer_scalar", "stated inline by driver-metal's texts"),
            // Both ways in `gemma_4::project`'s own mapping test, which loops
            // `k_eq_v` over `[false, true]`.
            ("v_from_k", "both ways in gemma_4::project's mapping test"),
            // Both ways in this module's own trace test. No shipped Metal
            // binding turns it off -- the arm is for a deployment whose loader
            // states an explicit residual landing per block -- so a fixture
            // stating `false` would describe a deployment that does not exist.
            (
                "fuse_residual_gemv",
                "both ways in this module's own trace test",
            ),
            // Both ways in `forward`'s own trace test,
            // `the_fused_qk_norm_rope_replaces_the_pair_and_only_when_it_may`,
            // which states the fusion's four conditions one at a time.
            //
            // No fixture states it TRUE because no shipped Metal binding
            // does: there is no `.metal` kernel behind `rms_rope_bfloat16`,
            // only a name in `kernels-metal`'s census so that `model-ir` can
            // check a VULKAN text -- Vulkan consumes the metal-flavoured
            // plan. A fixture stating `true` would describe a Metal
            // deployment that does not exist, and every trace assertion in
            // this tree would then read a text no device could run.
            (
                "fused_qk_rope",
                "both ways in forward's own trace test",
            ),
            // NOT A BRANCH. `llama_like_metal` reads it as `let _ =` and sets
            // `paged = true` unconditionally: it is a fact about the DRIVER's
            // allocation and not about the fire.
            ("paged_multi_batch", "read and discarded, with a reason"),
            // Not a branch either: it is an ARGUMENT to `router_topk`, so the
            // false side is the same symbol computing something else.
            ("norm_topk_prob", "an argument, not a branch"),
            // ANDed with the FIRE CLASS: `multi_batch = class !=
            // FireClass::Decode`, so one fixture takes both arms.
            (
                "qmm_multi_batch",
                "the conjunct beside it is the fire class",
            ),
            // ANDed with a SHAPE fact -- `f.qkv_bias`, `f.o_bias`,
            // `f.router_bias` -- which the seven `LlamaLikeFacts` fixtures
            // state both ways.
            ("add_bias", "the conjunct beside it is a shape fact"),
            // A NUMBER rather than a bool. Four in all three fixtures, but
            // `kernels-metal` instantiates the 8-bit widths too and
            // `mlx-community` publishes 8-bit snapshots of these same rows.
            // Stated by `driver-metal`'s `text_conformance`, whose
            // `the_two_affine_widths_are_an_exchange_not_a_default` asserts
            // each text names ONE width and that they differ.
            ("affine_bits", "stated inline by driver-metal's texts"),
            // TRUE in all three, and the reason is the catalog rather than
            // the predicate: `qmm_t.metal` stamps
            // `affine_qmm_t_fp16_precast` at `gs = 64, b = 4` alone, and
            // every Metal row this family ships is g64/b4 -- gpt-oss's
            // `config.json` reads `group_size: 32` at the top level, which
            // is the MXFP4 EXPERT mode, while its 98 dense tensors are 64/4
            // like the rest.
            //
            // The false arm is production's, not a hypothetical: the 8-bit
            // affine snapshots the `affine_bits` excuse above names take it,
            // and so does any family-9 device, where the `bfloat` matrix
            // unit is real and the staging pass is pure cost. Both arms are
            // asserted directly on the predicate by
            // `project::tests::the_staged_gemm_is_the_g64_b4_codecs_alone`,
            // which is the statement a fixture pair would have made.
            (
                "qmm_fp16_precast",
                "one codec is stamped and this catalog is all of it",
            ),
            // A BUILD's stamp, like `qmm_tile` beside it -- which escapes
            // this sweep only because it serialises as an array and the
            // filter above keeps booleans and numbers. The answer is which
            // kernels were compiled, so it does not vary by model family and
            // three family fixtures would state it identically however they
            // were written.
            //
            // Both arms are asserted directly on the predicate by
            // `project::tests::the_partial_row_tolerance_is_the_builds_stamp_
            // and_not_the_familys`, which is the statement a fixture pair
            // would have made.
            (
                "qmm_partial_rows",
                "a build's stamp, asserted directly in project::tests",
            ),
        ];

        let scalars = |m: &LlamaLikeMetalFacts| match serde_json::to_value(m) {
            Ok(serde_json::Value::Object(o)) => o
                .into_iter()
                .filter(|(_, v)| v.is_boolean() || v.is_number())
                .map(|(k, v)| (k, v.to_string()))
                .collect::<BTreeMap<_, _>>(),
            other => panic!("these facts serialise as a struct, not {other:?}"),
        };
        let every = [
            scalars(&LlamaLikeMetalFacts::synthetic()),
            scalars(&LlamaLikeMetalFacts::gpt_oss_20b()),
            scalars(&LlamaLikeMetalFacts::gemma_like()),
        ];
        // Non-vacuity: `serde_json` handing back an empty map would make every
        // assertion below pass over nothing.
        assert!(
            every[0].len() >= 20,
            "these facts carry more scalars than this: {:?}",
            every[0].keys().collect::<Vec<_>>()
        );

        let dark: BTreeSet<&str> = every[0]
            .keys()
            .filter(|name| every.iter().all(|f| f[*name] == every[0][*name]))
            .map(String::as_str)
            .collect();
        let excused: BTreeSet<&str> = EXCUSED.iter().map(|(f, _)| *f).collect();

        let opened: Vec<&&str> = dark.difference(&excused).collect();
        assert!(
            opened.is_empty(),
            "branch(es) every fixture states IDENTICALLY, and none is \
             excused. A predicate one value reaches compiles and is never \
             emitted the other way: {opened:?}"
        );
        let closed: Vec<&&str> = excused.difference(&dark).collect();
        assert!(
            closed.is_empty(),
            "excuse(s) that stopped being needed -- a fixture now states \
             more than one value, so the entry says something false about \
             this crate: {closed:?}"
        );
    }

    /// gpt-oss alternates: even layers see a window, odd layers see all.
    ///
    /// The fixture's callers are GPU tests in `driver-metal`, `driver-vulkan`
    /// and `driver-cuda`, none of which run without a device — so an
    /// alternation that silently became uniform would make every one of those
    /// tests agree with itself about the wrong model.
    #[test]
    fn the_gpt_oss_fixture_alternates_its_window_layer_by_layer() {
        let facts = LlamaLikeMetalFacts::gpt_oss_20b();
        assert!(facts.attn_sinks, "every layer carries a per-head sink");
        for layer in 0..24 {
            let expected = if layer % 2 == 0 { 128 } else { -1 };
            assert_eq!(
                facts.window_left_at(layer),
                expected,
                "layer {layer}'s window"
            );
            assert_eq!(
                facts.is_full_attention(layer),
                layer % 2 == 1,
                "layer {layer} attends the whole context only when unwindowed"
            );
        }
    }

    /// The window list is shorter than any real stack, and that is the rule.
    ///
    /// `window_left_at` clamps: the LAST entry covers the tail, so a model
    /// deeper than its own fixture keeps the final layer's window rather than
    /// falling off the end.
    #[test]
    fn a_layer_past_the_end_of_the_list_keeps_the_last_entry() {
        // The tail entry is a REAL window rather than -1 on purpose: asking
        // this of `gpt_oss_20b` compares nothing, because its last layer is odd
        // and a fallback that stopped clamping would agree with it.
        let facts = LlamaLikeMetalFacts {
            window_left: vec![128, 256],
            ..LlamaLikeMetalFacts::synthetic()
        };
        assert_eq!(facts.window_left_at(0), 128);
        assert_eq!(facts.window_left_at(1), 256);
        for layer in [2, 3, 47, 9_999] {
            assert_eq!(
                facts.window_left_at(layer),
                256,
                "layer {layer} is past the end, where the tail entry governs"
            );
        }
    }

    /// An empty list means every layer attends everything — the case every
    /// non-windowed family takes, so the one that would be noticed last.
    #[test]
    fn an_unstated_window_is_no_window_on_every_layer() {
        let facts = LlamaLikeMetalFacts::synthetic();
        assert!(
            facts.window_left.is_empty(),
            "the synthetic deployment states no window"
        );
        for layer in [0, 1, 7, 100] {
            assert_eq!(facts.window_left_at(layer), -1, "layer {layer}");
            assert!(facts.is_full_attention(layer));
        }
    }

    /// gemma slides five layers in six, and rotates them at a DIFFERENT base.
    ///
    /// The fact whose loss is silent: a driver that read the single config
    /// value rotated twenty of twenty-four layers at the wrong frequencies.
    /// Not a load error, not a crash — wrong output at every position but
    /// zero.
    #[test]
    fn gemmas_sliding_layers_rotate_at_the_sliding_base() {
        let facts = LlamaLikeMetalFacts::gemma_like();
        assert_eq!(facts.rope_theta, 1_000_000.0);
        assert_eq!(facts.rope_theta_sliding, 10_000.0);
        assert_ne!(
            facts.rope_theta, facts.rope_theta_sliding,
            "the two bases differ, which is the whole reason for the accessor"
        );

        let mut sliding = 0;
        for layer in 0..24 {
            if facts.is_full_attention(layer) {
                assert_eq!(
                    facts.rope_theta_at(layer),
                    facts.rope_theta,
                    "full layer {layer} takes the global base"
                );
            } else {
                sliding += 1;
                assert_eq!(
                    facts.rope_theta_at(layer),
                    facts.rope_theta_sliding,
                    "sliding layer {layer} takes the sliding base"
                );
            }
        }
        assert_eq!(sliding, 20, "twenty of twenty-four layers slide");
    }

    /// A deployment that states ONE base uses it on every layer.
    ///
    /// `rope_theta_sliding` of zero is "not stated", and the accessor has to
    /// read that as "the one base" rather than as a base of zero.
    #[test]
    fn a_single_rotary_base_governs_the_windowed_layers_too() {
        let facts = LlamaLikeMetalFacts::gpt_oss_20b();
        assert_eq!(
            facts.rope_theta_sliding, 0.0,
            "gpt-oss states one base for a stack that does alternate windows"
        );
        for layer in 0..24 {
            assert_eq!(
                facts.rope_theta_at(layer),
                150_000.0,
                "layer {layer} rotates at the only base there is"
            );
        }
    }

    /// The gemma fixture carries the facts that make it gemma.
    ///
    /// Its doc is explicit that the WIDTHS are plausible rather than measured;
    /// these are the structural claims, and a fixture that lost one would make
    /// the gemma path test the llama path under a gemma name. `embed_scale` is
    /// on the list because zero is a BRANCH: `llama_like_metal` reads
    /// `embed_scale > 0.0`.
    #[test]
    fn the_gemma_fixture_is_gemma_shaped_rather_than_llama_shaped() {
        let facts = LlamaLikeMetalFacts::gemma_like();
        assert!(matches!(facts.activation, Activation::Geglu));
        assert_eq!(facts.logit_softcap, 30.0);
        assert_eq!(facts.per_layer_emb_dim, 256);
        assert_eq!(facts.kv_shared_layers, 4);
        assert!(facts.dense_beside_moe);
        assert!(
            facts.embed_scale > 0.0,
            "a gemma scales its embedding, and zero picks the other statement"
        );

        let plain = LlamaLikeMetalFacts::synthetic();
        assert!(
            matches!(plain.activation, Activation::SiluMul),
            "and the base fixture it derives from is the llama-like one"
        );
        assert_eq!(plain.logit_softcap, 0.0);
        assert_eq!(plain.per_layer_emb_dim, 0);
        assert_eq!(plain.embed_scale, 0.0);
    }
}
