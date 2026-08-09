//! `llama_like`'s load-time facts: the semantic ones, and one struct per
//! backend for what a deployment bound.

use serde::{Deserialize, Serialize};
// The shared vocabulary stayed with the toolchain -- more than one family
// is written in these words. Re-exported so a declaration reaches its
// facts and the words they are stated in from one place.
pub use model_compiler::facts::{NormPlacement, QkNorm};
use model_compiler::dsl::WeightRepr;
use model_compiler::trace::{NormVariant, RopeKind};

/// The llama_like family's facts: covers qwen3, mistral3, phi3, olmo2/3
/// (pie-application-plan.md §7 stage 3 scope). Declared so far: the qwen3
/// configuration — pre-norm, per-head qk-norm, standard rope, fused QKV
/// binding, dense MLP — the phi3 one, which drops the qk-norm and the
/// embedding tie, the mistral (7B v0.3) one, which pairs the fused
/// binding with no qk-norm, and the olmo2 (1B) one — the first to change
/// the declaration itself: post-norm placement and the global qk-norm.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LlamaLikeFacts {
    pub hidden: u32,
    pub layers: u32,
    pub q_heads: u32,
    pub kv_heads: u32,
    pub head_dim: u32,
    pub intermediate: u32,
    pub vocab: u32,
    pub rope: RopeKind,
    pub norm_variant: NormVariant,
    /// Norm-vs-residual order per sub-layer; `Pre` for every configuration
    /// before olmo2. Serde-defaulted so pre-olmo facts JSON (none is
    /// persisted today, but the goldens' discipline applies) reads back
    /// unchanged.
    #[serde(default)]
    pub norm_placement: NormPlacement,
    /// RMSNorm on Q/K before rope: off, per-head (qwen3) or global (olmo2).
    pub qk_norm: QkNorm,
    /// The deployment bound one packed `[q + 2kv, hidden]` projection.
    /// This is a *binding* fact, not an architecture fact: the declaration
    /// writes one matmul either way, and with `false` it traces three.
    pub fused_qkv: bool,
    /// The lm_head weight is the embedding table (weight tying).
    pub tied_embeddings: bool,
    /// Qwen-2 family attention biases: the checkpoint ships
    /// `{q,k,v}_proj.bias` and the forward adds them to the raw
    /// projections (after the lora correction, before norms/rope — the
    /// hand-written `maybe_add_bias` position). Serde-defaulted so
    /// pre-bias facts JSON reads back unchanged (append-only discipline).
    #[serde(default)]
    pub qkv_bias: bool,
}

impl LlamaLikeFacts {
    /// This family's projection into the DSL's family-neutral
    /// [`ModelShape`](model_compiler::dsl::ModelShape) — the dense-transformer weight
    /// namespace, and nothing about llama in particular.
    ///
    /// The toolchain cannot name `LlamaLikeFacts` -- that edge would point the
    /// wrong way -- so the projection is written here, on the family side,
    /// once per family.
    pub fn shape(&self) -> model_compiler::dsl::ModelShape {
        model_compiler::dsl::ModelShape {
            hidden: self.hidden,
            intermediate: self.intermediate,
            vocab: self.vocab,
            head_dim: self.head_dim,
            q_width: self.q_width(),
            kv_width: self.kv_width(),
            qk_norm: self.qk_norm,
            norm_variant: self.norm_variant,
            tied_embeddings: self.tied_embeddings,
            // DENSE, because these are the SEMANTIC facts: a trace with
            // no backend cannot name the kernel a scaled weight needs,
            // so the representation reaches the namespace from the
            // BACKEND facts (`llama_like_cuda` overrides it below).
            proj_repr: model_compiler::dsl::WeightRepr::Bf16,
        }
    }

    pub fn q_width(&self) -> u32 {
        self.q_heads * self.head_dim
    }

    pub fn kv_width(&self) -> u32 {
        self.kv_heads * self.head_dim
    }

    /// Qwen2.5-1.5B-Instruct (Qwen/Qwen2.5-1.5B-Instruct config.json):
    /// the fifth llama_like configuration, and the first with attention
    /// biases (`qkv_bias: true` — Qwen2ForCausalLM binds
    /// `{q,k,v}_proj.bias` and the forward adds them to the raw
    /// projections; the AddBias rung). GQA (12 q / 2 kv heads), head_dim
    /// 128 (hidden 1536 / 12 — no config key; the derivation matches the
    /// driver's), no qk-norm, tied embeddings (`tie_word_embeddings:
    /// true`). `rope_theta: 1e6` and `sliding_window` (unused:
    /// `use_sliding_window: false`) are backend cfg the trace
    /// deliberately lacks. `fused_qkv: true` is the binding fact: the
    /// checkpoint ships three raw bf16 projections under canonical names
    /// and the dense join re-fuses the WEIGHTS (biases stay separate
    /// tensors, added after the split — the hand-written order).
    pub fn qwen2_5_1_5b() -> Self {
        Self {
            hidden: 1536,
            layers: 28,
            q_heads: 12,
            kv_heads: 2,
            head_dim: 128,
            intermediate: 8960,
            vocab: 151_936,
            rope: RopeKind::Standard,
            norm_variant: NormVariant::Plain,
            norm_placement: NormPlacement::Pre,
            qk_norm: QkNorm::Off,
            fused_qkv: true,
            tied_embeddings: true,
            qkv_bias: true,
        }
    }

    /// Qwen3-0.6B, the workspace's parity model.
    pub fn qwen3_0_6b() -> Self {
        Self {
            hidden: 1024,
            layers: 28,
            q_heads: 16,
            kv_heads: 8,
            head_dim: 128,
            intermediate: 3072,
            vocab: 151_936,
            rope: RopeKind::Standard,
            norm_variant: NormVariant::Plain,
            norm_placement: NormPlacement::Pre,
            qk_norm: QkNorm::PerHead,
            fused_qkv: true,
            tied_embeddings: true,
            qkv_bias: false,
        }
    }

    /// Phi-3-mini-4k-instruct (microsoft/Phi-3-mini-4k-instruct
    /// config.json): the second llama_like configuration the declaration
    /// covers. MHA (32 q = 32 kv heads), no qk-norm, untied lm_head
    /// (`tie_word_embeddings: false`). `head_dim` is the logical 96
    /// (hidden 3072 / 32 heads); the kernel-side 96 → 128 pad is backend
    /// knowledge the trace deliberately lacks, as are `sliding_window:
    /// 2047` and rope scaling (null here) — none of them change WHAT the
    /// pass computes, only how the driver launches it. `fused_qkv: false`
    /// is the binding fact, and a mildly surprising one: the checkpoint
    /// ships `qkv_proj` pre-fused, but the loader contract SPLITS it into
    /// banded q/k/v views (`llama_like_contract.hpp` phi3_fused_splits)
    /// and the CUDA dense join only re-fuses raw source tensors — a
    /// contract-derived band is not one — so the deployment binds three
    /// projections and the trace writes three matmuls (verified against
    /// the live binding: the declared-forward trace reports the 387-op
    /// unfused form, 12 ops x 32 layers + 3).
    pub fn phi3_mini() -> Self {
        Self {
            hidden: 3072,
            layers: 32,
            q_heads: 32,
            kv_heads: 32,
            head_dim: 96,
            intermediate: 8192,
            vocab: 32_064,
            rope: RopeKind::Standard,
            norm_variant: NormVariant::Plain,
            norm_placement: NormPlacement::Pre,
            qk_norm: QkNorm::Off,
            fused_qkv: false,
            tied_embeddings: false,
            qkv_bias: false,
        }
    }

    /// Mistral-7B-Instruct-v0.3 (mistralai/Mistral-7B-Instruct-v0.3
    /// config.json): the third llama_like configuration the declaration
    /// covers, and the first to combine the fused QKV binding with no
    /// qk-norm — qwen3 is fused + qk-norm, phi3 unfused + no qk-norm, so
    /// every fact here exercises an existing branch and only the
    /// combination is new. GQA (32 q / 8 kv heads), head_dim 128 (hidden
    /// 4096 / 32 — no kernel pad, unlike phi3's 96), untied lm_head
    /// (`tie_word_embeddings: false`, and the checkpoint ships
    /// `lm_head.weight`). `rope_theta: 1e6`, null rope scaling and
    /// `sliding_window: null` are backend cfg the trace deliberately
    /// lacks. `fused_qkv: true` is the binding fact, the mirror image of
    /// phi3's: the checkpoint ships three raw BF16 q/k/v projections
    /// under the canonical names, and the CUDA dense join
    /// (`contract.hpp::dense_fused_projection_joins`) re-fuses exactly
    /// such raw source tensors into `qkv_proj.fused` — so the deployment
    /// binds one packed projection and the trace writes Matmul(qkv) +
    /// SplitQkv (verified against the live binding: the declared-forward
    /// trace reports the 355-op fused form, 11 ops x 32 layers + 3).
    pub fn mistral_7b_v03() -> Self {
        Self {
            hidden: 4096,
            layers: 32,
            q_heads: 32,
            kv_heads: 8,
            head_dim: 128,
            intermediate: 14_336,
            vocab: 32_768,
            rope: RopeKind::Standard,
            norm_variant: NormVariant::Plain,
            norm_placement: NormPlacement::Pre,
            qk_norm: QkNorm::Off,
            fused_qkv: true,
            tied_embeddings: false,
            qkv_bias: false,
        }
    }

    /// OLMo-2-0425-1B-Instruct (allenai/OLMo-2-0425-1B-Instruct
    /// config.json): the fourth llama_like configuration, and the first
    /// that extends the declaration itself rather than recombining
    /// existing branches. Two genuinely new facts:
    ///
    /// * `norm_placement: Post` — the checkpoint ships
    ///   `post_attention_layernorm` + `post_feedforward_layernorm` and NO
    ///   `input_layernorm`; each sub-layer reads the residual stream raw,
    ///   norms its own output, and a separate residual add lands it
    ///   (`kernels::norm::residual_add_bf16` in the hand-written post-norm walk).
    /// * `qk_norm: Global` — the checkpoint's `q_norm`/`k_norm` weights
    ///   are shape `[2048]` = heads x head_dim (verified against the
    ///   safetensors header), NOT `[128]`: one RMSNorm over the flattened
    ///   projection, the `rmsnorm_qk` global branch. (The "per-head for
    ///   OLMo-2 small" note in llama_like.cpp is wrong for this 1B
    ///   checkpoint — the tensor shape is the truth.)
    ///
    /// MHA (16 q = 16 kv heads), head_dim 128 (hidden 2048 / 16 — no
    /// config `head_dim` key; the derivation matches the driver's),
    /// untied lm_head (`tie_word_embeddings: false`, `lm_head.weight`
    /// ships). `attention_bias: false`, so no qkv-bias branch is needed.
    /// `rope_theta: 5e5` and null rope scaling are backend cfg the trace
    /// deliberately lacks. `fused_qkv: false` is the binding fact:
    /// although the dense join re-fuses the raw q/k/v into
    /// `qkv_proj.fused`, `bind_olmo3` (qwen3.cpp) never reads the fused
    /// names — it binds the per-projection views — so the deployment runs
    /// three projection GEMMs and the trace writes three matmuls. (Same
    /// for gate/up: bound unfused, but that is emitter dispatch on the
    /// single traced `gate_up` matmul, not a fact.)
    pub fn olmo2_1b() -> Self {
        Self {
            hidden: 2048,
            layers: 16,
            q_heads: 16,
            kv_heads: 16,
            head_dim: 128,
            intermediate: 8192,
            vocab: 100_352,
            rope: RopeKind::Standard,
            norm_variant: NormVariant::Plain,
            norm_placement: NormPlacement::Post,
            qk_norm: QkNorm::Global,
            fused_qkv: false,
            tied_embeddings: false,
            qkv_bias: false,
        }
    }
}

/// CUDA backend facts for a LOWERED llama_like trace
/// (`family::llama_like_cuda`; north-star-dsl.md).
///
/// Everything here is load-time — env defaults, kernel-support predicates
/// over the head geometry, what the deployment's binding materialized —
/// exactly the terms `context.cpp:1419` and the executor's path booleans
/// derive today. The declaration's class arms consume them the way they
/// consume model facts; the driver's job shrinks to VALIDATING at boot
/// that its own derivation agrees (the `declared_facts.cpp` pattern), not
/// choosing.
///
/// Like every fact struct: measured per deployment, provenance-pinned in
/// the constructors, never silently derived.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LlamaLikeCudaFacts {
    /// XQA decode eligibility: `PIE_CUDA_XQA_DECODE` (default on) &&
    /// `xqa_decode_bf16_supported(heads, head_dim_kernel, page_size,
    /// window)` && all-full-attention && native-bf16 cache && !HND layout
    /// (context.cpp:1419-1425, llama_like.cpp:693-701).
    pub xqa_decode: bool,
    /// The fused decode-QKV epilogue is live: `decode_fused_post_enabled`
    /// (env, default on) && native-bf16 cache && unpadded head_dim &&
    /// no qkv bias — the load-time terms of `fused_decode_qkv_post`
    /// (declared_forward.cpp:465-479). The trace-time terms (`fused_qkv`,
    /// per-head qk-norm, Standard rope) live on [`LlamaLikeFacts`] and
    /// the declaration checks both.
    pub decode_fused_post: bool,
    /// The workspace carries a rope table (`ws.rope_table` non-empty), so
    /// the fused arm's first layer states [`model_compiler::trace::OpKind::RopeTableBuild`];
    /// without it the fused kernel derives cos/sin from theta in-kernel
    /// and no table launch exists.
    pub rope_table: bool,
    /// FlashInfer's decode kernel set lacks this model's GQA ratio
    /// (`!flashinfer_decode_supports_gqa`, context.cpp:1413-1414): decode
    /// fires fall back to dequant + the prefill kernel
    /// ([`model_compiler::trace::AttnKernel::PrefillDequantDecode`]). XQA, when
    /// eligible, overrides this (context.cpp:1427).
    pub force_prefill_path: bool,
    /// The attention kernels run at a padded `head_dim_kernel` wider than
    /// the logical head dim (Phi-3-mini: 96 → 128). A load-time fact
    /// (`cfg.head_dim != cfg.head_dim_kernel`): the generated form stages
    /// zero-padded q/k/v copies around the KV write, overrides the
    /// softmax scale to `1/sqrt(d)`, and strips the attention output —
    /// the hand-written `head_dim_padded` branches, resolved at emission.
    /// Serde-defaulted (append-only discipline).
    #[serde(default)]
    pub head_dim_padded: bool,
    /// The head width the ATTENTION kernels run at (Phi-3-mini: 128 for
    /// a logical 96), or 0 for a deployment that runs at the logical
    /// one.
    ///
    /// [`Self::head_dim_padded`] is exactly `head_dim_kernel != 0`, and
    /// both are here because the bool is in the digest and the WIDTH is
    /// what a statement needs: `cuda::pad_head_dim` produces a value
    /// whose shape is `heads * head_dim_kernel`, and a shape is not
    /// something a boolean gives. Sixteen executor sites read the bool
    /// and re-derived the width from config; the pads and the strip are
    /// statements now, so the width crosses with them.
    ///
    /// Serde-defaulted (append-only discipline).
    #[serde(default)]
    pub head_dim_kernel: u32,
    /// The checkpoint materialised a packed gate‖up bank
    /// (`w.layers[l].gate_up_proj_fused != nullptr`), so the MLP's packed
    /// GEMM lands in one buffer and the activation is the CHUNKED swiglu
    /// over it; without it the projection writes two buffers and the
    /// activation is the pair form.
    ///
    /// A pure WEIGHT-BINDING fact, which is why it belongs here rather
    /// than in the walk. The executor derived it per layer as
    /// `gate_up_proj_fused != nullptr && !ws.gate_up_fused.empty()`, but
    /// the second term is dead: `workspace.cpp` allocates that buffer
    /// unconditionally ("Always allocated … lets the forward dispatch
    /// decide per layer"). So the binding alone decides, the binding is
    /// known at model construction, and the taxonomy's first row applies
    /// — a load-time fact is a trace-time `match`, erased.
    ///
    /// Stating it deletes a runtime branch from three places at once: the
    /// interpreter's `arm_swiglu`, the generated `.inc`'s
    /// `if (gate_up_fused_N)` per layer, and the flat list's residue.
    /// Serde-defaulted (append-only discipline).
    #[serde(default)]
    pub gate_up_fused: bool,
    /// How this deployment STORES its linear projections — the weight
    /// representation axis ([`model_compiler::dsl::WeightRepr`]).
    ///
    /// A pure binding fact, like [`Self::gate_up_fused`], and the last
    /// one the driver was answering for itself: `make_weight_view` built
    /// a `WeightView` out of a per-layer `QuantMeta` the statement never
    /// mentioned and `gemm::act_x_w` routed on it — ten call sites here
    /// and eight in qwen3.5, every one of them the driver knowing
    /// something the declaration did not.
    ///
    /// ONE repr for the whole deployment rather than one per projection,
    /// because a checkpoint quantizes uniformly and the build gate
    /// refuses a mixed binding by name. Where a checkpoint ever does
    /// mix, this becomes a field per projection and the text asks the
    /// facts per handle — nothing else changes, which is the point of
    /// putting the axis on the WEIGHT.
    ///
    /// Serde-defaulted to dense (append-only discipline), so a fixture
    /// written before this field reads exactly as it did.
    #[serde(default)]
    pub proj_repr: WeightRepr,
    /// How many ranks this deployment shards its layers across
    /// (`LlamaLikeForwardCfg::tp_size`), or 0/1 for a single GPU.
    ///
    /// SHARDING NEEDS NO VOCABULARY: a rank's trace states ITS widths,
    /// and the text divides by this the way it divides by anything
    /// else. What needs vocabulary is the point where the shards are
    /// recombined, because that is a launch — `dist::all_reduce_bf16`
    /// and its two friends, which the text states.
    ///
    /// So this fact does two things and neither is a switch the driver
    /// reads: it narrows the projection widths, and it decides whether
    /// the landing statements exist at all.
    ///
    /// Serde-defaulted (append-only discipline); 0 reads as one rank.
    #[serde(default)]
    pub tp_size: u32,
    /// The SLIDING WINDOW each layer attends over, `-1` for none.
    ///
    /// Empty means every layer is `-1` — a deployment with no window at
    /// all — which is why the accessor and not the field is what texts
    /// read ([`Self::window_left_at`]).
    ///
    /// A load-time fact: a config's `sliding_window`, or its per-layer
    /// list where the architecture alternates (OLMo-3, Mistral). Eleven
    /// executor sites across four families derived it by reaching into
    /// `fwd_cfg.per_layer_window_left` — a per-layer array no statement
    /// mentioned — and the dispatch statements carry it now.
    ///
    /// The per-FIRE override (`runtime_window_left`) is NOT this. That
    /// is a runtime input and wants a guard predicate;
    /// `DeclineReason::SlidingWindow` still names it.
    ///
    /// Serde-defaulted (append-only discipline).
    #[serde(default)]
    pub window_left: Vec<i32>,
    /// Rows below which an all-reduce takes the NVLink P2P kernel
    /// instead of NCCL, or 0 for a deployment that always takes NCCL.
    ///
    /// `NcclComm::all_reduce_bf16` asks `can_handle(bytes)` and routes
    /// on the answer — a driver picking between two implementations,
    /// which is what this replaces. The text states the pair as a guard
    /// and this is its predicate's payload.
    ///
    /// It is a ROW count and the kernel's test is BYTES, which is the
    /// same question once `hidden` is known: a row is `hidden` bf16
    /// elements. Converting here rather than in the arm is the point —
    /// a load-time fact becomes a trace-time constant.
    ///
    /// ZERO also covers the deployment that registered no P2P buffers.
    /// The kernel reads only registered memory, which is a placement
    /// fact rather than a size one, and a deployment that has none has
    /// no threshold either.
    ///
    /// Serde-defaulted (append-only discipline).
    #[serde(default)]
    pub all_reduce_p2p_max_rows: u32,
}

/// The METAL backend's load-time facts — what the Metal deployment
/// resolved before any fire, the way [`LlamaLikeCudaFacts`] carries
/// CUDA's.
///
/// UNVERIFIED (2026-08-05): no Metal deployment has produced these yet.
/// The Metal driver cannot even build on the box we have (`xcrun --find
/// metal` fails — the shader compiler ships with full Xcode), so every
/// field here is read off the driver's SOURCE
/// (`crates/driver-metal/csrc/src/batch/decode_psos.cpp`, `model/qwen3_5/decode_step.hpp`)
/// rather than measured. `.wiki/tart/macos.md` records the ladder; the
/// precedent for refusing to call an unmeasured fact set measured is
/// [`Qwen35CudaFacts::qwen3_5_0_8b_synthetic`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LlamaLikeMetalFacts {
    /// The projection GEMV folds the block residual in its epilogue
    /// (`affine_qmv_fast_residual`, `Dispatch::fuse_residual`,
    /// `PIE_FUSE_RESIDUAL`), so a `beta_one` matmul states one launch
    /// instead of a projection plus a `residual_add`.
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
    /// Same role `LlamaLikeCudaFacts::proj_repr` plays and for the same
    /// reason: `LlamaLikeFacts::shape()` answers `Bf16` because the semantic
    /// facts carry no backend, and a trace with no backend cannot name the
    /// kernel a scaled weight needs. The backend facts do carry one, so this
    /// is where the representation reaches the namespace.
    ///
    /// It is load-bearing beyond the kernel name: an affine kernel reads
    /// THREE tensors, and `MatW::scale_names` is what makes the statement say
    /// so. A text that left this dense named `affine_qmv_fast` while stating
    /// one weight, and the driver would have had to derive the other two from
    /// a naming convention it was never told.
    ///
    /// Serde-defaulted, so a fixture written before this field reads as it did.
    #[serde(default)]
    pub proj_repr: model_compiler::dsl::WeightRepr,
    /// Bits per packed weight element — 4 or 8.
    ///
    /// The affine entrypoints are instantiated over `(group size × bit
    /// width)`, so the SYMBOL a statement names carries both. `proj_repr`
    /// already carries the group; this is the other half, and it is a separate
    /// field because `WeightRepr::Scaled` has nowhere to put it (a bit width
    /// is a property of the weight's dtype, which the trace does not spell).
    ///
    /// This is the `_d_256` lesson generalised: a symbol whose axis point is
    /// wrong does not fail, it reads the wrong bytes. A symbol that is a bare
    /// STEM does not resolve at all, which is the better failure and the one
    /// the runtime compiler reports by listing what the shader does export.
    #[serde(default)]
    pub affine_bits: u32,
    /// The GEMM's `(row tile, column tile)`, as the entrypoint spells them.
    ///
    /// `affine_qmm_t` is instantiated over `(group × bits × bm × bn)`, so the
    /// batched projection's symbol carries a TILE — and a tile is chosen from
    /// the ROW COUNT, which is a fire-time fact a trace cannot know.
    ///
    /// So it is a load-time fact instead, and that is the honest reading of
    /// what the driver was doing: `qmm_bm` picks the widest rung at or under
    /// `n`, and a deployment that always fires the same window always picks
    /// the same one. A deployment that wants the tile chosen per fire needs
    /// the row count on a guard axis, which is a change to `Row` and not to
    /// this text.
    ///
    /// Serde-defaulted to `(0, 0)`, which spells no tile — right for the
    /// GEMV-only deployments, wrong loudly for a GEMM one, because a symbol
    /// with no tile does not resolve.
    #[serde(default)]
    pub qmm_tile: (u32, u32),
    /// The deployment bound ONE packed `gate‖up` bank.
    ///
    /// A binding fact, exactly as `LlamaLikeFacts::fused_qkv` is, and the
    /// Metal answer to it is normally **false**: `compile_load_plan` authors
    /// with `Projections::InPlace`, and `dense_fused_projection_joins` returns
    /// before doing anything under that policy. So the MLX path publishes
    /// `mlp.gate_proj` and `mlp.up_proj` separately and the text must state
    /// two projections.
    ///
    /// It matters more than a name: `mlp/gated.metal::silu_mul` takes **gate
    /// and up as two buffers**, so a text that states one packed value binds
    /// the OUTPUT where `up` belongs and leaves the output unbound. That is a
    /// fire that runs.
    #[serde(default)]
    pub gate_up_fused: bool,
}

impl LlamaLikeMetalFacts {
    /// A SYNTHETIC fixture, not a measurement — see the struct comment.
    /// These are the driver's own defaults as its source reads them.
    pub fn synthetic() -> Self {
        Self {
            fuse_residual_gemv: true,
            paged_multi_batch: true,
            qmm_multi_batch: true,
            // The symbols this text names are the affine ones
            // (`affine_qmv_fast`, `embed_gather_4bit`), so the deployment
            // they describe is a quantized checkpoint. MLX stores the pair
            // beside the packed weight as `.scales` and `.biases`, which is
            // a zero-point layout.
            proj_repr: model_compiler::dsl::WeightRepr::Scaled {
                layout: model_compiler::dsl::ScaleLayout::PerGroup,
                group: 64,
                axis: 0,
                zero_point: true,
            },
            affine_bits: 4,
            // The narrowest rung `qmm_bm` can pick, so it is the one a short
            // window fires; `bn = 32` is the only column tile the residual
            // variant is instantiated at.
            qmm_tile: (16, 32),
            // `Projections::InPlace` is what `compile_load_plan` authors with,
            // and the join declines under it.
            gate_up_fused: false,
        }
    }
}

impl LlamaLikeCudaFacts {
    /// The window layer `l` attends over, `-1` for none.
    ///
    /// A short list is not an error: a deployment whose config carries
    /// ONE `sliding_window` states a one-element list and every layer
    /// reads it, which is what the drivers' `per_layer_window_left`
    /// fallback meant.
    pub fn window_left_at(&self, l: u32) -> i32 {
        model_compiler::facts::window_left_at(&self.window_left, l)
    }

    /// Qwen3-0.6B on L40S, default env — MEASURED 2026-08-02 against the
    /// driver's own derivation via the rung-3 digest print
    /// (`PIE_DECLARED_FORWARD_TRACE=1` + `..._GENERATED=1`; the live
    /// digest is the provenance):
    /// `.../qk1/fq1/te0/xqa0/dfp1/rt1/fpp0`.
    ///
    /// `xqa_decode: false` — the first version of this constructor
    /// guessed `true` from the geometry and called it measured; the
    /// digest mechanism caught the lie on its first live run
    /// (`fwd_cfg.use_xqa_decode` derives false on this deployment).
    /// The decode class therefore states the FlashInfer decode kernel.
    /// Note the deployment's BINDING also unties the lm_head
    /// (`w.lm_head != w.embed`, live `te0`) even though the checkpoint's
    /// config ties it — the model-facts fixture [`LlamaLikeFacts::qwen3_0_6b`]
    /// keeps the config-level fact for the semantic goldens; emission
    /// against the live deployment overrides it (`emit-cuda`).
    pub fn qwen3_0_6b_l40s() -> Self {
        Self {
            xqa_decode: false,
            decode_fused_post: true,
            rope_table: true,
            force_prefill_path: false,
            head_dim_padded: false,
            head_dim_kernel: 0,
            // The loader's `dense_fused_projection_joins` contract packs
            // BF16 dense groups and declines quantized ones, so a plain
            // BF16 deployment carries the bank. VERIFIED LIVE, not
            // assumed — the boot log's declared-facts line is what says
            // so, and the digest refuses the deployment if this fixture
            // and the binding disagree.
            gate_up_fused: true,
            // Dense. The same contract line says so: a group it packs is
            // a BF16 one, so a deployment that carries the bank cannot
            // be quantized.
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

