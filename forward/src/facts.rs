//! Load-time facts a declaration traces against.
//!
//! These are the `config.json`-derived values that the hand-written
//! `LlamaLikeForwardCfg` + `HfConfig` pair carries into the forward today,
//! reduced to what the *declaration* needs: everything here is resolved at
//! trace time and none of it survives into the traced form except as
//! constants and op choices.

use serde::{Deserialize, Serialize};

use crate::trace::{NormVariant, RopeKind};

/// Where each sub-layer's norm sits relative to the residual add.
///
/// `Pre` is the standard Llama shape: norm the residual stream *into* the
/// sub-layer, accumulate the sub-layer's projection straight back onto the
/// stream (the `beta=1` GEMM). `Post` is the OLMo-2/OLMo-3 shape: the
/// sub-layer reads the residual stream raw, the norm applies to the
/// sub-layer's OUTPUT, and only then does a separate residual add land it —
/// a genuinely different op ORDER, which is why it is a fact and not an
/// emitter choice. Mirrors the driver's `NormPlacement`
/// (`driver/cuda/src/model/llama_like/llama_like.hpp`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum NormPlacement {
    #[default]
    Pre,
    Post,
}

/// Which q/k-norm convention the checkpoint ships, if any.
///
/// Two conventions exist in the wild (the driver's `rmsnorm_qk` dispatch,
/// `llama_like.cpp`): *per-head* (qwen3, gemma-3 — weight shape
/// `[head_dim]`, each head's channels normalised independently) and
/// *global* (OLMo-2, OLMo-3 — weight shape `[heads * head_dim]`, ONE
/// RMSNorm over the flattened projection). They are different arithmetic —
/// the global form shares one scale across heads — so the tri-state is a
/// fact, and the traced ops differ: per-head traces `RmsnormPerHead`,
/// global traces a plain row `Rmsnorm` (which is exactly what the kernel
/// launches: `launch_rmsnorm_bf16` over `[N, heads * head_dim]`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum QkNorm {
    #[default]
    Off,
    PerHead,
    Global,
}

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
    ///   (`launch_residual_add_bf16` in the hand-written post-norm walk).
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
    /// the fused arm's first layer states [`crate::trace::OpKind::RopeTableBuild`];
    /// without it the fused kernel derives cos/sin from theta in-kernel
    /// and no table launch exists.
    pub rope_table: bool,
    /// FlashInfer's decode kernel set lacks this model's GQA ratio
    /// (`!flashinfer_decode_supports_gqa`, context.cpp:1413-1414): decode
    /// fires fall back to dequant + the prefill kernel
    /// ([`crate::trace::AttnKernel::PrefillDequantDecode`]). XQA, when
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
}

/// The METAL backend's load-time facts — what the Metal deployment
/// resolved before any fire, the way [`LlamaLikeCudaFacts`] carries
/// CUDA's.
///
/// UNVERIFIED (2026-08-05): no Metal deployment has produced these yet.
/// The Metal driver cannot even build on the box we have (`xcrun --find
/// metal` fails — the shader compiler ships with full Xcode), so every
/// field here is read off the driver's SOURCE
/// (`driver/metal/src/kernels/decode_psos.cpp`, `model/qwen3_5/decode_step.hpp`)
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
}

impl LlamaLikeMetalFacts {
    /// A SYNTHETIC fixture, not a measurement — see the struct comment.
    /// These are the driver's own defaults as its source reads them.
    pub fn synthetic() -> Self {
        Self {
            fuse_residual_gemv: true,
            paged_multi_batch: true,
            qmm_multi_batch: true,
        }
    }
}

impl LlamaLikeCudaFacts {
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
            // The loader's `dense_fused_projection_joins` contract packs
            // BF16 dense groups and declines quantized ones, so a plain
            // BF16 deployment carries the bank. VERIFIED LIVE, not
            // assumed — the boot log's declared-facts line is what says
            // so, and the digest refuses the deployment if this fixture
            // and the binding disagree.
            gate_up_fused: true,
        }
    }
}

/// Facts for one qwen3_5_moe-family MoE MLP block — a traced FRAGMENT, not
/// a model.
///
/// Deliberately narrower than a `Qwen35MoeFacts` would be: the qwen3.5
/// family alternates GDN linear-attention layers with full-attention layers
/// (the HYBRID part, `driver/cuda/src/model/qwen3_5/qwen3_5_forward.cpp`),
/// and declaring the MoE MLP inside the llama_like skeleton would trace a
/// model that does not exist. So these facts describe exactly the unit the
/// qwen3_5 hybrid composes per layer — `y += moe_mlp(rmsnorm(y))`, the
/// [`Qwen35MlpKind::Moe`] arm of [`Qwen35HybridFacts`] — and
/// `family::qwen3_5_moe_mlp_block` traces that unit standalone. The GDN
/// attention half is [`Qwen35GdnFacts`] / `family::qwen3_5_gdn_block` —
/// its own fragment, same reasoning.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Qwen35MoeMlpFacts {
    pub hidden: u32,
    /// Routed expert count (HF `num_experts` / `num_local_experts`).
    pub num_experts: u32,
    /// Experts per token (HF `num_experts_per_tok`) — the router's k.
    pub top_k: u32,
    /// Per-expert intermediate width (HF `moe_intermediate_size`).
    pub moe_intermediate: u32,
    /// Shared-expert intermediate width (HF
    /// `shared_expert_intermediate_size`); 0 means no shared expert, which
    /// is the qwen3_moe shape — the hand-written pass skips the whole
    /// shared block when the bind wired no `shared_*` pointers.
    pub shared_expert_intermediate: u32,
    /// qwen3.5/3.6 use the Gemma `(1 + w)` fold
    /// (`qwen3_5_moe_forward.cpp::uses_gemma_rmsnorm`: everything but plain
    /// `qwen3_moe`).
    pub norm_variant: NormVariant,
}

impl Qwen35MoeMlpFacts {
    /// Qwen3.5-35B-A3B, the small qwen3_5_moe checkpoint.
    ///
    /// No config.json is committed in this tree, so these dims are pinned
    /// from the driver's own measured notes on this checkpoint
    /// (`qwen3_5_moe_forward.cpp`): 256 routed experts ("with 256 experts
    /// holding only a few routes each"); gate_up bytes per expert 4.2 MB at
    /// tp=1 / 2.1 MB at tp=2 = `2 * moe_intermediate * hidden * 2B` with
    /// `moe_intermediate = 512, hidden = 2048`; top-k 8 (the profiled
    /// N=128 decode step's "352 blocks for ~252 active experts" matches
    /// the aligned-decode block formula only at `routes = N * 8`); and a
    /// shared expert with `Is == Im` (the precondition of the shared-fold
    /// experiment, which rode it along "as one more expert").
    pub fn qwen3_5_35b_a3b() -> Self {
        Self {
            hidden: 2048,
            num_experts: 256,
            top_k: 8,
            moe_intermediate: 512,
            shared_expert_intermediate: 512,
            norm_variant: NormVariant::Gemma,
        }
    }
}

/// Facts for one qwen3_5 GDN (gated-deltanet) linear-attention block — the
/// second traced FRAGMENT, and the other layer kind of the qwen3.5 hybrid.
///
/// Describes exactly the unit the qwen3_5 hybrid composes on a `Linear`
/// layer — `y += gdn(l, rmsnorm(y, attn_norm))` (plan.md Part 1's
/// `match layers[l] { ..., Linear => gdn(l, x, h) }`) — traced standalone by
/// `family::qwen3_5_gdn_block`, mirroring
/// `qwen3_5_forward.cpp::linear_attn_layer_body` launch for launch. The
/// full-attention layer kind is [`Qwen35FullAttnFacts`] /
/// `family::qwen3_5_full_attn_block` — its own fragment, same reasoning —
/// and [`Qwen35HybridFacts`] composes both into the full model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Qwen35GdnFacts {
    pub hidden: u32,
    /// GDN key heads (HF `linear_num_key_heads`).
    pub key_heads: u32,
    /// GDN value heads (HF `linear_num_value_heads`); a multiple of
    /// `key_heads` (GQA share) or equal to it.
    pub value_heads: u32,
    /// Per-head key width (HF `linear_key_head_dim`).
    pub key_head_dim: u32,
    /// Per-head value width (HF `linear_value_head_dim`).
    pub value_head_dim: u32,
    /// Depthwise conv window (HF `linear_conv_kernel_dim`).
    pub conv_kernel: u32,
    /// The deployment bound the fused `in_proj_qkvz` + `in_proj_ba` banks.
    /// A *binding* fact, llama_like's `fused_qkv` precedent: the checkpoint
    /// ships four raw projections (`in_proj_{qkv,z,b,a}`) and the CUDA
    /// contract's `gdn_fused_in_proj_joins` re-joins them — but only behind
    /// `PIE_QWEN35_FUSED_GDN_PROJ` (default OFF,
    /// `qwen35_fused_gdn_projection_enabled()`), so the default deployment
    /// binds four projections and the trace writes four matmuls; with the
    /// join enabled it writes two matmuls + two `SplitGdn`s
    /// (`qwen3_5_forward.cpp` branches on `la_in_proj_qkvz`/`la_in_proj_ba`
    /// the same way).
    pub fused_in_proj: bool,
    /// qwen3.5/3.6 use the Gemma `(1 + w)` fold for the block norms
    /// (`launch_rmsnorm_gemma_bf16` on the pre-attention norm). The GATED
    /// norm inside the block is not governed by this: its weight fold is
    /// plain by kernel contract (`rmsnorm.hpp`).
    pub norm_variant: NormVariant,
}

impl Qwen35GdnFacts {
    /// `key_heads * key_head_dim` — one leg of the packed conv input.
    pub fn key_width(&self) -> u32 {
        self.key_heads * self.key_head_dim
    }

    /// `value_heads * value_head_dim` — the v leg, the z gate width, and
    /// the o_proj input width.
    pub fn value_width(&self) -> u32 {
        self.value_heads * self.value_head_dim
    }

    /// The packed `[q | k | v]` conv width: `2 * key_width + value_width`.
    pub fn conv_dim(&self) -> u32 {
        2 * self.key_width() + self.value_width()
    }

    /// Qwen3.5-0.8B, the workspace's linear-attention parity checkpoint
    /// (`driver/cuda/tests/parity_qwen3_5_multireq.py` defaults to
    /// `Qwen/Qwen3.5-0.8B-Base`).
    ///
    /// No config.json is committed in this tree, so every dimension is
    /// pinned from the drivers' own statements of this checkpoint:
    ///
    /// * `driver/metal/src/model/qwen3_5/geometry.hpp` (`DecodeGeometry`
    ///   defaults, the Metal driver's 0.8B target): `hidden = 1024`,
    ///   `gdn_k_heads = 16`, `gdn_v_heads = 16`, `gdn_k_dim = 128`,
    ///   `gdn_v_dim = 128`, `gdn_conv_k = 4`, `gdn_conv_dim = 6144`,
    ///   `gdn_v_total = 2048` — and `conv_dim()`/`value_width()` here
    ///   reproduce those last two (2·2048 + 2048 = 6144, 16·128 = 2048).
    /// * `driver/metal/src/model/qwen3_5/decode_consts.cpp` corroborates
    ///   the widths as launch geometry: in-proj 1024 → 6144, z 1024 →
    ///   2048, out-proj 2048 → 1024, "in_proj_a / in_proj_b — DENSE bf16
    ///   GEMV [16, 1024]" (= value_heads × hidden).
    /// * `driver/cuda/src/model/config.hpp:357` pins the conv window: 4.
    /// * `fused_in_proj: false` is the live default binding
    ///   (`PIE_QWEN35_FUSED_GDN_PROJ` unset — see the field doc).
    /// * `norm_variant: Gemma`: `qwen3_5_forward.cpp` launches
    ///   `launch_rmsnorm_gemma_bf16` for every block norm, and the Metal
    ///   port states "All RMSNorm gains use the Gemma (1+w) convention"
    ///   (`driver/metal/tests/mlx/model/qwen3_5.hpp`).
    pub fn qwen3_5_0_8b() -> Self {
        Self {
            hidden: 1024,
            key_heads: 16,
            value_heads: 16,
            key_head_dim: 128,
            value_head_dim: 128,
            conv_kernel: 4,
            fused_in_proj: false,
            norm_variant: NormVariant::Gemma,
        }
    }
}

/// Facts for one qwen3_5 FULL-attention block — the third traced FRAGMENT,
/// and the last layer kind the qwen3.5 hybrid needed.
///
/// This is NOT llama_like's attention, which is why it gets its own facts
/// instead of a `LlamaLikeFacts` configuration: the q projection is 2× wide
/// with an interleaved per-head `[query | gate]` split
/// (`launch_split_q_gate_bf16`), the attention output is multiplied by
/// `sigmoid(gate)` (`launch_sigmoid_gate_inplace_bf16` — no residual, not
/// the shared-expert `SigmoidGateAdd`), rope is PARTIAL
/// (`partial_rotary_factor`, `launch_rope_partial_bf16`), and the per-head
/// q/k norms fold Gemma-style (`launch_rmsnorm_gemma_bf16` over `N * heads`
/// rows of `head_dim`). The qk-norm is not a tri-state here: the
/// hand-written `full_attn_layer_body` launches the per-head pair
/// unconditionally, so the declaration does too, and only the fold is a
/// fact ([`Qwen35FullAttnFacts::norm_variant`]).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Qwen35FullAttnFacts {
    pub hidden: u32,
    pub q_heads: u32,
    pub kv_heads: u32,
    pub head_dim: u32,
    /// Partial-rotary width: the leading channels of each head that rotate
    /// (`OpKind::Rope`'s `partial`). Stated as the resolved channel count,
    /// not HF's factor — the driver's `max(2, 2 * int(0.5 *
    /// partial_rotary_factor * head_dim))` derivation
    /// (`qwen3_5_forward.cpp`) is config parsing, and the fixture pins its
    /// result with provenance.
    pub rotary_dim: u32,
    /// The deployment bound one packed `[2q | k | v]` projection
    /// (`fa_qgkv_proj_fused`). A *binding* fact, llama_like's `fused_qkv`
    /// precedent: the join is env-gated default-OFF
    /// (`PIE_QWEN35_FUSED_FULL_ATTN_QGKV`,
    /// `qwen3_5.cpp::fused_full_attn_qgkv_weights_enabled`), so the default
    /// deployment binds three projections and the trace writes three
    /// matmuls; with the join enabled it writes Matmul(qgkv) + SplitQkv
    /// whose "q" leg is the 2×-wide `[query | gate]` bank
    /// (`full_attn_layer_body`'s `use_fused_qgkv` branch:
    /// `launch_split_qkv_bf16(packed, qg, k, v, N, 2*Hq, Hk)`).
    pub fused_qkv: bool,
    /// qwen3.5 folds `(1 + w)` on every norm of this block — the
    /// pre-attention norm AND the per-head q/k norms
    /// (`launch_rmsnorm_gemma_bf16` throughout `full_attn_layer_body`).
    pub norm_variant: NormVariant,
}

impl Qwen35FullAttnFacts {
    pub fn q_width(&self) -> u32 {
        self.q_heads * self.head_dim
    }

    pub fn kv_width(&self) -> u32 {
        self.kv_heads * self.head_dim
    }

    /// Qwen3.5-0.8B's full-attention geometry, pinned from the drivers'
    /// own statements of this checkpoint (no config.json is committed;
    /// same provenance discipline as [`Qwen35GdnFacts::qwen3_5_0_8b`]):
    ///
    /// * `driver/metal/src/model/qwen3_5/geometry.hpp` (`DecodeGeometry`
    ///   defaults, the Metal driver's 0.8B target): `hidden = 1024`,
    ///   `n_q_heads = 8`, `n_kv_heads = 2`, `head_dim = 256`,
    ///   `rotary_dims = 64` ("derived from partial_rotary_factor *
    ///   head_dim").
    /// * `driver/metal/src/model/qwen3_5/decode_consts.cpp` corroborates
    ///   the widths as launch geometry: "2×-wide gated q_proj (4096)" =
    ///   2 · 8 · 256, k/v 1024 → 512 = 2 · 256, o_proj 2048 → 1024.
    /// * `rotary_dim = 64`: `partial_rotary_factor = 0.25` (the family
    ///   default `driver/metal/src/batch/forward.hpp` states and
    ///   `driver/cuda/src/model/qwen3_5/qwen3_5.hpp` documents —
    ///   "`partial_rotary_factor=0.25` — only the first 25% of head_dim is
    ///   rotated"); the CUDA derivation `max(2, 2·int(0.5·0.25·256))`
    ///   (`qwen3_5_forward.cpp`) and Metal's `rotary_dims` both land on 64.
    /// * `fused_qkv: false` is the live default binding
    ///   (`PIE_QWEN35_FUSED_FULL_ATTN_QGKV` unset — see the field doc).
    /// * `norm_variant: Gemma`: `full_attn_layer_body` launches
    ///   `launch_rmsnorm_gemma_bf16` for the block norm and both per-head
    ///   q/k norms, and the Metal port states "All RMSNorm gains use the
    ///   Gemma (1+w) convention" (`driver/metal/tests/mlx/model/qwen3_5.hpp`).
    pub fn qwen3_5_0_8b() -> Self {
        Self {
            hidden: 1024,
            q_heads: 8,
            kv_heads: 2,
            head_dim: 256,
            rotary_dim: 64,
            fused_qkv: false,
            norm_variant: NormVariant::Gemma,
        }
    }
}

/// Which MLP the qwen3_5 hybrid runs on every layer: the dense SwiGLU block
/// (qwen3.5 dense checkpoints — `qwen35_dense_mlp_block`) or the MoE block
/// (qwen3.5/3.6-MoE — `run_moe_mlp`, the [`Qwen35MoeMlpFacts`] fragment).
/// One enum for the whole model because the family applies the same MLP
/// kind to every layer (`qwen3_5_forward.cpp` has no per-layer MLP switch;
/// the per-layer axis of this family is the ATTENTION kind).
///
/// WHICH ARM A CHECKPOINT TAKES IS A READING OF ITS CONFIG, not of its
/// `model_type`. Qwen3.6-27B is `model_type: qwen3_5` and takes `Dense`
/// (no `num_experts`, `intermediate_size` 17408); the MoE arm is the
/// 35B-A3B-shaped checkpoints'. Worth stating because the opposite was
/// once assumed here, and it aimed a stretch of work at a branch the
/// checkpoint in question never reaches.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Qwen35MlpKind {
    Dense { intermediate: u32 },
    Moe(Qwen35MoeMlpFacts),
}

/// Facts for the full qwen3_5 HYBRID model — the declaration that composes
/// the three fragments: plan.md Part 1's
/// `match layers[l] { Full => full_attn(l, x), Linear => gdn(l, x) }`,
/// a static match resolved at trace time.
///
/// # How the layer kinds are known
///
/// The checkpoint states them explicitly: `config.json` ships a
/// `layer_types` array of `"linear_attention"` / `"full_attention"` with one
/// entry per layer, which is the CUDA driver's sole source
/// (`HfConfig::layer_types`, parsed in `driver/cuda/src/model/config.cpp`;
/// `qwen3_5.cpp` refuses a length mismatch). The qwen3.5 checkpoints ship a
/// REGULAR pattern — one full-attention layer every
/// `full_attention_interval`, the rest linear
/// (`driver/metal/tests/mlx/model/qwen3_5.hpp`) — and the Metal driver
/// reduces the array to exactly that interval, refusing irregular arrays
/// (`driver/metal/src/batch/forward.hpp`: "-1: `layer_types` is irregular,
/// refuse"; `driver/metal/src/model/qwen3_5/geometry.hpp::is_full_attn`).
/// These facts state the interval, mirroring that reduction: a hypothetical
/// irregular checkpoint is outside this declaration's vocabulary, exactly
/// as it is outside the Metal driver's.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Qwen35HybridFacts {
    pub layers: u32,
    /// One full-attention layer every `full_attn_interval`-th, at the END
    /// of each block: `is_full_attn(l) = l % interval == interval - 1`
    /// (the Metal geometry's formula, verbatim). `1` makes every layer
    /// full attention.
    pub full_attn_interval: u32,
    pub vocab: u32,
    /// The lm_head weight is the embedding table (weight tying).
    pub tied_embeddings: bool,
    /// The fold of the FINAL norm (the per-block norms carry their own
    /// variant inside the sub-facts; qwen3.5 folds Gemma everywhere).
    pub norm_variant: NormVariant,
    /// The full-attention layer kind.
    pub attn: Qwen35FullAttnFacts,
    /// The GDN linear-attention layer kind.
    pub gdn: Qwen35GdnFacts,
    /// The (uniform) per-layer MLP.
    pub mlp: Qwen35MlpKind,
}

/// CUDA backend facts for a LOWERED qwen3_5 hybrid trace
/// (`family::qwen3_5_hybrid_cuda`; north-star-dsl.md rung 4c).
///
/// Everything here is load-time, [`LlamaLikeCudaFacts`]-style: env
/// defaults and kernel-eligibility predicates the hand-written
/// `linear_attn_layer_body` / `declared_forward.cpp` derive per fire
/// today, hoisted to where a fact belongs. The N-thresholds are VALUES
/// carried into [`crate::trace::GuardPred`]s — the one branch kind a
/// lowered trace keeps — because only N varies per fire; the predicates
/// AROUND them (env gates, head geometry) resolve here.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Qwen35CudaFacts {
    /// The recurrent-state store dtype is bf16 (vs fp32) — the
    /// `state_bf16` parameter every GDN recurrence launcher family
    /// suffixes (`launch_recurrent_gated_delta_step_batched[_state_bf16]`
    /// and the chunked prefill families).
    pub state_bf16: bool,
    /// The warp-tiled prefill arm EXISTS at all: `K_d <= 256` && the
    /// state-persist env gate
    /// (`qwen35_gdn_warp_tiled_state_persist_enabled`,
    /// `PIE_QWEN35_GDN_WARP_TILED_STATE_PERSIST`). The hand-written
    /// predicate reads `!write_state || state_persist_enabled()` — but
    /// for the normal-fire classes (Decode/Prefill) `write_state` is
    /// always true, so the env term is the WHOLE eligibility here; the
    /// verify-frozen service classes where `write_state` is false are
    /// rung 4c-iv's, not this struct's. (`commit_lens == nullptr`, the
    /// other hand-written term, is likewise a CLASS — CommitAdvance —
    /// not a fact.)
    pub warp_tiled: bool,
    /// `qwen35_gdn_warp_tiled_max_tokens()`
    /// (`PIE_QWEN35_GDN_WARP_TILED_MAX_TOKENS`, default 64) — an
    /// env-tunable driver constant, resolved into the trace as the
    /// warp-tiled arm's `TokensLE` payload the way every fact resolves.
    pub warp_tiled_max: u32,
    /// `qwen35_gdn_cached_prefill_max_tokens()`
    /// (`PIE_QWEN35_GDN_CACHED_PREFILL_MAX_TOKENS`, default 0 = the
    /// cached family off) — the cached arm's `TokensLE` payload.
    pub cached_max: u32,
    /// The deployment configures the verify hidden stash
    /// (`RecurrentStateCache::configure_verify_hidden_stash`): the
    /// engine owns that configuration, so — like [`Self::state_bf16`],
    /// whose dtype the engine likewise decides — the fact is stated here
    /// and the driver cross-checks its own derivation per fire rather
    /// than choosing. With the stash live, the CommitAdvance class
    /// replays each linear layer's in-proj outputs from the stash
    /// (`cuda::verify_stash_load`) and skips the GEMMs; without it, the
    /// commit pass re-runs the in-projections against whatever the
    /// workspace holds. Serde-defaulted so pre-4c-iv facts JSON reads
    /// back unchanged (the append-only discipline).
    #[serde(default)]
    pub verify_stash: bool,
    /// `Qwen3_5MoeMlpWorkspace::cutlass_max_rows` — `min(max_tokens, 512)`
    /// when `ops::flashinfer_cutlass_moe_enabled()` sized a workspace,
    /// else 0. Zero means the fused leg does not exist on this
    /// deployment; non-zero is the ROW BOUND of the MoE text, and fires
    /// above it decline rather than the declaration guessing which of
    /// the remaining three legs the pass would have taken.
    #[serde(default)]
    pub moe_cutlass_max_rows: u32,
    /// `add_to_residual` — tp==1, so the MoE block's output lands on the
    /// residual stream inside this pass. At tp>1 the block writes to
    /// scratch and an allreduce follows, which is a different (and
    /// unstated) shape.
    #[serde(default)]
    pub moe_residual_fold: bool,
    /// The shared expert's gate weight is bound and unquantized, so its
    /// landing is the fused dot form. False sends it to the
    /// `[Tokens, 1]` GEMM plus a separate scalar-gate add, which this
    /// text does not state.
    #[serde(default)]
    pub moe_shared_gate_dot: bool,
    /// `Lw.expert_cache != nullptr` — the experts are paged one at a
    /// time, so every device-side leg that strides a fused slab is off
    /// the table and the pass takes the host-routed path.
    #[serde(default)]
    pub moe_streamed_experts: bool,
    /// `qwen35_moe_force_general_path()` — the env that pins the pass to
    /// the host-routed path regardless of shape.
    #[serde(default)]
    pub moe_force_general: bool,
    /// The DENSE MLP's gate_up BINDING — `Lw.gate_up_proj_fused !=
    /// nullptr`, so the packed GEMM lands in one buffer and the
    /// activation is the CHUNKED swiglu over it; without it the
    /// projection writes two and the activation is the pair form.
    ///
    /// [`LlamaLikeCudaFacts::gate_up_fused`]'s reasoning applies
    /// verbatim, including why the workspace term the executor also
    /// tested is dead. Only the MoE arm's shared expert is unaffected:
    /// it always binds a packed bank, so its text states the chunked
    /// form outright.
    #[serde(default)]
    pub gate_up_fused: bool,
}

impl Qwen35CudaFacts {
    /// SYNTHETIC fixture — NOT a measurement. These values (bf16 state,
    /// warp-tiled arm live, thresholds 64 / 4096 — plausible defaults)
    /// pin the GOLDEN FORM of the lowered qwen3_5 traces only: the live
    /// derivation and its digest validation against the driver's own
    /// booleans are rung 4c-iii. The precedent for refusing to call this
    /// "measured" is [`LlamaLikeCudaFacts::qwen3_0_6b_l40s`]: its first
    /// version guessed `xqa_decode: true` and called it measured, and the
    /// rung-3 digest caught the lie on its first live run. This
    /// constructor makes no such claim — every consumer of these goldens
    /// must treat the arm structure as the artifact under review, not the
    /// deployment's truth.
    ///
    /// And the digest DID catch this one too (its fourth catch): the
    /// LIVE L40S default-env derivation is
    /// `warp_tiled: false, cached_max: 0` (both env-gated off), which is
    /// what the emission fact set in `bin/emit-cuda.rs` uses. This
    /// fixture keeps the synthetic values deliberately: the goldens pin
    /// the guard-chain STRUCTURE (a warp arm that exists, a cached arm
    /// with a real threshold), which the live-default set would erase.
    pub fn qwen3_5_0_8b_synthetic() -> Self {
        Self {
            state_bf16: true,
            warp_tiled: true,
            warp_tiled_max: 64,
            cached_max: 4096,
            verify_stash: true,
            // The MoE fields describe the fused leg as the driver has it
            // today: the CUTLASS workspace is always sized
            // (`flashinfer_cutlass_moe_enabled()` returns true
            // unconditionally), 512 is `kFusedMoeMaxRows`, tp=1 folds the
            // residual, and neither the streamed-expert cache nor the
            // force-general env is on by default. Synthetic like the rest
            // of this fixture — the 0.8B checkpoint is dense and reaches
            // none of them; they pin the MoE block's golden form.
            moe_cutlass_max_rows: 512,
            moe_residual_fold: true,
            moe_shared_gate_dot: true,
            moe_streamed_experts: false,
            moe_force_general: false,
            // 0.8B binds the packed bank (qwen3_5.cpp's loader takes the
            // same `dense_fused_projection_joins` contract llama_like's
            // does), so the chunked form is the golden's shape.
            gate_up_fused: true,
        }
    }
}

impl Qwen35HybridFacts {
    /// Whether layer `l` runs full attention —
    /// `driver/metal/src/model/qwen3_5/geometry.hpp::is_full_attn`.
    pub fn is_full_attn(&self, l: u32) -> bool {
        self.full_attn_interval <= 1
            || l % self.full_attn_interval == self.full_attn_interval - 1
    }

    /// The model's hidden size (the sub-facts each carry it for standalone
    /// tracing; [`crate::family::qwen3_5_hybrid`] asserts they agree).
    pub fn hidden(&self) -> u32 {
        self.attn.hidden
    }

    /// Qwen3.5-0.8B, the workspace's hybrid parity checkpoint
    /// (`driver/cuda/tests/parity_qwen3_5_multireq.py` defaults to
    /// `Qwen/Qwen3.5-0.8B-Base`). Sub-facts are the provenance-pinned 0.8B
    /// fixtures ([`Qwen35FullAttnFacts::qwen3_5_0_8b`],
    /// [`Qwen35GdnFacts::qwen3_5_0_8b`]); the model-level dims are pinned
    /// from `driver/metal/src/model/qwen3_5/geometry.hpp` (`DecodeGeometry`
    /// defaults, the Metal driver's 0.8B target): `n_layers = 24`,
    /// `full_attn_interval = 4` (layers 3, 7, 11, 15, 19, 23 full — the
    /// family's 3:1 linear:full pattern), `vocab = 248320`
    /// (`decode_consts.cpp` corroborates: lm_head 1024 → 248320),
    /// `tied_embeddings = true`. The MLP is DENSE with `intermediate =
    /// 3584` (geometry.hpp; `decode_consts.cpp`: gate/up 1024 → 3584,
    /// down 3584 → 1024) — "Dense only on the 0.8B target (MoE deferred)"
    /// (`driver/metal/tests/mlx/model/qwen3_5.hpp`), and the CUDA dense
    /// family (`model_type: qwen3_5`, `qwen3_5_forward.cpp`) runs
    /// `qwen35_dense_mlp_block` on every layer.
    pub fn qwen3_5_0_8b() -> Self {
        Self {
            layers: 24,
            full_attn_interval: 4,
            vocab: 248_320,
            tied_embeddings: true,
            norm_variant: NormVariant::Gemma,
            attn: Qwen35FullAttnFacts::qwen3_5_0_8b(),
            gdn: Qwen35GdnFacts::qwen3_5_0_8b(),
            mlp: Qwen35MlpKind::Dense { intermediate: 3584 },
        }
    }

    /// Qwen3.6-27B — the DENSE hybrid, read from the checkpoint's own
    /// `config.json` (`text_config`), not inferred from the family name.
    ///
    /// Every value here is a field of that file or the driver's stated
    /// derivation from one: 64 layers, `full_attention_interval` 4,
    /// `vocab_size` 248320, `tie_word_embeddings` false,
    /// `intermediate_size` 17408 (no `num_experts` — this checkpoint
    /// takes the `Dense` arm, see [`Qwen35MlpKind`]), hidden 5120,
    /// 24 q heads over 4 kv heads at `head_dim` 256, and
    /// `partial_rotary_factor` 0.25 → `rotary_dim` 64 by the driver's
    /// `max(2, 2 * int(0.5 * f * head_dim))`. The GDN half is the
    /// `linear_*` block: 16 key heads, 48 value heads (a GQA ratio of 3,
    /// which `family.rs`'s gdn body already branches on), 128/128 head
    /// dims, `linear_conv_kernel_dim` 4.
    ///
    /// `fused_in_proj` / `fused_qkv` are false because both joins are
    /// env-gated default-off, the same as 0.8B's.
    ///
    /// NOT reachable on an L40S at bf16 — 27B is ~55 GB against 46. An
    /// FP8 checkpoint of the same geometry is what would boot here; the
    /// traced form is identical either way, which is why the fixture is
    /// worth having before the hardware is.
    pub fn qwen3_6_27b() -> Self {
        Self {
            layers: 64,
            full_attn_interval: 4,
            vocab: 248_320,
            tied_embeddings: false,
            norm_variant: NormVariant::Gemma,
            attn: Qwen35FullAttnFacts {
                hidden: 5120,
                q_heads: 24,
                kv_heads: 4,
                head_dim: 256,
                rotary_dim: 64,
                fused_qkv: false,
                norm_variant: NormVariant::Gemma,
            },
            gdn: Qwen35GdnFacts {
                hidden: 5120,
                key_heads: 16,
                value_heads: 48,
                key_head_dim: 128,
                value_head_dim: 128,
                conv_kernel: 4,
                fused_in_proj: false,
                norm_variant: NormVariant::Gemma,
            },
            mlp: Qwen35MlpKind::Dense {
                intermediate: 17_408,
            },
        }
    }
}

/// Facts for the GEMMA-4 family — the third declared family, and the
/// first whose per-layer axis carries TWO HEAD DIMS.
///
/// # What makes this family its own declaration
///
/// gemma-4 alternates sliding and full attention on a regular interval,
/// like qwen3_5's hybrid — but where qwen3_5's two layer kinds are two
/// different ATTENTIONS, gemma-4's are the same attention at different
/// geometry: `head_dim` 256 on a sliding layer, `global_head_dim` 512 on
/// a full one, with partial rope on the full layers only. The window
/// itself is not trace vocabulary at all: it is a scalar `window_left`
/// the driver reads per layer from its own config, which is why nothing
/// here names it.
///
/// Two things ARE structural and have no analogue in the families
/// declared so far:
///
/// * **KV sharing.** The last [`Self::kv_shared_layers`] layers project
///   no k/v, norm no k/v, rope no k, and write no cache — they attend
///   through the pages of the last earlier layer of the SAME kind. The
///   elision is per layer and total, so it is a fact the trace reads to
///   decide which statements exist, not a runtime branch.
/// * **PLE** (per-layer embeddings). A prologue that embeds a SECOND
///   table, projects it to `layers * ple_dim`, norms and scales it, and
///   transposes so each layer reads a contiguous slice; then a per-layer
///   epilogue gates that slice into the residual stream. It is the
///   reason gemma-4 cannot be a `llama_like` configuration.
///
/// (There is no altup here. That is gemma3n's mechanism —
/// `driver/cuda/src/model/gemma4/` never mentions it.)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Gemma4Facts {
    pub hidden: u32,
    pub layers: u32,
    /// Full attention every `interval`-th layer, `l % interval ==
    /// interval - 1` — qwen3_5's formula, and the config agrees
    /// (E4B: full at 5, 11, …, 41 with interval 6).
    pub full_attn_interval: u32,
    pub q_heads: u32,
    pub kv_heads: u32,
    /// The SLIDING layers' head dim (`head_dim`).
    pub head_dim: u32,
    /// The FULL layers' head dim (`global_head_dim`). Different from
    /// [`Self::head_dim`] on E4B (512 vs 256), which is why the two
    /// kinds cannot share one width the way qwen3_5's do.
    pub global_head_dim: u32,
    /// Partial-rotary width on the FULL layers, resolved the driver's
    /// way (`max(2, 2 * int(0.5 * factor * head_dim))`): 0.25 × 512
    /// gives 128. Sliding layers rotate fully.
    pub global_rotary_dim: u32,
    pub intermediate: u32,
    pub vocab: u32,
    pub tied_embeddings: bool,
    /// `num_kv_shared_layers` — the count of TRAILING layers that reuse
    /// an earlier layer's pages. `first_shared = layers - this`.
    pub kv_shared_layers: u32,
    /// `hidden_size_per_layer_input` — the PLE slice width per layer.
    pub ple_dim: u32,
    /// `use_double_wide_mlp`: the KV-SHARED layers carry an MLP of
    /// `2 * intermediate`. E2B sets it, E4B does not — the first
    /// gemma-4 axis where two deployments of one family disagree about
    /// a WIDTH rather than a count, so it is a fact and the widths it
    /// implies erase at trace time.
    pub double_wide_shared: bool,
    /// `final_logit_softcapping`; 0 means no cap.
    pub logit_softcap: f32,
}

impl Gemma4Facts {
    /// Whether layer `l` runs FULL attention — the same predicate
    /// [`Qwen35HybridFacts::is_full_attn`] states, because the two
    /// families schedule their layer kinds the same way.
    pub fn is_full_attn(&self, l: u32) -> bool {
        self.full_attn_interval <= 1
            || l % self.full_attn_interval == self.full_attn_interval - 1
    }

    /// Whether layer `l` reuses another layer's KV pages, projecting and
    /// writing none of its own.
    pub fn is_kv_shared(&self, l: u32) -> bool {
        l >= self.layers.saturating_sub(self.kv_shared_layers)
    }

    /// This layer's MLP width. The double-wide variant widens exactly
    /// the KV-shared layers, which is why it keys on the same predicate
    /// rather than on a second count.
    pub fn intermediate_of(&self, l: u32) -> u32 {
        if self.double_wide_shared && self.is_kv_shared(l) {
            self.intermediate * 2
        } else {
            self.intermediate
        }
    }

    /// The layer whose pages `l` attends through: the last EARLIER layer
    /// of the same kind (`gemma4.cpp`'s load-time search). `None` for a
    /// layer that owns its pages.
    pub fn kv_source(&self, l: u32) -> Option<u32> {
        if !self.is_kv_shared(l) {
            return None;
        }
        let first_shared = self.layers - self.kv_shared_layers;
        (0..first_shared)
            .rev()
            .find(|&j| self.is_full_attn(j) == self.is_full_attn(l))
    }

    /// This layer's head dim — the per-layer axis that makes gemma-4 its
    /// own family.
    pub fn head_dim_of(&self, l: u32) -> u32 {
        if self.is_full_attn(l) {
            self.global_head_dim
        } else {
            self.head_dim
        }
    }

    /// `google/gemma-4-E4B-it`, read from the checkpoint's own
    /// `config.json` (`text_config`) — every value a field of that file
    /// or the driver's stated derivation from one.
    pub fn gemma_4_e4b() -> Self {
        Self {
            hidden: 2560,
            layers: 42,
            full_attn_interval: 6,
            q_heads: 8,
            kv_heads: 2,
            head_dim: 256,
            global_head_dim: 512,
            // partial_rotary_factor 0.25 on `global_head_dim` 512.
            global_rotary_dim: 128,
            intermediate: 10_240,
            vocab: 262_144,
            tied_embeddings: true,
            kv_shared_layers: 18,
            ple_dim: 256,
            double_wide_shared: false,
            logit_softcap: 30.0,
        }
    }
}

#[cfg(test)]
mod gemma4_tests {
    use super::Gemma4Facts;

    /// The layer-kind schedule the checkpoint states, reproduced by the
    /// formula. `config.json` lists `full_attention` at 5, 11, 17, 23,
    /// 29, 35, 41 — seven of forty-two — and the interval must generate
    /// exactly that set, not merely contain it.
    #[test]
    fn the_schedule_is_the_one_the_config_lists() {
        let f = Gemma4Facts::gemma_4_e4b();
        let full: Vec<u32> = (0..f.layers).filter(|&l| f.is_full_attn(l)).collect();
        assert_eq!(full, vec![5, 11, 17, 23, 29, 35, 41]);
    }

    /// KV sharing: the trailing 18 layers own no pages, and each attends
    /// through the last EARLIER layer of its own kind. On E4B that means
    /// every shared SLIDING layer lands on 22 and every shared FULL one
    /// on 23 — the driver's load-time search, as a fact.
    #[test]
    fn every_shared_layer_finds_a_source_of_its_own_kind() {
        let f = Gemma4Facts::gemma_4_e4b();
        assert_eq!(f.layers - f.kv_shared_layers, 24);
        for l in 0..f.layers {
            match f.kv_source(l) {
                None => assert!(!f.is_kv_shared(l), "layer {l} shares but found no source"),
                Some(src) => {
                    assert!(f.is_kv_shared(l));
                    assert!(src < 24, "layer {l} sources from a sharing layer {src}");
                    assert_eq!(
                        f.is_full_attn(src),
                        f.is_full_attn(l),
                        "layer {l} sources from the other attention kind"
                    );
                }
            }
        }
        assert_eq!(f.kv_source(24), Some(22));
        assert_eq!(f.kv_source(29), Some(23));
        assert_eq!(f.kv_source(41), Some(23));
    }

    /// The two head dims are the per-layer axis. A family that had one
    /// would not need this fact at all, which is why it is worth a test
    /// that says the widths actually differ.
    #[test]
    fn the_two_layer_kinds_have_different_head_dims() {
        let f = Gemma4Facts::gemma_4_e4b();
        assert_ne!(f.head_dim, f.global_head_dim);
        assert_eq!(f.head_dim_of(0), 256);
        assert_eq!(f.head_dim_of(5), 512);
        assert_eq!(f.global_rotary_dim, 2 * (0.5 * 0.25 * 512.0) as u32);
    }
}

/// The CUDA backend's load-time facts for gemma-4 — the BINDING
/// questions its class traces resolve at trace time.
///
/// Three, and all three are "what did the loader materialise", which is
/// the taxonomy's first row: a load-time fact is a trace-time `match`,
/// erased.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Gemma4CudaFacts {
    /// The loader bound one packed `[Hq + 2*Hk, hidden]` projection
    /// (`qkv_proj_fused`) — llama_like's `fused_qkv`, same question.
    pub fused_qkv: bool,
    /// The loader bound a packed gate‖up bank — llama_like's
    /// `gate_up_fused`, same question, different activation behind it.
    pub gate_up_fused: bool,
    /// The KV cache is native bf16, so the fused decode post may write
    /// pages directly. One of the four terms
    /// `can_fuse_packed_qkv_post` reads; the other three are the
    /// declaration's own (`partial` is a layer-kind fact, hooks and the
    /// fire class are class/guard vocabulary).
    pub kv_native_bf16: bool,
}

impl Gemma4CudaFacts {
    /// SYNTHETIC fixture — the same standing caveat every `*CudaFacts`
    /// constructor here carries: it pins the GOLDEN FORM of the traced
    /// arms, not a deployment's truth. The live derivation and its
    /// digest are the executor rung's, and the digest is what corrects a
    /// guess on first boot.
    pub fn gemma_4_e4b_synthetic() -> Self {
        Self {
            fused_qkv: true,
            gate_up_fused: true,
            kv_native_bf16: true,
        }
    }
}

// ── gpt-oss ────────────────────────────────────────────────────────────

/// gpt-oss's shape. The family rides `mixtral.cpp`, so these are the
/// facts that text reads — not a checkpoint dump.
///
/// Two of them are here because the driver ANSWERS them per layer and
/// the declaration would otherwise have to re-derive them per fire: the
/// alternating window kind, and whether a layer carries attention sinks.
/// Both are load-time, so both erase at trace time.
#[derive(Debug, Clone, PartialEq)]
pub struct GptOssFacts {
    pub hidden: u32,
    pub layers: u32,
    pub q_heads: u32,
    pub kv_heads: u32,
    pub head_dim: u32,
    /// One expert's MLP width (`intermediate_size`). gpt-oss's is equal
    /// to `hidden`, which is a coincidence of this checkpoint and not a
    /// rule the text may lean on.
    pub intermediate: u32,
    pub experts: u32,
    pub top_k: u32,
    pub vocab: u32,
    pub tied_embeddings: bool,
    /// `swiglu_limit`; 0 means the plain SwiGLU. gpt-oss clamps at 7.0,
    /// and the clamp is a DIFFERENT KERNEL, so this decides which
    /// activation the text states rather than being a runtime scalar.
    pub swiglu_limit: f32,
    /// Whether the checkpoint biases q/k/v/o, the router, and the expert
    /// projections (`attention_bias`). gpt-oss biases all of them; the
    /// q/k/v biases FOLD INTO the projection's epilogue and the rest are
    /// their own launches.
    pub attention_bias: bool,
    /// Whether this deployment's rope is the YaRN-paper one. gpt-oss's
    /// config asks for it (factor 32 over an original 4096 context) and
    /// the driver resolves it at load, so it is a fact and not a fire's
    /// question — and a WRONG one here is not a crash but a silently
    /// unscaled rotation, which is how it went unnoticed.
    pub rope_yarn_original: bool,
    /// Every layer carries `attn_sinks` on gpt-oss. The driver asks
    /// `layer.attn_sinks != nullptr` per layer and only requests an LSE
    /// from attention where the answer is yes — so this is what decides
    /// whether the attention statement produces one value or two.
    pub attn_sinks: bool,
}

impl GptOssFacts {
    /// Whether layer `l` attends over the SLIDING window. gpt-oss
    /// alternates from layer 0 (`layer_types` reads
    /// sliding, full, sliding, full, …), which the driver reaches
    /// through `per_layer_window_left` — a scalar the text does not
    /// state, since the window is an argument and not a kernel.
    pub fn is_sliding(&self, l: u32) -> bool {
        l % 2 == 0
    }

    /// openai/gpt-oss-20b, read from the checkpoint's `config.json`
    /// (2026-08-06). `layer_types` alternates from sliding; the yarn
    /// `rope_scaling` is NOT in this list because the driver never
    /// applies it — `mixtral.cpp:320` passes the plain `rope_theta`, a
    /// latent bug this declaration must not launder into a fact.
    pub fn gpt_oss_20b() -> Self {
        Self {
            hidden: 2880,
            layers: 24,
            q_heads: 64,
            kv_heads: 8,
            head_dim: 64,
            intermediate: 2880,
            experts: 32,
            top_k: 4,
            vocab: 201088,
            tied_embeddings: false,
            swiglu_limit: 7.0,
            attention_bias: true,
            rope_yarn_original: true,
            attn_sinks: true,
        }
    }
}

/// The CUDA backend's answers for a gpt-oss deployment — the bindings
/// and the admission thresholds, all resolved at load.
#[derive(Debug, Clone, PartialEq)]
pub struct GptOssCudaFacts {
    /// Whether the layer bank carries the per-expert POINTER ARRAYS the
    /// fused decode GEMV indexes (`expert_gate_up_packed_ptrs`). Built
    /// by the `RoutedDecode` MXFP4 policy, which is the engine default;
    /// the `NativeGemm` policy binds marlin views instead and reaches
    /// the experts through a per-expert loop no rectangle spells.
    pub mxfp4_decode_gemv: bool,
    /// `mxfp4_decode_max_routes` — the fused leg's admission threshold
    /// in ROUTES (`N * top_k`), default `32 * experts`. A fire past it
    /// takes the host-routed walk, which this declaration refuses by
    /// name rather than states.
    pub mxfp4_decode_max_routes: u32,
    /// Whether the experts are STREAMED through a slab cache. A streamed
    /// layer reaches the same fused kernels, but only after a host
    /// round-trip that decides what to page in — so a streamed
    /// deployment is outside the flat list until that is stated.
    pub streamed_experts: bool,
}

impl GptOssCudaFacts {
    /// The L40S deployment's set, as the driver derives it: no
    /// streaming, the default policy's pointer arrays, and the default
    /// cap. SYNTHETIC until a live digest judges it — the standing
    /// contract for every `*_synthetic` fixture in this file.
    pub fn gpt_oss_20b_synthetic() -> Self {
        Self {
            mxfp4_decode_gemv: true,
            mxfp4_decode_max_routes: 32 * 32,
            streamed_experts: false,
        }
    }
}
