//! `llama_like`'s SHAPE: the numbers a checkpoint of this family has.
//!
//! Ungated, because a row is written in these words and a row is the
//! crate's identity under every aspect -- `chat` asks which template
//! speaks for it, `contract` asks how to author it, `forward` asks what
//! to trace. One struct, three readers; that is the whole point of the
//! catalog, and it cannot hold if the struct only exists when the tracer
//! is compiled in.
//!
//! What stayed behind in `forward/facts.rs` is the per-backend BINDING
//! facts, which name kernels and therefore belong to the aspect that
//! has them.

use serde::{Deserialize, Serialize};
// The shared vocabulary stayed with the toolchain -- more than one family
// is written in these words. Re-exported so a declaration reaches its
// facts and the words they are stated in from one place.
pub use model_ir::facts::{NormPlacement, QkNorm};
use model_ir::trace::{NormVariant, RopeKind};

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
    /// Experts in this deployment's mixture, and 0 for a dense one.
    ///
    /// A llama-like architecture with a ROUTED FFN is still llama-like: the
    /// attention is unchanged and only the block between the two norms
    /// differs. Stating it as a fact rather than a family is the whole tart
    /// argument -- one supergraph, more polymorphism -- and it is what lets
    /// qwen3-moe reach the device without a second text.
    #[serde(default)]
    pub n_experts: u32,
    /// How many of them a row goes to.
    #[serde(default)]
    pub experts_per_token: u32,
    /// One expert's inner width.
    #[serde(default)]
    pub moe_intermediate: u32,
    /// The dense expert's inner width, 0 for a mixture without one.
    #[serde(default)]
    pub shared_intermediate: u32,
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
    /// The ATTENTION LANDING's bias: the checkpoint ships `o_proj.bias` and
    /// the forward adds it after the landing, residual and all.
    ///
    /// gpt-oss's. Separate from [`Self::qkv_bias`] because the two are
    /// separate publications -- Qwen-2 ships the first three and no fourth,
    /// and a single flag would make a text state a tensor its checkpoint
    /// does not have. `GptOssCudaFacts::attention_bias` is the same claim on
    /// the other backend.
    ///
    /// After the residual and not before, which is not a rounding detail
    /// worth being casual about: the CUDA text folds the residual into the
    /// landing (`beta=1`) and then adds the bias, so a Metal text that added
    /// the bias first would accumulate in a different order than the
    /// reference this is checked against.
    #[serde(default)]
    pub o_bias: bool,
    /// The ROUTER's bias: one number per expert, added to the logits before
    /// the top-k.
    ///
    /// The most consequential bias a mixture publishes and the least
    /// forgiving. The others shift an activation that a norm largely
    /// absorbs; this one shifts a RANKING. A text that drops it does not
    /// compute a slightly different answer -- it sends each token to
    /// different experts, and every number after that is unrelated.
    #[serde(default)]
    pub router_bias: bool,
}

impl LlamaLikeFacts {
    /// This family's projection into the DSL's family-neutral
    /// [`ModelShape`](model_dsl::ModelShape) — the dense-transformer weight
    /// namespace, and nothing about llama in particular.
    ///
    /// The toolchain cannot name `LlamaLikeFacts` -- that edge would point the
    /// wrong way -- so the projection is written here, on the family side,
    /// once per family.
    pub fn shape(&self) -> model_dsl::ModelShape {
        model_dsl::ModelShape {
            hidden: self.hidden,
            intermediate: self.intermediate,
            n_experts: self.n_experts,
            moe_intermediate: self.moe_intermediate,
            shared_intermediate: self.shared_intermediate,
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
            proj_repr: model_dsl::WeightRepr::Bf16,
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
            // DENSE: no mixture. Stated rather than defaulted because a
            // fixture is a measurement of a real checkpoint, and "this one has
            // no experts" is part of the measurement.
            n_experts: 0,
            experts_per_token: 0,
            moe_intermediate: 0,
            shared_intermediate: 0,
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
            o_bias: false,
            router_bias: false,
        }
    }

    /// Llama-3.2-1B-Instruct, the checkpoint every Metal device gate
    /// measures against MLX — and, until this fixture, the only one of them
    /// with no llama_like fact of its own.
    ///
    /// It is here for a shape no other fixture has: `head_dim` **64**
    /// without sinks. gpt-oss is the only other 64-wide entry and it has
    /// them, so a text built from these fixtures could reach
    /// `sdpa_paged_mma_sink` and never `sdpa_paged_mma` — a kernel the
    /// driver dispatches on this very checkpoint, twelve times per device
    /// run, that no conformance text named.
    ///
    /// GQA (32 q / 8 kv heads), head_dim 64 STATED by the config rather
    /// than derived (2048 / 32 agrees, but llama-3 states it and a family
    /// that states it is not one to derive it for). Dense, plain RMS norms
    /// pre-attention, no qk-norm, no biases anywhere, tied embeddings, and
    /// the three projections ship separately in the MLX layout.
    ///
    /// `rope: Standard` where the config says `rope_type: "llama3"`, and it
    /// is not a mistranscription. [`RopeKind`] names the SHAPE of the
    /// rotation, and llama-3's scaling is a remap of the inverse-frequency
    /// table -- factor 32 above 8192 positions, interpolated between the low
    /// and high frequency cuts -- computed once into the table the kernel
    /// reads. There is no second kernel and no third value to state, the
    /// same way phi-3's LongRoPE has none.
    pub fn llama_3_2_1b() -> Self {
        Self {
            // DENSE: no mixture. Stated rather than defaulted because a
            // fixture is a measurement of a real checkpoint, and "this one has
            // no experts" is part of the measurement.
            n_experts: 0,
            experts_per_token: 0,
            moe_intermediate: 0,
            shared_intermediate: 0,
            hidden: 2048,
            layers: 16,
            q_heads: 32,
            kv_heads: 8,
            head_dim: 64,
            intermediate: 8192,
            vocab: 128_256,
            rope: RopeKind::Standard,
            norm_variant: NormVariant::Plain,
            norm_placement: NormPlacement::Pre,
            qk_norm: QkNorm::Off,
            fused_qkv: false,
            tied_embeddings: true,
            qkv_bias: false,
            o_bias: false,
            router_bias: false,
        }
    }

    /// Qwen3-0.6B, the workspace's parity model.
    pub fn qwen3_0_6b() -> Self {
        Self {
            // DENSE: no mixture. Stated rather than defaulted because a
            // fixture is a measurement of a real checkpoint, and "this one has
            // no experts" is part of the measurement.
            n_experts: 0,
            experts_per_token: 0,
            moe_intermediate: 0,
            shared_intermediate: 0,
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
            o_bias: false,
            router_bias: false,
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
    /// Qwen3-30B-A3B: llama-like attention, ROUTED FFN.
    ///
    /// The fixture that makes the mixture reachable. Every attention number
    /// here is a qwen3 number and every one of them is shared with
    /// [`Self::qwen3_0_6b`]'s shape -- which is the point: the only thing this
    /// deployment does differently is the block between the two norms.
    ///
    /// A SYNTHETIC fixture in the same sense as the rest: the numbers are the
    /// published config's, not a measurement of a running checkpoint.
    pub fn qwen3_30b_a3b() -> Self {
        Self {
            hidden: 2048,
            layers: 48,
            q_heads: 32,
            kv_heads: 4,
            head_dim: 128,
            // The DENSE inner width, which a mixture has no use for. Stated as
            // zero rather than left at the dense value so a text that reached
            // for the wrong one computes nothing visible instead of something
            // plausible.
            intermediate: 0,
            n_experts: 128,
            experts_per_token: 8,
            moe_intermediate: 768,
            // No shared expert: qwen3-moe dropped the one qwen2-moe had.
            shared_intermediate: 0,
            vocab: 151_936,
            rope: RopeKind::Standard,
            norm_variant: NormVariant::Plain,
            norm_placement: NormPlacement::Pre,
            qk_norm: QkNorm::PerHead,
            // An ATTENTION number, so the doc above binds it: qwen3-0.6b
            // states `true` and this said `false`.
            fused_qkv: true,
            tied_embeddings: false,
            qkv_bias: false,
            o_bias: false,
            router_bias: false,
        }
    }

    /// gpt-oss-20b: llama-like attention with SINKS, a routed FFN, and its
    /// own SwiGLU.
    ///
    /// The numbers are the published config's. What makes it interesting is
    /// how little of it is new: the attention is a qwen3 attention with one
    /// extra weight per layer, the mixture is the one qwen3-moe already
    /// proved, and the activation is a symbol. `sliding_window: 128` alternates
    /// with full attention every other layer, which `window_left` states.
    pub fn gpt_oss_20b() -> Self {
        Self {
            hidden: 2880,
            layers: 24,
            q_heads: 64,
            kv_heads: 8,
            head_dim: 64,
            // The DENSE inner width, which a mixture has no use for.
            intermediate: 0,
            n_experts: 32,
            experts_per_token: 4,
            moe_intermediate: 2880,
            // No shared expert.
            shared_intermediate: 0,
            vocab: 201_088,
            // gpt-oss is the only YaRN row in this table, and this said
            // `Standard` until `the_llama_like_fixture_measures_the_same_checkpoint`
            // was made to compare the whole struct.
            rope: RopeKind::Yarn,
            norm_variant: NormVariant::Plain,
            norm_placement: NormPlacement::Pre,
            // No QK norm; gpt-oss normalizes neither.
            qk_norm: QkNorm::Off,
            fused_qkv: false,
            tied_embeddings: false,
            // `attention_bias: true` — every projection carries one, and
            // "every" includes the LANDING and the router. The checkpoint
            // ships `self_attn.o_proj.bias [2880]` and `mlp.router.bias
            // [32]` beside q/k/v, `gpt_oss::project::metal_shape` states
            // all three, and these two read `false` here for as long as
            // the guard compared fields one at a time.
            qkv_bias: true,
            o_bias: true,
            router_bias: true,
        }
    }

    pub fn phi3_mini() -> Self {
        Self {
            // DENSE: no mixture. Stated rather than defaulted because a
            // fixture is a measurement of a real checkpoint, and "this one has
            // no experts" is part of the measurement.
            n_experts: 0,
            experts_per_token: 0,
            moe_intermediate: 0,
            shared_intermediate: 0,
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
            o_bias: false,
            router_bias: false,
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
            // DENSE: no mixture. Stated rather than defaulted because a
            // fixture is a measurement of a real checkpoint, and "this one has
            // no experts" is part of the measurement.
            n_experts: 0,
            experts_per_token: 0,
            moe_intermediate: 0,
            shared_intermediate: 0,
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
            o_bias: false,
            router_bias: false,
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
            // DENSE: no mixture. Stated rather than defaulted because a
            // fixture is a measurement of a real checkpoint, and "this one has
            // no experts" is part of the measurement.
            n_experts: 0,
            experts_per_token: 0,
            moe_intermediate: 0,
            shared_intermediate: 0,
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
            o_bias: false,
            router_bias: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every fixture, so a mistyped constant is a failure here rather
    /// than a model that loads and answers wrongly.
    fn all() -> Vec<(&'static str, LlamaLikeFacts)> {
        vec![
            ("qwen2.5-1.5b", LlamaLikeFacts::qwen2_5_1_5b()),
            ("qwen3-0.6b", LlamaLikeFacts::qwen3_0_6b()),
            ("qwen3-30b-a3b", LlamaLikeFacts::qwen3_30b_a3b()),
            ("gpt-oss-20b", LlamaLikeFacts::gpt_oss_20b()),
            // The CATALOG ids, so `every_fixture_is_the_row_it_names` can
            // resolve them. These read "phi3-mini" and "olmo2-1b" until it
            // tried, and a label that names no row is a label nothing can
            // check.
            ("phi-3-mini-4k", LlamaLikeFacts::phi3_mini()),
            ("mistral-7b-v0.3", LlamaLikeFacts::mistral_7b_v03()),
            ("olmo-2-1b", LlamaLikeFacts::olmo2_1b()),
        ]
    }

    /// The same question as `every_metal_predicate_is_stated_more_than_
    /// one_way_or_excused`, asked of the SHAPE half.
    ///
    /// `LlamaLikeMetalFacts` describes what a deployment binds and this
    /// describes what a checkpoint IS; `llama_like_metal` branches on
    /// both. A predicate every fixture states identically is a branch
    /// that compiles and is never emitted -- which is what `o_bias` and
    /// `router_bias` were until the gpt-oss row's total comparison found
    /// them false against a projection that says true.
    ///
    /// Numbers and not only booleans, because the two worst defects this
    /// pair of structs has had were numeric: `embed_scale: 0.0` and
    /// `attn_scale: 0.0`, both of them SENTINELS meaning "derive it",
    /// both left in a fixture named for a family that derives nothing of
    /// the kind. A number identical in every fixture is the same dark
    /// branch as a boolean nobody flips, and asking for two distinct
    /// values rather than for a zero avoids inventing a rule about which
    /// numbers may be zero -- widths vary on their own.
    ///
    /// Seven fixtures, and the whole struct produces ONE excuse.
    #[test]
    fn every_shape_predicate_is_stated_more_than_one_way_or_excused() {
        use std::collections::{BTreeMap, BTreeSet};

        // `(field, why)` -- a fact about the crate, not an intention.
        const EXCUSED: &[(&str, &str)] = &[
            // Zero in all seven, and correctly: no row in this catalog
            // that Metal serves has a shared expert. The rows that do --
            // glm-5, kimi-k2, kimi-k3, qwen3.5-moe -- state their own
            // texts and none of them is `llama_like_metal`. The branch is
            // exercised by `forward`'s own test, which sets it to 512
            // inline and checks the dense leg is blended in.
            //
            // The DRIVER side had the same hole, seen from the kernel:
            // `shared_expert_combine` is compiled into every Metal build
            // and no text named it, so `driver-metal`'s slot conformance
            // had never inspected it. Closed there by a "qwen3-moe, shared
            // expert" text at this same 512 -- which is `Qwen3.6-35B-A3B`'s
            // measured `shared_expert_intermediate_size`, a real number
            // rather than a plausible one.
            (
                "shared_intermediate",
                "no Metal-served row has a shared expert",
            ),
        ];

        let scalars = |f: &LlamaLikeFacts| match serde_json::to_value(f) {
            Ok(serde_json::Value::Object(o)) => o
                .into_iter()
                .filter(|(_, v)| v.is_boolean() || v.is_number())
                .map(|(k, v)| (k, v.to_string()))
                .collect::<BTreeMap<_, _>>(),
            other => panic!("these facts serialise as a struct, not {other:?}"),
        };
        let every: Vec<BTreeMap<String, String>> = all().iter().map(|(_, f)| scalars(f)).collect();
        assert!(every.len() >= 7, "the fixture table shrank");
        // Not vacuous: a serialisation that stopped emitting scalars would
        // make every assertion below trivially true.
        assert!(every[0].len() >= 15, "this struct is mostly scalars");

        let dark: BTreeSet<&str> = every[0]
            .keys()
            .filter(|name| every.iter().all(|f| f[*name] == every[0][*name]))
            .map(String::as_str)
            .collect();
        let excused: BTreeSet<&str> = EXCUSED.iter().map(|(f, _)| *f).collect();

        let opened: Vec<&&str> = dark.difference(&excused).collect();
        assert!(
            opened.is_empty(),
            "shape predicate(s) every fixture states identically, and none \
             is excused. A branch only one value reaches compiles and is \
             never emitted: {opened:?}"
        );
        let closed: Vec<&&str> = excused.difference(&dark).collect();
        assert!(
            closed.is_empty(),
            "excuse(s) that stopped being needed: {closed:?}"
        );
    }

    /// Every fixture IS a catalog row's own shape, field for field.
    ///
    /// These fixtures exist so that a text can be traced without a
    /// deployment, and every one of them is named for a row that this
    /// build actually serves — so the two are one document read twice,
    /// and the only honest relation between them is equality.
    ///
    /// Written as a total comparison and not a list of fields, because
    /// the two drifts found the day this was added were both in fields
    /// no list contained: `gemma_like`'s `embed_scale` fell through its
    /// base fixture at zero, which is the branch that picks
    /// `embed_gather` over `embed_gather_scaled`, and the gpt-oss
    /// fixture said `Standard` rope and no landing or router bias
    /// against a row that states YaRN and both. Each was guarded by an
    /// enumeration whose doc named exactly that defect.
    #[test]
    fn every_fixture_is_the_row_it_names() {
        let rows: Vec<(&str, LlamaLikeFacts)> = crate::qwen_2::VARIANTS
            .iter()
            .map(|v| (v.id, v.shape.clone()))
            .chain(
                crate::qwen_3::VARIANTS
                    .iter()
                    .map(|v| (v.id, v.shape.clone())),
            )
            .chain(
                crate::phi_3::VARIANTS
                    .iter()
                    .map(|v| (v.id, v.shape.clone())),
            )
            .chain(
                crate::mistral_3::VARIANTS
                    .iter()
                    .map(|v| (v.id, v.shape.clone())),
            )
            .chain(
                crate::olmo_2::VARIANTS
                    .iter()
                    .map(|v| (v.id, v.shape.clone())),
            )
            .collect();
        for (name, f) in all() {
            // gpt-oss is not a llama-like row: it has its own facts type
            // and its own text, and `metal_shape` is the projection that
            // makes it one. `gpt_oss::tests::
            // the_llama_like_fixture_measures_the_same_checkpoint` holds
            // that comparison, against the projection AND against the row.
            if name == "gpt-oss-20b" {
                continue;
            }
            let (_, row) = rows
                .iter()
                .find(|(id, _)| *id == name)
                .unwrap_or_else(|| panic!("{name}: no catalog row carries that id"));
            assert_eq!(&f, row, "{name}: the fixture and the row have drifted");
        }
    }

    #[test]
    fn every_fixture_states_a_stack_that_could_exist() {
        for (name, f) in all() {
            assert!(f.hidden > 0, "{name}: no residual width");
            assert!(f.layers > 0, "{name}: no layers");
            assert!(f.q_heads > 0, "{name}: no query heads");
            assert!(f.kv_heads > 0, "{name}: no kv heads");
            assert!(f.head_dim > 0, "{name}: no head dim");
            assert!(f.vocab > 0, "{name}: no vocabulary");
            assert_eq!(
                f.q_heads % f.kv_heads,
                0,
                "{name}: {} query heads do not group into {} kv heads",
                f.q_heads,
                f.kv_heads
            );
            assert!(
                f.kv_heads <= f.q_heads,
                "{name}: more kv heads than query heads is not GQA, it is nothing"
            );
        }
    }

    /// The two widths the load plan splits a fused `qkv` on. Stated as
    /// a product here and re-derived nowhere else.
    #[test]
    fn the_projection_widths_are_the_heads_times_the_head_dim() {
        for (name, f) in all() {
            assert_eq!(f.q_width(), f.q_heads * f.head_dim, "{name}");
            assert_eq!(f.kv_width(), f.kv_heads * f.head_dim, "{name}");
            assert!(f.kv_width() <= f.q_width(), "{name}");
        }
    }

    /// A mixture states an expert count, an expert width and a
    /// per-token top-k; a dense stack states none of the three. The
    /// half-stated case is the one that loads and then indexes past the
    /// end of a router.
    #[test]
    fn a_mixture_states_all_three_of_its_numbers_or_none() {
        for (name, f) in all() {
            let mixture = f.n_experts > 0;
            assert_eq!(
                mixture,
                f.experts_per_token > 0,
                "{name}: experts and top-k disagree"
            );
            assert_eq!(
                mixture,
                f.moe_intermediate > 0,
                "{name}: experts and expert width disagree"
            );
            if mixture {
                assert!(
                    f.experts_per_token <= f.n_experts,
                    "{name}: routes to more experts than it has"
                );
            }
        }
    }

    /// A dense stack has a dense MLP. A mixture may have none — its
    /// width lives on the experts.
    #[test]
    fn a_dense_stack_states_a_dense_width() {
        for (name, f) in all() {
            if f.n_experts == 0 {
                assert!(f.intermediate > 0, "{name}: a dense stack with no MLP");
            }
        }
    }

    /// The projection into the toolchain's namespace carries the row's
    /// numbers unchanged, and states `Bf16` regardless of how the
    /// checkpoint stores them — the representation is the BACKEND's
    /// answer, not the semantic shape's.
    #[test]
    fn the_dsl_shape_is_the_row_restated_in_the_toolchains_words() {
        for (name, f) in all() {
            let s = f.shape();
            assert_eq!(s.hidden, f.hidden, "{name}");
            assert_eq!(s.intermediate, f.intermediate, "{name}");
            assert_eq!(s.n_experts, f.n_experts, "{name}");
            assert_eq!(s.moe_intermediate, f.moe_intermediate, "{name}");
            assert_eq!(s.shared_intermediate, f.shared_intermediate, "{name}");
            assert_eq!(s.vocab, f.vocab, "{name}");
            assert_eq!(s.head_dim, f.head_dim, "{name}");
            assert_eq!(s.q_width, f.q_width(), "{name}");
            assert_eq!(s.kv_width, f.kv_width(), "{name}");
            assert_eq!(s.qk_norm, f.qk_norm, "{name}");
            assert_eq!(s.norm_variant, f.norm_variant, "{name}");
            assert_eq!(s.tied_embeddings, f.tied_embeddings, "{name}");
            assert_eq!(
                s.proj_repr,
                model_dsl::WeightRepr::Bf16,
                "{name}: the semantic shape cannot name a backend's encoding"
            );
        }
    }

    /// The fixtures are MEASUREMENTS, so the ones that differ have to
    /// keep differing — a copy-paste that collapsed two of them into
    /// the same stack would pass every test above.
    #[test]
    fn the_fixtures_are_distinct_measurements() {
        let rows = all();
        for (i, (a_name, a)) in rows.iter().enumerate() {
            for (b_name, b) in rows.iter().skip(i + 1) {
                assert_ne!(
                    (
                        a.hidden, a.layers, a.q_heads, a.kv_heads, a.head_dim, a.vocab
                    ),
                    (
                        b.hidden, b.layers, b.q_heads, b.kv_heads, b.head_dim, b.vocab
                    ),
                    "{a_name} and {b_name} state the same stack"
                );
            }
        }
    }

    /// The four the redesign named as things a row must STATE rather
    /// than a derivation must sniff: pre/post norm, the qk-norm
    /// granularity, whether the qkv arrives fused, and whether the
    /// embeddings are tied. Asserted on the fixtures that carry the
    /// unusual answer, because a default would have swallowed them.
    #[test]
    fn the_sniffed_facts_are_stated() {
        let olmo = LlamaLikeFacts::olmo2_1b();
        assert_eq!(
            olmo.norm_placement,
            NormPlacement::Post,
            "olmo2 is the post-norm stack; this was a `starts_with(\"olmo\")`"
        );
        assert_eq!(
            olmo.qk_norm,
            QkNorm::Global,
            "a whole-projection norm, not one per head"
        );
        assert!(!olmo.tied_embeddings, "olmo2 ships a separate head");

        let qwen3 = LlamaLikeFacts::qwen3_0_6b();
        assert_eq!(qwen3.norm_placement, NormPlacement::Pre);
        assert_eq!(
            qwen3.qk_norm,
            QkNorm::PerHead,
            "the derivation asked `elems_of(q_norm) == head_dim`"
        );
        assert!(
            qwen3.tied_embeddings,
            "a tie is an ABSENCE, so it must be stated"
        );

        let qwen2 = LlamaLikeFacts::qwen2_5_1_5b();
        assert!(
            qwen2.qkv_bias,
            "qwen2.5 is the generation that ships qkv biases"
        );
        assert_eq!(qwen2.qk_norm, QkNorm::Off);
    }
}
