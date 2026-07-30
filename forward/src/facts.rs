//! Load-time facts a declaration traces against.
//!
//! These are the `config.json`-derived values that the hand-written
//! `LlamaLikeForwardCfg` + `HfConfig` pair carries into the forward today,
//! reduced to what the *declaration* needs: everything here is resolved at
//! trace time and none of it survives into the traced form except as
//! constants and op choices.

use serde::{Deserialize, Serialize};

use crate::trace::{NormVariant, RopeKind};

/// The llama_like family's facts: covers qwen3, mistral3, phi3, olmo3
/// (pie-application-plan.md §7 stage 3 scope). Declared so far: the qwen3
/// configuration — pre-norm, per-head qk-norm, standard rope, fused QKV
/// binding, dense MLP — the phi3 one, which drops the qk-norm and the
/// embedding tie, and the mistral (7B v0.3) one, which pairs the fused
/// binding with no qk-norm.
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
    /// Per-head RMSNorm on Q/K before rope (qwen3, olmo2-small).
    pub qk_norm: bool,
    /// The deployment bound one packed `[q + 2kv, hidden]` projection.
    /// This is a *binding* fact, not an architecture fact: the declaration
    /// writes one matmul either way, and with `false` it traces three.
    pub fused_qkv: bool,
    /// The lm_head weight is the embedding table (weight tying).
    pub tied_embeddings: bool,
}

impl LlamaLikeFacts {
    pub fn q_width(&self) -> u32 {
        self.q_heads * self.head_dim
    }

    pub fn kv_width(&self) -> u32 {
        self.kv_heads * self.head_dim
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
            qk_norm: true,
            fused_qkv: true,
            tied_embeddings: true,
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
            qk_norm: false,
            fused_qkv: false,
            tied_embeddings: false,
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
            qk_norm: false,
            fused_qkv: true,
            tied_embeddings: false,
        }
    }
}
