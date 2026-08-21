use crate::catalog::Variant;
use crate::llama_3::Llama3;
use crate::shared::llama_like::spec::LlamaLikeFacts;
use model_ir::facts::{NormPlacement, QkNorm};
use model_ir::trace::{NormVariant, RopeKind};

pub const TINY_LLAMA: &str = "test-tiny-llama";

pub const VARIANTS: &[Llama3] = &[Llama3 {
    id: TINY_LLAMA,
    shape: LlamaLikeFacts {
        hidden: 64,
        layers: 2,
        q_heads: 4,
        kv_heads: 2,
        head_dim: 16,
        n_experts: 0,
        experts_per_token: 0,
        moe_intermediate: 0,
        shared_intermediate: 0,
        intermediate: 96,
        vocab: 128,
        rope: RopeKind::Standard,
        norm_variant: NormVariant::Plain,
        norm_placement: NormPlacement::Pre,
        qk_norm: QkNorm::Off,
        fused_qkv: true,
        tied_embeddings: false,
        qkv_bias: false,
        o_bias: false,
        router_bias: false,
    },
    rope_theta: 10_000.0,
    norm_eps: 1e-6,
    window: -1,

    rope_factor: 1.0,
}];

#[must_use]
pub fn rows() -> &'static [&'static dyn Variant] {
    static ROWS: std::sync::OnceLock<Vec<&'static dyn Variant>> = std::sync::OnceLock::new();
    ROWS.get_or_init(|| VARIANTS.iter().map(|v| v as &'static dyn Variant).collect())
}
