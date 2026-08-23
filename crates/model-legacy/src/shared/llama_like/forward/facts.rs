use model_dsl::WeightRepr;
pub use model_ir::facts::{NormPlacement, QkNorm};
use serde::{Deserialize, Serialize};

pub use super::super::spec::LlamaLikeFacts;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LlamaLikeCudaFacts {
    pub xqa_decode: bool,

    pub decode_fused_post: bool,

    pub rope_table: bool,

    pub force_prefill_path: bool,

    #[serde(default)]
    pub head_dim_padded: bool,

    #[serde(default)]
    pub head_dim_kernel: u32,

    #[serde(default)]
    pub gate_up_fused: bool,

    #[serde(default)]
    pub proj_repr: WeightRepr,

    #[serde(default)]
    pub tp_size: u32,

    #[serde(default)]
    pub window_left: Vec<i32>,

    #[serde(default)]
    pub all_reduce_p2p_max_rows: u32,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub enum Activation {
    #[default]
    SiluMul,

    SwiGlu {
        limit: f32,

        alpha: f32,
    },

    Geglu,
}

fn default_moe_tile() -> Option<(u32, u32)> {
    Some(crate::shared::llama_like::project::ROUTED_QMM_TILE)
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LlamaLikeMetalFacts {
    pub fuse_residual_gemv: bool,

    pub paged_multi_batch: bool,

    pub qmm_multi_batch: bool,

    #[serde(default)]
    pub proj_repr: model_dsl::WeightRepr,

    #[serde(default)]
    pub affine_bits: u32,

    #[serde(default)]
    pub moe_repr: Option<model_dsl::WeightRepr>,

    #[serde(default)]
    pub moe_bits: u32,

    #[serde(default)]
    pub router_repr: Option<model_dsl::WeightRepr>,

    #[serde(default)]
    pub router_bits: u32,

    #[serde(default)]
    pub qmm_tile: (u32, u32),

    pub qmm_partial_rows: bool,

    #[serde(default = "default_moe_tile")]
    pub moe_tile: Option<(u32, u32)>,

    #[serde(default)]
    pub qmm_fp16_precast: bool,

    #[serde(default)]
    pub routed_qmm_fp16: bool,

    #[serde(default)]
    pub gate_up_fused: bool,

    #[serde(default)]
    pub rms_eps: f32,

    #[serde(default)]
    pub add_bias: bool,

    #[serde(default)]
    pub fused_qk_rope: bool,

    #[serde(default)]
    pub rope_theta: f32,

    #[serde(default)]
    pub rope_theta_sliding: f32,

    #[serde(default)]
    pub global_head_dim: u32,

    #[serde(default)]
    pub global_kv_heads: u32,

    #[serde(default)]
    pub full_partial_rotary: f32,

    #[serde(default)]
    pub v_from_k: bool,

    #[serde(default)]
    pub dense_beside_moe: bool,

    #[serde(default)]
    pub router_input_norm: bool,

    #[serde(default)]
    pub router_expert_scale: bool,

    #[serde(default)]
    pub norm_topk_prob: bool,

    #[serde(default)]
    pub per_layer_scalar: bool,

    #[serde(default)]
    pub embed_scale: f32,

    #[serde(default)]
    pub attn_scale: f32,

    #[serde(default)]
    pub v_norm: bool,

    #[serde(default)]
    pub per_layer_emb_dim: u32,

    #[serde(default)]
    pub kv_shared_layers: u32,

    #[serde(default)]
    pub logit_softcap: f32,

    #[serde(default)]
    pub attn_sinks: bool,

    #[serde(default)]
    pub activation: Activation,

    #[serde(default)]
    pub rope_freq_table: bool,

    #[serde(default)]
    pub rope_proportional: bool,

    #[serde(default)]
    pub window_left: Vec<i32>,
}

impl LlamaLikeMetalFacts {
    #[must_use]
    pub fn gpt_oss_20b() -> Self {
        Self {
            attn_scale: 0.226_657_55,
            attn_sinks: true,

            activation: Activation::SwiGlu {
                limit: 7.0,
                alpha: 1.702,
            },

            rope_theta: 150_000.0,
            rope_freq_table: true,
            rms_eps: 1e-5,

            window_left: (0..24).map(|l| if l % 2 == 0 { 128 } else { -1 }).collect(),

            moe_repr: Some(model_dsl::WeightRepr::Mxfp4Marlin),
            moe_bits: 4,

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

    #[must_use]
    pub fn gemma_like() -> Self {
        Self {
            activation: Activation::Geglu,

            rope_proportional: true,
            logit_softcap: 30.0,
            per_layer_emb_dim: 256,
            kv_shared_layers: 4,
            dense_beside_moe: true,
            router_input_norm: true,
            router_expert_scale: true,

            v_norm: true,

            attn_scale: 1.0,

            embed_scale: 32.0,

            window_left: (0..28).map(|l| if l % 6 == 5 { -1 } else { 512 }).collect(),

            global_head_dim: 256,
            global_kv_heads: 4,

            full_partial_rotary: 0.25,
            rope_theta: 1_000_000.0,

            rope_theta_sliding: 10_000.0,

            routed_qmm_fp16: true,
            ..Self::synthetic()
        }
    }

    pub fn window_left_at(&self, l: u32) -> i32 {
        model_ir::facts::window_left_at(&self.window_left, l)
    }

    pub fn rope_theta_at(&self, l: u32) -> f32 {
        if self.rope_theta_sliding > 0.0 && self.window_left_at(l) >= 0 {
            self.rope_theta_sliding
        } else {
            self.rope_theta
        }
    }

    pub fn is_full_attention(&self, l: u32) -> bool {
        self.window_left_at(l) < 0
    }

    pub fn head_dim_at(&self, l: u32, sliding: u32) -> u32 {
        if self.global_head_dim > 0 && self.is_full_attention(l) {
            self.global_head_dim
        } else {
            sliding
        }
    }

    pub fn kv_heads_at(&self, l: u32, sliding: u32) -> u32 {
        if self.global_kv_heads > 0 && self.is_full_attention(l) {
            self.global_kv_heads
        } else {
            sliding
        }
    }

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

    pub fn synthetic() -> Self {
        Self {
            qmm_partial_rows: false,

            qmm_fp16_precast: true,
            fuse_residual_gemv: true,
            paged_multi_batch: true,
            qmm_multi_batch: true,

            add_bias: true,

            fused_qk_rope: false,

            norm_topk_prob: true,

            proj_repr: model_dsl::WeightRepr::Scaled {
                layout: model_dsl::ScaleLayout::PerGroup,
                group: 64,
                axis: 0,
                zero_point: true,
            },

            routed_qmm_fp16: false,
            affine_bits: 4,

            moe_repr: None,
            moe_bits: 0,

            router_repr: None,
            router_bits: 0,

            qmm_tile: (32, 32),
            moe_tile: default_moe_tile(),

            gate_up_fused: false,

            rms_eps: 1e-6,

            rope_theta: 1_000_000.0,

            rope_theta_sliding: 0.0,

            global_head_dim: 0,
            global_kv_heads: 0,
            full_partial_rotary: 0.0,

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

            logit_softcap: 0.0,
            attn_sinks: false,
            activation: Activation::SiluMul,

            rope_freq_table: false,

            rope_proportional: false,

            window_left: Vec::new(),
        }
    }
}

impl LlamaLikeCudaFacts {
    pub fn window_left_at(&self, l: u32) -> i32 {
        model_ir::facts::window_left_at(&self.window_left, l)
    }

    pub fn qwen3_0_6b_l40s() -> Self {
        Self {
            xqa_decode: false,
            decode_fused_post: true,
            rope_table: true,
            force_prefill_path: false,
            head_dim_padded: false,
            head_dim_kernel: 0,

            gate_up_fused: true,

            proj_repr: WeightRepr::Bf16,

            tp_size: 1,

            window_left: Vec::new(),

            all_reduce_p2p_max_rows: 0,
        }
    }
}
