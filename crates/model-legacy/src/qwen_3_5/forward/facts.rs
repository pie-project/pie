use model_dsl::WeightRepr;
use serde::{Deserialize, Serialize};

pub use super::super::spec::{
    Qwen35FullAttnFacts, Qwen35GdnFacts, Qwen35HybridFacts, Qwen35MlpKind, Qwen35MoeMlpFacts,
};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Qwen35CudaFacts {
    pub state_bf16: bool,

    pub warp_tiled: bool,

    pub warp_tiled_max: u32,

    pub cached_max: u32,

    #[serde(default)]
    pub verify_stash: bool,

    #[serde(default)]
    pub moe_cutlass_max_rows: u32,

    #[serde(default)]
    pub prefill_decode: bool,

    #[serde(default)]
    pub moe_residual_fold: bool,

    #[serde(default)]
    pub moe_shared_gate_dot: bool,

    #[serde(default)]
    pub moe_streamed_experts: bool,

    #[serde(default)]
    pub moe_force_general: bool,

    #[serde(default)]
    pub gate_up_fused: bool,

    #[serde(default)]
    pub proj_repr: WeightRepr,

    #[serde(default)]
    pub window_left: Vec<i32>,
}

impl Qwen35CudaFacts {
    pub fn qwen3_5_0_8b_synthetic() -> Self {
        Self {
            window_left: Vec::new(),
            state_bf16: true,
            warp_tiled: true,
            warp_tiled_max: 64,
            cached_max: 4096,
            verify_stash: true,

            moe_cutlass_max_rows: 512,
            prefill_decode: false,
            moe_residual_fold: true,
            moe_shared_gate_dot: true,
            moe_streamed_experts: false,
            moe_force_general: false,

            gate_up_fused: true,

            proj_repr: WeightRepr::Bf16,
        }
    }
}
