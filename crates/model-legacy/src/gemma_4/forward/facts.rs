use serde::{Deserialize, Serialize};

pub use super::super::spec::{Gemma4Facts, Gemma4Mixture};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Gemma4CudaFacts {
    pub fused_qkv: bool,

    pub gate_up_fused: bool,

    pub kv_native_bf16: bool,

    #[serde(default)]
    pub window_left: Vec<i32>,

    #[serde(default)]
    pub layer_scalars: Vec<f32>,
}

impl Gemma4CudaFacts {
    pub fn gemma_4_e4b_synthetic() -> Self {
        Self {
            window_left: Vec::new(),
            fused_qkv: true,
            gate_up_fused: true,
            kv_native_bf16: true,

            layer_scalars: Vec::new(),
        }
    }
}
