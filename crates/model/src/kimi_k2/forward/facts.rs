use serde::{Deserialize, Serialize};

pub use super::super::spec::{KimiFacts, KimiMlaFacts, KimiMoeFacts};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KimiCudaFacts {

    pub q_kv_a_fused: bool,

    pub rope_yarn_original: bool,
}

impl KimiCudaFacts {

    pub fn kimi_k2_synthetic() -> Self {
        KimiCudaFacts {
            q_kv_a_fused: true,
            rope_yarn_original: true,
        }
    }
}
