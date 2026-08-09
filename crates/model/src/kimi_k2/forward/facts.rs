//! kimi_k2's per-backend binding facts.
//!
//! The SHAPE moved to `../spec.rs` (ungated: a row is written in it, and
//! a row has to exist under every aspect). What a deployment BOUND — one
//! fused latent GEMM instead of two, a YaRN rope the config asked for —
//! is per-backend and known only when that backend's aspect is compiled,
//! so it stays here.

use serde::{Deserialize, Serialize};

/// The shape, re-exported so a declaration reaches its facts and the
/// words they are stated in from one place.
pub use super::super::spec::{KimiFacts, KimiMlaFacts, KimiMoeFacts};

/// The CUDA reading's deployment facts — the choices the hand-written
/// pass makes from the BINDING and the config, resolved once at load.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KimiCudaFacts {
    /// `Lw.q_kv_a_fused != nullptr`: one GEMM for the query's and the
    /// KV's latents instead of two, and a STRIDED norm over the query
    /// half — neither latent is a contiguous block of the result.
    pub q_kv_a_fused: bool,
    /// The config asks for YaRN (`rope_scaling_kind`), so the rope is
    /// `kernels::rope::rope_yarn_original_bf16` rather than the plain one.
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
