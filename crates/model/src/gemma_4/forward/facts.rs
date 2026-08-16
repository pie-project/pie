//! `gemma-4`'s load-time facts: what the LOADER materialised, which is a
//! question only a build that traces can ask. The semantic shape is the
//! catalog row's and is re-exported from `../spec.rs` below.

use serde::{Deserialize, Serialize};

/// The SEMANTIC shape, re-exported so the traced text below keeps naming it
/// where it always did.
pub use super::super::spec::{Gemma4Facts, Gemma4Mixture};

/// The CUDA backend's load-time facts for gemma-4 — the BINDING
/// questions its class traces resolve at trace time.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Gemma4CudaFacts {
    /// The loader bound one packed `[Hq + 2*Hk, hidden]` projection
    /// (`qkv_proj_fused`) — llama_like's `fused_qkv`, same question.
    pub fused_qkv: bool,
    /// The loader bound a packed gate‖up bank — llama_like's
    /// `gate_up_fused`, same question, different activation behind it.
    pub gate_up_fused: bool,
    /// The KV cache is native bf16, so the fused decode post may write pages
    /// directly. One of the four terms `can_fuse_packed_qkv_post` reads.
    pub kv_native_bf16: bool,
    /// The SLIDING WINDOW each layer attends over, `-1` for none — read
    /// through [`model_ir::facts::window_left_at`], which documents its shape.
    #[serde(default)]
    pub window_left: Vec<i32>,
}

impl Gemma4CudaFacts {
    /// SYNTHETIC fixture: it pins the GOLDEN FORM of the traced arms, not a
    /// deployment's truth. The live derivation and its digest are the executor's.
    pub fn gemma_4_e4b_synthetic() -> Self {
        Self {
            // The fixture attends the whole context; a live gemma-4
            // deployment states its per-layer list.
            window_left: Vec::new(),
            fused_qkv: true,
            gate_up_fused: true,
            kv_native_bf16: true,
        }
    }
}

// ── gpt-oss ────────────────────────────────────────────────────────────
