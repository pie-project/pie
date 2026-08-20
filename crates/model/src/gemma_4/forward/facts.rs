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
    /// Each layer's `layer_scalar`, the `[1]` tensor its PLE landing
    /// multiplies the residual by.
    ///
    /// A CHECKPOINT'S NUMBER, so it belongs in the statement rather than in a
    /// fact the fire asks for (`.wiki/migration.md` §11.20's rule: two fires of
    /// the same deployment cannot see different answers here). It reaches this
    /// struct from `Deployed::layer_scalars`, which the loader fills by reading
    /// each tensor to the host once.
    ///
    /// Empty means "no row states one", and the landing then scales by the
    /// identity — which is what every synthetic fixture does and what this
    /// text did for every layer of every deployment until the walk against
    /// transformers was able to run and disagree.
    #[serde(default)]
    pub layer_scalars: Vec<f32>,
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
            // A fixture states no checkpoint, so it states no scalar and the
            // landing scales by the identity.
            layer_scalars: Vec::new(),
        }
    }
}

// ── gpt-oss ────────────────────────────────────────────────────────────
