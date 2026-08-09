//! `gemma-4`'s load-time facts.

use serde::{Deserialize, Serialize};

/// The SEMANTIC shape, which now lives in [`crate::gemma_4::spec`] and
/// is re-exported here so the traced text below keeps naming it where it
/// always did.
///
/// It moved because a catalog row is `const` and ungated: `manifest()`,
/// `load_shape()` and `deployment()` are answered in builds with no
/// tracer compiled in, and a struct behind `#[cfg(feature = "forward")]`
/// cannot be the words a row is written in. What stayed is
/// [`Gemma4CudaFacts`] — what the LOADER materialised, which is a
/// question only a build that traces can ask.
pub use super::super::spec::{Gemma4Facts, Gemma4Mixture};

/// The CUDA backend's load-time facts for gemma-4 — the BINDING
/// questions its class traces resolve at trace time.
///
/// Three, and all three are "what did the loader materialise", which is
/// the taxonomy's first row: a load-time fact is a trace-time `match`,
/// erased.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Gemma4CudaFacts {
    /// The loader bound one packed `[Hq + 2*Hk, hidden]` projection
    /// (`qkv_proj_fused`) — llama_like's `fused_qkv`, same question.
    pub fused_qkv: bool,
    /// The loader bound a packed gate‖up bank — llama_like's
    /// `gate_up_fused`, same question, different activation behind it.
    pub gate_up_fused: bool,
    /// The KV cache is native bf16, so the fused decode post may write
    /// pages directly. One of the four terms
    /// `can_fuse_packed_qkv_post` reads; the other three are the
    /// declaration's own (`partial` is a layer-kind fact, hooks and the
    /// fire class are class/guard vocabulary).
    pub kv_native_bf16: bool,
    /// The SLIDING WINDOW each layer attends over, `-1` for none —
    /// read through [`model_compiler::facts::window_left_at`], which is
    /// where the shape of this list is documented.
    ///
    /// The dispatch statements carry it, so no executor reaches into
    /// `fwd_cfg.per_layer_window_left` for it. Serde-defaulted, and
    /// empty reads as "no window", which is what every fixture written
    /// before this field meant.
    #[serde(default)]
    pub window_left: Vec<i32>,
}

impl Gemma4CudaFacts {
    /// SYNTHETIC fixture — the same standing caveat every `*CudaFacts`
    /// constructor here carries: it pins the GOLDEN FORM of the traced
    /// arms, not a deployment's truth. The live derivation and its
    /// digest are the executor rung's, and the digest is what corrects a
    /// guess on first boot.
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
