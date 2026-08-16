//! `gpt-oss`'s per-backend binding facts.
//!
//! The SHAPE lives in `../spec.rs`, ungated, because a row must exist under
//! every aspect. What a deployment RESOLVED is known only once that backend's
//! aspect is compiled.

/// The shape, re-exported so a declaration reaches its facts from one place.
pub use super::super::spec::GptOssFacts;

/// The CUDA backend's answers for a gpt-oss deployment, resolved at load.
#[derive(Debug, Clone, PartialEq)]
pub struct GptOssCudaFacts {
    /// Whether the layer bank carries the per-expert POINTER ARRAYS the fused
    /// decode GEMV indexes — built by the default `RoutedDecode` MXFP4 policy,
    /// where `NativeGemm` binds marlin views instead.
    pub mxfp4_decode_gemv: bool,
    /// The fused leg's admission threshold in ROUTES (`N * top_k`), default
    /// `32 * experts`; a fire past it takes the host-routed walk, which this
    /// declaration refuses by name.
    pub mxfp4_decode_max_routes: u32,
    /// Whether the experts are STREAMED through a slab cache: the same fused
    /// kernels, but only after a host round-trip deciding what to page in.
    pub streamed_experts: bool,
}

impl GptOssCudaFacts {
    /// The L40S deployment's set. SYNTHETIC until a live digest judges it.
    pub fn gpt_oss_20b_synthetic() -> Self {
        Self {
            mxfp4_decode_gemv: true,
            mxfp4_decode_max_routes: 32 * 32,
            streamed_experts: false,
        }
    }
}
