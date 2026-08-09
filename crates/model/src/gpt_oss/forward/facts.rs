//! `gpt-oss`'s per-backend binding facts.
//!
//! The SHAPE moved to `../spec.rs` (ungated: a row is written in it, and
//! a row must exist under every aspect). What a deployment RESOLVED --
//! the device's MXFP4 policy, a fused leg's route ceiling, whether the
//! expert slabs are streamed -- is known only when that backend's aspect
//! is compiled, so it stays here.

/// The shape, re-exported so a declaration reaches its facts and the
/// words they are stated in from one place.
pub use super::super::spec::GptOssFacts;

/// The CUDA backend's answers for a gpt-oss deployment — the bindings
/// and the admission thresholds, all resolved at load.
#[derive(Debug, Clone, PartialEq)]
pub struct GptOssCudaFacts {
    /// Whether the layer bank carries the per-expert POINTER ARRAYS the
    /// fused decode GEMV indexes (`expert_gate_up_packed_ptrs`). Built
    /// by the `RoutedDecode` MXFP4 policy, which is the engine default;
    /// the `NativeGemm` policy binds marlin views instead and reaches
    /// the experts through a per-expert loop no rectangle spells.
    pub mxfp4_decode_gemv: bool,
    /// `mxfp4_decode_max_routes` — the fused leg's admission threshold
    /// in ROUTES (`N * top_k`), default `32 * experts`. A fire past it
    /// takes the host-routed walk, which this declaration refuses by
    /// name rather than states.
    pub mxfp4_decode_max_routes: u32,
    /// Whether the experts are STREAMED through a slab cache. A streamed
    /// layer reaches the same fused kernels, but only after a host
    /// round-trip that decides what to page in — so a streamed
    /// deployment is outside the flat list until that is stated.
    pub streamed_experts: bool,
    /// The SLIDING WINDOW each layer attends over, `-1` for none —
    /// read through [`model_compiler::facts::window_left_at`], which is
    /// where the shape of this list is documented.
    ///
    /// The dispatch statements carry it, so no executor reaches into
    /// `fwd_cfg.per_layer_window_left` for it. Serde-defaulted, and
    /// empty reads as "no window", which is what every fixture written
    /// before this field meant.
    pub window_left: Vec<i32>,
}

impl GptOssCudaFacts {
    /// The L40S deployment's set, as the driver derives it: no
    /// streaming, the default policy's pointer arrays, and the default
    /// cap. SYNTHETIC until a live digest judges it — the standing
    /// contract for every `*_synthetic` fixture in this file.
    pub fn gpt_oss_20b_synthetic() -> Self {
        Self {
            // The fixture attends the whole context; a live gpt-oss
            // states its alternating list.
            window_left: Vec::new(),
            mxfp4_decode_gemv: true,
            mxfp4_decode_max_routes: 32 * 32,
            streamed_experts: false,
        }
    }
}
