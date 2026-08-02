//! `llama_like` — the shape a dozen generations share.
//!
//! A FAMILY, not a generation: llama 2/3, mistral, qwen 2/3, phi-3, olmo 2/3
//! and the rest are one forward pass parameterized by facts. It lives under
//! `families/` for the reason ChatML does — what more than one generation
//! binds is not any one generation's property.

/// The forward pass: a semantic text that names operations and never kernels,
/// and one lowered text per backend.
#[cfg(feature = "forward")]
pub mod forward;
