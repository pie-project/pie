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

/// The SHAPE: the numbers a checkpoint of this family has.
///
/// Ungated. A catalog row is written in these words, and a row must
/// exist under every aspect.
pub mod spec;

/// The three projections a row of this family makes: its tensor
/// manifest, its `Deployment`, and its traced text.
pub mod project;
