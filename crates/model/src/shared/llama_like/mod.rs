//! `llama_like` — the shape a dozen generations share.
//!
//! A FAMILY, not a generation: llama 2/3, mistral, qwen 2/3, phi-3, olmo 2/3
//! and the rest are one forward pass parameterized by facts. It lives under
//! `shared/` for the reason ChatML does — what more than one generation
//! binds is not any one generation's property.

/// The forward pass: a semantic text that names operations and never kernels,
/// and one lowered text per backend.
pub mod forward;

/// The SHAPE: the numbers a checkpoint of this family has.
///
/// Ungated. A catalog row is written in these words, and a row must
/// exist under every aspect.
pub mod spec;

/// The three projections a row of this family makes: its tensor
/// manifest, its `Deployment`, and its traced text.
pub mod project;

/// The AUTHORING passes, which are this family's and not one generation's.
///
/// `author_llama_like` serves llama-3, mistral and qwen-2/3;
/// `author_dense` serves gemma-2/3, gemma-3n, mistral-3, olmo-2/3 and
/// qwen-2; `author_llama_mlx` is their Metal lowering. Ten generations
/// bind three functions.
///
/// They lived in `llama_3/contract.rs` because llama-3 wrote them down
/// first, and every other generation reached across a sibling edge to
/// call them — the exact shape `tests/sibling_isolation.rs` forbids and
/// the exact shape that made `qwen_2` depend on `qwen_3` back when these
/// were crates. Being written first is a fact about who wrote it, not a
/// claim of ownership.
#[cfg(feature = "contract")]
pub mod contract;
