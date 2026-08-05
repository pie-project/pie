//! What every model generation shares — and nothing about any one of them.
//!
//! The test for whether something belongs here: it must be *about models in
//! general* — an authoring pass, a decoder that classifies a token stream, a
//! policy type a contract author is parameterized by — and never a fact about
//! one generation's checkpoints or templates. Those facts live in that
//! generation's own crate.
//!
//! The rule is not left to prose: `pie-model`'s `tests/common_is_thin.rs`
//! fails the build if a family name reaches this crate's code, or if a `pub`
//! item here is named after one.
//!
//! **Thin means no knowledge, not few lines.** `builder.rs` is the largest
//! file under `model/` and that is correct — it is the authoring DSL, and
//! every generation calls its passes directly. Bulk in a shared crate is only
//! a smell when it is knowledge; a pass is mechanism.
//!
//! Split by aspect, exactly as the generation crates are:
//!
//! - `contract`: the authoring DSL ([`builder`]), the vocabulary it is
//!   parameterized by ([`facts`], [`policy`]), and the shared shapes
//!   ([`probe`], [`moe`], [`mlx`]).
//! - `chat`: the [`instruct`] trait and events every generation implements,
//!   and the [`decoders`] behind them.
//!
//! Host-side multimodal decode is deliberately NOT here. It is a third
//! aspect and it is family-aware — `GemmaImageConfig`, `QwenVisionConfig`, a
//! `VisionArch` it dispatches on — so it stays in `pie-model` until it is
//! split per generation the way chat and contract already are. Putting it
//! here would have meant exempting it from the guard, which is how a rule
//! becomes a comment.
//!
//! [`metadata`] is outside both: it is what a served model's compiled
//! metadata *is*, which the worker reads without linking either aspect.

// ── The contract aspect ──────────────────────────────────────────────
#[cfg(feature = "contract")]
pub mod builder;
#[cfg(feature = "contract")]
pub mod facts;
#[cfg(feature = "contract")]
pub mod mlx;
#[cfg(feature = "contract")]
pub mod moe;
#[cfg(feature = "contract")]
pub mod policy;
#[cfg(feature = "contract")]
pub mod probe;

// ── The chat aspect ──────────────────────────────────────────────────
#[cfg(feature = "chat")]
pub mod decoders;
#[cfg(feature = "chat")]
pub mod instruct;

// ── Neither aspect ───────────────────────────────────────────────────
mod metadata;
pub use metadata::ModelMetadata;
