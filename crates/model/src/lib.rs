//! What a model IS: the registry that turns a model's name into an
//! implementation, and the implementations it dispatches to.
//!
//! [`contract`] writes a row's load contract, [`instruct`] formats for it,
//! and `<generation>::forward` is the forward pass in `model-compiler`'s
//! tracing eDSL. The first two are aspects of a row; the third is reached by
//! a driver naming a family's text directly, so it sits on the generation
//! module. Generations that share an implementation get no module of their
//! own, and `tests/sibling_isolation.rs` enforces the layout rule: a
//! generation module may name the shared root, never a sibling.


/// What a driver needs to serve a checkpoint, with no family name in it.
pub mod deployment;

#[cfg(feature = "chat")]
pub mod instruct;

pub mod shared;


/// The tensor manifest: what a checkpoint of a row MUST contain. Identity and
/// validation are the same operation here; ungated, so every aspect that can
/// name a row can ask what it is made of.
pub mod manifest;

/// The catalog: one row per model, one row per answer.
pub mod catalog;

/// What a checkpoint's FILES say about how its numbers are stored: a property
/// of a file, where a row is a property of a model — Qwen3-8B is one row and four downloads.
pub mod encoding;

/// The load path itself: a row in, a plan out, stated once for every driver.
#[cfg(feature = "contract")]
pub mod boot;
#[cfg(feature = "contract")]
pub mod contract;
/// The ingest aspect: a foreign checkpoint vocabulary in, this crate's out.
/// Sits with `contract` because it is the layer below it -- the same
/// question, one step earlier.
#[cfg(feature = "contract")]
pub mod ingest;
#[cfg(feature = "chat")]
pub mod multimodal;

// ── The generations ──────────────────────────────────────────────────
//
// LOAD-BEARING TEXT: `catalog::tests::a_generation_with_no_rows_says_so`
// splits this file on the header above to read the generation list from
// what `lib.rs` declares rather than from the filesystem. Named
// `<vendor>_<generation>`, a version's dots as underscores.
pub mod csm;
pub mod deepseek_r1;
pub mod deepseek_v4;
pub mod gemma_2;
pub mod gemma_3;
pub mod gemma_3n;
pub mod gemma_4;
pub mod glm_5;
pub mod gpt_oss;
pub mod kimi_k2;
pub mod kimi_k3;
pub mod llama_2;
pub mod llama_3;
pub mod mistral_3;
pub mod nemotron_h;
pub mod olmo_2;
pub mod olmo_3;
pub mod phi_3;
pub mod qwen_2;
pub mod qwen_3;
pub mod qwen_3_5;

/// One row that describes no real checkpoint, so that a test can afford to
/// write one. Absent unless asked for.
#[cfg(feature = "test-rows")]
pub mod test_rows;

// What a served model's compiled metadata IS; the worker reads it without
// linking either aspect.
mod metadata;
pub use metadata::ModelMetadata;
