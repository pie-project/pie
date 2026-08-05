//! What each model family is, backend-blind.
//!
//! Organized family-first: one directory per lineage (`llama/`, `qwen3/`,
//! `gemma/`, …), one file per aspect inside it — `chat/` today; `contract.rs`
//! and `forward.rs` as those aspects migrate out of the drivers. `common/`
//! holds what every family shares and knows about none of them.
//!
//! The aspects deliberately partition the model space differently: templates
//! split by release (llama2 vs llama3), contracts by storage schema (phi3
//! loads as llama), forward passes by compute graph. So directories organize
//! and never dispatch — each aspect has its own registry keyed by
//! `model_type`, with every N:1 reuse written out as a row
//! (`instruct::create` is the chat aspect's).
//!
//! This crate also still carries the model *service* — the global model/
//! tokenizer cache below — which is runtime machinery rather than family
//! knowledge; it is slated to move back under `runtime/` once the contract
//! aspect lands and the crate's consumers split into knowledge-only (capi)
//! and runtime callers.

// The aspects' shared vocabulary and traits.
pub mod common;
#[cfg(feature = "contract")]
pub mod contract;
#[cfg(feature = "chat")]
pub mod instruct;
#[cfg(feature = "chat")]
pub mod multimodal;

// The families, one directory per lineage.
#[cfg(feature = "contract")]
pub mod csm;
#[cfg(feature = "contract")]
pub mod deepseek_v4;
pub mod gemma;
#[cfg(feature = "contract")]
pub mod glm5;
pub mod gptoss;
pub mod kimi;
pub mod llama;
pub mod mistral;
#[cfg(feature = "contract")]
pub mod nemotron_h;
pub mod olmo;
pub mod phi;
pub mod qwen3;
#[cfg(feature = "contract")]
pub mod qwen3_5;
pub mod r1;

// ── The model service ────────────────────────────────────────────────
//
// The runtime's global model/tokenizer cache — service machinery, not family
// knowledge, gated with the chat aspect it serves and re-exported so its
// consumers' paths (`pie_model::register`, …) survive the split. It is
// slated to move back under `runtime/` (see the module doc above).
#[cfg(feature = "chat")]
mod service;
#[cfg(feature = "chat")]
pub use service::*;
