//! The two registries that turn a model's name into an implementation.
//!
//! The implementations are not here. Each model *generation* is its own crate
//! under `model/` — `pie-model-llama-3`, `pie-model-qwen-3-5`, twenty of them
//! — named `<vendor>_<generation>` with a version's dots as underscores. A
//! generation is a crate rather than a module because the generations differ
//! enough to be worth separating (templates, storage schema, and the forward
//! programs still to land), and because a crate boundary is the only thing
//! Rust has that stops one generation reaching into another.
//!
//! What this crate holds is the dispatch:
//!
//! * [`contract`] — `model_type` to the author that writes its load contract.
//! * [`instruct`] — `arch_name` to the chat template that formats for it.
//!
//! The two deliberately partition the model space differently: templates
//! split by release, contracts by storage schema. So a generation crate is
//! reached through a *row*, never as a module path, and every N:1 reuse is
//! written out — a dozen `model_type`s share Llama 3's dense author, and
//! Qwen3's ChatML template serves four vendors.
//!
//! A generation with no implementation of its own gets no crate: `mixtral`,
//! `deepseek_v2`, `deepseek_v3`, `ministral3`, `gemma3n`, `kimi_k25` and
//! `qwen3_6` are rows naming the generation whose implementation they share.
//!
//! [`multimodal`] is the exception still here: host-side image and audio
//! decode is a third aspect and it is family-aware, so it has not been split
//! per generation the way chat and contract have.
//!
//! The model *service* — the global model/tokenizer cache — used to live
//! here too. It was runtime machinery rather than family knowledge, and it
//! moved to `runtime/engine/src/model.rs` once its consumers split cleanly:
//! everything but [`ModelMetadata`] was read by the engine alone.

// The aspects' shared vocabulary and traits.
#[cfg(feature = "contract")]
pub mod contract;
#[cfg(feature = "chat")]
pub mod instruct;

// The model generations are crates now, one per generation, under `model/`.
// This crate names them in its two registries (`contract::HF_ROWS`,
// `instruct::create`) and nowhere else — a generation is reached through the
// row that dispatches to it, never as a module path.
//
// A generation with no implementation of its own has no crate: it is a row
// naming the generation whose implementation it shares. `mixtral`,
// `deepseek_v2`, `deepseek_v3`, `ministral3`, `gemma3n`, `kimi_k25` and
// `qwen3_6` are those today.

// ── What a served model's metadata is ───────────────────────────────
//
// The shape an artifact's compiled metadata arrives in. The *service* that
// used to sit here — the global `OnceLock` cache, `register`/`model()` — was
// runtime machinery rather than family knowledge and moved to
// `runtime/engine/src/model.rs`; this stayed, because the worker reads it
// without linking the runtime.
// Re-exported so a consumer's path survives the split: `ModelMetadata` and
// the multimodal decode moved to `pie-model-common`, which is where a crate
// that needs them without the generations can reach them.
pub use pie_model_common::ModelMetadata;
// Host-side image/audio decode. Family-aware — it dispatches on a
// `VisionArch` and carries per-generation configs — so it is *not* in
// `pie-model-common`, whose guard would rightly refuse it. Splitting it per
// generation is the same move chat and contract have already made.
#[cfg(feature = "chat")]
pub mod multimodal;
