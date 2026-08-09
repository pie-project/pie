//! What more than one generation binds.
//!
//! A generation module may name this and the shared root, and nothing else —
//! that is the whole of the sibling-isolation rule
//! (`tests/sibling_isolation.rs`), and this module is the half of it that
//! makes the rule livable. Without somewhere legitimate to share from, real
//! sharing has to hide as a sibling dependency, which is exactly what happened
//! while the generations were crates: `qwen_2` depended on `qwen_3`.
//!
//! The bar for landing here is **more than one generation binds it**, not
//! "it looks reusable". A thing one generation uses stays in that generation,
//! however general it looks; it moves here the day a second one wants it.
//!
//! What that is NOT is a place for the shared root's contents. The root holds
//! what is true of models *in general* — the authoring DSL, the decoders, the
//! policy vocabulary. `families/` holds a specific implementation that a
//! specific set of generations happen to have in common.

/// ChatML: the `<|im_start|>` / `<|im_end|>` conversation format.
///
/// Qwen3 wrote it down first, which is why it lived in `qwen_3/chat.rs` and
/// why `qwen_2` reached across for it. But `qwen3`, `qwen3_5`, `nemotron_h`,
/// `glm_moe_dsa` and `qwen2` all bind it, with the differences between them
/// expressed entirely as [`chatml::ChatMLConfig`] — thinking on or off, tools
/// on or off, which stop tokens, what generation suffix. A format five
/// generations parameterize is not one generation's fact.
#[cfg(feature = "chat")]
pub mod chatml;

/// Gemma's `<start_of_turn>` conversation format.
///
/// Gemma 3 wrote it down first, which is why it lived in
/// `gemma_3/chat.rs` — and gemma-3n bound it from there, across a
/// sibling edge the isolation rule forbids. The template itself already
/// knew: [`gemma_chat::Gemma3Variant`] has named a gemma-3n arm since
/// the day it was written. Two generations, one format, so it is here
/// and `gemma_3::chat` re-exports it.
#[cfg(feature = "chat")]
pub mod gemma_chat;

/// DeepSeek's `<｜User｜>` / `<｜Assistant｜>` conversation format.
///
/// DeepSeek-R1 wrote it down first, which is why it lived in
/// `deepseek_r1/chat.rs`. The old `instruct::create` pointed a second
/// architecture string at that one constructor — `"deepseek_v4"`, a
/// generation with its own directory, its own contract and its own text
/// — which was a sibling reach dressed as a table row. Now that a row
/// answers the chat question itself, the reach would be a `use`, so the
/// format is here and `deepseek_r1::chat` re-exports it.
#[cfg(feature = "chat")]
pub mod deepseek;

/// Kimi's `<|im_middle|>` conversation format.
///
/// Kimi K2 wrote it down first, which is why it lived in
/// `kimi_k2/chat.rs`. The old `instruct::create` pointed three
/// architecture strings at that one constructor — `"kimi_k2"`,
/// `"kimi_k25"` and `"kimi_k3"` — which was a sibling reach dressed as
/// a table row. Now that a row answers the chat question itself, the
/// reach would be a `use`, so the format is here and `kimi_k2::chat`
/// re-exports it.
#[cfg(feature = "chat")]
pub mod kimi;

pub mod llama_like;
