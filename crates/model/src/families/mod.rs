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
pub mod llama_like;
