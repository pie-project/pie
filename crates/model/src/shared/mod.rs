//! The vocabulary a generation is allowed to name.
//!
//! A generation module may name this module and nothing else — that is the
//! whole of the sibling-isolation rule (`tests/sibling_isolation.rs`), and
//! this module is the half of it that makes the rule livable. Without
//! somewhere legitimate to share from, real sharing has to hide as a sibling
//! dependency, which is exactly what happened while the generations were
//! crates: `qwen_2` depended on `qwen_3`.
//!
//! # Two kinds of shared thing, one directory
//!
//! These used to be two places — a `shared/` directory and a scatter of
//! modules at the crate root — and the distinction they drew was real:
//!
//! * **General vocabulary.** [`builder`], [`policy`], [`probe`], [`moe`],
//!   [`mlx`], [`weight_names`], [`decoders`] — true of models *in general*.
//!   The authoring DSL, the quantization policy, the tensor-name rules.
//! * **Shared implementations.** [`llama_like`], [`chatml`], [`gemma_chat`],
//!   [`deepseek`], [`kimi`] — one specific answer that a specific set of
//!   generations happen to have in common.
//!
//! The distinction survives as prose because it is worth knowing, but it
//! stopped earning a directory boundary the moment the crate root became the
//! catalog's own surface. What belongs at the root is the table and its
//! answers — [`catalog`](crate::catalog), [`manifest`](crate::manifest),
//! [`deployment`](crate::deployment) — plus one directory per generation.
//! Everything a generation *reaches for* is here, so "may I name this?" has
//! one answer and not two.
//!
//! The bar for landing here is unchanged and is the important part: **more
//! than one generation binds it**, not "it looks reusable". A thing one
//! generation uses stays in that generation, however general it looks; it
//! moves here the day a second one wants it.

// ── General vocabulary: the contract aspect ──────────────────────────

/// The authoring DSL: what a contract pass says about a checkpoint.
#[cfg(feature = "contract")]
pub mod builder;

/// MLX-published checkpoints, whose quantization the author must lower.
#[cfg(feature = "contract")]
pub mod mlx;

/// Routed-expert banks, and how an author binds one.
#[cfg(feature = "contract")]
pub mod moe;

/// The quantization vocabulary — an encoding is a POLICY, never an
/// identity, which is why it is here and not a column of the table.
#[cfg(feature = "contract")]
pub mod policy;

/// Asking the checkpoint what it actually shipped.
#[cfg(feature = "contract")]
pub mod probe;

/// The tensor-naming rules an author resolves a logical name through.
///
/// [`catalog::LoadShape`](crate::catalog::LoadShape) is what it reads, and
/// that is the authoring aspect's vocabulary.
#[cfg(feature = "contract")]
pub mod weight_names;

// ── General vocabulary: the chat aspect ──────────────────────────────

/// Incremental decoding: bytes out of tokens, for a streaming turn.
#[cfg(feature = "chat")]
pub mod decoders;

// ── Shared implementations ───────────────────────────────────────────

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
