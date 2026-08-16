//! The vocabulary a generation is allowed to name.
//!
//! A generation module may name this module and nothing else — the whole of
//! the sibling-isolation rule (`tests/sibling_isolation.rs`), and this is the
//! half of it that makes the rule livable. The bar for landing here is **more
//! than one generation binds it**, not "it looks reusable": what one
//! generation uses moves here the day a second one wants it.

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

/// The quantization vocabulary — an encoding is a POLICY, never an identity.
#[cfg(feature = "contract")]
pub mod policy;

/// Asking the checkpoint what it actually shipped.
#[cfg(feature = "contract")]
pub mod probe;

/// The tensor-naming rules an author resolves a logical name through.
#[cfg(feature = "contract")]
pub mod weight_names;

/// The tensor names a multimodal TOWER publishes, in launcher order.
#[cfg(feature = "contract")]
pub mod tower_names;

// ── General vocabulary: the chat aspect ──────────────────────────────

/// Incremental decoding: bytes out of tokens, for a streaming turn.
#[cfg(feature = "chat")]
pub mod decoders;

// ── Shared implementations ───────────────────────────────────────────

/// ChatML: the `<|im_start|>` / `<|im_end|>` conversation format.
#[cfg(feature = "chat")]
pub mod chatml;

/// Gemma's `<start_of_turn>` conversation format.
#[cfg(feature = "chat")]
pub mod gemma_chat;

/// DeepSeek's `<｜User｜>` / `<｜Assistant｜>` conversation format.
#[cfg(feature = "chat")]
pub mod deepseek;

/// Kimi's `<|im_middle|>` conversation format.
#[cfg(feature = "chat")]
pub mod kimi;

/// The tensor names a generation publishes, in every vocabulary that
/// spells them — the table `<generation>/import.rs` fills in.
pub mod vocabulary;

pub mod llama_like;
