//! DeepSeek V4's tensor names, in every vocabulary that spells them.
//!
//! The one generation here whose names are not HuggingFace's usual shape.
//! `attn.wq`, `attn.wkv`, `attn.wo`, `attn_norm` and `mlp_norm` are the
//! release's own spelling -- `attn` rather than `self_attn`, `w*` rather than
//! `*_proj`, and the norms named for the block they precede rather than for
//! where they sit.
//!
//! That makes this the generation where the `pie` and `hf` columns being
//! equal says the most. It is not that pie adopted HuggingFace's convention
//! here; it is that pie adopted THIS release's, which is a different
//! convention again -- and until this table existed, that was recorded
//! nowhere but in the rows.
//!
//! `attn.wo_a` / `attn.wo_b` beside `attn.wo` is a division, not a spelling:
//! some releases factor the output projection and some do not.
//!
//! # What the `hf` column is grounded in, and what it is not
//!
//! Every row below is a name a row in this generation asks for, lowered
//! through [`crate::manifest::Observed::logical`] -- which is the same
//! lowering identification uses, so the table is checked against the catalog
//! rather than written from memory. What it is NOT is a reading of a
//! checkpoint on disk: pie's import is the identity today, so a checkpoint
//! that spells a tensor some other way was never renamed and never had to be.
//! The `hf` column therefore states what these rows imply, and a family whose
//! release turns out to spell something differently corrects it here, in the
//! one place that would then matter.

use crate::shared::vocabulary::{Member, Vocab};

/// Every tensor a DeepSeek V4 publishes, and what each vocabulary calls it.
///
/// No `gguf` column: [`crate::ingest`] has no arm for this generation, and a
/// column filled by guessing at llama.cpp's spelling would be a claim no file
/// on disk was read for. An unmapped GGUF is refused by name, which is the
/// answer that can be acted on.
pub const VOCAB: Vocab = Vocab(&[
    // ── Inside a decoder layer ───────────────────────────────────────
    Member::same("model.layers.{layer}.attn.kv_norm"),
    Member::same("model.layers.{layer}.attn.q_norm"),
    Member::same("model.layers.{layer}.attn.wkv"),
    Member::same("model.layers.{layer}.attn.wo"),
    Member::same("model.layers.{layer}.attn.wo_a"),
    Member::same("model.layers.{layer}.attn.wo_b"),
    Member::same("model.layers.{layer}.attn.wq"),
    Member::same("model.layers.{layer}.attn.wq_a"),
    Member::same("model.layers.{layer}.attn.wq_b"),
    Member::same("model.layers.{layer}.attn_norm"),
    Member::same("model.layers.{layer}.ffn.gate"),
    Member::same("model.layers.{layer}.mlp.down_proj"),
    Member::same("model.layers.{layer}.mlp.gate_proj"),
    Member::same("model.layers.{layer}.mlp.up_proj"),
    Member::same("model.layers.{layer}.mlp_norm"),
    // ── Outside it ───────────────────────────────────────────────────
    Member::same("model.embed_tokens"),
    Member::same("lm_head"),
    Member::same("model.norm"),
]);
