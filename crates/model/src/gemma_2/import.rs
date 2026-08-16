//! Gemma 2's tensor names, in every vocabulary that spells them.
//!
//! Four norms per layer, where a llama-like family has two. Gemma sandwiches
//! both the attention and the MLP, so `pre_feedforward_layernorm` and
//! `post_feedforward_layernorm` are rows here and are absent from
//! [`crate::shared::llama_like::import`].
//!
//! There is no GGUF column. `general.architecture` is `gemma2` in llama.cpp
//! and [`crate::ingest`] has no arm for it; the arm that exists is `gemma3`,
//! whose table is [`crate::gemma_3::import`] and whose norm rows are folded
//! by one. Copying that file's `folded_constant` here on the strength of the
//! shared family name is the exact mistake its own documentation exists to
//! prevent.
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

/// Every tensor a Gemma 2 publishes, and what each vocabulary calls it.
///
/// No `gguf` column: [`crate::ingest`] has no arm for this generation, and a
/// column filled by guessing at llama.cpp's spelling would be a claim no file
/// on disk was read for. An unmapped GGUF is refused by name, which is the
/// answer that can be acted on.
pub const VOCAB: Vocab = Vocab(&[
    // ── Inside a decoder layer ───────────────────────────────────────
    Member::same("model.layers.{layer}.input_layernorm"),
    Member::same("model.layers.{layer}.mlp.down_proj"),
    Member::same("model.layers.{layer}.mlp.gate_proj"),
    Member::same("model.layers.{layer}.mlp.up_proj"),
    Member::same("model.layers.{layer}.post_attention_layernorm"),
    Member::same("model.layers.{layer}.post_feedforward_layernorm"),
    Member::same("model.layers.{layer}.pre_feedforward_layernorm"),
    Member::same("model.layers.{layer}.self_attn.k_proj"),
    Member::same("model.layers.{layer}.self_attn.o_proj"),
    Member::same("model.layers.{layer}.self_attn.q_norm"),
    Member::same("model.layers.{layer}.self_attn.q_proj"),
    Member::same("model.layers.{layer}.self_attn.v_proj"),
    // ── Outside it ───────────────────────────────────────────────────
    Member::same("model.embed_tokens"),
    Member::same("lm_head"),
    Member::same("model.norm"),
]);
