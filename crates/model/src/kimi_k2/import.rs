//! Kimi K2's tensor names, in every vocabulary that spells them.
//!
//! Multi-head latent attention, the same seventeen rows as
//! [`crate::glm_5::import`] -- and written out here rather than shared,
//! because the two agreeing today is a fact about two releases and not a
//! lineage. `shared`'s bar is "more than one generation BINDS it", and what
//! binds these is a coincidence of shape: DeepSeek's MLA naming was adopted
//! independently by both, and either can move without the other.
//!
//! `mlp.gate` is the router; see [`crate::glm_5::import`] for the collision
//! with `mlp.gate_proj` that makes it worth saying twice.
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

/// Every tensor a Kimi K2 publishes, and what each vocabulary calls it.
///
/// No `gguf` column: [`crate::ingest`] has no arm for this generation, and a
/// column filled by guessing at llama.cpp's spelling would be a claim no file
/// on disk was read for. An unmapped GGUF is refused by name, which is the
/// answer that can be acted on.
pub const VOCAB: Vocab = Vocab(&[
    // ── Inside a decoder layer ───────────────────────────────────────
    Member::same("model.layers.{layer}.input_layernorm"),
    Member::same("model.layers.{layer}.mlp.down_proj"),
    Member::same("model.layers.{layer}.mlp.gate"),
    Member::same("model.layers.{layer}.mlp.gate_proj"),
    Member::same("model.layers.{layer}.mlp.shared_experts.gate_proj"),
    Member::same("model.layers.{layer}.post_attention_layernorm"),
    Member::same("model.layers.{layer}.self_attn.kv_a_layernorm"),
    Member::same("model.layers.{layer}.self_attn.kv_a_proj_with_mqa"),
    Member::same("model.layers.{layer}.self_attn.kv_b_proj"),
    Member::same("model.layers.{layer}.self_attn.o_proj"),
    Member::same("model.layers.{layer}.self_attn.q_a_layernorm"),
    Member::same("model.layers.{layer}.self_attn.q_a_proj"),
    Member::same("model.layers.{layer}.self_attn.q_b_proj"),
    Member::same("model.layers.{layer}.self_attn.q_proj"),
    // ── Outside it ───────────────────────────────────────────────────
    Member::same("model.embed_tokens"),
    Member::same("lm_head"),
    Member::same("model.norm"),
]);
