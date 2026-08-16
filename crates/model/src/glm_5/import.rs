//! GLM-5's tensor names, in every vocabulary that spells them.
//!
//! Multi-head latent attention, so the projections are the compressed pair
//! rather than Q/K/V: `q_a_proj` down to a latent, `q_b_proj` back up, and
//! `kv_a_proj_with_mqa` carrying the rope half alongside the latent. The two
//! `*_a_layernorm` rows sit on the latents, which is why they are named for
//! the projection and not for a position in the block.
//!
//! `self_attn.q_proj` is here beside `q_a_proj`/`q_b_proj` because a variant
//! that does not compress queries publishes the plain one, and the row that
//! asks for it is not the row that asks for the pair.
//!
//! `mlp.gate` is the ROUTER -- a gate that selects experts, not the
//! `gate_proj` half of a gated MLP. The two are one edit apart and one is
//! `[E, H]` while the other is `[I, H]`.
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

/// Every tensor a GLM-5 publishes, and what each vocabulary calls it.
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
