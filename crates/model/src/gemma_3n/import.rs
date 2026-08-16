//! Gemma 3n's tensor names, in every vocabulary that spells them.
//!
//! The per-layer-input machinery is what makes this table long. `altup`,
//! `laurel`, `per_layer_input_gate`, `per_layer_projection` and the two
//! model-level `per_layer_*` rows are all Gemma 3n's, and none of them
//! appears in any other generation here.
//!
//! `embed_tokens_per_layer` is the row to read against `embed_tokens`: two
//! embedding tables, one of which is indexed per layer, and a table that
//! collapsed them would be a model that loads and reads the wrong one.
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

/// Every tensor a Gemma 3n publishes, and what each vocabulary calls it.
///
/// No `gguf` column: [`crate::ingest`] has no arm for this generation, and a
/// column filled by guessing at llama.cpp's spelling would be a claim no file
/// on disk was read for. An unmapped GGUF is refused by name, which is the
/// answer that can be acted on.
pub const VOCAB: Vocab = Vocab(&[
    // ── Inside a decoder layer ───────────────────────────────────────
    Member::same("model.layers.{layer}.altup.modality_router"),
    Member::same("model.layers.{layer}.altup.router_norm"),
    Member::same("model.layers.{layer}.input_layernorm"),
    Member::same("model.layers.{layer}.laurel.linear_left"),
    Member::same("model.layers.{layer}.laurel.linear_right"),
    Member::same("model.layers.{layer}.laurel.post_laurel_norm"),
    Member::same("model.layers.{layer}.mlp.down_proj"),
    Member::same("model.layers.{layer}.mlp.gate_proj"),
    Member::same("model.layers.{layer}.mlp.up_proj"),
    Member::same("model.layers.{layer}.per_layer_input_gate"),
    Member::same("model.layers.{layer}.per_layer_projection"),
    Member::same("model.layers.{layer}.post_attention_layernorm"),
    Member::same("model.layers.{layer}.post_feedforward_layernorm"),
    Member::same("model.layers.{layer}.post_per_layer_input_norm"),
    Member::same("model.layers.{layer}.pre_feedforward_layernorm"),
    Member::same("model.layers.{layer}.self_attn.k_norm"),
    Member::same("model.layers.{layer}.self_attn.k_proj"),
    Member::same("model.layers.{layer}.self_attn.o_proj"),
    Member::same("model.layers.{layer}.self_attn.q_norm"),
    Member::same("model.layers.{layer}.self_attn.q_proj"),
    Member::same("model.layers.{layer}.self_attn.v_norm"),
    Member::same("model.layers.{layer}.self_attn.v_proj"),
    // ── Outside it ───────────────────────────────────────────────────
    Member::same("model.embed_tokens"),
    Member::same("model.embed_tokens_per_layer"),
    Member::same("lm_head"),
    Member::same("model.norm"),
    Member::same("model.per_layer_model_projection"),
    Member::same("model.per_layer_projection_norm"),
]);
