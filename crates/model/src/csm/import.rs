//! CSM's tensor names, in every vocabulary that spells them.
//!
//! Five towers, which is what makes this the longest table here: a
//! `backbone_model` transformer, a `depth_decoder` with its own transformer
//! and embeddings, and a `codec_model` holding an encoder, a decoder and two
//! residual vector quantizers.
//!
//! Only one of the five sits under `model.`, and `depth_decoder.model.` is
//! `model.` NESTED under a tower -- which is the case
//! [`crate::manifest::Observed::logical`] runs its prefix stripping to a
//! fixed point for. Whole names are the only way a table states that; a stem
//! table would have needed a prefix per tower and then a rule for which one.
//!
//! `codec_model.quantizer.*_residual_vector_quantizer.layers.{layer}.codebook.embed_sum`
//! is a codebook and not a weight in the forward pass sense, and the two
//! quantizers -- acoustic and semantic -- are separate rows because they are
//! separate tables with different depths.
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

/// Every tensor a CSM publishes, and what each vocabulary calls it.
///
/// No `gguf` column: [`crate::ingest`] has no arm for this generation, and a
/// column filled by guessing at llama.cpp's spelling would be a claim no file
/// on disk was read for. An unmapped GGUF is refused by name, which is the
/// answer that can be acted on.
pub const VOCAB: Vocab = Vocab(&[
    // ── Inside a decoder layer ───────────────────────────────────────
    Member::same("backbone_model.layers.{layer}.input_layernorm"),
    Member::same("backbone_model.layers.{layer}.mlp.down_proj"),
    Member::same("backbone_model.layers.{layer}.mlp.gate_proj"),
    Member::same("backbone_model.layers.{layer}.mlp.up_proj"),
    Member::same("backbone_model.layers.{layer}.post_attention_layernorm"),
    Member::same("backbone_model.layers.{layer}.self_attn.k_proj"),
    Member::same("backbone_model.layers.{layer}.self_attn.o_proj"),
    Member::same("backbone_model.layers.{layer}.self_attn.q_proj"),
    Member::same("backbone_model.layers.{layer}.self_attn.v_proj"),
    Member::same("codec_model.decoder.layers.{layer}.conv"),
    Member::same("codec_model.decoder_transformer.layers.{layer}.self_attn.q_proj"),
    Member::same("codec_model.decoder_transformer.layers.{layer}.self_attn_layer_scale.scale"),
    Member::same("codec_model.encoder.layers.{layer}.conv"),
    Member::same("codec_model.encoder_transformer.layers.{layer}.self_attn.q_proj"),
    Member::same("codec_model.quantizer.acoustic_residual_vector_quantizer.layers.{layer}.codebook.embed_sum"),
    Member::same("codec_model.quantizer.semantic_residual_vector_quantizer.layers.{layer}.codebook.embed_sum"),
    Member::same("depth_decoder.model.layers.{layer}.input_layernorm"),
    Member::same("depth_decoder.model.layers.{layer}.mlp.down_proj"),
    Member::same("depth_decoder.model.layers.{layer}.mlp.gate_proj"),
    Member::same("depth_decoder.model.layers.{layer}.mlp.up_proj"),
    Member::same("depth_decoder.model.layers.{layer}.post_attention_layernorm"),
    Member::same("depth_decoder.model.layers.{layer}.self_attn.k_proj"),
    Member::same("depth_decoder.model.layers.{layer}.self_attn.o_proj"),
    Member::same("depth_decoder.model.layers.{layer}.self_attn.q_proj"),
    Member::same("depth_decoder.model.layers.{layer}.self_attn.v_proj"),
    // ── Outside it ───────────────────────────────────────────────────
    Member::same("backbone_model.embed_tokens.embed_audio_tokens"),
    Member::same("backbone_model.norm"),
    Member::same("codec_model.downsample.conv"),
    Member::same("codec_model.quantizer.acoustic_residual_vector_quantizer.output_proj"),
    Member::same("codec_model.quantizer.semantic_residual_vector_quantizer.input_proj"),
    Member::same("codec_model.upsample.conv"),
    Member::same("depth_decoder.codebooks_head"),
    Member::same("depth_decoder.model.embed_tokens"),
    Member::same("depth_decoder.model.inputs_embeds_projector"),
    Member::same("depth_decoder.model.norm"),
    Member::same("model.embed_text_tokens"),
    Member::same("lm_head"),
]);
