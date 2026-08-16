//! Qwen3.5's tensor names, in every vocabulary that spells them.
//!
//! Qwen3.5 is the widest table here that is not a multi-tower model, and the
//! reason is that it is three architectures at once: a gated-delta linear
//! attention (`linear_attn.*`), a normal attention, and a mixture with both
//! routed experts and a shared one. A layer picks one, so no checkpoint holds
//! every row below -- the table answers for the generation and the row
//! answers for the variant.
//!
//! `linear_attn.in_proj_qkv` beside `in_proj_a`, `in_proj_b` and `in_proj_z`
//! is the pair worth reading twice: releases disagree about how many pieces
//! the input projection ships in, which is a DIVISION and not a spelling, so
//! it is a row each rather than one row with two names.
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

/// Every tensor a Qwen3.5 publishes, and what each vocabulary calls it.
///
/// No `gguf` column: [`crate::ingest`] has no arm for this generation, and a
/// column filled by guessing at llama.cpp's spelling would be a claim no file
/// on disk was read for. An unmapped GGUF is refused by name, which is the
/// answer that can be acted on.
pub const VOCAB: Vocab = Vocab(&[
    // ── Inside a decoder layer ───────────────────────────────────────
    Member::same("model.layers.{layer}.input_layernorm"),
    Member::same("model.layers.{layer}.linear_attn.A_log"),
    Member::same("model.layers.{layer}.linear_attn.conv1d"),
    Member::same("model.layers.{layer}.linear_attn.dt_bias"),
    Member::same("model.layers.{layer}.linear_attn.in_proj_a"),
    Member::same("model.layers.{layer}.linear_attn.in_proj_b"),
    Member::same("model.layers.{layer}.linear_attn.in_proj_qkv"),
    Member::same("model.layers.{layer}.linear_attn.in_proj_z"),
    Member::same("model.layers.{layer}.linear_attn.norm"),
    Member::same("model.layers.{layer}.linear_attn.out_proj"),
    Member::same("model.layers.{layer}.mlp.down_proj"),
    Member::same("model.layers.{layer}.mlp.experts.{expert}.down_proj"),
    Member::same("model.layers.{layer}.mlp.experts.{expert}.gate_proj"),
    Member::same("model.layers.{layer}.mlp.gate"),
    Member::same("model.layers.{layer}.mlp.gate_proj"),
    Member::same("model.layers.{layer}.mlp.shared_expert.gate_proj"),
    Member::same("model.layers.{layer}.mlp.switch_mlp.down_proj"),
    Member::same("model.layers.{layer}.mlp.switch_mlp.gate_proj"),
    Member::same("model.layers.{layer}.mlp.up_proj"),
    Member::same("model.layers.{layer}.post_attention_layernorm"),
    Member::same("model.layers.{layer}.self_attn.k_norm"),
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
