//! Kimi K3's tensor names, in every vocabulary that spells them.
//!
//! K2's latent attention plus a gated-delta path -- `self_attn.A_log`,
//! `f_a_proj`, `f_b_proj`, `g_proj`, `b_proj` and `o_norm` are K3's and
//! appear in no other generation here. The mixture moved too:
//! `block_sparse_moe.gate` and `block_sparse_moe.shared_expert.gate_proj`
//! where K2 says `mlp.gate` and `mlp.shared_experts.gate_proj`.
//!
//! `self_attention_res_norm` and `self_attention_res_proj` sit OUTSIDE
//! `self_attn.`, which is a real distinction and not a typo: they are the
//! residual path around the block, not part of the attention module.
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

/// Every tensor a Kimi K3 publishes, and what each vocabulary calls it.
///
/// No `gguf` column: [`crate::ingest`] has no arm for this generation, and a
/// column filled by guessing at llama.cpp's spelling would be a claim no file
/// on disk was read for. An unmapped GGUF is refused by name, which is the
/// answer that can be acted on.
pub const VOCAB: Vocab = Vocab(&[
    // ── Inside a decoder layer ───────────────────────────────────────
    Member::same("model.layers.{layer}.block_sparse_moe.gate"),
    Member::same("model.layers.{layer}.block_sparse_moe.shared_expert.gate_proj"),
    Member::same("model.layers.{layer}.input_layernorm"),
    Member::same("model.layers.{layer}.mlp.down_proj"),
    Member::same("model.layers.{layer}.mlp.gate_proj"),
    Member::same("model.layers.{layer}.post_attention_layernorm"),
    Member::same("model.layers.{layer}.self_attention_res_norm"),
    Member::same("model.layers.{layer}.self_attention_res_proj"),
    Member::same("model.layers.{layer}.self_attn.A_log"),
    Member::same("model.layers.{layer}.self_attn.b_proj"),
    Member::same("model.layers.{layer}.self_attn.f_a_proj"),
    Member::same("model.layers.{layer}.self_attn.f_b_proj"),
    Member::same("model.layers.{layer}.self_attn.g_proj"),
    Member::same("model.layers.{layer}.self_attn.k_proj"),
    Member::same("model.layers.{layer}.self_attn.kv_a_layernorm"),
    Member::same("model.layers.{layer}.self_attn.kv_a_proj_with_mqa"),
    Member::same("model.layers.{layer}.self_attn.kv_b_proj"),
    Member::same("model.layers.{layer}.self_attn.o_norm"),
    Member::same("model.layers.{layer}.self_attn.o_proj"),
    Member::same("model.layers.{layer}.self_attn.q_a_layernorm"),
    Member::same("model.layers.{layer}.self_attn.q_a_proj"),
    Member::same("model.layers.{layer}.self_attn.q_b_proj"),
    Member::same("model.layers.{layer}.self_attn.q_proj"),
    Member::same("model.layers.{layer}.self_attn.v_proj"),
    // ── Outside it ───────────────────────────────────────────────────
    Member::same("model.embed_tokens"),
    Member::same("lm_head"),
    Member::same("model.norm"),
]);
