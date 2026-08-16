//! gpt-oss's tensor names, in every vocabulary that spells them.
//!
//! `self_attn.sinks` is the row that is not a weight in the usual sense: a
//! per-head learned attention sink, a vector, published under a name with no
//! `.weight` on it at all. It is why [`crate::shared::vocabulary`] matches a
//! name whole when its last segment is not a suffix.
//!
//! The expert biases appear twice on purpose. `gate_up_proj_bias` and
//! `down_proj_bias` are OpenAI's spelling, and `experts.gate_proj.bias` /
//! `experts.up_proj.bias` / `experts.down_proj.bias` are MLX's, which splits
//! the fused gate/up bias in two. That is a DIVISION, not a spelling, which
//! is why the rows carry it as alternatives rather than the table carrying it
//! as one name.
//!
//! There is no GGUF column, and [`crate::ingest`] carries the measurement of
//! why: every gpt-oss GGUF published carries the experts as a self-contained
//! 17-byte MXFP4 block that the placement algebra cannot split into the
//! `_blocks` / `_scales` pair this generation's contract declares, and
//! requantizes attention and the embeddings to Q8_0 besides.
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

/// Every tensor a gpt-oss publishes, and what each vocabulary calls it.
///
/// No `gguf` column: [`crate::ingest`] has no arm for this generation, and a
/// column filled by guessing at llama.cpp's spelling would be a claim no file
/// on disk was read for. An unmapped GGUF is refused by name, which is the
/// answer that can be acted on.
pub const VOCAB: Vocab = Vocab(&[
    // ── Inside a decoder layer ───────────────────────────────────────
    Member::same("model.layers.{layer}.input_layernorm"),
    Member::same("model.layers.{layer}.mlp.experts.down_proj.bias"),
    Member::same("model.layers.{layer}.mlp.experts.down_proj_bias"),
    Member::same("model.layers.{layer}.mlp.experts.gate_proj.bias"),
    Member::same("model.layers.{layer}.mlp.experts.gate_up_proj_bias"),
    Member::same("model.layers.{layer}.mlp.experts.up_proj.bias"),
    Member::same("model.layers.{layer}.mlp.router"),
    Member::same("model.layers.{layer}.post_attention_layernorm"),
    Member::same("model.layers.{layer}.self_attn.k_proj"),
    Member::same("model.layers.{layer}.self_attn.o_proj"),
    Member::same("model.layers.{layer}.self_attn.q_proj"),
    Member::same("model.layers.{layer}.self_attn.sinks"),
    Member::same("model.layers.{layer}.self_attn.v_proj"),
    // ── Outside it ───────────────────────────────────────────────────
    Member::same("model.embed_tokens"),
    Member::same("lm_head"),
    Member::same("model.norm"),
]);
