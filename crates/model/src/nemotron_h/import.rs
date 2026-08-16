//! Nemotron-H's tensor names, in every vocabulary that spells them.
//!
//! The only generation here that does not put its decoder under `model.`.
//! Nemotron-H publishes `backbone.layers.{layer}.` and `backbone.norm_f`,
//! with `lm_head` outside as usual -- so a naming table that stored member
//! STEMS and rebuilt `model.layers.{i}.` around them could not hold this
//! family at all without a second knob for the prefix. It is why the rows
//! below are whole names.
//!
//! `mixer` rather than `self_attn` or `mlp`, for the same reason: a layer is
//! a Mamba-2 mixer, an attention, or a mixture, and the release gives all
//! three one module name. `A_log`, `D`, `conv1d` and `dt_bias` are the
//! state-space parameters; `in_proj` / `out_proj` are its projections, and
//! they are NOT `q_proj` / `o_proj` under another name.
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

/// Every tensor a Nemotron-H publishes, and what each vocabulary calls it.
///
/// No `gguf` column: [`crate::ingest`] has no arm for this generation, and a
/// column filled by guessing at llama.cpp's spelling would be a claim no file
/// on disk was read for. An unmapped GGUF is refused by name, which is the
/// answer that can be acted on.
pub const VOCAB: Vocab = Vocab(&[
    // ── Inside a decoder layer ───────────────────────────────────────
    Member::same("backbone.layers.{layer}.mixer.A_log"),
    Member::same("backbone.layers.{layer}.mixer.D"),
    Member::same("backbone.layers.{layer}.mixer.conv1d"),
    Member::same("backbone.layers.{layer}.mixer.down_proj"),
    Member::same("backbone.layers.{layer}.mixer.dt_bias"),
    Member::same("backbone.layers.{layer}.mixer.experts.{expert}.down_proj"),
    Member::same("backbone.layers.{layer}.mixer.experts.{expert}.up_proj"),
    Member::same("backbone.layers.{layer}.mixer.gate_proj"),
    Member::same("backbone.layers.{layer}.mixer.in_proj"),
    Member::same("backbone.layers.{layer}.mixer.k_norm"),
    Member::same("backbone.layers.{layer}.mixer.k_proj"),
    Member::same("backbone.layers.{layer}.mixer.norm"),
    Member::same("backbone.layers.{layer}.mixer.o_proj"),
    Member::same("backbone.layers.{layer}.mixer.out_proj"),
    Member::same("backbone.layers.{layer}.mixer.q_norm"),
    Member::same("backbone.layers.{layer}.mixer.q_proj"),
    Member::same("backbone.layers.{layer}.mixer.up_proj"),
    Member::same("backbone.layers.{layer}.mixer.v_proj"),
    Member::same("backbone.layers.{layer}.norm"),
    // ── Outside it ───────────────────────────────────────────────────
    Member::same("backbone.embeddings"),
    Member::same("backbone.norm_f"),
    Member::same("lm_head"),
]);
