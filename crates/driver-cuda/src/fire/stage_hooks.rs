//! The stage-hook dispatch surface a model body threads through itself. Ports
//! the header-only `model/stage_hooks.hpp`: the hooks struct, the sideband that
//! crosses with each call, and the inline [`invoke_stage_hook`] guard chain the
//! generated bodies call. `sideband_arena` is not a field — Rust needs it `&mut`
//! for a slot acquire, so it travels beside the hooks as `attn_score` and
//! `page_mask` already do.

use std::ffi::c_void;

use super::attn_score::{AttentionObservation, AttentionScores};
use super::page_mask::AttentionMaskSink;

/// Where in the attention stage a hook fires. Mirrors `model::StageHookPoint`,
/// discriminants included.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum StageHookPoint {
    /// After the QKV projection, before attention.
    OnAttnProj = 1,
    /// After attention.
    OnAttn = 2,
}

/// What crosses the model → dispatch boundary with the hook: the fire's KV
/// geometry, the layer's captured scores, and a page-mask sink's destination.
/// Carried by the call so a hook cannot read another fire's sideband.
#[derive(Debug, Clone, Copy, Default)]
pub struct StageHookSideband<'a> {
    /// The fire's KV geometry; filled from the hooks' own observation when the
    /// call site leaves it `None`.
    pub observation: Option<&'a AttentionObservation<'a>>,
    /// The layer's captured scores, or `None` when nothing was captured — the
    /// PTIR side then fails loudly instead of reading a stale row.
    pub scores: Option<&'a AttentionScores>,
    /// Where a program's `attn_page_mask` sink writes. Owned by the model body,
    /// only filled by the dispatch — hence a raw pointer, as in the C++.
    pub mask_sink: *mut AttentionMaskSink,
}

/// The dispatch side of a hook — the C++ `context` + `execute` pair as one trait
/// object. `&self` because a fire invokes hooks many times; a recording impl
/// uses interior mutability.
pub trait StageHookExecute {
    /// One hook invocation. Argument order is the C++ fn pointer's.
    #[allow(clippy::too_many_arguments)]
    fn execute(
        &self,
        point: StageHookPoint,
        query_data: *const c_void,
        query_rows: u32,
        query_columns: u32,
        layer: u32,
        stream: *mut c_void,
        query_is_f32: bool,
        sideband: &StageHookSideband<'_>,
    );
}

/// The fire-level replay seam: `prepare_replay` and `verify_replay_capture`.
pub trait PrepareReplay {
    /// Hoist every attention-phase prepare to fire level; return a fingerprint
    /// of what a captured body would bake, or 0 when the fire must take the
    /// eager body.
    fn prepare_replay(&self, stream: *mut c_void) -> u64;
    /// Assert the body consumed every prepared invocation.
    fn verify_replay_capture(&self);
}

/// The fire's hook set. Mirrors `model::StageHooks`.
#[derive(Default, Clone, Copy)]
pub struct StageHooks<'a> {
    /// The fire's PTIR programs read `AttnScore` at `OnAttn`.
    pub wants_attn_score: bool,
    /// Query rows observed at the tail of a prefill chunk when scores are
    /// wanted; decode's window is 1 by construction.
    pub attn_score_window: u32,
    /// The fire's PTIR programs write `attn_page_mask` at `OnAttnProj`.
    pub wants_page_mask: bool,
    /// Leading request rows belonging to no attention-stage program — the
    /// hook-free prefix. `0` = no row is provably hook-free.
    pub hook_free_prefix_rows: u32,
    /// The hook rows' own truncation; `u32::MAX` = full. The body invokes hook
    /// stages only at layers below this.
    pub hook_rows_k: u32,
    /// The fire's KV geometry, set by the body-invocation choke point.
    pub observation: Option<&'a AttentionObservation<'a>>,
    /// The dispatch, or `None` for "no program attached".
    pub execute: Option<&'a dyn StageHookExecute>,
    /// The replay seam, or `None` when the frame did not wire it.
    pub prepare_replay: Option<&'a dyn PrepareReplay>,
}

impl std::fmt::Debug for StageHooks<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StageHooks")
            .field("wants_attn_score", &self.wants_attn_score)
            .field("attn_score_window", &self.attn_score_window)
            .field("wants_page_mask", &self.wants_page_mask)
            .field("hook_free_prefix_rows", &self.hook_free_prefix_rows)
            .field("hook_rows_k", &self.hook_rows_k)
            .field("execute", &self.execute.is_some())
            .finish_non_exhaustive()
    }
}

/// The C++ default: everything off, `hook_rows_k` at full — a zero-default
/// would silently disable every hook.
#[must_use]
pub fn default_hooks<'a>() -> StageHooks<'a> {
    StageHooks { hook_rows_k: u32::MAX, ..StageHooks::default() }
}

impl<'a> StageHooks<'a> {
    /// The slice the score captures read — what the C++ passes as `StageHooks*`
    /// and `attn_score.cu` dereferences three fields of.
    #[must_use]
    pub const fn score_view(&self) -> super::attn_score::ScoreHookView<'a> {
        super::attn_score::ScoreHookView {
            wants_attn_score: self.wants_attn_score,
            observation: self.observation,
        }
    }
}

/// Invoke one stage hook, or nothing. Null hooks or null execute cost one
/// branch; a layer at or past `hook_rows_k` is skipped (its rows are frozen and
/// an invocation would observe garbage); a sideband with no observation is
/// filled from the hooks' own.
#[allow(clippy::too_many_arguments)]
pub fn invoke_stage_hook<'a>(
    hooks: Option<&StageHooks<'a>>,
    point: StageHookPoint,
    query_data: *const c_void,
    query_rows: u32,
    query_columns: u32,
    layer: u32,
    stream: *mut c_void,
    query_is_f32: bool,
    mut sideband: StageHookSideband<'a>,
) {
    let Some(hooks) = hooks else { return };
    let Some(execute) = hooks.execute else { return };
    if layer >= hooks.hook_rows_k {
        return;
    }
    if sideband.observation.is_none() {
        sideband.observation = hooks.observation;
    }
    execute.execute(
        point,
        query_data,
        query_rows,
        query_columns,
        layer,
        stream,
        query_is_f32,
        &sideband,
    );
}
