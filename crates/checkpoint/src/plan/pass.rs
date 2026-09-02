//! The pass pipeline: a pass is a named function over a finished plan that
//! reports how many rewrites it made, so a pass that never fires is visible
//! rather than inferred.

use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::plan::LoadPlan;

/// What a pass is allowed to do, and therefore where it may sit in the order.
/// A check's proof only holds if nothing rewrites the plan afterwards;
/// [`run_all`] enforces the ordering.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage {
    /// May change the plan.
    Rewrite,

    /// May only refuse.
    Check,
}

/// One rewrite or check over a finished plan. A struct rather than a trait,
/// since no pass has state of its own, and it lets [`super::passes::all`] be
/// a `static` the compiler lays out once.
#[derive(Clone, Copy)]
pub struct Pass {
    pub name: &'static str,

    /// Whether this pass may rewrite, or may only refuse.
    pub stage: Stage,

    /// Whether this pass exists for the arena (the allocation a plan's
    /// persistent buffers and staging region are laid out in). True of
    /// exactly the two passes that rewrite buffer-relative writes into
    /// arena-absolute ones and hoist `Allocate`s to the schedule head — both
    /// wrong for a load with no arena, so [`crate::plan::compile_streaming`]
    /// runs the pipeline without them.
    pub for_arena: bool,

    /// Rewrite the plan, returning how many rewrites were made. A validator
    /// returns `0`.
    pub run: fn(&mut LoadPlan) -> Result<usize>,
}

/// What one pass did, kept on the plan.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PassStats {
    pub pass: String,
    pub instrs_before: usize,
    pub instrs_after: usize,
    pub rewrites: usize,
}

/// Run the standard pipeline, in order, recording what each pass did —
/// including passes that rewrote nothing, so a dead pass is distinguishable
/// from one not in the pipeline. The [`Stage`] split is enforced here rather
/// than trusted.
pub fn run_all(plan: &mut LoadPlan) -> Result<Vec<PassStats>> {
    run_passes(plan, super::passes::all())
}

/// The same pipeline with the arena's own passes left out — see
/// [`Pass::for_arena`] and [`crate::plan::compile_streaming`]. Filtered from
/// [`super::passes::all`] rather than written out a second time.
pub fn run_arenaless(plan: &mut LoadPlan) -> Result<Vec<PassStats>> {
    let pipeline: Vec<Pass> = super::passes::all()
        .iter()
        .copied()
        .filter(|pass| !pass.for_arena)
        .collect();
    run_passes(plan, &pipeline)
}

/// Run a pipeline. Split out from [`run_all`] so the ordering rule can be
/// tested against a pipeline that breaks it; the real one never does.
pub(super) fn run_passes(plan: &mut LoadPlan, passes: &[Pass]) -> Result<Vec<PassStats>> {
    let mut stats = Vec::new();
    let mut checking = false;
    for pass in passes {
        match pass.stage {
            Stage::Check => checking = true,
            Stage::Rewrite if checking => {
                return Err(crate::error::Error::Internal(format!(
                    "pass '{}' rewrites the plan after a validator has already \
                     checked it; every rewrite must come before every check",
                    pass.name
                )));
            }
            Stage::Rewrite => {}
        }
        let before = plan.instrs.len();
        let rewrites = (pass.run)(plan)?;
        if pass.stage == Stage::Check && rewrites != 0 {
            return Err(crate::error::Error::Internal(format!(
                "pass '{}' is a validator but reports {rewrites} rewrites",
                pass.name
            )));
        }
        stats.push(PassStats {
            pass: pass.name.to_string(),
            instrs_before: before,
            instrs_after: plan.instrs.len(),
            rewrites,
        });
    }
    Ok(stats)
}
