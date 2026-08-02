//! The pass pipeline.
//!
//! A pass is a named function over a finished plan that says how many rewrites
//! it made. Both halves matter. The name means a new pass is a line in
//! `passes::all()` rather than a line buried in the middle of the compiler; the
//! count means a pass that never fires is *visible* rather than inferred.
//!
//! v1 had neither. Its seven passes were seven statements in the middle of
//! `StorageCompiler::lower`, and its eighth — the `optimizer`, 276 lines — was
//! a separate stage over a separate IR that reported `rewrites: 0` on all
//! fourteen goldens because the frontend could not construct the patterns it
//! matched. Nobody noticed, because the only reader of the report was a debug
//! dump.

use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::plan::LoadPlan;

/// What a pass is allowed to do, and therefore where it may sit in the order.
///
/// The distinction is load-bearing rather than descriptive: a check proves
/// something about the plan the compiler hands back, and that proof only holds
/// if nothing rewrites the plan afterwards. [`run_all`] enforces it, so the
/// guarantee survives someone appending a rewrite to the end of
/// [`super::passes::all`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage {
    /// May change the plan.
    Rewrite,

    /// May only refuse.
    Check,
}

/// One rewrite or check over a finished plan.
///
/// A pass is a name and a function, so this is a struct rather than a trait: no
/// pass has ever had state of its own, and a trait with one implementor buys
/// dispatch nobody calls for. Keeping it concrete makes [`super::passes::all`] a
/// `static` the compiler lays out once, instead of nine boxes built per compile.
pub struct Pass {
    pub name: &'static str,

    /// Whether this pass may rewrite, or may only refuse.
    pub stage: Stage,

    /// Rewrite the plan, returning how many rewrites were made.
    ///
    /// A validator returns `0`: it is the honest answer, and it is why the
    /// count is "rewrites" rather than "did something".
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

/// Run the standard pipeline, in order, recording what each pass did.
///
/// Every pass is recorded, including the ones that rewrote nothing. Dropping
/// the zeroes is what let v1's dead optimizer hide: a pass that never fires
/// then looks exactly like a pass that is not in the pipeline, and telling
/// those two apart is the whole reason the count is kept.
///
/// The [`Stage`] split is enforced here rather than trusted: a rewrite after a
/// check would silently void what the check proved, and a check that reports a
/// rewrite is not a check.
pub fn run_all(plan: &mut LoadPlan) -> Result<Vec<PassStats>> {
    run_passes(plan, super::passes::all())
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
