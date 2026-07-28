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

/// One rewrite or check over a finished plan.
///
/// A pass is a name and a function, so this is a struct rather than a trait: no
/// pass has ever had state of its own, and a trait with one implementor buys
/// dispatch nobody calls for. Keeping it concrete makes [`super::passes::all`] a
/// `static` the compiler lays out once, instead of nine boxes built per compile.
pub struct Pass {
    pub name: &'static str,

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
/// Only passes that fired are recorded, which keeps the report the same size it
/// was under v1 while making the passes that never fire discoverable: run with
/// every pass recorded and the dead ones are the ones missing from the list.
pub fn run_all(plan: &mut LoadPlan) -> Result<Vec<PassStats>> {
    let mut stats = Vec::new();
    for pass in super::passes::all() {
        let before = plan.instrs.len();
        let started = std::time::Instant::now();
        let rewrites = (pass.run)(plan)?;
        if crate::planner_debug_enabled() {
            eprintln!(
                "[pie-loader] pass {:<34} {:>6} ms  {before} -> {} instrs, {rewrites} rewrites",
                pass.name,
                started.elapsed().as_millis(),
                plan.instrs.len()
            );
        }
        if rewrites == 0 {
            continue;
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
