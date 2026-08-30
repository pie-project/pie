//! The load-time reading of a baked plan: what a shell may refuse before it
//! has fired once.
//!
//! Neutral IR reasoning, and it was written twice. Both shells asked the same
//! question of the same two objects — a `Trace` and a `CompiledModel` — and
//! got there by the same walk; the only thing that differed was which
//! backend's planner the prose named.

use model_compiler::CompiledModel;
use model_ir::{Attention, Def, Operation, Trace};

use crate::fire::MaskSpan;
use crate::store::{Fault, Result};

/// **THE BAKE-TIME HALF OF THE WINDOW ARGUMENT**: no attention schedule may
/// be built over more classes than the node consuming it runs in.
///
/// A schedule is not a row-shaped table that slices — it is a carving. How
/// many requests it batches, where each request's query rows start, how its
/// work items split the kv, and how much of the grant it padded to are all
/// fixed when the backend's `plan_prefill` walks the window it was dispatched
/// in. The consumers then hand it their OWN rebased qo boundaries
/// ([`rebase`]), and a consumer standing in a narrower window hands it a
/// vector that ends before its work items do. Nothing faults: the reads land
/// in whatever follows a `[lanes + 1]` vector in the staging store, and the
/// answer is wrong logits. A handle table narrows the blast radius — a cut is
/// bounds-checked against its buffer — and does not close it, because the
/// vector that follows is inside the same reservation and is therefore a
/// legal read.
///
/// It is a property of the BAKE, not of a fire — region masks are static —
/// so it is asked once at load, where the sentence can name the model text
/// that has to change. What produces it is one plan value shared by arms in
/// different classes: the compiler narrows a prepare node by demand to the
/// union of the classes reading its struct (design build log 7), which is
/// the right answer for a shared value and the wrong SHAPE for two windowed
/// readers. gemma's text is the standing instance — `plan_p` feeds both
/// `attention.prefill` and `attention.masked`, so its region carries the two
/// classes and each arm carries one.
///
/// Equality rather than containment, deliberately. A schedule built over
/// FEWER classes than its reader is the same failure from the other side
/// (the reader's later requests index past the schedule's batch), and a
/// consumer that is not windowed at all does not consume a plan.
///
/// # Errors
///
/// [`Fault::Straddled`], naming the value, the consuming node, and the two
/// class sets.
pub fn no_schedule_straddles_its_readers(trace: &Trace, compiled: &CompiledModel) -> Result<()> {
    // Which region each node stands in, and therefore which classes it runs.
    let mut region_of: Vec<usize> = vec![0; trace.nodes.len()];
    for (at, region) in compiled.template().iter().enumerate() {
        for node in region.nodes.clone() {
            if let Some(slot) = region_of.get_mut(node as usize) {
                *slot = at;
            }
        }
    }
    let mask_of = |node: usize| &compiled.template()[region_of[node]].mask;

    for (at, node) in trace.nodes.iter().enumerate() {
        let Operation::Attention(op) = &node.op else {
            continue;
        };
        // Only the launches, never the builders: a builder DEFINES the
        // schedule and so stands in the window it is carved at by
        // construction.
        let consumed = match op {
            Attention::Decode { plan, .. }
            | Attention::DecodeLse { plan, .. }
            | Attention::Prefill { plan, .. }
            | Attention::PrefillLse { plan, .. }
            | Attention::Masked { plan, .. } => *plan,
            _ => continue,
        };
        let Some(Def::Op(built_by)) = trace.values.get(consumed.0 as usize).map(|v| &v.def) else {
            continue;
        };
        let planned = mask_of(*built_by as usize);
        let reader = mask_of(at);
        if planned != reader {
            return Err(Fault::Straddled {
                value: consumed.0,
                node: at as u32,
                planned: format!("{:?}", planned.iter().collect::<Vec<_>>()),
                consumed: format!("{:?}", reader.iter().collect::<Vec<_>>()),
            });
        }
    }
    Ok(())
}

/// The window's qo boundaries, rebased so the first is 0.
#[must_use]
pub fn rebase(indptr: &[i32], span: MaskSpan) -> Vec<i32> {
    let first = span.lane_offset as usize;
    let last = first + span.lanes as usize;
    let Some(cut) = indptr.get(first..=last) else {
        return vec![0];
    };
    let base = cut[0];
    cut.iter().map(|bound| bound - base).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_window_s_boundaries_start_at_its_own_zero() {
        let indptr = [0, 3, 7, 12, 20];
        let span = MaskSpan {
            row_offset: 7,
            rows: 5,
            lane_offset: 2,
            lanes: 1,
        };
        assert_eq!(rebase(&indptr, span), vec![0, 5]);
    }

    #[test]
    fn the_whole_fire_rebases_to_itself() {
        let indptr = [0, 3, 7, 12, 20];
        let span = MaskSpan {
            row_offset: 0,
            rows: 20,
            lane_offset: 0,
            lanes: 4,
        };
        assert_eq!(rebase(&indptr, span), vec![0, 3, 7, 12, 20]);
    }

    #[test]
    fn a_span_past_the_vector_yields_one_bound_rather_than_a_panic() {
        let indptr = [0, 3];
        let span = MaskSpan {
            row_offset: 0,
            rows: 3,
            lane_offset: 1,
            lanes: 4,
        };
        assert_eq!(rebase(&indptr, span), vec![0]);
    }
}
