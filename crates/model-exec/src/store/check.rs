//! The load-time reading of a baked plan: what a shell may refuse before it
//! has fired once.

use model_compiler::CompiledModel;
use model_ir::{Attention, Def, Operation, Trace};

use crate::fire::MaskSpan;
use crate::store::{Fault, Result};

/// No attention schedule may be built over more classes than the node
/// consuming it runs in. A narrower reader hands the schedule a rebased qo
/// boundary vector ([`rebase`]) that ends before its work items do; nothing
/// faults, the read lands past the vector into whatever follows in the
/// staging store, and the answer is wrong logits rather than an error.
///
/// A property of the bake, not of a fire (region masks are static), so it is
/// asked once at load. Equality rather than containment: fewer classes than
/// the reader is the same failure from the other side.
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
        // Only the launches, never the builders: a builder defines the schedule.
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
pub fn rebase(indptr: &[i32], span: MaskSpan) -> Result<Vec<i32>> {
    let first = span.lane_offset as usize;
    let last = first + span.lanes as usize;
    let Some(cut) = indptr.get(first..=last) else {
        return Err(Fault::Ceiling {
            what: "qo boundaries",
            need: last as u64 + 1,
            have: indptr.len() as u64,
        });
    };
    let base = cut[0];
    Ok(cut.iter().map(|bound| bound - base).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_span_past_the_vector_is_refused() {
        let indptr = [0, 3, 5];
        let span = MaskSpan {
            row_offset: 0,
            rows: 5,
            lane_offset: 1,
            lanes: 4,
        };
        assert!(matches!(
            rebase(&indptr, span),
            Err(Fault::Ceiling {
                what: "qo boundaries",
                need: 6,
                have: 3
            })
        ));
    }
}
