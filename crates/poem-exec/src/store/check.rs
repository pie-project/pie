use poem_compiler::CompiledModel;
use poem_ir::{Attn, Def, Operation, Trace};

use crate::fire::MaskSpan;
use crate::store::{Fault, Result};

pub fn no_schedule_straddles_its_readers(trace: &Trace, compiled: &CompiledModel) -> Result<()> {
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
        let op = match &node.op {
            Operation::Attn(op) => op,
            Operation::Fused(poem_ir::Fused::LseSink { attn, .. }) => &**attn,
            _ => continue,
        };
        let consumed = match op {
            Attn::Decode {
                plan,
                selection: None,
                bias: None,
                lse: None,
                ..
            }
            | Attn::Decode {
                plan,
                selection: None,
                bias: None,
                lse: Some(_),
                ..
            }
            | Attn::Prefill {
                plan,
                mask: None,
                selection: None,
                bias: None,
                lse: None,
                ..
            }
            | Attn::Prefill {
                plan,
                mask: None,
                selection: None,
                bias: None,
                lse: Some(_),
                ..
            }
            | Attn::Decode {
                plan,
                selection: None,
                bias: Some(_),
                lse: None,
                ..
            }
            | Attn::Prefill {
                plan,
                mask: None,
                selection: None,
                bias: Some(_),
                lse: None,
                ..
            }
            | Attn::Prefill {
                plan,
                mask: Some(_),
                selection: None,
                bias: None,
                lse: None,
                ..
            }
            | Attn::Prefill {
                plan,
                mask: Some(_),
                selection: None,
                bias: None,
                lse: Some(_),
                ..
            }
            | Attn::Decode {
                plan,
                selection: Some(_),
                bias: None,
                lse: None,
                ..
            }
            | Attn::Prefill {
                plan,
                mask: None,
                selection: Some(_),
                bias: None,
                lse: None,
                ..
            } => *plan,
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
