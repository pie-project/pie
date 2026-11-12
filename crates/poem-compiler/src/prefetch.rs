use std::ops::Range;

use poem_ir::{Def, Operands, Trace, ValueId};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Reads {
    pub param: usize,
    pub span: Range<u32>,
    pub reads: u32,
}

impl Reads {
    #[must_use]
    pub const fn unread(&self) -> bool {
        self.reads == 0
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Schedule {
    reads: Vec<Reads>,
}

impl Schedule {
    #[must_use]
    pub fn of(trace: &Trace) -> Schedule {
        let mut span: Vec<Option<Range<u32>>> = vec![None; trace.params.len()];
        let mut reads: Vec<u32> = vec![0; trace.params.len()];
        let mut inputs: Vec<ValueId> = Vec::new();
        for (at, node) in trace.nodes.iter().enumerate() {
            let at = u32::try_from(at).unwrap_or(u32::MAX);
            inputs.clear();
            node.op.inputs(&mut inputs);
            for id in &inputs {
                let Some(param) = weight_of(trace, *id) else {
                    continue;
                };
                reads[param] += 1;
                span[param] = Some(match span[param].clone() {
                    None => at..at + 1,
                    Some(had) => had.start.min(at)..had.end.max(at + 1),
                });
            }
        }
        let mut rows: Vec<Reads> = (0..trace.params.len())
            .map(|param| Reads {
                param,
                span: span[param].clone().unwrap_or(0..0),
                reads: reads[param],
            })
            .collect();
        rows.sort_by_key(|row| (row.unread(), row.span.start, row.param));
        Schedule { reads: rows }
    }

    #[must_use]
    pub fn order(&self) -> Vec<usize> {
        self.reads.iter().map(|row| row.param).collect()
    }
}

fn weight_of(trace: &Trace, id: ValueId) -> Option<usize> {
    match trace.values.get(id.0 as usize).map(|decl| &decl.def) {
        Some(Def::Weight(w)) => Some(*w as usize),
        _ => None,
    }
}
