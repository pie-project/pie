//! What every backend reads of a trace before it plans: the request classes
//! its guards cut it into, which values an expert region holds, where it
//! states collectives, the row axes it states and each node's layer. Nothing
//! here depends on how the device runs the trace; the temporal plan
//! ([`crate::temporal`]) and the spatial plan (`spatial`) both start
//! from it.

use poem_ir::{ClassTable, Def, Operands, RowAxis, Trace, Ty, resolve_classes};

use crate::{Error, MAX_CLASSES, MAX_FACTS};

#[derive(Debug, Clone, PartialEq)]
pub struct Analysis {
    /// The request classes the trace's guards cut it into, and each node's.
    pub classes: ClassTable,
    /// [`crate::expert::held`]: the values an expert region makes, by id.
    pub expert: Vec<bool>,
    /// The nodes that are collectives ([`crate::collectives`]).
    pub collectives: Vec<u32>,
    /// The row axes the trace's values state.
    pub axes: Vec<RowAxis>,
    /// Each node's decoder layer, where the model states one.
    pub layers: Vec<Option<u32>>,
}

/// The analysis of `trace`, or why no backend can plan it: too many guard
/// facts, guards that leave a request uncovered, or a struct read across
/// classes.
pub fn analyze(trace: &Trace) -> Result<Analysis, Error> {
    let facts = poem_ir::fact_width(trace);
    if facts > MAX_FACTS {
        return Err(Error::TooManyFacts { facts });
    }
    let classes = resolve_classes(trace).map_err(Error::Classes)?;
    if classes.classes.len() > MAX_CLASSES {
        return Err(Error::TooManyClasses {
            classes: classes.classes.len(),
        });
    }
    if classes.node_mask.len() != trace.nodes.len() {
        return Err(Error::MaskLength {
            masks: classes.node_mask.len(),
            nodes: trace.nodes.len(),
        });
    }
    struct_readers_share_one_window(trace, &classes)?;
    Ok(Analysis {
        expert: crate::expert::held(trace),
        collectives: crate::collectives(trace),
        axes: crate::unit::axes_stated(trace),
        layers: trace.nodes.iter().map(|node| node.layer).collect(),
        classes,
    })
}

fn struct_readers_share_one_window(trace: &Trace, classes: &ClassTable) -> Result<(), Error> {
    let structs: Vec<bool> = trace
        .values
        .iter()
        .map(|value| matches!(value.ty, Ty::Struct(_)))
        .collect();

    let mut inputs = Vec::new();
    for (at, node) in trace.nodes.iter().enumerate() {
        inputs.clear();
        node.op.inputs(&mut inputs);
        for &read in &inputs {
            if !structs.get(read.0 as usize).copied().unwrap_or(false) {
                continue;
            }
            let Some(Def::Op(built_by)) = trace.values.get(read.0 as usize).map(|v| &v.def) else {
                continue;
            };
            let planned = &classes.node_mask[*built_by as usize];
            let reader = &classes.node_mask[at];
            if planned != reader {
                return Err(Error::Straddled {
                    value: read,
                    node: at as u32,
                    planned: planned.iter().collect(),
                    consumed: reader.iter().collect(),
                });
            }
        }
    }
    Ok(())
}
