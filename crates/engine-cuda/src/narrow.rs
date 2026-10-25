//! The `kernel.ssm.gdn_out_bf16` rewrite: a chunked gated delta walk whose only
//! reader is a gated norm (the gate not its own output, no output holding it)
//! lands that output in bf16. Every other walk keeps f32.

use poem_ir::{Dtype, Elemwise, Operands, Operation, Ssm, Trace, Ty};

/// The rewrite, or the trace as it was when `on` is false.
#[must_use]
pub fn gdn_out_bf16(mut trace: Trace, on: bool) -> Trace {
    if !on {
        return trace;
    }
    let mut readers = vec![0u32; trace.values.len()];
    let mut normed = vec![false; trace.values.len()];
    let mut ins = Vec::new();
    for node in &trace.nodes {
        ins.clear();
        node.op.inputs(&mut ins);
        for v in &ins {
            readers[v.0 as usize] += 1;
        }
        if let Operation::Elemwise(Elemwise::RmsnormGated { x, gate, .. }) = &node.op
            && x != gate
        {
            normed[x.0 as usize] = true;
        }
    }
    let mut held = vec![false; trace.values.len()];
    for output in &trace.outputs {
        for v in output.values() {
            held[v.0 as usize] = true;
        }
    }
    let narrow: Vec<usize> = trace
        .nodes
        .iter()
        .filter_map(|node| match &node.op {
            Operation::Ssm(Ssm::GatedDeltaChunked { y, .. }) => Some(y.0 as usize),
            _ => None,
        })
        .filter(|&y| readers[y] == 1 && normed[y] && !held[y])
        .collect();
    for y in narrow {
        if let Ty::Tensor { dtype, .. } = &mut trace.values[y].ty
            && *dtype == Dtype::F32
        {
            *dtype = Dtype::Bf16;
        }
    }
    trace
}
