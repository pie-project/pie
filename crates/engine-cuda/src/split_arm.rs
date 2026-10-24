//! A gated-delta mixer cuts its packed projection (`gather.split_rows`)
//! before the switch between its one-token step and its chunked walk, so
//! the split runs under no guard while the step's ops run under the arm's,
//! and `attention.ssm_gdn_step` (whose ops share one guard) never forms.
//! This rewrite gives the step's arm a split of its own under the arm's
//! guard and leaves the first one the other arm's guard, so a decode fire
//! runs the fused step and no split.

use poem_ir::{
    Def, Elemwise, Gather, Guard, Node, Operands, Operation, Ssm, Trace, ValueDecl, ValueId,
};

/// The rewrite over every such split.
#[must_use]
pub fn split_into_arms(mut trace: Trace) -> Trace {
    let mut at = 0;
    while at < trace.nodes.len() {
        if let Some(next) = split_one(&mut trace, at) {
            at = next;
        } else {
            at += 1;
        }
    }
    trace
}

/// Where `op` reads `from`, `to` instead; false for an op it does not know.
fn redirect(op: &mut Operation, from: ValueId, to: ValueId) -> bool {
    let swap = |v: &mut ValueId| {
        if *v == from {
            *v = to;
        }
    };
    match op {
        Operation::Ssm(Ssm::CausalConv1d { x, .. }) => swap(x),
        Operation::Ssm(Ssm::GatedDelta { qkv, z, .. }) => {
            swap(qkv);
            swap(z);
        }
        Operation::Elemwise(Elemwise::RmsnormGated { x, gate, .. }) => {
            swap(x);
            swap(gate);
        }
        _ => return false,
    }
    true
}

/// Splits node `at` into its arms when it is an unguarded split whose rows
/// a guarded one-token conv reads; the index to resume from.
fn split_one(trace: &mut Trace, at: usize) -> Option<usize> {
    let Operation::Gather(Gather::SplitRows {
        x,
        width,
        left,
        right,
    }) = trace.nodes[at].op
    else {
        return None;
    };
    if trace.nodes[at].guard != Guard::Always {
        return None;
    }
    let mut ins = Vec::new();
    let readers: Vec<usize> = (at + 1..trace.nodes.len())
        .filter(|&i| {
            ins.clear();
            trace.nodes[i].op.inputs(&mut ins);
            ins.contains(&left) || ins.contains(&right)
        })
        .collect();
    let arm = readers.iter().find_map(|&i| match &trace.nodes[i] {
        Node {
            op: Operation::Ssm(Ssm::CausalConv1d { x, .. }),
            guard,
            ..
        } if *x == left && *guard != Guard::Always => Some(guard.clone()),
        _ => None,
    })?;
    let (mine, rest): (Vec<usize>, Vec<usize>) =
        readers.iter().partition(|&&i| trace.nodes[i].guard == arm);
    let other = rest.first().map(|&i| trace.nodes[i].guard.clone());
    // The values a merge or a seam reads stay the first split's.
    let merged = trace.values.iter().any(|decl| {
        matches!(&decl.def, Def::Merge(arms) if arms.iter().any(|(v, _)| *v == left || *v == right))
    }) || trace
        .seams
        .iter()
        .any(|seam| seam.values.contains(&left) || seam.values.contains(&right));
    if merged
        || rest
            .iter()
            .any(|&i| Some(&trace.nodes[i].guard) != other.as_ref())
    {
        return None;
    }
    let mut ops: Vec<Operation> = mine.iter().map(|&i| trace.nodes[i].op.clone()).collect();
    let (l2, r2) = (
        ValueId(trace.values.len() as u32),
        ValueId(trace.values.len() as u32 + 1),
    );
    for op in &mut ops {
        if !(redirect(op, left, l2) && redirect(op, right, r2)) {
            return None;
        }
    }
    for (&i, op) in mine.iter().zip(ops) {
        trace.nodes[i].op = op;
    }
    let born = at + 1;
    for decl in &mut trace.values {
        if let Def::Op(i) = &mut decl.def
            && *i as usize >= born
        {
            *i += 1;
        }
    }
    for v in [left, right] {
        let ty = trace.values[v.0 as usize].ty.clone();
        trace.values.push(ValueDecl {
            def: Def::Op(born as u32),
            ty,
        });
    }
    let layer = trace.nodes[at].layer;
    trace.nodes.insert(
        born,
        Node {
            op: Operation::Gather(Gather::SplitRows {
                x,
                width,
                left: l2,
                right: r2,
            }),
            guard: arm,
            layer,
        },
    );
    if let Some(other) = other {
        trace.nodes[at].guard = other;
    }
    Some(born + 1)
}
