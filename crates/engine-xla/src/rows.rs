//! Values the model text rows by `Tokens` on a readout-rowed path.
//!
//! A few plan builders state their outputs' rows as `Tokens` (or
//! `TokensTimes(k)`) whatever their inputs' rows are (poem's MoE
//! routers and routed matmuls), so an MTP head that runs over `Readouts`
//! rows feeds token-rowed routing tables, expert rows and sums into its
//! readout-rowed residual. A GPU engine never notices: its buffers are sized
//! at the axis ceiling and every kernel counts rows off its leading operand.
//! Here a handle's rows are its axis' rows in this fire, so the shapes
//! disagree. Such a value is found at load (an output rowed by `Tokens` of a
//! node whose row-carrying inputs are all readout-rowed, in node order, so a
//! chain carries through) and its handle is cut to the fire's readout rows
//! (`Run::cut`): `k` of them for `TokensTimes(k)`.

use poem_ir::{Def, Dim, Operands, Trace, Ty, ValueId};

fn lead(trace: &Trace, v: ValueId) -> Option<Dim> {
    match &trace.values.get(v.0 as usize)?.ty {
        Ty::Tensor { shape, .. } => shape.first().copied(),
        Ty::Struct(_) => None,
    }
}

/// Per value: whether it is token-declared and readout-rowed.
#[must_use]
pub fn readout_rowed(trace: &Trace) -> Vec<bool> {
    let mut derived = vec![false; trace.values.len()];
    let mut inputs = Vec::new();
    let mut outputs = Vec::new();
    for node in &trace.nodes {
        inputs.clear();
        outputs.clear();
        node.op.inputs(&mut inputs);
        node.op.outputs(&mut outputs);
        let mut readouts = false;
        let mut tokens = false;
        for &v in &inputs {
            if matches!(trace.values[v.0 as usize].def, Def::Weight(_)) {
                continue;
            }
            match lead(trace, v) {
                Some(Dim::Readouts) => readouts = true,
                Some(Dim::Tokens | Dim::TokensTimes(_)) if derived[v.0 as usize] => {
                    readouts = true;
                }
                Some(Dim::Const(_)) | None => {}
                Some(_) => tokens = true,
            }
        }
        if !readouts || tokens {
            continue;
        }
        for &v in &outputs {
            if !inputs.contains(&v)
                && matches!(lead(trace, v), Some(Dim::Tokens | Dim::TokensTimes(_)))
            {
                derived[v.0 as usize] = true;
            }
        }
    }
    derived
}

/// A value rowed by `Readouts` that the arena seats, whose handle tells a
/// fire's readout rows.
#[must_use]
pub fn readout_probe(trace: &Trace) -> Option<ValueId> {
    trace
        .values
        .iter()
        .enumerate()
        .find(|(_, decl)| {
            matches!(decl.def, Def::Op(_))
                && matches!(&decl.ty, Ty::Tensor { shape, .. } if shape.first() == Some(&Dim::Readouts))
        })
        .map(|(at, _)| ValueId(at as u32))
}

/// The values a readout program reads to land readout output `name` at any
/// rows of a fire: the token-rowed values the output's backward slice over
/// readout-rowed nodes reads (the final hidden rows, for a plain model's
/// logits), with the slice's nodes in order; or why it cannot, when the
/// slice writes a cache (an MTP head's draft, whose attention appends kv).
pub fn readout_slice(trace: &Trace, name: &str) -> Result<(Vec<ValueId>, Vec<usize>), String> {
    let out = (trace.outputs.iter())
        .find(|o| o.name == name)
        .ok_or_else(|| format!("the trace states no output `{name}`"))?;
    let rowed = |v: ValueId| matches!(lead(trace, v), Some(Dim::Readouts));
    let mut wanted: Vec<ValueId> = vec![out.value];
    let mut slice: Vec<usize> = Vec::new();
    let mut entries: Vec<ValueId> = Vec::new();
    for (at, node) in trace.nodes.iter().enumerate().rev() {
        let mut outs = Vec::new();
        node.op.outputs(&mut outs);
        if !outs.iter().any(|v| wanted.contains(v)) {
            continue;
        }
        if matches!(node.op, poem_ir::Operation::Attn(_)) {
            return Err(format!(
                "output `{name}` is computed by attention over readout rows (node {at}), which \
                 writes the kv cache"
            ));
        }
        slice.push(at);
        let mut ins = Vec::new();
        node.op.inputs(&mut ins);
        for v in ins {
            let def = &trace.values[v.0 as usize].def;
            if rowed(v) {
                if !wanted.contains(&v) {
                    wanted.push(v);
                }
            } else if matches!(def, Def::Op(_) | Def::Merge(_))
                && matches!(lead(trace, v), Some(Dim::Tokens | Dim::TokensTimes(_)))
                && !entries.contains(&v)
            {
                entries.push(v);
            }
        }
    }
    slice.reverse();
    Ok((entries, slice))
}
