//! Horizontal packing: products of one activation by weights stored alike
//! are joined into one product by the weights joined along their rows, its
//! result split back into theirs. A package states each weight it means (q,
//! k and v; a gate and an up); this pass is where they become the packed
//! weight a kernel reads at once, and [`joins`] is what an import reads them
//! into, so the trace and the artifact agree.
//!
//! What joins: products with no input-scale operand and no logits head, each by a checkpoint weight of one
//! segment cut along its rows that nothing else reads, under one guard and
//! one layer, at one precision, the weights' planes alike, each stating the
//! heads it holds or none of them.
//! Two such products a gated unit reads as its gate and its up become one
//! packed unit; three whose last two are as wide (q, k, v) a split into
//! three; two others a split into two. Anything else runs as written. A
//! joined weight's static input scale (one product quantizes its
//! activations under one) is its first leg's: the legs read one activation,
//! so a calibration states them alike, and an import whose legs' differ is
//! refused.

use std::collections::{BTreeMap, BTreeSet};

use poem::Join;
use poem_ir::{
    Def, Dim, Elemwise, Gather, Linear, Node, Operands, Operation, Param, ParamLayout,
    ParamPlacement, ParamSource, Shard, Trace, Ty, ValueDecl, ValueId,
};

/// The joins [`pack`] makes of `trace`, in the order it makes them.
#[must_use]
pub fn joins(trace: &Trace) -> Vec<Join> {
    let name = |p: &u32| trace.params[*p as usize].name.clone();
    groups(trace)
        .iter()
        .flat_map(|g| {
            let weight = Join {
                name: g.name.clone(),
                legs: g.params.iter().map(name).collect(),
            };
            let bias = g.biases.as_ref().map(|b| Join {
                name: g.bias_name.clone(),
                legs: b.iter().map(|b| name(&b.param)).collect(),
            });
            std::iter::once(weight).chain(bias)
        })
        .collect()
}

/// `trace` with each join made.
#[must_use]
pub fn pack(trace: Trace) -> Trace {
    let groups = groups(&trace);
    if groups.is_empty() {
        return trace;
    }
    apply(trace, &groups)
}

/// How a join's result is split back.
#[derive(Debug)]
enum Form {
    /// Read by one gated unit as its gate (the first) and its up.
    Glu { at: usize },
    /// Three results, the last two as wide.
    Qkv { q: u32, kv: u32 },
    /// Two results.
    Rows { left: u32 },
}

/// A product's bias, added by the op right after it: that op, the bias's
/// param and value, and the op's result, which the leg then stands for.
#[derive(Debug, Clone, Copy)]
struct Biased {
    at: usize,
    param: u32,
    bias: ValueId,
    out: ValueId,
}

#[derive(Debug)]
struct Group {
    name: String,
    /// Each leg's bias, where they are biased (all or none).
    biases: Option<Vec<Biased>>,
    bias_name: String,
    /// The products, ascending.
    nodes: Vec<usize>,
    /// Each product's weight's param, and its planes' (suffix, param).
    params: Vec<u32>,
    planes: Vec<Vec<(String, u32)>>,
    widths: Vec<u64>,
    form: Form,
}

fn reads(trace: &Trace) -> Vec<u32> {
    let mut count = vec![0u32; trace.values.len()];
    let mut ins = Vec::new();
    for node in &trace.nodes {
        ins.clear();
        node.op.inputs(&mut ins);
        for v in &ins {
            count[v.0 as usize] += 1;
        }
    }
    for seam in &trace.seams {
        for v in &seam.values {
            count[v.0 as usize] += 1;
        }
    }
    count
}

/// A param a join may take as a leg: a checkpoint weight on the device in
/// its natural order, one segment cut along its rows.
fn leg(param: &Param) -> bool {
    param.source == ParamSource::Checkpoint
        && param.placement == ParamPlacement::Device
        && param.layout == ParamLayout::Natural
        && param.experts.is_none()
        && matches!(&param.shard, Shard::Cut { axis: 0, segments, .. } if segments.len() == 1)
}

/// Whether `param`'s cut states the heads its segment holds: legs that do
/// join only with legs that do (a joined cut either holds each segment's
/// heads whole or cuts every segment evenly).
fn headed(param: &Param) -> bool {
    matches!(&param.shard, Shard::Cut { heads, .. } if !heads.is_empty())
}

/// The planes stored beside the param named `name` (`name.scales`, ...),
/// by suffix.
fn planes_of(trace: &Trace, name: &str) -> Vec<(String, u32)> {
    trace
        .params
        .iter()
        .enumerate()
        .filter_map(|(at, p)| {
            let suffix = p.name.strip_prefix(name)?.strip_prefix('.')?;
            Some((suffix.to_string(), at as u32))
        })
        .collect()
}

fn weight_param(trace: &Trace, v: ValueId) -> Option<u32> {
    match trace.values[v.0 as usize].def {
        Def::Weight(p) => Some(p),
        _ => None,
    }
}

/// The bias the op reading `y` (and nothing else does) adds to it, where
/// it is a weight a join may take as a leg.
fn biased(trace: &Trace, read: &[u32], y: ValueId) -> Option<Biased> {
    if read[y.0 as usize] != 1 {
        return None;
    }
    let mut ins = Vec::new();
    trace.nodes.iter().enumerate().find_map(|(at, node)| {
        let Operation::Elemwise(Elemwise::AddBias { bias, out, out_out }) = &node.op else {
            return None;
        };
        if *out != y {
            return None;
        }
        ins.clear();
        node.op.inputs(&mut ins);
        let param = weight_param(trace, *bias)?;
        (read[bias.0 as usize] == 1 && leg(&trace.params[param as usize])).then_some(Biased {
            at,
            param,
            bias: *bias,
            out: *out_out,
        })
    })
}

fn groups(trace: &Trace) -> Vec<Group> {
    let read = reads(trace);
    // Each candidate product, keyed by what must agree for a join.
    let mut keyed: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    for (at, node) in trace.nodes.iter().enumerate() {
        let Operation::Linear(Linear::Matmul {
            act,
            w,
            input_scales: None,
            scales,
            quant,
            logits: false,
            y,
        }) = &node.op
        else {
            continue;
        };
        let Some(p) = weight_param(trace, *w) else {
            continue;
        };
        let param = &trace.params[p as usize];
        if read[w.0 as usize] != 1 || !leg(param) || param.shape.len() < 2 {
            continue;
        }
        if let Some(s) = scales {
            let Some(sp) = weight_param(trace, *s) else {
                continue;
            };
            if read[s.0 as usize] != 1 || !leg(&trace.params[sp as usize]) {
                continue;
            }
        }
        let planes: Vec<String> = planes_of(trace, &param.name)
            .into_iter()
            .map(|(suffix, _)| suffix)
            .collect();

        let bias = biased(trace, &read, *y).map(|b| {
            (
                headed(&trace.params[b.param as usize]),
                trace.params[b.param as usize].dtype,
            )
        });
        let key = format!(
            "{}|{quant:?}|{:?}|{:?}|{}|{planes:?}|{:?}|{:?}|{:?}|{}|{bias:?}",
            act.0,
            param.dtype,
            &param.shape[1..],
            scales.is_some(),
            node.guard,
            node.layer,
            rows_of(&trace.values[y.0 as usize].ty),
            headed(param),
        );
        keyed.entry(key).or_default().push(at);
    }

    let mut out = Vec::new();
    for nodes in keyed.into_values() {
        if nodes.len() < 2 {
            continue;
        }
        let biases: Option<Vec<Biased>> = nodes
            .iter()
            .map(|&at| match &trace.nodes[at].op {
                Operation::Linear(Linear::Matmul { y, .. }) => biased(trace, &read, *y),
                _ => unreachable!("grouped nodes are products"),
            })
            .collect();
        // What each leg stands for: its product, or the bias added to it.
        let ys: Vec<ValueId> = nodes
            .iter()
            .enumerate()
            .map(|(i, &at)| match (&biases, &trace.nodes[at].op) {
                (Some(b), _) => b[i].out,
                (None, Operation::Linear(Linear::Matmul { y, .. })) => *y,
                _ => unreachable!("grouped nodes are products"),
            })
            .collect();
        let params: Vec<u32> = nodes
            .iter()
            .map(|&at| match &trace.nodes[at].op {
                Operation::Linear(Linear::Matmul { w, .. }) => {
                    weight_param(trace, *w).expect("a leg's weight is a param")
                }
                _ => unreachable!("grouped nodes are products"),
            })
            .collect();
        let widths: Vec<u64> = params
            .iter()
            .map(|p| trace.params[*p as usize].shape[0])
            .collect();
        let form = match ys.as_slice() {
            [gate, up] => {
                let glu = trace.nodes.iter().position(|n| {
                    matches!(&n.op, Operation::Linear(Linear::Glu { packed: None, gate: Some(g), up: Some(u), .. })
                        if g == gate && u == up)
                });
                match glu {
                    Some(at) if read[gate.0 as usize] == 1 && read[up.0 as usize] == 1 => {
                        Form::Glu { at }
                    }
                    _ => Form::Rows {
                        left: u32::try_from(widths[0]).unwrap_or(u32::MAX),
                    },
                }
            }
            [_, _, _] if widths[1] == widths[2] => Form::Qkv {
                q: u32::try_from(widths[0]).unwrap_or(u32::MAX),
                kv: u32::try_from(widths[1]).unwrap_or(u32::MAX),
            },
            _ => continue,
        };
        let names: Vec<&str> = params
            .iter()
            .map(|p| trace.params[*p as usize].name.as_str())
            .collect();
        let planes = params
            .iter()
            .map(|p| planes_of(trace, &trace.params[*p as usize].name))
            .collect();
        let name = Join::name_of(&names);
        // A bias named for its weight (`q_bias` beside `q`) is named so for
        // the joined weight (`qkv_bias`).
        let bias_name = biases.as_ref().map_or_else(String::new, |b| {
            let bias_names: Vec<&str> = b
                .iter()
                .map(|b| trace.params[b.param as usize].name.as_str())
                .collect();
            let suffixes: Option<Vec<&str>> = bias_names
                .iter()
                .zip(&names)
                .map(|(bias, leg)| bias.strip_prefix(leg))
                .collect();
            match suffixes {
                Some(s) if !s[0].is_empty() && s.iter().all(|x| *x == s[0]) => {
                    format!("{name}{}", s[0])
                }
                _ => Join::name_of(&bias_names),
            }
        });
        out.push(Group {
            biases,
            bias_name,
            name,
            nodes,
            params,
            planes,
            widths,
            form,
        });
    }
    out.sort_by_key(|g| g.nodes[0]);
    out
}

/// `legs` joined along their rows under `name`: each one's cut a segment of
/// the joined one's, with its heads; a param no rank cuts (a scalar input
/// scale) is the first leg's.
fn joined(name: String, legs: &[&Param]) -> Param {
    let first = legs[0];
    let Shard::Cut { .. } = first.shard else {
        return Param {
            name,
            ..first.clone()
        };
    };
    let mut segments = Vec::new();
    let mut heads = Vec::new();
    let mut rows = 0;
    for leg in legs {
        if let Shard::Cut {
            segments: s,
            heads: h,
            ..
        } = &leg.shard
        {
            segments.extend(s.iter().copied());
            heads.push(h.clone());
        }
        rows += leg.shape[0];
    }
    let heads = if heads.iter().all(|h| !h.is_empty()) {
        heads.concat()
    } else {
        Vec::new()
    };
    let mut shape = first.shape.clone();
    shape[0] = rows;
    Param {
        name,
        shape,
        shard: Shard::Cut {
            axis: 0,
            segments,
            heads,
        },
        ..first.clone()
    }
}

fn apply(mut trace: Trace, groups: &[Group]) -> Trace {
    // Params: each join's in its first leg's place, its planes after it.
    let mut joined_at: BTreeMap<u32, Vec<Param>> = BTreeMap::new();
    let mut gone: BTreeSet<u32> = BTreeSet::new();
    // A leg's main or plane param, and the joined param that takes its place.
    let mut into: BTreeMap<u32, (u32, usize)> = BTreeMap::new();
    // Each join's bias, in its first leg's bias's place.
    let mut bias_at: BTreeMap<u32, (u32, Param)> = BTreeMap::new();
    let mut bias_into: BTreeMap<u32, u32> = BTreeMap::new();
    for (g, group) in groups.iter().enumerate() {
        let legs: Vec<&Param> = group
            .params
            .iter()
            .map(|p| &trace.params[*p as usize])
            .collect();
        let mut made = vec![joined(group.name.clone(), &legs)];
        for (i, (suffix, _)) in group.planes[0].iter().enumerate() {
            let planes: Vec<&Param> = group
                .planes
                .iter()
                .map(|planes| &trace.params[planes[i].1 as usize])
                .collect();
            made.push(joined(format!("{}.{suffix}", group.name), &planes));
        }
        if let Some(biases) = &group.biases {
            let legs: Vec<&Param> = biases
                .iter()
                .map(|b| &trace.params[b.param as usize])
                .collect();
            let at = biases[0].param;
            bias_at.insert(at, (g as u32, joined(group.bias_name.clone(), &legs)));
            for b in biases {
                gone.insert(b.param);
                bias_into.insert(b.param, g as u32);
            }
        }
        let first = group.params[0];
        for (leg, p) in group.params.iter().enumerate() {
            gone.insert(*p);
            into.insert(*p, (g as u32, 0));
            for (i, (_, plane)) in group.planes[leg].iter().enumerate() {
                gone.insert(*plane);
                into.insert(*plane, (g as u32, i + 1));
            }
        }
        joined_at.insert(first, made);
    }
    let mut params = Vec::with_capacity(trace.params.len());
    let mut new_index: BTreeMap<u32, u32> = BTreeMap::new();
    let mut group_index: BTreeMap<(u32, usize), u32> = BTreeMap::new();
    let mut bias_index: BTreeMap<u32, u32> = BTreeMap::new();
    for (at, param) in trace.params.iter().enumerate() {
        let at = at as u32;
        if let Some(made) = joined_at.remove(&at) {
            let g = into[&at].0;
            for (i, p) in made.into_iter().enumerate() {
                group_index.insert((g, i), params.len() as u32);
                params.push(p);
            }
        }
        if let Some((g, p)) = bias_at.remove(&at) {
            bias_index.insert(g, params.len() as u32);
            params.push(p);
        }
        if !gone.contains(&at) {
            new_index.insert(at, params.len() as u32);
            params.push(param.clone());
        }
    }
    for (old, (g, i)) in &into {
        new_index.insert(*old, group_index[&(*g, *i)]);
    }
    for (old, g) in &bias_into {
        new_index.insert(*old, bias_index[g]);
    }
    trace.params = params;
    for decl in &mut trace.values {
        if let Def::Weight(p) = &mut decl.def {
            *p = new_index[p];
        }
    }

    // Nodes: each join's product in its first leg's place, its split after;
    // the other legs' products gone.
    let mut drop: BTreeSet<usize> = BTreeSet::new();
    let mut after: BTreeMap<usize, Vec<Node>> = BTreeMap::new();
    for group in groups {
        let first = group.nodes[0];
        // What each leg stands for: its product, or the bias added to it.
        let ys: Vec<ValueId> = match &group.biases {
            Some(biases) => biases.iter().map(|b| b.out).collect(),
            None => group
                .nodes
                .iter()
                .map(|&at| match &trace.nodes[at].op {
                    Operation::Linear(Linear::Matmul { y, .. }) => *y,
                    _ => unreachable!("grouped nodes are products"),
                })
                .collect(),
        };
        let whole: u64 = group.widths.iter().sum();
        let y_ty = match &trace.values[ys[0].0 as usize].ty {
            Ty::Tensor { shape, dtype } => Ty::Tensor {
                shape: vec![shape[0], Dim::Const(whole)],
                dtype: *dtype,
            },
            other => other.clone(),
        };
        let y = ValueId(trace.values.len() as u32);
        trace.values.push(ValueDecl {
            def: Def::Op(u32::MAX),
            ty: y_ty,
        });
        // The joined weight is the first leg's value, as tall as its legs'.
        let legs: Vec<(ValueId, Option<ValueId>)> = group
            .nodes
            .iter()
            .map(|&at| match &trace.nodes[at].op {
                Operation::Linear(Linear::Matmul { w, scales, .. }) => (*w, *scales),
                _ => unreachable!("grouped nodes are products"),
            })
            .collect();
        let tall = |trace: &Trace, vs: Vec<ValueId>| -> u64 {
            vs.iter()
                .map(|v| match &trace.values[v.0 as usize].ty {
                    Ty::Tensor { shape, .. } => match shape.first() {
                        Some(Dim::Const(n)) => *n,
                        _ => 0,
                    },
                    Ty::Struct(_) => 0,
                })
                .sum()
        };
        let w_rows = tall(&trace, legs.iter().map(|l| l.0).collect());
        let s_rows = tall(&trace, legs.iter().filter_map(|l| l.1).collect());
        for (v, rows) in [(Some(legs[0].0), w_rows), (legs[0].1, s_rows)] {
            if let Some(v) = v
                && let Ty::Tensor { shape, .. } = &mut trace.values[v.0 as usize].ty
            {
                shape[0] = Dim::Const(rows);
            }
        }
        if let Operation::Linear(Linear::Matmul { y: out, .. }) = &mut trace.nodes[first].op {
            *out = y;
        }
        for &at in &group.nodes[1..] {
            drop.insert(at);
        }
        // A biased join adds the joined bias to the joined product, and the
        // split reads that.
        let mut extra = Vec::new();
        let y = match &group.biases {
            None => y,
            Some(biases) => {
                let ty = trace.values[y.0 as usize].ty.clone();
                let z = ValueId(trace.values.len() as u32);
                trace.values.push(ValueDecl {
                    def: Def::Op(u32::MAX),
                    ty,
                });
                let rows: u64 = group.widths.iter().sum();
                if let Ty::Tensor { shape, .. } = &mut trace.values[biases[0].bias.0 as usize].ty {
                    shape[0] = Dim::Const(rows);
                }
                for b in biases {
                    drop.insert(b.at);
                }
                let node = &trace.nodes[first];
                extra.push(Node {
                    op: Operation::Elemwise(Elemwise::AddBias {
                        bias: biases[0].bias,
                        out: y,
                        out_out: z,
                    }),
                    guard: node.guard.clone(),
                    layer: node.layer,
                });
                z
            }
        };
        match group.form {
            Form::Glu { at } => {
                if let Operation::Linear(Linear::Glu {
                    packed, gate, up, ..
                }) = &mut trace.nodes[at].op
                {
                    *packed = Some(y);
                    *gate = None;
                    *up = None;
                }
            }
            Form::Qkv { q, kv } => {
                let node = &trace.nodes[first];
                extra.push(Node {
                    op: Operation::Gather(Gather::SplitQkv {
                        packed: y,
                        q_width: q,
                        kv_width: kv,
                        q: ys[0],
                        k: ys[1],
                        v: ys[2],
                    }),
                    guard: node.guard.clone(),
                    layer: node.layer,
                });
            }
            Form::Rows { left } => {
                let node = &trace.nodes[first];
                extra.push(Node {
                    op: Operation::Gather(Gather::SplitRows {
                        x: y,
                        width: left,
                        left: ys[0],
                        right: ys[1],
                    }),
                    guard: node.guard.clone(),
                    layer: node.layer,
                });
            }
        }
        after.insert(first, extra);
    }
    let old = std::mem::take(&mut trace.nodes);
    for (at, node) in old.into_iter().enumerate() {
        if drop.contains(&at) {
            continue;
        }
        trace.nodes.push(node);
        trace.nodes.extend(after.remove(&at).unwrap_or_default());
    }
    compact(trace)
}

/// A value's leading axis, which every leg's result must share.
fn rows_of(ty: &Ty) -> Option<Dim> {
    match ty {
        Ty::Tensor { shape, .. } => shape.first().copied(),
        Ty::Struct(_) => None,
    }
}

/// `trace` without the values nothing reads or makes, renumbered, every
/// maker found again.
fn compact(mut trace: Trace) -> Trace {
    let mut live: BTreeSet<ValueId> = BTreeSet::new();
    let mut ids = Vec::new();
    for node in &trace.nodes {
        node.op.inputs(&mut ids);
        node.op.outputs(&mut ids);
    }
    live.extend(ids.drain(..));
    for seam in &trace.seams {
        live.extend(seam.values.iter().copied());
    }
    // A merge's arms are live with it; inputs and caches are kept as they are.
    for (v, decl) in trace.values.iter().enumerate() {
        match &decl.def {
            Def::Input(_) | Def::Cache(_) => {
                live.insert(ValueId(v as u32));
            }
            Def::Merge(arms) if live.contains(&ValueId(v as u32)) => {
                live.extend(arms.iter().map(|(arm, _)| *arm));
            }
            _ => {}
        }
    }
    let renumber: BTreeMap<ValueId, ValueId> = live
        .iter()
        .enumerate()
        .map(|(new, old)| (*old, ValueId(new as u32)))
        .collect();
    let at = |v: ValueId| renumber[&v];
    let values = live
        .iter()
        .map(|v| {
            let mut decl = trace.values[v.0 as usize].clone();
            if let Def::Merge(arms) = &mut decl.def {
                for (arm, _) in arms.iter_mut() {
                    *arm = at(*arm);
                }
            }
            decl
        })
        .collect();
    trace.values = values;
    for node in &mut trace.nodes {
        node.op = crate::tree::remap(&node.op, &at);
    }
    for seam in &mut trace.seams {
        for v in &mut seam.values {
            *v = at(*v);
        }
    }
    let mut outs = Vec::new();
    for (i, node) in trace.nodes.iter().enumerate() {
        outs.clear();
        node.op.outputs(&mut outs);
        for v in &outs {
            if let Def::Op(maker) = &mut trace.values[v.0 as usize].def {
                *maker = i as u32;
            }
        }
    }
    trace
}
