//! Horizontal packing: products of one activation by weights stored alike
//! are joined into one product by the weights joined along their rows, its
//! result split back into theirs. A package states each weight it means (q,
//! k and v; a gate and an up); this pass is where they become the packed
//! weight a kernel reads at once, and [`joins`] is what an import reads them
//! into, so the trace and the artifact agree.
//!
//! What joins: products with no input-scale operand and no logits head,
//! each by a checkpoint weight its dataflow cuts along its rows (one rank's
//! inference, [`shape`]) that nothing else reads, under one guard and one
//! layer, at one precision, the weights' planes alike, held alike by the
//! ranks: cut, or whole where their count cannot split what one rank's
//! dataflow cuts, the joined weight then held whole. What reads their
//! results says how: two a gated unit reads as its gate and its up become
//! one packed unit; legs a concat reads side by side (GDN's beta and alpha)
//! are the one product the concat was; legs left that cut by more than one
//! segment, when they are all those left (GDN's qkv and z), are split back
//! after the one product, its rows cut after it.
//! The rest, each of one segment and stating its heads as the others do,
//! join by their activation alone: three whose last two are as wide (q, k,
//! v) a split into three, two a split into two, more the last peeled off by
//! rows until three or two are left. Routed selects of a gate's and an up's
//! banks over one route join along the banks' rows, their biases and static
//! input scales (one an expert) with them. A joined weight's static input
//! scale is its first leg's: the legs read one activation, so a calibration
//! states them alike, and an import whose legs' differ is refused.

use std::collections::{BTreeMap, BTreeSet};

use poem::Join;
use poem_ir::{
    Def, Dim, Elemwise, Gather, Linear, Node, Operands, Operation, Param, ParamLayout,
    ParamPlacement, ParamSource, Shard, Trace, Ty, ValueDecl, ValueId,
};

/// The joins [`pack`] makes of `trace`, in the order it makes them.
#[must_use]
pub fn joins(trace: &Trace) -> Vec<Join> {
    joins_as(trace, &trace.params)
}

/// The joins [`pack_as`] makes of `trace` as `shape` cuts its params.
#[must_use]
pub fn joins_as(trace: &Trace, shape: &[Param]) -> Vec<Join> {
    let name = |p: &u32| trace.params[*p as usize].name.clone();
    groups(trace, shape)
        .iter()
        .flat_map(|g| {
            let weight = Join {
                name: g.name.clone(),
                legs: g.params.iter().map(name).collect(),
                axis: g.axis as u32,
            };
            let bias = g.biases.as_ref().map(|b| Join {
                name: g.bias_name.clone(),
                legs: b.iter().map(|b| name(&b.param)).collect(),
                axis: g.axis as u32,
            });
            std::iter::once(weight).chain(bias)
        })
        .fold(Vec::new(), |mut joins: Vec<Join>, join| {
            if !joins.iter().any(|j| j.name == join.name) {
                joins.push(join);
            }
            joins
        })
}

/// `trace` with each join made.
#[must_use]
pub fn pack(trace: Trace) -> Trace {
    let shape = trace.params.clone();
    pack_as(trace, &shape)
}

/// `trace` with each join made that its params as `shape` cuts them make:
/// legs join as their dataflow cuts them (`shape`, the trace's params cut
/// as one rank's dataflow would, before a count of ranks put any back
/// whole), and the joined weight is held as the legs are, cut or whole.
#[must_use]
pub fn pack_as(trace: Trace, shape: &[Param]) -> Trace {
    let groups = groups(&trace, shape);
    if groups.is_empty() {
        return trace;
    }
    apply(trace, &groups)
}

/// `trace`'s params as its dataflow cuts them, for [`pack_as`]: the cuts
/// one rank infers of those the package leaves to the compiler, the rest
/// as stated.
#[must_use]
pub fn shape(trace: &Trace) -> Vec<Param> {
    let mut params = trace.params.clone();
    if params.iter().any(|p| p.shard == Shard::Inferred) {
        let cuts = crate::shard::infer::infer(trace);
        for (param, cut) in params.iter_mut().zip(cuts) {
            if param.shard == Shard::Inferred {
                param.shard = cut;
            }
        }
    }
    params
}

/// How a join's result is split back.
#[derive(Debug)]
enum Form {
    /// Read by one gated unit as its gate (the first) and its up.
    Glu { at: usize },
    /// Read side by side, in the legs' order, by one concat (node `at`):
    /// the joined result is what it makes.
    Concat { at: usize },
    /// Split back into each leg's result: two by their rows; three whose
    /// last two are as wide as q, k and v; more, the last peeled off by
    /// rows until three or two are left.
    Split,
}

/// A product's bias, added by the op right after it: that op, the bias's
/// param and value, and the op's result, which the leg then stands for. A
/// routed select's bias is its own operand: `at` is the select.
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
    /// The products, in the order their rows join.
    nodes: Vec<usize>,
    /// Each product's weight's param, and its planes' (suffix, param).
    params: Vec<u32>,
    planes: Vec<Vec<(String, u32)>>,
    widths: Vec<u64>,
    form: Form,
    /// The axis the weights join along: their rows, a bank's axis 1.
    axis: usize,
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
    for output in &trace.outputs {
        for v in output.values() {
            count[v.0 as usize] += 1;
        }
    }
    count
}

/// A param a join may take as a leg, as its dataflow cuts it: a checkpoint
/// weight on the device in its natural order, one segment cut along its
/// rows.
fn leg(param: &Param) -> bool {
    leg_along(param, 0) && segments(param) == 1
}

/// A checkpoint weight on the device in its natural order, its dataflow
/// cutting it along `along` (a routed bank's rows are its axis 1).
fn leg_along(param: &Param, along: u32) -> bool {
    param.source == ParamSource::Checkpoint
        && param.placement == ParamPlacement::Device
        && param.layout == ParamLayout::Natural
        && param.experts.is_none()
        && matches!(&param.shard, Shard::Cut { axis, .. } if *axis == along)
}

/// Whether the ranks cut `param`, or hold it whole where the count of
/// them cannot split what its dataflow cuts: legs join only with legs held
/// alike.
fn cut(param: &Param) -> bool {
    matches!(param.shard, Shard::Cut { .. })
}

/// The segments `param`'s cut holds.
fn segments(param: &Param) -> usize {
    match &param.shard {
        Shard::Cut { segments, .. } => segments.len(),
        _ => 0,
    }
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
fn biased(trace: &Trace, shape: &[Param], read: &[u32], y: ValueId) -> Option<Biased> {
    if read[y.0 as usize] != 1 {
        return None;
    }
    trace.nodes.iter().enumerate().find_map(|(at, node)| {
        let Operation::Elemwise(Elemwise::AddBias { bias, out, out_out }) = &node.op else {
            return None;
        };
        if *out != y {
            return None;
        }
        let param = weight_param(trace, *bias)?;
        leg(&shape[param as usize]).then_some(Biased {
            at,
            param,
            bias: *bias,
            out: *out_out,
        })
    })
}

/// The legs of one key, by what reads their results: a gated unit's gate
/// and up and a concat's pieces join as they read them, and every leg left,
/// when one cuts by more than one segment, in the order they are traced; the
/// rest (each of one segment) join by their activation alone, two, three
/// whose last two are as wide, or more. Each is the legs' indices in the
/// order their rows join, and its form.
fn subgroups(
    trace: &Trace,
    read: &[u32],
    ys: &[ValueId],
    single: &[bool],
    headed: &[bool],
    widths: &[u64],
) -> Vec<(Vec<usize>, Form)> {
    let index: BTreeMap<ValueId, usize> = ys.iter().enumerate().map(|(i, y)| (*y, i)).collect();
    let mut left: BTreeSet<usize> = (0..ys.len()).collect();
    let mut out = Vec::new();
    // The legs `vs` are, if each is a leg not yet taken, once.
    let take = |left: &BTreeSet<usize>, vs: &[ValueId]| -> Option<Vec<usize>> {
        let legs: Vec<usize> = vs.iter().map(|v| index.get(v).copied()).collect::<Option<_>>()?;
        let distinct: BTreeSet<usize> = legs.iter().copied().collect();
        (legs.len() >= 2 && distinct.len() == legs.len() && legs.iter().all(|i| left.contains(i)))
            .then_some(legs)
    };
    for (at, node) in trace.nodes.iter().enumerate() {
        match &node.op {
            Operation::Linear(Linear::Glu {
                packed: None,
                gate: Some(g),
                up: Some(u),
                ..
            }) if read[g.0 as usize] == 1 && read[u.0 as usize] == 1 => {
                if let Some(legs) = take(&left, &[*g, *u]) {
                    left.retain(|i| !legs.contains(i));
                    out.push((legs, Form::Glu { at }));
                }
            }
            Operation::Gather(Gather::ConcatCols { xs, .. })
                if xs.iter().all(|x| read[x.0 as usize] == 1) =>
            {
                if let Some(legs) = take(&left, xs) {
                    left.retain(|i| !legs.contains(i));
                    out.push((legs, Form::Concat { at }));
                }
            }
            _ => {}
        }
    }
    // Every leg left, when one cuts by more than one segment, is one
    // projection's rows cut after it (GDN's qkv and z).
    let all: Vec<usize> = left.iter().copied().collect();
    if all.len() >= 2
        && all.iter().any(|i| !single[*i])
        && splits(&all.iter().map(|i| widths[*i]).collect::<Vec<_>>())
    {
        left.clear();
        out.push((all, Form::Split));
    }
    // The rest join with the rest that state their heads as they do.
    for heads in [false, true] {
        let rest: Vec<usize> = (left.iter().copied())
            .filter(|i| single[*i] && headed[*i] == heads)
            .collect();
        let rest_widths: Vec<u64> = rest.iter().map(|i| widths[*i]).collect();
        let joins = match rest_widths.as_slice() {
            [_, _] => true,
            [_, a, b] => a == b,
            _ => rest.len() > 3 && splits(&rest_widths),
        };
        if joins {
            out.push((rest, Form::Split));
        }
    }
    out
}

/// Whether results `widths` wide split back from one: the last peeled off
/// by rows until three whose last two are as wide, or two, are left.
fn splits(widths: &[u64]) -> bool {
    match widths {
        [_, _] => true,
        [_, a, b] if a == b => true,
        [rest @ .., _] if rest.len() >= 2 => splits(rest),
        _ => false,
    }
}

fn groups(trace: &Trace, shape: &[Param]) -> Vec<Group> {
    let read = reads(trace);
    // A param as its dataflow cuts it, whatever the ranks hold.
    let cut_as = |p: u32| &shape[p as usize];
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
        if !leg_along(cut_as(p), 0) || param.shape.len() < 2 {
            continue;
        }
        if let Some(s) = scales {
            let Some(sp) = weight_param(trace, *s) else {
                continue;
            };
            if !leg_along(cut_as(sp), 0) {
                continue;
            }
        }
        let planes: Vec<String> = planes_of(trace, &param.name)
            .into_iter()
            .map(|(suffix, _)| suffix)
            .collect();

        let bias = biased(trace, shape, &read, *y).map(|b| {
            (
                headed(cut_as(b.param)),
                cut(&trace.params[b.param as usize]),
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
            cut(param),
        );
        keyed.entry(key).or_default().push(at);
    }

    // Routed selects of one activation over one route by banks stored alike
    // (a gate's and an up's), joined along the banks' rows, each one's bias
    // (a bank of rows an expert) with it.
    for (at, node) in trace.nodes.iter().enumerate() {
        let Operation::Linear(Linear::MoeSelect {
            x,
            bank,
            bias,
            routes,
            input_scales,
            quantized,
            quant,
            y,
        }) = &node.op
        else {
            continue;
        };
        let Some(p) = weight_param(trace, *bank) else {
            continue;
        };
        let param = &trace.params[p as usize];
        // A bank's static input scales (one an expert) are a plane of it,
        // the joined bank's its first leg's.
        let scaled = match input_scales {
            None => false,
            Some(v) => match weight_param(trace, *v) {
                Some(sp) if planes_of(trace, &param.name).iter().any(|(_, q)| *q == sp) => true,
                _ => continue,
            },
        };
        if !leg_along(cut_as(p), 1) || segments(cut_as(p)) != 1 || param.shape.len() < 3 {
            continue;
        }
        let bias = match bias {
            None => None,
            Some(b) => match weight_param(trace, *b) {
                Some(bp)
                    if leg_along(cut_as(bp), 1)
                        && segments(cut_as(bp)) == 1
                        && cut(&trace.params[bp as usize]) == cut(param) =>
                {
                    Some(trace.params[bp as usize].dtype)
                }
                _ => continue,
            },
        };
        let planes: Vec<String> = planes_of(trace, &param.name)
            .into_iter()
            .map(|(suffix, _)| suffix)
            .collect();
        let mut rest = param.shape.clone();
        rest.remove(1);
        let key = format!(
            "select|{}|{}|{quantized}|{quant:?}|{scaled}|{:?}|{rest:?}|{planes:?}|{:?}|{:?}|{:?}|{}|{bias:?}",
            x.0,
            routes.0,
            param.dtype,
            node.guard,
            node.layer,
            rows_of(&trace.values[y.0 as usize].ty),
            cut(param),
        );
        keyed.entry(key).or_default().push(at);
    }

    let mut out = Vec::new();
    for nodes in keyed.into_values() {
        if nodes.len() < 2 {
            continue;
        }
        let selects = matches!(
            trace.nodes[nodes[0]].op,
            Operation::Linear(Linear::MoeSelect { .. })
        );
        if selects {
            let at = |i: usize| match &trace.nodes[nodes[i]].op {
                Operation::Linear(Linear::MoeSelect { bank, bias, y, .. }) => (*bank, *bias, *y),
                _ => unreachable!("grouped nodes are selects"),
            };
            if nodes.len() != 2 {
                continue;
            }
            let ((gate_bank, gate_bias, gate), (up_bank, up_bias, up)) = (at(0), at(1));
            let glu = trace.nodes.iter().position(|n| {
                matches!(&n.op, Operation::Linear(Linear::Glu { packed: None, gate: Some(g), up: Some(u), .. })
                    if *g == gate && *u == up)
            });
            let Some(glu) = glu.filter(|_| read[gate.0 as usize] == 1 && read[up.0 as usize] == 1)
            else {
                continue;
            };
            let params = [gate_bank, up_bank].map(|b| weight_param(trace, b).expect("a bank"));
            let names: Vec<&str> = params
                .iter()
                .map(|p| trace.params[*p as usize].name.as_str())
                .collect();
            let name = Join::name_of(&names);
            let biases = match (gate_bias, up_bias) {
                (Some(g), Some(u)) => Some(
                    [(g, nodes[0], gate), (u, nodes[1], up)]
                        .map(|(b, at, out)| Biased {
                            at,
                            param: weight_param(trace, b).expect("a bias is a param"),
                            bias: b,
                            out,
                        })
                        .to_vec(),
                ),
                _ => None,
            };
            let bias_name = biases.as_ref().map_or_else(String::new, |b| {
                bias_name_of(trace, &name, &names, b)
            });
            out.push(Group {
                name,
                biases,
                bias_name,
                widths: params
                    .iter()
                    .map(|p| trace.params[*p as usize].shape[1])
                    .collect(),
                planes: params
                    .iter()
                    .map(|p| planes_of(trace, &trace.params[*p as usize].name))
                    .collect(),
                params: params.to_vec(),
                nodes,
                form: Form::Glu { at: glu },
                axis: 1,
            });
            continue;
        }
        let all_biases: Option<Vec<Biased>> = nodes
            .iter()
            .map(|&at| match &trace.nodes[at].op {
                Operation::Linear(Linear::Matmul { y, .. }) => biased(trace, shape, &read, *y),
                _ => unreachable!("grouped nodes are products"),
            })
            .collect();
        // What each leg stands for: its product, or the bias added to it.
        let ys: Vec<ValueId> = nodes
            .iter()
            .enumerate()
            .map(|(i, &at)| match (&all_biases, &trace.nodes[at].op) {
                (Some(b), _) => b[i].out,
                (None, Operation::Linear(Linear::Matmul { y, .. })) => *y,
                _ => unreachable!("grouped nodes are products"),
            })
            .collect();
        let all_params: Vec<u32> = nodes
            .iter()
            .map(|&at| match &trace.nodes[at].op {
                Operation::Linear(Linear::Matmul { w, .. }) => {
                    weight_param(trace, *w).expect("a leg's weight is a param")
                }
                _ => unreachable!("grouped nodes are products"),
            })
            .collect();
        let all_widths: Vec<u64> = all_params
            .iter()
            .map(|p| trace.params[*p as usize].shape[0])
            .collect();
        let single: Vec<bool> = all_params
            .iter()
            .map(|p| leg(cut_as(*p)))
            .collect();
        let heads: Vec<bool> = all_params
            .iter()
            .map(|p| headed(cut_as(*p)))
            .collect();
        for (legs, form) in subgroups(trace, &read, &ys, &single, &heads, &all_widths) {
            let params: Vec<u32> = legs.iter().map(|i| all_params[*i]).collect();
            let widths: Vec<u64> = legs.iter().map(|i| all_widths[*i]).collect();
            let biases: Option<Vec<Biased>> = all_biases
                .as_ref()
                .map(|b| legs.iter().map(|i| b[*i]).collect());
            let names: Vec<&str> = params
                .iter()
                .map(|p| trace.params[*p as usize].name.as_str())
                .collect();
            let planes = params
                .iter()
                .map(|p| planes_of(trace, &trace.params[*p as usize].name))
                .collect();
            let name = Join::name_of(&names);
            let bias_name = biases.as_ref().map_or_else(String::new, |b| {
                bias_name_of(trace, &name, &names, b)
            });
            out.push(Group {
                axis: 0,
                biases,
                bias_name,
                name,
                nodes: legs.iter().map(|i| nodes[*i]).collect(),
                params,
                planes,
                widths,
                form,
            });
        }
    }
    out.sort_by_key(|g| g.nodes.iter().copied().min());
    consistent(trace, out)
}

/// The name a join's biases go by: a bias named for its weight (`q_bias`
/// beside `q`) is named so for the joined weight (`qkv_bias`), any other
/// as its legs join.
fn bias_name_of(trace: &Trace, name: &str, names: &[&str], biases: &[Biased]) -> String {
    let bias_names: Vec<&str> = biases
        .iter()
        .map(|b| trace.params[b.param as usize].name.as_str())
        .collect();
    let suffixes: Option<Vec<&str>> = bias_names
        .iter()
        .zip(names)
        .map(|(bias, leg)| bias.strip_prefix(leg))
        .collect();
    match suffixes {
        Some(s) if !s[0].is_empty() && s.iter().all(|x| *x == s[0]) => {
            format!("{name}{}", s[0])
        }
        _ => Join::name_of(&bias_names),
    }
}

/// The groups whose legs every reader of them joins alike: a weight read at
/// several sites (a layer run twice) joins only where every site reads it
/// in a group of the same legs, so one joined weight serves them all.
fn consistent(trace: &Trace, groups: Vec<Group>) -> Vec<Group> {
    // The nodes reading each param, through any of its values.
    let mut readers: Vec<BTreeSet<usize>> = vec![BTreeSet::new(); trace.params.len()];
    let mut ins = Vec::new();
    for (at, node) in trace.nodes.iter().enumerate() {
        ins.clear();
        node.op.inputs(&mut ins);
        for v in &ins {
            if let Def::Weight(p) = trace.values[v.0 as usize].def {
                readers[p as usize].insert(at);
            }
        }
    }
    let legs_of = |g: &Group| -> Vec<u32> {
        let mut legs = g.params.clone();
        if let Some(b) = &g.biases {
            legs.extend(b.iter().map(|b| b.param));
        }
        legs
    };
    let sites_of = |g: &Group| -> BTreeSet<usize> {
        let mut sites: BTreeSet<usize> = g.nodes.iter().copied().collect();
        if let Some(b) = &g.biases {
            sites.extend(b.iter().map(|b| b.at));
        }
        sites
    };
    let mut by_leg: BTreeMap<u32, Vec<usize>> = BTreeMap::new();
    for (i, g) in groups.iter().enumerate() {
        for p in legs_of(g) {
            by_leg.entry(p).or_default().push(i);
        }
    }
    let ok: Vec<bool> = groups
        .iter()
        .map(|g| {
            legs_of(g).iter().all(|p| {
                let mine = &by_leg[p];
                let sites: BTreeSet<usize> =
                    mine.iter().flat_map(|&i| sites_of(&groups[i])).collect();
                mine.iter().all(|&i| legs_of(&groups[i]) == legs_of(g))
                    && readers[*p as usize].is_subset(&sites)
            })
        })
        .collect();
    groups
        .into_iter()
        .zip(ok)
        .filter_map(|(g, ok)| ok.then_some(g))
        .collect()
}

/// `legs` joined along their rows under `name`: each one's cut a segment of
/// the joined one's, with its heads; legs every rank holds whole, whole. A
/// plane of the weights' `rank` or a row each of their `rows` joins so; one
/// of neither (a scalar input scale) is the first leg's.
fn joined(name: String, legs: &[&Param], axis: usize, rank: usize, rows: &[u64]) -> Param {
    let first = legs[0];
    let Shard::Cut { .. } = first.shard else {
        let mut shape = first.shape.clone();
        let along = |leg: &Param, i: usize| {
            leg.shape.len() == rank || leg.shape.get(axis) == rows.get(i)
        };
        if legs.iter().enumerate().all(|(i, leg)| along(leg, i)) && shape.len() > axis {
            shape[axis] = legs.iter().map(|leg| leg.shape[axis]).sum();
        }
        return Param {
            name,
            shape,
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
        rows += leg.shape[axis];
    }
    let heads = if heads.iter().all(|h| !h.is_empty()) {
        heads.concat()
    } else {
        Vec::new()
    };
    let mut shape = first.shape.clone();
    shape[axis] = rows;
    Param {
        name,
        shape,
        shard: Shard::Cut {
            axis: axis as u32,
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
        let rank = legs[0].shape.len();
        let rows: Vec<u64> = legs.iter().map(|leg| leg.shape[group.axis]).collect();
        let mut made = vec![joined(group.name.clone(), &legs, group.axis, rank, &rows)];
        for (i, (suffix, _)) in group.planes[0].iter().enumerate() {
            let planes: Vec<&Param> = group
                .planes
                .iter()
                .map(|planes| &trace.params[planes[i].1 as usize])
                .collect();
            made.push(joined(
                format!("{}.{suffix}", group.name),
                &planes,
                group.axis,
                rank,
                &rows,
            ));
        }
        if let Some(biases) = &group.biases {
            let legs: Vec<&Param> = biases
                .iter()
                .map(|b| &trace.params[b.param as usize])
                .collect();
            let at = biases[0].param;
            let rank = legs[0].shape.len();
            bias_at.insert(
                at,
                (
                    g as u32,
                    joined(group.bias_name.clone(), &legs, group.axis, rank, &rows),
                ),
            );
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
        // A weight several sites read joins once; each site is rewritten.
        joined_at.entry(first).or_insert(made);
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

    // Nodes: each join's product in its earliest leg's place, its split
    // after; the other legs' products gone.
    let mut drop: BTreeSet<usize> = BTreeSet::new();
    let mut after: BTreeMap<usize, Vec<Node>> = BTreeMap::new();
    let mut retyped: BTreeSet<ValueId> = BTreeSet::new();
    // Values a join's result stands for: a concat's, which it is.
    let mut subst: BTreeMap<ValueId, ValueId> = BTreeMap::new();
    for group in groups {
        let first = *group.nodes.iter().min().expect("a join has legs");
        // What each leg stands for: its product, or the bias added to it.
        let ys: Vec<ValueId> = match &group.biases {
            Some(biases) => biases.iter().map(|b| b.out).collect(),
            None => group
                .nodes
                .iter()
                .map(|&at| match &trace.nodes[at].op {
                    Operation::Linear(Linear::Matmul { y, .. } | Linear::MoeSelect { y, .. }) => *y,
                    _ => unreachable!("grouped nodes are products"),
                })
                .collect(),
        };
        let whole: u64 = group.widths.iter().sum();
        let y_ty = match &trace.values[ys[0].0 as usize].ty {
            Ty::Tensor { shape, dtype } => {
                let mut shape = shape.clone();
                if let Some(last) = shape.last_mut() {
                    *last = Dim::Const(whole);
                }
                Ty::Tensor {
                    shape,
                    dtype: *dtype,
                }
            }
            other => other.clone(),
        };
        let fresh = |trace: &mut Trace, ty: Ty| {
            let v = ValueId(trace.values.len() as u32);
            trace.values.push(ValueDecl {
                def: Def::Op(u32::MAX),
                ty,
            });
            v
        };
        let y = fresh(&mut trace, y_ty.clone());
        // The joined weight is the first leg's value, as tall as its legs'.
        let legs: Vec<(ValueId, Option<ValueId>)> = group
            .nodes
            .iter()
            .map(|&at| match &trace.nodes[at].op {
                Operation::Linear(Linear::Matmul { w, scales, .. }) => (*w, *scales),
                Operation::Linear(Linear::MoeSelect { bank, .. }) => (*bank, None),
                _ => unreachable!("grouped nodes are products"),
            })
            .collect();
        let axis = group.axis;
        let tall = |trace: &Trace, vs: Vec<ValueId>| -> u64 {
            vs.iter()
                .map(|v| match &trace.values[v.0 as usize].ty {
                    Ty::Tensor { shape, .. } => match shape.get(axis) {
                        Some(Dim::Const(n)) => *n,
                        _ => 0,
                    },
                    Ty::Struct(_) => 0,
                })
                .sum()
        };
        let w_rows = tall(&trace, legs.iter().map(|l| l.0).collect());
        let s_rows = tall(&trace, legs.iter().filter_map(|l| l.1).collect());
        // A weight value several sites read is retyped once, at the first.
        for (v, rows) in [(Some(legs[0].0), w_rows), (legs[0].1, s_rows)] {
            if let Some(v) = v
                && retyped.insert(v)
                && let Ty::Tensor { shape, .. } = &mut trace.values[v.0 as usize].ty
            {
                shape[axis] = Dim::Const(rows);
            }
        }
        // A routed select's biases join as its banks do, the select adding
        // the joined one.
        let selected_bias = match &group.biases {
            Some(biases) if axis == 1 => {
                let rows: u64 = group.widths.iter().sum();
                if let Ty::Tensor { shape, .. } = &mut trace.values[biases[0].bias.0 as usize].ty {
                    shape[axis] = Dim::Const(rows);
                }
                Some(biases[0].bias)
            }
            _ => None,
        };
        match &mut trace.nodes[first].op {
            Operation::Linear(Linear::Matmul { w, scales, y: out, .. }) => {
                (*w, *scales, *out) = (legs[0].0, legs[0].1, y);
            }
            Operation::Linear(Linear::MoeSelect {
                bank, bias, y: out, ..
            }) => {
                (*bank, *out) = (legs[0].0, y);
                if selected_bias.is_some() {
                    *bias = selected_bias;
                }
            }
            _ => unreachable!("grouped nodes are products"),
        }
        for &at in &group.nodes {
            if at != first {
                drop.insert(at);
            }
        }
        // A biased join adds the joined bias to the joined product, and the
        // split reads that.
        let mut extra = Vec::new();
        let y = match &group.biases {
            Some(biases) if selected_bias.is_none() => {
                let z = fresh(&mut trace, y_ty.clone());
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
            _ => y,
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
            Form::Concat { at } => {
                if let Operation::Gather(Gather::ConcatCols { y: made, .. }) = &trace.nodes[at].op {
                    subst.insert(*made, y);
                }
                drop.insert(at);
            }
            Form::Split => {
                let node = &trace.nodes[first];
                let (guard, layer) = (node.guard.clone(), node.layer);
                split(&mut trace, &mut extra, y, &group.widths, &ys, &guard, layer);
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
    if !subst.is_empty() {
        let to = |v: ValueId| subst.get(&v).copied().unwrap_or(v);
        for node in &mut trace.nodes {
            node.op = crate::tree::remap(&node.op, &to);
        }
        for output in &mut trace.outputs {
            for v in output.values_mut() {
                *v = to(*v);
            }
        }
    }
    compact(trace)
}

/// The nodes that split `y` back into `ys`, `widths` wide: two by their
/// rows, three whose last two are as wide as q, k and v, more the last
/// peeled off by rows and the rest split so.
fn split(
    trace: &mut Trace,
    extra: &mut Vec<Node>,
    y: ValueId,
    widths: &[u64],
    ys: &[ValueId],
    guard: &poem_ir::Guard,
    layer: Option<u32>,
) {
    let width = |w: u64| u32::try_from(w).unwrap_or(u32::MAX);
    let op = match (widths, ys) {
        ([q, kv, _], [a, b, c]) if widths[1] == widths[2] => Gather::SplitQkv {
            packed: y,
            q_width: width(*q),
            kv_width: width(*kv),
            q: *a,
            k: *b,
            v: *c,
        },
        ([left, _], [a, b]) => Gather::SplitRows {
            x: y,
            width: width(*left),
            left: *a,
            right: *b,
        },
        _ => {
            let n = widths.len() - 1;
            let rest_width: u64 = widths[..n].iter().sum();
            let ty = match &trace.values[y.0 as usize].ty {
                Ty::Tensor { shape, dtype } => {
                    let mut shape = shape.clone();
                    if let Some(last) = shape.last_mut() {
                        *last = Dim::Const(rest_width);
                    }
                    Ty::Tensor {
                        shape,
                        dtype: *dtype,
                    }
                }
                other => other.clone(),
            };
            let rest = ValueId(trace.values.len() as u32);
            trace.values.push(ValueDecl {
                def: Def::Op(u32::MAX),
                ty,
            });
            extra.push(Node {
                op: Operation::Gather(Gather::SplitRows {
                    x: y,
                    width: width(rest_width),
                    left: rest,
                    right: ys[n],
                }),
                guard: guard.clone(),
                layer,
            });
            return split(trace, extra, rest, &widths[..n], &ys[..n], guard, layer);
        }
    };
    extra.push(Node {
        op: Operation::Gather(op),
        guard: guard.clone(),
        layer,
    });
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
    for output in &trace.outputs {
        live.extend(output.values());
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
    for output in &mut trace.outputs {
        for v in output.values_mut() {
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
