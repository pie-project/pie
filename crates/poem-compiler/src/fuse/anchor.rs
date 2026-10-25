//! Anchored regions: a matmul and the pointwise ops over its product tile,
//! lowered to one kernels-ir [`Anchored`] (a contraction with an epilogue)
//! that a backend's contraction frames fill their hooks with. The peer of
//! the row chains ([`super::chain`]): the same forming, lowering and
//! folding, anchored on a product instead of a row.
//!
//! **Forming.** An anchor is a `Matmul` over bf16 rows and a bf16 weight
//! stored as it is (no quantized codes, no activation scales), landing bf16
//! rows; its epilogue the run of ops after it, each reading what the run
//! computes, under the matmul's guard and over the product's rows, as a
//! chain's run is. The ops a region lowers ([`super::chain`]'s) and a
//! product's activations: a GLU of the packed `[gate | up]` product (the
//! contraction then lands two parts, [`Layout::Packed`]) and the ungated
//! ones. Of the run, the longest leading part whose anchored region
//! kernels-ir verifies (pointwise over the product tile: no reduction, no
//! gather across it) and the backend serves (`serves`) is the anchor's; a
//! run of no op is no anchor, and the matmul keeps its own kernel.
//!
//! **Lowering.** The product's parts are the epilogue's first parameters;
//! each value it reads and does not compute is an input, by how it is laid
//! out: a tile of the product's rows ([`Input::Tile`], a residual), a
//! vector a column ([`Input::Column`], a bias), one element a row
//! ([`Input::Row`]) or one value ([`Input::One`]). Each value the run
//! computes rounds to its dtype where it is defined, as a chain's do; one
//! anything past the run may read is a result. The product itself is no
//! value: nothing past the run reads it, or the run is no anchor.
//!
//! **Folding** ([`fold`]) replaces each anchor's ops with one
//! `fused.anchored` op, landed where its last op was, as a chain's region
//! lands, and hands back each anchored region by key. Anchors fold before
//! the row chains do, so a product's epilogue is the contraction's, and the
//! chains form over what is left.
//!
//! Anchors are formed and folded only under `compiler.synth_anchors`.

use std::collections::BTreeMap;

use kernels_ir::{Anchored, Contraction, Input, Layout, Pointwise, Quant, Region, Shape};
use poem_ir::{Def, Dim, Dtype, Fused, Linear, Operands, Operation, Trace, Ty, ValueId};

use super::chain::{self, Builder, Escapes, Form, In, Out};

/// An anchor: a matmul and its epilogue's ops, lowered to one anchored
/// region.
#[derive(Debug, Clone, PartialEq)]
pub struct Anchor {
    /// The matmul and the epilogue's ops, in trace order.
    pub nodes: Vec<u32>,
    pub act: ValueId,
    pub w: ValueId,
    /// `ins[k]` is the epilogue's input `k`.
    pub ins: Vec<ValueId>,
    /// `outs[k]` is the epilogue's result `k`.
    pub outs: Vec<ValueId>,
    pub anchored: Anchored,
}

/// Whether a backend serves an anchored region at every count of rows a
/// fire may hand it, its inputs and results stored in the dtypes given (an
/// anchored region states no storage).
pub type Serves<'a> = &'a dyn Fn(&Anchored, &[Dtype], &[Dtype]) -> bool;

/// `trace` with each anchor `serves` takes replaced by one `fused.anchored`
/// op, and the anchored regions its ops name, by key
/// ([`Anchored::key`]).
#[must_use]
pub fn fold(trace: Trace, serves: Serves<'_>) -> (Trace, BTreeMap<u64, Anchored>) {
    let mut anchored = BTreeMap::new();
    let mut folded: Vec<(Vec<u32>, Fused)> = Vec::new();
    for anchor in anchors(&trace, serves) {
        let key = anchor.anchored.key();
        folded.push((
            anchor.nodes,
            Fused::Anchored {
                act: anchor.act,
                w: anchor.w,
                ins: anchor.ins,
                key,
                outs: anchor.outs,
            },
        ));
        anchored.entry(key).or_insert(anchor.anchored);
    }
    if folded.is_empty() {
        return (trace, anchored);
    }
    (
        super::search::compact(chain::replace(trace, folded)),
        anchored,
    )
}

/// Every anchor of `trace` that `serves` takes, in trace order; the ops
/// outside them run as they are.
#[must_use]
pub fn anchors(trace: &Trace, serves: Serves<'_>) -> Vec<Anchor> {
    let escapes = Escapes::of(trace);
    let count = u32::try_from(trace.nodes.len()).unwrap_or(u32::MAX);
    let mut taken = vec![false; trace.nodes.len()];
    let mut out = Vec::new();
    for at in 0..count {
        if taken[at as usize] {
            continue;
        }
        let Some(product) = product_of(trace, at) else {
            continue;
        };
        // The run from the matmul: each op that joins it, up to the first
        // op that reads what it computes and does not join it.
        let mut run = vec![at];
        let mut next = at + 1;
        while next < count && next <= at + chain::REACH && run.len() <= chain::MAX_CHAIN {
            if !taken[next as usize] && joins(trace, &run, next, &product) {
                run.push(next);
            } else if chain::reads_from(trace, &run, next) {
                break;
            }
            next += 1;
        }
        // The longest leading part that is an anchor the backend serves.
        let found = (2..=run.len()).rev().find_map(|len| {
            let part = &run[..len];
            if !chain::keeps_order(trace, part) {
                return None;
            }
            let anchor = lower(trace, part, &product, &escapes)?;
            let stored = |vs: &[ValueId]| -> Vec<Dtype> {
                vs.iter().map(|v| chain::dtype_of(trace, *v)).collect()
            };
            let served = anchor.anchored.verify().is_ok()
                && serves(
                    &anchor.anchored,
                    &stored(&anchor.ins),
                    &stored(&anchor.outs),
                );
            served.then_some(anchor)
        });
        if let Some(anchor) = found {
            for &n in &anchor.nodes {
                taken[n as usize] = true;
            }
            out.push(anchor);
        }
    }
    out
}

/// The product a matmul lands, as an anchor reads it.
#[derive(Debug, Clone, Copy)]
struct Product {
    act: ValueId,
    w: ValueId,
    y: ValueId,
    dim: Dim,
    /// The weight's rows: every part's columns.
    rows: u32,
    k: u32,
}

/// The product node `at` lands, if it is a matmul an anchor holds: bf16
/// rows of `k` against a bf16 weight of `rows x k` stored as it is, into
/// bf16 rows of `rows`.
fn product_of(trace: &Trace, at: u32) -> Option<Product> {
    let Operation::Linear(Linear::Matmul {
        act,
        w,
        input_scales: None,
        scales: None,
        quant: None,
        y,
        ..
    }) = trace.nodes[at as usize].op
    else {
        return None;
    };
    let rows_of = |v: ValueId| match &trace.values[v.0 as usize].ty {
        Ty::Tensor {
            shape,
            dtype: Dtype::Bf16,
        } => match shape.as_slice() {
            [dim, Dim::Const(width)] if !matches!(dim, Dim::Const(_)) => Some((*dim, *width)),
            _ => None,
        },
        _ => None,
    };
    let (dim, k) = rows_of(act)?;
    let (out_dim, rows) = rows_of(y)?;
    let Def::Weight(param) = trace.values[w.0 as usize].def else {
        return None;
    };
    let weight = &trace.values[w.0 as usize].ty;
    let stored = trace.params[param as usize].dtype;
    let laid = matches!(weight, Ty::Tensor { shape, .. }
        if shape.as_slice() == [Dim::Const(rows), Dim::Const(k)]);
    if out_dim != dim || stored != Dtype::Bf16 || !laid {
        return None;
    }
    Some(Product {
        act,
        w,
        y,
        dim,
        rows: u32::try_from(rows).ok()?,
        k: u32::try_from(k).ok()?,
    })
}

/// The width of each half of the packed product `node`'s GLU reads, where
/// it is one reading the product as its `[gate | up]` halves.
fn glu_of(trace: &Trace, node: u32, product: &Product) -> Option<u32> {
    match trace.nodes[node as usize].op {
        Operation::Linear(Linear::Glu {
            packed: Some(packed),
            gate: None,
            up: None,
            weights: None,
            limit: None,
            intermediate,
            ..
        }) if packed == product.y && intermediate.checked_mul(2) == Some(product.rows) => {
            Some(intermediate)
        }
        _ => None,
    }
}

/// The epilogue's width over `run` (a matmul then its ops) and its layout:
/// the product's own, or a packed GLU's halves where the run's first op
/// reads it so.
fn layout_of(trace: &Trace, run: &[u32], product: &Product) -> (u32, Layout) {
    match run.get(1).and_then(|&n| glu_of(trace, n, product)) {
        Some(half) => (half, Layout::Packed(2)),
        None => (product.rows, Layout::Dense),
    }
}

/// Whether `node` extends `run`: under the matmul's guard, reading what the
/// run computes, and an op over the epilogue's rows (or the packed
/// product's GLU, read straight off it).
fn joins(trace: &Trace, run: &[u32], node: u32, product: &Product) -> bool {
    let first = &trace.nodes[run[0] as usize];
    if trace.nodes[node as usize].guard != first.guard || !chain::reads_from(trace, run, node) {
        return false;
    }
    if run.len() == 1 && glu_of(trace, node, product).is_some() {
        return true;
    }
    let (width, _) = layout_of(trace, run, product);
    let rows = chain::Rows {
        dim: product.dim,
        width: u64::from(width),
    };
    match &trace.nodes[node as usize].op {
        Operation::Linear(Linear::GeluTanh { x, y } | Linear::Relu2 { x, y }) => [x, y].iter().all(
            |v| matches!(chain::form(trace, **v), Some(Form::Rows { dim, width }) if dim == rows.dim && width == rows.width),
        ),
        _ => chain::row_of(trace, node) == Some(rows),
    }
}

/// The anchor `run` (the matmul, then its epilogue's ops) lowers to, or
/// `None` where an op of it does not lower, an input is not one of the
/// product tile's, or the product is read past the run.
fn lower(trace: &Trace, run: &[u32], product: &Product, escapes: &Escapes) -> Option<Anchor> {
    if escapes.leaves(product.y, run) {
        return None;
    }
    let (width, layout) = layout_of(trace, run, product);
    let parts = layout.parts();
    let epilogue = &run[1..];

    // The inputs: what the epilogue reads and does not compute, in
    // first-read order, each a value of the product tile's.
    let mut computed: Vec<ValueId> = vec![product.y];
    let mut ins: Vec<(ValueId, Input)> = Vec::new();
    let mut ids = Vec::new();
    for &node in epilogue {
        let op = &trace.nodes[node as usize].op;
        for read in chain::reads_of(op, width) {
            let v = read.value;
            if computed.contains(&v) || ins.iter().any(|(held, _)| *held == v) {
                continue;
            }
            if read.by.is_some() || read.from > 0 || read.half {
                return None;
            }
            let input = match chain::form(trace, v)? {
                Form::Rows { dim, width: w } if dim == product.dim && w == u64::from(width) => {
                    Input::Tile
                }
                Form::Shared { width: w } if w == u64::from(width) => Input::Column,
                Form::One => Input::One,
                Form::Each => Input::Row,
                _ => return None,
            };
            ins.push((v, input));
        }
        ids.clear();
        op.outputs(&mut ids);
        computed.extend_from_slice(&ids);
    }

    let mut b = Builder::new(parts + u32::try_from(ins.len()).ok()?, width);
    if parts == 1 {
        b.of.insert(product.y, 0);
    } else {
        for p in 0..parts {
            b.parts.insert((product.y, p * width), p);
        }
    }
    for (k, (v, _)) in (parts..).zip(&ins) {
        b.of.insert(*v, k);
    }
    for &node in epilogue {
        b.node(trace, &trace.nodes[node as usize].op)?;
    }

    let outs: Vec<ValueId> = computed[1..]
        .iter()
        .copied()
        .filter(|&v| escapes.leaves(v, run))
        .collect();
    let results = outs
        .iter()
        .map(|v| b.get(*v))
        .collect::<Option<Vec<u32>>>()?;
    let mut rounds: Vec<(u32, Dtype)> = Vec::new();
    for &v in &computed[1..] {
        let narrow = chain::dtype_of(trace, v);
        if !matches!(narrow, Dtype::Bf16 | Dtype::F16) {
            continue;
        }
        let id = b.get(v)?;
        if id >= b.b.params && !rounds.iter().any(|(held, _)| *held == id) {
            rounds.push((id, narrow));
        }
    }
    let params = (0..parts)
        .map(|_| Shape::Row)
        .chain(ins.iter().map(|(_, input)| input.shape()))
        .collect();
    let region = Region {
        width,
        params,
        ops: b.b.ops,
        results,
        rounds,
    };
    // The op lands each result on a plane apart from every one it reads.
    let held = |v: ValueId| In {
        value: v,
        dtype: chain::dtype_of(trace, v),
        shared: false,
        one: false,
        each: false,
        by: None,
        from: 0,
    };
    let reads: Vec<In> = [product.act]
        .into_iter()
        .chain(ins.iter().map(|(v, _)| *v))
        .map(held)
        .collect();
    let lands: Vec<Out> = outs
        .iter()
        .map(|&v| Out {
            value: v,
            dtype: chain::dtype_of(trace, v),
        })
        .collect();
    if !chain::lands_apart_of(trace, run, &reads, &lands) {
        return None;
    }
    let anchored = Anchored {
        contraction: Contraction {
            n: width,
            k: product.k,
            layout,
            quant: Quant::Plain(Dtype::Bf16),
            acc: Some(Dtype::Bf16),
        },
        prologue: None,
        epilogue: Some(Pointwise {
            region,
            inputs: ins.iter().map(|(_, input)| *input).collect(),
        }),
    };
    Some(Anchor {
        nodes: run.to_vec(),
        act: product.act,
        w: product.w,
        ins: ins.into_iter().map(|(v, _)| v).collect(),
        outs,
        anchored,
    })
}
