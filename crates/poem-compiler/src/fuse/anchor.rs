//! Anchored regions: a matmul and the pointwise ops over its product tile,
//! lowered to one kernels-ir [`Anchored`] (a contraction with an epilogue)
//! that a backend's contraction skeletons fill their hooks with. The peer of
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
//! ones. And a split of the product into pieces (a packed qkv row's q, k
//! and v, a row's two halves) and the ops over each piece's heads (a head's
//! norm, q's and k's turn): each piece computed over the whole tile and
//! landed in its window. Of the run, the longest leading part whose
//! anchored region kernels-ir verifies (pointwise over the product tile, or
//! combining columns only within a head's span: no sum across the row, no
//! gather past a head) and the backend serves (`serves`) is the anchor's; a
//! run of no op is no anchor, and the matmul keeps its own kernel.
//!
//! **Lowering.** The product's parts are the epilogue's first parameters;
//! each value it reads and does not compute is an input, by how it is laid
//! out: a tile of the product's rows ([`Input::Tile`], a residual), a
//! vector a column ([`Input::Column`], a bias), a head's vector every head
//! reads ([`Input::Head`], a head's norm weight), one element a row
//! ([`Input::Row`], a position) or one value ([`Input::One`]). The product
//! is no value (nothing past the run reads it, or the run is no anchor) and
//! keeps its accumulator, f32, through the epilogue: a matmul's product and
//! the bias or activation over it are one computation, as `nn.Linear` and
//! every reference keep them, the bf16 the trace states of the product a
//! store the fusion leaves out. Each value the epilogue's ops compute rounds to
//! its dtype where it is defined, as a chain's do; one anything past the
//! run may read is a result.
//!
//! **Folding** ([`fold`]) replaces each anchor's ops with one
//! `fused.anchored` op, landed where its last op was, as a chain's region
//! lands, and hands back each anchored region by key. Anchors fold before
//! the row chains do, so a product's epilogue is the contraction's, and the
//! chains form over what is left.
//!
//! **Norms** ([`anchors_with`], `compiler.fold_norms`): an anchor whose
//! operand is a row's RMS norm, its normed row read by the product alone,
//! takes the norm in, rewritten as a scale of each row's products
//! (`rmsnorm(x, g) · wᵀ = r ⊙ (x · (g ⊙ w)ᵀ)`, kernels-ir's
//! `Anchored::normed`): its operand the norm's input (the residual stream,
//! still written for whatever else reads it), the scale the epilogue's
//! `Input::Norm`, the contraction's own as it streams each row, and the
//! gain folded into the bank once at load, in place, as `fold_bank`
//! declares (the bank read by this product alone and held whole on the
//! device, `Folds`). The fold's extra rounding of each weight is the
//! rewrite's, held to the unfolded program within each side's rounding.
//!
//! Anchors are formed and folded only under `compiler.synth_anchors`.

use std::collections::BTreeMap;

use kernels_ir::{
    Anchored, Binary, Contraction, Gain, Input, Layout, Op, Pointwise, Quant, Region, RowNorm,
    Shape, Window,
};
use poem_ir::{
    Def, Dim, Dtype, Elemwise, Fused, Gather, Linear, NormAxis, NormWeight, Operands, Operation,
    Trace, Ty, ValueId,
};

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
    /// Where the anchor norms its operand's rows: the norm's gain, which a
    /// load folds into `w` (`act` then the norm's input).
    pub gain: Option<ValueId>,
}

/// Whether a backend serves an anchored region at every count of rows a
/// fire may hand it, its inputs and results stored in the dtypes given (an
/// anchored region states no storage).
pub type Serves<'a> = &'a dyn Fn(&Anchored, &[Dtype], &[Dtype]) -> bool;

/// Whether a load may fold a norm's gain into the bank `w` in place: a
/// plane it holds whole on the device for the life of the load (not one
/// streamed from the host, whose bytes it would serve unfolded).
pub type Folds<'a> = &'a dyn Fn(ValueId) -> bool;

/// `trace` with each anchor `serves` takes replaced by one `fused.anchored`
/// op, and the anchored regions its ops name, by key
/// ([`Anchored::key`]).
#[must_use]
pub fn fold(trace: Trace, serves: Serves<'_>) -> (Trace, BTreeMap<u64, Anchored>) {
    fold_with(trace, serves, None)
}

/// [`fold`], and where `folds` is given, each anchor over an RMS-normed
/// operand whose bank it takes folded with its norm ([`anchors_with`]).
#[must_use]
pub fn fold_with(
    trace: Trace,
    serves: Serves<'_>,
    folds: Option<Folds<'_>>,
) -> (Trace, BTreeMap<u64, Anchored>) {
    let mut anchored = BTreeMap::new();
    let mut folded: Vec<(Vec<u32>, Fused)> = Vec::new();
    for anchor in anchors_with(&trace, serves, folds) {
        let key = anchor.anchored.key();
        folded.push((
            anchor.nodes,
            Fused::Anchored {
                act: anchor.act,
                w: anchor.w,
                ins: anchor.ins,
                key,
                outs: anchor.outs,
                gain: anchor.gain,
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
    anchors_with(trace, serves, None)
}

/// [`anchors`], and where `folds` is given, each one whose operand is an RMS norm's
/// row the norm's too ([`normed_of`]): the norm rewritten as a scale of each
/// row's products (`rmsnorm(x, g) · wᵀ = r ⊙ (x · (g ⊙ w)ᵀ)`), its gain
/// folded into the bank at load, the norm's op gone. Only where the anchor
/// has an epilogue already, the scale one more op of it: over a bare
/// product, the unfused leg it would add outweighs the norm it saves at a
/// prefill's rows (an L40S's 4096 x 4096 bank, 6 to 9% past 256 rows),
/// while over an epilogue the fold ties or wins at every row (a qkv's, a
/// SwiGLU's). The backend's tuner picks how the folded region runs at each
/// shape; the fold itself is this rule's.
#[must_use]
pub fn anchors_with(trace: &Trace, serves: Serves<'_>, folds: Option<Folds<'_>>) -> Vec<Anchor> {
    let escapes = Escapes::of(trace);
    let mut readers = vec![0u32; trace.values.len()];
    if folds.is_some() {
        let mut ins = Vec::new();
        for node in &trace.nodes {
            ins.clear();
            node.op.inputs(&mut ins);
            for v in &ins {
                readers[v.0 as usize] += 1;
            }
        }
    }
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
            let held = |anchor: &Anchor| {
                anchor.anchored.verify().is_ok()
                    && serves(
                        &anchor.anchored,
                        &stored(&anchor.ins),
                        &stored(&anchor.outs),
                    )
            };
            // Its operand's norm too, where the norm is the anchor's and the
            // backend serves it so; else the anchor over the normed rows.
            if let Some(folds) = folds
                && folds(product.w)
                && let Some(normed) = normed_of(trace, at, &product, &readers, &escapes)
                    .filter(|n| !taken[n.node as usize])
                    .and_then(|n| with_norm(trace, &anchor, n))
                && held(&normed)
            {
                return Some(normed);
            }
            held(&anchor).then_some(anchor)
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

/// An RMS norm whose normed row is a product's operand, as an anchor folds
/// it ([`normed_of`]).
#[derive(Debug, Clone, Copy)]
struct Normed {
    node: u32,
    x: ValueId,
    norm: RowNorm,
    gain: Option<ValueId>,
}

/// The norm whose row the matmul `at` reads, where an anchor may fold it: a
/// row's RMS norm under the matmul's guard, its normed row bf16 and read by
/// this matmul alone (so it needs no plane), its input bf16 rows of the
/// product's (still written where anything else reads it: the residual
/// stream), its gain none or a vector a column (bf16 or f32); and the
/// product's bank read by this matmul alone, a load folding the gain into
/// it in place.
fn normed_of(
    trace: &Trace,
    at: u32,
    product: &Product,
    readers: &[u32],
    escapes: &Escapes,
) -> Option<Normed> {
    let Def::Op(node) = trace.values[product.act.0 as usize].def else {
        return None;
    };
    let Operation::Elemwise(Elemwise::Rmsnorm {
        x,
        weight,
        axis: NormAxis::Row,
        eps,
        y,
    }) = trace.nodes[node as usize].op
    else {
        return None;
    };
    let alone = |v: ValueId| readers.get(v.0 as usize) == Some(&1);
    if y != product.act
        || !alone(y)
        || escapes.leaves(y, &[at])
        || !alone(product.w)
        || trace.nodes[node as usize].guard != trace.nodes[at as usize].guard
        || chain::dtype_of(trace, y) != Dtype::Bf16
        || chain::dtype_of(trace, x) != Dtype::Bf16
    {
        return None;
    }
    let rows = matches!(chain::form(trace, x), Some(Form::Rows { dim, width })
        if dim == product.dim && width == u64::from(product.k));
    let (gain, g) = match weight {
        NormWeight::None => (Gain::None, None),
        NormWeight::Scale(g) => (Gain::Scale, Some(g)),
        NormWeight::PlusOne(g) => (Gain::PlusOne, Some(g)),
    };
    let laid = g.is_none_or(|g| {
        matches!(chain::form(trace, g), Some(Form::Shared { width }) if width == u64::from(product.k))
            && matches!(chain::dtype_of(trace, g), Dtype::Bf16 | Dtype::F32)
    });
    (rows && laid).then_some(Normed {
        node,
        x,
        norm: RowNorm { eps, gain },
        gain: g,
    })
}

/// `anchor` with its operand's norm folded in ([`Anchored::normed`]): its
/// operand the norm's input, the norm's op its first, its gain the load's
/// to fold into the bank.
fn with_norm(trace: &Trace, anchor: &Anchor, normed: Normed) -> Option<Anchor> {
    let anchored = anchor.anchored.normed(normed.norm)?;
    let mut nodes = anchor.nodes.clone();
    nodes.insert(0, normed.node);
    if !chain::keeps_order(trace, &nodes) {
        return None;
    }
    Some(Anchor {
        nodes,
        act: normed.x,
        anchored,
        gain: normed.gain,
        ..anchor.clone()
    })
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

/// The pieces `node` cuts a value of `windows` (the product itself, its
/// whole tile, where `windows` holds nothing) into, where it is a split:
/// a packed qkv row's q, k and v ([`Gather::SplitQkv`]), or a row's two
/// halves ([`Gather::SplitRows`], Muse Spark's `qkv | gate`), each a
/// window of the product's tile.
fn split_of(
    trace: &Trace,
    node: u32,
    product: &Product,
    windows: &BTreeMap<ValueId, Window>,
) -> Option<Vec<(ValueId, Window)>> {
    let whole = |v: ValueId| {
        if v == product.y {
            Some(Window {
                from: 0,
                width: product.rows,
            })
        } else {
            windows.get(&v).copied()
        }
    };
    match trace.nodes[node as usize].op {
        Operation::Gather(Gather::SplitQkv {
            packed,
            q_width,
            kv_width,
            q,
            k,
            v,
        }) => {
            let at = whole(packed)?;
            if q_width.checked_add(kv_width.checked_mul(2)?)? != at.width {
                return None;
            }
            let piece = |from: u32, width: u32| Window {
                from: at.from + from,
                width,
            };
            Some(vec![
                (q, piece(0, q_width)),
                (k, piece(q_width, kv_width)),
                (v, piece(q_width + kv_width, kv_width)),
            ])
        }
        Operation::Gather(Gather::SplitRows {
            x,
            width,
            left,
            right,
        }) => {
            let at = whole(x)?;
            if width == 0 || width >= at.width {
                return None;
            }
            Some(vec![
                (
                    left,
                    Window {
                        from: at.from,
                        width,
                    },
                ),
                (
                    right,
                    Window {
                        from: at.from + width,
                        width: at.width - width,
                    },
                ),
            ])
        }
        _ => None,
    }
}

/// The window of the product each value `run` computes covers, where the
/// run's first op splits it ([`split_of`]): a split's pieces (and a split
/// of a piece's), and what each op over one piece computes (a pair's
/// rotation, each of its two).
fn windows_of(trace: &Trace, run: &[u32], product: &Product) -> BTreeMap<ValueId, Window> {
    let mut windows = BTreeMap::new();
    if run
        .get(1)
        .is_none_or(|&n| split_of(trace, n, product, &windows).is_none())
    {
        return windows;
    }
    let (mut ins, mut outs) = (Vec::new(), Vec::new());
    for &node in &run[1..] {
        if let Some(pieces) = split_of(trace, node, product, &windows) {
            windows.extend(pieces);
            continue;
        }
        let op = &trace.nodes[node as usize].op;
        if let Operation::Elemwise(Elemwise::Rope {
            q,
            k: Some(k),
            q_out,
            k_out: Some(k_out),
            ..
        }) = *op
        {
            for (from, to) in [(q, q_out), (k, k_out)] {
                if let Some(&w) = windows.get(&from) {
                    windows.insert(to, w);
                }
            }
            continue;
        }
        ins.clear();
        outs.clear();
        op.inputs(&mut ins);
        op.outputs(&mut outs);
        if let Some(&w) = ins.iter().find_map(|v| windows.get(v)) {
            for v in &outs {
                windows.insert(*v, w);
            }
        }
    }
    windows
}

/// Whether `node`, an op over the windows `windows`, reads them as one op
/// over each head of its own piece: every piece it reads one (the pair a
/// rotation turns, q's and k's, each its own), of whole heads where it
/// names one, and the op pointwise or a head's alone (a head's norm, a
/// rotation), so that the epilogue computes it over the whole tile and
/// lands its piece.
fn over_pieces(trace: &Trace, node: u32, windows: &BTreeMap<ValueId, Window>) -> bool {
    let op = &trace.nodes[node as usize].op;
    let mut ins = Vec::new();
    op.inputs(&mut ins);
    let pieces: Vec<Window> = ins.iter().filter_map(|v| windows.get(v).copied()).collect();
    let Some(&piece) = pieces.first() else {
        return false;
    };
    let whole = |head: u32, w: &Window| {
        head > 0 && w.from.is_multiple_of(head) && w.width.is_multiple_of(head)
    };
    let rows = |w: Window| {
        chain::Rows {
            dim: match trace.values[ins[0].0 as usize].ty {
                Ty::Tensor { ref shape, .. } => shape[0],
                Ty::Struct(_) => return None,
            },
            width: u64::from(w.width),
        }
        .into()
    };
    match *op {
        Operation::Elemwise(Elemwise::Rope {
            q,
            k: Some(k),
            positions,
            head_dim,
            freqs: None,
            turn,
            theta,
            ..
        }) => {
            let (Some(qw), Some(kw)) = (windows.get(&q), windows.get(&k)) else {
                return false;
            };
            whole(head_dim, qw)
                && whole(head_dim, kw)
                && matches!(chain::form(trace, positions), Some(chain::Form::Each))
                && chain::rotation(head_dim, head_dim, theta, &turn).is_some()
        }
        Operation::Elemwise(Elemwise::Rmsnorm {
            axis: NormAxis::Head(head),
            ..
        }) => {
            pieces.iter().all(|w| *w == piece && whole(head, w))
                && chain::row_of(trace, node) == rows(piece)
        }
        Operation::Elemwise(
            Elemwise::Act { .. } | Elemwise::Binary { .. } | Elemwise::Clamp { .. },
        ) => pieces.iter().all(|w| *w == piece) && chain::row_of(trace, node) == rows(piece),
        _ => false,
    }
}

/// Whether `node` extends `run`: under the matmul's guard, reading what the
/// run computes, and an op over the epilogue's rows (or the packed
/// product's GLU, read straight off it; or a split of it into a packed
/// qkv row's pieces, and an op over each of its pieces' heads).
fn joins(trace: &Trace, run: &[u32], node: u32, product: &Product) -> bool {
    let first = &trace.nodes[run[0] as usize];
    if trace.nodes[node as usize].guard != first.guard || !chain::reads_from(trace, run, node) {
        return false;
    }
    if run.len() == 1 && glu_of(trace, node, product).is_some() {
        return true;
    }
    if run.len() == 1 && split_of(trace, node, product, &BTreeMap::new()).is_some() {
        return true;
    }
    let windows = windows_of(trace, run, product);
    if !windows.is_empty() {
        return split_of(trace, node, product, &windows).is_some()
            || over_pieces(trace, node, &windows);
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
    // A split's pieces: each value over one computed over the whole tile,
    // landing its piece.
    let windows = windows_of(trace, run, product);

    // The inputs: what the epilogue reads and does not compute, in
    // first-read order, each a value of the product tile's (or, read by an
    // op over a piece, one every row or every head reads alike).
    let mut computed: Vec<ValueId> = vec![product.y];
    let mut ins: Vec<(ValueId, Input)> = Vec::new();
    let mut ids = Vec::new();
    for &node in epilogue {
        let op = &trace.nodes[node as usize].op;
        let mut read_ids = Vec::new();
        op.inputs(&mut read_ids);
        let pieced = read_ids.iter().any(|v| windows.contains_key(v));
        // A head's norm weight, one head wide, every head reading it.
        let head = match *op {
            Operation::Elemwise(Elemwise::Rmsnorm {
                axis: NormAxis::Head(head),
                weight: NormWeight::Scale(w) | NormWeight::PlusOne(w),
                ..
            }) => Some((w, head)),
            _ => None,
        };
        for read in chain::reads_of(op, width) {
            let v = read.value;
            if computed.contains(&v) || ins.iter().any(|(held, _)| *held == v) {
                continue;
            }
            if read.by.is_some() || read.from > 0 || read.half {
                return None;
            }
            let input = match chain::form(trace, v)? {
                Form::Shared { width: w }
                    if head.is_some_and(|(weight, size)| weight == v && u64::from(size) == w)
                        && w < u64::from(width) =>
                {
                    Input::Head(u32::try_from(w).ok()?)
                }
                Form::Rows { dim, width: w }
                    if !pieced && dim == product.dim && w == u64::from(width) =>
                {
                    Input::Tile
                }
                Form::Shared { width: w } if !pieced && w == u64::from(width) => Input::Column,
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
        let op = &trace.nodes[node as usize].op;
        match *op {
            // Each piece is what it is cut from, over the whole tile.
            Operation::Gather(Gather::SplitQkv {
                packed, q, k, v, ..
            }) => {
                let whole = b.get(packed)?;
                for piece in [q, k, v] {
                    b.of.insert(piece, whole);
                }
            }
            Operation::Gather(Gather::SplitRows { x, left, right, .. }) => {
                let whole = b.get(x)?;
                for piece in [left, right] {
                    b.of.insert(piece, whole);
                }
            }
            // q's and k's heads turned as one row: each column its own
            // piece's value, one rotation over both.
            Operation::Elemwise(Elemwise::Rope {
                q,
                k: Some(k),
                positions,
                head_dim,
                theta,
                ref turn,
                freqs: None,
                q_out,
                k_out: Some(k_out),
            }) => {
                let turn = chain::rotation(width, head_dim, theta, turn)?;
                let (xq, xk) = (b.get(q)?, b.get(k)?);
                let x = if xq == xk {
                    xq
                } else {
                    let col = b.b.push(Op::Column);
                    let edge = b.b.u32(windows.get(&k)?.from);
                    let in_q = b.b.push(Op::Binary(Binary::Lt, col, edge));
                    b.b.push(Op::Select(in_q, xq, xk))
                };
                let turned = b.b.rotate(x, b.get(positions)?, &turn);
                b.of.insert(q_out, turned);
                b.of.insert(k_out, turned);
            }
            _ => b.node(trace, op)?,
        }
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
    // Each result's piece: a split's, or (beside one) the whole tile.
    let cut = if windows.is_empty() {
        Vec::new()
    } else {
        outs.iter()
            .map(|v| windows.get(v).copied().unwrap_or(Window { from: 0, width }))
            .collect()
    };
    let anchored = Anchored {
        norm: None,
        windows: cut,
        contraction: Contraction {
            n: width,
            k: product.k,
            layout,
            quant: Quant::Plain(Dtype::Bf16),
            // The product keeps its accumulator through the epilogue, as a
            // matmul and its bias are one sum: no op between them rounds it.
            acc: None,
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
        gain: None,
    })
}
