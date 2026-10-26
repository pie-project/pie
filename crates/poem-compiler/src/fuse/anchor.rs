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
    Anchored, Binary, CacheDtype, CachePlane, Contraction, Gain, Input, Layout, Op, Paged,
    Pointwise, Quant, Region, RowNorm, Shape, Window,
};
use poem_ir::{
    Attn, CacheRow, Def, Dim, Dtype, Elemwise, Fused, Gather, Linear, NormAxis, NormWeight,
    Operands, Operation, Trace, Ty, ValueId,
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
    /// Where it lands results in a paged cache (a KV append's k and v):
    /// that cache. Its paged results name no value of `outs`.
    pub cache: Option<ValueId>,
    /// Its product landed apart by the first of `nodes`, which stays: the
    /// anchor is that product's epilogue (`fused.epilogue`), `act` and `w`
    /// the product op's.
    pub landed: bool,
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
        if anchor.landed {
            // the product's own op stays; the epilogue reads what it lands
            let y = landed_y(&trace, anchor.nodes[0]);
            folded.push((
                anchor.nodes[1..].to_vec(),
                Fused::Epilogue {
                    y,
                    ins: anchor.ins,
                    key,
                    outs: anchor.outs,
                    cache: anchor.cache,
                },
            ));
            anchored.entry(key).or_insert(anchor.anchored);
            continue;
        }
        folded.push((
            anchor.nodes,
            Fused::Anchored {
                act: anchor.act,
                w: anchor.w,
                ins: anchor.ins,
                key,
                outs: anchor.outs,
                gain: anchor.gain,
                cache: anchor.cache,
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

/// The product node `at` lands apart ([`landed_of`]).
fn landed_y(trace: &Trace, at: u32) -> ValueId {
    landed_of(trace, at).map_or(ValueId(0), |p| p.y)
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
        let Some(product) = product_of(trace, at).or_else(|| landed_of(trace, at)) else {
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
            // A product landed apart is anchored only where its epilogue lands
            // a KV append: the rest is the row chains'.
            if product.landed && anchor.cache.is_none() {
                return None;
            }
            // A sum across the whole row (a row one head wide, its norm) runs
            // on the row skeleton alone: an anchor takes one only where it
            // lands a KV append (a shared-kv latent's k row), and keeps its
            // contraction skeletons elsewhere.
            let reduces = anchor
                .anchored
                .epilogue
                .as_ref()
                .is_some_and(|e| e.region.ops.iter().any(|op| matches!(op, Op::Reduce(..))));
            if reduces && anchor.cache.is_none() {
                return None;
            }
            if let Some(folds) = folds
                && !product.landed
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
    /// Landed apart by its own op (a quantized bank's, a projection's half):
    /// its anchor is that op's epilogue alone (`fused.epilogue`).
    landed: bool,
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
        landed: false,
    })
}

/// The product node `at` lands where no contraction skeleton holds it but
/// an epilogue may read it as stored: bf16 rows of a matmul of any bank (a
/// quantized one's), or the left half `matmul_split_rows` lands (Muse
/// Spark's `qkv` of its `qkv | gate`).
fn landed_of(trace: &Trace, at: u32) -> Option<Product> {
    let (act, w, y) = match trace.nodes[at as usize].op {
        Operation::Linear(Linear::Matmul {
            act,
            w,
            y,
            logits: false,
            ..
        }) => (act, w, y),
        Operation::Fused(Fused::MatmulSplitRows { act, w, left, .. }) => (act, w, left),
        _ => return None,
    };
    let rows_of = |v: ValueId| match &trace.values[v.0 as usize].ty {
        Ty::Tensor { shape, dtype } => match shape.as_slice() {
            [dim, Dim::Const(width)] if !matches!(dim, Dim::Const(_)) => {
                Some((*dim, *width, *dtype))
            }
            _ => None,
        },
        Ty::Struct(_) => None,
    };
    let (dim, k, _) = rows_of(act)?;
    let (out_dim, rows, dtype) = rows_of(y)?;
    (out_dim == dim && dtype == Dtype::Bf16).then_some(Product {
        act,
        w,
        y,
        dim,
        rows: u32::try_from(rows).ok()?,
        k: u32::try_from(k).ok()?,
        landed: true,
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
    full: &[ValueId],
    windows: &BTreeMap<ValueId, Window>,
) -> Option<Vec<(ValueId, Window)>> {
    let whole = |v: ValueId| {
        if full.contains(&v) {
            Some(Window::of(0, product.rows))
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
            if !at.whole() || q_width.checked_add(kv_width.checked_mul(2)?)? != at.width {
                return None;
            }
            let piece = |from: u32, width: u32| Window::of(at.from + from, width);
            Some(vec![
                (q, piece(0, q_width)),
                (k, piece(q_width, kv_width)),
                (v, piece(q_width + kv_width, kv_width)),
            ])
        }
        // q and its gate interleaved a head (Qwen 3.5's): each a head of every
        // two, head-major windows of the piece
        Operation::Gather(Gather::SplitQGate {
            packed,
            head_dim,
            q,
            gate,
        }) => {
            let at = whole(packed)?;
            let d = head_dim;
            if !at.whole() || d == 0 || !at.width.is_multiple_of(2 * d) {
                return None;
            }
            let width = at.width / 2;
            Some(vec![
                (q, Window::strided(at.from, width, d, 2 * d)),
                (gate, Window::strided(at.from + d, width, d, 2 * d)),
            ])
        }
        Operation::Gather(Gather::SplitRows {
            x,
            width,
            left,
            right,
        }) => {
            let at = whole(x)?;
            if !at.whole() || width == 0 || width >= at.width {
                return None;
            }
            Some(vec![
                (left, Window::of(at.from, width)),
                (right, Window::of(at.from + width, at.width - width)),
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
    let mut full = vec![product.y];
    let (mut ins, mut outs) = (Vec::new(), Vec::new());
    for &node in &run[1..] {
        if let Some(pieces) = split_of(trace, node, product, &full, &windows) {
            windows.extend(pieces);
            continue;
        }
        let op = &trace.nodes[node as usize].op;
        // Before the first split, the run's values are the whole tile's (a
        // bias over the product, the split then cutting its sum).
        if windows.is_empty() {
            outs.clear();
            op.outputs(&mut outs);
            full.extend(outs.iter().copied().filter(|&v| is_full(trace, v, product)));
            continue;
        }
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

/// Whether `v` is a row of the whole product tile: as wide as every column
/// of a dense product.
fn is_full(trace: &Trace, v: ValueId, product: &Product) -> bool {
    matches!(chain::form(trace, v), Some(Form::Rows { dim, width })
        if dim == product.dim && width == u64::from(product.rows))
}

/// The values of the whole product tile `run` computes before any split:
/// the product, and what each op of it lands as wide.
fn full_of(trace: &Trace, run: &[u32], product: &Product) -> Vec<ValueId> {
    let mut full = vec![product.y];
    let mut outs = Vec::new();
    for &node in &run[1..] {
        outs.clear();
        trace.nodes[node as usize].op.outputs(&mut outs);
        full.extend(outs.iter().copied().filter(|&v| is_full(trace, v, product)));
    }
    full
}

/// Whether a rotation's stated frequencies (`freqs`, none: `theta`'s) are
/// one a pair it turns, where no ramp stretches them too.
fn table_fits(turn: &kernels_ir::build::Turn, freqs: Option<&poem_ir::RopeFreqs>) -> bool {
    let Some(freqs) = freqs else {
        return true;
    };
    let pairs = match turn.pairs {
        kernels_ir::build::Pairs::Halves { rotary, .. }
        | kernels_ir::build::Pairs::Tail { rotary, .. } => rotary / 2,
    };
    turn.ramp.is_none()
        && turn.mscale.is_none()
        && freqs.inv_freq.len() == pairs as usize
        && freqs.mscale.is_finite()
}

/// Whether `node`, an op over the windows `windows`, reads them as one op
/// over each head of its own piece: every piece it reads one (the pair a
/// rotation turns, q's and k's, each its own), of whole heads where it
/// names one, and the op pointwise or a head's alone (a head's norm, a
/// rotation), so that the epilogue computes it over the whole tile and
/// lands its piece.
fn over_pieces(
    trace: &Trace,
    node: u32,
    product: &Product,
    windows: &BTreeMap<ValueId, Window>,
) -> bool {
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
            ref freqs,
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
                && chain::rotation(head_dim, head_dim, theta, &turn)
                    .is_some_and(|t| table_fits(&t, freqs.as_ref()))
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
        Operation::Attn(Attn::KvAppend { .. }) => {
            append_of(trace, node, product, windows).is_some()
        }
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
    let windows = windows_of(trace, run, product);
    if windows.is_empty() {
        // a split of the product, or of what the run lands of its whole tile
        if layout_of(trace, run, product).1 == Layout::Dense
            && split_of(
                trace,
                node,
                product,
                &full_of(trace, run, product),
                &windows,
            )
            .is_some()
        {
            return true;
        }
        // a KV append of the whole tile (a shared-kv latent's k row)
        if layout_of(trace, run, product).1 == Layout::Dense
            && append_of(trace, node, product, &windows).is_some()
        {
            return true;
        }
    } else {
        return split_of(trace, node, product, &[], &windows).is_some()
            || over_pieces(trace, node, product, &windows);
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
    // A KV append the run ends in: its k and v paged into the cache.
    let append = epilogue
        .iter()
        .find_map(|&node| append_of(trace, node, product, &windows));

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
            // the cache a paged store lands in, no value the epilogue reads
            if append.is_some_and(|a| a.cache == v) {
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
            Operation::Gather(Gather::SplitQGate {
                packed, q, gate, ..
            }) => {
                let whole = b.get(packed)?;
                for piece in [q, gate] {
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
                ref freqs,
                q_out,
                k_out: Some(k_out),
            }) => {
                let mut turn = chain::rotation(width, head_dim, theta, turn)?;
                // frequencies stated outright: each pair's, cos and sin
                // carrying their scale
                if !table_fits(&turn, freqs.as_ref()) {
                    return None;
                }
                let table = freqs.as_ref().map(|f| f.inv_freq.as_slice());
                if let Some(f) = freqs
                    && f.mscale != 1.0
                {
                    turn.mscale = Some(turn.mscale.unwrap_or(1.0) * f.mscale);
                }
                let (xq, xk) = (b.get(q)?, b.get(k)?);
                let x = if xq == xk {
                    xq
                } else {
                    let col = b.b.push(Op::Column);
                    let edge = b.b.u32(windows.get(&k)?.from);
                    let in_q = b.b.push(Op::Binary(Binary::Lt, col, edge));
                    b.b.push(Op::Select(in_q, xq, xk))
                };
                let turned = b.b.rotate_with(x, b.get(positions)?, &turn, table);
                b.of.insert(q_out, turned);
                b.of.insert(k_out, turned);
            }
            // its k and v land as paged results, below
            Operation::Attn(Attn::KvAppend { .. }) if append.is_some() => {}
            _ => b.node(trace, op)?,
        }
    }

    let outs: Vec<ValueId> = computed[1..]
        .iter()
        .copied()
        .filter(|&v| escapes.leaves(v, run))
        .collect();
    // The paged results past them: the append's k into the cache's keys,
    // its v into its values.
    let input_of = |v: ValueId| {
        ins.iter()
            .position(|(held, _)| *held == v)
            .and_then(|at| u32::try_from(at).ok())
    };
    let mut paged: Vec<(ValueId, Paged)> = Vec::new();
    if let Some(a) = append {
        let (page, slot) = (input_of(a.page)?, input_of(a.slot)?);
        let land = |plane| Paged {
            plane,
            dtype: a.dtype,
            page,
            slot,
        };
        paged.push((a.k, land(CachePlane::Keys)));
        if let Some(v) = a.v {
            paged.push((v, land(CachePlane::Values)));
        }
    }
    let results = outs
        .iter()
        .chain(paged.iter().map(|(v, _)| v))
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
            .chain(paged.iter().map(|(v, _)| v))
            .map(|v| windows.get(v).copied().unwrap_or(Window::of(0, width)))
            .collect()
    };
    let paged: Vec<Option<Paged>> = if paged.is_empty() {
        Vec::new()
    } else {
        outs.iter()
            .map(|_| None)
            .chain(paged.iter().map(|(_, p)| Some(*p)))
            .collect()
    };
    let anchored = Anchored {
        norm: None,
        paged,
        windows: cut,
        contraction: Contraction {
            n: width,
            k: product.k,
            layout,
            quant: Quant::Plain(Dtype::Bf16),
            // The product keeps its accumulator through the epilogue, as a
            // matmul and its bias are one sum: no op between them rounds it;
            // one landed apart is read as its op stored it.
            acc: product.landed.then_some(Dtype::Bf16),
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
        cache: append.map(|a| a.cache),
        landed: product.landed,
    })
}

/// A KV append a windowed run ends in, as its anchor lands it: k (and v)
/// paged into `cache`'s planes at the rows `page` and `slot` name.
#[derive(Debug, Clone, Copy)]
struct Append {
    k: ValueId,
    v: Option<ValueId>,
    cache: ValueId,
    page: ValueId,
    slot: ValueId,
    dtype: CacheDtype,
}

/// The KV append node `node` is, where an anchor's paged store lands it: k
/// and v pieces of the product (`windows`) as wide as the cache's planes, a
/// cache of bf16 or e4m3 (a unit scale, vLLM's `fp8`) holding whole heads,
/// and the slot mapping one i32 a row.
/// The block-scaled rows kernels-ir lands of a trace's declared format.
#[must_use]
pub fn block_rows(format: poem_ir::KvRows) -> kernels_ir::BlockRows {
    match format {
        poem_ir::KvRows::V4 => kernels_ir::BlockRows::V4,
        poem_ir::KvRows::Mx8 => kernels_ir::BlockRows::MX8,
        poem_ir::KvRows::Nv4 => kernels_ir::BlockRows::NV4,
    }
}

fn append_of(
    trace: &Trace,
    node: u32,
    product: &Product,
    windows: &BTreeMap<ValueId, Window>,
) -> Option<Append> {
    let Operation::Attn(Attn::KvAppend {
        k,
        v,
        cache,
        write_page,
        write_offset,
    }) = trace.nodes[node as usize].op
    else {
        return None;
    };
    let Def::Cache(row) = trace.values[cache.0 as usize].def else {
        return None;
    };
    let Some(CacheRow::Kv {
        planes,
        dtype,
        live: None,
        rows,
        ..
    }) = trace.caches.get(row as usize)
    else {
        return None;
    };
    let (dtype, blocks) = match (dtype, rows) {
        (Dtype::Bf16, None) => (CacheDtype::Bf16, None),
        (Dtype::E4m3, None) => (CacheDtype::E4m3 { scale: 1.0 }, None),
        // DeepSeek's rows: a u8 plane a row of the format's bytes
        (Dtype::U8, Some(format)) => {
            let rows = block_rows(*format);
            (CacheDtype::Blocks(rows), Some(rows))
        }
        _ => return None,
    };
    // a piece of the product, or (no split) its whole tile
    let piece = |x: ValueId| {
        windows.get(&x).copied().filter(|w| w.whole()).or_else(|| {
            (windows.is_empty() && is_full(trace, x, product))
                .then_some(Window::of(0, product.rows))
        })
    };
    let kw = piece(k)?;
    // a plane of the window's columns, or of a block-scaled row's bytes
    // where the window is the row's columns
    let wide = |w: Window, plane: usize| match blocks {
        None => planes.get(plane) == Some(&u64::from(w.width)),
        Some(rows) => {
            w.width == rows.width() && planes.get(plane) == Some(&u64::from(rows.row_bytes()))
        }
    };
    let laid = match v {
        Some(v) => wide(kw, 0) && piece(v).is_some_and(|vw| wide(vw, 1)),
        None => planes.len() == 1 && wide(kw, 0),
    };
    let each = |x: ValueId| matches!(chain::form(trace, x), Some(Form::Each));
    (laid && each(write_page) && each(write_offset)).then_some(Append {
        k,
        v,
        cache,
        page: write_page,
        slot: write_offset,
        dtype,
    })
}
