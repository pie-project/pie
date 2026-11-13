//! Mixture-of-experts: routers, expert selection matmuls, combines.
//!
//! Routes are an i32 `[tokens, top_k]` plane (expert id per slot; a negative
//! id is an unrouted slot) with an f32 `[tokens, top_k]` weight plane beside
//! it, exactly as the GPU engines land them. Expert banks are resident
//! (design §4): the matmuls read the whole bank plane and pick experts
//! in-graph.
//!
//! Routed matmuls read the bank as `crate::pack` lands it (codes one per
//! element in a native narrow type, or mxfp4 pre-scaled to e5m2) and read
//! only the experts a fire routes to where that is cheaper than the bank:
//! - one or two tokens' pairs: each pair's expert is dynamic-sliced out of
//!   the bank straight into its own dot (independent dots XLA overlaps);
//! - more pairs (decode at batch): the pairs are sorted by expert and a
//!   loop runs one dot per active expert (per row tile of it), two a step,
//!   over the rows routed to it; when more experts are active than the loop
//!   is worth (a tile's dot costs ~14 us however small the expert, v6e) the
//!   whole bank streams instead: every row against every expert in one dot
//!   behind a free convert (group factors, when the bank has them, scale
//!   the per-group partials), each pair's row picked out. The fire's routes
//!   pick between the two at run time (each runs in a loop whose trip count
//!   is zero when not taken);
//! - many rows (prefill): stable-sort the pairs by expert, gather the
//!   activation rows in that order, decode the whole bank to bf16 once and
//!   run one `chlo.ragged_dot`, then un-permute with the inverse sort.
//!
//! Where the bank is plain (bf16, or pre-scaled f8, no group factors) and a
//! block of it fits VMEM, the grouped-matmul kernel (`crate::linear::moe_gmm`)
//! is preferred over all of these past the slice case. `kernel.moe.path`
//! may force any formulation.
//!
//! The bytes floor is the active experts' bytes: at batch 64 with top-4 of
//! 32 experts and diverse tokens nearly every expert is active and the
//! whole bank streams (e5m2: ~0.55 ms a gpt-oss layer on v6e). An exact
//! 4-bit form does not help there: decoding e2m1 with its group scale
//! inside the dot's operand takes more VPU work per weight than streaming
//! e5m2 costs (measured: 0.8 ms for the gate/up bank as `i4` codes, 2 ms
//! as `f4E2M1FN`, 0.45 ms pre-scaled).
//!
//! References: kernels-slang `linear/moe/{route, select}.slang`; the sink
//! router from kernels-cuda `linear/moe/route.cu`.
#![allow(clippy::too_many_arguments)]

use dtype::Dtype;

use crate::check::expect;
use crate::error::{Error, refuse};
use crate::hlo::{Cmp, Elem, Fold, Func, GatherDims, Ty, Val};
use crate::linear::quant::{
    Codes, TILE, code_values, e8m0, grouped_dot, materialize, offset_dot, read_codes, tile_factor,
};
use crate::tensor::{Bank, Landing, Operand, Stored};
use crate::tuning::{MoePath, Tuning};

const MXFP4_BLOCK: u32 = 32;

fn nonzero(op: &'static str, what: &str, v: u32) -> Result<u32, Error> {
    if v == 0 {
        return Err(refuse(op, format!("`{what}` is zero")));
    }
    Ok(v)
}

fn planes(
    op: &'static str,
    rows: u32,
    top_k: u32,
    routes: impl Stored,
    weights: impl Stored,
) -> Result<(), Error> {
    expect(op, routes, &[Dtype::I32])?;
    expect(op, weights, &[Dtype::F32])?;
    if routes.rows() != rows
        || weights.rows() != rows
        || routes.width() != top_k
        || weights.width() != top_k
    {
        return Err(refuse(
            op,
            format!(
                "routes {}x{} / weights {}x{} for {rows} rows at fan-out {top_k}",
                routes.rows(),
                routes.width(),
                weights.rows(),
                weights.width()
            ),
        ));
    }
    Ok(())
}

fn router(
    op: &'static str,
    logits: impl Stored,
    experts: u32,
    top_k: u32,
    routes: impl Stored,
    weights: impl Stored,
) -> Result<(), Error> {
    expect(op, logits, &[Dtype::Bf16, Dtype::F32])?;
    nonzero(op, "the expert count", experts)?;
    nonzero(op, "the fan-out", top_k)?;
    if logits.width() != experts {
        return Err(refuse(
            op,
            format!(
                "the logits are {} wide for {experts} experts",
                logits.width()
            ),
        ));
    }
    planes(op, logits.rows(), top_k, routes, weights)
}

/// `v` as a flat f32 `[n]` vector.
fn flat_f32(f: &mut Func, v: Val) -> Result<Val, Error> {
    let n = f.ty(v).elements();
    let v = f.convert(v, Elem::F32);
    Ok(f.reshape(v, &[n])?)
}

/// The `k` largest of `rank` (`[rows, n]` f32) per row, ties to the lowest
/// index, NaN never before a number: `(values [rows, k], index [rows, k] i32)`.
/// `carry` (same shape as `rank`) rides the sort and is returned picked the
/// same way.
fn pick_top(
    f: &mut Func,
    rank: Val,
    carry: Option<Val>,
    k: i64,
) -> Result<(Val, Val, Option<Val>), Error> {
    let dims = f.dims(rank).to_vec();
    let (rows, n) = (dims[0], dims[1]);
    let clean = {
        let nan = f.compare(Cmp::Ne, rank, rank)?;
        let lo = f.const_f(Elem::F32, f64::NEG_INFINITY, &dims);
        f.select(nan, lo, rank)?
    };
    let iota = f.iota(Elem::I32, &dims, 1);
    let mut xs = vec![clean, iota];
    if let Some(c) = carry {
        xs.push(c);
    }
    let sorted = f.sort(&xs, 1, true, |fq, a, b| {
        let gt = fq.compare(Cmp::Gt, a[0], b[0])?;
        let eq = fq.compare(Cmp::Eq, a[0], b[0])?;
        let lt = fq.compare(Cmp::Lt, a[1], b[1])?;
        let tie = fq.and(eq, lt)?;
        fq.or(gt, tie)
    })?;
    let k = k.min(n);
    let v = f.slice(sorted[0], &[0, 0], &[rows, k], &[1, 1])?;
    let i = f.slice(sorted[1], &[0, 0], &[rows, k], &[1, 1])?;
    let c = match carry {
        Some(_) => Some(f.slice(sorted[2], &[0, 0], &[rows, k], &[1, 1])?),
        None => None,
    };
    Ok((v, i, c))
}

/// Widens `[rows, picks]` to `[rows, width]` with `fill` in the tail.
fn pad_cols(f: &mut Func, v: Val, width: i64, fill: Val) -> Result<Val, Error> {
    let picks = f.dims(v)[1];
    if picks == width {
        return Ok(v);
    }
    Ok(f.pad(v, fill, &[0, 0], &[0, width - picks], &[0, 0])?)
}

/// Per-row `sum(w)`, broadcast back over `[rows, k]`.
fn row_sum(f: &mut Func, w: Val) -> Result<Val, Error> {
    let dims = f.dims(w).to_vec();
    let s = f.reduce(w, &[1], Fold::Sum)?;
    Ok(f.broadcast(s, &dims, &[0])?)
}

/// `w * scaling`, or `w * scaling / sum(w)` when renormalizing and the row's
/// sum is positive (the WGSL routers' rule).
fn renorm(f: &mut Func, w: Val, renormalize: bool, scaling: f32) -> Result<Val, Error> {
    let scaled = f.scale(w, f64::from(scaling))?;
    if !renormalize {
        return Ok(scaled);
    }
    let sum = row_sum(f, w)?;
    let zero = f.like_f(sum, 0.0);
    let pos = f.compare(Cmp::Gt, sum, zero)?;
    let divided = f.div(scaled, sum)?;
    Ok(f.select(pos, divided, scaled)?)
}

fn softmax_router(
    f: &mut Func,
    op: &'static str,
    logits: Operand,
    scale: Option<Operand>,
    experts: u32,
    top_k: u32,
    routes: Landing,
    weights: Landing,
) -> Result<(Val, Val), Error> {
    router(op, logits, experts, top_k, routes, weights)?;
    if let Some(s) = scale {
        expect(op, s, &[Dtype::Bf16])?;
        if s.elements() != u64::from(experts) {
            return Err(refuse(
                op,
                format!(
                    "the gain holds {} entries for {experts} experts",
                    s.elements()
                ),
            ));
        }
    }
    let k = i64::from(top_k);
    let x = logits.f32(f);
    let (v, i, _) = top_k_fill(f, x, k)?;
    // Softmax over the chosen logits.
    let w = f.softmax(v, 1)?;
    let w = match scale {
        Some(s) => {
            let s = s.val;
            let s = flat_f32(f, s)?;
            let picked = take_flat(f, s, i)?;
            f.mul(w, picked)?
        }
        None => w,
    };
    Ok((i, w))
}

/// `top_k` for a fan-out that may exceed the row: picks past the row's end
/// hold `-inf`/index `-1` (they get zero softmax weight).
fn top_k_fill(f: &mut Func, x: Val, k: i64) -> Result<(Val, Val, Option<Val>), Error> {
    let (v, i, c) = pick_top(f, x, None, k)?;
    let lo = f.const_f(Elem::F32, f64::NEG_INFINITY, &[]);
    let none = f.const_i(Elem::I32, -1, &[]);
    let v = pad_cols(f, v, k, lo)?;
    let i = pad_cols(f, i, k, none)?;
    Ok((v, i, c))
}

/// `table[ids]` elementwise for a flat `[n]` table and any-shaped i32 `ids`
/// (clamped into range).
fn take_flat(f: &mut Func, table: Val, ids: Val) -> Result<Val, Error> {
    let n = f.dims(table)[0];
    let dims = f.dims(ids).to_vec();
    let lo = f.const_i(Elem::I32, 0, &[]);
    let hi = f.const_i(Elem::I32, n - 1, &[]);
    let ids = f.clamp(lo, ids, hi)?;
    let mut with = dims.clone();
    with.push(1);
    let ids = f.reshape(ids, &with)?;
    Ok(f.gather(
        table,
        ids,
        &GatherDims {
            offset_dims: vec![],
            collapsed_slice_dims: vec![0],
            start_index_map: vec![0],
            index_vector_dim: dims.len() as i64,
            ..GatherDims::default()
        },
        &[1],
    )?)
}

/// `x[r, ids[r, j]]` for `[rows, n]` `x` and `[rows, k]` i32 `ids` (clamped).
fn take_cols(f: &mut Func, x: Val, ids: Val) -> Result<Val, Error> {
    let (rows, n) = (f.dims(x)[0], f.dims(x)[1]);
    let k = f.dims(ids)[1];
    let flat = f.reshape(x, &[rows * n])?;
    let base = f.iota(Elem::I32, &[rows, k], 0);
    let stride = f.const_i(Elem::I32, n, &[rows, k]);
    let base = f.mul(base, stride)?;
    let lo = f.const_i(Elem::I32, 0, &[]);
    let hi = f.const_i(Elem::I32, n - 1, &[]);
    let ids = f.clamp(lo, ids, hi)?;
    let at = f.add(base, ids)?;
    take_flat(f, flat, at)
}

pub fn topk_softmax(
    f: &mut Func,
    logits: Operand,
    experts: u32,
    top_k: u32,
    routes: Landing,
    weights: Landing,
) -> Result<(Val, Val), Error> {
    softmax_router(
        f,
        "linear.moe_route",
        logits,
        None,
        experts,
        top_k,
        routes,
        weights,
    )
}

pub fn topk_softmax_scaled(
    f: &mut Func,
    logits: Operand,
    scale: Operand,
    experts: u32,
    top_k: u32,
    routes: Landing,
    weights: Landing,
) -> Result<(Val, Val), Error> {
    softmax_router(
        f,
        "linear.moe_route",
        logits,
        Some(scale),
        experts,
        top_k,
        routes,
        weights,
    )
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Score {
    Sigmoid,
    /// `sqrt(max(softplus(x), 0))`, softplus as `log(1 + e^x)` below 20.
    SqrtSoftplus,
}

fn score(f: &mut Func, x: Val, how: Score) -> Result<Val, Error> {
    Ok(match how {
        Score::Sigmoid => {
            // 1 / (1 + e^-x), as the routers compute it.
            let nx = f.neg(x);
            let e = f.exp(nx);
            let d = f.offset(e, 1.0)?;
            let one = f.like_f(d, 1.0);
            f.div(one, d)?
        }
        Score::SqrtSoftplus => {
            let e = f.exp(x);
            let sp = f.offset(e, 1.0)?;
            let sp = f.log(sp);
            let twenty = f.like_f(x, 20.0);
            let big = f.compare(Cmp::Gt, x, twenty)?;
            let sp = f.select(big, x, sp)?;
            let zero = f.like_f(sp, 0.0);
            let sp = f.max(sp, zero)?;
            f.sqrt(sp)
        }
    })
}

/// The sigmoid-family routers: rank by `score + bias`, weigh by `score`,
/// slots past the expert count land expert 0 at weight 0.
fn scored_router(
    f: &mut Func,
    op: &'static str,
    how: Score,
    logits: Operand,
    bias: Option<Operand>,
    experts: u32,
    top_k: u32,
    renormalize: bool,
    scaling: f32,
    routes: Landing,
    weights: Landing,
) -> Result<(Val, Val), Error> {
    router(op, logits, experts, top_k, routes, weights)?;
    if let Some(b) = bias {
        expect(op, b, &[Dtype::F32])?;
        if b.elements() != u64::from(experts) {
            return Err(refuse(
                op,
                format!(
                    "the correction bias holds {} entries for {experts} experts",
                    b.elements()
                ),
            ));
        }
    }
    let (rows, n, k) = (i64::from(logits.rows), i64::from(experts), i64::from(top_k));
    let x = logits.f32(f);
    let s = score(f, x, how)?;
    let rank = match bias {
        Some(b) => {
            let b = b.val;
            let b = flat_f32(f, b)?;
            let b = f.broadcast(b, &[rows, n], &[1])?;
            f.add(s, b)?
        }
        None => s,
    };
    let (_, i, w) = pick_top(f, rank, Some(s), k)?;
    let w = w.expect("the score rides the sort");
    let zero_f = f.const_f(Elem::F32, 0.0, &[]);
    let zero_i = f.const_i(Elem::I32, 0, &[]);
    let w = pad_cols(f, w, k, zero_f)?;
    let i = pad_cols(f, i, k, zero_i)?;
    let w = renorm(f, w, renormalize, scaling)?;
    Ok((i, w))
}

pub fn topk_sigmoid(
    f: &mut Func,
    logits: Operand,
    experts: u32,
    top_k: u32,
    renormalize: bool,
    scaling: f32,
    routes: Landing,
    weights: Landing,
) -> Result<(Val, Val), Error> {
    scored_router(
        f,
        "linear.moe_route",
        Score::Sigmoid,
        logits,
        None,
        experts,
        top_k,
        renormalize,
        scaling,
        routes,
        weights,
    )
}

pub fn topk_sigmoid_biased(
    f: &mut Func,
    logits: Operand,
    bias: Operand,
    experts: u32,
    top_k: u32,
    renormalize: bool,
    scaling: f32,
    routes: Landing,
    weights: Landing,
) -> Result<(Val, Val), Error> {
    scored_router(
        f,
        "linear.moe_route",
        Score::Sigmoid,
        logits,
        Some(bias),
        experts,
        top_k,
        renormalize,
        scaling,
        routes,
        weights,
    )
}

pub fn topk_sqrt_softplus(
    f: &mut Func,
    logits: Operand,
    bias: Operand,
    experts: u32,
    top_k: u32,
    renormalize: bool,
    scaling: f32,
    routes: Landing,
    weights: Landing,
) -> Result<(Val, Val), Error> {
    scored_router(
        f,
        "linear.moe_route",
        Score::SqrtSoftplus,
        logits,
        Some(bias),
        experts,
        top_k,
        renormalize,
        scaling,
        routes,
        weights,
    )
}

pub fn predict_route(
    f: &mut Func,
    logits: Operand,
    bias: Operand,
    experts: u32,
    top_k: u32,
    routes: Landing,
    weights: Landing,
) -> Result<(Val, Val), Error> {
    scored_router(
        f,
        "linear.moe_route",
        Score::SqrtSoftplus,
        logits,
        Some(bias),
        experts,
        top_k,
        false,
        1.0,
        routes,
        weights,
    )
}

/// Sigmoid top-k over the first `experts` scores, then `sink` always-on
/// experts (`experts..experts + sink`) appended; every weight is scaled by
/// `scaling * global_scale / (sum + 1e-20)`.
/// Reference: `kernels-cuda/kernels/linear/moe.cuh` `moe_topk_sigmoid_sink`
/// (CUDA only; `global_scale` is an f32 scalar, the bias an f32 `[experts]`).
pub fn topk_sigmoid_sink(
    f: &mut Func,
    logits: Operand,
    correction_bias: Option<Operand>,
    global_scale: Option<Operand>,
    experts: u32,
    top_k: u32,
    sink: u32,
    scaling: f32,
    routes: Landing,
    weights: Landing,
) -> Result<(Val, Val), Error> {
    const OP: &str = "linear.moe_route";
    expect(OP, logits, &[Dtype::Bf16])?;
    nonzero(OP, "the expert count", experts)?;
    nonzero(OP, "the fan-out", top_k)?;
    let width = experts
        .checked_add(sink)
        .ok_or_else(|| refuse(OP, "the expert count does not count"))?;
    if logits.width != width {
        return Err(refuse(
            OP,
            format!(
                "the router's row is {} wide and the statement names {experts} routed + {sink} sink experts",
                logits.width
            ),
        ));
    }
    planes(OP, logits.rows, top_k + sink, routes, weights)?;
    if let Some(b) = correction_bias {
        expect(OP, b, &[Dtype::F32])?;
        if b.elements() < u64::from(experts) {
            return Err(refuse(
                OP,
                "the correction bias is narrower than the expert count",
            ));
        }
    }
    if let Some(g) = global_scale {
        expect(OP, g, &[Dtype::F32])?;
    }
    let (rows, n, k, s) = (
        i64::from(logits.rows),
        i64::from(experts),
        i64::from(top_k),
        i64::from(sink),
    );
    let x = logits.f32(f);
    let sc = score(f, x, Score::Sigmoid)?;
    let routed = f.slice(sc, &[0, 0], &[rows, n], &[1, 1])?;
    let rank = match correction_bias {
        Some(b) => {
            let b = b.val;
            let b = flat_f32(f, b)?;
            let b = f.slice(b, &[0], &[n], &[1])?;
            let b = f.broadcast(b, &[rows, n], &[1])?;
            f.add(routed, b)?
        }
        None => routed,
    };
    let (_, i, w) = pick_top(f, rank, Some(routed), k)?;
    let w = w.expect("the score rides the sort");
    let zero_f = f.const_f(Elem::F32, 0.0, &[]);
    let none = f.const_i(Elem::I32, -1, &[]);
    let mut w = pad_cols(f, w, k, zero_f)?;
    let mut i = pad_cols(f, i, k, none)?;
    if s > 0 {
        let sw = f.slice(sc, &[0, n], &[rows, n + s], &[1, 1])?;
        let si = f.iota(Elem::I32, &[rows, s], 1);
        let base = f.const_i(Elem::I32, n, &[rows, s]);
        let si = f.add(si, base)?;
        w = f.concat(&[w, sw], 1)?;
        i = f.concat(&[i, si], 1)?;
    }
    let sum = row_sum(f, w)?;
    let sum = f.offset(sum, 1e-20)?;
    let mut scale = f.like_f(sum, f64::from(scaling));
    if let Some(g) = global_scale {
        let g = g.val;
        let g = flat_f32(f, g)?;
        let g = f.slice(g, &[0], &[1], &[1])?;
        let g = f.reshape(g, &[])?;
        let g = f.splat(g, &[rows, k + s])?;
        scale = f.mul(scale, g)?;
    }
    let scale = f.div(scale, sum)?;
    let w = f.mul(w, scale)?;
    Ok((i, w))
}

/// Routes by a token-id hash table: `tid2eid[id]` (i64 `[vocab, top_k]`)
/// names each slot's expert, weighed by `sqrt(softplus(logit))` of that
/// expert; an id past the vocabulary reads row 0, and a table entry outside
/// `0..experts` lands its (truncated) id at weight 0.
pub fn hash_route(
    f: &mut Func,
    ids: Operand,
    tid2eid: Operand,
    logits: Operand,
    vocab: u32,
    top_k: u32,
    renormalize: bool,
    scaling: f32,
    routes: Landing,
    weights: Landing,
) -> Result<(Val, Val), Error> {
    const OP: &str = "linear.moe_route";
    expect(OP, ids, &[Dtype::U32, Dtype::I32])?;
    expect(OP, tid2eid, &[Dtype::I64, Dtype::U32])?;
    // An i64 table may land as its u32 words, two a value.
    let words = if tid2eid.dtype == Dtype::U32 { 2 } else { 1 };
    expect(OP, logits, &[Dtype::Bf16])?;
    nonzero(OP, "the fan-out", top_k)?;
    nonzero(OP, "the vocabulary", vocab)?;
    let experts = nonzero(OP, "the expert count the logits span", logits.width)?;
    planes(OP, logits.rows, top_k, routes, weights)?;
    if ids.elements() != u64::from(routes.rows) {
        return Err(refuse(
            OP,
            format!("{} token ids for {} rows", ids.elements(), routes.rows),
        ));
    }
    if tid2eid.elements() < u64::from(vocab) * u64::from(top_k) * words {
        return Err(refuse(OP, "the hash table is smaller than vocab x top_k"));
    }
    let (rows, n, k, v) = (
        i64::from(routes.rows),
        i64::from(experts),
        i64::from(top_k),
        i64::from(vocab),
    );
    let raw = ids.val;
    let raw = f.reshape(raw, &[rows])?;
    // Unsigned compare against the vocabulary, as the WGSL reads a u32.
    let raw = f.convert(raw, Elem::I64);
    let raw = if ids.dtype == Dtype::I32 {
        let m = f.const_i(Elem::I64, 0xFFFF_FFFF, &[rows]);
        f.and(raw, m)?
    } else {
        raw
    };
    let vv = f.const_i(Elem::I64, v, &[rows]);
    let inside = f.compare(Cmp::Lt, raw, vv)?;
    let zero = f.const_i(Elem::I64, 0, &[rows]);
    let tid = f.select(inside, raw, zero)?;
    let tid = f.convert(tid, Elem::I32);
    let table = tid2eid.val;
    let total = f.ty(table).elements();
    let w = k * words as i64;
    let table = f.reshape(table, &[total / w, w])?;
    let eid = f.take_rows(table, tid)?; // [rows, k] i64, or its words
    let eid = if words == 2 {
        // Each value from its (low, high) words, little-endian.
        let eid = f.reshape(eid, &[rows, k, 2])?;
        let lo = f.slice_axis(eid, 2, 0, 1)?;
        let hi = f.slice_axis(eid, 2, 1, 2)?;
        let lo = f.reshape(lo, &[rows, k])?;
        let hi = f.reshape(hi, &[rows, k])?;
        let lo = f.convert(lo, Elem::I64);
        let hi = f.convert(hi, Elem::I64);
        let s = f.const_i(Elem::I64, 32, &[rows, k]);
        let hi = f.shl(hi, s)?;
        f.or(hi, lo)?
    } else {
        eid
    };
    let lo = f.const_i(Elem::I64, 0, &[rows, k]);
    let hi = f.const_i(Elem::I64, n, &[rows, k]);
    let ge = f.compare(Cmp::Ge, eid, lo)?;
    let lt = f.compare(Cmp::Lt, eid, hi)?;
    let ok = f.and(ge, lt)?;
    let route = f.convert(eid, Elem::I32);
    let x = logits.f32(f);
    let picked = take_cols(f, x, route)?;
    // The WGSL's hash score: sqrt(softplus) with no clamp at zero.
    let e = f.exp(picked);
    let sp = f.offset(e, 1.0)?;
    let sp = f.log(sp);
    let twenty = f.like_f(picked, 20.0);
    let big = f.compare(Cmp::Gt, picked, twenty)?;
    let sp = f.select(big, picked, sp)?;
    let w = f.sqrt(sp);
    let zf = f.const_f(Elem::F32, 0.0, &[rows, k]);
    let w = f.select(ok, w, zf)?;
    let w = renorm(f, w, renormalize, scaling)?;
    Ok((route, w))
}

/// `routes[row, slot] = slot`: the identity routing a grouped matmul reads.
pub fn group_routes(f: &mut Func, groups: u32, routes: Landing) -> Result<Val, Error> {
    const OP: &str = "linear.group_routes";
    expect(OP, routes, &[Dtype::I32])?;
    nonzero(OP, "the group count", groups)?;
    if routes.width != groups {
        return Err(refuse(
            OP,
            format!("the routes are {} wide for {groups} groups", routes.width),
        ));
    }
    let v = f.iota(Elem::I32, &[i64::from(routes.rows), i64::from(groups)], 1);
    Ok(v)
}

// ------------------------------------------------------------ expert compute

/// How an expert bank is stored. Every form is expert-major and flat:
/// expert `e`, output row `n`, contracted column `k` sits at `(e*N + n)*K + k`
/// (one scale/zero point per `group` codes of a row), however the engine
/// shapes the plane; codes are one per element as `crate::pack` lands them,
/// pre-scaled mxfp4 weights (`E5m2`), or an integer plane packing several
/// (low bits first).
#[derive(Clone, Copy)]
enum Form<T = Operand> {
    /// bf16 weights.
    Dense(T),
    /// `scale * q + bias`, `bits`-wide unsigned codes, bf16 factors.
    Affine {
        codes: T,
        scales: T,
        biases: T,
        group: u32,
        bits: u32,
    },
    /// e2m1 codes, one e8m0 exponent byte per 32 (or pre-scaled weights).
    Mxfp4 { codes: T, scales: T },
    /// e4m3 codes, one f32 factor a 128x128 block of each expert's `[N, K]`
    /// (`[E·⌈N/128⌉, K/128]`): a block-FP8 release's experts as released.
    Tile { codes: T, scales: T },
}

impl<T> Form<T> {
    fn bits(self) -> u32 {
        match self {
            Self::Dense(_) => 16,
            Self::Affine { bits, .. } => bits,
            Self::Mxfp4 { .. } => 4,
            Self::Tile { .. } => 8,
        }
    }

    fn group(self) -> u32 {
        match self {
            Self::Dense(_) => 1,
            Self::Affine { group, .. } => group,
            Self::Mxfp4 { .. } => MXFP4_BLOCK,
            Self::Tile { .. } => TILE as u32,
        }
    }
}

fn form_of<T: Stored>(op: &'static str, bank: Bank<T>) -> Result<Form<T>, Error> {
    match (bank.biases, bank.group, bank.bits) {
        (Some(biases), group, bits @ (2 | 4 | 8)) if group > 0 => {
            expect(op, bank.scales, &[Dtype::Bf16])?;
            expect(op, biases, &[Dtype::Bf16])?;
            Ok(Form::Affine {
                codes: bank.codes,
                scales: bank.scales,
                biases,
                group,
                bits,
            })
        }
        (None, MXFP4_BLOCK, 4) => {
            expect(op, bank.scales, &[Dtype::U8, Dtype::E8m0])?;
            Ok(Form::Mxfp4 {
                codes: bank.codes,
                scales: bank.scales,
            })
        }
        (None, group, 8) if i64::from(group) == TILE && bank.codes.dtype() == Dtype::E4m3 => {
            expect(op, bank.scales, &[Dtype::F32])?;
            Ok(Form::Tile {
                codes: bank.codes,
                scales: bank.scales,
            })
        }
        (biases, group, bits) => Err(refuse(
            op,
            format!(
                "the bank is {} at {bits} bits in groups of {group}; this kernel decodes affine \
                 2/4/8-bit banks and mxfp4 (32-code blocks)",
                if biases.is_some() {
                    "affine"
                } else {
                    "symmetric"
                }
            ),
        )),
    }
}

/// The static facts of one routed matmul.
#[derive(Clone, Copy)]
struct Fan {
    tokens: i64,
    top_k: i64,
    /// `x` holds one row per token (shared by its slots), not one per pair.
    per_token: bool,
    k: i64,
    n: i64,
    experts: i64,
}

impl Fan {
    fn pairs(self) -> i64 {
        self.tokens * self.top_k
    }
}

/// Codes a codes plane stores, counted in `bits`-wide codes.
fn stored_codes(op: &'static str, codes: impl Stored, bits: u32) -> Result<u64, Error> {
    if matches!(codes.dtype(), Dtype::E5m2 | Dtype::E4m3) {
        return Ok(codes.elements());
    }
    if let Some(per) = crate::pack::codes_per_row(codes.dtype(), u64::from(codes.width())) {
        if crate::pack::code_bits(codes.dtype()) != Some(bits) {
            return Err(refuse(
                op,
                format!("a {:?} codes plane in a {bits}-bit bank", codes.dtype()),
            ));
        }
        return Ok(u64::from(codes.rows()) * per);
    }
    match crate::check::elem_of(op, codes.dtype()) {
        Ok(e) if e.is_int() => Ok(codes.elements() * u64::from(e.bits()) / u64::from(bits)),
        _ => Err(refuse(
            op,
            format!(
                "a {:?} codes plane; codes land one per element (`kernels_xla::pack`) or in an \
                 integer plane",
                codes.dtype()
            ),
        )),
    }
}

fn fan_of(
    op: &'static str,
    x_rows: u32,
    k: u32,
    n: u32,
    routes: impl Stored,
    form: Form<impl Stored>,
) -> Result<Fan, Error> {
    expect(op, routes, &[Dtype::I32])?;
    nonzero(op, "the routed fan-out", routes.width())?;
    nonzero(op, "K, the activation's width", k)?;
    nonzero(op, "N, the output width", n)?;
    let pairs = u64::from(routes.rows()) * u64::from(routes.width());
    let per_token = if x_rows == routes.rows() {
        true
    } else if u64::from(x_rows) == pairs {
        false
    } else {
        return Err(refuse(
            op,
            format!("the activation's {x_rows} rows are neither the fire's tokens nor its routes"),
        ));
    };
    let (k64, n64) = (u64::from(k), u64::from(n));
    let per = k64 * n64;
    let experts = match form {
        Form::Dense(bank) => {
            expect(op, bank, &[Dtype::Bf16])?;
            if !bank.elements().is_multiple_of(per) {
                return Err(refuse(
                    op,
                    format!(
                        "a {}-element bank is not whole {n}x{k} experts",
                        bank.elements()
                    ),
                ));
            }
            bank.elements() / per
        }
        Form::Affine { codes, scales, .. }
        | Form::Mxfp4 { codes, scales }
        | Form::Tile { codes, scales } => {
            let group = u64::from(form.group());
            if !k64.is_multiple_of(group) {
                return Err(refuse(
                    op,
                    format!("K is {k}: not whole {group}-code groups"),
                ));
            }
            let stored = stored_codes(op, codes, form.bits())?;
            if !stored.is_multiple_of(per) {
                return Err(refuse(
                    op,
                    format!("the codes plane's {stored} codes are not whole {n}x{k} experts"),
                ));
            }
            let experts = stored / per;
            // A block-FP8 factor covers 128 rows as well as 128 codes.
            let rows = if matches!(form, Form::Tile { .. }) {
                n64.div_ceil(group)
            } else {
                n64
            };
            let factors = experts * rows * (k64 / group);
            let planes = match form {
                Form::Affine { biases, .. } => vec![scales, biases],
                _ => vec![scales],
            };
            for p in planes {
                if p.elements() != factors {
                    return Err(refuse(
                        op,
                        format!(
                            "a factor plane holds {} entries for {experts} experts of {n}x{} groups",
                            p.elements(),
                            k64 / group
                        ),
                    ));
                }
            }
            experts
        }
    };
    if experts == 0 {
        return Err(refuse(op, "the bank holds no expert"));
    }
    Ok(Fan {
        tokens: i64::from(routes.rows()),
        top_k: i64::from(routes.width()),
        per_token,
        k: i64::from(k),
        n: i64::from(n),
        experts: experts as i64,
    })
}

/// A read bank over its `E·N` weight rows: the codes (or bf16 weights) and,
/// for codes, the f32 `[E·N, G]` factor and affine zero point.
#[derive(Clone, Copy)]
struct Read {
    codes: Codes,
    factor: Option<Val>,
    bias: Option<Val>,
}

fn read_bank(f: &mut Func, form: Form, fan: Fan) -> Result<Read, Error> {
    let rows = fan.experts * fan.n;
    let groups = fan.k / i64::from(form.group());
    Ok(match form {
        Form::Dense(bank) => {
            let w = bank.val;
            // A bank view is `[rows, k]` already (and has no text to reshape).
            let w = if f.bank_parts(w).is_some() && f.dims(w) == [rows, fan.k] {
                w
            } else {
                f.reshape(w, &[rows, fan.k])?
            };
            Read {
                codes: Codes::Weights(w),
                factor: None,
                bias: None,
            }
        }
        Form::Affine {
            codes,
            scales,
            biases,
            bits,
            ..
        } => {
            let c = read_codes(f, codes, bits, false, rows, fan.k)?;
            let s = scales.f32(f);
            let b = biases.f32(f);
            Read {
                codes: c,
                factor: Some(f.reshape(s, &[rows, groups])?),
                bias: Some(f.reshape(b, &[rows, groups])?),
            }
        }
        Form::Tile { codes, scales } => {
            let c = read_codes(f, codes, 8, false, rows, fan.k)?;
            // Each expert's block rows repeat over its rows.
            let nb = (fan.n + TILE - 1) / TILE;
            let s = f.reshape(scales.val, &[fan.experts, nb * groups])?;
            let s = f.reshape(s, &[fan.experts * nb, groups])?;
            let s = tile_factor(f, s, fan.experts * nb * TILE)?;
            let s = if nb * TILE == fan.n {
                s
            } else {
                let s = f.reshape(s, &[fan.experts, nb * TILE, groups])?;
                let s = f.slice_axis(s, 1, 0, fan.n)?;
                f.reshape(s, &[rows, groups])?
            };
            Read {
                codes: c,
                factor: Some(s),
                bias: None,
            }
        }
        Form::Mxfp4 { codes, scales } => {
            let c = read_codes(f, codes, 4, true, rows, fan.k)?;
            let factor = if let Codes::Weights(_) = c {
                None
            } else {
                let s = scales.val;
                let s = if f.elem(s) == Elem::U8 {
                    s
                } else {
                    f.bitcast(s, Elem::U8)?
                };
                let s = f.reshape(s, &[rows, groups])?;
                let s = f.convert(s, Elem::U32);
                Some(e8m0(f, s)?)
            };
            Read {
                codes: c,
                factor,
                bias: None,
            }
        }
    })
}

/// The bank decoded to bf16 `[E·N, K]`.
fn decode(f: &mut Func, bank: &Read, groups: i64) -> Result<Val, Error> {
    let q = code_values(f, bank.codes, Elem::Bf16)?;
    let Some(factor) = bank.factor else {
        return Ok(q);
    };
    let (rows, k) = (f.dims(q)[0], f.dims(q)[1]);
    let q = f.reshape(q, &[rows, groups, k / groups])?;
    let w = materialize(f, q, factor, bank.bias)?;
    Ok(f.convert(w, Elem::Bf16))
}

/// `x [r, K] · W_allᵀ`: every row against every expert, f32 `[r, E·N]`.
fn against_all(f: &mut Func, x: Val, bank: &Read, groups: i64) -> Result<Val, Error> {
    let Some(factor) = bank.factor else {
        let w = code_values(f, bank.codes, Elem::Bf16)?;
        return Ok(f.matmul_nt(x, w, Elem::F32)?);
    };
    let q = code_values(f, bank.codes, Elem::Bf16)?;
    let (rows, k) = (f.dims(q)[0], f.dims(q)[1]);
    let q = f.reshape(q, &[rows, groups, k / groups])?;
    let r = f.dims(x)[0];
    let xg = f.reshape(x, &[r, groups, k / groups])?;
    let mut out = grouped_dot(f, xg, q, factor)?;
    if let Some(b) = bank.bias {
        let o = offset_dot(f, x, b)?;
        out = f.add(out, o)?;
    }
    Ok(out)
}

/// `chlo.ragged_dot`: `x [P, K]` rows grouped contiguously by `sizes [E]`
/// (i32), group `e` contracted with `w[e]` (`[E, N, K]`), into f32 `[P, N]`.
/// Rows past the last group come out zero. PJRT's MLIR ingestion legalizes
/// it to XLA's `ragged-dot`, which the TPU compiler runs natively.
fn ragged_dot(f: &mut Func, x: Val, w: Val, sizes: Val) -> Result<Val, Error> {
    let (p, n) = (f.dims(x)[0], f.dims(w)[1]);
    let out = Ty::new(Elem::F32, &[p, n]);
    let attrs = "precision_config = [#chlo<precision DEFAULT>, #chlo<precision DEFAULT>], \
                 ragged_dot_dimension_numbers = #chlo.ragged_dot<lhs_batching_dimensions = [], \
                 rhs_batching_dimensions = [], lhs_contracting_dimensions = [1], \
                 rhs_contracting_dimensions = [2], lhs_ragged_dimensions = [0], \
                 rhs_group_dimensions = [0]>";
    Ok(f.op_named("chlo.ragged_dot", &[x, w, sizes], attrs, vec![out])[0])
}

/// `table[ids]` for a short i32 `table` (`[E]`) and `[n]` i32 `ids` in
/// range, as a one-hot select and sum: XLA lowers a gather from a table this
/// small to a serial chain of scalar selects (v6e: ~40 us over 768 ids).
fn pick_small(f: &mut Func, table: Val, ids: Val) -> Result<Val, Error> {
    let e = f.dims(table)[0];
    let n = f.dims(ids)[0];
    let is = f.broadcast(ids, &[n, e], &[0])?;
    let es = f.iota(Elem::I32, &[n, e], 1);
    let hit = f.compare(Cmp::Eq, is, es)?;
    let ts = f.broadcast(table, &[n, e], &[1])?;
    let zero = f.const_i(Elem::I32, 0, &[n, e]);
    let v = f.select(hit, ts, zero)?;
    Ok(f.reduce(v, &[1], Fold::Sum)?)
}

/// The grouped-matmul kernel (`crate::linear::moe_gmm`) for this bank, when its
/// weights are plain (bf16 or pre-scaled f8, no group factors) and a block
/// of them fits VMEM, unless the profile's `kernel.moe.path` rules it out.
///
/// It reads each routed expert once, at the DMA rate (~1.4 TB/s), with no
/// per-expert launch. vLLM-TPU's `gmm_v2` streams the same experts in the
/// same time: 5.3 ms of gate/up and 2.6 ms of down a step for gpt-oss at
/// 64 lanes. The loop over active experts runs the down experts at ~0.6
/// TB/s: 12.6 ms for both banks. Measured on v6e, gpt-oss-20b, ms/step with
/// logits kept, without and with it: 9.3 / 16.0 / 20.8 and 6.5 / 11.1 / 15.9
/// at widths 8 / 32 / 64.
/// Keep the XLA around the call free of gathers from short tables
/// (`pick_small`) and of per-row column offsets: XLA lowers both to serial
/// scalar code, which cost more than the kernel saved.
fn gmm_spec(
    fan: Fan,
    bank: &Read,
    elem: Elem,
    mxfp4: bool,
    tuning: Tuning,
) -> Option<crate::linear::moe_gmm::Gmm> {
    use crate::linear::moe_gmm::{Gmm, Orient};
    let off = !tuning.moe_gmm();
    // mxfp4 codes as landed (`f4E2M1FN`), their exponents beside them.
    let scaled = mxfp4 && matches!(bank.codes, Codes::E2m1(_)) && elem == Elem::F4E2m1fn;
    // Block-FP8 codes, their f32 block factors beside them (the codes land
    // `[K, E·N]`: see engine-xla's `gemm_only`).
    let tile =
        elem == Elem::F8E4m3fn && bank.factor.is_some() && matches!(bank.codes, Codes::Int(_));
    if off
        || (bank.factor.is_some() && !scaled && !tile)
        || !(matches!(bank.codes, Codes::Weights(_)) || scaled || tile)
        || !matches!(
            elem,
            Elem::Bf16 | Elem::F8E5m2 | Elem::F8E4m3fn | Elem::F4E2m1fn
        )
        || (scaled && (Orient::of(fan.experts * fan.n, fan.k) != Orient::KMajor || fan.k % 32 != 0))
    {
        return None;
    }
    let (p, e) = (fan.pairs(), fan.experts);
    let tm = if mxfp4 {
        gmm_rows_whole(p, e, tuning.moe_gmm_tm)
    } else {
        gmm_rows(p, e, tuning.moe_gmm_tm)
    };
    let spec = Gmm {
        // Every active expert's rows round up to whole tiles.
        tiles: (p + tm - 1) / tm + e.min(p),
        tm,
        block_bytes: tuning.moe_gmm_block_bytes,
        vmem_cap: tuning.vmem_cap(),
        k: fan.k,
        n: fan.n,
        experts: e,
        w: elem,
        orient: if tile {
            Orient::KMajor
        } else {
            Orient::of(e * fan.n, fan.k)
        },
        scaled,
        tile,
        bias: false,
        out: Elem::F32,
        // mxfp4 banks (pre-scaled) at decode widths that route every expert
        // a tile or less (v6e gpt-oss, 256 lanes: 33.8 -> 31.8 ms a step; at
        // 4 lanes, 16 pairs, 8.4 -> 8.7 ms, so not there).
        prefetch: !mxfp4 || (16 * e <= p && p <= e * tm),
        sums: None,
        run: 1,
        glu: None,
        pooled: None,
    };
    spec.out_width().ok().map(|_| spec)
}

/// `x [m, K] · wᵀ` for block-FP8 codes `codes [N, K]` (landed `[K, N]`)
/// with their f32 factors a row `factor [N, K/128]`: the grouped-matmul
/// kernel over one expert, `[m, N]` in `out` (f32, or rounded to bf16).
/// `None` when no block of the weight fits the kernel.
pub(crate) fn tile_projection(
    f: &mut Func,
    x: Val,
    codes: Val,
    factor: Val,
    tuning: Tuning,
    out: Elem,
    sums: Option<i64>,
) -> Result<Option<(Val, Option<Val>)>, Error> {
    use crate::linear::moe_gmm::{Gmm, Orient};
    let (m, k) = (f.dims(x)[0], f.dims(x)[1]);
    let n = f.dims(codes)[0];
    if !tuning.moe_gmm() || k % TILE != 0 {
        return Ok(None);
    }
    // The rows in one tile where their double-buffered block fits 64 MiB:
    // a prompt's tile widens each block of codes once for all its rows
    // (`moe_gmm::Gmm::widens`; v6e, 24576x6144 at 2048 rows: 0.73 ms in one
    // tile, as bf16's 0.72, against 1.34 widened in two). Rows past one tile
    // take the 128-deep dots in 512-row tiles instead.
    let most = (64 << 20) / (4 * k) / 16 * 16;
    let rows = ((m + 15) / 16) * 16;
    let tm = if rows <= most.max(512) { rows } else { 512 };
    let tiles = (m + tm - 1) / tm;
    let mut spec = Gmm {
        tiles,
        tm,
        block_bytes: tuning.moe_gmm_block_bytes,
        vmem_cap: tuning.vmem_cap(),
        k,
        n,
        experts: 1,
        w: Elem::F8E4m3fn,
        orient: Orient::KMajor,
        scaled: false,
        tile: true,
        bias: false,
        out,
        prefetch: false,
        sums,
        run: 1,
        glu: None,
        pooled: None,
    };
    if spec.out_width().ok() != Some(n) {
        return Ok(None);
    }
    // The head sums ride along where the blocks hold whole heads.
    let kernel = match spec.kernel() {
        Ok(kernel) => kernel,
        Err(_) if spec.sums.is_some() => {
            spec.sums = None;
            spec.kernel()?
        }
        Err(e) => return Err(e.into()),
    };
    let se = f.const_i(Elem::I32, 0, &[tiles]);
    let nt = f.const_i(Elem::I32, tiles, &[1]);
    let x = f.convert(x, Elem::Bf16);
    let xs = if tiles * tm > m {
        let z = f.const_f(Elem::Bf16, 0.0, &[]);
        f.pad(x, z, &[0, 0], &[tiles * tm - m, 0], &[0, 0])?
    } else {
        x
    };
    // The device's layouts: free transposes.
    let w = f.transpose(codes, &[1, 0])?;
    let s = f.transpose(factor, &[1, 0])?;
    let outs = kernel.call(f, &[se, nt, xs, w, s])?;
    let rows = |f: &mut Func, v: Val| -> Result<Val, Error> {
        Ok(if tiles * tm > m {
            f.slice_axis(v, 0, 0, m)?
        } else {
            v
        })
    };
    let y = rows(f, outs[0])?;
    let sums = spec.sums.map(|_| rows(f, outs[1])).transpose()?;
    Ok(Some((y, sums)))
}

/// A kernel spec landing its rows in bf16 (no bias follows).
fn bf16_rows(spec: Option<crate::linear::moe_gmm::Gmm>) -> Option<crate::linear::moe_gmm::Gmm> {
    spec.map(|s| crate::linear::moe_gmm::Gmm {
        out: Elem::Bf16,
        ..s
    })
}

/// The rows of one grouped-matmul tile for `p` pairs over `e` experts: the
/// profile's tile, doubled while an expert still holds twice its rows on
/// average, up to 128. A tile's dot loads its expert's block into the MXU
/// whatever its rows, so an expert's run cut into tiles of 16 loads the same
/// weights once per 16 rows: at a 640-row prefill of Gemma-4-26B's 32-expert
/// cut (160 rows an expert) 352 tiles where 72 of 128 rows read the same.
fn gmm_rows(p: i64, e: i64, base: i64) -> i64 {
    let per = p / e.min(p).max(1);
    let mut tm = base.max(16);
    while tm < 128 && tm * 2 <= per {
        tm *= 2;
    }
    tm
}

/// The rows of one tile over an mxfp4 bank (its e2m1 codes or e5m2, the
/// kernel converting them in VMEM): an expert's average rows, rounded up to
/// a power of two (up to 128: 256-row tiles padded gpt-oss's 1792-row
/// prefill's 7168 pairs to 15360 rows, whose MXU passes and plumbing cost
/// v5e 9% of its fire), so an expert's run is a step or two. A second tile on the same block
/// issues no DMA, and the next expert's block starts loading only a step
/// ahead: at a 640-row prefill of gpt-oss (80 rows an expert, two tiles of
/// 64) the bank streamed at 1.0 TB/s on v6e. No slack past the average: on
/// v5e the padded rows' MXU passes cost more than the bubble.
fn gmm_rows_whole(p: i64, e: i64, base: i64) -> i64 {
    let active = e.min(p).max(1);
    let per = (p + active - 1) / active;
    let mut tm = base.max(16);
    // A fire past 4096 tokens' pairs fills 256-row tiles nearly whole (v6e
    // gpt-oss 4096-row prefill: pf256 32.3k tok/s against 31.6k at 128).
    let cap = if p >= 16384 { 256 } else { 128 };
    while tm < cap && tm < per {
        tm *= 2;
    }
    // An expert holding two 256-row tiles or more takes them: each step
    // converts its block once for twice the rows, at the same padding
    // (DeepSeek-V4's 1792-row prompt, 672 rows an expert: its routed
    // matmuls 10.4 -> 8.0 of a v6e fire).
    if per >= 512 {
        tm = 256;
    }
    tm
}

/// A fire's routed pairs sorted by expert into the grouped-matmul kernel's
/// row tiles: each expert's run of rows padded to whole tiles of `tm` rows
/// (`tiles` at most). Two routed matmuls on the same routes and tile share
/// it, so one's tile-ordered result feeds the other with no un-permute and
/// re-gather between them.
#[derive(Clone, Copy, Debug)]
pub struct TilePlan {
    perm: Val,
    iota: Val,
    first: Val,
    tstart: Val,
    counts: Val,
    se: Val,
    nt: Val,
    ids: Val,
    valid: Val,
    tm: i64,
    tiles: i64,
    pairs: i64,
    experts: i64,
    top_k: i64,
}

impl TilePlan {
    /// The rows of a tile-ordered plane under this plan.
    #[must_use]
    pub const fn rows(&self) -> i64 {
        self.tiles * self.tm
    }
}

/// A routed matmul's result rows in tile order (`[tiles · tm, width]` bf16),
/// with the plan that orders them and the kernel's cut of its columns.
#[derive(Clone, Copy, Debug)]
pub struct Tiled {
    pub rows: Val,
    pub width: i64,
    pub plan: TilePlan,
    spec: crate::linear::moe_gmm::Gmm,
}

fn plan_tiles(
    f: &mut Func,
    fan: Fan,
    tm: i64,
    tiles: i64,
    r: Val,
    valid: Val,
    ids: Val,
) -> Result<TilePlan, Error> {
    let (p, e) = (fan.pairs(), fan.experts);
    let top = f.const_i(Elem::I32, e, &[p]);
    let key = f.select(valid, r, top)?;
    let iota = f.iota(Elem::I32, &[p], 0);
    let by = f.sort(&[key, iota], 0, true, |fq, a, b| {
        fq.compare(Cmp::Lt, a[0], b[0])
    })?;
    let (skey, perm) = (by[0], by[1]);
    // Rows per expert, and each expert's first sorted position and tile.
    let counts = {
        let ks = f.broadcast(skey, &[p, e], &[0])?;
        let es = f.iota(Elem::I32, &[p, e], 1);
        let hit = f.compare(Cmp::Eq, ks, es)?;
        let one = f.const_i(Elem::I32, 1, &[p, e]);
        let none = f.const_i(Elem::I32, 0, &[p, e]);
        let hit = f.select(hit, one, none)?;
        f.reduce(hit, &[0], Fold::Sum)?
    };
    let first = {
        let upto = f.scan(counts, 0, Fold::Sum)?;
        f.sub(upto, counts)?
    };
    let per = {
        let more = f.const_i(Elem::I32, tm - 1, &[e]);
        let c = f.add(counts, more)?;
        let tt = f.const_i(Elem::I32, tm, &[e]);
        f.div(c, tt)?
    };
    let tend = f.scan(per, 0, Fold::Sum)?;
    let tstart = f.sub(tend, per)?;
    let nt = f.slice(tend, &[e - 1], &[e], &[1])?;
    // Tile `t`'s expert: how many experts' tiles end at or before it.
    let se = {
        let ends = f.broadcast(tend, &[tiles, e], &[1])?;
        let ts = f.iota(Elem::I32, &[tiles, e], 0);
        let done = f.compare(Cmp::Le, ends, ts)?;
        let one = f.const_i(Elem::I32, 1, &[tiles, e]);
        let none = f.const_i(Elem::I32, 0, &[tiles, e]);
        let done = f.select(done, one, none)?;
        let se = f.reduce(done, &[1], Fold::Sum)?;
        let hi = f.const_i(Elem::I32, e - 1, &[tiles]);
        f.min(se, hi)?
    };
    Ok(TilePlan {
        perm,
        iota,
        first,
        tstart,
        counts,
        se,
        nt,
        ids,
        valid,
        tm,
        tiles,
        pairs: p,
        experts: e,
        top_k: fan.top_k,
    })
}

/// The activation rows in tile order: padded row `i` holds the `rank`-th
/// row routed to its tile's expert.
fn tile_rows(f: &mut Func, t: &TilePlan, per_token: bool, x: Val) -> Result<Val, Error> {
    let (tm, rows) = (t.tm, t.tiles * t.tm);
    let ri = f.iota(Elem::I32, &[rows], 0);
    let tt = f.const_i(Elem::I32, tm, &[rows]);
    let tile = f.div(ri, tt)?;
    let ex = pick_small(f, t.se, tile)?;
    let base = pick_small(f, t.tstart, ex)?;
    let base = f.mul(base, tt)?;
    let rank = f.sub(ri, base)?;
    let count = pick_small(f, t.counts, ex)?;
    let ntb = f.broadcast(t.nt, &[rows], &[0])?;
    let live = f.compare(Cmp::Lt, tile, ntb)?;
    let within = f.compare(Cmp::Lt, rank, count)?;
    let ok = f.and(live, within)?;
    let at = pick_small(f, t.first, ex)?;
    let at = f.add(at, rank)?;
    let pair = take_flat(f, t.perm, at)?;
    let src = if per_token {
        let kk = f.const_i(Elem::I32, t.top_k, &[rows]);
        f.div(pair, kk)?
    } else {
        pair
    };
    let zero = f.const_i(Elem::I32, 0, &[rows]);
    let src = f.select(ok, src, zero)?;
    Ok(f.take_rows(x, src)?)
}

/// The kernel over tile-ordered rows `xs`: `[tiles · tm, out_width]`.
fn tile_kernel(
    f: &mut Func,
    spec: crate::linear::moe_gmm::Gmm,
    fan: Fan,
    bank: &Read,
    scales: Option<Val>,
    bias: Option<Val>,
    t: &TilePlan,
    xs: Val,
) -> Result<Val, Error> {
    use crate::linear::moe_gmm::Orient;
    let (e, n) = (fan.experts, fan.n);
    let w = bank.codes.val();
    // e2m1 codes as their bits, bitcast in the bank's own layout (after the
    // transpose, XLA copies the whole bank to relay the u4 out).
    let w = if spec.scaled {
        f.bitcast(w, Elem::U4)?
    } else {
        w
    };
    let w = match spec.orient {
        // The bank's layout on the device: a free transpose.
        Orient::KMajor => f.transpose(w, &[1, 0])?,
        Orient::NMajor => w,
    };
    let kernel = spec.kernel()?;
    let bias = match bias {
        Some(b) if spec.bias => Some(f.reshape(b, &[e, 1, n])?),
        _ => None,
    };
    let (se, nt) = (t.se, t.nt);
    if spec.prefetching() {
        // Each tile's run (its expert's rank among the routed ones) and the
        // next run's expert (-1 after the last), as `sorted` gives them.
        let more = f.const_i(Elem::I32, t.tm - 1, &[e]);
        let per = f.add(t.counts, more)?;
        let tt = f.const_i(Elem::I32, t.tm, &[e]);
        let per = f.div(per, tt)?;
        let tend = f.add(t.tstart, per)?;
        let zero = f.const_i(Elem::I32, 0, &[e]);
        let one = f.const_i(Elem::I32, 1, &[e]);
        let busy = f.compare(Cmp::Gt, t.counts, zero)?;
        let busy = f.select(busy, one, zero)?;
        let upto = f.scan(busy, 0, Fold::Sum)?;
        let rank = f.sub(upto, busy)?;
        let run = pick_small(f, rank, se)?;
        let tend_t = pick_small(f, tend, se)?;
        let end = f.const_i(Elem::I32, t.tiles - 1, &[t.tiles]);
        let at = f.min(tend_t, end)?;
        let nx = pick_small(f, se, at)?;
        let ntb = f.broadcast(nt, &[t.tiles], &[0])?;
        let next = f.compare(Cmp::Lt, tend_t, ntb)?;
        let none = f.const_i(Elem::I32, -1, &[t.tiles]);
        let nxt = f.select(next, nx, none)?;
        let ops: Vec<Val> = [se, nt, run, nxt, xs, w].into_iter().chain(bias).collect();
        return Ok(kernel.call(f, &ops)?[0]);
    }
    Ok(if spec.scaled {
        // The exponent bytes `[E·N, K/32]`, read `[K/32, E·N]` as the codes
        // are (the TPU lays them `E·N`-minor too).
        let s = scales.ok_or_else(|| refuse("linear.moe", "scaled codes with no exponents"))?;
        let s = if f.elem(s) == Elem::U8 {
            s
        } else {
            f.bitcast(s, Elem::U8)?
        };
        let s = f.reshape(s, &[e * n, fan.k / 32])?;
        let s = f.transpose(s, &[1, 0])?;
        let ops: Vec<Val> = [se, nt, xs, w, s].into_iter().chain(bias).collect();
        kernel.call(f, &ops)?[0]
    } else if spec.tile {
        // The factors a row (`[E·N, K/128]`), read `[K/128, E·N]` as the
        // codes are.
        let s = scales.ok_or_else(|| refuse("linear.moe", "block-FP8 codes with no factors"))?;
        let s = f.transpose(s, &[1, 0])?;
        let ops: Vec<Val> = [se, nt, xs, w, s].into_iter().chain(bias).collect();
        kernel.call(f, &ops)?[0]
    } else {
        let ops: Vec<Val> = [se, nt, xs, w].into_iter().chain(bias).collect();
        kernel.call(f, &ops)?[0]
    })
}

/// Each pair's tile row, in pair order.
fn tile_dest(f: &mut Func, t: &TilePlan) -> Result<Val, Error> {
    let p = t.pairs;
    let back = f.sort(&[t.perm, t.iota], 0, true, |fq, a, b| {
        fq.compare(Cmp::Lt, a[0], b[0])
    })?[1];
    let base = pick_small(f, t.tstart, t.ids)?;
    let tt = f.const_i(Elem::I32, t.tm, &[p]);
    let base = f.mul(base, tt)?;
    let at = pick_small(f, t.first, t.ids)?;
    let rank = f.sub(back, at)?;
    Ok(f.add(base, rank)?)
}

/// The rows `dest` names of a tile-ordered result, each cut to its expert's
/// `n` columns (the kernel lands every row's outputs from column 0):
/// `[len(dest), n]`.
fn untile(f: &mut Func, n: i64, out: Val, dest: Val) -> Result<Val, Error> {
    let y = f.take_rows(out, dest)?;
    let p = f.dims(y)[0];
    if f.dims(y)[1] == n {
        return Ok(y);
    }
    Ok(f.slice(y, &[0, 0], &[p, n], &[1, 1])?)
}

/// The routed matmul through the grouped-matmul kernel: the pairs sorted by
/// expert (unrouted last), each expert's run of rows padded to whole tiles
/// of `tm` rows, one tile per grid step against its expert's weights, and
/// each pair's row picked back out. Only the routed experts' weights move.
/// `[P, N]`, and the result in tile order. Rows already in tile order under
/// the same plan (`x_tiled`, a previous routed matmul's on these routes)
/// feed the kernel as they are.
fn gmm(
    f: &mut Func,
    spec: crate::linear::moe_gmm::Gmm,
    fan: Fan,
    bank: &Read,
    scales: Option<Val>,
    bias: Option<Val>,
    x: Val,
    r: Val,
    valid: Val,
    ids: Val,
    x_tiled: Option<Tiled>,
) -> Result<(Val, Tiled), Error> {
    let (plan, xs) = match x_tiled {
        Some(t)
            if t.plan.tm == spec.tm
                && t.plan.tiles == spec.tiles
                && t.plan.pairs == fan.pairs()
                && t.plan.experts == fan.experts
                && t.width == fan.k
                && !fan.per_token =>
        {
            (t.plan, t.rows)
        }
        _ => {
            let plan = plan_tiles(f, fan, spec.tm, spec.tiles, r, valid, ids)?;
            let xs = tile_rows(f, &plan, fan.per_token, x)?;
            (plan, xs)
        }
    };
    let out = tile_kernel(f, spec, fan, bank, scales, bias, &plan, xs)?;
    let dest = tile_dest(f, &plan)?;
    let y = untile(f, fan.n, out, dest)?;
    let width = spec.out_width()?;
    Ok((
        y,
        Tiled {
            rows: out,
            width,
            plan,
            spec,
        },
    ))
}

/// The routed matmul through the grouped-matmul kernel: the pairs sorted by
/// expert (unrouted last), each expert's run of rows padded to whole tiles
/// of `tm` rows, one tile per grid step against its expert's weights, and
/// each pair's row picked back out. Only the routed experts' weights move.
/// `[P, N]`, f32 or bf16 as the kernel lands it. (A dense bank's: an mxfp4
/// bank's rows go through [`gmm`], which chains its select's tiles.)
fn gmm_sorted(
    f: &mut Func,
    spec: crate::linear::moe_gmm::Gmm,
    fan: Fan,
    bank: &Read,
    x: Val,
    r: Val,
    valid: Val,
) -> Result<Val, Error> {
    let (p, n) = (fan.pairs(), fan.n);
    let sorted = sorted(f, spec, fan, x, r, valid, None)?;
    let out = tiles_of(f, spec, bank, &sorted, sorted.xs)?;
    let dest = dest(f, &sorted, p)?;
    untile(f, n, out, dest)
}

/// A fire's pairs sorted by expert into the grouped-matmul kernel's tiles.
struct Sorted {
    /// Each tile's expert, and the tile count.
    se: Val,
    nt: Val,
    /// The activation rows tile by tile, and the token each comes from.
    xs: Val,
    src: Val,
    /// The pair each row holds (a pad row: `>= p`).
    held: Val,
    /// The weight each row carries (0 for a pad), when the sort carried one.
    ws: Option<Val>,
    /// Each tile's run (its expert's rank among the routed ones) and the
    /// next run's expert (-1 after the last), for a kernel that fetches
    /// the next expert's weights while a run's tiles go.
    run: Val,
    nxt: Val,
    /// Each expert's tiles, the tiles up to its last, and its rank among
    /// the routed.
    per: Val,
    tend: Val,
    rank: Val,
}

/// A stepping kernel's schedule (`moe_gmm::Gmm::stepping`): each step's
/// expert, the step count, each step's run and the next run's expert, its
/// first tile and its tile count, i32 `[steps]` (`[1]` the count).
fn steps(
    f: &mut Func,
    spec: crate::linear::moe_gmm::Gmm,
    sorted: &Sorted,
) -> Result<[Val; 6], Error> {
    let (e, j, n) = (spec.experts, spec.run, spec.steps());
    let more = f.const_i(Elem::I32, j - 1, &[e]);
    let sp = f.add(sorted.per, more)?;
    let jj = f.const_i(Elem::I32, j, &[e]);
    let sp = f.div(sp, jj)?;
    let send = f.scan(sp, 0, Fold::Sum)?;
    let sstart = f.sub(send, sp)?;
    let ns = f.slice(send, &[e - 1], &[e], &[1])?;
    let sexp = {
        let ends = f.broadcast(send, &[n, e], &[1])?;
        let ss = f.iota(Elem::I32, &[n, e], 0);
        let done = f.compare(Cmp::Le, ends, ss)?;
        let one = f.const_i(Elem::I32, 1, &[n, e]);
        let none = f.const_i(Elem::I32, 0, &[n, e]);
        let done = f.select(done, one, none)?;
        let se = f.reduce(done, &[1], Fold::Sum)?;
        let hi = f.const_i(Elem::I32, e - 1, &[n]);
        f.min(se, hi)?
    };
    let si = f.iota(Elem::I32, &[n], 0);
    let first = pick_small(f, sstart, sexp)?;
    let local = f.sub(si, first)?;
    let jn = f.const_i(Elem::I32, j, &[n]);
    let done = f.mul(local, jn)?;
    let tstart = f.sub(sorted.tend, sorted.per)?;
    let t0 = pick_small(f, tstart, sexp)?;
    let stile = f.add(t0, done)?;
    let per = pick_small(f, sorted.per, sexp)?;
    let left = f.sub(per, done)?;
    let scnt = f.min(left, jn)?;
    let one = f.const_i(Elem::I32, 1, &[n]);
    let scnt = f.max(scnt, one)?;
    let srun = pick_small(f, sorted.rank, sexp)?;
    let end = pick_small(f, send, sexp)?;
    let top = f.const_i(Elem::I32, n - 1, &[n]);
    let at = f.min(end, top)?;
    let nx = pick_small(f, sexp, at)?;
    let nsb = f.broadcast(ns, &[n], &[0])?;
    let more = f.compare(Cmp::Lt, end, nsb)?;
    let none = f.const_i(Elem::I32, -1, &[n]);
    let snxt = f.select(more, nx, none)?;
    Ok([sexp, ns, srun, snxt, stile, scnt])
}

/// The pairs sorted by expert (unrouted last), each expert's run of rows
/// padded to whole tiles of `tm` rows, and `x`'s rows laid out so. The pad
/// rows sort in with the pairs (expert `e`'s after its pairs, the rest
/// last), so the sort lays out the padded rows itself: picking each row's
/// pair out of a sorted order cost a scalar gather (v6e, Gemma-4-26B's
/// 32-expert cut, 1792 tokens: 109 us a layer).
fn sorted(
    f: &mut Func,
    spec: crate::linear::moe_gmm::Gmm,
    fan: Fan,
    x: Val,
    r: Val,
    valid: Val,
    weights: Option<Val>,
) -> Result<Sorted, Error> {
    let (p, e, tm) = (fan.pairs(), fan.experts, spec.tm);
    let (tiles, rows) = (spec.tiles, spec.tiles * spec.tm);
    let top = f.const_i(Elem::I32, e, &[p]);
    let key = f.select(valid, r, top)?;
    // Rows per expert, and each expert's first tile.
    let counts = {
        let ks = f.broadcast(key, &[p, e], &[0])?;
        let es = f.iota(Elem::I32, &[p, e], 1);
        let hit = f.compare(Cmp::Eq, ks, es)?;
        let one = f.const_i(Elem::I32, 1, &[p, e]);
        let none = f.const_i(Elem::I32, 0, &[p, e]);
        let hit = f.select(hit, one, none)?;
        f.reduce(hit, &[0], Fold::Sum)?
    };
    let per = {
        let more = f.const_i(Elem::I32, tm - 1, &[e]);
        let c = f.add(counts, more)?;
        let tt = f.const_i(Elem::I32, tm, &[e]);
        f.div(c, tt)?
    };
    let tend = f.scan(per, 0, Fold::Sum)?;
    let nt = f.slice(tend, &[e - 1], &[e], &[1])?;
    // Tile `t`'s expert: how many experts' tiles end at or before it.
    let se = {
        let ends = f.broadcast(tend, &[tiles, e], &[1])?;
        let ts = f.iota(Elem::I32, &[tiles, e], 0);
        let done = f.compare(Cmp::Le, ends, ts)?;
        let one = f.const_i(Elem::I32, 1, &[tiles, e]);
        let none = f.const_i(Elem::I32, 0, &[tiles, e]);
        let done = f.select(done, one, none)?;
        let se = f.reduce(done, &[1], Fold::Sum)?;
        let hi = f.const_i(Elem::I32, e - 1, &[tiles]);
        f.min(se, hi)?
    };
    // Pad row `c` is the `c / tm`-th routed expert's `c % tm`-th, live while
    // that expert's run lacks as many rows to a whole tile (at most
    // `min(e, p)` experts are routed, and `rows - p >= min(e, p)·tm`).
    let fill = rows - p;
    let pads = {
        let tt = f.const_i(Elem::I32, tm, &[e]);
        let whole = f.mul(per, tt)?;
        f.sub(whole, counts)?
    };
    let zero = f.const_i(Elem::I32, 0, &[e]);
    let routed = f.compare(Cmp::Gt, counts, zero)?;
    let one = f.const_i(Elem::I32, 1, &[e]);
    let ones = f.select(routed, one, zero)?;
    let rank = f.scan(ones, 0, Fold::Sum)?;
    let rank = f.sub(rank, ones)?;
    let c = f.iota(Elem::I32, &[fill], 0);
    let tf = f.const_i(Elem::I32, tm, &[fill]);
    let nth = f.div(c, tf)?;
    let fs = f.rem(c, tf)?;
    let (fe, lacks) = {
        let jb = f.broadcast(nth, &[fill, e], &[0])?;
        let rb = f.broadcast(rank, &[fill, e], &[1])?;
        let hit = f.compare(Cmp::Eq, jb, rb)?;
        let on = f.broadcast(routed, &[fill, e], &[1])?;
        let hit = f.and(hit, on)?;
        let none = f.const_i(Elem::I32, 0, &[fill, e]);
        let es = f.iota(Elem::I32, &[fill, e], 1);
        let fe = f.select(hit, es, none)?;
        let pb = f.broadcast(pads, &[fill, e], &[1])?;
        let lacks = f.select(hit, pb, none)?;
        (
            f.reduce(fe, &[1], Fold::Sum)?,
            f.reduce(lacks, &[1], Fold::Sum)?,
        )
    };
    let live = f.compare(Cmp::Lt, fs, lacks)?;
    // Keys: pairs `2e` (unrouted `2E`), live pads `2e + 1`, the rest last.
    let two = f.const_i(Elem::I32, 2, &[p]);
    let kp = f.mul(key, two)?;
    let two = f.const_i(Elem::I32, 2, &[fill]);
    let kf = f.mul(fe, two)?;
    let one = f.const_i(Elem::I32, 1, &[fill]);
    let kf = f.add(kf, one)?;
    let end = f.const_i(Elem::I32, 2 * e + 1, &[fill]);
    let kf = f.select(live, kf, end)?;
    let keys = f.concat(&[kp, kf], 0)?;
    let iota = f.iota(Elem::I32, &[rows], 0);
    let mut carried = vec![keys, iota];
    if let Some(w) = weights {
        let none = f.const_f(Elem::F32, 0.0, &[fill]);
        carried.push(f.concat(&[w, none], 0)?);
    }
    let by = f.sort(&carried, 0, true, |fq, a, b| {
        fq.compare(Cmp::Lt, a[0], b[0])
    })?;
    let held = by[1];
    // Padded row `i` holds pair `held[i]` (a pad when `>= p`).
    let src = {
        let pp = f.const_i(Elem::I32, p, &[rows]);
        let pair = f.compare(Cmp::Lt, held, pp)?;
        let src = if fan.per_token {
            let kk = f.const_i(Elem::I32, fan.top_k, &[rows]);
            f.div(held, kk)?
        } else {
            held
        };
        let zero = f.const_i(Elem::I32, 0, &[rows]);
        f.select(pair, src, zero)?
    };
    let xs = f.take_rows(x, src)?;
    let run = pick_small(f, rank, se)?;
    let tend_t = pick_small(f, tend, se)?;
    let end = f.const_i(Elem::I32, tiles - 1, &[tiles]);
    let at = f.min(tend_t, end)?;
    let nx = pick_small(f, se, at)?;
    let ntb = f.broadcast(nt, &[tiles], &[0])?;
    let more = f.compare(Cmp::Lt, tend_t, ntb)?;
    let none = f.const_i(Elem::I32, -1, &[tiles]);
    let nxt = f.select(more, nx, none)?;
    Ok(Sorted {
        se,
        nt,
        xs,
        src,
        held,
        ws: weights.map(|_| by[2]),
        run,
        nxt,
        per,
        tend,
        rank,
    })
}

/// Pair `i`'s row of the kernel's result: where the sort put it.
fn dest(f: &mut Func, sorted: &Sorted, p: i64) -> Result<Val, Error> {
    let rows = f.dims(sorted.held)[0];
    let iota = f.iota(Elem::I32, &[rows], 0);
    let back = f.sort(&[sorted.held, iota], 0, true, |fq, a, b| {
        fq.compare(Cmp::Lt, a[0], b[0])
    })?[1];
    Ok(f.slice(back, &[0], &[p], &[1])?)
}

/// The kernel over `xs`, rows laid out tile by tile as `sorted` lays them.
fn tiles_of(
    f: &mut Func,
    spec: crate::linear::moe_gmm::Gmm,
    bank: &Read,
    sorted: &Sorted,
    xs: Val,
) -> Result<Val, Error> {
    use crate::linear::moe_gmm::Orient;
    let w = bank.codes.val();
    let w = match spec.orient {
        // The bank's layout on the device: a free transpose.
        Orient::KMajor => f.transpose(w, &[1, 0])?,
        Orient::NMajor => w,
    };
    let kernel = spec.kernel()?;
    if spec.stepping() {
        let st = steps(f, spec, sorted)?;
        let args = pooled_args(f, &[(spec, w)], st.to_vec(), vec![xs, w])?;
        return Ok(kernel.call(f, &args)?[0]);
    }
    if spec.prefetching() {
        let pre = vec![sorted.se, sorted.nt, sorted.run, sorted.nxt];
        let args = pooled_args(f, &[(spec, w)], pre, vec![xs, w])?;
        return Ok(kernel.call(f, &args)?[0]);
    }
    Ok(kernel.call(f, &[sorted.se, sorted.nt, xs, w])?[0])
}

/// A kernel's operands with its banks in the pool: each pooled bank's table
/// after the scalar-prefetched ones (in `banks` order), and the pool in the
/// bank's place.
fn pooled_args(
    f: &mut Func,
    banks: &[(crate::linear::moe_gmm::Gmm, Val)],
    mut pre: Vec<Val>,
    mut ins: Vec<Val>,
) -> Result<Vec<Val>, Error> {
    for &(spec, bank) in banks {
        if let Some(at) = f.bank_parts(bank).filter(|_| spec.pooled.is_some()) {
            pre.push(at.table);
            for v in &mut ins {
                if *v == bank {
                    *v = at.pool;
                }
            }
        }
    }
    pre.extend(ins);
    Ok(pre)
}

/// `spec` reading a bank view in place, where its kernel takes a pooled
/// bank (one expert fetched whole, a run at a time); `None` where it does
/// not (the bank's experts are then sliced or the bank read whole).
fn pooled_spec(
    f: &Func,
    spec: Option<crate::linear::moe_gmm::Gmm>,
    bank: &Read,
) -> Option<crate::linear::moe_gmm::Gmm> {
    let Some(at) = f.bank_parts(bank.codes.val()) else {
        return spec;
    };
    let rows = f.dims(at.pool)[0];
    let spec = crate::linear::moe_gmm::Gmm {
        pooled: Some(crate::linear::moe_gmm::Pooled {
            rows,
            offset: at.offset,
        }),
        ..spec?
    };
    (spec.prefetching() && spec.kernel().is_ok()).then_some(spec)
}

/// The routed matmul every select entry lowers to: `y[t*k + s] = W[routes[t,
/// s]] · x_row (+ bias[e])`; a slot routed outside `0..experts` keeps `y`'s
/// row as it was (the GPU kernels skip its store).
#[allow(clippy::too_many_arguments)]
fn routed(
    f: &mut Func,
    form: Form,
    fan: Fan,
    x: Val,
    routes: Val,
    mut bias: Option<Val>,
    prev: Val,
    tuning: Tuning,
    x_tiled: Option<Tiled>,
) -> Result<(Val, Option<Tiled>), Error> {
    let mut tiled = None;
    let (p, e, n, k) = (fan.pairs(), fan.experts, fan.n, fan.k);
    let groups = k / i64::from(form.group());
    let r = f.reshape(routes, &[p])?;
    let zero = f.const_i(Elem::I32, 0, &[p]);
    let top = f.const_i(Elem::I32, e, &[p]);
    let ge = f.compare(Cmp::Ge, r, zero)?;
    let lt = f.compare(Cmp::Lt, r, top)?;
    let valid = f.and(ge, lt)?;
    let lo = f.const_i(Elem::I32, 0, &[]);
    let hi = f.const_i(Elem::I32, e - 1, &[]);
    let ids = f.clamp(lo, r, hi)?;
    let bank = read_bank(f, form, fan)?;
    let x = f.convert(x, Elem::Bf16);
    let rows = f.dims(x)[0];
    let r8 = (rows + 7) / 8 * 8;
    let fold = if bank.factor.is_some() { groups } else { 1 };
    let unit = match f.elem(bank.codes.val()) {
        Elem::U4 | Elem::F4E2m1fn => 1,
        Elem::Bf16 | Elem::F16 => 4,
        _ => 2,
    };
    let bank_bytes = e * n * k * unit / 2;
    let expert_bytes = bank_bytes / e;
    // How many active experts (row tiles) the loop runs for less than the
    // whole bank, whose dot also lands a partial per row and expert.
    let tile = loop_tile(fan, tuning);
    let c = tuning.moe_costs;
    let whole_ns =
        bank_bytes / c.whole_bytes_per_ns.max(1) + r8 * e * n * c.whole_partial_fs / 1_000_000;
    let tile_ns =
        (expert_bytes / c.slice_bytes_per_ns.max(1)).max(c.loop_tile_ns + tile * c.loop_row_ns);
    let worth = whole_ns / tile_ns;
    // A forced `gmm` falls back to `all` for a bank the kernel does not read.
    let mxfp4 = matches!(form, Form::Mxfp4 { .. });
    let grouped = gmm_spec(fan, &bank, f.elem(bank.codes.val()), mxfp4, tuning);
    // A bank view (experts in the pool): the kernel reads it in place, or
    // each pair's expert is sliced out of it; any other path reads it whole.
    let viewed = f.bank_parts(bank.codes.val()).is_some();
    let grouped = pooled_spec(f, grouped, &bank);
    let path = match tuning.moe_path {
        _ if viewed && grouped.is_some() && !(p <= c.slice_pairs && fan.tokens == 1) => {
            MoePath::Gmm
        }
        _ if viewed && grouped.is_none() && p <= c.slice_pairs => MoePath::Slice,
        MoePath::Gmm if grouped.is_none() => MoePath::All,
        // One token's pairs route to as many experts: sliced out, they read
        // what the grouped matmul would, without its sort (v6e,
        // Qwen3.6-35B-A3B's 4 MiB experts: Raw decode 133 -> 176 tok/s).
        MoePath::Auto { .. }
            if p <= c.slice_pairs
                // (a block-FP8 bank's pairs run the kernel: sliced, they
                // would contract in XLA's group-batched dot)
                && !grouped.is_some_and(|g| g.tile)
                && ((fan.tokens == 1 && grouped.is_some())
                    || p * expert_bytes.max(c.slice_bytes) < bank_bytes) =>
        {
            MoePath::Slice
        }
        // Past a token or two: each routed expert read once, at the DMA
        // rate, with no per-expert launch (see `gmm_spec`).
        MoePath::Auto { .. } if grouped.is_some() => MoePath::Gmm,
        MoePath::Auto { .. } if r8 * e * n * fold > c.dense_all => MoePath::Ragged,
        MoePath::Auto { .. } if worth < 2 => MoePath::All,
        MoePath::Auto { .. } if worth >= p => MoePath::Loop,
        MoePath::Auto { .. } => MoePath::Active,
        forced => forced,
    };
    let bank = if viewed && !matches!(path, MoePath::Gmm | MoePath::Slice) {
        let w = f.bank_solid(bank.codes.val(), "a routed select off the kernel")?;
        Read {
            codes: bank.codes.with(w),
            ..bank
        }
    } else {
        bank
    };
    let y = match path {
        MoePath::Slice => {
            // Each pair's expert sliced out of the bank and multiplied by its
            // row alone: independent dots XLA overlaps, no join.
            let mut ys = Vec::with_capacity(p as usize);
            for i in 0..p {
                let id = f.slice(ids, &[i], &[i + 1], &[1])?;
                let id = f.reshape(id, &[])?;
                let one = expert(f, &bank, id, n)?;
                let row = if fan.per_token { i / fan.top_k } else { i };
                let xr = f.slice(x, &[row, 0], &[row + 1, k], &[1, 1])?;
                ys.push(against_all(f, xr, &one, groups)?);
            }
            f.concat(&ys, 0)?
        }
        MoePath::All => every_expert(f, &bank, fan, x, ids, groups)?,
        MoePath::Gmm => match grouped {
            // A dense bank: without an expert bias the sums only round to
            // bf16 after the kernel, which lands them bf16.
            Some(mut spec) if !mxfp4 && !spec.tile => {
                spec.out = if bias.is_some() {
                    Elem::F32
                } else {
                    Elem::Bf16
                };
                gmm_sorted(f, spec, fan, &bank, x, r, valid)?
            }
            Some(mut spec) => {
                let scales = match form {
                    Form::Mxfp4 { scales, .. } => Some(scales.val),
                    Form::Tile { .. } => bank.factor,
                    _ => None,
                };
                // The bias joins the product inside the kernel where its cut
                // is whole blocks, and the rows come back rounded, as
                // `finish` rounds them; a bias it cannot take stays outside,
                // over the f32 rows.
                let whole = spec.out_width().ok() == Some(n);
                spec.bias = bias.is_some() && whole;
                spec.out = if bias.is_none() || whole {
                    Elem::Bf16
                } else {
                    Elem::F32
                };
                let inside = if spec.bias { bias.take() } else { None };
                let (y, t) = gmm(
                    f, spec, fan, &bank, scales, inside, x, r, valid, ids, x_tiled,
                )?;
                // Rows that round as the select's own do, its bias inside
                // (mxfp4 banks: the tile-order chain is measured on gpt-oss).
                if mxfp4 && spec.out == Elem::Bf16 && bias.is_none() {
                    tiled = Some(t);
                }
                y
            }
            None => every_expert(f, &bank, fan, x, ids, groups)?,
        },
        MoePath::Loop => {
            let plan = plan_active(f, fan, x, r, valid, tuning)?;
            run_active(f, &plan, &bank, fan, groups, None, tuning)?
        }
        MoePath::Active => {
            // The loop over the experts the fire routes to, unless more are
            // active than it is worth: then the whole bank. Both run inside
            // loops whose trip count the fire's routes decide (0 steps for
            // the one not taken); a `case` would copy the bank into its
            // branch (v6e: 3x the whole step).
            let plan = plan_active(f, fan, x, r, valid, tuning)?;
            let most = f.const_i(Elem::I32, worth, &[]);
            let many = f.compare(Cmp::Gt, plan.steps, most)?;
            let looped = run_active(f, &plan, &bank, fan, groups, Some(many), tuning)?;
            let zeros = f.const_f(Elem::F32, 0.0, &[p, n]);
            let whole = f.while_loop(
                &[many, zeros],
                |_, a| Ok(a[0]),
                |fq, _| {
                    let f = fq;
                    let all = every_expert(f, &bank, fan, x, ids, groups).map_err(sealed)?;
                    let done = f.const_i(Elem::Pred, 0, &[]);
                    Ok(vec![done, all])
                },
            )?[1];
            let many = f.broadcast(many, &[p, n], &[])?;
            f.select(many, whole, looped)?
        }
        MoePath::Ragged | MoePath::Auto { .. } => {
            // Sort the pairs by expert (unrouted last), run one ragged dot over
            // the whole bank, and put the rows back.
            let top = f.const_i(Elem::I32, e, &[p]);
            let key = f.select(valid, r, top)?;
            let iota = f.iota(Elem::I32, &[p], 0);
            let by = f.sort(&[key, iota], 0, true, |fq, a, b| {
                fq.compare(Cmp::Lt, a[0], b[0])
            })?;
            let (skey, perm) = (by[0], by[1]);
            let ks = f.broadcast(skey, &[p, e], &[0])?;
            let es = f.iota(Elem::I32, &[p, e], 1);
            let hit = f.compare(Cmp::Eq, ks, es)?;
            let one = f.const_i(Elem::I32, 1, &[p, e]);
            let none = f.const_i(Elem::I32, 0, &[p, e]);
            let hit = f.select(hit, one, none)?;
            let sizes = f.reduce(hit, &[0], Fold::Sum)?;
            let src = if fan.per_token {
                let kk = f.const_i(Elem::I32, fan.top_k, &[p]);
                f.div(perm, kk)?
            } else {
                perm
            };
            let xs = f.take_rows(x, src)?;
            let w = decode(f, &bank, groups)?;
            let w = f.reshape(w, &[e, n, k])?;
            let ys = ragged_dot(f, xs, w, sizes)?;
            let back = f.sort(&[perm, iota], 0, true, |fq, a, b| {
                fq.compare(Cmp::Lt, a[0], b[0])
            })?;
            f.take_rows(ys, back[1])?
        }
    };
    Ok((finish(f, fan, y, ids, valid, bias, prev)?, tiled))
}

/// Expert `e`'s (an i32 scalar) rows of a read bank: `[N, ..]` of each plane.
fn expert(f: &mut Func, bank: &Read, e: Val, n: i64) -> Result<Read, Error> {
    if f.bank_parts(bank.codes.val()).is_some() {
        let w = f.bank_expert(bank.codes.val(), e)?;
        return Ok(Read {
            codes: bank.codes.with(w),
            factor: None,
            bias: None,
        });
    }
    let nn = f.const_i(Elem::I32, n, &[]);
    let at = f.mul(e, nn)?;
    let zero = f.const_i(Elem::I32, 0, &[]);
    let rows = |f: &mut Func, v: Val| -> Result<Val, Error> {
        let w = f.dims(v)[1];
        Ok(f.dynamic_slice(v, &[at, zero], &[n, w])?)
    };
    Ok(Read {
        codes: bank.codes.with(rows(f, bank.codes.val())?),
        factor: match bank.factor {
            Some(fq) => Some(rows(f, fq)?),
            None => None,
        },
        bias: match bank.bias {
            Some(b) => Some(rows(f, b)?),
            None => None,
        },
    })
}

/// Every activation row against every expert in one dot (the whole bank
/// streams once), each pair's row picked: f32 `[P, N]`.
fn every_expert(
    f: &mut Func,
    bank: &Read,
    fan: Fan,
    x: Val,
    ids: Val,
    groups: i64,
) -> Result<Val, Error> {
    let (p, e, n) = (fan.pairs(), fan.experts, fan.n);
    let rows = f.dims(x)[0];
    let r8 = (rows + 7) / 8 * 8;
    let xp = if r8 > rows {
        let z = f.const_f(Elem::Bf16, 0.0, &[]);
        f.pad(x, z, &[0, 0], &[r8 - rows, 0], &[0, 0])?
    } else {
        x
    };
    let all = against_all(f, xp, bank, groups)?;
    let all = f.reshape(all, &[r8 * e, n])?;
    let pair = f.iota(Elem::I32, &[p], 0);
    let row = if fan.per_token {
        let kk = f.const_i(Elem::I32, fan.top_k, &[p]);
        f.div(pair, kk)?
    } else {
        pair
    };
    let ee = f.const_i(Elem::I32, e, &[p]);
    let at = f.mul(row, ee)?;
    let at = f.add(at, ids)?;
    Ok(f.take_rows(all, at)?)
}

/// The rows of one tile of the loop over active experts: the fire's tokens
/// (a token routes to an expert once), at most `kernel.moe.loop_tile_max`.
fn loop_tile(fan: Fan, tuning: Tuning) -> i64 {
    let p = fan.pairs();
    let most = tuning.moe_loop_tile_max;
    ((fan.tokens.max(1) + 7) / 8 * 8)
        .min(most)
        .min((p + 7) / 8 * 8)
}

/// The pairs sorted by expert and cut into row tiles, for [`run_active`].
struct Active {
    /// How many tiles (i32 scalar): the loop's trip count.
    steps: Val,
    /// The tiles' first sorted positions, in order, then `P` (padding).
    order: Val,
    /// The sorted experts (unrouted as `E`), then a valid id (padding).
    skey: Val,
    perm: Val,
    /// The activation rows in sorted order, `tile` zero rows after.
    xs: Val,
    tile: i64,
}

/// Sorts the pairs by expert (unrouted last) and cuts each expert's run
/// into tiles of at most `tile` rows (a token routes to an expert once, so
/// one tile per active expert unless a router repeats one).
fn plan_active(
    f: &mut Func,
    fan: Fan,
    x: Val,
    r: Val,
    valid: Val,
    tuning: Tuning,
) -> Result<Active, Error> {
    let (p, e) = (fan.pairs(), fan.experts);
    let tile = loop_tile(fan, tuning);
    let top = f.const_i(Elem::I32, e, &[p]);
    let key = f.select(valid, r, top)?;
    let iota = f.iota(Elem::I32, &[p], 0);
    let by = f.sort(&[key, iota], 0, true, |fq, a, b| {
        fq.compare(Cmp::Lt, a[0], b[0])
    })?;
    let (skey, perm) = (by[0], by[1]);
    let before = {
        let head = f.const_i(Elem::I32, e + 1, &[1]);
        let body = f.slice(skey, &[0], &[p - 1], &[1])?;
        f.concat(&[head, body], 0)?
    };
    let first = f.compare(Cmp::Ne, skey, before)?;
    let zero = f.const_i(Elem::I32, 0, &[p]);
    let at_first = f.select(first, iota, zero)?;
    let begins = f.scan(at_first, 0, Fold::Max)?;
    let offset = f.sub(iota, begins)?;
    let tt = f.const_i(Elem::I32, tile, &[p]);
    let within = f.rem(offset, tt)?;
    let starts = f.compare(Cmp::Eq, within, zero)?;
    let routed = f.compare(Cmp::Lt, skey, top)?;
    let starts = f.and(starts, routed)?;
    let one = f.const_i(Elem::I32, 1, &[p]);
    let counted = f.select(starts, one, zero)?;
    let steps = f.reduce(counted, &[0], Fold::Sum)?;
    let pp = f.const_i(Elem::I32, p, &[p]);
    let late = f.add(iota, pp)?;
    let order = f.select(starts, iota, late)?;
    let order = f.sort(&[order], 0, true, |fq, a, b| {
        fq.compare(Cmp::Lt, a[0], b[0])
    })?[0];
    // Past the tiles every entry is `P` or more: clamp to `P` (the padding
    // rows), and give that position a valid expert.
    let cap = f.const_i(Elem::I32, p, &[]);
    let lo = f.const_i(Elem::I32, 0, &[]);
    let order = f.clamp(lo, order, cap)?;
    let pad = f.const_i(Elem::I32, p, &[1]);
    let order = f.concat(&[order, pad], 0)?;
    let first_expert = f.slice(skey, &[0], &[1], &[1])?;
    let hi = f.const_i(Elem::I32, e - 1, &[]);
    let first_expert = f.clamp(lo, first_expert, hi)?;
    let skey = f.concat(&[skey, first_expert], 0)?;
    let src = if fan.per_token {
        let kk = f.const_i(Elem::I32, fan.top_k, &[p]);
        f.div(perm, kk)?
    } else {
        perm
    };
    let xs = f.take_rows(x, src)?;
    let z = f.const_f(Elem::Bf16, 0.0, &[]);
    let xs = f.pad(xs, z, &[0, 0], &[tile, 0], &[0, 0])?;
    Ok(Active {
        steps,
        order,
        skey,
        perm,
        xs,
        tile,
    })
}

/// The loop over [`plan_active`]'s tiles, two per step (a step's two dots
/// overlap): each tile's rows against its expert's slice of the bank,
/// landed in sorted order, then put back in pair order. f32 `[P, N]`; no
/// step runs when `skip` (an i1 scalar) holds.
fn run_active(
    f: &mut Func,
    plan: &Active,
    bank: &Read,
    fan: Fan,
    groups: i64,
    skip: Option<Val>,
    tuning: Tuning,
) -> Result<Val, Error> {
    let (p, n, k) = (fan.pairs(), fan.n, fan.k);
    let tile = plan.tile;
    let ys = f.const_f(Elem::F32, 0.0, &[p + tile, n]);
    let j0 = f.const_i(Elem::I32, 0, &[]);
    let per = tuning.moe_loop_unroll;
    let per_v = f.const_i(Elem::I32, per, &[]);
    let more = f.const_i(Elem::I32, per - 1, &[]);
    let steps = f.add(plan.steps, more)?;
    let steps = f.div(steps, per_v)?;
    let steps = match skip {
        Some(skip) => {
            let none = f.const_i(Elem::I32, 0, &[]);
            f.select(skip, none, steps)?
        }
        None => steps,
    };
    let (order, skey, xs) = (plan.order, plan.skey, plan.xs);
    let bank = *bank;
    let out = f.while_loop(
        &[j0, ys],
        |fq, a| fq.compare(Cmp::Lt, a[0], steps),
        |fq, a| {
            let f = fq;
            let (j, mut ys) = (a[0], a[1]);
            let step = (|| -> Result<Val, Error> {
                let per_v = f.const_i(Elem::I32, per, &[]);
                let first = f.mul(j, per_v)?;
                let mut outs = Vec::with_capacity(per as usize);
                for u in 0..per {
                    // Tiles past the count read `order`'s padding entry
                    // (`P`): they run over the padding rows.
                    let du = f.const_i(Elem::I32, u, &[]);
                    let t = f.add(first, du)?;
                    let s = scalar_at(f, order, t)?;
                    // The slice's start is `expert · N` computed here: read
                    // from a per-tile table instead, XLA stopped fusing the
                    // slice into the dot (v6e, gpt-oss: +4 ms at 64 rows).
                    let ex = scalar_at(f, skey, s)?;
                    let w = expert(f, &bank, ex, n)?;
                    let zero = f.const_i(Elem::I32, 0, &[]);
                    let xe = f.dynamic_slice(xs, &[s, zero], &[tile, k])?;
                    let ye = against_all(f, xe, &w, groups)?;
                    outs.push((s, ye));
                }
                let zero = f.const_i(Elem::I32, 0, &[]);
                for (s, ye) in outs {
                    ys = f.dynamic_update_slice(ys, ye, &[s, zero])?;
                }
                Ok(ys)
            })()
            .map_err(sealed)?;
            let one = f.const_i(Elem::I32, 1, &[]);
            let j = f.add(j, one)?;
            Ok(vec![j, step])
        },
    )?;
    let sorted = f.slice(out[1], &[0, 0], &[p, n], &[1, 1])?;
    let iota = f.iota(Elem::I32, &[p], 0);
    let back = f.sort(&[plan.perm, iota], 0, true, |fq, a, b| {
        fq.compare(Cmp::Lt, a[0], b[0])
    })?;
    Ok(f.take_rows(sorted, back[1])?)
}

/// The i32 scalar `v[at]` of a rank-1 `v` (`at` an i32 scalar).
fn scalar_at(f: &mut Func, v: Val, at: Val) -> Result<Val, Error> {
    let one = f.dynamic_slice(v, &[at], &[1])?;
    Ok(f.reshape(one, &[])?)
}

/// A kernel error raised inside a region, as the builder's.
fn sealed(e: Error) -> crate::hlo::Malformed {
    match e {
        Error::Malformed(m) => m,
        other => crate::hlo::Malformed {
            op: "linear.moe",
            detail: other.to_string(),
        },
    }
}

/// `y` (f32 `[P, N]`) plus each pair's expert bias, rounded, with the
/// unrouted pairs keeping `prev`.
fn finish(
    f: &mut Func,
    fan: Fan,
    y: Val,
    ids: Val,
    valid: Val,
    bias: Option<Val>,
    prev: Val,
) -> Result<Val, Error> {
    let (p, e, n) = (fan.pairs(), fan.experts, fan.n);
    let y = match bias {
        Some(b) => {
            let b = f.reshape(b, &[e, n])?;
            let rows = f.take_rows(b, ids)?;
            let rows = f.convert(rows, Elem::F32);
            f.add(y, rows)?
        }
        None => y,
    };
    let y = f.convert(y, Elem::Bf16);
    let keep = f.broadcast(valid, &[p, n], &[0])?;
    let prev = f.reshape(prev, &[p, n])?;
    let prev = f.convert(prev, Elem::Bf16);
    Ok(f.select(keep, y, prev)?)
}

fn select_entry(
    f: &mut Func,
    op: &'static str,
    x: Operand,
    form: Form,
    bias: Option<Operand>,
    routes: Operand,
    y: Operand,
    tuning: Tuning,
    x_tiled: Option<Tiled>,
) -> Result<(Val, Option<Tiled>), Error> {
    expect(op, x, &[Dtype::Bf16])?;
    expect(op, y, &[Dtype::Bf16])?;
    let fan = fan_of(op, x.rows, x.width, y.width, routes, form)?;
    if i64::from(y.rows) != fan.pairs() {
        return Err(refuse(
            op,
            format!(
                "the result has {} rows for {} routed pairs",
                y.rows,
                fan.pairs()
            ),
        ));
    }
    if let Some(b) = bias {
        expect(op, b, &[Dtype::Bf16])?;
        if b.elements() != (fan.experts * fan.n) as u64 {
            return Err(refuse(
                op,
                format!(
                    "the expert bias holds {} entries for {} experts of {}",
                    b.elements(),
                    fan.experts,
                    fan.n
                ),
            ));
        }
    }
    let xv = x.val;
    let rv = routes.val;
    let bv = bias.map(|b| b.val);
    let prev = y.val;
    routed(f, form, fan, xv, rv, bv, prev, tuning, x_tiled)
}

/// The GLU a routed MLP runs between its legs ([`experts`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExpertGlu {
    Swiglu,
    GegluTanh,
}

/// A routed MLP over dense banks on one sort: the gate-up select, its GLU
/// and the down select on the same routes run tile by tile in the
/// grouped-matmul kernel's sorted layout, so the rows between the legs are
/// never put back in pair order and gathered again; each row weighted
/// before the down leg, and each token's rows summed in f32. Each valid
/// pair's rows are the ones the three ops land apart, to bf16 rounding.
/// `None` (nothing landed) where the kernel does not
/// take both banks in one layout, or the fire is a slice's (`routed`).
#[allow(clippy::too_many_arguments)]
pub fn experts(
    f: &mut Func,
    x: Operand,
    gate_up: Operand,
    routes: Operand,
    packed: Landing,
    glu: ExpertGlu,
    intermediate: u32,
    down: Operand,
    routed: Landing,
    weights: Operand,
    y: Landing,
    tuning: Tuning,
) -> Result<Option<Val>, Error> {
    const OP: &str = "linear.moe_experts";
    expect(OP, x, &[Dtype::Bf16])?;
    expect(OP, weights, &[Dtype::F32])?;
    let up_fan = fan_of(
        OP,
        x.rows,
        x.width,
        packed.width,
        routes,
        Form::Dense(gate_up),
    )?;
    let down_fan = fan_of(
        OP,
        routes.rows() * routes.width(),
        intermediate,
        routed.width,
        routes,
        Form::Dense(down),
    )?;
    let p = up_fan.pairs();
    if !matches!(tuning.moe_path, MoePath::Auto { gmm: true })
        || p <= tuning.moe_costs.slice_pairs
        || i64::from(packed.rows) != p
        || i64::from(routed.rows) != p
        || packed.width != 2 * intermediate
        || down_fan.experts != up_fan.experts
        || weights.rows != y.rows
        || i64::from(weights.width) != up_fan.top_k
        || i64::from(y.rows) != up_fan.tokens
        || y.width != routed.width
    {
        return Ok(None);
    }
    let up_bank = read_bank(f, Form::Dense(gate_up), up_fan)?;
    let down_bank = read_bank(f, Form::Dense(down), down_fan)?;
    let elem = |f: &mut Func, b: &Read| f.elem(b.codes.val());
    let (ue, de) = (elem(f, &up_bank), elem(f, &down_bank));
    let (Some(up), Some(dn)) = (
        pooled_spec(
            f,
            bf16_rows(gmm_spec(up_fan, &up_bank, ue, false, tuning)),
            &up_bank,
        ),
        pooled_spec(
            f,
            bf16_rows(gmm_spec(down_fan, &down_bank, de, false, tuning)),
            &down_bank,
        ),
    ) else {
        return Ok(None);
    };
    let flat = |s: crate::linear::moe_gmm::Gmm| s.out_width().map(|w| w == s.n);
    if up.tm != dn.tm || up.tiles != dn.tiles || !flat(up)? || !flat(dn)? {
        return Ok(None);
    }
    let run = if up_fan.tokens >= tuning.moe_gmm_run_from {
        tuning.moe_gmm_run
    } else {
        1
    };
    let stepped = |s: crate::linear::moe_gmm::Gmm| crate::linear::moe_gmm::Gmm { run, ..s };
    let (up, dn) = if run > 1 && stepped(up).stepping() && stepped(dn).stepping() {
        let slack = |s: crate::linear::moe_gmm::Gmm| crate::linear::moe_gmm::Gmm {
            tiles: s.tiles + run - 1,
            ..stepped(s)
        };
        (slack(up), slack(dn))
    } else {
        (up, dn)
    };
    let e = up_fan.experts;
    let r = f.reshape(routes.val, &[p])?;
    let zero = f.const_i(Elem::I32, 0, &[p]);
    let top = f.const_i(Elem::I32, e, &[p]);
    let ge = f.compare(Cmp::Ge, r, zero)?;
    let lt = f.compare(Cmp::Lt, r, top)?;
    let valid = f.and(ge, lt)?;
    // Each pair's weight rides the sort to its row (an unrouted pair's 0).
    let t = up_fan.tokens;
    let w = f.reshape(weights.val, &[p])?;
    let none = f.const_f(Elem::F32, 0.0, &[p]);
    let w = f.select(valid, w, none)?;
    let sorted = sorted(f, up, up_fan, x.val, r, valid, Some(w))?;
    let rows = up.tiles * up.tm;
    // The down leg is linear: a row weighted before it lands weighted (the
    // f32 GLU rounds to bf16 once, as before).
    let ws = sorted
        .ws
        .ok_or_else(|| refuse(OP, "the sort carried no weights"))?;
    // The GLU and the weight in the gate-up kernel, its rows half as wide,
    // where an expert's tile holds 64 rows or more (at 16, a 64-lane decode
    // of Qwen3.6-35B-A3B's cut: 323 -> 340 us; at 64, 256 lanes: 359 -> 337)
    // and each half is whole lane tiles (Gemma-4-26B-A4B's 704: c64 3136 ->
    // 3096 tok/s), or the tile holds 128 rows (a prompt's: Gemma-4-26B-A4B's
    // 2176-row fire: 854 -> 776 us, its GLU's own pass gone).
    let inside = crate::linear::moe_gmm::Gmm {
        glu: Some(glu),
        ..up
    };
    let glu_inside = tuning.moe_gmm_glu
        && up.tm >= 64
        && (intermediate % 128 == 0 || up.tm >= 128)
        && up.stepping()
        && up.orient == crate::linear::moe_gmm::Orient::NMajor
        && inside.kernel().is_ok();
    // Both legs in one kernel, the GLU rows staying in VMEM between them.
    let fused = if glu_inside
        && tuning.moe_gmm_fuse_from > 0
        && t >= tuning.moe_gmm_fuse_from
        && let Ok(kernel) = inside.mlp_kernel(&dn)
    {
        let wl = f.broadcast(ws, &[rows, 128], &[0])?;
        let st = steps(f, inside, &sorted)?;
        let wu = up_bank.codes.val();
        let wd = down_bank.codes.val();
        let wd = match dn.orient {
            crate::linear::moe_gmm::Orient::KMajor => f.transpose(wd, &[1, 0])?,
            crate::linear::moe_gmm::Orient::NMajor => wd,
        };
        let args = pooled_args(
            f,
            &[(inside, wu), (dn, wd)],
            st.to_vec(),
            vec![sorted.xs, wu, wl, wd],
        )?;
        Some(kernel.call(f, &args)?[0])
    } else {
        None
    };
    let down_rows = match fused {
        Some(rows) => rows,
        None => {
            let hidden = if glu_inside {
                let wl = f.broadcast(ws, &[rows, 128], &[0])?;
                let st = steps(f, inside, &sorted)?;
                let w = up_bank.codes.val();
                let args = pooled_args(f, &[(inside, w)], st.to_vec(), vec![sorted.xs, w, wl])?;
                inside.kernel()?.call(f, &args)?[0]
            } else {
                let up_rows = tiles_of(f, up, &up_bank, &sorted, sorted.xs)?;
                let held = Operand {
                    val: up_rows,
                    rows: rows as u32,
                    width: packed.width,
                    dtype: Dtype::Bf16,
                };
                let into = Landing {
                    rows: rows as u32,
                    width: intermediate,
                    dtype: Dtype::Bf16,
                };
                let hidden = match glu {
                    ExpertGlu::Swiglu => crate::linear::mlp::swiglu(f, held, intermediate, into)?,
                    ExpertGlu::GegluTanh => {
                        crate::linear::mlp::geglu_tanh_packed(f, held, intermediate, into)?
                    }
                };
                let wb = f.broadcast(ws, &[rows, i64::from(intermediate)], &[0])?;
                let hidden = f.mul(hidden, wb)?;
                f.convert(hidden, Elem::Bf16)
            };
            tiles_of(f, dn, &down_bank, &sorted, hidden)?
        }
    };
    let n = i64::from(routed.width);
    // (A wide prompt's fire: a decode's lanes, and a fire of 640 rows, sum
    // by the one-hot dot, whose cells grow as the square of its tokens:
    // v6e, Qwen3.6-35B-A3B's 557-token Raw prompt 3% slower through the
    // blocks.)
    if tuning.moe_segment_tokens > 0
        && t >= tuning.moe_segment_from
        && let Some(y) = segment_sum(
            f,
            &sorted,
            up,
            up_fan,
            down_rows,
            r,
            valid,
            tuning,
            if y.dtype == Dtype::Bf16 {
                Elem::Bf16
            } else {
                Elem::F32
            },
        )?
    {
        return Ok(Some(y));
    }
    if rows * t > tuning.moe_costs.onehot_cells {
        // Each pair's (weighted) row picked back out and summed per token.
        let dest = dest(f, &sorted, p)?;
        let pairs = f.take_rows(down_rows, dest)?;
        let keep = f.broadcast(valid, &[p, n], &[0])?;
        let none = f.const_f(Elem::Bf16, 0.0, &[p, n]);
        let pairs = f.select(keep, pairs, none)?;
        let v = f.convert(pairs, Elem::F32);
        let v = f.reshape(v, &[t, up_fan.top_k, n])?;
        return Ok(Some(f.reduce(v, &[1], Fold::Sum)?));
    }
    // Each token's rows summed by one dot against the rows' tokens, one-hot:
    // picking each pair's row back out was a row gather (v6e, Gemma-4-26B's
    // 32-expert cut, 1792 tokens: 0.44 ms a layer, and 0.1 ms to sum them).
    // Rows of tiles past the count are unwritten: zeroed, not read.
    let live = {
        let tt = f.const_i(Elem::I32, up.tm, &[1]);
        let end = f.mul(sorted.nt, tt)?;
        let end = f.reshape(end, &[])?;
        let end = f.broadcast(end, &[rows, n], &[])?;
        let ri = f.iota(Elem::I32, &[rows, n], 0);
        f.compare(Cmp::Lt, ri, end)?
    };
    let zero = f.const_f(Elem::Bf16, 0.0, &[rows, n]);
    let ys = f.select(live, down_rows, zero)?;
    let hot = {
        let tb = f.broadcast(sorted.src, &[rows, t], &[0])?;
        let ts = f.iota(Elem::I32, &[rows, t], 1);
        let hit = f.compare(Cmp::Eq, tb, ts)?;
        let one = f.const_f(Elem::Bf16, 1.0, &[rows, t]);
        let off = f.const_f(Elem::Bf16, 0.0, &[rows, t]);
        f.select(hit, one, off)?
    };
    Ok(Some(f.dot_general_at(
        hot,
        ys,
        &[],
        &[],
        &[0],
        &[0],
        Elem::F32,
        false,
    )?))
}

/// Each token's rows of `ys` (`[rows, n]`, in `sorted`'s tile order) summed
/// a block of the profile's tokens at a time (`moe_gmm::SegmentSum`): each
/// block's pairs per expert, from the routes, place its run of each
/// expert's rows. `None` where the kernel's buffers do not fit.
fn segment_sum(
    f: &mut Func,
    sorted: &Sorted,
    spec: crate::linear::moe_gmm::Gmm,
    fan: Fan,
    ys: Val,
    r: Val,
    valid: Val,
    tuning: Tuning,
    out: Elem,
) -> Result<Option<Val>, Error> {
    use crate::linear::moe_gmm::SegmentSum;
    let (p, e, t, k, tm) = (fan.pairs(), fan.experts, fan.tokens, fan.top_k, spec.tm);
    let (tb, chunk) = (tuning.moe_segment_tokens, 16);
    let (rows, n) = (spec.tiles * tm, f.dims(ys)[1]);
    let nb = (t + tb - 1) / tb;
    let seg = SegmentSum {
        rows,
        n,
        experts: e,
        blocks: nb,
        tb,
        chunk,
        depth: SegmentSum::depth_of(tb, k, e, chunk),
        vmem_cap: tuning.vmem_cap(),
        out,
    };
    // (The cuts it is measured on: 32 experts, rows 2048 and 2816 wide.)
    if !seg.fits() || n % 128 != 0 || n > 4096 || e > 32 || tm % chunk != 0 {
        return Ok(None);
    }
    // Each block's pairs per expert (the pairs are token-major: block `b`
    // holds pairs `b·tb·k ..`), and the pairs of the blocks before it.
    let top = f.const_i(Elem::I32, e, &[p]);
    let key = f.select(valid, r, top)?;
    let key = if nb * tb * k > p {
        let none = f.const_i(Elem::I32, e, &[]);
        f.pad(key, none, &[0], &[nb * tb * k - p], &[0])?
    } else {
        key
    };
    let key = f.reshape(key, &[nb, tb * k])?;
    let counts = {
        let ks = f.broadcast(key, &[nb, tb * k, e], &[0, 1])?;
        let es = f.iota(Elem::I32, &[nb, tb * k, e], 2);
        let hit = f.compare(Cmp::Eq, ks, es)?;
        let one = f.const_i(Elem::I32, 1, &[nb, tb * k, e]);
        let none = f.const_i(Elem::I32, 0, &[nb, tb * k, e]);
        let hit = f.select(hit, one, none)?;
        f.reduce(hit, &[1], Fold::Sum)?
    };
    let before = f.scan(counts, 0, Fold::Sum)?;
    let before = f.sub(before, counts)?;
    // A block's run of an expert's rows, from the expert's first tile.
    let first = f.sub(sorted.tend, sorted.per)?;
    let tt = f.const_i(Elem::I32, tm, &[e]);
    let first = f.mul(first, tt)?;
    let first = f.broadcast(first, &[nb, e], &[1])?;
    let lo = f.add(first, before)?;
    let hi = f.add(lo, counts)?;
    // Whole chunks of rows from the one holding the run's first.
    let cc = f.const_i(Elem::I32, chunk, &[nb, e]);
    let start = f.div(lo, cc)?;
    let start = f.mul(start, cc)?;
    let more = f.const_i(Elem::I32, chunk - 1, &[nb, e]);
    let span = f.sub(hi, start)?;
    let span = f.add(span, more)?;
    let span = f.div(span, cc)?;
    let zero = f.const_i(Elem::I32, 0, &[nb, e]);
    let any = f.compare(Cmp::Gt, counts, zero)?;
    let cnt = f.select(any, span, zero)?;
    let off = f.scan(cnt, 1, Fold::Sum)?;
    let off = f.sub(off, cnt)?;
    let off = f.mul(off, cc)?;
    let start = f.reshape(start, &[nb * e])?;
    let cnt = f.reshape(cnt, &[nb * e])?;
    let off = f.reshape(off, &[nb * e])?;
    // Each row's token (-1 where no pair holds the row), a lane tile wide.
    let pp = f.const_i(Elem::I32, p, &[rows]);
    let pair = f.compare(Cmp::Lt, sorted.held, pp)?;
    let none = f.const_i(Elem::I32, -1, &[rows]);
    let tok = f.select(pair, sorted.src, none)?;
    let tok = f.broadcast(tok, &[rows, 128], &[0])?;
    let out = seg.kernel()?.call(f, &[start, cnt, off, tok, ys])?[0];
    Ok(Some(if nb * tb > t {
        f.slice_axis(out, 0, 0, t)?
    } else {
        out
    }))
}

/// A dense (bf16) bank: `bank` holds `[experts, N * K]` (any row shape).
pub fn matmul_select(
    f: &mut Func,
    x: Operand,
    bank: Operand,
    routes: Operand,
    y: Operand,
    tuning: Tuning,
) -> Result<Val, Error> {
    Ok(select_entry(
        f,
        "linear.moe_select",
        x,
        Form::Dense(bank),
        None,
        routes,
        y,
        tuning,
        None,
    )?
    .0)
}

pub fn matmul_select_bias(
    f: &mut Func,
    x: Operand,
    bank: Bank<Operand>,
    bias: Operand,
    routes: Operand,
    y: Operand,
    tuning: Tuning,
) -> Result<Val, Error> {
    const OP: &str = "linear.moe_select";
    let form = form_of(OP, bank)?;
    Ok(select_entry(f, OP, x, form, Some(bias), routes, y, tuning, None)?.0)
}

pub fn matmul_select_quant(
    f: &mut Func,
    x: Operand,
    bank: Bank<Operand>,
    routes: Operand,
    y: Operand,
    tuning: Tuning,
) -> Result<Val, Error> {
    const OP: &str = "linear.moe_select";
    let form = form_of(OP, bank)?;
    Ok(select_entry(f, OP, x, form, None, routes, y, tuning, None)?.0)
}

/// A quantized-bank select (with its bias or none) that also answers its
/// result in tile order where the grouped-matmul kernel ran, and reads `x`
/// in tile order where `x_tiled` (a select on the same routes) gives it.
#[allow(clippy::too_many_arguments)]
pub fn matmul_select_tiled(
    f: &mut Func,
    x: Operand,
    bank: Bank<Operand>,
    bias: Option<Operand>,
    routes: Operand,
    y: Operand,
    tuning: Tuning,
    x_tiled: Option<Tiled>,
) -> Result<(Val, Option<Tiled>), Error> {
    const OP: &str = "linear.moe_select";
    let form = form_of(OP, bank)?;
    select_entry(f, OP, x, form, bias, routes, y, tuning, x_tiled)
}

/// A tile-ordered result's rows mapped through `rows` (a row-wise op on
/// them, a GLU), the same plan ordering them.
#[must_use]
pub fn retile(t: Tiled, rows: Val, width: i64) -> Tiled {
    Tiled {
        rows,
        width,
        spec: crate::linear::moe_gmm::Gmm {
            out: Elem::Bf16,
            ..t.spec
        },
        ..t
    }
}

/// [`weighted_sum`] of a routed result held in tile order: each slot's rows
/// gathered slot-major (`[top_k, T, N]`, a free fold) where the pair-major
/// fold relays every row out. An unrouted slot adds nothing (its weight is
/// the router's zero; the select leaves its row as it was, which no sum
/// should read).
pub fn weighted_sum_tiled(
    f: &mut Func,
    t: Tiled,
    n: u32,
    weights: Operand,
    y: Landing,
) -> Result<Val, Error> {
    const OP: &str = "linear.moe_combine";
    expect(OP, weights, &[Dtype::F32])?;
    let plan = t.plan;
    let (tok, k, w) = (i64::from(y.rows), plan.top_k, i64::from(n));
    if tok * k != plan.pairs || i64::from(y.width) != w {
        return Err(refuse(OP, "the tiled rows do not fold into the landing"));
    }
    let slot_major = |f: &mut Func, v: Val| -> Result<Val, Error> {
        let v = f.reshape(v, &[tok, k])?;
        let v = f.transpose(v, &[1, 0])?;
        Ok(f.reshape(v, &[k * tok])?)
    };
    let dest = tile_dest(f, &plan)?;
    let dest = slot_major(f, dest)?;
    let rows = untile(f, w, t.rows, dest)?;
    let r = f.reshape(rows, &[k, tok, w])?;
    let r = f.convert(r, Elem::F32);
    let zero = f.const_f(Elem::F32, 0.0, &[tok, k]);
    let valid = f.reshape(plan.valid, &[tok, k])?;
    let g = f.select(valid, weights.val, zero)?;
    let g = f.transpose(g, &[1, 0])?;
    let g = f.broadcast(g, &[k, tok, w], &[0, 1])?;
    let v = f.mul(r, g)?;
    Ok(f.reduce(v, &[0], Fold::Sum)?)
}

/// A grouped matmul's weight: a quantized bank, or a dense bf16 plane.
#[derive(Clone, Copy)]
pub enum GroupedPlane<T = Operand> {
    Bank(Bank<T>),
    Dense(T),
}

/// `x [rows, groups*K]` split into `groups` slices, slice `g` of row `r`
/// multiplied by expert `routes[r, g]`, into `y [rows, groups*N]`.
pub fn matmul_grouped(
    f: &mut Func,
    x: Operand,
    plane: GroupedPlane,
    routes: Operand,
    groups: u32,
    y: Operand,
    tuning: Tuning,
) -> Result<Val, Error> {
    const OP: &str = "linear.matmul_grouped";
    let groups = nonzero(OP, "the group count", groups)?;
    expect(OP, x, &[Dtype::Bf16])?;
    expect(OP, y, &[Dtype::Bf16])?;
    if !x.width.is_multiple_of(groups)
        || !y.width.is_multiple_of(groups)
        || routes.width != groups
        || routes.rows != x.rows
        || y.rows != x.rows
    {
        return Err(refuse(
            OP,
            format!(
                "{groups} groups do not divide a {}x{} row into a {}x{} one, or the routes are {}x{}",
                x.rows, x.width, y.rows, y.width, routes.rows, routes.width
            ),
        ));
    }
    let form = match plane {
        GroupedPlane::Bank(bank) => form_of(OP, bank)?,
        GroupedPlane::Dense(dense) => Form::Dense(dense),
    };
    let (k, n) = (x.width / groups, y.width / groups);
    let fan = fan_of(OP, x.rows * groups, k, n, routes, form)?;
    let xv = x.val;
    let xv = f.reshape(xv, &[fan.pairs(), fan.k])?;
    let rv = routes.val;
    let prev = y.val;
    Ok(routed(f, form, fan, xv, rv, None, prev, tuning, None)?.0)
}

/// [`matmul_grouped`] under routes that send each row's slot `g` to group
/// `g` (`linear.group_routes`): `x [rows, groups*K]` against a dense bf16
/// `w [groups*N, K]`, one dot batched over the groups, f32. Sorting the
/// slots by a group every row has, and gathering them back, cost 0.8 of a
/// 68 ms DeepSeek-V4 prompt fire on a v6e a layer.
pub fn matmul_grouped_fixed(
    f: &mut Func,
    x: Operand,
    w: Operand,
    groups: u32,
    y: Landing,
) -> Result<Val, Error> {
    const OP: &str = "linear.matmul_grouped";
    let groups = nonzero(OP, "the group count", groups)?;
    expect(OP, x, &[Dtype::Bf16])?;
    expect(OP, w, &[Dtype::Bf16])?;
    let g = i64::from(groups);
    let (rows, k, n) = (
        i64::from(x.rows),
        i64::from(x.width) / g,
        i64::from(y.width) / g,
    );
    if !x.width.is_multiple_of(groups)
        || !y.width.is_multiple_of(groups)
        || y.rows != x.rows
        || i64::from(w.width) != k
        || i64::from(w.rows) != g * n
    {
        return Err(refuse(
            OP,
            format!(
                "{groups} groups do not split a {}x{} row against a {}x{} plane into a {}x{} one",
                x.rows, x.width, w.rows, w.width, y.rows, y.width
            ),
        ));
    }
    let xv = f.slice_axis(x.val, 0, 0, rows)?;
    let xv = f.reshape(xv, &[rows, g, k])?;
    let wv = f.reshape(w.val, &[g, n, k])?;
    let v = f.dot_general(xv, wv, &[1], &[0], &[2], &[2], Elem::F32)?;
    let v = f.transpose(v, &[1, 0, 2])?;
    Ok(f.reshape(v, &[rows, g * n])?)
}

/// `y[t] = sum_s weights[t, s] * routed[t*k + s]`, in f32.
pub fn weighted_sum(
    f: &mut Func,
    routed: Operand,
    weights: Operand,
    y: Landing,
) -> Result<Val, Error> {
    const OP: &str = "linear.moe_combine";
    expect(OP, routed, &[Dtype::Bf16])?;
    expect(OP, weights, &[Dtype::F32])?;
    nonzero(OP, "the token rows", y.rows)?;
    if !routed.rows.is_multiple_of(y.rows) || routed.width != y.width {
        return Err(refuse(
            OP,
            format!(
                "the routed {}x{} rectangle does not fold into {}x{}",
                routed.rows, routed.width, y.rows, y.width
            ),
        ));
    }
    let top_k = routed.rows / y.rows;
    if weights.rows != y.rows || weights.width != top_k {
        return Err(refuse(
            OP,
            format!(
                "the weights are {}x{} for fan-out {top_k}",
                weights.rows, weights.width
            ),
        ));
    }
    let (t, k, w) = (i64::from(y.rows), i64::from(top_k), i64::from(y.width));
    // Folded while bf16: the fold relays the rows out (a token's `k` rows
    // pad a sublane tile), at half the bytes f32 rows take.
    let r = f.reshape(routed.val, &[t, k, w])?;
    let r = f.convert(r, Elem::F32);
    let g = weights.val;
    let g = f.broadcast(g, &[t, k, w], &[0, 1])?;
    let v = f.mul(r, g)?;
    let v = f.reduce(v, &[1], Fold::Sum)?;
    Ok(v)
}

/// `y = x + sum_s weights[t, s] * bias[routes[t, s]]` over routed slots
/// (negative routes skipped; `bias` is `[experts, width]`).
pub fn bias_sum(
    f: &mut Func,
    x: Operand,
    bias: Operand,
    routes: Operand,
    weights: Operand,
    y: Landing,
) -> Result<Val, Error> {
    const OP: &str = "linear.moe_bias_sum";
    expect(OP, x, &[Dtype::Bf16])?;
    expect(OP, bias, &[Dtype::Bf16])?;
    expect(OP, routes, &[Dtype::I32])?;
    expect(OP, weights, &[Dtype::F32])?;
    let top_k = nonzero(OP, "the routed fan-out", routes.width)?;
    if x.rows != y.rows
        || x.width != y.width
        || routes.rows != y.rows
        || weights.rows != y.rows
        || weights.width != top_k
        || !bias.elements().is_multiple_of(u64::from(y.width))
        || bias.elements() == 0
    {
        return Err(refuse(
            OP,
            "the token, route, weight and bias planes do not agree",
        ));
    }
    let experts = (bias.elements() / u64::from(y.width)) as i64;
    let (t, k, w) = (i64::from(y.rows), i64::from(top_k), i64::from(y.width));
    let xv = x.f32(f);
    let b = bias.val;
    let b = f.reshape(b, &[experts, w])?;
    // Each row's weights spread over the experts (`[T, E]`, nothing where a
    // slot is unrouted) against the bias table in one f32 product, where a
    // bias row gathered a pair and folded cost ~0.2 ms a layer (v5e gpt-oss
    // 1792-row prefill).
    let r = f.reshape(routes.val, &[t, k])?;
    let r = f.broadcast(r, &[t, k, experts], &[0, 1])?;
    let at = f.iota(Elem::I32, &[t, k, experts], 2);
    let hit = f.compare(Cmp::Eq, r, at)?;
    let g = f.reshape(weights.val, &[t, k])?;
    let g = f.broadcast(g, &[t, k, experts], &[0, 1])?;
    let zf = f.const_f(Elem::F32, 0.0, &[t, k, experts]);
    let g = f.select(hit, g, zf)?;
    let g = f.reduce(g, &[1], Fold::Sum)?;
    let b = f.convert(b, Elem::F32);
    let v = f.dot_general(g, b, &[], &[], &[1], &[0], Elem::F32)?;
    let v = f.add(xv, v)?;
    Ok(v)
}

/// `y = routed + sigmoid(gate[row]) * shared`.
pub fn sigmoid_gate_add(
    f: &mut Func,
    routed: Operand,
    shared: Operand,
    gate: Operand,
    y: Landing,
) -> Result<Val, Error> {
    const OP: &str = "linear.moe_sigmoid_gate_add";
    expect(OP, routed, &[Dtype::Bf16])?;
    expect(OP, shared, &[Dtype::Bf16])?;
    expect(OP, gate, &[Dtype::Bf16])?;
    if shared.rows != routed.rows
        || shared.width != routed.width
        || gate.elements() != u64::from(routed.rows)
        || y.rows != routed.rows
        || y.width != routed.width
    {
        return Err(refuse(
            OP,
            "the routed, shared, gate and result planes do not agree",
        ));
    }
    let (t, w) = (i64::from(routed.rows), i64::from(routed.width));
    let r = routed.f32(f);
    let s = shared.f32(f);
    let g = gate.f32(f);
    let g = f.reshape(g, &[t])?;
    let g = score(f, g, Score::Sigmoid)?;
    let g = f.broadcast(g, &[t, w], &[0])?;
    let v = f.mul(g, s)?;
    let v = f.add(r, v)?;
    Ok(v)
}
