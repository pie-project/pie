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
//! The bytes floor is the active experts' bytes: at batch 64 with top-4 of
//! 32 experts and diverse tokens nearly every expert is active and the
//! whole bank streams (e5m2: ~0.55 ms a gpt-oss layer on v6e). An exact
//! 4-bit form does not help there: decoding e2m1 with its group scale
//! inside the dot's operand takes more VPU work per weight than streaming
//! e5m2 costs (measured: 0.8 ms for the gate/up bank as `i4` codes, 2 ms
//! as `f4E2M1FN`, 0.45 ms pre-scaled).
//!
//! References: `kernels-wgpu/src/linear/moe.rs` + `kernels/moe/{route,
//! qmv_routed}.wgsl`; the sink router from `kernels-cuda/kernels/linear/
//! moe.cuh` (`moe_topk_sigmoid_sink`).
#![allow(clippy::too_many_arguments)]

use dtype::Dtype;

use crate::cx::{Ctx, Cx, expect};
use crate::error::{Error, refuse};
use crate::hlo::{Cmp, Elem, Fold, Func, GatherDims, Ty, Val};
use crate::linear::quant::{
    Codes, code_values, e8m0, grouped_dot, materialize, offset_dot, read_codes,
};
use crate::tensor::{Bank, Tensor};

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
    routes: Tensor,
    weights: Tensor,
) -> Result<(), Error> {
    expect(op, routes, &[Dtype::I32])?;
    expect(op, weights, &[Dtype::F32])?;
    if routes.rows != rows
        || weights.rows != rows
        || routes.width != top_k
        || weights.width != top_k
    {
        return Err(refuse(
            op,
            format!(
                "routes {}x{} / weights {}x{} for {rows} rows at fan-out {top_k}",
                routes.rows, routes.width, weights.rows, weights.width
            ),
        ));
    }
    Ok(())
}

fn router(
    op: &'static str,
    logits: Tensor,
    experts: u32,
    top_k: u32,
    routes: Tensor,
    weights: Tensor,
) -> Result<(), Error> {
    expect(op, logits, &[Dtype::Bf16])?;
    nonzero(op, "the expert count", experts)?;
    nonzero(op, "the fan-out", top_k)?;
    if logits.width != experts {
        return Err(refuse(
            op,
            format!("the logits are {} wide for {experts} experts", logits.width),
        ));
    }
    planes(op, logits.rows, top_k, routes, weights)
}

/// `v` as a flat f32 `[n]` vector.
fn flat_f32(cx: &mut Cx<'_>, v: Val) -> Result<Val, Error> {
    let n = cx.ty(v).elements();
    let v = cx.convert(v, Elem::F32);
    Ok(cx.reshape(v, &[n])?)
}

/// The `k` largest of `rank` (`[rows, n]` f32) per row, ties to the lowest
/// index, NaN never before a number: `(values [rows, k], index [rows, k] i32)`.
/// `carry` (same shape as `rank`) rides the sort and is returned picked the
/// same way.
fn pick_top(
    cx: &mut Cx<'_>,
    rank: Val,
    carry: Option<Val>,
    k: i64,
) -> Result<(Val, Val, Option<Val>), Error> {
    let dims = cx.dims(rank).to_vec();
    let (rows, n) = (dims[0], dims[1]);
    let clean = {
        let nan = cx.compare(Cmp::Ne, rank, rank)?;
        let lo = cx.const_f(Elem::F32, f64::NEG_INFINITY, &dims);
        cx.select(nan, lo, rank)?
    };
    let iota = cx.iota(Elem::I32, &dims, 1);
    let mut xs = vec![clean, iota];
    if let Some(c) = carry {
        xs.push(c);
    }
    let sorted = cx.sort(&xs, 1, true, |f, a, b| {
        let gt = f.compare(Cmp::Gt, a[0], b[0])?;
        let eq = f.compare(Cmp::Eq, a[0], b[0])?;
        let lt = f.compare(Cmp::Lt, a[1], b[1])?;
        let tie = f.and(eq, lt)?;
        f.or(gt, tie)
    })?;
    let k = k.min(n);
    let v = cx.slice(sorted[0], &[0, 0], &[rows, k], &[1, 1])?;
    let i = cx.slice(sorted[1], &[0, 0], &[rows, k], &[1, 1])?;
    let c = match carry {
        Some(_) => Some(cx.slice(sorted[2], &[0, 0], &[rows, k], &[1, 1])?),
        None => None,
    };
    Ok((v, i, c))
}

/// Widens `[rows, picks]` to `[rows, width]` with `fill` in the tail.
fn pad_cols(cx: &mut Cx<'_>, v: Val, width: i64, fill: Val) -> Result<Val, Error> {
    let picks = cx.dims(v)[1];
    if picks == width {
        return Ok(v);
    }
    Ok(cx.pad(v, fill, &[0, 0], &[0, width - picks], &[0, 0])?)
}

/// Per-row `sum(w)`, broadcast back over `[rows, k]`.
fn row_sum(cx: &mut Cx<'_>, w: Val) -> Result<Val, Error> {
    let dims = cx.dims(w).to_vec();
    let s = cx.reduce(w, &[1], Fold::Sum)?;
    Ok(cx.broadcast(s, &dims, &[0])?)
}

/// `w * scaling`, or `w * scaling / sum(w)` when renormalizing and the row's
/// sum is positive (the WGSL routers' rule).
fn renorm(cx: &mut Cx<'_>, w: Val, renormalize: bool, scaling: f32) -> Result<Val, Error> {
    let scaled = cx.scale(w, f64::from(scaling))?;
    if !renormalize {
        return Ok(scaled);
    }
    let sum = row_sum(cx, w)?;
    let zero = cx.like_f(sum, 0.0);
    let pos = cx.compare(Cmp::Gt, sum, zero)?;
    let divided = cx.div(scaled, sum)?;
    Ok(cx.select(pos, divided, scaled)?)
}

fn softmax_router(
    ctx: &Ctx<'_>,
    op: &'static str,
    logits: Tensor,
    scale: Option<Tensor>,
    experts: u32,
    top_k: u32,
    routes: Tensor,
    weights: Tensor,
) -> Result<(), Error> {
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
    ctx.emit(&mut |cx| {
        let k = i64::from(top_k);
        let x = cx.read_f32(logits)?;
        let (v, i, _) = top_k_fill(cx, x, k)?;
        // Softmax over the chosen logits.
        let w = cx.softmax(v, 1)?;
        let w = match scale {
            Some(s) => {
                let s = cx.read(s)?;
                let s = flat_f32(cx, s)?;
                let picked = take_flat(cx, s, i)?;
                cx.mul(w, picked)?
            }
            None => w,
        };
        cx.write(routes, i)?;
        cx.write(weights, w)
    })
}

/// `top_k` for a fan-out that may exceed the row: picks past the row's end
/// hold `-inf`/index `-1` (they get zero softmax weight).
fn top_k_fill(cx: &mut Cx<'_>, x: Val, k: i64) -> Result<(Val, Val, Option<Val>), Error> {
    let (v, i, c) = pick_top(cx, x, None, k)?;
    let lo = cx.const_f(Elem::F32, f64::NEG_INFINITY, &[]);
    let none = cx.const_i(Elem::I32, -1, &[]);
    let v = pad_cols(cx, v, k, lo)?;
    let i = pad_cols(cx, i, k, none)?;
    Ok((v, i, c))
}

/// `table[ids]` elementwise for a flat `[n]` table and any-shaped i32 `ids`
/// (clamped into range).
fn take_flat(cx: &mut Cx<'_>, table: Val, ids: Val) -> Result<Val, Error> {
    let n = cx.dims(table)[0];
    let dims = cx.dims(ids).to_vec();
    let lo = cx.const_i(Elem::I32, 0, &[]);
    let hi = cx.const_i(Elem::I32, n - 1, &[]);
    let ids = cx.clamp(lo, ids, hi)?;
    let mut with = dims.clone();
    with.push(1);
    let ids = cx.reshape(ids, &with)?;
    Ok(cx.gather(
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
fn take_cols(cx: &mut Cx<'_>, x: Val, ids: Val) -> Result<Val, Error> {
    let (rows, n) = (cx.dims(x)[0], cx.dims(x)[1]);
    let k = cx.dims(ids)[1];
    let flat = cx.reshape(x, &[rows * n])?;
    let base = cx.iota(Elem::I32, &[rows, k], 0);
    let stride = cx.const_i(Elem::I32, n, &[rows, k]);
    let base = cx.mul(base, stride)?;
    let lo = cx.const_i(Elem::I32, 0, &[]);
    let hi = cx.const_i(Elem::I32, n - 1, &[]);
    let ids = cx.clamp(lo, ids, hi)?;
    let at = cx.add(base, ids)?;
    take_flat(cx, flat, at)
}

pub fn topk_softmax(
    ctx: &Ctx<'_>,
    logits: Tensor,
    experts: u32,
    top_k: u32,
    routes: Tensor,
    weights: Tensor,
) -> Result<(), Error> {
    softmax_router(
        ctx,
        "linear.moe_topk_softmax",
        logits,
        None,
        experts,
        top_k,
        routes,
        weights,
    )
}

pub fn topk_softmax_scaled(
    ctx: &Ctx<'_>,
    logits: Tensor,
    scale: Tensor,
    experts: u32,
    top_k: u32,
    routes: Tensor,
    weights: Tensor,
) -> Result<(), Error> {
    softmax_router(
        ctx,
        "linear.moe_topk_softmax_scaled",
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

fn score(cx: &mut Cx<'_>, x: Val, how: Score) -> Result<Val, Error> {
    Ok(match how {
        Score::Sigmoid => {
            // 1 / (1 + e^-x), as the routers compute it.
            let nx = cx.neg(x);
            let e = cx.exp(nx);
            let d = cx.offset(e, 1.0)?;
            let one = cx.like_f(d, 1.0);
            cx.div(one, d)?
        }
        Score::SqrtSoftplus => {
            let e = cx.exp(x);
            let sp = cx.offset(e, 1.0)?;
            let sp = cx.log(sp);
            let twenty = cx.like_f(x, 20.0);
            let big = cx.compare(Cmp::Gt, x, twenty)?;
            let sp = cx.select(big, x, sp)?;
            let zero = cx.like_f(sp, 0.0);
            let sp = cx.max(sp, zero)?;
            cx.sqrt(sp)
        }
    })
}

/// The sigmoid-family routers: rank by `score + bias`, weigh by `score`,
/// slots past the expert count land expert 0 at weight 0.
fn scored_router(
    ctx: &Ctx<'_>,
    op: &'static str,
    how: Score,
    logits: Tensor,
    bias: Option<Tensor>,
    experts: u32,
    top_k: u32,
    renormalize: bool,
    scaling: f32,
    routes: Tensor,
    weights: Tensor,
) -> Result<(), Error> {
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
    ctx.emit(&mut |cx| {
        let (rows, n, k) = (i64::from(logits.rows), i64::from(experts), i64::from(top_k));
        let x = cx.read_f32(logits)?;
        let s = score(cx, x, how)?;
        let rank = match bias {
            Some(b) => {
                let b = cx.read(b)?;
                let b = flat_f32(cx, b)?;
                let b = cx.broadcast(b, &[rows, n], &[1])?;
                cx.add(s, b)?
            }
            None => s,
        };
        let (_, i, w) = pick_top(cx, rank, Some(s), k)?;
        let w = w.expect("the score rides the sort");
        let zero_f = cx.const_f(Elem::F32, 0.0, &[]);
        let zero_i = cx.const_i(Elem::I32, 0, &[]);
        let w = pad_cols(cx, w, k, zero_f)?;
        let i = pad_cols(cx, i, k, zero_i)?;
        let w = renorm(cx, w, renormalize, scaling)?;
        cx.write(routes, i)?;
        cx.write(weights, w)
    })
}

pub fn topk_sigmoid(
    ctx: &Ctx<'_>,
    logits: Tensor,
    experts: u32,
    top_k: u32,
    renormalize: bool,
    scaling: f32,
    routes: Tensor,
    weights: Tensor,
) -> Result<(), Error> {
    scored_router(
        ctx,
        "linear.moe_topk_sigmoid",
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
    ctx: &Ctx<'_>,
    logits: Tensor,
    bias: Tensor,
    experts: u32,
    top_k: u32,
    renormalize: bool,
    scaling: f32,
    routes: Tensor,
    weights: Tensor,
) -> Result<(), Error> {
    scored_router(
        ctx,
        "linear.moe_topk_sigmoid",
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
    ctx: &Ctx<'_>,
    logits: Tensor,
    bias: Tensor,
    experts: u32,
    top_k: u32,
    renormalize: bool,
    scaling: f32,
    routes: Tensor,
    weights: Tensor,
) -> Result<(), Error> {
    scored_router(
        ctx,
        "linear.moe_topk_sqrt_softplus",
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
    ctx: &Ctx<'_>,
    logits: Tensor,
    bias: Tensor,
    experts: u32,
    top_k: u32,
    routes: Tensor,
    weights: Tensor,
) -> Result<(), Error> {
    scored_router(
        ctx,
        "linear.moe_predict_route",
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
    ctx: &Ctx<'_>,
    logits: Tensor,
    correction_bias: Option<Tensor>,
    global_scale: Option<Tensor>,
    experts: u32,
    top_k: u32,
    sink: u32,
    scaling: f32,
    routes: Tensor,
    weights: Tensor,
) -> Result<(), Error> {
    const OP: &str = "linear.moe_topk_sigmoid_sink";
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
    ctx.emit(&mut |cx| {
        let (rows, n, k, s) = (
            i64::from(logits.rows),
            i64::from(experts),
            i64::from(top_k),
            i64::from(sink),
        );
        let x = cx.read_f32(logits)?;
        let sc = score(cx, x, Score::Sigmoid)?;
        let routed = cx.slice(sc, &[0, 0], &[rows, n], &[1, 1])?;
        let rank = match correction_bias {
            Some(b) => {
                let b = cx.read(b)?;
                let b = flat_f32(cx, b)?;
                let b = cx.slice(b, &[0], &[n], &[1])?;
                let b = cx.broadcast(b, &[rows, n], &[1])?;
                cx.add(routed, b)?
            }
            None => routed,
        };
        let (_, i, w) = pick_top(cx, rank, Some(routed), k)?;
        let w = w.expect("the score rides the sort");
        let zero_f = cx.const_f(Elem::F32, 0.0, &[]);
        let none = cx.const_i(Elem::I32, -1, &[]);
        let mut w = pad_cols(cx, w, k, zero_f)?;
        let mut i = pad_cols(cx, i, k, none)?;
        if s > 0 {
            let sw = cx.slice(sc, &[0, n], &[rows, n + s], &[1, 1])?;
            let si = cx.iota(Elem::I32, &[rows, s], 1);
            let base = cx.const_i(Elem::I32, n, &[rows, s]);
            let si = cx.add(si, base)?;
            w = cx.concat(&[w, sw], 1)?;
            i = cx.concat(&[i, si], 1)?;
        }
        let sum = row_sum(cx, w)?;
        let sum = cx.offset(sum, 1e-20)?;
        let mut scale = cx.like_f(sum, f64::from(scaling));
        if let Some(g) = global_scale {
            let g = cx.read(g)?;
            let g = flat_f32(cx, g)?;
            let g = cx.slice(g, &[0], &[1], &[1])?;
            let g = cx.reshape(g, &[])?;
            let g = cx.splat(g, &[rows, k + s])?;
            scale = cx.mul(scale, g)?;
        }
        let scale = cx.div(scale, sum)?;
        let w = cx.mul(w, scale)?;
        cx.write(routes, i)?;
        cx.write(weights, w)
    })
}

/// Routes by a token-id hash table: `tid2eid[id]` (i64 `[vocab, top_k]`)
/// names each slot's expert, weighed by `sqrt(softplus(logit))` of that
/// expert; an id past the vocabulary reads row 0, and a table entry outside
/// `0..experts` lands its (truncated) id at weight 0.
pub fn hash_route(
    ctx: &Ctx<'_>,
    ids: Tensor,
    tid2eid: Tensor,
    logits: Tensor,
    vocab: u32,
    top_k: u32,
    renormalize: bool,
    scaling: f32,
    routes: Tensor,
    weights: Tensor,
) -> Result<(), Error> {
    const OP: &str = "linear.moe_hash_route";
    expect(OP, ids, &[Dtype::U32, Dtype::I32])?;
    expect(OP, tid2eid, &[Dtype::I64])?;
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
    if tid2eid.elements() < u64::from(vocab) * u64::from(top_k) {
        return Err(refuse(OP, "the hash table is smaller than vocab x top_k"));
    }
    ctx.emit(&mut |cx| {
        let (rows, n, k, v) = (
            i64::from(routes.rows),
            i64::from(experts),
            i64::from(top_k),
            i64::from(vocab),
        );
        let raw = cx.read(ids)?;
        let raw = cx.reshape(raw, &[rows])?;
        // Unsigned compare against the vocabulary, as the WGSL reads a u32.
        let raw = cx.convert(raw, Elem::I64);
        let raw = if ids.dtype == Dtype::I32 {
            let m = cx.const_i(Elem::I64, 0xFFFF_FFFF, &[rows]);
            cx.and(raw, m)?
        } else {
            raw
        };
        let vv = cx.const_i(Elem::I64, v, &[rows]);
        let inside = cx.compare(Cmp::Lt, raw, vv)?;
        let zero = cx.const_i(Elem::I64, 0, &[rows]);
        let tid = cx.select(inside, raw, zero)?;
        let tid = cx.convert(tid, Elem::I32);
        let table = cx.read(tid2eid)?;
        let total = cx.ty(table).elements();
        let table = cx.reshape(table, &[total / k, k])?;
        let eid = cx.take_rows(table, tid)?; // [rows, k] i64
        let lo = cx.const_i(Elem::I64, 0, &[rows, k]);
        let hi = cx.const_i(Elem::I64, n, &[rows, k]);
        let ge = cx.compare(Cmp::Ge, eid, lo)?;
        let lt = cx.compare(Cmp::Lt, eid, hi)?;
        let ok = cx.and(ge, lt)?;
        let route = cx.convert(eid, Elem::I32);
        let x = cx.read_f32(logits)?;
        let picked = take_cols(cx, x, route)?;
        // The WGSL's hash score: sqrt(softplus) with no clamp at zero.
        let e = cx.exp(picked);
        let sp = cx.offset(e, 1.0)?;
        let sp = cx.log(sp);
        let twenty = cx.like_f(picked, 20.0);
        let big = cx.compare(Cmp::Gt, picked, twenty)?;
        let sp = cx.select(big, picked, sp)?;
        let w = cx.sqrt(sp);
        let zf = cx.const_f(Elem::F32, 0.0, &[rows, k]);
        let w = cx.select(ok, w, zf)?;
        let w = renorm(cx, w, renormalize, scaling)?;
        cx.write(routes, route)?;
        cx.write(weights, w)
    })
}

/// `routes[row, slot] = slot`: the identity routing a grouped matmul reads.
pub fn group_routes(ctx: &Ctx<'_>, groups: u32, routes: Tensor) -> Result<(), Error> {
    const OP: &str = "linear.group_routes";
    expect(OP, routes, &[Dtype::I32])?;
    nonzero(OP, "the group count", groups)?;
    if routes.width != groups {
        return Err(refuse(
            OP,
            format!("the routes are {} wide for {groups} groups", routes.width),
        ));
    }
    ctx.emit(&mut |cx| {
        let v = cx.iota(Elem::I32, &[i64::from(routes.rows), i64::from(groups)], 1);
        cx.write(routes, v)
    })
}

// ------------------------------------------------------------ expert compute

/// Up to this many f32 partials (`rows · experts · N`, times the group count
/// for a codes bank) a routed matmul runs every activation row against every
/// expert in one dot and picks each pair's row: the whole bank streams
/// through the MXU once with only a free convert in front of it (measured on
/// v6e: gpt-oss's 32-expert gate/up bank in 0.45 ms pre-scaled, at HBM
/// speed). Past it, the pairs are sorted by expert into one ragged dot over
/// the bank decoded to bf16.
const DENSE_ALL: i64 = 1 << 27;

/// What multiplying one expert sliced out of the bank costs at least, in
/// bank bytes streamed: a dot over a dynamic slice of an expert's rows runs
/// ~15 us on v6e however small the expert (measured: gpt-oss's 16.6 MB
/// gate/up expert at the HBM rate, its 8.3 MB down expert in the same time).
/// Independent slices overlap; a loop over them pays this per step.
const SLICE_BYTES: i64 = 18 << 20;

/// Up to this many routed pairs, a fire slices each pair's expert out and
/// multiplies its row alone (one or two tokens' fan-out), when that costs
/// less than streaming the bank (gpt-oss at one token: 0.11 ms a layer
/// against 0.67 ms for the banks).
const SLICE_PAIRS: i64 = 8;

/// What one row tile of the loop over active experts costs, in
/// nanoseconds (v6e, gpt-oss at 64 tokens, real routing): an expert's dot
/// over a dynamic slice of the bank runs at the HBM rate (~1,200 bytes a ns)
/// but not under ~14 us, whatever the expert (its 8.3 MB down expert takes
/// 14.6 us, its 16.6 MB gate/up expert 13.8 us).
const LOOP_TILE_NS: i64 = 14_000;
const LOOP_ROW_NS: i64 = 10;

/// What streaming the whole bank through one dot costs, in nanoseconds:
/// the bank at ~1,450 bytes a ns plus ~5.5 ps a partial (every row
/// against every expert, then picked; the pick relayouts them), measured
/// as for the loop (gate/up at 64 rows 0.38 ms, down at 256 rows 0.33 ms).
const WHOLE_BYTES_PER_NS: i64 = 1_450;
const WHOLE_PARTIAL_FS: i64 = 5_500;

thread_local! {
    /// Tiles the loop over active experts multiplies per step
    /// (`PIE_XLA_MOE_UNROLL`, default 2).
    static LOOP_UNROLL: i64 = std::env::var("PIE_XLA_MOE_UNROLL")
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|&u: &i64| (1..=8).contains(&u))
        .unwrap_or(2);
}

/// How an expert bank is stored. Every form is expert-major and flat:
/// expert `e`, output row `n`, contracted column `k` sits at `(e*N + n)*K + k`
/// (one scale/zero point per `group` codes of a row), however the engine
/// shapes the plane; codes are one per element as `crate::pack` lands them,
/// pre-scaled mxfp4 weights (`E5m2`), or an integer plane packing several
/// (low bits first).
#[derive(Clone, Copy)]
enum Form {
    /// bf16 weights.
    Dense(Tensor),
    /// `scale * q + bias`, `bits`-wide unsigned codes, bf16 factors.
    Affine {
        codes: Tensor,
        scales: Tensor,
        biases: Tensor,
        group: u32,
        bits: u32,
    },
    /// e2m1 codes, one e8m0 exponent byte per 32 (or pre-scaled weights).
    Mxfp4 { codes: Tensor, scales: Tensor },
}

impl Form {
    fn bits(self) -> u32 {
        match self {
            Self::Dense(_) => 16,
            Self::Affine { bits, .. } => bits,
            Self::Mxfp4 { .. } => 4,
        }
    }

    fn group(self) -> u32 {
        match self {
            Self::Dense(_) => 1,
            Self::Affine { group, .. } => group,
            Self::Mxfp4 { .. } => MXFP4_BLOCK,
        }
    }
}

fn form_of(op: &'static str, bank: Bank) -> Result<Form, Error> {
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
fn stored_codes(op: &'static str, codes: Tensor, bits: u32) -> Result<u64, Error> {
    if codes.dtype == Dtype::E5m2 {
        return Ok(codes.elements());
    }
    if let Some(per) = crate::pack::codes_per_row(codes.dtype, u64::from(codes.width)) {
        if crate::pack::code_bits(codes.dtype) != Some(bits) {
            return Err(refuse(
                op,
                format!("a {:?} codes plane in a {bits}-bit bank", codes.dtype),
            ));
        }
        return Ok(u64::from(codes.rows) * per);
    }
    match crate::cx::elem_of(op, codes.dtype) {
        Ok(e) if e.is_int() => Ok(codes.elements() * u64::from(e.bits()) / u64::from(bits)),
        _ => Err(refuse(
            op,
            format!(
                "a {:?} codes plane; codes land one per element (`kernels_xla::pack`) or in an \
                 integer plane",
                codes.dtype
            ),
        )),
    }
}

fn fan_of(
    op: &'static str,
    x_rows: u32,
    k: u32,
    n: u32,
    routes: Tensor,
    form: Form,
) -> Result<Fan, Error> {
    expect(op, routes, &[Dtype::I32])?;
    nonzero(op, "the routed fan-out", routes.width)?;
    nonzero(op, "K, the activation's width", k)?;
    nonzero(op, "N, the output width", n)?;
    let pairs = u64::from(routes.rows) * u64::from(routes.width);
    let per_token = if x_rows == routes.rows {
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
        Form::Affine { codes, scales, .. } | Form::Mxfp4 { codes, scales } => {
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
            let factors = experts * n64 * (k64 / group);
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
        tokens: i64::from(routes.rows),
        top_k: i64::from(routes.width),
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

fn read_bank(cx: &mut Cx<'_>, form: Form, fan: Fan) -> Result<Read, Error> {
    let rows = fan.experts * fan.n;
    let groups = fan.k / i64::from(form.group());
    Ok(match form {
        Form::Dense(bank) => {
            let w = cx.read(bank)?;
            Read {
                codes: Codes::Weights(cx.reshape(w, &[rows, fan.k])?),
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
            let c = read_codes(cx, codes, bits, false, rows, fan.k)?;
            let s = cx.read_f32(scales)?;
            let b = cx.read_f32(biases)?;
            Read {
                codes: c,
                factor: Some(cx.reshape(s, &[rows, groups])?),
                bias: Some(cx.reshape(b, &[rows, groups])?),
            }
        }
        Form::Mxfp4 { codes, scales } => {
            let c = read_codes(cx, codes, 4, true, rows, fan.k)?;
            let factor = if let Codes::Weights(_) = c {
                None
            } else {
                let s = cx.read(scales)?;
                let s = if cx.elem(s) == Elem::U8 {
                    s
                } else {
                    cx.bitcast(s, Elem::U8)?
                };
                let s = cx.reshape(s, &[rows, groups])?;
                let s = cx.convert(s, Elem::U32);
                Some(e8m0(cx, s)?)
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
fn decode(cx: &mut Cx<'_>, bank: &Read, groups: i64) -> Result<Val, Error> {
    let q = code_values(cx, bank.codes, Elem::Bf16)?;
    let Some(factor) = bank.factor else {
        return Ok(q);
    };
    let (rows, k) = (cx.dims(q)[0], cx.dims(q)[1]);
    let q = cx.reshape(q, &[rows, groups, k / groups])?;
    let w = materialize(cx, q, factor, bank.bias)?;
    Ok(cx.convert(w, Elem::Bf16))
}

/// `x [r, K] · W_allᵀ`: every row against every expert, f32 `[r, E·N]`.
fn against_all(cx: &mut Cx<'_>, x: Val, bank: &Read, groups: i64) -> Result<Val, Error> {
    let Some(factor) = bank.factor else {
        let w = code_values(cx, bank.codes, Elem::Bf16)?;
        return Ok(cx.matmul_nt(x, w, Elem::F32)?);
    };
    let q = code_values(cx, bank.codes, Elem::Bf16)?;
    let (rows, k) = (cx.dims(q)[0], cx.dims(q)[1]);
    let q = cx.reshape(q, &[rows, groups, k / groups])?;
    let r = cx.dims(x)[0];
    let xg = cx.reshape(x, &[r, groups, k / groups])?;
    let mut out = grouped_dot(cx, xg, q, factor)?;
    if let Some(b) = bank.bias {
        let o = offset_dot(cx, x, b)?;
        out = cx.add(out, o)?;
    }
    Ok(out)
}

/// `chlo.ragged_dot`: `x [P, K]` rows grouped contiguously by `sizes [E]`
/// (i32), group `e` contracted with `w[e]` (`[E, N, K]`), into f32 `[P, N]`.
/// Rows past the last group come out zero. PJRT's MLIR ingestion legalizes
/// it to XLA's `ragged-dot`, which the TPU compiler runs natively.
fn ragged_dot(cx: &mut Cx<'_>, x: Val, w: Val, sizes: Val) -> Result<Val, Error> {
    let (p, n) = (cx.dims(x)[0], cx.dims(w)[1]);
    let out = Ty::new(Elem::F32, &[p, n]);
    let attrs = "precision_config = [#chlo<precision DEFAULT>, #chlo<precision DEFAULT>], \
                 ragged_dot_dimension_numbers = #chlo.ragged_dot<lhs_batching_dimensions = [], \
                 rhs_batching_dimensions = [], lhs_contracting_dimensions = [1], \
                 rhs_contracting_dimensions = [2], lhs_ragged_dimensions = [0], \
                 rhs_group_dimensions = [0]>";
    Ok(cx.op_named("chlo.ragged_dot", &[x, w, sizes], attrs, vec![out])[0])
}

/// The routed matmul every select entry lowers to: `y[t*k + s] = W[routes[t,
/// s]] · x_row (+ bias[e])`; a slot routed outside `0..experts` keeps `y`'s
/// row as it was (the GPU kernels skip its store).
fn routed(
    cx: &mut Cx<'_>,
    form: Form,
    fan: Fan,
    x: Val,
    routes: Val,
    bias: Option<Val>,
    prev: Val,
) -> Result<Val, Error> {
    let (p, e, n, k) = (fan.pairs(), fan.experts, fan.n, fan.k);
    let groups = k / i64::from(form.group());
    let r = cx.reshape(routes, &[p])?;
    let zero = cx.const_i(Elem::I32, 0, &[p]);
    let top = cx.const_i(Elem::I32, e, &[p]);
    let ge = cx.compare(Cmp::Ge, r, zero)?;
    let lt = cx.compare(Cmp::Lt, r, top)?;
    let valid = cx.and(ge, lt)?;
    let lo = cx.const_i(Elem::I32, 0, &[]);
    let hi = cx.const_i(Elem::I32, e - 1, &[]);
    let ids = cx.clamp(lo, r, hi)?;
    let bank = read_bank(cx, form, fan)?;
    let x = cx.convert(x, Elem::Bf16);
    let rows = cx.dims(x)[0];
    let r8 = (rows + 7) / 8 * 8;
    let fold = if bank.factor.is_some() { groups } else { 1 };
    let unit = match cx.elem(bank.codes.val()) {
        Elem::U4 | Elem::F4E2m1fn => 1,
        Elem::Bf16 | Elem::F16 => 4,
        _ => 2,
    };
    let bank_bytes = e * n * k * unit / 2;
    let expert_bytes = bank_bytes / e;
    // How many active experts (row tiles) the loop runs for less than the
    // whole bank, whose dot also lands a partial per row and expert.
    let tile = loop_tile(fan);
    let whole_ns = bank_bytes / WHOLE_BYTES_PER_NS + r8 * e * n * WHOLE_PARTIAL_FS / 1_000_000;
    let tile_ns = (expert_bytes / 1_200).max(LOOP_TILE_NS + tile * LOOP_ROW_NS);
    let worth = whole_ns / tile_ns;
    // `PIE_XLA_MOE_PATH=slice|loop|all|ragged` forces a formulation (tests
    // reach each with small shapes).
    let forced = std::env::var("PIE_XLA_MOE_PATH").ok();
    let path = match forced.as_deref() {
        Some(path) => path,
        None if p <= SLICE_PAIRS && p * expert_bytes.max(SLICE_BYTES) < bank_bytes => "slice",
        None if r8 * e * n * fold > DENSE_ALL => "ragged",
        None if worth < 2 => "all",
        None if worth >= p => "loop",
        None => "active",
    };
    let y = match path {
        "slice" => {
            // Each pair's expert sliced out of the bank and multiplied by its
            // row alone: independent dots XLA overlaps, no join.
            let mut ys = Vec::with_capacity(p as usize);
            for i in 0..p {
                let id = cx.slice(ids, &[i], &[i + 1], &[1])?;
                let id = cx.reshape(id, &[])?;
                let one = expert(cx, &bank, id, n)?;
                let row = if fan.per_token { i / fan.top_k } else { i };
                let xr = cx.slice(x, &[row, 0], &[row + 1, k], &[1, 1])?;
                ys.push(against_all(cx, xr, &one, groups)?);
            }
            cx.concat(&ys, 0)?
        }
        "all" => every_expert(cx, &bank, fan, x, ids, groups)?,
        "loop" => {
            let plan = plan_active(cx, fan, x, r, valid)?;
            run_active(cx, &plan, &bank, fan, groups, None)?
        }
        "active" => {
            // The loop over the experts the fire routes to, unless more are
            // active than it is worth: then the whole bank. Both run inside
            // loops whose trip count the fire's routes decide (0 steps for
            // the one not taken); a `case` would copy the bank into its
            // branch (v6e: 3x the whole step).
            let plan = plan_active(cx, fan, x, r, valid)?;
            let most = cx.const_i(Elem::I32, worth, &[]);
            let many = cx.compare(Cmp::Gt, plan.steps, most)?;
            let looped = run_active(cx, &plan, &bank, fan, groups, Some(many))?;
            let zeros = cx.const_f(Elem::F32, 0.0, &[p, n]);
            let whole = cx.while_loop(
                &[many, zeros],
                |_, a| Ok(a[0]),
                |f, _| {
                    let mut env = Sealed;
                    let mut cx = Cx::new(f, &mut env);
                    let all = every_expert(&mut cx, &bank, fan, x, ids, groups).map_err(sealed)?;
                    let done = cx.const_i(Elem::Pred, 0, &[]);
                    Ok(vec![done, all])
                },
            )?[1];
            let many = cx.broadcast(many, &[p, n], &[])?;
            cx.select(many, whole, looped)?
        }
        _ => {
            // Sort the pairs by expert (unrouted last), run one ragged dot over
            // the whole bank, and put the rows back.
            let top = cx.const_i(Elem::I32, e, &[p]);
            let key = cx.select(valid, r, top)?;
            let iota = cx.iota(Elem::I32, &[p], 0);
            let by = cx.sort(&[key, iota], 0, true, |f, a, b| {
                f.compare(Cmp::Lt, a[0], b[0])
            })?;
            let (skey, perm) = (by[0], by[1]);
            let ks = cx.broadcast(skey, &[p, e], &[0])?;
            let es = cx.iota(Elem::I32, &[p, e], 1);
            let hit = cx.compare(Cmp::Eq, ks, es)?;
            let one = cx.const_i(Elem::I32, 1, &[p, e]);
            let none = cx.const_i(Elem::I32, 0, &[p, e]);
            let hit = cx.select(hit, one, none)?;
            let sizes = cx.reduce(hit, &[0], Fold::Sum)?;
            let src = if fan.per_token {
                let kk = cx.const_i(Elem::I32, fan.top_k, &[p]);
                cx.div(perm, kk)?
            } else {
                perm
            };
            let xs = cx.take_rows(x, src)?;
            let w = decode(cx, &bank, groups)?;
            let w = cx.reshape(w, &[e, n, k])?;
            let ys = ragged_dot(cx, xs, w, sizes)?;
            let back = cx.sort(&[perm, iota], 0, true, |f, a, b| {
                f.compare(Cmp::Lt, a[0], b[0])
            })?;
            cx.take_rows(ys, back[1])?
        }
    };
    finish(cx, fan, y, ids, valid, bias, prev)
}

/// Expert `e`'s (an i32 scalar) rows of a read bank: `[N, ..]` of each plane.
fn expert(cx: &mut Cx<'_>, bank: &Read, e: Val, n: i64) -> Result<Read, Error> {
    let nn = cx.const_i(Elem::I32, n, &[]);
    let at = cx.mul(e, nn)?;
    let zero = cx.const_i(Elem::I32, 0, &[]);
    let rows = |cx: &mut Cx<'_>, v: Val| -> Result<Val, Error> {
        let w = cx.dims(v)[1];
        Ok(cx.dynamic_slice(v, &[at, zero], &[n, w])?)
    };
    Ok(Read {
        codes: bank.codes.with(rows(cx, bank.codes.val())?),
        factor: match bank.factor {
            Some(f) => Some(rows(cx, f)?),
            None => None,
        },
        bias: match bank.bias {
            Some(b) => Some(rows(cx, b)?),
            None => None,
        },
    })
}

/// Every activation row against every expert in one dot (the whole bank
/// streams once), each pair's row picked: f32 `[P, N]`.
fn every_expert(
    cx: &mut Cx<'_>,
    bank: &Read,
    fan: Fan,
    x: Val,
    ids: Val,
    groups: i64,
) -> Result<Val, Error> {
    let (p, e, n) = (fan.pairs(), fan.experts, fan.n);
    let rows = cx.dims(x)[0];
    let r8 = (rows + 7) / 8 * 8;
    let xp = if r8 > rows {
        let z = cx.const_f(Elem::Bf16, 0.0, &[]);
        cx.pad(x, z, &[0, 0], &[r8 - rows, 0], &[0, 0])?
    } else {
        x
    };
    let all = against_all(cx, xp, bank, groups)?;
    let all = cx.reshape(all, &[r8 * e, n])?;
    let pair = cx.iota(Elem::I32, &[p], 0);
    let row = if fan.per_token {
        let kk = cx.const_i(Elem::I32, fan.top_k, &[p]);
        cx.div(pair, kk)?
    } else {
        pair
    };
    let ee = cx.const_i(Elem::I32, e, &[p]);
    let at = cx.mul(row, ee)?;
    let at = cx.add(at, ids)?;
    Ok(cx.take_rows(all, at)?)
}

/// The rows of one tile of the loop over active experts: the fire's tokens
/// (a token routes to an expert once), at most 128 (`PIE_XLA_MOE_TILE`
/// lowers the cap, for tuning).
fn loop_tile(fan: Fan) -> i64 {
    let p = fan.pairs();
    let most = std::env::var("PIE_XLA_MOE_TILE")
        .ok()
        .and_then(|v| v.parse::<i64>().ok())
        .unwrap_or(128);
    ((fan.tokens.max(1) + 7) / 8 * 8).min(most).min((p + 7) / 8 * 8)
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
fn plan_active(cx: &mut Cx<'_>, fan: Fan, x: Val, r: Val, valid: Val) -> Result<Active, Error> {
    let (p, e) = (fan.pairs(), fan.experts);
    let tile = loop_tile(fan);
    let top = cx.const_i(Elem::I32, e, &[p]);
    let key = cx.select(valid, r, top)?;
    let iota = cx.iota(Elem::I32, &[p], 0);
    let by = cx.sort(&[key, iota], 0, true, |f, a, b| f.compare(Cmp::Lt, a[0], b[0]))?;
    let (skey, perm) = (by[0], by[1]);
    let before = {
        let head = cx.const_i(Elem::I32, e + 1, &[1]);
        let body = cx.slice(skey, &[0], &[p - 1], &[1])?;
        cx.concat(&[head, body], 0)?
    };
    let first = cx.compare(Cmp::Ne, skey, before)?;
    let zero = cx.const_i(Elem::I32, 0, &[p]);
    let at_first = cx.select(first, iota, zero)?;
    let begins = cx.scan(at_first, 0, Fold::Max)?;
    let offset = cx.sub(iota, begins)?;
    let tt = cx.const_i(Elem::I32, tile, &[p]);
    let within = cx.rem(offset, tt)?;
    let starts = cx.compare(Cmp::Eq, within, zero)?;
    let routed = cx.compare(Cmp::Lt, skey, top)?;
    let starts = cx.and(starts, routed)?;
    let one = cx.const_i(Elem::I32, 1, &[p]);
    let counted = cx.select(starts, one, zero)?;
    let steps = cx.reduce(counted, &[0], Fold::Sum)?;
    let pp = cx.const_i(Elem::I32, p, &[p]);
    let late = cx.add(iota, pp)?;
    let order = cx.select(starts, iota, late)?;
    let order = cx.sort(&[order], 0, true, |f, a, b| f.compare(Cmp::Lt, a[0], b[0]))?[0];
    // Past the tiles every entry is `P` or more: clamp to `P` (the padding
    // rows), and give that position a valid expert.
    let cap = cx.const_i(Elem::I32, p, &[]);
    let lo = cx.const_i(Elem::I32, 0, &[]);
    let order = cx.clamp(lo, order, cap)?;
    let pad = cx.const_i(Elem::I32, p, &[1]);
    let order = cx.concat(&[order, pad], 0)?;
    let first_expert = cx.slice(skey, &[0], &[1], &[1])?;
    let hi = cx.const_i(Elem::I32, e - 1, &[]);
    let first_expert = cx.clamp(lo, first_expert, hi)?;
    let skey = cx.concat(&[skey, first_expert], 0)?;
    let src = if fan.per_token {
        let kk = cx.const_i(Elem::I32, fan.top_k, &[p]);
        cx.div(perm, kk)?
    } else {
        perm
    };
    let xs = cx.take_rows(x, src)?;
    let z = cx.const_f(Elem::Bf16, 0.0, &[]);
    let xs = cx.pad(xs, z, &[0, 0], &[tile, 0], &[0, 0])?;
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
    cx: &mut Cx<'_>,
    plan: &Active,
    bank: &Read,
    fan: Fan,
    groups: i64,
    skip: Option<Val>,
) -> Result<Val, Error> {
    let (p, n, k) = (fan.pairs(), fan.n, fan.k);
    let tile = plan.tile;
    let ys = cx.const_f(Elem::F32, 0.0, &[p + tile, n]);
    let j0 = cx.const_i(Elem::I32, 0, &[]);
    let per = LOOP_UNROLL.with(|u| *u);
    let per_v = cx.const_i(Elem::I32, per, &[]);
    let more = cx.const_i(Elem::I32, per - 1, &[]);
    let steps = cx.add(plan.steps, more)?;
    let steps = cx.div(steps, per_v)?;
    let steps = match skip {
        Some(skip) => {
            let none = cx.const_i(Elem::I32, 0, &[]);
            cx.select(skip, none, steps)?
        }
        None => steps,
    };
    let (order, skey, xs) = (plan.order, plan.skey, plan.xs);
    let bank = *bank;
    let out = cx.while_loop(
        &[j0, ys],
        |f, a| f.compare(Cmp::Lt, a[0], steps),
        |f, a| {
            let mut env = Sealed;
            let mut cx = Cx::new(f, &mut env);
            let (j, mut ys) = (a[0], a[1]);
            let step = (|| -> Result<Val, Error> {
                let per_v = cx.const_i(Elem::I32, per, &[]);
                let first = cx.mul(j, per_v)?;
                let mut outs = Vec::with_capacity(per as usize);
                for u in 0..per {
                    // Tiles past the count read `order`'s padding entry
                    // (`P`): they run over the padding rows.
                    let du = cx.const_i(Elem::I32, u, &[]);
                    let t = cx.add(first, du)?;
                    let s = scalar_at(&mut cx, order, t)?;
                    // The slice's start is `expert · N` computed here: read
                    // from a per-tile table instead, XLA stopped fusing the
                    // slice into the dot (v6e, gpt-oss: +4 ms at 64 rows).
                    let ex = scalar_at(&mut cx, skey, s)?;
                    let w = expert(&mut cx, &bank, ex, n)?;
                    let zero = cx.const_i(Elem::I32, 0, &[]);
                    let xe = cx.dynamic_slice(xs, &[s, zero], &[tile, k])?;
                    let ye = against_all(&mut cx, xe, &w, groups)?;
                    outs.push((s, ye));
                }
                let zero = cx.const_i(Elem::I32, 0, &[]);
                for (s, ye) in outs {
                    ys = cx.dynamic_update_slice(ys, ye, &[s, zero])?;
                }
                Ok(ys)
            })()
            .map_err(sealed)?;
            let one = cx.const_i(Elem::I32, 1, &[]);
            let j = cx.add(j, one)?;
            Ok(vec![j, step])
        },
    )?;
    let sorted = cx.slice(out[1], &[0, 0], &[p, n], &[1, 1])?;
    let iota = cx.iota(Elem::I32, &[p], 0);
    let back = cx.sort(&[plan.perm, iota], 0, true, |f, a, b| {
        f.compare(Cmp::Lt, a[0], b[0])
    })?;
    Ok(cx.take_rows(sorted, back[1])?)
}

/// The i32 scalar `v[at]` of a rank-1 `v` (`at` an i32 scalar).
fn scalar_at(cx: &mut Cx<'_>, v: Val, at: Val) -> Result<Val, Error> {
    let one = cx.dynamic_slice(v, &[at], &[1])?;
    Ok(cx.reshape(one, &[])?)
}

/// The env of a region inside a kernel (a loop body, a case branch): its
/// values are built from the enclosing kernel's, never read from handles.
struct Sealed;

impl crate::cx::Env for Sealed {
    fn read(&mut self, _: &mut Func, _: Tensor) -> Result<Val, Error> {
        Err(refuse("linear.moe", "a region reads no handle"))
    }

    fn write(&mut self, _: &mut Func, _: Tensor, _: Val) -> Result<(), Error> {
        Err(refuse("linear.moe", "a region writes no handle"))
    }
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
    cx: &mut Cx<'_>,
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
            let b = cx.reshape(b, &[e, n])?;
            let rows = cx.take_rows(b, ids)?;
            let rows = cx.convert(rows, Elem::F32);
            cx.add(y, rows)?
        }
        None => y,
    };
    let y = cx.convert(y, Elem::Bf16);
    let keep = cx.broadcast(valid, &[p, n], &[0])?;
    let prev = cx.reshape(prev, &[p, n])?;
    let prev = cx.convert(prev, Elem::Bf16);
    Ok(cx.select(keep, y, prev)?)
}

fn select_entry(
    ctx: &Ctx<'_>,
    op: &'static str,
    x: Tensor,
    form: Form,
    bias: Option<Tensor>,
    routes: Tensor,
    y: Tensor,
) -> Result<(), Error> {
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
    ctx.emit(&mut |cx| {
        let xv = cx.read(x)?;
        let rv = cx.read(routes)?;
        let bv = match bias {
            Some(b) => Some(cx.read(b)?),
            None => None,
        };
        let prev = cx.read(y)?;
        let out = routed(cx, form, fan, xv, rv, bv, prev)?;
        cx.write(y, out)
    })
}

/// A dense (bf16) bank: `bank` holds `[experts, N * K]` (any row shape).
pub fn matmul_select(
    ctx: &Ctx<'_>,
    x: Tensor,
    bank: Tensor,
    routes: Tensor,
    y: Tensor,
) -> Result<(), Error> {
    select_entry(
        ctx,
        "linear.moe_matmul_select",
        x,
        Form::Dense(bank),
        None,
        routes,
        y,
    )
}

pub fn matmul_select_bias(
    ctx: &Ctx<'_>,
    x: Tensor,
    bank: Bank,
    bias: Tensor,
    routes: Tensor,
    y: Tensor,
) -> Result<(), Error> {
    const OP: &str = "linear.moe_matmul_select_bias";
    let form = form_of(OP, bank)?;
    select_entry(ctx, OP, x, form, Some(bias), routes, y)
}

pub fn matmul_select_quant(
    ctx: &Ctx<'_>,
    x: Tensor,
    bank: Bank,
    routes: Tensor,
    y: Tensor,
) -> Result<(), Error> {
    const OP: &str = "linear.moe_matmul_select_quant";
    let form = form_of(OP, bank)?;
    select_entry(ctx, OP, x, form, None, routes, y)
}

#[derive(Clone, Copy)]
pub enum GroupedPlane {
    Bank(Bank),
    Dense(Tensor),
}

/// `x [rows, groups*K]` split into `groups` slices, slice `g` of row `r`
/// multiplied by expert `routes[r, g]`, into `y [rows, groups*N]`.
pub fn matmul_grouped(
    ctx: &Ctx<'_>,
    x: Tensor,
    plane: GroupedPlane,
    routes: Tensor,
    groups: u32,
    y: Tensor,
) -> Result<(), Error> {
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
    ctx.emit(&mut |cx| {
        let xv = cx.read(x)?;
        let xv = cx.reshape(xv, &[fan.pairs(), fan.k])?;
        let rv = cx.read(routes)?;
        let prev = cx.read(y)?;
        let out = routed(cx, form, fan, xv, rv, None, prev)?;
        cx.write(y, out)
    })
}

/// `y[t] = sum_s weights[t, s] * routed[t*k + s]`, in f32.
pub fn weighted_sum(
    ctx: &Ctx<'_>,
    routed: Tensor,
    weights: Tensor,
    y: Tensor,
) -> Result<(), Error> {
    const OP: &str = "linear.moe_weighted_sum";
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
    ctx.emit(&mut |cx| {
        let (t, k, w) = (i64::from(y.rows), i64::from(top_k), i64::from(y.width));
        let r = cx.read_f32(routed)?;
        let r = cx.reshape(r, &[t, k, w])?;
        let g = cx.read(weights)?;
        let g = cx.broadcast(g, &[t, k, w], &[0, 1])?;
        let v = cx.mul(r, g)?;
        let v = cx.reduce(v, &[1], Fold::Sum)?;
        cx.write(y, v)
    })
}

/// `y = x + sum_s weights[t, s] * bias[routes[t, s]]` over routed slots
/// (negative routes skipped; `bias` is `[experts, width]`).
pub fn bias_sum(
    ctx: &Ctx<'_>,
    x: Tensor,
    bias: Tensor,
    routes: Tensor,
    weights: Tensor,
    y: Tensor,
) -> Result<(), Error> {
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
    ctx.emit(&mut |cx| {
        let (t, k, w) = (i64::from(y.rows), i64::from(top_k), i64::from(y.width));
        let xv = cx.read_f32(x)?;
        let b = cx.read(bias)?;
        let b = cx.reshape(b, &[experts, w])?;
        let r = cx.read(routes)?;
        let r = cx.reshape(r, &[t * k])?;
        let zero = cx.const_i(Elem::I32, 0, &[t * k]);
        let routed = cx.compare(Cmp::Ge, r, zero)?;
        let lo = cx.const_i(Elem::I32, 0, &[]);
        let hi = cx.const_i(Elem::I32, experts - 1, &[]);
        let ids = cx.clamp(lo, r, hi)?;
        let rows = cx.take_rows(b, ids)?;
        let rows = cx.convert(rows, Elem::F32);
        let g = cx.read(weights)?;
        let g = cx.reshape(g, &[t * k])?;
        let zf = cx.const_f(Elem::F32, 0.0, &[t * k]);
        let g = cx.select(routed, g, zf)?;
        let g = cx.broadcast(g, &[t * k, w], &[0])?;
        let v = cx.mul(rows, g)?;
        let v = cx.reshape(v, &[t, k, w])?;
        let v = cx.reduce(v, &[1], Fold::Sum)?;
        let v = cx.add(xv, v)?;
        cx.write(y, v)
    })
}

/// `y = routed + sigmoid(gate[row]) * shared`.
pub fn sigmoid_gate_add(
    ctx: &Ctx<'_>,
    routed: Tensor,
    shared: Tensor,
    gate: Tensor,
    y: Tensor,
) -> Result<(), Error> {
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
    ctx.emit(&mut |cx| {
        let (t, w) = (i64::from(routed.rows), i64::from(routed.width));
        let r = cx.read_f32(routed)?;
        let s = cx.read_f32(shared)?;
        let g = cx.read_f32(gate)?;
        let g = cx.reshape(g, &[t])?;
        let g = score(cx, g, Score::Sigmoid)?;
        let g = cx.broadcast(g, &[t, w], &[0])?;
        let v = cx.mul(g, s)?;
        let v = cx.add(r, v)?;
        cx.write(y, v)
    })
}
