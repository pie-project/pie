//! Device time of the decode-form recurrent kernels at a decode fire of
//! Qwen3.5-0.8B's shapes (and KDA at 32 heads of 128), with the state pool
//! donated as the engine donates it and larger than the fire (four slots per
//! lane, scattered). Asked for with `PIE_XLA_BENCH=1`; `PIE_XLA_BENCH_WIDTH`
//! sets the lanes (default 64), `PIE_XLA_BENCH_SLOTS` the pool's slots.
//!
//! Each figure is per rep of a module that repeats the kernel `REPS` times,
//! so the ~0.1 ms per-execute host overhead is amortized away.
//!
//! What it found (v6e-1, 64 lanes, 257-slot pool): a `[slots, stride]`
//! bank is tiled `(8, 128)` across slots, so a slot's row is strided over
//! tiles shared by eight slots and gathering / scattering the 64 state rows
//! alone costs ~2.5 ms. The same bank as `[slots · stride / 128, 128]` (each
//! slot whole tiles) moves them in ~0.28 ms, and the gated delta step on it
//! is ~0.29 ms whatever the math's form (broadcast-reduce, one-pass, or a
//! batched `dot_general`, which is slower at HIGHEST). Rounds of ≤ 32 MiB of
//! lanes keep each round's gathered and updated states on chip: ~0.245 ms,
//! against a gather + scatter floor of ~0.18 ms for the same rounds.
//! Scatter-with-combiner, a lane loop of in-place updates, and per-lane
//! `dynamic_update_slice` landing were no better (the last ~0.015 ms).

use dtype::Dtype;
use engine_xla::bench::Bench;
use kernels_xla::Tensor;
use kernels_xla::attn::ssm;
use kernels_xla::hlo::{Combine, Elem, Fold, Func, GatherDims, ScatterDims, Val};
use kernels_xla::RecurrentPool;

fn width() -> Option<u32> {
    std::env::var("PIE_XLA_BENCH").ok()?;
    Some(
        std::env::var("PIE_XLA_BENCH_WIDTH")
            .ok()
            .and_then(|w| w.parse().ok())
            .unwrap_or(64),
    )
}

/// The pool's slots (default four per lane, so the slab is larger than
/// the fire and does not stay on chip); the lanes' slots are scattered.
fn pool_slots(n: u32) -> u32 {
    std::env::var("PIE_XLA_BENCH_SLOTS")
        .ok()
        .and_then(|w| w.parse().ok())
        .unwrap_or(4 * n)
}

fn lane_slots(n: u32, slots: u32) -> Vec<i32> {
    let mut used = vec![false; slots as usize];
    let mut out = Vec::new();
    let mut at = 5u32 % slots;
    for _ in 0..n {
        while used[at as usize] {
            at = (at + 1) % slots;
        }
        used[at as usize] = true;
        out.push(at as i32);
        at = (at + 37) % slots;
    }
    out
}

fn report(what: &str, t: Option<f64>) {
    if let Some(t) = t {
        eprintln!("{what:<40} {:>8.3} ms", t * 1e3);
    }
}

/// Repeats in one module, so the mean is device time rather than the
/// per-execute host overhead (~0.1 ms).
const REPS: u32 = 10;

/// `time_in_place` of `body` emitted `REPS` times back to back, per rep.
fn dev(
    b: &mut Bench,
    in_place: &[Tensor],
    body: impl Fn(&kernels_xla::Ctx<'_>) -> Result<(), kernels_xla::Error>,
) -> Option<f64> {
    b.time_in_place(10, in_place, |ctx| {
        for _ in 0..REPS {
            body(ctx)?;
        }
        Ok(())
    })
    .unwrap()
    .map(|t| t / f64::from(REPS))
}

fn pool(state: Tensor, slots: Tensor) -> RecurrentPool {
    RecurrentPool {
        state,
        slots,
        conv_state: state,
        new_conv_state: state,
    }
}

const H: i64 = 16;
const D: i64 = 128;

/// `[R, H, D]` q/k (normed), v, and `[R, H]` g/beta from the bench inputs.
fn heads(f: &mut Func, qkv: Val, gates: Val) -> (Val, Val, Val, Val, Val) {
    let r = f.dims(qkv)[0];
    let qkv = f.convert(qkv, Elem::F32);
    let q = f.slice_axis(qkv, 1, 0, H * D).unwrap();
    let k = f.slice_axis(qkv, 1, H * D, 2 * H * D).unwrap();
    let v = f.slice_axis(qkv, 1, 2 * H * D, 3 * H * D).unwrap();
    let norm = |f: &mut Func, x: Val| {
        let x = f.reshape(x, &[r, H, D]).unwrap();
        let sq = f.mul(x, x).unwrap();
        let s = f.reduce(sq, &[2], Fold::Sum).unwrap();
        let s = f.offset(s, 1e-6).unwrap();
        let inv = f.rsqrt(s);
        let inv = f.broadcast(inv, &[r, H, D], &[0, 1]).unwrap();
        f.mul(x, inv).unwrap()
    };
    let q = norm(f, q);
    let k = norm(f, k);
    let v = f.reshape(v, &[r, H, D]).unwrap();
    let g = f.slice_axis(gates, 1, 0, H).unwrap();
    let b = f.slice_axis(gates, 1, H, 2 * H).unwrap();
    (q, k, v, g, b)
}

/// Gathers each lane's `[H*D, D]` block of a `[(slots)*H*D, D]` slab.
fn gather_tiles(f: &mut Func, slab: Val, slots: Val) -> Val {
    let n = f.dims(slab)[0] / (H * D);
    let r = f.dims(slots)[0];
    let s3 = f.reshape(slab, &[n, H * D, D]).unwrap();
    let i = f.reshape(slots, &[r, 1]).unwrap();
    f.gather(
        s3,
        i,
        &GatherDims {
            offset_dims: vec![1, 2],
            collapsed_slice_dims: vec![0],
            start_index_map: vec![0],
            index_vector_dim: 1,
            ..GatherDims::default()
        },
        &[1, H * D, D],
    )
    .unwrap()
}

fn scatter_tiles(f: &mut Func, slab: Val, slots: Val, rows: Val, unique: bool) -> Val {
    let n = f.dims(slab)[0] / (H * D);
    let r = f.dims(slots)[0];
    let s3 = f.reshape(slab, &[n, H * D, D]).unwrap();
    let i = f.reshape(slots, &[r, 1]).unwrap();
    let rows = f.reshape(rows, &[r, H * D, D]).unwrap();
    let out = f
        .scatter_hinted(
            s3,
            i,
            rows,
            &ScatterDims {
                update_window_dims: vec![1, 2],
                inserted_window_dims: vec![0],
                scatter_dims_to_operand_dims: vec![0],
                index_vector_dim: 1,
                ..ScatterDims::default()
            },
            Combine::Set,
            unique,
        )
        .unwrap();
    f.reshape(out, &[n * H * D, D]).unwrap()
}

/// `scatter_tiles` with a combiner of `(old, update)`.
fn scatter_tiles_with(
    f: &mut Func,
    slab: Val,
    slots: Val,
    rows: Val,
    combine: impl FnOnce(&mut Func, Val, Val) -> kernels_xla::hlo::Built<Val>,
) -> Val {
    let n = f.dims(slab)[0] / (H * D);
    let r = f.dims(slots)[0];
    let s3 = f.reshape(slab, &[n, H * D, D]).unwrap();
    let i = f.reshape(slots, &[r, 1]).unwrap();
    let rows = f.reshape(rows, &[r, H * D, D]).unwrap();
    let ty = f.ty(s3).clone();
    let sc = kernels_xla::hlo::Ty::scalar(Elem::F32);
    let attrs = "scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2], inserted_window_dims = [0], input_batching_dims = [], scatter_indices_batching_dims = [], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, indices_are_sorted = false, unique_indices = false";
    let out = f
        .op_region(
            "scatter",
            &[s3, i, rows],
            &[sc.clone(), sc],
            |f, b| Ok(vec![combine(f, b[0], b[1])?]),
            attrs,
            vec![ty],
        )
        .unwrap()[0];
    f.reshape(out, &[n * H * D, D]).unwrap()
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Fused {
    /// Reduce from the gather, then a multiply-scatter of the decay and an
    /// add-scatter of the rank-one update.
    MulAdd,
    /// Reduce from the gather, then one set-scatter of `a S + δ kᵀ` (S
    /// gathered again).
    Regather,
    /// Batched reduce from the gather, then a loop of in-place updates.
    LoopUpd,
    /// OnePass over static chunks of this many lanes, each gathered,
    /// updated and scattered on its own (a small working set stays on chip).
    Chunks(i64),
    /// `Chunks` moving the states only (`s + 1`), for the io floor.
    ChunksIo(i64),
    /// `ChunksIo` landing each lane with its own dynamic_update_slice.
    ChunksIoDus(i64),
    /// `LoopUpd` with this many lanes per iteration.
    Group(i64),
}

/// The fused variants: returns `(y [R, H, D], slab')`.
fn fused(f: &mut Func, how: Fused, slab: Val, slots: Val, qkv: Val, gates: Val) -> (Val, Val) {
    let r = f.dims(slots)[0];
    let (q, k, v, g, beta) = heads(f, qkv, gates);
    let a = f.exp(g);
    let a3 = f.broadcast(a, &[r, H, D], &[0, 1]).unwrap();
    let betab = f.broadcast(beta, &[r, H, D], &[0, 1]).unwrap();
    let kq = f.mul(k, q).unwrap();
    let kq = f.reduce(kq, &[2], Fold::Sum).unwrap();
    let kq = f.broadcast(kq, &[r, H, D], &[0, 1]).unwrap();
    let d4 = [r, H, D, D];
    // mem = a S k, sq = a S q; δ = β (v − mem); y = sq + δ (k·q).
    let finish = |f: &mut Func, sk: Val, sq: Val| {
        let mem = f.mul(sk, a3).unwrap();
        let sq = f.mul(sq, a3).unwrap();
        let delta = f.sub(v, mem).unwrap();
        let delta = f.mul(delta, betab).unwrap();
        let dq = f.mul(delta, kq).unwrap();
        (f.add(sq, dq).unwrap(), delta)
    };
    let matvecs = |f: &mut Func, s: Val| {
        let kb = f.broadcast(k, &d4, &[0, 1, 3]).unwrap();
        let qb = f.broadcast(q, &d4, &[0, 1, 3]).unwrap();
        let m = f.mul(s, kb).unwrap();
        let m = f.reduce(m, &[3], Fold::Sum).unwrap();
        let y = f.mul(s, qb).unwrap();
        let y = f.reduce(y, &[3], Fold::Sum).unwrap();
        (m, y)
    };
    match how {
        Fused::Chunks(c) | Fused::ChunksIo(c) | Fused::ChunksIoDus(c) => {
            let dus = matches!(how, Fused::ChunksIoDus(_));
            let io = matches!(how, Fused::ChunksIo(_)) || dus;
            let mut slab = slab;
            let mut ys = Vec::new();
            let mut lo = 0;
            while lo < r {
                let hi = (lo + c).min(r);
                let cut = |f: &mut Func, x: Val| f.slice_axis(x, 0, lo, hi).unwrap();
                let (sl, qc, kc, vc, gc, bc) = (cut(f, slots), cut(f, q), cut(f, k), cut(f, v), cut(f, g), cut(f, beta));
                let s = gather_tiles(f, slab, sl);
                let s = f.reshape(s, &[hi - lo, H, D, D]).unwrap();
                let (y, s2) = if io {
                    let one = f.like_f(s, 1.0);
                    (vc, f.add(s, one).unwrap())
                } else {
                    step(f, Math::Current, s, qc, kc, vc, gc, bc)
                };
                slab = if dus {
                    let nn = f.dims(slab)[0] / (H * D);
                    let mut cur = f.reshape(slab, &[nn, H * D, D]).unwrap();
                    let s2 = f.reshape(s2, &[hi - lo, H * D, D]).unwrap();
                    let sl1 = f.reshape(sl, &[hi - lo]).unwrap();
                    let zero = f.const_i(Elem::I32, 0, &[]);
                    for j in 0..hi - lo {
                        let at = f.slice_axis(sl1, 0, j, j + 1).unwrap();
                        let at = f.reshape(at, &[]).unwrap();
                        let u = f.slice_axis(s2, 0, j, j + 1).unwrap();
                        cur = f.dynamic_update_slice(cur, u, &[at, zero, zero]).unwrap();
                    }
                    f.reshape(cur, &[nn * H * D, D]).unwrap()
                } else {
                    scatter_tiles(f, slab, sl, s2, false)
                };
                ys.push(y);
                lo = hi;
            }
            (f.concat(&ys, 0).unwrap(), slab)
        }
        Fused::MulAdd | Fused::Regather => {
            let s = gather_tiles(f, slab, slots);
            let s = f.reshape(s, &d4).unwrap();
            let (sk, sq) = matvecs(f, s);
            let (y, delta) = finish(f, sk, sq);
            let db = f.broadcast(delta, &d4, &[0, 1, 2]).unwrap();
            let kb = f.broadcast(k, &d4, &[0, 1, 3]).unwrap();
            let upd = f.mul(db, kb).unwrap();
            let ab = f.broadcast(a, &d4, &[0, 1]).unwrap();
            let slab = if how == Fused::MulAdd {
                let slab = scatter_tiles_with(f, slab, slots, ab, |f, o, u| f.mul(o, u));
                scatter_tiles_with(f, slab, slots, upd, |f, o, u| f.add(o, u))
            } else {
                let s = gather_tiles(f, slab, slots);
                let s = f.reshape(s, &d4).unwrap();
                let s1 = f.mul(s, ab).unwrap();
                let s2 = f.add(s1, upd).unwrap();
                scatter_tiles(f, slab, slots, s2, false)
            };
            (y, slab)
        }
        Fused::LoopUpd | Fused::Group(_) => {
            let n = f.dims(slab)[0] / (H * D);
            let s3 = f.reshape(slab, &[n, H * D, D]).unwrap();
            let slots1 = f.reshape(slots, &[r]).unwrap();
            let zero = f.const_i(Elem::I32, 0, &[]);
            let s = gather_tiles(f, slab, slots);
            let s = f.reshape(s, &d4).unwrap();
            let (sk, sq) = matvecs(f, s);
            let (y, delta) = finish(f, sk, sq);
            let d3 = [H, D, D];
            let lane_update = |f: &mut Func, cur: Val, i: Val| -> kernels_xla::hlo::Built<Val> {
                let slot = f.dynamic_slice(slots1, &[i], &[1])?;
                let slot = f.reshape(slot, &[])?;
                let old = f.dynamic_slice(cur, &[slot, zero, zero], &[1, H * D, D])?;
                let old = f.reshape(old, &d3)?;
                let pick = |f: &mut Func, x: Val| -> kernels_xla::hlo::Built<Val> {
                    let d = f.dims(x).to_vec();
                    let mut st = vec![i];
                    st.extend(std::iter::repeat_n(zero, d.len() - 1));
                    let mut sz = d.clone();
                    sz[0] = 1;
                    let x = f.dynamic_slice(x, &st, &sz)?;
                    f.reshape(x, &d[1..])
                };
                let (ki, di, ai) = (pick(f, k)?, pick(f, delta)?, pick(f, a)?);
                let kb = f.broadcast(ki, &d3, &[0, 2])?;
                let db = f.broadcast(di, &d3, &[0, 1])?;
                let ab = f.broadcast(ai, &d3, &[0])?;
                let s1 = f.mul(old, ab)?;
                let u = f.mul(db, kb)?;
                let s2 = f.add(s1, u)?;
                let s2 = f.reshape(s2, &[1, H * D, D])?;
                f.dynamic_update_slice(cur, s2, &[slot, zero, zero])
            };
            let out = if let Fused::Group(g) = how {
                f.for_loop(r / g, &[s3], |f, i, c| {
                    let gv = f.const_i(Elem::I32, g, &[]);
                    let base = f.mul(i, gv)?;
                    let mut cur = c[0];
                    for j in 0..g {
                        let jv = f.const_i(Elem::I32, j, &[]);
                        let at = f.add(base, jv)?;
                        cur = lane_update(f, cur, at)?;
                    }
                    Ok(vec![cur])
                })
                .unwrap()[0]
            } else {
                f.for_loop(r, &[s3], |f, i, c| Ok(vec![lane_update(f, c[0], i)?])).unwrap()[0]
            };
            let slab = f.reshape(out, &[n * H * D, D]).unwrap();
            (y, slab)
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Math {
    /// The emitter's current form: decay, reduce, update, reduce.
    Current,
    /// Both matvecs from the undecayed state, broadcast-multiply-reduce.
    OnePass,
    /// Both matvecs as one batched `dot_general` (HIGHEST).
    Dot,
    /// Same, DEFAULT precision (bf16 operands), for reference only.
    DotLow,
}

/// One token over `s` `[R, H, dv, dk]`: returns `(y [R, H, dv], s')`.
fn step(f: &mut Func, math: Math, s: Val, q: Val, k: Val, v: Val, g: Val, beta: Val) -> (Val, Val) {
    let d = f.dims(s).to_vec();
    let r = d[0];
    let a = f.exp(g);
    let ab = f.broadcast(a, &d, &[0, 1]).unwrap();
    let kb = f.broadcast(k, &d, &[0, 1, 3]).unwrap();
    let betab = f.broadcast(beta, &[r, H, D], &[0, 1]).unwrap();
    match math {
        Math::Current => {
            let s1 = f.mul(s, ab).unwrap();
            let mem = f.mul(s1, kb).unwrap();
            let mem = f.reduce(mem, &[3], Fold::Sum).unwrap();
            let delta = f.sub(v, mem).unwrap();
            let delta = f.mul(delta, betab).unwrap();
            let db = f.broadcast(delta, &d, &[0, 1, 2]).unwrap();
            let upd = f.mul(db, kb).unwrap();
            let s2 = f.add(s1, upd).unwrap();
            let qb = f.broadcast(q, &d, &[0, 1, 3]).unwrap();
            let y = f.mul(s2, qb).unwrap();
            let y = f.reduce(y, &[3], Fold::Sum).unwrap();
            (y, s2)
        }
        Math::OnePass | Math::Dot | Math::DotLow => {
            let a3 = f.broadcast(a, &[r, H, D], &[0, 1]).unwrap();
            let ak = f.mul(k, a3).unwrap();
            let aq = f.mul(q, a3).unwrap();
            let (mem, sq) = if math == Math::OnePass {
                let akb = f.broadcast(ak, &d, &[0, 1, 3]).unwrap();
                let aqb = f.broadcast(aq, &d, &[0, 1, 3]).unwrap();
                let m = f.mul(s, akb).unwrap();
                let m = f.reduce(m, &[3], Fold::Sum).unwrap();
                let y = f.mul(s, aqb).unwrap();
                let y = f.reduce(y, &[3], Fold::Sum).unwrap();
                (m, y)
            } else {
                let ak = f.reshape(ak, &[r, H, 1, D]).unwrap();
                let aq = f.reshape(aq, &[r, H, 1, D]).unwrap();
                let x = f.concat(&[ak, aq], 2).unwrap();
                let m = f
                    .dot_general_at(s, x, &[0, 1], &[0, 1], &[3], &[3], Elem::F32, math == Math::Dot)
                    .unwrap();
                let mem = f.slice_axis(m, 3, 0, 1).unwrap();
                let mem = f.reshape(mem, &[r, H, D]).unwrap();
                let sq = f.slice_axis(m, 3, 1, 2).unwrap();
                let sq = f.reshape(sq, &[r, H, D]).unwrap();
                (mem, sq)
            };
            let delta = f.sub(v, mem).unwrap();
            let delta = f.mul(delta, betab).unwrap();
            let kq = f.mul(k, q).unwrap();
            let kq = f.reduce(kq, &[2], Fold::Sum).unwrap();
            let kq = f.broadcast(kq, &[r, H, D], &[0, 1]).unwrap();
            let dq = f.mul(delta, kq).unwrap();
            let y = f.add(sq, dq).unwrap();
            let s1 = f.mul(s, ab).unwrap();
            let db = f.broadcast(delta, &d, &[0, 1, 2]).unwrap();
            let upd = f.mul(db, kb).unwrap();
            let s2 = f.add(s1, upd).unwrap();
            (y, s2)
        }
    }
}

struct Fire {
    qkv: Tensor,
    gates: Tensor,
    slots: Tensor,
    y: Tensor,
}

fn fire(b: &mut Bench, n: u32) -> Fire {
    let w = (3 * H * D) as u32;
    Fire {
        qkv: b.bf16(n, w, &vec![0.01; (n * w) as usize]),
        gates: b.f32(n, 2 * H as u32, &vec![0.1; (n * 2 * H as u32) as usize]),
        slots: b.i32(n, 1, &lane_slots(n, pool_slots(n))),
        y: b.zeros(Dtype::F32, n, (H * D) as u32),
    }
}

#[test]
fn gated_delta_decode_variants() {
    let Some(n) = width() else {
        eprintln!("not asked: set PIE_XLA_BENCH");
        return;
    };
    let stride = (H * D * D) as u32;
    let mut b = Bench::new();
    let fr = fire(&mut b, n);
    let z = b.zeros(Dtype::Bf16, n, (H * D) as u32);
    let m = pool_slots(n) + 1;
    let bank = b.f32(m, stride, &vec![0.0; (m * stride) as usize]);
    let st = pool(bank, fr.slots);
    let entry = |ctx: &kernels_xla::Ctx<'_>| {
        ssm::gated_delta(ctx, fr.qkv, z, fr.gates, &st, 16, 16, 128, 128, fr.y)
    };
    report("gated_delta (copied slab)", b.time(20, entry).unwrap());
    report("gated_delta (donated)", b.time_in_place(50, &[bank], entry).unwrap());

    // The same slab as [(slots) * H * dv, dk]: every slot is whole tiles.
    let m = pool_slots(n) + 1;
    let tiled = b.f32(m * (H * D) as u32, D as u32, &vec![0.0; (m * stride) as usize]);
    let st2 = pool(tiled, fr.slots);
    let entry2 = |ctx: &kernels_xla::Ctx<'_>| {
        ssm::gated_delta(ctx, fr.qkv, z, fr.gates, &st2, 16, 16, 128, 128, fr.y)
    };
    if let Ok(t) = b.time_in_place(50, &[tiled], entry2) {
        report("gated_delta tiled slab (donated)", t);
    }

    // Row-major slab, gather/scatter only.
    report(
        "io 2d (donated)",
        dev(&mut b, &[bank], |ctx| {
            ctx.emit(&mut |cx| {
                let s = cx.read(bank)?;
                let i = cx.read(fr.slots)?;
                let i = cx.reshape(i, &[i64::from(n)])?;
                let r = cx.take_rows(s, i)?;
                let one = cx.like_f(r, 1.0);
                let r = cx.add(r, one)?;
                let s = cx.put_rows(s, i, r, Combine::Set)?;
                cx.write(bank, s)
            })
        }),
    );
    for unique in [false, true] {
        report(
            if unique { "io tiled unique (donated)" } else { "io tiled (donated)" },
            dev(&mut b, &[tiled], |ctx| {
                ctx.emit(&mut |cx| {
                    let s = cx.read(tiled)?;
                    let i = cx.read(fr.slots)?;
                    let f = cx.func();
                    let r = gather_tiles(f, s, i);
                    let one = f.like_f(r, 1.0);
                    let r = f.add(r, one)?;
                    let s = scatter_tiles(f, s, i, r, unique);
                    cx.write(tiled, s)
                })
            }),
        );
    }

    for math in [Math::Current, Math::OnePass, Math::Dot, Math::DotLow] {
        report(
            &format!("math {math:?} tiled (donated)"),
            dev(&mut b, &[tiled], |ctx| {
                ctx.emit(&mut |cx| {
                    let s = cx.read(tiled)?;
                    let i = cx.read(fr.slots)?;
                    let qkv = cx.read(fr.qkv)?;
                    let gates = cx.read(fr.gates)?;
                    let f = cx.func();
                    let (q, k, v, g, beta) = heads(f, qkv, gates);
                    let r = gather_tiles(f, s, i);
                    let r = f.reshape(r, &[i64::from(n), H, D, D])?;
                    let (y, s2) = step(f, math, r, q, k, v, g, beta);
                    let s = scatter_tiles(f, s, i, s2, false);
                    cx.write(fr.y, y)?;
                    cx.write(tiled, s)
                })
            }),
        );
    }
    for how in [
        Fused::MulAdd,
        Fused::Regather,
        Fused::LoopUpd,
        Fused::Group(8),
        Fused::Chunks(16),
        Fused::Chunks(32),
        Fused::ChunksIo(32),
        Fused::ChunksIoDus(32),
    ] {
        report(
            &format!("fused {how:?} tiled (donated)"),
            dev(&mut b, &[tiled], |ctx| {
                ctx.emit(&mut |cx| {
                    let s = cx.read(tiled)?;
                    let i = cx.read(fr.slots)?;
                    let qkv = cx.read(fr.qkv)?;
                    let gates = cx.read(fr.gates)?;
                    let (y, s) = fused(cx.func(), how, s, i, qkv, gates);
                    cx.write(fr.y, y)?;
                    cx.write(tiled, s)
                })
            }),
        );
    }
}

#[test]
fn state_block_pieces() {
    let Some(n) = width() else {
        return;
    };
    let stride = (H * D * D) as u32;
    let mut b = Bench::new();
    let fr = fire(&mut b, n);
    let m = pool_slots(n) + 1;
    let tiled = b.f32(m * (H * D) as u32, D as u32, &vec![0.0; (m * stride) as usize]);
    let rows = b.f32(n * (H * D) as u32, D as u32, &vec![0.5; (n * stride) as usize]);
    let small = b.zeros(Dtype::F32, n, (H * D) as u32);
    report(
        "empty (tiny add)",
        dev(&mut b, &[], |ctx| {
            ctx.emit(&mut |cx| {
                let s = cx.read(fr.gates)?;
                let s = cx.add(s, s)?;
                cx.write(fr.gates, s)
            })
        }),
    );
    report(
        "gather + reduce (chained)",
        dev(&mut b, &[small], |ctx| {
            ctx.emit(&mut |cx| {
                let s = cx.read(tiled)?;
                let i = cx.read(fr.slots)?;
                let acc = cx.read(small)?;
                let f = cx.func();
                let g = gather_tiles(f, s, i);
                let g = f.reshape(g, &[i64::from(n), H * D, D])?;
                let accb = f.broadcast(acc, &[i64::from(n), H * D, D], &[0, 1])?;
                let g = f.mul(g, accb)?;
                let g = f.reduce(g, &[2], Fold::Sum)?;
                cx.write(small, g)
            })
        }),
    );
    report(
        "gather + 2 reduce over lanes (chained)",
        dev(&mut b, &[small], |ctx| {
            ctx.emit(&mut |cx| {
                let s = cx.read(tiled)?;
                let i = cx.read(fr.slots)?;
                let acc = cx.read(small)?;
                let f = cx.func();
                let g = gather_tiles(f, s, i);
                let g = f.reshape(g, &[i64::from(n), H, D, D])?;
                let acc = f.reshape(acc, &[i64::from(n), H, D])?;
                let accb = f.broadcast(acc, &[i64::from(n), H, D, D], &[0, 1, 3])?;
                let one = f.like_f(accb, 1.0);
                let acc2 = f.add(accb, one)?;
                let x = f.mul(g, accb)?;
                let x = f.reduce(x, &[3], Fold::Sum)?;
                let y = f.mul(g, acc2)?;
                let y = f.reduce(y, &[3], Fold::Sum)?;
                let x = f.add(x, y)?;
                cx.write(small, x)
            })
        }),
    );
    report(
        "gather + 2 reduce over sublanes (chained)",
        dev(&mut b, &[small], |ctx| {
            ctx.emit(&mut |cx| {
                let s = cx.read(tiled)?;
                let i = cx.read(fr.slots)?;
                let acc = cx.read(small)?;
                let f = cx.func();
                let g = gather_tiles(f, s, i);
                let g = f.reshape(g, &[i64::from(n), H, D, D])?;
                let acc = f.reshape(acc, &[i64::from(n), H, D])?;
                let accb = f.broadcast(acc, &[i64::from(n), H, D, D], &[0, 1, 2])?;
                let one = f.like_f(accb, 1.0);
                let acc2 = f.add(accb, one)?;
                let x = f.mul(g, accb)?;
                let x = f.reduce(x, &[2], Fold::Sum)?;
                let y = f.mul(g, acc2)?;
                let y = f.reduce(y, &[2], Fold::Sum)?;
                let x = f.add(x, y)?;
                cx.write(small, x)
            })
        }),
    );
    report(
        "dus chain (donated)",
        dev(&mut b, &[tiled], |ctx| {
            ctx.emit(&mut |cx| {
                let s = cx.read(tiled)?;
                let i = cx.read(fr.slots)?;
                let r = cx.read(rows)?;
                let f = cx.func();
                let nn = f.dims(s)[0] / (H * D);
                let mut cur = f.reshape(s, &[nn, H * D, D])?;
                let r = f.reshape(r, &[i64::from(n), H * D, D])?;
                let i = f.reshape(i, &[i64::from(n)])?;
                let zero = f.const_i(Elem::I32, 0, &[]);
                for l in 0..i64::from(n) {
                    let at = f.slice_axis(i, 0, l, l + 1)?;
                    let at = f.reshape(at, &[])?;
                    let u = f.slice_axis(r, 0, l, l + 1)?;
                    cur = f.dynamic_update_slice(cur, u, &[at, zero, zero])?;
                }
                let s = f.reshape(cur, &[(nn) * H * D, D])?;
                cx.write(tiled, s)
            })
        }),
    );
    report(
        "scatter only (donated)",
        dev(&mut b, &[tiled], |ctx| {
            ctx.emit(&mut |cx| {
                let s = cx.read(tiled)?;
                let i = cx.read(fr.slots)?;
                let r = cx.read(rows)?;
                let s = scatter_tiles(cx.func(), s, i, r, false);
                cx.write(tiled, s)
            })
        }),
    );
    report(
        "scatter of r*2 (donated)",
        dev(&mut b, &[tiled], |ctx| {
            ctx.emit(&mut |cx| {
                let s = cx.read(tiled)?;
                let i = cx.read(fr.slots)?;
                let r = cx.read(rows)?;
                let r = cx.add(r, r)?;
                let s = scatter_tiles(cx.func(), s, i, r, false);
                cx.write(tiled, s)
            })
        }),
    );
    report(
        "in-place slab*2 (donated)",
        dev(&mut b, &[tiled], |ctx| {
            ctx.emit(&mut |cx| {
                let s = cx.read(tiled)?;
                let s = cx.add(s, s)?;
                cx.write(tiled, s)
            })
        }),
    );
}

/// The decode entries as the engine calls them, over a bank of one row per
/// slot and over a bank of whole blocks of rows per slot.
#[test]
fn decode_entries() {
    let Some(n) = width() else {
        return;
    };
    let m = pool_slots(n) + 1;
    let mut b = Bench::new();
    let fr = fire(&mut b, n);
    let z = b.zeros(Dtype::Bf16, n, (H * D) as u32);

    // Gated delta, Qwen3.5-0.8B: 16 heads of 128x128.
    let stride = (H * D * D) as u32;
    let only_blocked = std::env::var("PIE_XLA_BENCH_BLOCKED").is_ok();
    for blocked in [false, true] {
        if only_blocked && !blocked {
            continue;
        }
        let bank = if blocked {
            b.f32(m * (H * D) as u32, D as u32, &vec![0.0; (m * stride) as usize])
        } else {
            b.f32(m, stride, &vec![0.0; (m * stride) as usize])
        };
        let st = pool(bank, fr.slots);
        report(
            &format!("gated_delta {}", if blocked { "blocked" } else { "row/slot" }),
            dev(&mut b, &[bank], |ctx| {
                ssm::gated_delta(ctx, fr.qkv, z, fr.gates, &st, 16, 16, 128, 128, fr.y)
            }),
        );
    }

    // KDA: 32 heads of 128 (2 MiB of state per slot).
    let (kh, kd) = (32u32, 128u32);
    let wide = kh * kd;
    let mixed = b.bf16(n, 3 * wide, &vec![0.01; (n * 3 * wide) as usize]);
    let fp = b.bf16(n, wide, &vec![0.1; (n * wide) as usize]);
    let bp = b.bf16(n, kh, &vec![0.1; (n * kh) as usize]);
    let dt = b.f32(1, wide, &vec![0.0; wide as usize]);
    let al = b.f32(1, kh, &vec![0.0; kh as usize]);
    let ky = b.zeros(Dtype::F32, n, wide);
    let kstride = kh * kd * kd;
    for blocked in [false, true] {
        if only_blocked && !blocked {
            continue;
        }
        let bank = if blocked {
            b.f32(m * kh * kd, kd, &vec![0.0; (m * kstride) as usize])
        } else {
            b.f32(m, kstride, &vec![0.0; (m * kstride) as usize])
        };
        let st = pool(bank, fr.slots);
        report(
            &format!("kda_step {}", if blocked { "blocked" } else { "row/slot" }),
            dev(&mut b, &[bank], |ctx| {
                ssm::kda_step(ctx, mixed, fp, bp, dt, al, &st, kh, kd, 1e-6, 0.0, ky)
            }),
        );
    }

    // Causal conv over 6144 channels, width 4.
    let c = 6144u32;
    let x = b.bf16(n, c, &vec![0.01; (n * c) as usize]);
    let w = b.bf16(c, 4, &vec![0.1; (c * 4) as usize]);
    let cy = b.zeros(Dtype::Bf16, n, c);
    for blocked in [false, true] {
        if only_blocked && !blocked {
            continue;
        }
        let bank = if blocked {
            b.f32(m * 4, c, &vec![0.0; (m * 4 * c) as usize])
        } else {
            b.f32(m, 4 * c, &vec![0.0; (m * 4 * c) as usize])
        };
        let st = pool(bank, fr.slots);
        report(
            &format!("causal_conv1d {}", if blocked { "blocked" } else { "row/slot" }),
            dev(&mut b, &[bank], |ctx| ssm::causal_conv1d(ctx, x, w, &st, 4, 1, cy)),
        );
    }
}
