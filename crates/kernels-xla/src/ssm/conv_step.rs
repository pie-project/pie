//! A decode fire's causal conv as one Mosaic kernel over the chunk pool:
//! every lane's window of `hist` rows is copied out of the pool into VMEM
//! (all copies in flight together), each lane's taps weighed there in the
//! order the XLA form weighs them, the window rolled by the lane's row, and
//! written back to the pool where the lane writes. The XLA form gathers the
//! windows out of the pool and scatters them back, each a pass of its own
//! around the conv (and, on a TPU, a launch each over the pool).
//!
//! A window is `hist · C` bf16 elements, which in the pool's packed rows
//! order are `hist · C / 128` rows of 128 lanes of the pool read as bf16:
//! a row of the window is `C / 128` of them. The kernel returns the taps'
//! weighed sum (bias added), so the activation is the XLA form's own.

use crate::hlo::{Built, Cmp, Elem, Fold, Func, Val};
use crate::mosaic::{Body, Grid, ICmp, Kernel, MTy, Operand, Sem, Space, V, Window};

/// The most VMEM the kernel's windows take (both copies of every lane's).
const MOST_WINDOWS: i64 = 16 << 20;

/// `(acc [t, C] f32, pool)`: the conv's weighed sum for one row a lane, and
/// the pool with each writing lane's window rolled; `None` where the kernel
/// does not apply (no pool view, channels not whole rows of 128, too many
/// lanes for VMEM). `x` `[t, C]` bf16; `w` `[C, taps]` f32; `slab` the
/// windows' view of the pool, `per` its rows a window; `slot`, `write`
/// `[t]` i32 (a lane with `write` 0 lands nothing).
#[allow(clippy::too_many_arguments)]
pub(super) fn conv_step(
    f: &mut Func,
    x: Val,
    w: Val,
    bias: Option<Val>,
    slab: Val,
    per: i64,
    hist: i64,
    dil: i64,
    slot: Val,
    write: Val,
) -> Built<Option<(Val, Val)>> {
    let Some((pool, Elem::Bf16)) = f.view_parts(slab) else {
        return Ok(None);
    };
    if !crate::blocks::on() {
        return Ok(None);
    }
    let (t, c) = (f.dims(x)[0], f.dims(x)[1]);
    let taps = f.dims(w)[1];
    let width = f.dims(slab)[1];
    let r0 = c / 128;
    let blk = per * width / 128;
    if c % 128 != 0
        || per * width != hist * c
        || f.elem(x) != Elem::Bf16
        || 2 * t * blk * 128 * 2 > MOST_WINDOWS
        || blk % 16 != 0
    {
        return Ok(None);
    }
    let rows = f.dims(pool)[0];
    let windows = rows * 2 / blk;
    // The window each lane reads (a gather clamps), and whether it lands:
    // its own flag, and no later writing lane landing the same window (a
    // set scatter lands the last).
    let lo = f.like_i(slot, 0);
    let hi = f.like_i(slot, windows - 1);
    let at = f.clamp(lo, slot, hi)?;
    let a = f.broadcast(at, &[t, t], &[0])?;
    let b = f.broadcast(at, &[t, t], &[1])?;
    let same = f.compare(Cmp::Eq, a, b)?;
    let i = f.iota(Elem::I32, &[t, t], 0);
    let j = f.iota(Elem::I32, &[t, t], 1);
    let later = f.compare(Cmp::Gt, j, i)?;
    let writes = if f.elem(write) == Elem::Pred {
        write
    } else {
        let wz = f.like_i(write, 0);
        f.compare(Cmp::Ne, write, wz)?
    };
    let wj = f.broadcast(writes, &[t, t], &[1])?;
    let over = f.and(same, later)?;
    let over = f.and(over, wj)?;
    let over = f.convert(over, Elem::I32);
    let shadowed = f.reduce(over, &[1], Fold::Max)?;
    let zero = f.like_i(shadowed, 0);
    let last = f.compare(Cmp::Eq, shadowed, zero)?;
    let lands = f.and(writes, last)?;
    let lands = f.convert(lands, Elem::I32);
    // Operands in rows of 128.
    let xr = f.reshape(x, &[t, r0, 128])?;
    let wt = f.transpose(w, &[1, 0])?;
    let wr = f.reshape(wt, &[taps, r0, 128])?;
    let br = match bias {
        Some(b) => Some(f.reshape(b, &[1, r0, 128])?),
        None => None,
    };
    let c_ = kernel(t, r0, blk, hist, dil, taps, rows, br.is_some())?;
    let mut ops = vec![at, lands, xr, wr];
    ops.extend(br);
    ops.push(pool);
    let pool_at = ops.len() - 1;
    let out = c_.call_aliased(f, &ops, &[(1, pool_at)])?;
    let acc = f.reshape(out[0], &[t, c])?;
    let dims = f.dims(slab).to_vec();
    let slab = f.pool_view(out[1], Elem::Bf16, &dims)?;
    Ok(Some((acc, slab)))
}

#[allow(clippy::too_many_arguments)]
fn kernel(
    t: i64,
    r0: i64,
    blk: i64,
    hist: i64,
    dil: i64,
    taps: i64,
    rows: i64,
    biased: bool,
) -> Built<crate::mosaic::Compiled> {
    let whole = |array: Vec<i64>, elem: Elem| Operand {
        block: array.clone(),
        array,
        elem,
        window: Window::Blocked,
        index: Box::new(|b: &mut Body, _: &Grid| {
            let z = b.i32(0);
            Ok(vec![z, z, z])
        }),
    };
    let any = |array: Vec<i64>| Operand {
        array,
        elem: Elem::U32,
        block: vec![],
        window: Window::Any,
        index: Box::new(|_: &mut Body, _: &Grid| Ok(vec![])),
    };
    let mut ins = vec![
        whole(vec![t, r0, 128], Elem::Bf16),
        whole(vec![taps, r0, 128], Elem::F32),
    ];
    if biased {
        ins.push(whole(vec![1, r0, 128], Elem::F32));
    }
    ins.push(any(vec![rows, 128]));
    let windows_bytes = 2 * t * blk * 128 * 2;
    let vmem = windows_bytes + t * r0 * 128 * (2 + 4) + taps * r0 * 128 * 4 + (8 << 20);
    let k = Kernel {
        name: "conv_step".into(),
        grid: vec![1],
        semantics: vec![Sem::Arbitrary],
        prefetch: vec![vec![t], vec![t]],
        ins,
        outs: vec![whole(vec![t, r0, 128], Elem::F32), any(vec![rows, 128])],
        vmem_limit: Some(u64::try_from(vmem).unwrap_or(0)),
        scratch: vec![
            MTy::Ref(vec![t, blk, 128], Elem::Bf16, Space::Vmem),
            MTy::Ref(vec![t, blk, 128], Elem::Bf16, Space::Vmem),
            MTy::Sem(vec![2]),
        ],
    };
    k.build(|b, blk_| {
        let (slots, lands) = (blk_.grid.prefetch[0], blk_.grid.prefetch[1]);
        let (x_r, w_r) = (blk_.ins[0], blk_.ins[1]);
        let b_r = biased.then(|| blk_.ins[2]);
        let pool_in = b.ref_bitcast(blk_.ins[ins_pool(biased)], Elem::Bf16)?;
        let acc_r = blk_.outs[0];
        let pool_out = b.ref_bitcast(blk_.outs[1], Elem::Bf16)?;
        let (wins, next, sems) = (blk_.scratch[0], blk_.scratch[1], blk_.scratch[2]);
        let sem_of = |b: &mut Body, i: i64| -> Built<V> {
            let at = b.i32(i);
            let s = b.ref_slice(sems, &[at], &[1])?;
            b.ref_squeeze(s)
        };
        let lane_win = |b: &mut Body, buf: V, j: V| -> Built<V> {
            let z = b.i32(0);
            let r = b.ref_slice(buf, &[j, z, z], &[1, blk, 128])?;
            b.ref_squeeze(r)
        };
        let pool_win = |b: &mut Body, p: V, j: V| -> Built<V> {
            let sl = b.load(slots, &[j])?;
            let bv = b.i32(blk);
            let at = b.muli(sl, bv)?;
            let at = b.assume_multiple(at, 16)?;
            let z = b.i32(0);
            b.ref_slice(p, &[at, z], &[blk, 128])
        };
        let (z, one, n) = (b.i32(0), b.i32(1), b.i32(t));
        // every lane's window in
        let rs = sem_of(b, 0)?;
        b.for_range(z, n, one, |b, j| {
            let src = pool_win(b, pool_in, j)?;
            let dst = lane_win(b, wins, j)?;
            b.dma_start(src, dst, rs)
        })?;
        b.dma_wait(rs, wins, wins)?;
        let i0 = b.index(0);
        b.for_range(z, n, one, |b, j| {
            let ji = b.as_index(j)?;
            let row_of = |b: &mut Body, buf: V, r: i64| -> Built<V> {
                let at = b.index(r * r0);
                let v = b.vload(buf, &[ji, at, i0], &[1, r0, 128])?;
                b.shape_cast(v, &[r0, 128])
            };
            let xv = b.vload(x_r, &[ji, i0, i0], &[1, r0, 128])?;
            let xv = b.shape_cast(xv, &[r0, 128])?;
            let x32 = b.convert_f(xv, Elem::F32)?;
            // taps in the XLA form's order: tap k reads row
            // hist - 1 - (taps - 1 - k) · dil, the newest being x
            let mut acc: Option<V> = None;
            for kk in 0..taps {
                let pos = hist - 1 - (taps - 1 - kk) * dil;
                let tap = if pos + 1 == hist {
                    x32
                } else {
                    let v = row_of(b, wins, pos + 1)?;
                    b.convert_f(v, Elem::F32)?
                };
                let ki = b.index(kk);
                let wk = b.vload(w_r, &[ki, i0, i0], &[1, r0, 128])?;
                let wk = b.shape_cast(wk, &[r0, 128])?;
                let term = b.mulf(tap, wk)?;
                acc = Some(match acc {
                    None => term,
                    Some(a) => b.addf(a, term)?,
                });
            }
            let mut acc = acc.ok_or_else(|| crate::hlo::Malformed {
                op: "conv_step",
                detail: "no taps".into(),
            })?;
            if let Some(b_r) = b_r {
                let bv = b.vload(b_r, &[i0, i0, i0], &[1, r0, 128])?;
                let bv = b.shape_cast(bv, &[r0, 128])?;
                acc = b.addf(acc, bv)?;
            }
            let acc3 = b.shape_cast(acc, &[1, r0, 128])?;
            b.vstore(acc3, acc_r, &[ji, i0, i0])?;
            // the window rolled: rows 1.. then x
            for r in 1..hist {
                let v = row_of(b, wins, r)?;
                let v = b.shape_cast(v, &[1, r0, 128])?;
                let at = b.index((r - 1) * r0);
                b.vstore(v, next, &[ji, at, i0])?;
            }
            let xv3 = b.shape_cast(xv, &[1, r0, 128])?;
            let at = b.index((hist - 1) * r0);
            b.vstore(xv3, next, &[ji, at, i0])
        })?;
        // the writing lanes' windows out (each waited for by itself: a lane
        // that lands nothing copies nothing)
        let ws = sem_of(b, 1)?;
        let out = |b: &mut Body, j: V, wait: bool| -> Built<()> {
            let l = b.load(lands, &[j])?;
            let z = b.i32(0);
            let yes = b.cmpi(ICmp::Ne, l, z)?;
            b.when(yes, |b| {
                let src = lane_win(b, next, j)?;
                let dst = pool_win(b, pool_out, j)?;
                if wait {
                    b.dma_wait(ws, src, dst)
                } else {
                    b.dma_start(src, dst, ws)
                }
            })
        };
        b.for_range(z, n, one, |b, j| out(b, j, false))?;
        b.for_range(z, n, one, |b, j| out(b, j, true))
    })
}

fn ins_pool(biased: bool) -> usize {
    if biased { 3 } else { 2 }
}
