//! Attention among the voxels of each clip's lane.

#![allow(clippy::too_many_arguments)]

use crate::conv::{add_k, div_k, lanes_of, mul_k, nonzero, place, spread, zero_invalid};
use crate::error::{Error, refuse};
use crate::hlo::{Cmp, Elem, Fold, Func, Val};
use crate::tensor::{Landing, Operand, Stored};
use dtype::Dtype;

/// Queries a chunk of the dense score matrix holds, bounded by bytes.
const SCORE_BUDGET: i64 = 64 << 20;

/// Single-head attention of each row over the keys of its clip (or its block
/// of frames), `softmax(q·kᵀ · sm_scale) · v` in f32 with one bf16 rounding;
/// a row no lane holds lands zero. kernels-cuda `attn::voxel::attention`.
/// Computed as a dense masked product over all rows of the fire, in query
/// chunks of at most 64 MiB of scores.
pub fn attention(
    f: &mut Func,
    q: Operand,
    k_: Operand,
    v: Operand,
    grid: Operand,
    segment: Segment,
    sm_scale: f32,
    y: Landing,
) -> Result<Val, Error> {
    const OP: &str = "attn.voxel";
    if q.dtype != Dtype::Bf16 {
        return Err(Error::DtypeUnsupported {
            op: OP,
            dtype: q.dtype,
        });
    }
    for (what, t) in [("key", k_.shape()), ("value", v.shape()), ("output", y)] {
        if t.rows != q.rows || t.width != q.width || t.dtype != q.dtype {
            return Err(refuse(
                OP,
                format!(
                    "the {what} is {}x{} {:?}; the query is {}x{} {:?}, and all four share one shape",
                    t.rows, t.width, t.dtype, q.rows, q.width, q.dtype
                ),
            ));
        }
    }
    if segment == Segment::Frames(0) {
        return Err(refuse(
            OP,
            "a block of zero frames holds no keys; state `Segment::Lane`",
        ));
    }
    lanes_of(OP, "input", grid)?;
    nonzero(OP, "rows", q.rows)?;
    let rows = i64::from(q.rows);
    let chunk = (SCORE_BUDGET / (rows * 4)).clamp(1, rows);
    let gv = grid.val;
    let p = place(f, gv, rows)?;
    let (begin, end) = match segment {
        Segment::Lane | Segment::Frames(0) => {
            let vox = f.mul(p.b.h, p.b.w)?;
            let vox = f.mul(vox, p.b.t)?;
            let end = f.add(p.b.off, vox)?;
            (p.b.off, end)
        }
        Segment::Frames(n) => {
            let plane = f.mul(p.b.h, p.b.w)?;
            let first = div_k(f, p.v.t, i64::from(n))?;
            let first = mul_k(f, first, i64::from(n))?;
            let last = add_k(f, first, i64::from(n))?;
            let last = f.min(last, p.b.t)?;
            let b = f.mul(first, plane)?;
            let b = f.add(b, p.b.off)?;
            let e = f.mul(last, plane)?;
            let e = f.add(e, p.b.off)?;
            (b, e)
        }
    };
    let qv = q.val;
    let kv = k_.val;
    let vv = v.f32(f);
    let mut outs = Vec::new();
    let mut q0 = 0;
    while q0 < rows {
        let n = chunk.min(rows - q0);
        let qc = f.slice_axis(qv, 0, q0, q0 + n)?;
        let s = f.matmul_nt(qc, kv, Elem::F32)?;
        let s = f.scale(s, f64::from(sm_scale))?;
        let dims = [n, rows];
        let j = f.iota(Elem::I32, &dims, 1);
        let b = f.slice_axis(begin, 0, q0, q0 + n)?;
        let e = f.slice_axis(end, 0, q0, q0 + n)?;
        let b = spread(f, b, &dims, &[0])?;
        let e = spread(f, e, &dims, &[0])?;
        let lo = f.compare(Cmp::Ge, j, b)?;
        let hi = f.compare(Cmp::Lt, j, e)?;
        let seen = f.and(lo, hi)?;
        let ninf = f.like_f(s, f64::NEG_INFINITY);
        let s = f.select(seen, s, ninf)?;
        let m = f.reduce(s, &[1], Fold::Max)?;
        let fin = f.is_finite(m);
        let zm = f.like_f(m, 0.0);
        let m = f.select(fin, m, zm)?;
        let m = spread(f, m, &dims, &[0])?;
        let z = f.sub(s, m)?;
        let pr = f.exp(z);
        let zp = f.like_f(pr, 0.0);
        let pr = f.select(seen, pr, zp)?;
        let l = f.reduce(pr, &[1], Fold::Sum)?;
        let o = f.dot_general(pr, vv, &[], &[], &[1], &[0], Elem::F32)?;
        let zl = f.like_f(l, 0.0);
        let pos = f.compare(Cmp::Gt, l, zl)?;
        let one = f.like_f(l, 1.0);
        let safe = f.select(pos, l, one)?;
        let inv = f.div(one, safe)?;
        let inv = f.select(pos, inv, zl)?;
        let width = f.dims(o)[1];
        let inv = spread(f, inv, &[n, width], &[0])?;
        outs.push(f.mul(o, inv)?);
        q0 += n;
    }
    let o = f.concat(&outs, 0)?;
    let o = zero_invalid(f, o, p.valid)?;
    Ok(o)
}

/// Which keys a spatial attention row sees: its whole clip, or its block of
/// `n` frames (`poem_ir::VoxelSegment`).
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum Segment {
    #[default]
    Lane,
    Frames(u32),
}
