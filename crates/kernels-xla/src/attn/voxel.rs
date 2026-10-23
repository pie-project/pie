//! Attention among the voxels of each clip's lane.

#![allow(clippy::too_many_arguments)]

use crate::conv::{add_k, div_k, lanes_of, mul_k, nonzero, place, spread, zero_invalid};
use crate::cx::Ctx;
use crate::error::{Error, refuse};
use crate::hlo::{Cmp, Elem, Fold};
use crate::tensor::Tensor;
use dtype::Dtype;

/// Queries a chunk of the dense score matrix holds, bounded by bytes.
const SCORE_BUDGET: i64 = 64 << 20;

/// Single-head attention of each row over the keys of its clip (or its block
/// of frames), `softmax(q·kᵀ · sm_scale) · v` in f32 with one bf16 rounding;
/// a row no lane holds lands zero. kernels-cuda `attn::voxel::attention`.
/// Computed as a dense masked product over all rows of the fire, in query
/// chunks of at most 64 MiB of scores.
pub fn attention(
    ctx: &Ctx<'_>,
    q: Tensor,
    k_: Tensor,
    v: Tensor,
    grid: Tensor,
    segment: Segment,
    sm_scale: f32,
    y: Tensor,
) -> Result<(), Error> {
    const OP: &str = "attn.voxel";
    if q.dtype != Dtype::Bf16 {
        return Err(Error::DtypeUnsupported {
            op: OP,
            dtype: q.dtype,
        });
    }
    for (what, t) in [("key", k_), ("value", v), ("output", y)] {
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
    ctx.emit(&mut |cx| {
        let gv = cx.read(grid)?;
        let p = place(cx, gv, rows)?;
        let (begin, end) = match segment {
            Segment::Lane | Segment::Frames(0) => {
                let vox = cx.mul(p.b.h, p.b.w)?;
                let vox = cx.mul(vox, p.b.t)?;
                let end = cx.add(p.b.off, vox)?;
                (p.b.off, end)
            }
            Segment::Frames(n) => {
                let plane = cx.mul(p.b.h, p.b.w)?;
                let first = div_k(cx, p.v.t, i64::from(n))?;
                let first = mul_k(cx, first, i64::from(n))?;
                let last = add_k(cx, first, i64::from(n))?;
                let last = cx.min(last, p.b.t)?;
                let b = cx.mul(first, plane)?;
                let b = cx.add(b, p.b.off)?;
                let e = cx.mul(last, plane)?;
                let e = cx.add(e, p.b.off)?;
                (b, e)
            }
        };
        let qv = cx.read(q)?;
        let kv = cx.read(k_)?;
        let vv = cx.read_f32(v)?;
        let mut outs = Vec::new();
        let mut q0 = 0;
        while q0 < rows {
            let n = chunk.min(rows - q0);
            let qc = cx.slice_axis(qv, 0, q0, q0 + n)?;
            let s = cx.matmul_nt(qc, kv, Elem::F32)?;
            let s = cx.scale(s, f64::from(sm_scale))?;
            let dims = [n, rows];
            let j = cx.iota(Elem::I32, &dims, 1);
            let b = cx.slice_axis(begin, 0, q0, q0 + n)?;
            let e = cx.slice_axis(end, 0, q0, q0 + n)?;
            let b = spread(cx, b, &dims, &[0])?;
            let e = spread(cx, e, &dims, &[0])?;
            let lo = cx.compare(Cmp::Ge, j, b)?;
            let hi = cx.compare(Cmp::Lt, j, e)?;
            let seen = cx.and(lo, hi)?;
            let ninf = cx.like_f(s, f64::NEG_INFINITY);
            let s = cx.select(seen, s, ninf)?;
            let m = cx.reduce(s, &[1], Fold::Max)?;
            let fin = cx.is_finite(m);
            let zm = cx.like_f(m, 0.0);
            let m = cx.select(fin, m, zm)?;
            let m = spread(cx, m, &dims, &[0])?;
            let z = cx.sub(s, m)?;
            let pr = cx.exp(z);
            let zp = cx.like_f(pr, 0.0);
            let pr = cx.select(seen, pr, zp)?;
            let l = cx.reduce(pr, &[1], Fold::Sum)?;
            let o = cx.dot_general(pr, vv, &[], &[], &[1], &[0], Elem::F32)?;
            let zl = cx.like_f(l, 0.0);
            let pos = cx.compare(Cmp::Gt, l, zl)?;
            let one = cx.like_f(l, 1.0);
            let safe = cx.select(pos, l, one)?;
            let inv = cx.div(one, safe)?;
            let inv = cx.select(pos, inv, zl)?;
            let width = cx.dims(o)[1];
            let inv = spread(cx, inv, &[n, width], &[0])?;
            outs.push(cx.mul(o, inv)?);
            q0 += n;
        }
        let o = cx.concat(&outs, 0)?;
        let o = zero_invalid(cx, o, p.valid)?;
        cx.write(y, o)
    })
}

/// Which keys a spatial attention row sees: its whole clip, or its block of
/// `n` frames (`poem_ir::VoxelSegment`).
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum Segment {
    #[default]
    Lane,
    Frames(u32),
}
