//! Folding two partial readings by their log-sum-exps.

#![allow(clippy::too_many_arguments)]

use dtype::Dtype;

use crate::cx::{Ctx, expect};
use crate::error::{Error, refuse};
use crate::hlo::Cmp;
use crate::tensor::Tensor;

/// `o = (o1·w1 + o2·w2) / (w1 + w2)` with `w = 2^(lse − max)`, and
/// `lse = max + log2(w1 + w2)`; a side whose lse is not finite yields to the
/// other unchanged (base-2 lses, as the attention kernels publish them).
pub fn merge_lse(
    ctx: &Ctx<'_>,
    o1: Tensor,
    lse1: Tensor,
    o2: Tensor,
    lse2: Tensor,
    heads: u32,
    head_dim: u32,
    o: Tensor,
    lse: Tensor,
) -> Result<(), Error> {
    const OP: &str = "attention.merge_lse";
    expect(OP, o, &[Dtype::Bf16])?;
    if u64::from(heads) * u64::from(head_dim) != u64::from(o.width) {
        return Err(refuse(
            OP,
            format!(
                "the {heads} heads x {head_dim} this merge states are not its {}-wide row",
                o.width
            ),
        ));
    }
    for t in [o1, o2] {
        if t.rows != o.rows || t.width != o.width {
            return Err(refuse(
                OP,
                "the merged readings are one row per query row, as wide as the answer",
            ));
        }
    }
    for t in [lse1, lse2, lse] {
        if t.dtype != Dtype::F32 || t.rows != o.rows || t.width != heads {
            return Err(refuse(
                OP,
                "a log-sum-exp plane is one f32 per head per row",
            ));
        }
    }
    let (r, h, d) = (i64::from(o.rows), i64::from(heads), i64::from(head_dim));
    ctx.emit(&mut |cx| {
        let a = cx.read_f32(o1)?;
        let b = cx.read_f32(o2)?;
        let l1 = cx.read_f32(lse1)?;
        let l2 = cx.read_f32(lse2)?;
        let a = cx.reshape(a, &[r, h, d])?;
        let b = cx.reshape(b, &[r, h, d])?;
        let big = cx.like_f(l1, 3.0e38);
        let f1 = {
            let x = cx.abs(l1);
            cx.compare(Cmp::Lt, x, big)?
        };
        let f2 = {
            let x = cx.abs(l2);
            cx.compare(Cmp::Lt, x, big)?
        };
        let mx = cx.max(l1, l2)?;
        let d1 = cx.sub(l1, mx)?;
        let d1 = cx.scale(d1, std::f64::consts::LN_2)?;
        let w1 = cx.exp(d1);
        let d2 = cx.sub(l2, mx)?;
        let d2 = cx.scale(d2, std::f64::consts::LN_2)?;
        let w2 = cx.exp(d2);
        let total = cx.add(w1, w2)?;
        let one = cx.like_f(total, 1.0);
        let inv = cx.div(one, total)?;
        let w1 = cx.mul(w1, inv)?;
        let w2 = cx.mul(w2, inv)?;
        let w1 = cx.broadcast(w1, &[r, h, d], &[0, 1])?;
        let w2 = cx.broadcast(w2, &[r, h, d], &[0, 1])?;
        let x1 = cx.mul(a, w1)?;
        let x2 = cx.mul(b, w2)?;
        let merged = cx.add(x1, x2)?;
        let lt = cx.log(total);
        let lt = cx.scale(lt, std::f64::consts::LOG2_E)?;
        let lm = cx.add(mx, lt)?;
        // A non-finite side yields to the other.
        let f1b = cx.broadcast(f1, &[r, h, d], &[0, 1])?;
        let f2b = cx.broadcast(f2, &[r, h, d], &[0, 1])?;
        let ov = cx.select(f1b, merged, b)?;
        let ov = cx.select(f2b, ov, a)?;
        let lv = cx.select(f1, lm, l2)?;
        let lv = cx.select(f2, lv, l1)?;
        cx.write(o, ov)?;
        cx.write(lse, lv)
    })
}
