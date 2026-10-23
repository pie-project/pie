//! Folding two partial readings by their log-sum-exps.

#![allow(clippy::too_many_arguments)]

use dtype::Dtype;

use crate::cx::expect;
use crate::error::{Error, refuse};
use crate::hlo::{Cmp, Func, Val};
use crate::tensor::{Landing, Operand, Stored};

/// `o = (o1·w1 + o2·w2) / (w1 + w2)` with `w = 2^(lse − max)`, and
/// `lse = max + log2(w1 + w2)`; a side whose lse is not finite yields to the
/// other unchanged (base-2 lses, as the attention kernels publish them).
pub fn merge_lse(
    f: &mut Func,
    o1: Operand,
    lse1: Operand,
    o2: Operand,
    lse2: Operand,
    heads: u32,
    head_dim: u32,
    o: Landing,
    lse: Landing,
) -> Result<(Val, Val), Error> {
    const OP: &str = "attn.merge_lse";
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
    for t in [lse1.shape(), lse2.shape(), lse] {
        if t.dtype != Dtype::F32 || t.rows != o.rows || t.width != heads {
            return Err(refuse(
                OP,
                "a log-sum-exp plane is one f32 per head per row",
            ));
        }
    }
    let (r, h, d) = (i64::from(o.rows), i64::from(heads), i64::from(head_dim));
    let a = o1.f32(f);
    let b = o2.f32(f);
    let l1 = lse1.f32(f);
    let l2 = lse2.f32(f);
    let a = f.reshape(a, &[r, h, d])?;
    let b = f.reshape(b, &[r, h, d])?;
    let big = f.like_f(l1, 3.0e38);
    let f1 = {
        let x = f.abs(l1);
        f.compare(Cmp::Lt, x, big)?
    };
    let f2 = {
        let x = f.abs(l2);
        f.compare(Cmp::Lt, x, big)?
    };
    let mx = f.max(l1, l2)?;
    let d1 = f.sub(l1, mx)?;
    let d1 = f.scale(d1, std::f64::consts::LN_2)?;
    let w1 = f.exp(d1);
    let d2 = f.sub(l2, mx)?;
    let d2 = f.scale(d2, std::f64::consts::LN_2)?;
    let w2 = f.exp(d2);
    let total = f.add(w1, w2)?;
    let one = f.like_f(total, 1.0);
    let inv = f.div(one, total)?;
    let w1 = f.mul(w1, inv)?;
    let w2 = f.mul(w2, inv)?;
    let w1 = f.broadcast(w1, &[r, h, d], &[0, 1])?;
    let w2 = f.broadcast(w2, &[r, h, d], &[0, 1])?;
    let x1 = f.mul(a, w1)?;
    let x2 = f.mul(b, w2)?;
    let merged = f.add(x1, x2)?;
    let lt = f.log(total);
    let lt = f.scale(lt, std::f64::consts::LOG2_E)?;
    let lm = f.add(mx, lt)?;
    // A non-finite side yields to the other.
    let f1b = f.broadcast(f1, &[r, h, d], &[0, 1])?;
    let f2b = f.broadcast(f2, &[r, h, d], &[0, 1])?;
    let ov = f.select(f1b, merged, b)?;
    let ov = f.select(f2b, ov, a)?;
    let lv = f.select(f1, lm, l2)?;
    let lv = f.select(f2, lv, l1)?;
    Ok((ov, lv))
}
