//! Choosing from rows: argmax and top-k, and the block draft's selector walk.

pub mod selector_walk;

use dtype::Dtype;

use crate::cx::{Ctx, Cx};
use crate::error::{Error, refuse};
use crate::gather::{NONE, nonzero};
use crate::hlo::{Cmp, Elem, Val};
use crate::tensor::Tensor;

/// Writes the index of each row's largest value into column `column` of the
/// i32 `y`; ties go to the lowest index, NaN is never picked, an all-NaN row
/// answers 0.
pub fn argmax(ctx: &Ctx<'_>, x: Tensor, column: u32, y: Tensor) -> Result<(), Error> {
    const OP: &str = "sampling.argmax";
    if !matches!(x.dtype, Dtype::Bf16 | Dtype::F32 | Dtype::F16) {
        return Err(Error::DtypeUnsupported {
            op: OP,
            dtype: x.dtype,
        });
    }
    if y.dtype != Dtype::I32 {
        return Err(Error::DtypeUnsupported {
            op: OP,
            dtype: y.dtype,
        });
    }
    nonzero(OP, "rows", x.rows)?;
    nonzero(OP, "width", x.width)?;
    if column >= y.width {
        return Err(refuse(
            OP,
            format!(
                "column {column} is outside the {}-wide plane it writes",
                y.width
            ),
        ));
    }
    if x.rows != y.rows {
        return Err(refuse(
            OP,
            format!("{} rows ranked into {} rows", x.rows, y.rows),
        ));
    }
    ctx.emit(&mut |cx| {
        let v = cx.read_f32(x)?;
        let (i, _) = best(cx, v, None)?;
        let none = cx.like_i(i, NONE);
        let hit = cx.compare(Cmp::Ne, i, none)?;
        let zero = cx.like_i(i, 0);
        let i = cx.select(hit, i, zero)?;
        let i = cx.reshape(i, &[i64::from(x.rows), 1])?;
        if y.width == 1 {
            return cx.write(y, i);
        }
        let old = cx.read(y)?;
        let r = cx.const_i(Elem::I32, 0, &[]);
        let c = cx.const_i(Elem::I32, i64::from(column), &[]);
        let merged = cx.dynamic_update_slice(old, i, &[r, c])?;
        cx.write(y, merged)
    })
}

/// The `k` largest values of each row, descending, with their columns; ties
/// go to the lower column, NaN is never picked, and a slot no value fills
/// answers value 0 at column 0. Values land f32, indices i32. Read from
/// kernels-metal `sampling::topk`; the GPU stamps k = 8 and 16,
/// this takes any `k` up to the row width.
pub fn topk(
    ctx: &Ctx<'_>,
    x: Tensor,
    k: u32,
    values: Tensor,
    indices: Tensor,
) -> Result<(), Error> {
    const OP: &str = "sampling.top_k";
    if !matches!(x.dtype, Dtype::Bf16 | Dtype::F32 | Dtype::F16) {
        return Err(Error::DtypeUnsupported {
            op: OP,
            dtype: x.dtype,
        });
    }
    let rows = nonzero(OP, "rows", x.rows)?;
    nonzero(OP, "width", x.width)?;
    nonzero(OP, "k", k)?;
    if k > x.width {
        return Err(refuse(
            OP,
            format!("k = {k} is wider than the {}-wide row", x.width),
        ));
    }
    if values.rows != rows || values.width != k || values.dtype != Dtype::F32 {
        return Err(refuse(
            OP,
            format!("the values plane is not [{rows}, {k}] f32"),
        ));
    }
    if indices.rows != rows || indices.width != k || indices.dtype != Dtype::I32 {
        return Err(refuse(
            OP,
            format!("the indices plane is not [{rows}, {k}] i32"),
        ));
    }
    ctx.emit(&mut |cx| {
        let r = i64::from(rows);
        let dims = [r, i64::from(x.width)];
        let v = cx.read_f32(x)?;
        let col = cx.iota(Elem::I32, &dims, 1);
        let mut taken = cx.const_i(Elem::Pred, 0, &dims);
        let (mut vs, mut is) = (Vec::new(), Vec::new());
        for _ in 0..k {
            let (i, m) = best(cx, v, Some(taken))?;
            let wide = cx.broadcast(i, &dims, &[0])?;
            let hit = cx.compare(Cmp::Eq, col, wide)?;
            taken = cx.or(taken, hit)?;
            let none = cx.like_i(i, NONE);
            let found = cx.compare(Cmp::Ne, i, none)?;
            let zi = cx.like_i(i, 0);
            let zv = cx.like_f(m, 0.0);
            let i = cx.select(found, i, zi)?;
            let m = cx.select(found, m, zv)?;
            is.push(cx.reshape(i, &[r, 1])?);
            vs.push(cx.reshape(m, &[r, 1])?);
        }
        let vs = cx.concat(&vs, 1)?;
        let is = cx.concat(&is, 1)?;
        cx.write(values, vs)?;
        cx.write(indices, is)
    })
}

/// The best column of each row of an f32 `[rows, width]` — largest value,
/// lowest index on a tie, NaN never picked, nor any column `skip` marks —
/// as `(index, value)`, `[rows]` each. A row with nothing to pick answers
/// index [`NONE`].
fn best(cx: &mut Cx<'_>, x: Val, skip: Option<Val>) -> Result<(Val, Val), Error> {
    let dims = cx.dims(x).to_vec();
    let col = cx.iota(Elem::I32, &dims, 1);
    let mut ok = cx.compare(Cmp::Eq, x, x)?;
    if let Some(skip) = skip {
        let keep = cx.not(skip);
        ok = cx.and(ok, keep)?;
    }
    let none = cx.like_i(col, NONE);
    let idx = cx.select(ok, col, none)?;
    let lo = cx.const_f(Elem::F32, f64::NEG_INFINITY, &[]);
    let nil = cx.const_i(Elem::I32, NONE, &[]);
    let out = cx.reduce_with(&[x, idx], &[lo, nil], &[1], |f, a, b| {
        let (av, ai, bv, bi) = (a[0], a[1], b[0], b[1]);
        let n = f.const_i(Elem::I32, NONE, &[]);
        let a_ok = f.compare(Cmp::Ne, ai, n)?;
        let b_none = f.compare(Cmp::Eq, bi, n)?;
        let gt = f.compare(Cmp::Gt, av, bv)?;
        let eq = f.compare(Cmp::Eq, av, bv)?;
        let lt = f.compare(Cmp::Lt, ai, bi)?;
        let tie = f.and(eq, lt)?;
        let wins = f.or(gt, tie)?;
        let wins = f.or(b_none, wins)?;
        let take = f.and(a_ok, wins)?;
        let v = f.select(take, av, bv)?;
        let i = f.select(take, ai, bi)?;
        Ok(vec![v, i])
    })?;
    Ok((out[1], out[0]))
}
