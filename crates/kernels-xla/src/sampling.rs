//! Choosing from rows: argmax and top-k, and the block draft's selector walk.

pub mod selector_walk;

use dtype::Dtype;

use crate::error::{Error, refuse};
use crate::gather::{NONE, nonzero};
use crate::hlo::{Cmp, Elem, Func, Val};
use crate::tensor::{Landing, Operand};

/// Writes the index of each row's largest value into column `column` of the
/// i32 `y`; ties go to the lowest index, NaN is never picked, an all-NaN row
/// answers 0.
/// The `[rows, 1]` i32 column it returns lands in column `column` of `y`.
pub fn argmax(f: &mut Func, x: Operand, column: u32, y: Landing) -> Result<Val, Error> {
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
    let v = x.f32(f);
    let (i, _) = best(f, v, None)?;
    let none = f.like_i(i, NONE);
    let hit = f.compare(Cmp::Ne, i, none)?;
    let zero = f.like_i(i, 0);
    let i = f.select(hit, i, zero)?;
    Ok(f.reshape(i, &[i64::from(x.rows), 1])?)
}

/// The `k` largest values of each row, descending, with their columns; ties
/// go to the lower column, NaN is never picked, and a slot no value fills
/// answers value 0 at column 0. Values land f32, indices i32. Read from
/// kernels-metal `sampling::topk`; the GPU stamps k = 8 and 16,
/// this takes any `k` up to the row width.
pub fn topk(
    f: &mut Func,
    x: Operand,
    k: u32,
    values: Landing,
    indices: Landing,
) -> Result<(Val, Val), Error> {
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
    let r = i64::from(rows);
    let dims = [r, i64::from(x.width)];
    let v = x.f32(f);
    let col = f.iota(Elem::I32, &dims, 1);
    let mut taken = f.const_i(Elem::Pred, 0, &dims);
    let (mut vs, mut is) = (Vec::new(), Vec::new());
    for _ in 0..k {
        let (i, m) = best(f, v, Some(taken))?;
        let wide = f.broadcast(i, &dims, &[0])?;
        let hit = f.compare(Cmp::Eq, col, wide)?;
        taken = f.or(taken, hit)?;
        let none = f.like_i(i, NONE);
        let found = f.compare(Cmp::Ne, i, none)?;
        let zi = f.like_i(i, 0);
        let zv = f.like_f(m, 0.0);
        let i = f.select(found, i, zi)?;
        let m = f.select(found, m, zv)?;
        is.push(f.reshape(i, &[r, 1])?);
        vs.push(f.reshape(m, &[r, 1])?);
    }
    let vs = f.concat(&vs, 1)?;
    let is = f.concat(&is, 1)?;
    Ok((vs, is))
}

/// The best column of each row of an f32 `[rows, width]` — largest value,
/// lowest index on a tie, NaN never picked, nor any column `skip` marks —
/// as `(index, value)`, `[rows]` each. A row with nothing to pick answers
/// index [`NONE`].
fn best(f: &mut Func, x: Val, skip: Option<Val>) -> Result<(Val, Val), Error> {
    let dims = f.dims(x).to_vec();
    let col = f.iota(Elem::I32, &dims, 1);
    let mut ok = f.compare(Cmp::Eq, x, x)?;
    if let Some(skip) = skip {
        let keep = f.not(skip);
        ok = f.and(ok, keep)?;
    }
    let none = f.like_i(col, NONE);
    let idx = f.select(ok, col, none)?;
    let lo = f.const_f(Elem::F32, f64::NEG_INFINITY, &[]);
    let nil = f.const_i(Elem::I32, NONE, &[]);
    let out = f.reduce_with(&[x, idx], &[lo, nil], &[1], |fq, a, b| {
        let (av, ai, bv, bi) = (a[0], a[1], b[0], b[1]);
        let n = fq.const_i(Elem::I32, NONE, &[]);
        let a_ok = fq.compare(Cmp::Ne, ai, n)?;
        let b_none = fq.compare(Cmp::Eq, bi, n)?;
        let gt = fq.compare(Cmp::Gt, av, bv)?;
        let eq = fq.compare(Cmp::Eq, av, bv)?;
        let lt = fq.compare(Cmp::Lt, ai, bi)?;
        let tie = fq.and(eq, lt)?;
        let wins = fq.or(gt, tie)?;
        let wins = fq.or(b_none, wins)?;
        let take = fq.and(a_ok, wins)?;
        let v = fq.select(take, av, bv)?;
        let i = fq.select(take, ai, bi)?;
        Ok(vec![v, i])
    })?;
    Ok((out[1], out[0]))
}
