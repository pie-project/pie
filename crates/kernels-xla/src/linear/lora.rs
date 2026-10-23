//! Multi-adapter LoRA: `y += B[a] · (A[a] · x)` per row, `a = routes[row]`,
//! a negative (or unseated) adapter id adding nothing.
//!
//! The banks are `bank_a: [adapters, rank · in]` (adapter `a`'s `[rank, in]`
//! down projection, row-major) and `bank_b: [adapters, out · rank]` (its
//! `[out, rank]` up projection), as kernels-wgpu `gemm/lora.wgsl` reads them.
//! Rather than gather a per-row adapter, every row is contracted against the
//! whole down bank in one dot (`[rows, adapters · rank]`), the waist is
//! masked to the row's own adapter, and one dot over `(adapter, rank)` lands
//! the correction: two MXU matmuls, static shapes. The waist stays f32, as
//! the shader keeps it; the up dot takes it as two bf16 terms (16 significant
//! bits) against the exactly-bf16 up bank.

use dtype::Dtype;

use crate::check::expect;
use crate::error::{Error, refuse};
use crate::hlo::{Cmp, Elem, Func, Val};
use crate::linear::gemm::dot_split;
use crate::tensor::Operand;

pub fn correct(
    f: &mut Func,
    x: Operand,
    bank_a: Operand,
    bank_b: Operand,
    routes: Operand,
    y: Operand,
) -> Result<Option<Val>, Error> {
    const OP: &str = "linear.lora_correct";
    expect(OP, x, &[Dtype::Bf16])?;
    expect(OP, y, &[Dtype::Bf16])?;
    expect(OP, bank_a, &[Dtype::Bf16])?;
    expect(OP, bank_b, &[Dtype::Bf16])?;
    expect(OP, routes, &[Dtype::I32])?;
    let (rows, in_width, out_width) = (x.rows, x.width, y.width);
    if in_width == 0 || out_width == 0 {
        return Err(refuse(
            OP,
            "the correction's input and output widths are nonzero",
        ));
    }
    if y.rows != rows || routes.elements() != u64::from(rows) {
        return Err(refuse(
            OP,
            format!(
                "{rows} input rows land {} rows under {} adapter ids",
                y.rows,
                routes.elements()
            ),
        ));
    }
    if !bank_a.width.is_multiple_of(in_width) || bank_a.width == 0 {
        return Err(refuse(
            OP,
            format!(
                "the down bank is {} wide over an input of {in_width}, which is not a \
                 whole number of ranks",
                bank_a.width
            ),
        ));
    }
    let rank = bank_a.width / in_width;
    if bank_b.width != out_width.saturating_mul(rank) {
        return Err(refuse(
            OP,
            format!(
                "the up bank is {} wide where {out_width} x {rank} is {}",
                bank_b.width,
                out_width.saturating_mul(rank),
            ),
        ));
    }
    if bank_a.rows != bank_b.rows || bank_a.rows == 0 {
        return Err(refuse(
            OP,
            format!(
                "the bank's two planes seat {} and {} adapters",
                bank_a.rows, bank_b.rows
            ),
        ));
    }
    if rows == 0 {
        return Ok(None);
    }
    let (m, n_in, n_out) = (i64::from(rows), i64::from(in_width), i64::from(out_width));
    let (adapters, r) = (i64::from(bank_a.rows), i64::from(rank));
    let xv = x.val;
    let a = bank_a.val;
    let a = f.reshape(a, &[adapters * r, n_in])?;
    let t = f.matmul_nt(xv, a, Elem::F32)?;
    let t = f.reshape(t, &[m, adapters, r])?;

    let ids = routes.val;
    let ids = f.reshape(ids, &[m])?;
    let ids = f.broadcast(ids, &[m, adapters], &[0])?;
    let seat = f.iota(Elem::I32, &[m, adapters], 1);
    let mine = f.compare(Cmp::Eq, ids, seat)?;
    let mine = f.broadcast(mine, &[m, adapters, r], &[0, 1])?;
    let zero = f.like_f(t, 0.0);
    let t = f.select(mine, t, zero)?;

    let b = bank_b.val;
    let b = f.reshape(b, &[adapters, n_out, r])?;
    let corr = dot_split(f, t, b, 2, &[], &[], &[1, 2], &[0, 2])?;

    let yv = y.f32(f);
    let out = f.add(yv, corr)?;
    Ok(Some(out))
}
