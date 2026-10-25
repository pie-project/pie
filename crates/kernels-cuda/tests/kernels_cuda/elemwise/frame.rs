#![cfg(feature = "cuda")]

//! The row frame around a hand-written residual norm body lands the chain's
//! math (`super::within`): the stream `y` summed in place bit for bit and
//! the normed row within a norm's ulp, for a row of one to four vectors a
//! thread, a ragged last vector a thread, and the weight taken as is or as
//! `1 + weight`.

use crate::common::attn_ref::{ten, tuned};
use crate::common::{Gpu, Lcg};
use dtype::Dtype::Bf16;
use kernels_cuda::elemwise::{frame, norm, pointwise};
use kernels_cuda::jit::Ctx;
use kernels_cuda::tensor::Tensor;

use super::within::{EXACT, NORM, assert_within, deterministic, reference, row_of, widened};

/// The unfused ops' register forms (`kernel.elemwise.rows_vec`).
fn rows_vec(gpu: &Gpu) -> Ctx {
    gpu.ctx().with_tuning(tuned(|t| t.rows_vec = true))
}

pub(super) type Fire = fn(
    &Ctx,
    Tensor,
    &mut Tensor,
    Tensor,
    bool,
    f32,
    &mut Tensor,
) -> Result<(), kernels_cuda::error::Error>;

/// The chain unfused, as the engine fires its ops: `y += x` in place, then
/// the RMS norm of the stored `y` into `out`.
pub(super) fn unfused(
    ctx: &Ctx,
    x: Tensor,
    y: &mut Tensor,
    weight: Tensor,
    plus_one: bool,
    eps: f32,
    out: &mut Tensor,
) -> Result<(), kernels_cuda::error::Error> {
    pointwise::residual_add(ctx, x, y)?;
    if plus_one {
        norm::rmsnorm_plus_one(ctx, *y, weight, eps, out)
    } else {
        norm::rmsnorm(ctx, *y, weight, eps, out)
    }
}

/// How many elements of `got` are not `want`, and the most bf16 ulps (as
/// bit patterns of one sign) one is apart.
pub(super) fn apart(got: &[u16], want: &[u16]) -> (usize, u16) {
    got.iter()
        .zip(want)
        .filter(|(a, b)| a != b)
        .fold((0, 0), |(n, most), (a, b)| {
            (n + 1, most.max(a.abs_diff(*b)))
        })
}

/// One seeded draw: x, y, the weight.
fn draw(rows: usize, width: usize) -> (Vec<u16>, Vec<u16>, Vec<u16>) {
    let n = rows * width;
    let mut lcg = Lcg::seeded(0xf4a3e ^ ((width as u64) << 8) ^ rows as u64);
    let (x, _) = lcg.row(n);
    let (y, _) = lcg.row(n);
    let (w, _) = lcg.row(width);
    (x, y, w)
}

/// The chain's region by the interpreter over the draw: the stream, the
/// normed row.
fn math(rows: usize, width: usize, plus_one: bool) -> Vec<Vec<f32>> {
    use kernels_ir::{Binary, Literal, Op, Reduce, Region, Shape, Unary};
    let mut ops = vec![
        Op::Binary(Binary::Add, 1, 0),
        Op::Binary(Binary::Mul, 3, 3),
        Op::Reduce(Reduce::Sum, 4),
        Op::Const(Literal::F32(width as f32)),
        Op::Binary(Binary::Div, 5, 6),
        Op::Const(Literal::F32(1e-6)),
        Op::Binary(Binary::Add, 7, 8),
        Op::Unary(Unary::Rsqrt, 9),
        Op::Binary(Binary::Mul, 3, 10),
    ];
    if plus_one {
        ops.push(Op::Const(Literal::F32(1.0)));
        ops.push(Op::Binary(Binary::Add, 2, 12));
        ops.push(Op::Binary(Binary::Mul, 11, 13));
    } else {
        ops.push(Op::Binary(Binary::Mul, 11, 2));
    }
    let out = 2 + ops.len() as u32;
    let region = Region {
        width: width as u32,
        params: vec![Shape::Row; 3],
        ops,
        results: vec![3, out],
        rounds: vec![(3, Bf16), (out, Bf16)],
    };
    let (x, y, w) = draw(rows, width);
    reference(&region, rows, |r| {
        vec![
            row_of(&x, r, width),
            row_of(&y, r, width),
            kernels_ir::interp::Arg::Row(widened(&w)),
        ]
    })
}

/// `y` and the normed row after `fire` over one seeded draw.
fn land(fire: Fire, rows: usize, width: usize, plus_one: bool) -> (Vec<u16>, Vec<u16>) {
    let n = rows * width;
    let (x, y, w) = draw(rows, width);
    let mut gpu = Gpu::open();
    let (x_at, y_at, w_at, out_at) = (gpu.up(&x), gpu.up(&y), gpu.up(&w), gpu.zeros(n * 2));
    fire(
        &rows_vec(&gpu),
        ten(x_at, rows, width, Bf16),
        &mut ten(y_at, rows, width, Bf16),
        ten(w_at, width, 1, Bf16),
        plus_one,
        1e-6,
        &mut ten(out_at, rows, width, Bf16),
    )
    .expect("the residual norm fires");
    gpu.sync();
    (gpu.down(y_at, n), gpu.down(out_at, n))
}

#[test]
fn the_framed_residual_norm_lands_its_math() {
    for width in [4096usize, 7168, 8192, 2048, 1544, 6152] {
        for rows in [1usize, 7, 64] {
            for plus_one in [false, true] {
                let case = format!("{rows} rows of {width}, plus one {plus_one}");
                let (y, out) = land(frame::residual_add_rmsnorm, rows, width, plus_one);
                deterministic(
                    &case,
                    &out,
                    &land(frame::residual_add_rmsnorm, rows, width, plus_one).1,
                );
                let want = math(rows, width, plus_one);
                assert_within(&format!("{case}: the summed stream"), &y, &want[0], EXACT);
                assert_within(&format!("{case}: the normed row"), &out, &want[1], NORM);
                let (_, out_unfused) = land(unfused, rows, width, plus_one);
                let (n, _) = apart(&out, &out_unfused);
                eprintln!(
                    "{case}: {n} of {} normed elements apart from the unfused ops",
                    out.len()
                );
            }
        }
    }
}
