#![cfg(feature = "cuda")]

//! The row frame around a hand-written residual norm body lands the chain's
//! math (`super::within`): the stream `y` summed in place bit for bit and
//! the normed row within a norm's ulp, for a row of one to four vectors a
//! thread, a ragged last vector a thread, and the weight taken as is or as
//! `1 + weight`.

use crate::common::attn_ref::{ten, tuned};
use crate::common::{Gpu, Lcg, captured, forget, replay};
use dtype::Dtype::Bf16;
use kernels_cuda::cudarc::runtime::sys as rt;
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

/// The framed residual norm against its unfused ops (`residual_add`, then
/// `rmsnorm`) at a decode's and a prefill's rows over the widths of 4096-,
/// 7168- and 8192-wide models, each fire's bytes (the chain's own: x and y
/// read, y and the normed row written, the weight read) against the device's
/// copy rate, the roofline: `cargo test ... -- --ignored residual_norm_rate
/// --nocapture`. A fire is its graph's fastest replay over
/// its fires; the planes rotate over at least 256 MB so a fire reads HBM, not
/// the L2.
#[test]
#[ignore = "a rate bench: cargo test ... -- --ignored residual_norm_rate --nocapture"]
fn residual_norm_rate() {
    const FOOTPRINT: usize = 256 << 20;
    const FIRES: usize = 256;
    const REPLAYS: u32 = 20;
    let copy = copy_rate();
    eprintln!("device copy: {copy:.0} GB/s");
    eprintln!("rows  width  unfused us  GB/s  %copy  frame us  GB/s  %copy  frame/unfused");
    for rows in [1usize, 8, 64, 512, 4096] {
        for width in [4096usize, 7168, 8192] {
            let mut gpu = Gpu::open();
            let plane = rows * width * 2;
            let copies = FOOTPRINT.div_ceil(3 * plane).clamp(1, FIRES);
            let w = ten(gpu.zeros(width * 2), width, 1, Bf16);
            let sets: Vec<[u64; 3]> = (0..copies)
                .map(|_| [gpu.zeros(plane), gpu.zeros(plane), gpu.zeros(plane)])
                .collect();
            let fires = FIRES.max(copies) / copies * copies;
            let ctx = rows_vec(&gpu);
            let time = |fire: Fire| {
                let all = || {
                    for i in 0..fires {
                        let [x, y, out] = sets[i % copies];
                        fire(
                            &ctx,
                            ten(x, rows, width, Bf16),
                            &mut ten(y, rows, width, Bf16),
                            w,
                            false,
                            1e-6,
                            &mut ten(out, rows, width, Bf16),
                        )
                        .expect("the residual norm fires");
                    }
                };
                all();
                gpu.sync();
                let exec = captured(&ctx, all);
                replay(&ctx, exec);
                gpu.sync();
                let us = fastest(REPLAYS, || {
                    replay(&ctx, exec);
                    gpu.sync();
                });
                forget(exec);
                us / fires as f64
            };
            // x and y read, y and the normed row written, the weight read
            let bytes = (4 * plane + width * 2) as f64;
            let rate = |us: f64| bytes / us / 1e3;
            let (unfused_us, frame_us) = (time(unfused), time(frame::residual_add_rmsnorm));
            eprintln!(
                "{rows:>4}  {width:>5}  {unfused_us:>10.2}  {:>4.0}  {:>4.0}%  {frame_us:>8.2}  {:>4.0}  {:>4.0}%  {:>13.3}",
                rate(unfused_us),
                100.0 * rate(unfused_us) / copy,
                rate(frame_us),
                100.0 * rate(frame_us) / copy,
                frame_us / unfused_us,
            );
        }
    }
}

/// The device's copy rate in GB/s: bytes read plus bytes written by its
/// fastest 1 GiB device-to-device copy.
#[allow(unsafe_code)]
pub(super) fn copy_rate() -> f64 {
    const BYTES: usize = 1 << 30;
    let mut gpu = Gpu::open();
    let (src, dst) = (gpu.zeros(BYTES), gpu.zeros(BYTES));
    let stream = gpu.ctx().stream();
    let copy = || {
        // SAFETY: both ends are `BYTES` long allocations `gpu` holds until
        // its drop, which syncs the stream first.
        let code = unsafe {
            rt::cudaMemcpyAsync(
                dst as *mut _,
                src as *const _,
                BYTES,
                rt::cudaMemcpyKind::cudaMemcpyDeviceToDevice,
                stream.cast(),
            )
        };
        assert_eq!(
            code,
            rt::cudaError::cudaSuccess,
            "`cudaMemcpyAsync` answered {code:?}"
        );
    };
    copy();
    gpu.sync();
    let us = fastest(10, || {
        copy();
        gpu.sync();
    });
    2.0 * BYTES as f64 / us / 1e3
}

/// The fewest microseconds of `runs` runs of `run`: other work on a shared
/// card only ever slows one.
pub(super) fn fastest(runs: u32, mut run: impl FnMut()) -> f64 {
    (0..runs)
        .map(|_| {
            let start = std::time::Instant::now();
            run();
            start.elapsed().as_secs_f64() * 1e6
        })
        .fold(f64::INFINITY, f64::min)
}
