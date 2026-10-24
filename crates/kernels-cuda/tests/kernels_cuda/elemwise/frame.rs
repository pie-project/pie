#![cfg(feature = "cuda")]

//! The row frame around a hand-written `residual_add_rmsnorm` body lands the
//! hand register kernel's bits: the stream `y` summed in place and the normed
//! row, for a row of one to four vectors a thread, a ragged last vector a
//! thread, and the weight taken as is or as `1 + weight`.

use crate::common::attn_ref::{ten, tuned};
use crate::common::{Gpu, Lcg, captured, forget, replay};
use dtype::Dtype::Bf16;
use kernels_cuda::cudarc::runtime::sys as rt;
use kernels_cuda::elemwise::{frame, norm_residual};
use kernels_cuda::jit::Ctx;
use kernels_cuda::tensor::Tensor;

/// The hand kernel's register form (`kernel.elemwise.rows_vec`).
fn hand(gpu: &Gpu) -> Ctx {
    gpu.ctx().with_tuning(tuned(|t| t.rows_vec = true))
}

type Fire = fn(
    &Ctx,
    Tensor,
    &mut Tensor,
    Tensor,
    bool,
    f32,
    &mut Tensor,
) -> Result<(), kernels_cuda::error::Error>;

/// `y` and the normed row after `fire` over one seeded draw.
fn land(fire: Fire, rows: usize, width: usize, plus_one: bool) -> (Vec<u16>, Vec<u16>) {
    let n = rows * width;
    let mut lcg = Lcg::seeded(0xf4a3e ^ ((width as u64) << 8) ^ rows as u64);
    let (x, _) = lcg.row(n);
    let (y, _) = lcg.row(n);
    let (w, _) = lcg.row(width);
    let mut gpu = Gpu::open();
    let (x_at, y_at, w_at, out_at) = (gpu.up(&x), gpu.up(&y), gpu.up(&w), gpu.zeros(n * 2));
    fire(
        &hand(&gpu),
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
fn the_framed_residual_norm_lands_the_register_kernels_bits() {
    for width in [4096usize, 7168, 8192, 2048, 1544, 6152] {
        for rows in [1usize, 7, 64] {
            for plus_one in [false, true] {
                let case = format!("{rows} rows of {width}, plus one {plus_one}");
                let (y, out) = land(frame::residual_add_rmsnorm, rows, width, plus_one);
                let (y_want, out_want) =
                    land(norm_residual::residual_add_rmsnorm, rows, width, plus_one);
                assert!(y == y_want, "{case}: the summed stream");
                assert!(out == out_want, "{case}: the normed row");
            }
        }
    }
}

/// The framed and the register residual norm at a decode's and a prefill's
/// rows over the widths of 4096-, 7168- and 8192-wide models, each fire's
/// bytes against the device's copy rate: `cargo test ... -- --ignored
/// residual_norm_rate --nocapture`. A fire is its graph's fastest replay over
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
    eprintln!("rows  width  hand us  GB/s  %copy  frame us  GB/s  %copy  frame/hand");
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
            let ctx = hand(&gpu);
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
            let (hand_us, frame_us) = (
                time(norm_residual::residual_add_rmsnorm),
                time(frame::residual_add_rmsnorm),
            );
            eprintln!(
                "{rows:>4}  {width:>5}  {hand_us:>7.2}  {:>4.0}  {:>4.0}%  {frame_us:>8.2}  {:>4.0}  {:>4.0}%  {:>10.3}",
                rate(hand_us),
                100.0 * rate(hand_us) / copy,
                rate(frame_us),
                100.0 * rate(frame_us) / copy,
                frame_us / hand_us,
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
