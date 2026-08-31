//! **WHAT THE VECTOR POINT IS ACTUALLY BOUND BY**, and the two tables the
//! dense ladder's narrow band was decided on.
//!
//! `linear::quant`'s vector arm costs very nearly one bank read per row, and
//! the natural reading of that is a kernel repeating a fetch it could share.
//! This file is the measurement that says otherwise, and it is here rather
//! than in a notebook because two constants —
//! [`DeviceTuning::qmv_rows_max`] and [`DeviceTuning::qmv_rows_packs`] — are
//! set from it.
//!
//! # The two arms
//!
//! 1. **The roofline arm.** Both points over a bank sized past this machine's
//!    48 MiB system cache, so every weight byte is DRAM. The one-row point
//!    reaches 365 GB/s from two rows up, which is the M1 Max's ceiling.
//! 2. **The ALU arm.** The same points over a slice of the same bank small
//!    enough to sit IN that cache, scaled back to the full bank's size. If
//!    the kernel were bound by the read, this column would collapse. It does
//!    not move — so the vector point is arithmetic-bound, and the 365 GB/s is
//!    a coincidence of this machine's balance rather than the wall.
//!
//! Nothing here is asserted. The numbers are the output, and
//! `affine_floor` is where this family's answers are gated.
//!
//! ```text
//! cargo test -p engine-metal --release --test what_the_vector_point_is_bound_by -- --nocapture
//! ```
//!
//! [`DeviceTuning::qmv_rows_max`]: kernels_metal::tuning::DeviceTuning::qmv_rows_max
//! [`DeviceTuning::qmv_rows_packs`]: kernels_metal::tuning::DeviceTuning::qmv_rows_packs

#![cfg(target_vendor = "apple")]

use std::time::Instant;

use engine_metal::device::{self, Buffer, Context, Handles, Pipelines};
use engine_metal::encode::Sink;
use kernels_metal::linear::quant;
use kernels_metal::{Arg, Ctx, Fire, Grid, Tensor};
use model_ir::Dtype;

const QMV_FILE: &str = "linear/quant_qmv.metal";
const QMV_ROWS_FILE: &str = "linear/quant_qmv_rows.metal";
const QMV_GROUP: [u32; 3] = [32, 2, 1];
const GROUP: u32 = 64;
const BITS: u32 = 4;

/// A bank sized past the M1 Max's 48 MiB system cache, so the fires below
/// measure DRAM and not residency: 16384 x 8192 codes is 64 MiB, plus 8 MiB
/// of scales and zero points.
const N: u32 = 16384;
const K: u32 = 8192;
const ROWS: u32 = 16;

/// A slice whose codes are 4 MiB — comfortably inside the system cache.
const SMALL_N: u32 = 1024;

const REPS: usize = 32;

fn bytes_read() -> f64 {
    // codes + scales + biases, which is every byte a fire of one row group
    // touches on the weight side.
    f64::from(N) * f64::from(K) / 2.0 + 2.0 * f64::from(N) * f64::from(K) / f64::from(GROUP) * 2.0
}

#[test]
fn qmv_bandwidth() {
    if !device::present() {
        println!("SKIP: no Metal device");
        return;
    }
    let device = Context::bind().expect("device");
    let pipelines = Pipelines::new();

    let codes = Buffer::zeroed(&device, u64::from(N) * u64::from(K) / 2).expect("codes");
    let scales =
        Buffer::zeroed(&device, u64::from(N) * u64::from(K) / u64::from(GROUP) * 2).expect("s");
    let biases =
        Buffer::zeroed(&device, u64::from(N) * u64::from(K) / u64::from(GROUP) * 2).expect("b");
    let act = Buffer::zeroed(&device, u64::from(ROWS) * u64::from(K) * 2).expect("x");
    let y = Buffer::zeroed(&device, u64::from(ROWS) * u64::from(N) * 2).expect("y");

    let handles = Handles::new();
    let bind = |b: &Buffer| handles.bind(b, 0, b.bytes()).expect("bind");
    let t_codes = Tensor::new(bind(&codes), N, K, Dtype::MlxU4);
    let t_scales = Tensor::new(bind(&scales), N, K / GROUP, Dtype::Bf16);
    let t_biases = Tensor::new(bind(&biases), N, K / GROUP, Dtype::Bf16);
    let t_x = Tensor::new(bind(&act), ROWS, K, Dtype::Bf16);
    let t_y = Tensor::new(bind(&y), ROWS, N, Dtype::Bf16);

    let run = |label: &str, m: u32, fold: Option<(i32, i32)>| {
        let k = K as i32;
        let n = N as i32;
        let fire_one = |sink: &Ctx<'_>| -> Result<(), kernels_metal::Error> {
            match fold {
                None => {
                    let p = quant::qmv_point("bench", "fast", GROUP as i32, BITS as i32)?;
                    sink.fire(
                        Fire::at(QMV_FILE, p.entry)
                            .apply(Grid::of(quant::qmv_grid("bench", m as i32, n)?, QMV_GROUP)),
                        &[
                            t_codes.arg(),
                            t_scales.arg(),
                            t_biases.arg(),
                            t_x.arg(),
                            t_y.arg_mut(),
                            k.arg(),
                            n.arg(),
                        ],
                    )
                }
                Some((r, p)) => {
                    let pt = quant::qmv_rows_point("bench", GROUP as i32, BITS as i32, r, p)?;
                    sink.fire(
                        Fire::at(QMV_ROWS_FILE, pt.entry)
                            .stamp(pt.stamp)
                            .apply(Grid::of(
                                quant::qmv_rows_grid("bench", m as i32, r, n)?,
                                QMV_GROUP,
                            )),
                        &[
                            t_codes.arg(),
                            t_scales.arg(),
                            t_biases.arg(),
                            t_x.arg(),
                            t_y.arg_mut(),
                            k.arg(),
                            n.arg(),
                            (m as i32).arg(),
                        ],
                    )
                }
            }
        };
        // Every fire in ONE frame: a commit is a round trip worth a few
        // hundred microseconds, which is most of what a single fire costs.
        let burst = |n: usize| {
            let frame = device.frame().expect("frame");
            {
                let sink = Sink::new(&device, &frame, &pipelines, &handles);
                for _ in 0..n {
                    fire_one(&sink).expect("encode");
                }
            }
            frame.commit().expect("commit");
        };
        burst(4);
        let start = Instant::now();
        burst(REPS);
        let ms = start.elapsed().as_secs_f64() * 1000.0 / REPS as f64;
        let groups = match fold {
            None => f64::from(m),
            Some((r, _)) => (f64::from(m) / f64::from(r as u32)).ceil(),
        };
        let gbs = groups * bytes_read() / (ms / 1000.0) / 1e9;
        println!(
            "  {label:<18} m={m:<3} {ms:>8.3} ms  {gbs:>7.1} GB/s  (per-row {:>7.3} ms)",
            ms / f64::from(m)
        );
    };

    println!(
        "\nqmv bandwidth, bank {N}x{K} gs{GROUP} b{BITS} = {:.1} MiB",
        bytes_read() / 1048576.0
    );
    for m in [1u32, 2, 3, 4, 6, 8, 12, 16] {
        run("one-row", m, None);
    }
    for &(r, p) in &[(2, 2), (2, 1), (4, 2), (4, 1), (8, 1), (8, 2)] {
        println!("  -- fold r={r} p={p}");
        for m in [2u32, 4, 8, 16] {
            if m >= r as u32 {
                run("fold", m, Some((r, p)));
            }
        }
    }

    // ── The ALU question. Same kernels, a slice narrow enough to sit in the
    //    48 MiB system cache, so the weight read is nearly free and what is
    //    left is arithmetic. If the per-row time collapses here, the fold's
    //    ceiling is bandwidth-related; if it does not, the fold's ceiling is
    //    the arithmetic and no fold can move it.
    println!(
        "\n  -- cache-resident: the same points over a {} MiB slice",
        SMALL_N * K / 2 / 1048576
    );
    let run_small = |label: &str, m: u32, fold: Option<(i32, i32)>| {
        let k = K as i32;
        let n = SMALL_N as i32;
        let fire_one = |sink: &Ctx<'_>| -> Result<(), kernels_metal::Error> {
            match fold {
                None => {
                    let p = quant::qmv_point("bench", "fast", GROUP as i32, BITS as i32)?;
                    sink.fire(
                        Fire::at(QMV_FILE, p.entry)
                            .apply(Grid::of(quant::qmv_grid("bench", m as i32, n)?, QMV_GROUP)),
                        &[
                            t_codes.arg(),
                            t_scales.arg(),
                            t_biases.arg(),
                            t_x.arg(),
                            t_y.arg_mut(),
                            k.arg(),
                            n.arg(),
                        ],
                    )
                }
                Some((r, p)) => {
                    let pt = quant::qmv_rows_point("bench", GROUP as i32, BITS as i32, r, p)?;
                    sink.fire(
                        Fire::at(QMV_ROWS_FILE, pt.entry)
                            .stamp(pt.stamp)
                            .apply(Grid::of(
                                quant::qmv_rows_grid("bench", m as i32, r, n)?,
                                QMV_GROUP,
                            )),
                        &[
                            t_codes.arg(),
                            t_scales.arg(),
                            t_biases.arg(),
                            t_x.arg(),
                            t_y.arg_mut(),
                            k.arg(),
                            n.arg(),
                            (m as i32).arg(),
                        ],
                    )
                }
            }
        };
        let burst = |n: usize| {
            let frame = device.frame().expect("frame");
            {
                let sink = Sink::new(&device, &frame, &pipelines, &handles);
                for _ in 0..n {
                    fire_one(&sink).expect("encode");
                }
            }
            frame.commit().expect("commit");
        };
        burst(4);
        let start = Instant::now();
        burst(REPS);
        let ms = start.elapsed().as_secs_f64() * 1000.0 / REPS as f64;
        // Scaled to the full bank, so the column is comparable to the ones
        // above: this slice is SMALL_N / N of the whole.
        let scaled = ms * f64::from(N) / f64::from(SMALL_N);
        println!(
            "  {label:<18} m={m:<3} {ms:>8.3} ms  (per-row, scaled to the full bank {:>7.3} ms)",
            scaled / f64::from(m)
        );
    };
    for m in [1u32, 2, 4, 8, 16] {
        run_small("one-row/cached", m, None);
    }
    for m in [2u32, 4, 8, 16] {
        run_small("fold r2p1/cached", m, Some((2, 1)));
    }
    for m in [4u32, 8, 16] {
        run_small("fold r4p1/cached", m, Some((4, 1)));
    }
}
