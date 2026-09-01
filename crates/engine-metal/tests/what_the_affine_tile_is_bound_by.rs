//! **WHAT THE AFFINE TILE IS ACTUALLY BOUND BY**, at every row rung it is
//! stamped at — the measurement the qmv file's twin makes for the vector arm,
//! made here for the GEMM.
//!
//! `what_the_vector_point_is_bound_by` settled the matvec: it is arithmetic
//! and not the bus. The tile had never been asked the same question over its
//! own ladder. Every number the bench wiki carries for it is either a
//! whole-model prefill (which mixes four kernels and a scan) or the retired
//! C++ `roofline_probe` (which is not in this tree any more), and a lever
//! aimed at the tile wants a bench that fires ONLY the tile.
//!
//! # The five arms
//!
//! 1. **The row ladder.** The shipped point at the dense projection shape
//!    `K = N = 5120`, at every `m` from the 8 rung to a prefill block, with
//!    `bm` chosen the way [`quant::bm_rung`] chooses it. Two columns, because
//!    the tile changes regime across this ladder and one column hides it:
//!    **TFLOP/s**, which is the right denominator once `m` is tall enough to
//!    reuse a staged tile, and **effective bank GB/s**, which is the right one
//!    while it is not. A fire of `m` rows makes `ceil(m / bm)` passes over the
//!    weight bank, so the bytes are `ceil(m / bm) * bank`.
//! 2. **The column tile.** `bn` at 16/32/64 over the same ladder — the axis
//!    §12 swept from inside a model, swept here in isolation.
//! 3. **The group size.** gs32/64/128 at one shape. This is the only axis
//!    §15 left open: llama.cpp's `Q4_K` super-block is 256 elements against
//!    our 64, so it touches a quarter as many scale/bias pairs inside the K
//!    loop, and whether that is worth anything is a question about the
//!    FORMAT that this arm answers in the kernel's own terms.
//! 4. **The width.** b2/b4/b8 at one shape, because the 2-bit prefill rides
//!    the same tiles through the group-parametric stamps and a change to the
//!    loader has to be priced at every width it serves.
//! 5. **The pre-cast pair**, which is the point a dense prefill ACTUALLY
//!    fires — arms 1 to 4 are all the plain bf16 tile, and §12's trace says
//!    79% of a 1024-row fire is `affine_qmm_t_strided_fp16_precast`. The
//!    GEMM alone and then the staging dispatch beside it, at two column
//!    tiles, because the two points do not agree about that axis.
//!
//! Nothing here is asserted. The numbers are the output, and `affine_floor`
//! is where this family's ANSWERS are gated — a loader change that moves a
//! number here and a bit there is a regression whatever this file prints.
//!
//! ```text
//! cargo test -p engine-metal --release --test what_the_affine_tile_is_bound_by -- --nocapture
//! ```
//!
//! # What it said the first time, on the reference M1 Max
//!
//! Medians of three, 24-core M1 Max, `.wiki/macos-bench.md` §21.
//!
//! **THE GB/s COLUMN FALLS WHILE THE TFLOP/s COLUMN RISES**, and that one
//! fact is the answer to the question this file was opened for:
//!
//! ```text
//!   m=8      0.338 ms   1.9 TFLOP/s   68 GB/s eff
//!   m=64     0.816 ms   3.7 TFLOP/s   16 GB/s eff
//!   m=1024  10.611 ms   5.1 TFLOP/s   22 GB/s eff
//! ```
//!
//! A tile reads the bank once per ROW BLOCK, so above the narrow rungs the
//! read is amortized and a bandwidth roofline is simply the wrong
//! denominator — an effective 22 GB/s against this machine's 389.5 GB/s
//! stream is not a kernel seven times off the bus, it is a kernel that has
//! stopped asking the bus for anything. The right denominator is §15's
//! measured GEMM ceiling: 6.29-6.33 TFLOP/s at M=128 and 7.08 at M=512,
//! taken with the `matmul2d` probe in fp16 with no quantization anywhere.
//!
//! **AGAINST THAT, ARM 5 IS THE ANSWER AND ARM 1 IS NOT**, because a dense
//! prefill does not fire the plain bf16 point at all — §12's trace of a
//! 1024-row fire is 79% `affine_qmm_t_strided_fp16_precast`. The pre-cast
//! pair measures 5.74 TFLOP/s at m=128 and 6.36 at m=512, which is **91% and
//! 90% of an unquantized fp16 ceiling while unpacking 4-bit weights**, and
//! the staging dispatch that buys it costs 0.5%. §15 put this at 92% by
//! dividing a whole-model number; this is the same finding taken off the
//! kernel directly, and the two agree.
//!
//! **THE FORMAT AXIS IS CLOSED, WHICH IS WHAT §15 LEFT OPEN.** That section
//! ended on llama.cpp's `Q4_K` super-block being 256 elements against our 64,
//! so it touches a quarter as many scale/bias pairs inside the K loop, and
//! named it "the open question" behind llama.cpp's 5% on a dense prefill.
//! Arm 3 asks it directly and the answer is nothing: 5.06 / 5.05 / 5.05
//! TFLOP/s at gs 32 / 64 / 128, m=1024. Four times the factor traffic and
//! a quarter of it land in the same place.
//!
//! **AND SO IS THE LOADER, FROM THE OTHER SIDE.** Arm 4 changes the code
//! bytes and the unpack ALU together by 4x and moves 2.8%: 5.21 / 5.05 /
//! 5.04 TFLOP/s at 2 / 4 / 8 bits, m=1024. That is the first MEASUREMENT of
//! the quantity `mlx_quantized_block.metal`'s header predicts by arithmetic
//! as `3 / (2 * BM)` = 2.3%, and it brackets it. Nothing that makes the
//! weight read cheaper or the unpack shorter can be worth more than that
//! number, which is why §21 rejected a vectorized affine loader that was
//! bit-identical and measured at 0 to −0.8%.
//!
//! **`bn = 64` WINS ON THE PLAIN POINT AND LOSES ON THE ONE THAT SHIPS**,
//! and reading arm 2 without arm 5 is how a phantom lever gets reported.
//! Arm 2 says 9.28 ms against `bn = 32`'s 10.61 at m=1024, +14%, which looks
//! like `quant::bn_unsplit`'s "NEVER 64" leaving something on the floor. Arm
//! 5 fires the same two column tiles through the pre-cast point and the sign
//! flips: 8.28 ms at `bn = 32` against 8.59 at 64. The rule is right, §12's
//! in-model measurement (104.6 against 107.2 tok/s on Qwen3.6-27B) is right,
//! and the 14% belongs to a point no dense prefill fires.
//!
//! Two caveats to carry off every row above, both learned here. **Measure
//! the point the model actually selects** — the plain tile and the pre-cast
//! tile disagree about an axis by 14% and in opposite directions. And **a
//! number from this file is not a number about a fire**: nothing here has a
//! scan or an attention beside it competing for threadgroups, which §12's
//! ablation shows is the only reason those kernels are free.

#![cfg(target_vendor = "apple")]

use std::time::Instant;

use engine_metal::device::{self, Buffer, Context, Handles, Pipelines};
use engine_metal::encode::Sink;
use kernels_metal::linear::quant;
use kernels_metal::{Arg, Encode, Fire, Grid, Tensor};
use model_ir::Dtype;

const QMM_FILE: &str = "linear/quant_qmm_t.metal";
const QMM_STAMP: &str = "PIE_STAMP_qmm_t";

/// The dense projection shape §15 priced the fp16 ceiling at: Qwen3.6-27B and
/// gemma-4-31b both carry 5120-wide projections, and it is the shape the
/// `matmul2d` probe's 6.29/6.33 TFLOP/s row was taken over.
const K: u32 = 5120;
const N: u32 = 5120;

/// The tallest row block any arm launches, and so the rows the activation and
/// result rectangles are reserved at.
const ROWS: u32 = 1024;

const REPS: usize = 16;

/// Bytes of the weight bank one pass over it reads: codes at `bits`, plus a
/// scale and a bias per group.
fn bank_bytes(group: u32, bits: u32) -> f64 {
    let codes = f64::from(N) * f64::from(K) * f64::from(bits) / 8.0;
    let factors = 2.0 * f64::from(N) * f64::from(K) / f64::from(group) * 2.0;
    codes + factors
}

#[test]
fn affine_tile_bandwidth() {
    if !device::present() {
        println!("SKIP: no Metal device");
        return;
    }
    let device = Context::bind().expect("device");
    let pipelines = Pipelines::new();

    // Reserved at the WIDEST width the arms below reach (8 bits, one byte an
    // element) and the FINEST group (32, the most factors), so one set of
    // rectangles serves every (group, bits) pair.
    let codes = Buffer::zeroed(&device, u64::from(N) * u64::from(K)).expect("codes");
    let scales = Buffer::zeroed(&device, u64::from(N) * u64::from(K) / 32 * 2).expect("scales");
    let biases = Buffer::zeroed(&device, u64::from(N) * u64::from(K) / 32 * 2).expect("biases");
    let act = Buffer::zeroed(&device, u64::from(ROWS) * u64::from(K) * 2).expect("x");
    let half_x = Buffer::zeroed(&device, u64::from(ROWS) * u64::from(K) * 2).expect("half x");
    let y = Buffer::zeroed(&device, u64::from(ROWS) * u64::from(N) * 2).expect("y");

    let handles = Handles::new();
    let bind = |b: &Buffer| handles.bind(b, 0, b.bytes()).expect("bind");
    let t_codes = Tensor::new(bind(&codes), N, K, Dtype::U4g64);
    let t_scales = Tensor::new(bind(&scales), N, K / 32, Dtype::Bf16);
    let t_biases = Tensor::new(bind(&biases), N, K / 32, Dtype::Bf16);
    let t_x = Tensor::new(bind(&act), ROWS, K, Dtype::Bf16);
    let t_half_x = Tensor::new(bind(&half_x), ROWS, K, Dtype::F16);
    let t_y = Tensor::new(bind(&y), ROWS, N, Dtype::Bf16);

    // ms per fire of the plain stamped tile at (group, bits, m, bm, bn).
    let time = |group: i32, bits: i32, m: u32, bm: i32, bn: i32| -> f64 {
        let point = quant::qmm_point("bench.qmm_t", "", QMM_STAMP, group, bits, bm, bn)
            .expect("an axis point");
        let (k, n) = (K as i32, N as i32);
        let grid = quant::qmm_grid("bench.qmm_t", n, bn, m as i32, bm, 1).expect("a grid");
        let burst = |reps: usize| {
            let frame = device.frame().expect("frame");
            {
                let sink = Sink::new(&device, &frame, &pipelines, &handles);
                for _ in 0..reps {
                    sink.fire(
                        Fire::at(QMM_FILE, point.entry)
                            .stamp(point.stamp)
                            .apply(Grid::of(grid, quant::qmm_group(bm))),
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
                    .expect("encode");
                }
            }
            frame.commit().expect("commit");
        };
        burst(4);
        let start = Instant::now();
        burst(REPS);
        start.elapsed().as_secs_f64() * 1000.0 / REPS as f64
    };

    let report = |label: &str, group: i32, bits: i32, m: u32, bm: i32, bn: i32| {
        let ms = time(group, bits, m, bm, bn);
        let flops = 2.0 * f64::from(m) * f64::from(N) * f64::from(K);
        let tflops = flops / (ms / 1000.0) / 1e12;
        let passes = f64::from(m.div_ceil(bm.unsigned_abs()));
        let gbs = passes * bank_bytes(group.unsigned_abs(), bits.unsigned_abs())
            / (ms / 1000.0)
            / 1e9;
        println!(
            "  {label:<16} m={m:<5} bm={bm:<3} bn={bn:<3} {ms:>8.3} ms  \
             {tflops:>6.2} TFLOP/s  {gbs:>7.1} GB/s eff"
        );
    };

    println!("\naffine tile, bank {N}x{K}, plain stamped point");
    println!(
        "  gs64 b4 bank = {:.1} MiB per pass",
        bank_bytes(64, 4) / 1048576.0
    );

    println!("\n  -- 1. the row ladder, bn=32, bm as `bm_rung` picks it");
    for m in [8u32, 16, 32, 64, 128, 256, 512, 1024] {
        let bm = quant::bm_rung(m as i32);
        report("ladder", 64, 4, m, bm, 32);
    }

    println!("\n  -- 2. the column tile");
    for bn in [16i32, 32, 64] {
        for m in [8u32, 128, 1024] {
            report("bn", 64, 4, m, quant::bm_rung(m as i32), bn);
        }
    }

    println!("\n  -- 3. the group size — §15's one open axis");
    for group in [32i32, 64, 128] {
        for m in [8u32, 128, 1024] {
            report("gs", group, 4, m, quant::bm_rung(m as i32), 32);
        }
    }

    println!("\n  -- 4. the width, all three the ladder serves");
    for bits in [2i32, 4, 8] {
        for m in [8u32, 128, 1024] {
            report("bits", 64, bits, m, quant::bm_rung(m as i32), 32);
        }
    }

    // ── 5. THE POINT A DENSE PREFILL ACTUALLY FIRES. Everything above is the
    //    plain bf16 tile, and §12's trace of a 1024-row fire is 79%
    //    `affine_qmm_t_strided_fp16_precast` — the pre-cast pair, which
    //    stages the activation to `half` once and hands the matrix unit the
    //    instruction M1 has rather than the bfloat16 one it emulates. A
    //    ladder measured on the plain point is not the ladder the losing
    //    prefill cells climb, which is the trap this arm exists to avoid.
    //
    //    The GEMM is timed ALONE and then the PAIR, because the staging
    //    dispatch is a real cost the driver pays once per fire and reporting
    //    only the GEMM would flatter the arm.
    let precast = |label: &str, m: u32, bm: i32, bn: i32, with_stage: bool| {
        let entry = quant::precast_point("bench.precast", "", bm, bn).expect("a precast point");
        let (k, n) = (K as i32, N as i32);
        let grid = quant::qmm_grid("bench.precast", n, bn, m as i32, bm, 1).expect("a grid");
        let burst = |reps: usize| {
            let frame = device.frame().expect("frame");
            {
                let sink = Sink::new(&device, &frame, &pipelines, &handles);
                for _ in 0..reps {
                    if with_stage {
                        let count = m as i32 * k;
                        let mut cast = vec![sink.absent().expect("absent"); 3];
                        cast.push(t_x.arg());
                        for _ in 4..12 {
                            cast.push(sink.absent().expect("absent"));
                        }
                        cast.push(t_half_x.arg_mut());
                        cast.push(count.arg());
                        sink.fire(
                            Fire::at(QMM_FILE, quant::PRECAST_STAGE).apply(
                                quant::precast_stage("bench.precast", m as i32, k)
                                    .expect("a stage"),
                            ),
                            &cast,
                        )
                        .expect("encode");
                    }
                    let mut gemm = vec![
                        t_codes.arg(),
                        t_scales.arg(),
                        t_biases.arg(),
                        sink.absent().expect("absent"),
                        t_y.arg_mut(),
                        k.arg(),
                        n.arg(),
                    ];
                    for _ in 7..12 {
                        gemm.push(sink.absent().expect("absent"));
                    }
                    gemm.push(t_half_x.arg());
                    sink.fire(
                        Fire::at(QMM_FILE, entry)
                            .apply(Grid::of(grid, quant::qmm_group(bm))),
                        &gemm,
                    )
                    .expect("encode");
                }
            }
            frame.commit().expect("commit");
        };
        burst(4);
        let start = Instant::now();
        burst(REPS);
        let ms = start.elapsed().as_secs_f64() * 1000.0 / REPS as f64;
        let tflops = 2.0 * f64::from(m) * f64::from(N) * f64::from(K) / (ms / 1000.0) / 1e12;
        println!(
            "  {label:<16} m={m:<5} bm={bm:<3} bn={bn:<3} {ms:>8.3} ms  {tflops:>6.2} TFLOP/s"
        );
    };

    println!("\n  -- 5. the PRE-CAST pair — the point a dense prefill fires");
    for bn in [32i32, 64] {
        for m in [8u32, 128, 512, 1024] {
            precast("precast gemm", m, quant::bm_rung(m as i32), bn, false);
        }
        for m in [128u32, 1024] {
            precast("precast +stage", m, quant::bm_rung(m as i32), bn, true);
        }
    }
}
