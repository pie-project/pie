#![cfg(feature = "cuda")]

//! A region over small vectors one a row and a repeat over them, as the row
//! skeleton prints it: a row's vector read from its own f32 row, one every
//! row reads, a repeat renormalizing their product a fixed number of passes,
//! that vector landed as a result of its own (an f32 per value, a row) and
//! one of its values scaling the row. Held to the kernels-ir interpreter
//! (`super::within`): the repeat's scalar work in the thread's own f32, the
//! row's product then one rounding.

use crate::common::attn_ref::ten;
use crate::common::{Gpu, Lcg};
use dtype::Dtype::{Bf16, F32};
use kernels_cuda::synth::{self, In, Out, Plane, Region, Schedule, Synthesized};
use kernels_ir::interp::Arg;
use kernels_ir::{Binary, Literal, Op, Shape, Unary};

use super::within::{NORM, assert_within, deterministic, reference, row_of};

const N: u32 = 4;
const PASSES: u32 = 5;

/// Parameters: the row, its vector, the vector every row reads, the row's
/// weight.
fn region(width: u32) -> Region {
    Region {
        width,
        params: vec![Shape::Row, Shape::Vector(N), Shape::Vector(N), Shape::Row],
        ops: vec![
            Op::Binary(Binary::Mul, 1, 2),   // 4: m · w
            Op::Const(Literal::F32(1e-3)),   // 5
            Op::Carry(4),                    // 6
            Op::Pick(6, 0),                  // 7
            Op::Pick(6, 1),                  // 8
            Op::Pick(6, 2),                  // 9
            Op::Pick(6, 3),                  // 10
            Op::Unary(Unary::Abs, 7),        // 11
            Op::Unary(Unary::Abs, 8),        // 12
            Op::Unary(Unary::Abs, 9),        // 13
            Op::Unary(Unary::Abs, 10),       // 14
            Op::Binary(Binary::Add, 11, 12), // 15
            Op::Binary(Binary::Add, 15, 13), // 16
            Op::Binary(Binary::Add, 16, 14), // 17
            Op::Binary(Binary::Add, 17, 5),  // 18
            Op::Binary(Binary::Div, 6, 18),  // 19
            Op::Unary(Unary::Tanh, 8),       // 20
            Op::Insert(19, 1, 20),           // 21
            Op::Repeat(PASSES, 6, 21),       // 22
            Op::Pick(22, 1),                 // 23
            Op::Binary(Binary::Mul, 0, 23),  // 24
            Op::Binary(Binary::Mul, 24, 3),  // 25
        ],
        results: vec![25, 22],
        rounds: vec![(25, Bf16)],
    }
}

const INS: [In; 4] = [
    In::Rows(Bf16),
    In::Rows(F32),
    In::Shared(F32),
    In::Shared(Bf16),
];

fn generated(width: u32) -> Synthesized {
    let outs = [Out::Rows(Bf16), Out::Rows(F32)];
    synth::supports(&region(width), &INS, &outs).expect("the skeleton takes the region");
    let kernel = synth::synth(&region(width), &INS, 2).expect("synthesized");
    kernel
        .load()
        .expect("the generated kernel compiles and loads");
    kernel
}

struct Draw {
    x: Vec<u16>,
    m: Vec<f32>,
    w: Vec<f32>,
    weight: Vec<u16>,
}

fn draw(rows: usize, width: usize) -> Draw {
    let mut lcg = Lcg::seeded(0x7ec ^ ((width as u64) << 8) ^ rows as u64);
    Draw {
        x: lcg.row(rows * width).0,
        m: (0..rows * N as usize).map(|_| lcg.unit()).collect(),
        w: (0..N as usize).map(|_| 0.5 + lcg.unit()).collect(),
        weight: lcg.row(width).0,
    }
}

/// What the kernel lands over the draw: the row, then the vectors.
fn land(
    kernel: Synthesized,
    rows: usize,
    width: usize,
    schedule: Option<Schedule>,
) -> (Vec<u16>, Vec<f32>) {
    let d = draw(rows, width);
    let mut gpu = Gpu::open();
    let (x, m, w, weight) = (gpu.up(&d.x), gpu.up(&d.m), gpu.up(&d.w), gpu.up(&d.weight));
    let (y, r) = (
        gpu.zeros(rows * width * 2),
        gpu.zeros(rows * N as usize * 4),
    );
    let ins = [
        Plane::rows(&ten(x, rows, width, Bf16)),
        Plane::rows(&ten(m, rows, N as usize, F32)),
        Plane::shared(&ten(w, 1, N as usize, F32)),
        Plane::shared(&ten(weight, 1, width, Bf16)),
    ];
    let outs = [
        Plane::rows(&ten(y, rows, width, Bf16)),
        Plane::rows(&ten(r, rows, N as usize, F32)),
    ];
    let ctx = gpu.ctx();
    match schedule {
        None => kernel.fire(&ctx, &ins, &outs, rows as u32),
        Some(s) => kernel.fire_laid(&ctx, &ins, &outs, (rows as u32, 1), s),
    }
    .expect("the generated kernel fires");
    gpu.sync();
    (gpu.down(y, rows * width), gpu.down(r, rows * N as usize))
}

#[test]
fn a_repeat_over_a_rows_small_vectors_lands_its_math() {
    for width in [512usize, 4096] {
        let kernel = generated(width as u32);
        for rows in [1usize, 7, 300] {
            let d = draw(rows, width);
            let n = N as usize;
            let args = |r: usize| {
                vec![
                    row_of(&d.x, r, width),
                    Arg::Row(d.m[r * n..(r + 1) * n].to_vec()),
                    Arg::Row(d.w.clone()),
                    row_of(&d.weight, 0, width),
                ]
            };
            let want = reference(&region(width as u32), rows, args);
            // the vector as the interpreter computes it, in f32 (the
            // reference's rows are bf16 planes)
            let vectors: Vec<f32> = (0..rows)
                .flat_map(|r| {
                    kernels_ir::interp::eval(&region(width as u32), &args(r))
                        .expect("the interpreter runs the region")
                        .remove(1)
                })
                .collect();
            let case = format!("{rows} rows of {width}");
            let (y, r) = land(kernel, rows, width, None);
            assert_within(&format!("{case}: the row"), &y, &want[0], NORM);
            // the vector in f32, within a few of its ulps of the reference's
            for (got, want) in r.iter().zip(&vectors) {
                assert!(
                    (got - want).abs() <= 4e-6 * want.abs().max(1.0),
                    "{case}: a vector's value {got} against {want}"
                );
            }
            deterministic(&case, &y, &land(kernel, rows, width, None).0);
            for schedule in Schedule::ALL {
                let (laid, _) = land(kernel, rows, width, Some(schedule));
                assert_within(&format!("{case}, {schedule:?}"), &laid, &want[0], NORM);
            }
        }
    }
}
