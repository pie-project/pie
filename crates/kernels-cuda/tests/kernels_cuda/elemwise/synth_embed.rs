#![cfg(feature = "cuda")]

//! `fused.embed_scale_add`'s chain, Gemma 4's per-layer input (a table's
//! row by token id, scaled, added to the layer's projected row and scaled
//! again), as the fold forms it once the rule is off: one generated row
//! kernel, against its math by the interpreter (`super::within`) and the
//! hand fused kernel, what each lands.
//!
//! The region is the one poem-compiler's `fuse::chain` lowers the four ops
//! to: the table read where each row's id picks it (an id past the table
//! picks row 0, as the gather does), each scale as the bf16 the scaling
//! kernel holds it, each value rounded to bf16 where an op defines it, and
//! only what is read past the chain (the scaled sum) written. The hand
//! kernel writes the gathered row, its scaled row and the sum too, which
//! the trace reads no further.

use crate::common::attn_ref::ten;
use crate::common::{Gpu, Lcg};
use dtype::Dtype::{Bf16, I32};
use kernels_cuda::gather;
use kernels_cuda::jit::Ctx;
use kernels_cuda::synth::{self, In, Out, Plane, Region, Synthesized};
use kernels_ir::interp::Arg;
use kernels_ir::skeleton::round_bf16;
use kernels_ir::{Binary, Literal, Op, Shape};

use super::within::{EXACT, assert_within, reference, row_of};

/// Gemma 4's per-layer row: 256 wide, its table's scale `sqrt(256)` and the
/// sum's `1 / sqrt(2)`.
const WIDTH: usize = 256;
const EMBED_SCALE: f32 = 16.0;
const OUT_SCALE: f32 = core::f32::consts::FRAC_1_SQRT_2;
const VOCAB: usize = 1 << 16;

/// Parameters: the table (by id), the ids, the layer's row.
fn region() -> Region {
    Region {
        width: WIDTH as u32,
        params: vec![Shape::Row, Shape::Scalar, Shape::Row],
        ops: vec![
            Op::Const(Literal::F32(round_bf16(EMBED_SCALE))),
            Op::Binary(Binary::Mul, 0, 3),
            Op::Binary(Binary::Add, 2, 4),
            Op::Const(Literal::F32(round_bf16(OUT_SCALE))),
            Op::Binary(Binary::Mul, 5, 6),
        ],
        results: vec![7],
        rounds: vec![(4, Bf16), (5, Bf16), (7, Bf16)],
    }
}

fn generated() -> Synthesized {
    let ins = [In::Gathered(Bf16), In::Each(I32), In::Rows(Bf16)];
    synth::supports(&region(), &ins, &[Out::Rows(Bf16)]).expect("the skeleton takes the region");
    let kernel = synth::synth(&region(), &ins, 1).expect("synthesized");
    kernel
        .load()
        .expect("the generated kernel compiles and loads");
    kernel
}

/// One fire's planes: the ids, the layer's row, and what lands.
#[derive(Clone, Copy)]
struct Planes {
    ids: u64,
    y: u64,
    e: u64,
    e_scaled: u64,
    y_scaled: u64,
}

impl Planes {
    fn zeros(gpu: &mut Gpu, rows: usize) -> Planes {
        let plane = rows * WIDTH * 2;
        Planes {
            ids: gpu.zeros(rows * 4),
            y: gpu.zeros(plane),
            e: gpu.zeros(plane),
            e_scaled: gpu.zeros(plane),
            y_scaled: gpu.zeros(plane),
        }
    }
}

fn fire_hand(ctx: &Ctx, table: u64, rows: usize, p: Planes) {
    let t = |at| ten(at, rows, WIDTH, Bf16);
    gather::embed_scale_add(
        ctx,
        ten(p.ids, rows, 1, I32),
        ten(table, VOCAB, WIDTH, Bf16),
        VOCAB as u32,
        &mut t(p.e),
        EMBED_SCALE,
        &mut t(p.e_scaled),
        &mut t(p.y),
        OUT_SCALE,
        &mut t(p.y_scaled),
    )
    .expect("the hand kernel fires");
}

fn fire_region(ctx: &Ctx, kernel: Synthesized, table: u64, rows: usize, p: Planes) {
    let ids = ten(p.ids, rows, 1, I32);
    kernel
        .fire(
            ctx,
            &[
                Plane::gathered(&ten(table, VOCAB, WIDTH, Bf16), &ids),
                Plane::each(&ids),
                Plane::rows(&ten(p.y, rows, WIDTH, Bf16)),
            ],
            &[Plane::rows(&ten(p.y_scaled, rows, WIDTH, Bf16))],
            rows as u32,
        )
        .expect("the generated kernel fires");
}

/// The region lands its math's bits (IEEE ops and declared roundings), ids
/// past the table (and below it) picking row 0 as the gather picks it; the
/// hand kernel lands the same.
#[test]
fn a_per_layer_inputs_region_lands_its_math() {
    let kernel = generated();
    for rows in [1usize, 7, 64, 300] {
        let mut lcg = Lcg::seeded(0xe3b ^ rows as u64);
        let table_rows = lcg.row(VOCAB * WIDTH).0;
        let y = lcg.row(rows * WIDTH).0;
        let ids: Vec<i32> = (0..rows)
            .map(|r| match r % 9 {
                3 => VOCAB as i32 + 5,
                5 => -2,
                _ => ((r * 7919 + 13) % VOCAB) as i32,
            })
            .collect();
        let mut gpu = Gpu::open();
        let table = gpu.up(&table_rows);
        let drawn = |gpu: &mut Gpu| Planes {
            ids: gpu.up(&ids),
            y: gpu.up(&y),
            ..Planes::zeros(gpu, rows)
        };
        let (hand, ours) = (drawn(&mut gpu), drawn(&mut gpu));
        fire_hand(&gpu.ctx(), table, rows, hand);
        fire_region(&gpu.ctx(), kernel, table, rows, ours);
        gpu.sync();
        let n = rows * WIDTH;
        let ours = gpu.down::<u16>(ours.y_scaled, n);
        let picked = |r: usize| -> usize {
            usize::try_from(ids[r])
                .ok()
                .filter(|&id| id < VOCAB)
                .unwrap_or(0)
        };
        let want = reference(&region(), rows, |r| {
            vec![
                row_of(&table_rows, picked(r), WIDTH),
                Arg::Scalar(ids[r] as f32),
                row_of(&y, r, WIDTH),
            ]
        });
        let case = format!("{rows} rows");
        assert_within(&case, &ours, &want[0], EXACT);
        assert_within(
            &format!("{case}, the hand kernel"),
            &gpu.down::<u16>(hand.y_scaled, n),
            &want[0],
            EXACT,
        );
    }
}
