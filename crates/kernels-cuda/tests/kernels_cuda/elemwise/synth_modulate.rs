#![cfg(feature = "cuda")]

//! A DiT block's norm and its modulation by the lane's scale and shift
//! (after the gated residual), the chains CUDA forms where the hand rules
//! `fuse.norm_modulate` and `fuse.gated_residual_norm_modulate` once did:
//! one generated row kernel each, against the ops they replace, what each
//! lands.
//!
//! Each region is the one poem-compiler's `fuse::chain` lowers the ops to:
//! the stream `y + g ⊙ x` in one rounding (the gated add's `fmaf`), the
//! norm over it (layernorm, no affine), its row rounded to bf16 as the
//! norm's plane stores it, then `n (1 + scale) + shift` in one rounding,
//! the f32 modulation picked by the row's lane, its scale the first half of
//! the lane's row and its shift the second. It is held to that math by the
//! interpreter (`super::within`), from a decode's rows (every parameter
//! loaded up front) to a prefill's (the scale and shift issued after the
//! norm's sums); the ops are shown beside it, not held to it.

use crate::common::attn_ref::ten;
use crate::common::{Gpu, Lcg};
use dtype::Dtype::{Bf16, F32, I32};
use kernels_cuda::elemwise::{modulate, norm};
use kernels_cuda::jit::Ctx;
use kernels_cuda::synth::{self, In, Out, Plane, Region, Synthesized};
use kernels_cuda::tensor::Tensor;
use kernels_ir::interp::Arg;
use kernels_ir::{Binary, Literal, Op, Reduce, Shape, Unary};

use super::within::{EXACT, NORM_SCALED, apart, assert_within, deterministic, reference, row_of};

const EPS: f32 = 1e-6;
/// The lanes a fire's rows are picked among.
const LANES: usize = 4;
/// The catalog's widths: LTX 2.5's, Wan 2.2's and Flux 2's, LTX's audio.
const WIDTHS: [usize; 3] = [2048, 3072, 4096];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Chain {
    /// `layernorm` then `modulate` (Flux 2's, Wan 2.2's).
    NormModulate,
    /// the gated `residual_add`, then the two.
    Gated,
    /// a scale-free `rmsnorm` then `modulate` (LTX 2.5's, Z-Image's).
    RmsModulate,
    /// the gated `residual_add`, then those.
    GatedRms,
}

impl Chain {
    const BOTH: [Chain; 4] = [
        Chain::NormModulate,
        Chain::Gated,
        Chain::RmsModulate,
        Chain::GatedRms,
    ];

    const fn gated(self) -> bool {
        matches!(self, Chain::Gated | Chain::GatedRms)
    }

    const fn rms(self) -> bool {
        matches!(self, Chain::RmsModulate | Chain::GatedRms)
    }

    /// Parameters: x, scale, shift, the lanes; gated, the stream y and the
    /// gate before them (y, g, x, lanes, scale, shift).
    fn region(self, width: usize) -> Region {
        let w = width as f32;
        let mut ops = Vec::new();
        let (params, x, scale, shift, results) = if self.gated() {
            ops.push(Op::Fma(1, 2, 0));
            (6, 6, 4, 5, vec![6])
        } else {
            (4u32, 0u32, 1u32, 2u32, Vec::new())
        };
        let id = |ops: &Vec<Op>| params + ops.len() as u32 - 1;
        // the row less its mean (layernorm), or as it is (rmsnorm)
        let d = if self.rms() {
            x
        } else {
            ops.push(Op::Reduce(Reduce::Sum, x));
            let sum = id(&ops);
            ops.push(Op::Const(Literal::F32(w)));
            ops.push(Op::Binary(Binary::Div, sum, sum + 1));
            let mean = id(&ops);
            ops.push(Op::Binary(Binary::Sub, x, mean));
            id(&ops)
        };
        ops.push(Op::Binary(Binary::Mul, d, d));
        ops.push(Op::Reduce(Reduce::Sum, id(&ops)));
        let ss = id(&ops);
        ops.push(Op::Const(Literal::F32(w)));
        ops.push(Op::Binary(Binary::Div, ss, ss + 1));
        ops.push(Op::Const(Literal::F32(EPS)));
        ops.push(Op::Binary(Binary::Add, ss + 2, ss + 3));
        ops.push(Op::Unary(Unary::Rsqrt, ss + 4));
        ops.push(Op::Binary(Binary::Mul, d, ss + 5));
        let n = id(&ops);
        ops.push(Op::Const(Literal::F32(1.0)));
        ops.push(Op::Binary(Binary::Add, n + 1, scale));
        ops.push(Op::Fma(n, n + 2, shift));
        let out = id(&ops);
        let mut rounds = vec![(n, Bf16), (out, Bf16)];
        if self.gated() {
            rounds.push((6, Bf16));
        }
        let mut results = results;
        results.push(out);
        let mut shapes = vec![Shape::Row; params as usize];
        shapes[3] = Shape::Scalar;
        Region {
            width: width as u32,
            params: shapes,
            ops,
            results,
            rounds,
        }
    }

    fn ins(self) -> Vec<In> {
        match self.gated() {
            false => {
                vec![
                    In::Rows(Bf16),
                    In::Gathered(F32),
                    In::Gathered(F32),
                    In::Each(I32),
                ]
            }
            true => vec![
                In::Rows(Bf16),
                In::Gathered(F32),
                In::Rows(Bf16),
                In::Each(I32),
                In::Gathered(F32),
                In::Gathered(F32),
            ],
        }
    }

    fn outs(self) -> usize {
        match self {
            Chain::NormModulate | Chain::RmsModulate => 1,
            Chain::Gated | Chain::GatedRms => 2,
        }
    }
}

fn generated(chain: Chain, width: usize) -> Synthesized {
    let region = chain.region(width);
    let ins = chain.ins();
    let outs = vec![Out::Rows(Bf16); chain.outs()];
    synth::supports(&region, &ins, &outs).unwrap_or_else(|why| panic!("{chain:?}: {why}"));
    let kernel = synth::synth(&region, &ins, chain.outs()).expect("synthesized");
    kernel
        .load()
        .expect("the generated kernel compiles and loads");
    kernel
}

/// One fire's planes: the stream (and its sum), the branch, the normed
/// row, the result; the gate and modulation planes, the lanes.
#[derive(Clone, Copy)]
struct Planes {
    y: u64,
    x: u64,
    sum: u64,
    normed: u64,
    out: u64,
}

#[derive(Clone, Copy)]
struct Fixed {
    g: u64,
    m: u64,
    lanes: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum By {
    Ops,
    Region,
}

#[allow(clippy::too_many_arguments)]
fn fire(
    ctx: &Ctx,
    chain: Chain,
    by: By,
    kernel: Option<Synthesized>,
    rows: usize,
    width: usize,
    p: Planes,
    f: Fixed,
) {
    let t = |at| ten(at, rows, width, Bf16);
    let lanes = ten(f.lanes, rows, 1, I32);
    let g = ten(f.g, LANES, width, F32);
    let m = ten(f.m, LANES, 2 * width, F32);
    let normed = |x: Tensor, y: &mut Tensor| {
        if chain.rms() {
            norm::rmsnorm_no_scale(ctx, x, 0, EPS, y)
        } else {
            norm::layernorm_no_scale(ctx, x, EPS, y)
        }
        .expect("norm");
    };
    match (by, chain.gated()) {
        (By::Region, _) => {
            let scale = Plane::cols(&m, 0).by(&lanes);
            let shift = Plane::cols(&m, width as u32).by(&lanes);
            let ins = match chain.gated() {
                false => {
                    vec![Plane::rows(&t(p.x)), scale, shift, Plane::each(&lanes)]
                }
                true => vec![
                    Plane::rows(&t(p.y)),
                    Plane::cols(&g, 0).by(&lanes),
                    Plane::rows(&t(p.x)),
                    Plane::each(&lanes),
                    scale,
                    shift,
                ],
            };
            let outs = match chain.gated() {
                false => vec![Plane::rows(&t(p.out))],
                true => vec![Plane::rows(&t(p.sum)), Plane::rows(&t(p.out))],
            };
            kernel
                .expect("a kernel")
                .fire(ctx, &ins, &outs, rows as u32)
                .expect("the generated kernel fires");
        }
        (By::Ops, false) => {
            normed(t(p.x), &mut t(p.normed));
            modulate::scale_shift(ctx, t(p.normed), m, Some(lanes), &mut t(p.out))
                .expect("modulate");
        }
        (By::Ops, true) => {
            modulate::gated_residual_add(ctx, t(p.y), g, t(p.x), Some(lanes), &mut t(p.sum))
                .expect("add");
            normed(t(p.sum), &mut t(p.normed));
            modulate::scale_shift(ctx, t(p.normed), m, Some(lanes), &mut t(p.out))
                .expect("modulate");
        }
    }
}

/// One seeded draw: the stream, the branch, the gate and modulation
/// planes, each row's lane.
struct Draw {
    y: Vec<u16>,
    x: Vec<u16>,
    g: Vec<f32>,
    m: Vec<f32>,
    lanes: Vec<i32>,
}

fn draw(rows: usize, width: usize) -> Draw {
    let mut lcg = Lcg::seeded(0x30d ^ ((width as u64) << 12) ^ rows as u64);
    Draw {
        y: lcg.row(rows * width).0,
        x: lcg.row(rows * width).0,
        g: (0..LANES * width).map(|_| 0.5 * lcg.unit()).collect(),
        m: (0..LANES * 2 * width).map(|_| 0.25 * lcg.unit()).collect(),
        lanes: (0..rows).map(|r| ((r * 5 + 1) % LANES) as i32).collect(),
    }
}

fn drawn(gpu: &mut Gpu, rows: usize, width: usize) -> (Planes, Fixed) {
    let d = draw(rows, width);
    let plane = rows * width * 2;
    let p = Planes {
        y: gpu.up(&d.y),
        x: gpu.up(&d.x),
        sum: gpu.zeros(plane),
        normed: gpu.zeros(plane),
        out: gpu.zeros(plane),
    };
    let f = Fixed {
        g: gpu.up(&d.g),
        m: gpu.up(&d.m),
        lanes: gpu.up(&d.lanes),
    };
    (p, f)
}

/// The chain's region by the interpreter over the draw.
fn math(chain: Chain, rows: usize, width: usize) -> Vec<Vec<f32>> {
    let d = draw(rows, width);
    let lane = |r: usize| d.lanes[r] as usize;
    let scale = |r: usize| Arg::Row(d.m[lane(r) * 2 * width..lane(r) * 2 * width + width].to_vec());
    let shift =
        |r: usize| Arg::Row(d.m[lane(r) * 2 * width + width..(lane(r) + 1) * 2 * width].to_vec());
    reference(&chain.region(width), rows, |r| match chain.gated() {
        false => vec![
            row_of(&d.x, r, width),
            scale(r),
            shift(r),
            Arg::Scalar(d.lanes[r] as f32),
        ],
        true => vec![
            row_of(&d.y, r, width),
            Arg::Row(d.g[lane(r) * width..(lane(r) + 1) * width].to_vec()),
            row_of(&d.x, r, width),
            Arg::Scalar(d.lanes[r] as f32),
            scale(r),
            shift(r),
        ],
    })
}

/// Each chain at a decode's, a ragged and a prefill's rows: the region
/// lands its math, the stream the reference's bits (one `fmaf`) and the
/// result within [`NORM_SCALED`] (a norm's ulp, then the modulation's
/// rounding), and two fires the same bits.
#[test]
fn a_modulated_norms_region_lands_its_math() {
    for chain in Chain::BOTH {
        for width in WIDTHS {
            let kernel = generated(chain, width);
            for rows in [1usize, 7, 64, 600] {
                let n = rows * width;
                let land = |by: By| {
                    let mut gpu = Gpu::open();
                    let (p, f) = drawn(&mut gpu, rows, width);
                    fire(&gpu.ctx(), chain, by, Some(kernel), rows, width, p, f);
                    gpu.sync();
                    (gpu.down::<u16>(p.sum, n), gpu.down::<u16>(p.out, n))
                };
                let region = land(By::Region);
                let case = format!("{chain:?}, {rows} rows of {width}");
                deterministic(&case, &region.1, &land(By::Region).1);
                let want = math(chain, rows, width);
                if chain.gated() {
                    assert_within(&format!("{case}: the stream"), &region.0, &want[0], EXACT);
                }
                assert_within(
                    &format!("{case}: the result"),
                    &region.1,
                    want.last().expect("a result"),
                    NORM_SCALED,
                );
                let (apart_n, most) = apart(&land(By::Ops).1, want.last().expect("a result"));
                eprintln!("{case}: the ops {apart_n} of {n} apart from the math, {most} ulps at most");
            }
        }
    }
}
