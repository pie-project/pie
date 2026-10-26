#![cfg(feature = "cuda")]

//! `fused.res_blend_rmsnorm`'s chain, AttnRes's blend of a row's closed
//! blocks and its running prefix and the RMS norm of the blended row (Kimi
//! K3's sublayer inputs), as the fold forms it: one generated row kernel,
//! against its math by the interpreter (`super::within`) and the blend and
//! norm fired apart.
//!
//! The region is the one poem-compiler's `fuse::chain` lowers the two ops
//! to: each candidate (the blocks, then the prefix) scaled by the norm's
//! weight and the projection and summed, times its RMS norm's scale, into
//! its logit (its square sum and its dot one pass's); their
//! softmax (`expf` of each less the most); the candidates summed under it,
//! the sum rounded to bf16 as the blend's plane stores it; then its norm
//! under the next weight. Up to eight blocks and the prefix, 7168 wide, are
//! more rows than a thread's registers hold, so the skeleton streams them
//! through its passes, each read once to score it and once to blend it.

use crate::common::attn_ref::ten;
use crate::common::{Gpu, Lcg, to_bf16};
use dtype::Dtype::Bf16;
use kernels_cuda::elemwise::{modulate, norm};
use kernels_cuda::jit::Ctx;
use kernels_cuda::synth::{self, In, Out, Plane, Region, Schedule, Synthesized};
use kernels_cuda::tensor::Tensor;
use kernels_ir::build::Ops;
use kernels_ir::{Binary, Op, Reduce, Shape, Unary};

use super::within::{NORM_SCALED, apart, assert_within, deterministic, reference, row_of};

const EPS: f32 = 1e-5;
const NORM_EPS: f32 = 1e-5;

/// Kimi K3's hidden widths: the release's and the fixture's.
const WIDTHS: [usize; 2] = [7168, 2048];

/// The blocks a blend reads: none (the fixture's first), a few, and the
/// release's most.
const BLOCKS: [usize; 4] = [0, 1, 4, 8];

/// Parameters: the prefix, the blocks, the norm's weight, the projection
/// and the next norm's weight, as the chain reads them.
fn region(width: usize, blocks: usize) -> Region {
    let params = (blocks + 4) as u32;
    let (nw, pw, w) = (blocks as u32 + 1, blocks as u32 + 2, blocks as u32 + 3);
    let mut b = Ops::new(params);
    let candidates: Vec<u32> = (1..=blocks as u32).chain([0]).collect();
    let logits: Vec<u32> = candidates
        .iter()
        .map(|&x| {
            let sq = b.push(Op::Binary(Binary::Mul, x, x));
            let ss = b.push(Op::Reduce(Reduce::Sum, sq));
            let n = b.f32(width as f32);
            let ms = b.push(Op::Binary(Binary::Div, ss, n));
            let e = b.f32(EPS);
            let shifted = b.push(Op::Binary(Binary::Add, ms, e));
            let r = b.push(Op::Unary(Unary::Rsqrt, shifted));
            let xw = b.push(Op::Binary(Binary::Mul, x, nw));
            let d = b.push(Op::Binary(Binary::Mul, xw, pw));
            let dot = b.push(Op::Reduce(Reduce::Sum, d));
            b.push(Op::Binary(Binary::Mul, dot, r))
        })
        .collect();
    let most = logits[1..]
        .iter()
        .fold(logits[0], |m, &l| b.push(Op::Binary(Binary::Max, m, l)));
    let exps: Vec<u32> = logits
        .iter()
        .map(|&l| {
            let shifted = b.push(Op::Binary(Binary::Sub, l, most));
            b.push(Op::Unary(Unary::Exp, shifted))
        })
        .collect();
    let sum = exps[1..]
        .iter()
        .fold(exps[0], |a, &e| b.push(Op::Binary(Binary::Add, a, e)));
    let mut blend = None;
    for (&x, &e) in candidates.iter().zip(&exps) {
        let share = b.push(Op::Binary(Binary::Div, e, sum));
        let term = b.push(Op::Binary(Binary::Mul, share, x));
        blend = Some(match blend {
            None => term,
            Some(acc) => b.push(Op::Binary(Binary::Add, acc, term)),
        });
    }
    let h = blend.expect("a candidate");
    let n = b.rms_normed(h, width as u32, NORM_EPS);
    let out = b.push(Op::Binary(Binary::Mul, n, w));
    let shapes = vec![Shape::Row; params as usize];
    Region {
        width: width as u32,
        params: shapes,
        ops: b.ops,
        results: vec![out],
        rounds: vec![(h, Bf16), (out, Bf16)],
    }
}

fn ins(blocks: usize) -> Vec<In> {
    let mut ins = vec![In::Rows(Bf16); blocks + 1];
    ins.extend([In::Shared(Bf16); 3]);
    ins
}

fn generated(width: usize, blocks: usize) -> Synthesized {
    let region = region(width, blocks);
    let ins = ins(blocks);
    synth::supports(&region, &ins, &[Out::Rows(Bf16)])
        .unwrap_or_else(|why| panic!("{blocks} blocks of {width}: {why}"));
    let kernel = synth::synth(&region, &ins, 1).expect("synthesized");
    kernel
        .load()
        .expect("the generated kernel compiles and loads");
    kernel
}

/// One draw's planes: the prefix, the blocks, the weights, and what lands.
struct Planes {
    prefix: u64,
    blocks: Vec<u64>,
    nw: u64,
    pw: u64,
    w: u64,
    out: u64,
    blend: u64,
}

struct Draw {
    prefix: Vec<u16>,
    blocks: Vec<Vec<u16>>,
    nw: Vec<u16>,
    pw: Vec<u16>,
    w: Vec<u16>,
}

fn draw(rows: usize, width: usize, blocks: usize) -> Draw {
    let mut lcg =
        Lcg::seeded(0xb1e ^ ((width as u64) << 8) ^ ((blocks as u64) << 20) ^ rows as u64);
    let near_one = |lcg: &mut Lcg| -> Vec<u16> {
        (0..width)
            .map(|_| to_bf16(1.0 + 0.25 * lcg.unit()))
            .collect()
    };
    Draw {
        prefix: lcg.row(rows * width).0,
        blocks: (0..blocks).map(|_| lcg.row(rows * width).0).collect(),
        nw: near_one(&mut lcg),
        // a projection that spreads the logits a few apart
        pw: (0..width).map(|_| to_bf16(0.2 * lcg.unit())).collect(),
        w: near_one(&mut lcg),
    }
}

fn drawn(gpu: &mut Gpu, rows: usize, width: usize, blocks: usize) -> Planes {
    let d = draw(rows, width, blocks);
    Planes {
        prefix: gpu.up(&d.prefix),
        blocks: d.blocks.iter().map(|b| gpu.up(b)).collect(),
        nw: gpu.up(&d.nw),
        pw: gpu.up(&d.pw),
        w: gpu.up(&d.w),
        out: gpu.zeros(rows * width * 2),
        blend: gpu.zeros(rows * width * 2),
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum By {
    /// The blend and the norm, each its hand kernel.
    Ops,
    Region(Option<Schedule>),
}

fn fire(ctx: &Ctx, by: By, kernel: Option<Synthesized>, rows: usize, width: usize, p: &Planes) {
    let t = |at| ten(at, rows, width, Bf16);
    let weight = |at| ten(at, 1, width, Bf16);
    let blocks: Vec<Tensor> = p.blocks.iter().map(|&b| t(b)).collect();
    match by {
        By::Ops => {
            modulate::res_blend(
                ctx,
                t(p.prefix),
                &blocks,
                weight(p.nw),
                EPS,
                weight(p.pw),
                &mut t(p.blend),
            )
            .expect("the blend fires");
            norm::rmsnorm(ctx, t(p.blend), weight(p.w), NORM_EPS, &mut t(p.out))
                .expect("the norm fires");
        }
        By::Region(schedule) => {
            let mut ins = vec![Plane::rows(&t(p.prefix))];
            ins.extend(blocks.iter().map(Plane::rows));
            ins.extend([p.nw, p.pw, p.w].map(|at| Plane::shared(&weight(at))));
            let kernel = kernel.expect("a kernel");
            let outs = [Plane::rows(&t(p.out))];
            match schedule {
                None => kernel.fire(ctx, &ins, &outs, rows as u32),
                Some(s) => kernel.fire_laid(ctx, &ins, &outs, (rows as u32, 1), s),
            }
            .expect("the generated kernel fires");
        }
    }
}

fn land(by: By, kernel: Option<Synthesized>, rows: usize, width: usize, blocks: usize) -> Vec<u16> {
    let mut gpu = Gpu::open();
    let p = drawn(&mut gpu, rows, width, blocks);
    fire(&gpu.ctx(), by, kernel, rows, width, &p);
    gpu.sync();
    gpu.down(p.out, rows * width)
}

/// The region by the interpreter over the draw.
fn math(rows: usize, width: usize, blocks: usize) -> Vec<f32> {
    let d = draw(rows, width, blocks);
    let shared = |v: &[u16]| row_of(v, 0, width);
    reference(&region(width, blocks), rows, |r| {
        let mut args = vec![row_of(&d.prefix, r, width)];
        args.extend(d.blocks.iter().map(|b| row_of(b, r, width)));
        args.extend([shared(&d.nw), shared(&d.pw), shared(&d.w)]);
        args
    })
    .remove(0)
}

/// Each blend at a decode's, a ragged and a prefill's rows, under each
/// schedule: the region lands its math within [`NORM_SCALED`] (the blend's
/// row rounded, then its norm's), two fires the same bits; the blend and
/// the norm fired apart are shown beside it, not held to it.
#[test]
fn a_blends_region_streams_its_candidates_and_lands_its_math() {
    for width in WIDTHS {
        for blocks in BLOCKS {
            let kernel = generated(width, blocks);
            for rows in [1usize, 7, 300] {
                let case = format!("{blocks} blocks of {width}, {rows} rows");
                let want = math(rows, width, blocks);
                let region = land(By::Region(None), Some(kernel), rows, width, blocks);
                deterministic(
                    &case,
                    &region,
                    &land(By::Region(None), Some(kernel), rows, width, blocks),
                );
                assert_within(&case, &region, &want, NORM_SCALED);
                for schedule in Schedule::ALL {
                    let laid = land(
                        By::Region(Some(schedule)),
                        Some(kernel),
                        rows,
                        width,
                        blocks,
                    );
                    assert_within(&format!("{case}, {schedule:?}"), &laid, &want, NORM_SCALED);
                }
                let (n, most) = apart(&land(By::Ops, None, rows, width, blocks), &want);
                eprintln!(
                    "{case}: the ops {n} of {} apart from the math, {most} ulps at most",
                    rows * width
                );
            }
        }
    }
}
