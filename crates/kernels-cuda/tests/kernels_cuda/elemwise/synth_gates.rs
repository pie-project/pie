#![cfg(feature = "cuda")]

//! A hyper-connection's gates and the norm of the row they pre-mix
//! (DeepSeek-V4's sublayer input), the chain CUDA forms where the hand rule
//! `fuse.hc_gates_norm` once did: one generated row kernel, against its
//! math by the interpreter (`super::within`) and the hand kernels.
//!
//! The region is the one poem-compiler's `fuse::chain` lowers the two ops
//! to: each gate a sigmoid of its mix scaled and biased (small vectors one a
//! row, and two every row reads), the pre gates plus eps; the combine a
//! softmax of each row of its logits plus eps, its columns normalized, then
//! `sinkhorn - 1` passes normalizing its rows and its columns (a repeat over
//! the small vector); the post and combine mixes landed as small vectors of
//! their own; the pre-mixed row the pre-weighted sum of the four streams
//! (parts of the row of streams), rounded to bf16 as its plane stores it,
//! then its norm.

use crate::common::attn_ref::ten;
use crate::common::{Gpu, Lcg};
use dtype::Dtype::{Bf16, F32};
use kernels_cuda::elemwise::{hc, norm};
use kernels_cuda::jit::Ctx;
use kernels_cuda::synth::{self, In, Out, Plane, Region, Schedule, Synthesized};
use kernels_ir::build::Ops;
use kernels_ir::interp::Arg;
use kernels_ir::{Binary, Op, Shape, Unary};

use super::within::{NORM, apart, assert_within, deterministic, reference, row_of};

const M: u32 = 4;
const MIXES: u32 = 2 * M + M * M;
const SINKHORN: u32 = 20;
const GATE_EPS: f32 = 1e-6;
const ALPHA: f32 = 2.0;
const EPS: f32 = 1e-6;
/// DeepSeek-V4 Flash's hidden width, and a narrower one.
const WIDTHS: [usize; 2] = [4096, 1024];

fn sigmoid(b: &mut Ops, t: u32) -> u32 {
    let one = b.f32(1.0);
    let neg = b.push(Op::Unary(Unary::Neg, t));
    let exp = b.push(Op::Unary(Unary::Exp, neg));
    let den = b.push(Op::Binary(Binary::Add, one, exp));
    b.push(Op::Unary(Unary::Recip, den))
}

fn packed(b: &mut Ops, values: &[u32]) -> u32 {
    let mut v = b.push(Op::Splat(values[0], values.len() as u32));
    for (i, &x) in (1u32..).zip(&values[1..]) {
        v = b.push(Op::Insert(v, i, x));
    }
    v
}

fn normalized(b: &mut Ops, c: &mut [u32], eps: u32, rows: bool) {
    let at = |line: u32, k: u32| (if rows { line * M + k } else { k * M + line }) as usize;
    for line in 0..M {
        let mut sum = c[at(line, 0)];
        for k in 1..M {
            sum = b.push(Op::Binary(Binary::Add, sum, c[at(line, k)]));
        }
        let sum = b.push(Op::Binary(Binary::Add, sum, eps));
        for k in 0..M {
            c[at(line, k)] = b.push(Op::Binary(Binary::Div, c[at(line, k)], sum));
        }
    }
}

/// Parameters: the mixes, the four streams, the scale, the bias, the norm's
/// weight. Results: the post mix, the combine, the normed row.
fn region(width: u32) -> Region {
    let (mix, sc, bs, w) = (0, M + 1, M + 2, M + 3);
    let mut b = Ops::new(M + 4);
    let eps = b.f32(GATE_EPS);
    let logit = |b: &mut Ops, i: u32, s: u32| {
        let x = b.push(Op::Pick(mix, i));
        let k = b.push(Op::Pick(sc, s));
        let scaled = b.push(Op::Binary(Binary::Mul, x, k));
        let bias = b.push(Op::Pick(bs, i));
        b.push(Op::Binary(Binary::Add, scaled, bias))
    };
    let pre: Vec<u32> = (0..M)
        .map(|i| {
            let l = logit(&mut b, i, 0);
            let g = sigmoid(&mut b, l);
            b.push(Op::Binary(Binary::Add, g, eps))
        })
        .collect();
    let a = b.f32(ALPHA);
    let post: Vec<u32> = (0..M)
        .map(|i| {
            let l = logit(&mut b, M + i, 1);
            let g = sigmoid(&mut b, l);
            b.push(Op::Binary(Binary::Mul, g, a))
        })
        .collect();
    let post = packed(&mut b, &post);
    let mut c: Vec<u32> = (0..M * M).map(|t| logit(&mut b, 2 * M + t, 2)).collect();
    for r in 0..M {
        let row: Vec<u32> = (0..M).map(|j| c[(r * M + j) as usize]).collect();
        let most = row[1..]
            .iter()
            .fold(row[0], |acc, &v| b.push(Op::Binary(Binary::Max, acc, v)));
        let e: Vec<u32> = row
            .iter()
            .map(|&v| {
                let shifted = b.push(Op::Binary(Binary::Sub, v, most));
                b.push(Op::Unary(Unary::Exp, shifted))
            })
            .collect();
        let sum = e[1..]
            .iter()
            .fold(e[0], |acc, &v| b.push(Op::Binary(Binary::Add, acc, v)));
        for j in 0..M {
            let q = b.push(Op::Binary(Binary::Div, e[j as usize], sum));
            c[(r * M + j) as usize] = b.push(Op::Binary(Binary::Add, q, eps));
        }
    }
    normalized(&mut b, &mut c, eps, false);
    let start = packed(&mut b, &c);
    let carry = b.push(Op::Carry(start));
    let mut c: Vec<u32> = (0..M * M).map(|t| b.push(Op::Pick(carry, t))).collect();
    normalized(&mut b, &mut c, eps, true);
    normalized(&mut b, &mut c, eps, false);
    let next = packed(&mut b, &c);
    let comb = b.push(Op::Repeat(SINKHORN - 1, carry, next));
    let mut x = None;
    for (i, &g) in (1..=M).zip(&pre) {
        let term = b.push(Op::Binary(Binary::Mul, g, i));
        x = Some(match x {
            None => term,
            Some(acc) => b.push(Op::Binary(Binary::Add, acc, term)),
        });
    }
    let x = x.expect("a stream");
    let n = b.rms_normed(x, width, EPS);
    let y = b.push(Op::Binary(Binary::Mul, n, w));
    let mut params = vec![Shape::Vector(MIXES)];
    params.extend([Shape::Row; M as usize]);
    params.extend([Shape::Vector(3), Shape::Vector(MIXES), Shape::Row]);
    Region {
        width,
        params,
        ops: b.ops,
        results: vec![post, comb, y],
        rounds: vec![(x, Bf16), (y, Bf16)],
    }
}

fn ins() -> Vec<In> {
    let mut ins = vec![In::Rows(F32)];
    ins.extend([In::Rows(Bf16); M as usize]);
    ins.extend([In::Shared(F32), In::Shared(F32), In::Shared(Bf16)]);
    ins
}

fn generated(width: usize) -> Synthesized {
    let outs = [Out::Rows(F32), Out::Rows(F32), Out::Rows(Bf16)];
    synth::supports(&region(width as u32), &ins(), &outs).expect("the skeleton takes the region");
    let kernel = synth::synth(&region(width as u32), &ins(), 3).expect("synthesized");
    kernel
        .load()
        .expect("the generated kernel compiles and loads");
    kernel
}

struct Draw {
    mixes: Vec<f32>,
    streams: Vec<u16>,
    scale: Vec<f32>,
    base: Vec<f32>,
    weight: Vec<u16>,
}

fn draw(rows: usize, width: usize) -> Draw {
    let mut lcg = Lcg::seeded(0x9a7e ^ ((width as u64) << 8) ^ rows as u64);
    Draw {
        mixes: (0..rows * MIXES as usize)
            .map(|_| 2.0 * lcg.unit())
            .collect(),
        streams: lcg.row(rows * M as usize * width).0,
        scale: vec![0.75, 1.25, 0.5],
        base: (0..MIXES).map(|_| 0.5 * lcg.unit()).collect(),
        weight: lcg.row(width).0,
    }
}

/// One fire's planes.
#[derive(Clone, Copy)]
struct Planes {
    mixes: u64,
    streams: u64,
    scale: u64,
    base: u64,
    weight: u64,
    post: u64,
    comb: u64,
    x: u64,
    y: u64,
}

fn drawn(gpu: &mut Gpu, rows: usize, width: usize) -> Planes {
    let d = draw(rows, width);
    Planes {
        mixes: gpu.up(&d.mixes),
        streams: gpu.up(&d.streams),
        scale: gpu.up(&d.scale),
        base: gpu.up(&d.base),
        weight: gpu.up(&d.weight),
        post: gpu.zeros(rows * M as usize * 4),
        comb: gpu.zeros(rows * (M * M) as usize * 4),
        x: gpu.zeros(rows * width * 2),
        y: gpu.zeros(rows * width * 2),
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum By {
    /// The gates and the norm, each its hand kernel (the L40S's path).
    Ops,
    /// The hand fused kernel (`kernel.elemwise.hc_tile`'s).
    Hand,
    Region(Option<Schedule>),
}

fn fire(ctx: &Ctx, by: By, kernel: Option<Synthesized>, rows: usize, width: usize, p: Planes) {
    let streams = ten(p.streams, rows, M as usize * width, Bf16);
    let mixes = ten(p.mixes, rows, MIXES as usize, F32);
    let (scale, base) = (ten(p.scale, 1, 3, F32), ten(p.base, 1, MIXES as usize, F32));
    let weight = ten(p.weight, 1, width, Bf16);
    let mut post = ten(p.post, rows, M as usize, F32);
    let mut comb = ten(p.comb, rows, (M * M) as usize, F32);
    let mut y = ten(p.y, rows, width, Bf16);
    match by {
        By::Ops => {
            let mut x = ten(p.x, rows, width, Bf16);
            hc::gates(
                ctx, mixes, streams, scale, base, M, GATE_EPS, ALPHA, SINKHORN, true, &mut x,
                &mut post, &mut comb,
            )
            .expect("the gates fire");
            norm::rmsnorm(ctx, x, weight, EPS, &mut y).expect("the norm fires");
        }
        By::Hand => {
            let gate = hc::Gate {
                scale,
                base,
                gate_eps: GATE_EPS,
                alpha: ALPHA,
                sinkhorn: SINKHORN,
            };
            let landed = hc::gates_norm(
                ctx, mixes, streams, M, gate, weight, EPS, &mut post, &mut comb, &mut y,
            )
            .expect("the hand kernel fires");
            assert!(landed, "the hand kernel takes the shape");
        }
        By::Region(schedule) => {
            let mut ins = vec![Plane::rows(&mixes)];
            ins.extend((0..M).map(|i| Plane::cols(&streams, i * width as u32)));
            ins.extend([
                Plane::shared(&scale),
                Plane::shared(&base),
                Plane::shared(&weight),
            ]);
            let outs = [Plane::rows(&post), Plane::rows(&comb), Plane::rows(&y)];
            let kernel = kernel.expect("a kernel");
            match schedule {
                None => kernel.fire(ctx, &ins, &outs, rows as u32),
                Some(s) => kernel.fire_laid(ctx, &ins, &outs, (rows as u32, 1), s),
            }
            .expect("the generated kernel fires");
        }
    }
}

/// What lands over the draw: the normed row, the post mixes, the combines.
fn land(
    by: By,
    kernel: Option<Synthesized>,
    rows: usize,
    width: usize,
) -> (Vec<u16>, Vec<f32>, Vec<f32>) {
    let mut gpu = Gpu::open();
    let p = drawn(&mut gpu, rows, width);
    let ctx = if by == By::Hand {
        gpu.ctx()
            .with_tuning(crate::common::attn_ref::tuned(|t| t.hc_tile = true))
    } else {
        gpu.ctx()
    };
    fire(&ctx, by, kernel, rows, width, p);
    gpu.sync();
    (
        gpu.down(p.y, rows * width),
        gpu.down(p.post, rows * M as usize),
        gpu.down(p.comb, rows * (M * M) as usize),
    )
}

/// The region by the interpreter over the draw: the normed rows (bf16, as
/// their plane holds them) and the two mixes (f32).
fn math(rows: usize, width: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let d = draw(rows, width);
    let region = region(width as u32);
    let m = MIXES as usize;
    let args = |r: usize| {
        let mut args = vec![Arg::Row(d.mixes[r * m..(r + 1) * m].to_vec())];
        let row = &d.streams[r * M as usize * width..(r + 1) * M as usize * width];
        args.extend((0..M as usize).map(|i| row_of(&row[i * width..(i + 1) * width], 0, width)));
        args.extend([
            Arg::Row(d.scale.clone()),
            Arg::Row(d.base.clone()),
            row_of(&d.weight, 0, width),
        ]);
        args
    };
    let normed = reference(&region, rows, args).remove(2);
    let (mut post, mut comb) = (Vec::new(), Vec::new());
    for r in 0..rows {
        let out = kernels_ir::interp::eval(&region, &args(r)).expect("the interpreter runs");
        post.extend(&out[0]);
        comb.extend(&out[1]);
    }
    (normed, post, comb)
}

fn near(case: &str, what: &str, got: &[f32], want: &[f32]) {
    for (k, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - w).abs() <= 1e-5 * w.abs().max(1.0),
            "{case}: {what} {k}: {g} against {w}"
        );
    }
}

/// The gates' region at a decode's, a ragged and a prefill's rows, under
/// each schedule: the normed row within [`NORM`] of its math, the two mixes
/// within a few f32 ulps (`expf` against the host's `exp`, through the
/// Sinkhorn's passes), two fires the same bits; the hand kernels beside it.
#[test]
fn a_hyper_connections_gates_and_norm_land_their_math() {
    for width in WIDTHS {
        let kernel = generated(width);
        for rows in [1usize, 7, 300] {
            let case = format!("{rows} rows of {width}");
            let (normed, post, comb) = math(rows, width);
            let (y, p, c) = land(By::Region(None), Some(kernel), rows, width);
            assert_within(&case, &y, &normed, NORM);
            near(&case, "post", &p, &post);
            near(&case, "combine", &c, &comb);
            deterministic(
                &case,
                &y,
                &land(By::Region(None), Some(kernel), rows, width).0,
            );
            for schedule in Schedule::ALL {
                let (laid, ..) = land(By::Region(Some(schedule)), Some(kernel), rows, width);
                assert_within(&format!("{case}, {schedule:?}"), &laid, &normed, NORM);
            }
            for by in [By::Ops, By::Hand] {
                let (n, most) = apart(&land(by, None, rows, width).0, &normed);
                eprintln!(
                    "{case}: {by:?} {n} of {} apart from the math, {most} ulps at most",
                    rows * width
                );
            }
        }
    }
}
