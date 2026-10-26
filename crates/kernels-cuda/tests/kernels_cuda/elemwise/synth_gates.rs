#![cfg(feature = "cuda")]

//! A hyper-connection's gates and the norm of the row they pre-mix
//! (DeepSeek-V4.1's sublayer input), the chain CUDA forms where the hand
//! rule `fuse.hc_gates_norm` once did: one generated row kernel, against
//! its math by the interpreter (`super::within`) and the gates and the norm
//! apart.
//!
//! The region is the one poem-compiler's `fuse::chain` lowers the two ops
//! to, vector-wide: the logits a small vector one a row (the mixes scaled,
//! the pre, post and combine scales laid beside their mixes, and biased),
//! every gate a sigmoid of its logit, the pre gates plus eps and the post
//! gates times alpha; the combine a softmax of each row of its logits plus
//! eps, its columns normalized, then `sinkhorn - 1` passes normalizing its
//! rows and its columns (a repeat over the small vector), each sum laid
//! beside the values it divides; the post and combine mixes landed as
//! small vectors of their own; the pre-mixed row the pre-weighted sum of
//! the four streams (parts of the row of streams), rounded to bf16 as its
//! plane stores it, then its norm. The skeleton prints the small vectors
//! over a warp's lanes, the mixes on the row's first warp or, at a
//! decode's spread rows, a side block.

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
/// DeepSeek-V4 Flash's hidden width, a narrower one, and one of eight
/// lanes a row (a small vector three slots a thread).
const WIDTHS: [usize; 3] = [4096, 1024, 64];

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

/// Each row's (or column's) fold by `op` of an `M` x `M` vector (row by
/// row), plus `eps` where given, laid beside each of its values.
fn lines(b: &mut Ops, v: u32, op: Binary, eps: Option<u32>, rows: bool) -> u32 {
    let at = |line: u32, k: u32| if rows { line * M + k } else { k * M + line };
    let folds: Vec<u32> = (0..M)
        .map(|line| {
            let mut acc = b.push(Op::Pick(v, at(line, 0)));
            for k in 1..M {
                let x = b.push(Op::Pick(v, at(line, k)));
                acc = b.push(Op::Binary(op, acc, x));
            }
            match eps {
                Some(eps) => b.push(Op::Binary(Binary::Add, acc, eps)),
                None => acc,
            }
        })
        .collect();
    let laid: Vec<u32> = (0..M * M)
        .map(|t| folds[(if rows { t / M } else { t % M }) as usize])
        .collect();
    packed(b, &laid)
}

/// `v` with each row (or column) divided by its sum plus `eps`.
fn normalized(b: &mut Ops, v: u32, eps: u32, rows: bool) -> u32 {
    let sums = lines(b, v, Binary::Add, Some(eps), rows);
    b.push(Op::Binary(Binary::Div, v, sums))
}

/// Parameters: the mixes, the four streams, the scale, the bias, the norm's
/// weight. Results: the post mix, the combine, the normed row.
fn region(width: u32) -> Region {
    let (mix, sc, bs, w) = (0, M + 1, M + 2, M + 3);
    let mut b = Ops::new(M + 4);
    let eps = b.f32(GATE_EPS);
    // each logit its mix scaled (the pre, post and combine scales laid
    // beside their mixes) and biased; every gate a sigmoid of its logit
    let s: Vec<u32> = (0..3).map(|k| b.push(Op::Pick(sc, k))).collect();
    let per: Vec<u32> = (0..MIXES).map(|i| s[(i / M).min(2) as usize]).collect();
    let scales = packed(&mut b, &per);
    let scaled = b.push(Op::Binary(Binary::Mul, mix, scales));
    let logits = b.push(Op::Binary(Binary::Add, scaled, bs));
    let gates = sigmoid(&mut b, logits);
    let pre: Vec<u32> = (0..M)
        .map(|i| {
            let g = b.push(Op::Pick(gates, i));
            b.push(Op::Binary(Binary::Add, g, eps))
        })
        .collect();
    let a = b.f32(ALPHA);
    let post: Vec<u32> = (0..M)
        .map(|i| {
            let g = b.push(Op::Pick(gates, M + i));
            b.push(Op::Binary(Binary::Mul, g, a))
        })
        .collect();
    let post = packed(&mut b, &post);
    // the combine: a softmax of each row of its logits plus eps, its
    // columns normalized, then `SINKHORN - 1` passes of its rows and columns
    let c: Vec<u32> = (0..M * M)
        .map(|t| b.push(Op::Pick(logits, 2 * M + t)))
        .collect();
    let c = packed(&mut b, &c);
    let most = lines(&mut b, c, Binary::Max, None, true);
    let shifted = b.push(Op::Binary(Binary::Sub, c, most));
    let e = b.push(Op::Unary(Unary::Exp, shifted));
    let sums = lines(&mut b, e, Binary::Add, None, true);
    let q = b.push(Op::Binary(Binary::Div, e, sums));
    let c = b.push(Op::Binary(Binary::Add, q, eps));
    let start = normalized(&mut b, c, eps, false);
    let carry = b.push(Op::Carry(start));
    let rows = normalized(&mut b, carry, eps, true);
    let next = normalized(&mut b, rows, eps, false);
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
    /// The gates and the norm, each its own kernel.
    Ops,
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
    fire(&gpu.ctx(), by, kernel, rows, width, p);
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
/// Sinkhorn's passes), two fires the same bits; the ops apart beside it.
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
            // the mixes on the row's lead warp (packed) or a side block
            // (spread) alike
            for schedule in Schedule::ALL {
                let at = format!("{case}, {schedule:?}");
                let (laid, p, c) = land(By::Region(Some(schedule)), Some(kernel), rows, width);
                assert_within(&at, &laid, &normed, NORM);
                near(&at, "post", &p, &post);
                near(&at, "combine", &c, &comb);
            }
            let (n, most) = apart(&land(By::Ops, None, rows, width).0, &normed);
            eprintln!(
                "{case}: the ops {n} of {} apart from the math, {most} ulps at most",
                rows * width
            );
        }
    }
}
