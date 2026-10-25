#![cfg(feature = "cuda")]

//! The hand fused kernels whose chains the fold forms as one generated row
//! kernel once their rule is off (`fuse.residual_add_rmsnorm`,
//! `fuse.rmsnorm_residual_add`), against that kernel: what each lands and
//! how fast, so a rule is deleted on its numbers.
//!
//! Each region here is the one poem-compiler's `fuse::chain` lowers the
//! rule's ops to on the catalog's models, op for op and in its parameters'
//! order (the first read first), its results the values read past the
//! chain, and every value an op defines rounded to bf16 where it is
//! defined, as the unfused ops store it:
//!
//! - `residual_add_rmsnorm`: `h = y + x`, `rmsnorm(h) · w`. The stream
//!   leaves (a layer's residual norms), or only the normed row does (the
//!   final norm: nothing reads the stream past it); either way the norm
//!   reads the stream as stored.
//! - `rmsnorm_residual_add`: `y + rmsnorm(x) · w` (Gemma's sandwich norm),
//!   then optionally the next norm over the sum. The normed `t` is read by
//!   the add alone, so it never leaves, but the add reads it rounded, as the
//!   hand kernel does.
//!
//! So each region lands the hand kernel's bits, its sums of squares taken
//! two at a time as the hand kernels take them (`a += h₀² + h₁²`), but
//! where a row's threads hold it otherwise than the hand kernel's: Gemma's
//! norm-and-fold at 5376 splits the row over 512 threads, the frame over
//! 256, so a sum's tree differs in its last bit and a normed element by an
//! ulp, about once in a hundred thousand.
//!
//! The chains a rule forms that no region holds (Gemma's layer scalar, a
//! one-element weight) are not here: with their rule off they run as more
//! kernels, not as one.

use crate::common::attn_ref::{ten, tuned};
use crate::common::{Gpu, Lcg, captured, forget, from_bf16, replay, to_bf16};
use dtype::Dtype::Bf16;
use kernels_cuda::elemwise::norm_residual::{self, Keep, PostNorm};
use kernels_cuda::jit::Ctx;
use kernels_cuda::synth::{self, In, Out, Plane, Region, Synthesized};
use kernels_cuda::tuning::Tuning;
use kernels_ir::{Binary, Literal, Op, Reduce, Shape, Unary};

const EPS: f32 = 1e-6;

/// Gemma 4's layer scale, as a bf16.
const SCALE: f32 = 0.71875;

/// One chain a hand rule forms and the fold forms as a region.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Chain {
    /// `residual_add_rmsnorm`; `stream` whether the sum is read past it.
    ResidualNorm { plus_one: bool, stream: bool },
    /// `rmsnorm_residual_add`; `post` whether the sum is normed again.
    NormResidual { post: bool },
    /// `rmsnorm_residual_add` with Gemma 4's layer scale (a weight of one
    /// element) on the sum, then the next norm.
    ScaledNormResidual,
}

impl Chain {
    /// The chains the bench times, as the catalog forms them most.
    const TIMED: [Chain; 5] = [
        Chain::ResidualNorm {
            plus_one: false,
            stream: true,
        },
        Chain::ResidualNorm {
            plus_one: false,
            stream: false,
        },
        Chain::NormResidual { post: false },
        Chain::NormResidual { post: true },
        Chain::ScaledNormResidual,
    ];

    fn name(self) -> &'static str {
        match self {
            Chain::ResidualNorm { stream: true, .. } => "residual_add_rmsnorm",
            Chain::ResidualNorm { stream: false, .. } => "residual_add_rmsnorm (final)",
            Chain::NormResidual { post: false } => "rmsnorm_residual_add",
            Chain::NormResidual { post: true } => "rmsnorm_residual_add + post",
            Chain::ScaledNormResidual => "rmsnorm_residual_add scaled",
        }
    }

    /// The widths of the catalog's models that form it.
    fn widths(self) -> [usize; 5] {
        match self {
            Chain::ResidualNorm { .. } => [2048, 4096, 5120, 6144, 8192],
            // 8192 is past the frame with post's six planes; Gemma's widths
            Chain::NormResidual { .. } | Chain::ScaledNormResidual => {
                [1536, 2560, 2816, 3840, 5376]
            }
        }
    }

    /// Which parameters are one row every row reads.
    fn shared(self) -> &'static [usize] {
        match self {
            Chain::ResidualNorm { .. } => &[2],
            Chain::NormResidual { post: false } => &[1],
            Chain::NormResidual { post: true } => &[1, 3],
            Chain::ScaledNormResidual => &[1, 4],
        }
    }

    /// Which parameters are one element every row reads.
    fn ones(self) -> &'static [usize] {
        match self {
            Chain::ScaledNormResidual => &[3],
            _ => &[],
        }
    }

    fn outs(self) -> usize {
        match self {
            Chain::ResidualNorm { stream, .. } => 1 + usize::from(stream),
            Chain::NormResidual { post } => 1 + usize::from(post),
            Chain::ScaledNormResidual => 2,
        }
    }

    /// The bytes the chain must move over `rows` rows of `width`: each
    /// parameter row read, each result row written, each weight once.
    fn bytes(self, rows: usize, width: usize) -> usize {
        let plane = rows * width * 2;
        let rows_in = self.region(width).params.len() - self.shared().len() - self.ones().len();
        (rows_in + self.outs()) * plane + self.shared().len() * width * 2 + self.ones().len() * 2
    }

    /// The region `fuse::chain` lowers the chain to.
    fn region(self, width: usize) -> Region {
        let mut b = Build::new(width);
        match self {
            Chain::ResidualNorm { plus_one, stream } => {
                // parameters x, y, w
                b.params(3);
                let h = b.stored(Op::Binary(Binary::Add, 1, 0));
                let n = b.rms_normed(h);
                let out = b.scaled(n, 2, plus_one);
                b.round(out);
                b.finish(if stream { vec![h, out] } else { vec![out] })
            }
            Chain::ScaledNormResidual => {
                // parameters x, w, y, the scale, the post norm's weight
                b.params(5);
                let n = b.rms_normed(0);
                let t = b.scaled(n, 1, false);
                b.round(t);
                let sum = b.stored(Op::Binary(Binary::Add, 2, t));
                let scaled = b.stored(Op::Binary(Binary::Mul, sum, 3));
                let n = b.rms_normed(scaled);
                let out = b.scaled(n, 4, false);
                b.round(out);
                b.finish_with(vec![scaled, out], &[3])
            }
            Chain::NormResidual { post } => {
                // parameters x, w, y, then the post norm's weight
                b.params(if post { 4 } else { 3 });
                let n = b.rms_normed(0);
                let t = b.scaled(n, 1, false);
                b.round(t);
                let sum = b.stored(Op::Binary(Binary::Add, 2, t));
                if post {
                    let n = b.rms_normed(sum);
                    let out = b.scaled(n, 3, false);
                    b.round(out);
                    b.finish(vec![sum, out])
                } else {
                    b.finish(vec![sum])
                }
            }
        }
    }
}

/// A region's ops, built as `fuse::chain`'s builder builds them.
struct Build {
    width: usize,
    params: u32,
    ops: Vec<Op>,
    rounds: Vec<(u32, dtype::Dtype)>,
}

impl Build {
    fn new(width: usize) -> Build {
        Build {
            width,
            params: 0,
            ops: Vec::new(),
            rounds: Vec::new(),
        }
    }

    fn params(&mut self, n: u32) {
        self.params = n;
    }

    fn push(&mut self, op: Op) -> u32 {
        self.ops.push(op);
        self.params + u32::try_from(self.ops.len()).expect("a value id") - 1
    }

    /// `v`, a trace value an op defines, rounded where it is defined.
    fn round(&mut self, v: u32) {
        self.rounds.push((v, Bf16));
    }

    /// `op`'s value, a trace value, rounded.
    fn stored(&mut self, op: Op) -> u32 {
        let v = self.push(op);
        self.round(v);
        v
    }

    fn konst(&mut self, x: f32) -> u32 {
        self.push(Op::Const(Literal::F32(x)))
    }

    /// `x · rsqrt(sum(x²) / width + eps)`.
    fn rms_normed(&mut self, x: u32) -> u32 {
        let sq = self.push(Op::Binary(Binary::Mul, x, x));
        let sum = self.push(Op::Reduce(Reduce::Sum, sq));
        let n = self.konst(self.width as f32);
        let ms = self.push(Op::Binary(Binary::Div, sum, n));
        let eps = self.konst(EPS);
        let shifted = self.push(Op::Binary(Binary::Add, ms, eps));
        let r = self.push(Op::Unary(Unary::Rsqrt, shifted));
        self.push(Op::Binary(Binary::Mul, x, r))
    }

    /// `n · w`, or `n · (w + 1)`.
    fn scaled(&mut self, n: u32, w: u32, plus_one: bool) -> u32 {
        let w = if plus_one {
            let one = self.konst(1.0);
            self.push(Op::Binary(Binary::Add, w, one))
        } else {
            w
        };
        self.push(Op::Binary(Binary::Mul, n, w))
    }

    fn finish(self, results: Vec<u32>) -> Region {
        self.finish_with(results, &[])
    }

    /// The region, the parameters `ones` names scalars.
    fn finish_with(self, results: Vec<u32>, ones: &[usize]) -> Region {
        Region {
            width: u32::try_from(self.width).expect("a width"),
            params: (0..self.params as usize)
                .map(|k| {
                    if ones.contains(&k) {
                        Shape::Scalar
                    } else {
                        Shape::Row
                    }
                })
                .collect(),
            ops: self.ops,
            results,
            rounds: self.rounds,
        }
    }
}

/// The chain's generated kernel at `width`, loaded.
fn generated(chain: Chain, width: usize) -> Synthesized {
    let region = chain.region(width);
    let ins: Vec<In> = (0..region.params.len())
        .map(|k| {
            if chain.ones().contains(&k) {
                In::One(Bf16)
            } else if chain.shared().contains(&k) {
                In::Shared(Bf16)
            } else {
                In::Rows(Bf16)
            }
        })
        .collect();
    let outs = vec![Out::Rows(Bf16); region.results.len()];
    synth::supports(&region, &ins, &outs).expect("the frame takes the region");
    let kernel = synth::synth(&region, &ins, outs.len()).expect("synthesized");
    kernel
        .load()
        .expect("the generated kernel compiles and loads");
    kernel
}

/// The knobs the hand kernels run under: none (the L40S's profile, and
/// every chip's default), or the register forms and unread rows skipped
/// (B200's and H200's profiles, `kernel.elemwise.{rows_vec,
/// norm_fold_block}`, the rows nothing reads left unwritten as the engine
/// left them when it fired the hand kernel).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Knobs {
    Default,
    Tuned,
}

impl Knobs {
    const BOTH: [Knobs; 2] = [Knobs::Default, Knobs::Tuned];

    fn ctx(self, gpu: &Gpu) -> Ctx {
        match self {
            Knobs::Default => gpu.ctx().with_tuning(Tuning::DEFAULT),
            Knobs::Tuned => gpu.ctx().with_tuning(tuned(|t| {
                t.rows_vec = true;
                t.norm_fold_block = 128;
            })),
        }
    }

    /// The rows the hand norm-and-fold lands: tuned, the
    /// engine leaves the normed row it alone reads unwritten, and the sum
    /// too where only its scaled copy goes on.
    fn keep(self, chain: Chain) -> Keep {
        match (self, chain) {
            (Knobs::Default, _) => Keep::ALL,
            (Knobs::Tuned, Chain::ScaledNormResidual) => Keep { t: false, y: false },
            (Knobs::Tuned, _) => Keep { t: false, y: true },
        }
    }
}

/// One set of a fire's planes: the chain's row parameters and results, and
/// the hand kernel's scratch, each `rows` rows.
#[derive(Clone, Copy)]
struct Planes {
    x: u64,
    y: u64,
    /// The hand norm-and-fold's normed row.
    t: u64,
    /// The region's stream (the hand kernels sum it into `y`).
    h: u64,
    out: u64,
}

impl Planes {
    fn zeros(gpu: &mut Gpu, plane: usize) -> Planes {
        Planes {
            x: gpu.zeros(plane),
            y: gpu.zeros(plane),
            t: gpu.zeros(plane),
            h: gpu.zeros(plane),
            out: gpu.zeros(plane),
        }
    }
}

/// The weights a chain reads.
#[derive(Clone, Copy)]
struct Weights {
    w: u64,
    w1: u64,
    /// A layer's scale: one element.
    s: u64,
}

/// The hand kernel of `chain` over `p`: the stream summed into `y`.
#[allow(clippy::too_many_arguments)]
fn fire_hand(
    ctx: &Ctx,
    knobs: Knobs,
    chain: Chain,
    rows: usize,
    width: usize,
    p: Planes,
    w: Weights,
) {
    let t = |at| ten(at, rows, width, Bf16);
    let weight = |at| ten(at, width, 1, Bf16);
    match chain {
        Chain::ResidualNorm { plus_one, .. } => norm_residual::residual_add_rmsnorm(
            ctx,
            t(p.x),
            &mut t(p.y),
            weight(w.w),
            plus_one,
            EPS,
            &mut t(p.out),
        ),
        Chain::NormResidual { post } => {
            let mut out = t(p.out);
            norm_residual::rmsnorm_residual_add_keeping(
                ctx,
                t(p.x),
                weight(w.w),
                EPS,
                &mut t(p.t),
                &mut t(p.y),
                None,
                post.then(|| PostNorm {
                    weight: weight(w.w1),
                    plus_one: false,
                    eps: EPS,
                    out: &mut out,
                }),
                knobs.keep(chain),
            )
        }
        Chain::ScaledNormResidual => {
            let (mut out, mut scaled) = (t(p.out), t(p.h));
            norm_residual::rmsnorm_residual_add_keeping(
                ctx,
                t(p.x),
                weight(w.w),
                EPS,
                &mut t(p.t),
                &mut t(p.y),
                Some((ten(w.s, 1, 1, Bf16), &mut scaled)),
                Some(PostNorm {
                    weight: weight(w.w1),
                    plus_one: false,
                    eps: EPS,
                    out: &mut out,
                }),
                knobs.keep(chain),
            )
        }
    }
    .expect("the hand kernel fires");
}

/// The generated kernel of `chain` over `p`: the stream into `h`.
fn fire_region(
    ctx: &Ctx,
    kernel: Synthesized,
    chain: Chain,
    rows: usize,
    width: usize,
    p: Planes,
    w: Weights,
) {
    let row = |at| Plane::rows(&ten(at, rows, width, Bf16));
    let weight = |at| Plane::shared(&ten(at, 1, width, Bf16));
    let (ins, outs) = match chain {
        Chain::ResidualNorm { stream: true, .. } => (
            vec![row(p.x), row(p.y), weight(w.w)],
            vec![row(p.h), row(p.out)],
        ),
        Chain::ResidualNorm { stream: false, .. } => {
            (vec![row(p.x), row(p.y), weight(w.w)], vec![row(p.out)])
        }
        Chain::NormResidual { post: false } => {
            (vec![row(p.x), weight(w.w), row(p.y)], vec![row(p.h)])
        }
        Chain::NormResidual { post: true } => (
            vec![row(p.x), weight(w.w), row(p.y), weight(w.w1)],
            vec![row(p.h), row(p.out)],
        ),
        Chain::ScaledNormResidual => (
            vec![
                row(p.x),
                weight(w.w),
                row(p.y),
                Plane::shared(&ten(w.s, 1, 1, Bf16)),
                weight(w.w1),
            ],
            vec![row(p.h), row(p.out)],
        ),
    };
    kernel
        .fire(ctx, &ins, &outs, u32::try_from(rows).expect("rows"))
        .expect("the generated kernel fires");
}

/// How far apart `a` and `b` are, in bf16 ulps at the scale of the larger
/// of them, or of a half where both are smaller: the chain's rows are of
/// order one, and a sum that cancels to near zero would count a rounding of
/// its addends in the sum's own, far finer, ulps.
fn ulps(a: u16, b: u16) -> f32 {
    let (a, b) = (from_bf16(a), from_bf16(b));
    let scale = a.abs().max(b.abs()).max(0.5);
    let spacing = 2f32.powi(scale.log2().floor() as i32 - 7);
    (a - b).abs() / spacing
}

/// The elements of `got` not `want`, and the most ulps apart one is.
fn apart(got: &[u16], want: &[u16]) -> (usize, f32) {
    got.iter()
        .zip(want)
        .filter(|(a, b)| a != b)
        .fold((0, 0.0), |(n, most), (a, b)| {
            (n + 1, most.max(ulps(*a, *b)))
        })
}

/// What one draw lands: the stream and the chain's last result.
struct Landed {
    stream: Option<Vec<u16>>,
    out: Vec<u16>,
}

/// A seeded draw of `rows` rows of `width`: the planes x and y, weights near
/// one.
fn draw(rows: usize, width: usize) -> (Vec<u16>, Vec<u16>, Vec<u16>, Vec<u16>) {
    let mut lcg = Lcg::seeded(0x5e7d_0003 ^ ((width as u64) << 12) ^ rows as u64);
    let x = lcg.row(rows * width).0;
    let y = lcg.row(rows * width).0;
    let mut near_one = || -> Vec<u16> {
        (0..width)
            .map(|_| to_bf16(1.0 + 0.25 * lcg.unit()))
            .collect()
    };
    let (w, w1) = (near_one(), near_one());
    (x, y, w, w1)
}

/// What `chain` lands over one draw, by its hand kernel under `knobs` or
/// (`None`) by its generated kernel.
fn land(chain: Chain, rows: usize, width: usize, by: Option<Knobs>) -> Landed {
    let n = rows * width;
    let (x, y, w, w1) = draw(rows, width);
    let mut gpu = Gpu::open();
    let p = Planes {
        x: gpu.up(&x),
        y: gpu.up(&y),
        t: gpu.zeros(n * 2),
        h: gpu.zeros(n * 2),
        out: gpu.zeros(n * 2),
    };
    let weights = Weights {
        w: gpu.up(&w),
        w1: gpu.up(&w1),
        s: gpu.up(&[to_bf16(SCALE)]),
    };
    let stream_at = match by {
        Some(knobs) => {
            fire_hand(&knobs.ctx(&gpu), knobs, chain, rows, width, p, weights);
            // the hand kernel sums the stream into `y`, and lands a scaled
            // one in its own plane
            if chain == Chain::ScaledNormResidual {
                p.h
            } else {
                p.y
            }
        }
        None => {
            let kernel = generated(chain, width);
            fire_region(&gpu.ctx(), kernel, chain, rows, width, p, weights);
            p.h
        }
    };
    gpu.sync();
    let (stream, out) = match chain {
        Chain::ResidualNorm { stream: true, .. }
        | Chain::NormResidual { post: true }
        | Chain::ScaledNormResidual => (Some(gpu.down(stream_at, n)), gpu.down(p.out, n)),
        Chain::ResidualNorm { stream: false, .. } => (None, gpu.down(p.out, n)),
        Chain::NormResidual { post: false } => (None, gpu.down(stream_at, n)),
    };
    Landed { stream, out }
}

/// The chain's own math on the host, in f32: values only the chain reads
/// kept unrounded, each result rounded as stored and read on as rounded.
fn host(chain: Chain, rows: usize, width: usize) -> Landed {
    let (x, y, w, w1) = draw(rows, width);
    let f = |v: &[u16]| -> Vec<f32> { v.iter().map(|&b| from_bf16(b)).collect() };
    let (x, y, w, w1) = (f(&x), f(&y), f(&w), f(&w1));
    let inv = |row: &[f32]| {
        let ss: f32 = row.iter().map(|v| v * v).sum();
        1.0 / (ss / width as f32 + EPS).sqrt()
    };
    let (mut stream, mut out) = (Vec::new(), Vec::new());
    for r in 0..rows {
        let at = r * width..(r + 1) * width;
        let (x, y) = (&x[at.clone()], &y[at]);
        match chain {
            Chain::ResidualNorm { plus_one, .. } => {
                let h: Vec<f32> = (0..width)
                    .map(|e| from_bf16(to_bf16(y[e] + x[e])))
                    .collect();
                let r = inv(&h);
                stream.extend(h.iter().map(|&v| to_bf16(v)));
                out.extend((0..width).map(|e| {
                    let w = if plus_one { w[e] + 1.0 } else { w[e] };
                    to_bf16(h[e] * r * w)
                }));
            }
            Chain::NormResidual { post } => {
                let r = inv(x);
                let sum: Vec<f32> = (0..width)
                    .map(|e| from_bf16(to_bf16(y[e] + from_bf16(to_bf16(x[e] * r * w[e])))))
                    .collect();
                stream.extend(sum.iter().map(|&v| to_bf16(v)));
                if post {
                    let r = inv(&sum);
                    out.extend((0..width).map(|e| to_bf16(sum[e] * r * w1[e])));
                }
            }
            Chain::ScaledNormResidual => {
                let r = inv(x);
                let scaled: Vec<f32> = (0..width)
                    .map(|e| {
                        let t = from_bf16(to_bf16(x[e] * r * w[e]));
                        let sum = from_bf16(to_bf16(y[e] + t));
                        from_bf16(to_bf16(sum * from_bf16(to_bf16(SCALE))))
                    })
                    .collect();
                stream.extend(scaled.iter().map(|&v| to_bf16(v)));
                let r = inv(&scaled);
                out.extend((0..width).map(|e| to_bf16(scaled[e] * r * w1[e])));
            }
        }
    }
    match chain {
        Chain::ResidualNorm { stream: true, .. }
        | Chain::NormResidual { post: true }
        | Chain::ScaledNormResidual => Landed {
            stream: Some(stream),
            out,
        },
        Chain::ResidualNorm { stream: false, .. } => Landed { stream: None, out },
        Chain::NormResidual { post: false } => Landed {
            stream: None,
            out: stream,
        },
    }
}

/// Every chain, at a decode's and a prefill's rows and a ragged count:
/// each generated kernel lands its own math within an ulp (its sums of
/// squares are added in another order than the host's), and is set
/// against each hand kernel, element for element.
#[test]
fn each_rules_region_lands_its_math_and_is_set_against_the_hand_kernel() {
    let chains = Chain::TIMED.into_iter().chain([Chain::ResidualNorm {
        plus_one: true,
        stream: true,
    }]);
    for chain in chains {
        for width in chain.widths() {
            for rows in [1usize, 7, 64] {
                let case = format!("{} ({chain:?}): {rows} rows of {width}", chain.name());
                let region = land(chain, rows, width, None);
                let math = host(chain, rows, width);
                for (what, got, want) in [
                    ("stream", region.stream.as_deref(), math.stream.as_deref()),
                    (
                        "result",
                        Some(region.out.as_slice()),
                        Some(math.out.as_slice()),
                    ),
                ] {
                    let (Some(got), Some(want)) = (got, want) else {
                        continue;
                    };
                    // `t` an ulp of itself apart (the host's `1 / sqrt` is
                    // not `rsqrtf`) is a sum up to four of the sum's scale
                    // apart where it cancels, and a post norm scales that
                    // up again: a sanity bound; the hand kernel's bits are
                    // the test
                    let within = match (chain, what) {
                        (Chain::NormResidual { .. } | Chain::ScaledNormResidual, _) => 8.0,
                        _ => 1.0,
                    };
                    let (_, most) = apart(got, want);
                    assert!(
                        most <= within,
                        "{case}: the {what} is {most} ulps from its math at worst"
                    );
                }
                for knobs in Knobs::BOTH {
                    let hand = land(chain, rows, width, Some(knobs));
                    let stream = region
                        .stream
                        .as_deref()
                        .zip(hand.stream.as_deref())
                        .map(|(got, want)| apart(got, want));
                    let (n, most) = apart(&region.out, &hand.out);
                    eprintln!(
                        "{case}, hand {knobs:?}: stream (apart, ulps) {stream:?}, result {n} of {} \
                         apart, {most} ulps at most",
                        rows * width
                    );
                    // The hand kernel's bits but where a sum of squares
                    // lands in its last bit apart: a norm an ulp apart, and
                    // the sum a norm's row joins (or the norm after it) an
                    // ulp of its addends apart, about once in a million.
                    let total = rows * width;
                    for (what, apart) in [("stream", stream), ("result", Some((n, most)))] {
                        let Some((n, most)) = apart else { continue };
                        assert!(
                            n * 10_000 <= total.max(10_000) && most <= 2.0,
                            "{case}, hand {knobs:?}: {n} of {total} {what} elements apart, \
                             {most} ulps at most"
                        );
                    }
                }
            }
        }
    }
}

/// The median of `runs` runs of `run`, in microseconds.
fn median(runs: u32, mut run: impl FnMut()) -> f64 {
    let mut us: Vec<f64> = (0..runs)
        .map(|_| {
            let start = std::time::Instant::now();
            run();
            start.elapsed().as_secs_f64() * 1e6
        })
        .collect();
    us.sort_by(f64::total_cmp);
    us[us.len() / 2]
}

/// Each chain's hand kernel, under the default knobs and the tuned ones,
/// against its generated kernel, at a decode's and a prefill's rows over
/// the widths of the catalog's models that form it; each fire's bytes (the
/// chain's own: what its parameters read and its results write) against the
/// device's copy rate: `cargo test --release ... -- --ignored
/// synthesized_rule_rate --nocapture`. A fire is a graph of its fires over
/// planes rotating through at least 1 GiB, so a prefill's fire reads HBM,
/// not the L2, even where it touches two planes of a set (the hand kernels
/// sum the stream in place; a region writes its own plane): the median over
/// rounds, each graph in turn, of each round's median replay.
#[test]
#[ignore = "a rate bench: cargo test --release ... -- --ignored synthesized_rule_rate --nocapture"]
fn synthesized_rule_rate() {
    use super::frame::copy_rate;
    // a set's five planes; a fire may touch two of them (an in-place
    // stream), so the sets span four times that of any card's L2
    const FOOTPRINT: usize = 1 << 30;
    const FIRES: usize = 256;
    const REPLAYS: u32 = 11;
    const ROUNDS: usize = 7;
    let copy = copy_rate();
    eprintln!("device copy: {copy:.0} GB/s");
    eprintln!(
        "{:<28}  rows  width  default us  %copy  tuned us  %copy  region us  %copy  region/best",
        "chain"
    );
    for chain in Chain::TIMED {
        for rows in [1usize, 8, 64, 512, 4096] {
            for width in chain.widths() {
                let kernel = generated(chain, width);
                let mut gpu = Gpu::open();
                let plane = rows * width * 2;
                let copies = FOOTPRINT.div_ceil(5 * plane).clamp(1, FIRES);
                let weights = Weights {
                    w: gpu.zeros(width * 2),
                    w1: gpu.zeros(width * 2),
                    s: gpu.zeros(16),
                };
                let sets: Vec<Planes> = (0..copies)
                    .map(|_| Planes::zeros(&mut gpu, plane))
                    .collect();
                let fires = FIRES.max(copies) / copies * copies;
                let graph = |ctx: &Ctx, fire: &dyn Fn(Planes)| {
                    let all = || {
                        for i in 0..fires {
                            fire(sets[i % copies]);
                        }
                    };
                    all();
                    gpu.sync();
                    let exec = captured(ctx, all);
                    replay(ctx, exec);
                    gpu.sync();
                    exec
                };
                let contexts = [Knobs::Default.ctx(&gpu), Knobs::Tuned.ctx(&gpu), gpu.ctx()];
                let execs = [
                    graph(&contexts[0], &|p| {
                        fire_hand(&contexts[0], Knobs::Default, chain, rows, width, p, weights);
                    }),
                    graph(&contexts[1], &|p| {
                        fire_hand(&contexts[1], Knobs::Tuned, chain, rows, width, p, weights);
                    }),
                    graph(&contexts[2], &|p| {
                        fire_region(&contexts[2], kernel, chain, rows, width, p, weights);
                    }),
                ];
                // rounds of each graph in turn, so a burst of another
                // process's work on a shared card lands in a round of each
                let mut rounds = [const { Vec::new() }; 3];
                for _ in 0..ROUNDS {
                    for (k, exec) in execs.iter().enumerate() {
                        rounds[k].push(median(REPLAYS, || {
                            replay(&contexts[k], *exec);
                            gpu.sync();
                        }));
                    }
                }
                for exec in execs {
                    forget(exec);
                }
                let [default_us, tuned_us, region_us] = rounds.map(|mut us| {
                    us.sort_by(f64::total_cmp);
                    us[us.len() / 2] / fires as f64
                });
                let share = |us: f64| 100.0 * chain.bytes(rows, width) as f64 / us / 1e3 / copy;
                eprintln!(
                    "{:<28}  {rows:>4}  {width:>5}  {default_us:>10.2}  {:>4.0}%  {tuned_us:>8.2}  {:>4.0}%  {region_us:>9.2}  {:>4.0}%  {:>11.3}",
                    chain.name(),
                    share(default_us),
                    share(tuned_us),
                    share(region_us),
                    region_us / default_us.min(tuned_us),
                );
            }
        }
    }
}
