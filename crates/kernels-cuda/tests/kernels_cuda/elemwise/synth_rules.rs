#![cfg(feature = "cuda")]

//! The chains the fold forms as one generated row kernel on CUDA (a residual
//! add and its RMS norm, either way round), each against its own ops fired
//! unfused and against the kernels-ir interpreter: what each lands and how
//! fast against the bandwidth roofline.
//!
//! Each region here is the one poem-compiler's `fuse::chain` lowers the
//! chain's ops to on the catalog's models, op for op and in its parameters'
//! order (the first read first), its results the values read past the
//! chain, and every value an op defines rounded to bf16 where it is
//! defined, as the unfused ops store it:
//!
//! - a residual norm: `h = y + x`, `rmsnorm(h) · w`. The stream leaves (a
//!   layer's residual norms), or only the normed row does (the final norm:
//!   nothing reads the stream past it); either way the norm reads the
//!   stream as stored.
//! - a norm and fold: `y + rmsnorm(x) · w` (Gemma's sandwich norm), then
//!   optionally the next norm over the sum. The normed `t` is read by the
//!   add alone, so it never leaves, but the add reads it rounded, as the
//!   unfused add does.
//!
//! So each region is held to its math by the interpreter (`super::within`):
//! its sums of squares in the frame's own order, within a norm's ulp. Gemma
//! 4's layer scale, a weight of one element, is the region's scalar
//! parameter. The unfused ops are set beside it, not held to it.

use crate::common::attn_ref::{ten, tuned};
use crate::common::{Gpu, Lcg, captured, forget, from_bf16, replay, to_bf16};
use dtype::Dtype::Bf16;
use kernels_cuda::elemwise::{norm, pointwise};
use kernels_cuda::jit::Ctx;
use kernels_cuda::synth::{self, In, Out, Plane, Region, Synthesized};
use kernels_cuda::tuning::Tuning;
use kernels_ir::interp::{self, Arg};
use kernels_ir::{Binary, Literal, Op, Reduce, Shape, Unary};

use super::within::{EXACT, NORM, apart, assert_within, deterministic, widened};

const EPS: f32 = 1e-6;

/// Gemma 4's layer scale, as a bf16.
const SCALE: f32 = 0.71875;

/// One chain the fold forms as a region.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Chain {
    /// A residual add, then the norm of the sum; `stream` whether the sum
    /// is read past it.
    ResidualNorm { plus_one: bool, stream: bool },
    /// A norm folded into the stream; `post` whether the sum is normed
    /// again.
    NormResidual { post: bool },
    /// A norm folded into the stream with Gemma 4's layer scale (a weight
    /// of one element) on the sum, then the next norm.
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
            Chain::ResidualNorm { stream: true, .. } => "residual, norm",
            Chain::ResidualNorm { stream: false, .. } => "residual, norm (final)",
            Chain::NormResidual { post: false } => "norm, residual",
            Chain::NormResidual { post: true } => "norm, residual, norm",
            Chain::ScaledNormResidual => "norm, residual, scale, norm",
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

/// The knobs the unfused ops run under: none (the L40S's profile, and every
/// chip's default), or their register forms (B200's and H200's profiles,
/// `kernel.elemwise.rows_vec`).
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
            Knobs::Tuned => gpu.ctx().with_tuning(tuned(|t| t.rows_vec = true)),
        }
    }
}

/// One set of a fire's planes: the chain's row parameters and results, and
/// the unfused ops' scratch, each `rows` rows.
#[derive(Clone, Copy)]
struct Planes {
    x: u64,
    y: u64,
    /// The unfused norm's normed row.
    t: u64,
    /// The region's stream (the unfused add sums it into `y`).
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

/// The ops of `chain` over `p`, unfused, as the engine fires them: the
/// stream summed (and scaled) in place into `y`.
fn fire_unfused(ctx: &Ctx, chain: Chain, rows: usize, width: usize, p: Planes, w: Weights) {
    let t = |at| ten(at, rows, width, Bf16);
    let weight = |at| ten(at, width, 1, Bf16);
    let fired = || -> Result<(), kernels_cuda::error::Error> {
        match chain {
            Chain::ResidualNorm { plus_one, .. } => {
                pointwise::residual_add(ctx, t(p.x), &mut t(p.y))?;
                if plus_one {
                    norm::rmsnorm_plus_one(ctx, t(p.y), weight(w.w), EPS, &mut t(p.out))
                } else {
                    norm::rmsnorm(ctx, t(p.y), weight(w.w), EPS, &mut t(p.out))
                }
            }
            Chain::NormResidual { post } => {
                norm::rmsnorm(ctx, t(p.x), weight(w.w), EPS, &mut t(p.t))?;
                pointwise::residual_add(ctx, t(p.t), &mut t(p.y))?;
                if post {
                    norm::rmsnorm(ctx, t(p.y), weight(w.w1), EPS, &mut t(p.out))?;
                }
                Ok(())
            }
            Chain::ScaledNormResidual => {
                norm::rmsnorm(ctx, t(p.x), weight(w.w), EPS, &mut t(p.t))?;
                pointwise::residual_add(ctx, t(p.t), &mut t(p.y))?;
                pointwise::scale(ctx, ten(w.s, 1, 1, Bf16), &mut t(p.y))?;
                norm::rmsnorm(ctx, t(p.y), weight(w.w1), EPS, &mut t(p.out))
            }
        }
    };
    fired().expect("the unfused ops fire");
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

/// What `chain` lands over one draw, by its unfused ops under `knobs` or
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
            fire_unfused(&knobs.ctx(&gpu), chain, rows, width, p, weights);
            // the unfused add sums the stream into `y`
            p.y
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

/// The chain's region by the kernels-ir interpreter, row by row: the
/// region's own math in f32, each value rounded where it is defined.
fn math(chain: Chain, rows: usize, width: usize) -> Landed {
    let (x, y, w, w1) = draw(rows, width);
    let f = |v: &[u16]| -> Vec<f32> { v.iter().map(|&b| from_bf16(b)).collect() };
    let (x, y, w, w1) = (f(&x), f(&y), f(&w), f(&w1));
    let region = chain.region(width);
    let scale = from_bf16(to_bf16(SCALE));
    let (mut stream, mut out) = (Vec::new(), Vec::new());
    for r in 0..rows {
        let at = r * width..(r + 1) * width;
        let (x, y) = (Arg::Row(x[at.clone()].to_vec()), Arg::Row(y[at].to_vec()));
        let args = match chain {
            Chain::ResidualNorm { .. } => vec![x, y, Arg::Row(w.clone())],
            Chain::NormResidual { post: false } => vec![x, Arg::Row(w.clone()), y],
            Chain::NormResidual { post: true } => {
                vec![x, Arg::Row(w.clone()), y, Arg::Row(w1.clone())]
            }
            Chain::ScaledNormResidual => vec![
                x,
                Arg::Row(w.clone()),
                y,
                Arg::Scalar(scale),
                Arg::Row(w1.clone()),
            ],
        };
        let results = interp::eval(&region, &args).expect("the interpreter runs the region");
        let bf16 = |row: &[f32]| row.iter().map(|&v| to_bf16(v)).collect::<Vec<u16>>();
        match results.as_slice() {
            [last] => out.extend(bf16(last)),
            [sum, last] => {
                stream.extend(bf16(sum));
                out.extend(bf16(last));
            }
            _ => unreachable!("a chain lands one or two results"),
        }
    }
    Landed {
        stream: (chain.outs() == 2).then_some(stream),
        out,
    }
}

/// Every chain, at a decode's and a prefill's rows and a ragged count:
/// each generated kernel lands its region's math by the interpreter within
/// its tolerance (`super::within`), and two fires land the same bits. The
/// chain's unfused ops are shown beside it, not held to it.
#[test]
fn each_chains_region_lands_its_math() {
    let chains = Chain::TIMED.into_iter().chain([Chain::ResidualNorm {
        plus_one: true,
        stream: true,
    }]);
    for chain in chains {
        for width in chain.widths() {
            for rows in [1usize, 7, 64] {
                let case = format!("{} ({chain:?}): {rows} rows of {width}", chain.name());
                let region = land(chain, rows, width, None);
                let again = land(chain, rows, width, None);
                deterministic(&case, &region.out, &again.out);
                let math = math(chain, rows, width);
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
                    // The stream of a residual norm is IEEE adds; a norm and
                    // fold's sum joins a normed row an ulp of itself apart
                    // (`NORM`), up to four of the sum's own ulps where it
                    // cancels, and a post norm scales that up again.
                    let within = match (chain, what) {
                        (Chain::ResidualNorm { .. }, "stream") => EXACT,
                        (Chain::NormResidual { .. } | Chain::ScaledNormResidual, _) => 8.0,
                        _ => NORM,
                    };
                    assert_within(&format!("{case}: the {what}"), got, &widened(want), within);
                }
                for knobs in Knobs::BOTH {
                    let unfused = land(chain, rows, width, Some(knobs));
                    let (n, most) = apart(&region.out, &widened(&unfused.out));
                    eprintln!(
                        "{case}, unfused {knobs:?}: the result {n} of {} apart, {most} ulps at most",
                        rows * width
                    );
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

/// Each chain's unfused ops, under the default knobs and the tuned ones,
/// against its generated kernel, at a decode's and a prefill's rows over
/// the widths of the catalog's models that form it; each fire's bytes (the
/// chain's own: what its parameters read and its results write) against the
/// device's copy rate, the roofline: `cargo test --release ... -- --ignored
/// synthesized_rule_rate --nocapture`. A fire is a graph of its fires over
/// planes rotating through at least 1 GiB, so a prefill's fire reads HBM,
/// not the L2, even where it touches several planes of a set (the unfused
/// ops land a normed row between them and sum the stream in place; a region
/// writes its own plane): the median over
/// rounds, each graph in turn, of each round's median replay.
#[test]
#[ignore = "a rate bench: cargo test --release ... -- --ignored synthesized_rule_rate --nocapture"]
fn synthesized_rule_rate() {
    use super::frame::copy_rate;
    // a set's five planes; a fire may touch several of them (an in-place
    // stream), so the sets span four times that of any card's L2
    const FOOTPRINT: usize = 1 << 30;
    const FIRES: usize = 256;
    const REPLAYS: u32 = 11;
    const ROUNDS: usize = 7;
    let copy = copy_rate();
    eprintln!("device copy: {copy:.0} GB/s");
    eprintln!(
        "{:<28}  rows  width  unfused us  %copy  tuned us  %copy  region us  %copy  region/best",
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
                        fire_unfused(&contexts[0], chain, rows, width, p, weights);
                    }),
                    graph(&contexts[1], &|p| {
                        fire_unfused(&contexts[1], chain, rows, width, p, weights);
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
