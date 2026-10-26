#![cfg(feature = "cuda")]

//! `fused.norm_rope`'s chain, a head's RMS norm and the rotation of its
//! heads at each row's position, as the fold forms it once the rule is off:
//! one generated row kernel, against the two ops it replaces and (a turn of
//! the tail) the hand fused kernel, what each lands. A turn of the first
//! dims across a wide head (Gemma 4's global q) is pair-packed where its
//! rows are: its turned vectors on whole warps, the rest's warps skipping
//! the turn.
//!
//! Each region is the one poem-compiler's `fuse::chain` lowers the two ops
//! to (kernels-ir's `build`, the builder it lowers with): the norm over
//! each head (the row read as its heads, one head's weight across them, or
//! a row one head wide normed as a row), its row rounded to bf16 as the
//! norm's plane stores it, then each head turned as the rotation kernels
//! turn it: `powf` frequencies, the skeleton's `__sincosf` angles, a column and
//! its mate (read back through shared memory) landing `a cos ∓ b sin`.
//!
//! Each is held to its math by the interpreter (`super::within`): the norm
//! sums a head in the skeleton's order, and the turn's angle is the skeleton's
//! `__sincosf`, within [`TURN`] ulps. The ops and the hand fused kernel are
//! set beside it, not held to it: the hand kernel turns each pair from the
//! norm's f32 values, not from its rounded row.

use crate::common::attn_ref::{ten, tuned};
use crate::common::{Gpu, Lcg, to_bf16};
use dtype::Dtype::{Bf16, I32};
use kernels_cuda::elemwise::{norm, norm_rope, rope};
use kernels_cuda::jit::Ctx;
use kernels_cuda::synth::{self, In, Out, Plane, Region, Schedule, Synthesized};
use kernels_cuda::tuning::Tuning;
use kernels_ir::build::{Ops, Pairs, Ramp, Turn};
use kernels_ir::interp::Arg;
use kernels_ir::{Binary, Op, Shape};

use super::within::{TURN, apart, assert_within, deterministic, reference, row_of, widened};

/// One norm-and-turn the catalog forms.
#[derive(Clone, Copy, Debug)]
struct Case {
    name: &'static str,
    width: u32,
    head: u32,
    /// The norm's weight: one head wide (`Some(head)`), a row one head
    /// wide normed as a row, or none (scale-free).
    weighted: bool,
    eps: f32,
    theta: f32,
    rotary: u32,
    /// `Some(span)`: the first pairs `span` apart (`Turn::Partial`); else
    /// the tail interleaved (`Turn::Last`).
    span: Option<u32>,
    yarn: Option<rope::Yarn>,
}

const DSV41_YARN: rope::Yarn = rope::Yarn {
    factor: 16.0,
    beta_fast: 32.0,
    beta_slow: 1.0,
    original_max_position: 65536,
};

/// The chains the catalog's models form, at their widths.
const CASES: [Case; 7] = [
    // Gemma 4's local layers' q: 8 or 32 heads of 256, a full split turn
    Case {
        name: "gemma4 local q",
        width: 2048,
        head: 256,
        weighted: true,
        eps: 1e-6,
        theta: 10_000.0,
        rotary: 256,
        span: Some(128),
        yarn: None,
    },
    Case {
        name: "gemma4 local q (31b)",
        width: 8192,
        head: 256,
        weighted: true,
        eps: 1e-6,
        theta: 10_000.0,
        rotary: 256,
        span: Some(128),
        yarn: None,
    },
    // its global layers': heads of 512, a proportional quarter turned
    Case {
        name: "gemma4 global q",
        width: 4096,
        head: 512,
        weighted: true,
        eps: 1e-6,
        theta: 1_000_000.0,
        rotary: 128,
        span: Some(256),
        yarn: None,
    },
    // DeepSeek-V4.1's rows one 128-wide head: a 64-wide tail, YaRN
    Case {
        name: "dsv41 q-lora row",
        width: 128,
        head: 128,
        weighted: true,
        eps: 1e-20,
        theta: 160_000.0,
        rotary: 64,
        span: None,
        yarn: Some(DSV41_YARN),
    },
    // DeepSeek-V4's q: 16 scale-free heads of 128
    Case {
        name: "dsv4 q heads",
        width: 2048,
        head: 128,
        weighted: false,
        eps: 1e-5,
        theta: 10_000.0,
        rotary: 64,
        span: None,
        yarn: None,
    },
    // dots3's 64-wide k row, wholly turned interleaved
    Case {
        name: "dots3 k row",
        width: 64,
        head: 64,
        weighted: true,
        eps: 1e-5,
        theta: 50_000.0,
        rotary: 64,
        span: None,
        yarn: None,
    },
    Case {
        name: "dots3 k row (global)",
        width: 64,
        head: 64,
        weighted: true,
        eps: 1e-5,
        theta: 80_000_000.0,
        rotary: 64,
        span: None,
        yarn: None,
    },
];

impl Case {
    fn turn(self) -> Turn {
        let ramp = self.yarn.filter(|y| y.factor > 1.0).map(|y| {
            let (low, high) = rope::ramp_bounds(
                self.rotary as i32,
                self.theta,
                y.beta_fast,
                y.beta_slow,
                y.original_max_position as i32,
            );
            Ramp {
                factor: y.factor,
                low,
                high,
            }
        });
        Turn {
            head_dim: self.head,
            pairs: match self.span {
                Some(span) => Pairs::Halves {
                    rotary: self.rotary,
                    span,
                },
                None => Pairs::Tail {
                    rotary: self.rotary,
                    interleaved: true,
                },
            },
            theta: self.theta,
            inverse: false,
            ramp,
            mscale: None,
        }
    }

    /// The region `fuse::chain` lowers the two ops to: parameters x, the
    /// weight (where there is one), the position. Every op is a head's
    /// alone, so the chain cuts each row into its heads (`Cut::Heads`),
    /// the region a function of one head; `Cut::Row` keeps the row whole,
    /// read as its heads.
    fn region(self, cut: Cut) -> Region {
        let params = if self.weighted { 3 } else { 2 };
        let pos = params - 1;
        let width = self.kernel_width(cut);
        let mut b = Ops::new(params);
        let n = if self.head == width {
            b.rms_normed(0, width, self.eps)
        } else {
            b.rms_normed_heads(0, self.head, self.eps)
        };
        // a head's weight, the plane repeating it across the row's heads
        let y = if self.weighted {
            b.push(Op::Binary(Binary::Mul, n, 1))
        } else {
            n
        };
        let out = b.rotate(y, pos, &self.turn());
        let mut params = vec![Shape::Row];
        if self.weighted {
            params.push(Shape::Row);
        }
        params.push(Shape::Scalar);
        Region {
            width,
            params,
            ops: b.ops,
            results: vec![out],
            rounds: vec![(y, Bf16), (out, Bf16)],
        }
    }

    /// The width of the kernel's row.
    fn kernel_width(self, cut: Cut) -> u32 {
        match cut {
            Cut::Heads => self.head,
            Cut::Row => self.width,
        }
    }

    fn ins(self) -> Vec<In> {
        let mut ins = vec![In::Rows(Bf16)];
        if self.weighted {
            ins.push(In::Shared(Bf16));
        }
        ins.push(In::Each(I32));
        ins
    }

    /// The weight's width.
    fn weight_width(self) -> usize {
        self.head as usize
    }
}

/// Whether the kernel's row is a head (the planes' rows cut into their
/// heads) or the row whole, its heads within it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Cut {
    Heads,
    Row,
}

/// The case's generated kernel, loaded.
fn generated(case: Case, cut: Cut) -> Synthesized {
    let region = case.region(cut);
    let ins = case.ins();
    let outs = [Out::Rows(Bf16)];
    synth::supports(&region, &ins, &outs).unwrap_or_else(|why| panic!("{}: {why}", case.name));
    let kernel = synth::synth(&region, &ins, 1).expect("synthesized");
    kernel
        .load()
        .expect("the generated kernel compiles and loads");
    kernel
}

/// How each case runs: its two ops, the hand fused kernel, or the
/// generated one, cut into heads or whole.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum By {
    Ops,
    Hand,
    Region(Cut),
    /// The region cut into heads under a schedule of the test's choosing.
    Laid(Schedule),
}

/// The knobs the hand kernels run under: none (the L40S's profile, every
/// chip's default), or the register and warp forms B200's and H200's turn
/// on (`kernel.elemwise.{rows_vec, hc_tile}`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Knobs {
    Default,
    Tuned,
}

impl Knobs {
    const BOTH: [Knobs; 2] = [Knobs::Default, Knobs::Tuned];

    fn tuning(self) -> Tuning {
        match self {
            Knobs::Default => Tuning::DEFAULT,
            Knobs::Tuned => tuned(|t| {
                t.rows_vec = true;
                t.hc_tile = true;
            }),
        }
    }
}

/// One fire's planes: x, the weight, the positions, and the row landed.
#[derive(Clone, Copy)]
struct Planes {
    x: u64,
    y: u64,
}

#[derive(Clone, Copy)]
struct Fixed {
    w: u64,
    positions: u64,
}

fn fire(
    ctx: &Ctx,
    case: Case,
    by: By,
    kernel: Option<Synthesized>,
    rows: usize,
    p: Planes,
    f: Fixed,
) {
    let width = case.width as usize;
    let x = ten(p.x, rows, width, Bf16);
    let mut y = ten(p.y, rows, width, Bf16);
    let w = ten(f.w, 1, case.weight_width(), Bf16);
    let positions = ten(f.positions, rows, 1, I32);
    let row_norm = case.head == case.width;
    match (by, case.span) {
        (By::Region(_) | By::Laid(_), _) => {
            let mut ins = vec![Plane::rows(&x)];
            if case.weighted {
                ins.push(Plane::shared(&w));
            }
            ins.push(Plane::each(&positions));
            let kernel = kernel.expect("a generated kernel");
            let rows = u32::try_from(rows).expect("rows");
            match by {
                By::Region(cut) => kernel.fire_parts(
                    ctx,
                    &ins,
                    &[Plane::rows(&y)],
                    rows,
                    case.width / case.kernel_width(cut),
                ),
                By::Laid(schedule) => kernel.fire_laid(
                    ctx,
                    &ins,
                    &[Plane::rows(&y)],
                    (rows, case.width / case.head),
                    schedule,
                ),
                _ => unreachable!("a region's fire"),
            }
            .expect("the generated kernel fires");
        }
        (By::Hand, Some(_)) => {
            unreachable!("CUDA holds no hand kernel for a turn of the first dims")
        }
        (By::Hand, None) => norm_rope::norm_rope_last(
            ctx,
            x,
            case.weighted.then_some(w),
            case.head,
            case.eps,
            positions,
            case.rotary,
            case.theta,
            true,
            false,
            case.yarn,
            &mut y,
        )
        .expect("the hand kernel fires"),
        (By::Ops, span) => {
            match (case.weighted, row_norm) {
                (true, true) => norm::rmsnorm(ctx, x, w, case.eps, &mut y),
                (true, false) => norm::rmsnorm_per_head(ctx, x, w, case.head, case.eps, &mut y),
                (false, _) => norm::rmsnorm_no_scale(ctx, x, case.head, case.eps, &mut y),
            }
            .expect("the norm fires");
            match span {
                Some(span) => rope::partial_q(
                    ctx,
                    &mut y,
                    positions,
                    case.rotary,
                    case.head,
                    case.theta,
                    span,
                ),
                None => rope::partial_last(
                    ctx,
                    &mut y,
                    positions,
                    case.rotary,
                    case.head,
                    case.theta,
                    true,
                    false,
                    case.yarn,
                ),
            }
            .expect("the rotation fires");
        }
    }
}

/// What `case` lands over a seeded draw of `rows` rows, by `by` under
/// `knobs`.
fn land(case: Case, rows: usize, by: By, knobs: Knobs) -> Vec<u16> {
    land_staged(case, rows, by, knobs, None)
}

/// [`land`], under a staged window of `(live, first)` rows where given.
fn land_staged(
    case: Case,
    rows: usize,
    by: By,
    knobs: Knobs,
    window: Option<[u32; 2]>,
) -> Vec<u16> {
    let width = case.width as usize;
    let (x, w, positions) = drawn(case, rows);
    let mut gpu = Gpu::open();
    let p = Planes {
        x: gpu.up(&x),
        y: gpu.zeros(rows * width * 2),
    };
    let f = Fixed {
        w: gpu.up(&w),
        positions: gpu.up(&positions),
    };
    let kernel = match by {
        By::Region(cut) => Some(generated(case, cut)),
        By::Laid(_) => Some(generated(case, Cut::Heads)),
        _ => None,
    };
    let ctx = gpu.ctx().with_tuning(knobs.tuning());
    if let Some(window) = window {
        ctx.arm_stage(gpu.up(&window));
    }
    fire(&ctx, case, by, kernel, rows, p, f);
    gpu.sync();
    gpu.down(p.y, rows * width)
}

/// A seeded draw of `rows` rows: x, the weight near one, the positions (a
/// long prompt's and its decodes').
fn drawn(case: Case, rows: usize) -> (Vec<u16>, Vec<u16>, Vec<i32>) {
    let width = case.width as usize;
    let mut lcg = Lcg::seeded(0x0d0e_0003 ^ (u64::from(case.width) << 16) ^ rows as u64);
    let x = lcg.row(rows * width).0;
    let w: Vec<u16> = (0..case.weight_width())
        .map(|_| to_bf16(1.0 + 0.25 * lcg.unit()))
        .collect();
    let positions: Vec<i32> = (0..rows)
        .map(|r| ((r * 7919 + 11) % 32768) as i32)
        .collect();
    (x, w, positions)
}

/// The case's region (cut into heads) by the interpreter over the draw of
/// `rows` rows: each row's heads, each its own row of the region.
fn math(case: Case, rows: usize) -> Vec<f32> {
    let (x, w, positions) = drawn(case, rows);
    let head = case.head as usize;
    let parts = case.width as usize / head;
    let region = case.region(Cut::Heads);
    reference(&region, rows * parts, |r| {
        let mut args = vec![row_of(&x, r, head)];
        if case.weighted {
            args.push(Arg::Row(widened(&w)));
        }
        args.push(Arg::Scalar(positions[r / parts] as f32));
        args
    })
    .swap_remove(0)
}

/// Under a staged window (a captured graph's live rows from an offset),
/// each case lands the window's rows as its math has them and leaves the
/// rest.
#[test]
fn a_staged_window_lands_its_rows_alone() {
    for case in CASES {
        let window = Some([23u32, 5]);
        let region = land_staged(case, 64, By::Region(Cut::Heads), Knobs::Default, window);
        let width = case.width as usize;
        let want = math(case, 64);
        assert_within(
            case.name,
            &region[5 * width..28 * width],
            &want[5 * width..28 * width],
            TURN,
        );
        assert!(
            region[..5 * width].iter().all(|&v| v == 0),
            "{}: rows before the window",
            case.name
        );
        assert!(
            region[28 * width..].iter().all(|&v| v == 0),
            "{}: rows past it",
            case.name
        );
        assert!(
            region[5 * width..28 * width].iter().any(|&v| v != 0),
            "{}: the window's rows",
            case.name
        );
    }
}

/// Every case at a decode's, a ragged and a prefill's rows: the generated
/// kernel against its math by the interpreter, within [`TURN`] (the angle
/// through the skeleton's `__sincosf`, a norm's sum in its own order), its
/// heads cut from the row or held within it alike, and two fires the same
/// bits. The ops and the hand fused kernel are shown beside it, not held
/// to it.
#[test]
fn a_norm_and_its_turn_land_their_math() {
    for case in CASES {
        for rows in [1usize, 7, 64] {
            let region = land(case, rows, By::Region(Cut::Heads), Knobs::Default);
            let total = rows * case.width as usize;
            let name = format!("{} ({rows} rows of {})", case.name, case.width);
            deterministic(
                &name,
                &region,
                &land(case, rows, By::Region(Cut::Heads), Knobs::Default),
            );
            let want = math(case, rows);
            let (n, most) = apart(&region, &want);
            eprintln!("{name}: the region {n} of {total} apart from its math, {most} ulps at most");
            assert_within(&name, &region, &want, TURN);
            if case.head != case.width {
                let whole = land(case, rows, By::Region(Cut::Row), Knobs::Default);
                assert_within(&format!("{name}, whole"), &whole, &want, TURN);
            }
            // each schedule the kernel holds (packed where it holds none other)
            for schedule in Schedule::ALL {
                let laid = land(case, rows, By::Laid(schedule), Knobs::Default);
                assert_within(&format!("{name}, {schedule:?}"), &laid, &want, TURN);
            }
            for knobs in Knobs::BOTH {
                // the hand fused kernel where CUDA still holds one: a turn
                // of the tail
                let hand: &[By] = if case.span.is_some() {
                    &[]
                } else {
                    &[By::Hand]
                };
                for &by in [By::Ops].iter().chain(hand) {
                    let (n, most) = apart(&land(case, rows, by, knobs), &want);
                    eprintln!(
                        "{name}: {by:?} {knobs:?}: {n} of {total} apart from the math, {most} ulps at most"
                    );
                }
            }
        }
    }
}
