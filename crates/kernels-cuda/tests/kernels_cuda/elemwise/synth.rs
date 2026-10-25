#![cfg(feature = "cuda")]

//! A generated row kernel (`kernels_cuda::synth`) lands what the region it
//! was generated from states (`super::within`'s contract): the printed
//! bodies of the regions poem lowers against the interpreter, and the
//! skeleton's hand residual norm body passed through the same path, against
//! the skeleton's own fire of it.

use crate::common::attn_ref::{ten, tuned};
use crate::common::{Gpu, Lcg, captured, forget, from_bf16, replay, to_bf16};
use dtype::Dtype::Bf16;
use kernels_cuda::elemwise::skeleton;
use kernels_cuda::jit::Ctx;
use kernels_cuda::synth::{self, Body, In, Out, Plane, Region};
use kernels_ir::interp::Arg;
use kernels_ir::{Binary, Literal, Op, Reduce, Shape, Unary};

use super::within::{EXACT, NORM, assert_within, deterministic, reference, row_of, widened};

const EPS: f32 = 1e-6;

const WIDTHS: [usize; 6] = [4096, 7168, 8192, 2048, 1544, 6152];
const ROWS: [usize; 3] = [1, 7, 64];

/// The unfused kernels' register forms (`kernel.elemwise.rows_vec`).
fn hand(gpu: &Gpu) -> Ctx {
    gpu.ctx().with_tuning(tuned(|t| t.rows_vec = true))
}

fn w(width: usize) -> u32 {
    u32::try_from(width).expect("a width")
}

/// `h = y + x` and `rmsnorm(h) * w`, as poem lowers a residual norm:
/// parameters x, y, w; results the stream h and the normed row.
fn residual_norm(width: usize) -> Region {
    Region {
        width: w(width),
        params: vec![Shape::Row; 3],
        ops: vec![
            Op::Binary(Binary::Add, 1, 0),
            Op::Binary(Binary::Mul, 3, 3),
            Op::Reduce(Reduce::Sum, 4),
            Op::Const(Literal::F32(width as f32)),
            Op::Binary(Binary::Div, 5, 6),
            Op::Const(Literal::F32(EPS)),
            Op::Binary(Binary::Add, 7, 8),
            Op::Unary(Unary::Rsqrt, 9),
            Op::Binary(Binary::Mul, 3, 10),
            Op::Binary(Binary::Mul, 11, 2),
        ],
        results: vec![3, 12],
        rounds: vec![(3, Bf16), (12, Bf16)],
    }
}

/// The Phase 1 hand body (`elemwise/skeleton.cu`), its eps a literal, in a
/// generated body's form: parameters x, y, w; results the normed row and
/// the stream.
fn hand_body() -> Body {
    let source = format!(
        "struct Body {{
  template <class F>
  __device__ __forceinline__ void operator()(F& f) const {{
    float h[F::N];
    float ss = 0.f;
#pragma unroll
    for (int e = 0; e < F::N; e += 2) {{
      h[e] = f.st(1, e, f.ld(1, e) + f.ld(0, e));
      h[e + 1] = f.st(1, e + 1, f.ld(1, e + 1) + f.ld(0, e + 1));
      if (f.valid(e)) ss += h[e] * h[e] + h[e + 1] * h[e + 1];
    }}
    f.template flush<1>();
    const float r = rsqrtf(f.reduce_sum(ss) / static_cast<float>(f.width) + {EPS:?}f);
#pragma unroll
    for (int e = 0; e < F::N; ++e) f.st(0, e, h[e] * r * f.ld(2, e));
    f.template flush<0>();
  }}
}};
"
    );
    Body {
        hash: fnv1a64(source.as_bytes()),
        source,
        passes: 2,
        shares: false,
        table: 0,
    }
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

/// `body` loaded for `ins` parameters and `outs` results at `width`.
fn loaded(body: &Body, ins: usize, outs: usize, width: usize) -> synth::Synthesized {
    let kernel = synth::synth_body(
        body,
        &vec![In::Rows(Bf16); ins],
        outs,
        u32::try_from(width).expect("a width"),
    )
    .expect("synthesized");
    kernel
        .load()
        .expect("the generated kernel compiles and loads");
    kernel
}

/// `region`'s body, the skeleton taking it with every plane a bf16 row but
/// those `shared` names.
fn body_of(region: &Region, shared: &[usize]) -> Body {
    let ins: Vec<In> = (0..region.params.len())
        .map(|k| {
            if shared.contains(&k) {
                In::Shared(Bf16)
            } else {
                In::Rows(Bf16)
            }
        })
        .collect();
    let outs = vec![Out::Rows(Bf16); region.results.len()];
    synth::supports(region, &ins, &outs).expect("the skeleton takes the region");
    Body::of(region).expect("the row core schedules the region")
}

/// Seeded bf16 planes: `rows` rows of `width` each, a weight row last.
fn draw(rows: usize, width: usize, planes: usize) -> (Vec<Vec<u16>>, Vec<u16>) {
    let mut lcg = Lcg::seeded(0x5e7d ^ ((width as u64) << 8) ^ rows as u64);
    let rows: Vec<Vec<u16>> = (0..planes).map(|_| lcg.row(rows * width).0).collect();
    (rows, lcg.row(width).0)
}

/// The stream and normed row `fire` lands over the draw of `rows` rows of
/// `width`: the skeleton's hand body or the chain's unfused ops.
fn residual_norm_by(
    fire: super::skeleton::Fire,
    rows: usize,
    width: usize,
) -> (Vec<u16>, Vec<u16>) {
    let n = rows * width;
    let (planes, w) = draw(rows, width, 2);
    let mut gpu = Gpu::open();
    let (x, y, w, out) = (
        gpu.up(&planes[0]),
        gpu.up(&planes[1]),
        gpu.up(&w),
        gpu.zeros(n * 2),
    );
    fire(
        &hand(&gpu),
        ten(x, rows, width, Bf16),
        &mut ten(y, rows, width, Bf16),
        ten(w, width, 1, Bf16),
        false,
        EPS,
        &mut ten(out, rows, width, Bf16),
    )
    .expect("the residual norm fires");
    gpu.sync();
    (gpu.down(y, n), gpu.down(out, n))
}

#[test]
fn the_hand_body_through_synth_lands_the_skeletons_bits() {
    let body = hand_body();
    for width in WIDTHS {
        let kernel = loaded(&body, 3, 2, width);
        for rows in ROWS {
            let n = rows * width;
            let (planes, w) = draw(rows, width, 2);
            let mut gpu = Gpu::open();
            let (x, y, w, h, out) = (
                gpu.up(&planes[0]),
                gpu.up(&planes[1]),
                gpu.up(&w),
                gpu.zeros(n * 2),
                gpu.zeros(n * 2),
            );
            let plane = |at| Plane::rows(&ten(at, rows, width, Bf16));
            // The stream into a plane of its own: a generated kernel reads
            // its parameters read-only, so no result lands over one.
            let ins = [plane(x), plane(y), Plane::shared(&ten(w, 1, width, Bf16))];
            let refused = kernel
                .fire(&gpu.ctx(), &ins, &[plane(out), plane(y)], rows as u32)
                .expect_err("a result over a parameter is refused");
            assert!(
                refused.to_string().contains("over a parameter"),
                "{refused}"
            );
            kernel
                .fire(&gpu.ctx(), &ins, &[plane(out), plane(h)], rows as u32)
                .expect("the hand body fires");
            gpu.sync();
            let (y_want, out_want) = residual_norm_by(skeleton::residual_add_rmsnorm, rows, width);
            let case = format!("{rows} rows of {width}");
            assert!(gpu.down::<u16>(h, n) == y_want, "{case}: the summed stream");
            assert!(
                gpu.down::<u16>(out, n) == out_want,
                "{case}: the normed row"
            );
        }
    }
}

/// The residual norm's region against the interpreter: the stream is IEEE
/// adds and lands the reference's bits; the norm sums its squares in the
/// skeleton's order, within [`NORM`] of the reference; two fires land the same
/// bits. The unfused ops are shown beside it, not held to it.
#[test]
fn a_residual_norms_region_lands_its_math() {
    for width in WIDTHS {
        let region = residual_norm(width);
        let kernel = loaded(&body_of(&region, &[2]), 3, 2, width);
        for rows in ROWS {
            let n = rows * width;
            let (planes, w) = draw(rows, width, 2);
            let fire = || {
                let mut gpu = Gpu::open();
                let (x, y, w, h, out) = (
                    gpu.up(&planes[0]),
                    gpu.up(&planes[1]),
                    gpu.up(&w),
                    gpu.zeros(n * 2),
                    gpu.zeros(n * 2),
                );
                let plane = |at| Plane::rows(&ten(at, rows, width, Bf16));
                kernel
                    .fire(
                        &gpu.ctx(),
                        &[plane(x), plane(y), Plane::shared(&ten(w, 1, width, Bf16))],
                        &[plane(h), plane(out)],
                        rows as u32,
                    )
                    .expect("the printed body fires");
                gpu.sync();
                (gpu.down::<u16>(h, n), gpu.down::<u16>(out, n))
            };
            let (h, out) = fire();
            let case = format!("{rows} rows of {width}");
            deterministic(&case, &out, &fire().1);
            let want = reference(&region, rows, |r| {
                vec![
                    row_of(&planes[0], r, width),
                    row_of(&planes[1], r, width),
                    Arg::Row(widened(&w)),
                ]
            });
            assert_within(&format!("{case}: the stream"), &h, &want[0], EXACT);
            assert_within(&format!("{case}: the normed row"), &out, &want[1], NORM);
            let (_, unfused) = residual_norm_by(super::skeleton::unfused, rows, width);
            let apart = out.iter().zip(&unfused).filter(|(a, b)| a != b).count();
            eprintln!("{case}: {apart} of {n} normed elements apart from the unfused ops");
        }
    }
}

/// `z = x + y` and `o = z * g`, both leaving: IEEE ops, the reference's
/// bits, the product of the sum as stored.
#[test]
fn an_add_then_mul_region_lands_its_math() {
    for width in WIDTHS {
        let region = Region {
            width: w(width),
            params: vec![Shape::Row; 3],
            ops: vec![Op::Binary(Binary::Add, 0, 1), Op::Binary(Binary::Mul, 3, 2)],
            results: vec![3, 4],
            rounds: vec![(3, Bf16)],
        };
        let kernel = loaded(&body_of(&region, &[]), 3, 2, width);
        for rows in ROWS {
            let n = rows * width;
            let (planes, _) = draw(rows, width, 3);
            let mut gpu = Gpu::open();
            let ins: Vec<u64> = planes.iter().map(|p| gpu.up(p)).collect();
            let (z, o) = (gpu.zeros(n * 2), gpu.zeros(n * 2));
            let t = |at| ten(at, rows, width, Bf16);
            kernel
                .fire(
                    &gpu.ctx(),
                    &[
                        Plane::rows(&t(ins[0])),
                        Plane::rows(&t(ins[1])),
                        Plane::rows(&t(ins[2])),
                    ],
                    &[Plane::rows(&t(z)), Plane::rows(&t(o))],
                    rows as u32,
                )
                .expect("the printed body fires");
            gpu.sync();
            let want = reference(&region, rows, |r| {
                (0..3).map(|k| row_of(&planes[k], r, width)).collect()
            });
            let case = format!("{rows} rows of {width}");
            assert_within(
                &format!("{case}: the sum"),
                &gpu.down::<u16>(z, n),
                &want[0],
                EXACT,
            );
            assert_within(
                &format!("{case}: the product"),
                &gpu.down::<u16>(o, n),
                &want[1],
                EXACT,
            );
        }
    }
}

/// `y + x + z` with the partial sum only the chain reads (the double
/// residual of a hyper-connected block): held in f32, never rounded, so the
/// kernel lands `bf16((y + x) + z)` in f32 bit for bit, which two hand
/// `residual_add`s, rounding between, do not.
#[test]
fn a_double_residual_region_rounds_once() {
    for width in WIDTHS {
        let region = Region {
            width: w(width),
            params: vec![Shape::Row; 3],
            ops: vec![Op::Binary(Binary::Add, 1, 0), Op::Binary(Binary::Add, 3, 2)],
            results: vec![4],
            rounds: vec![],
        };
        let kernel = loaded(&body_of(&region, &[]), 3, 1, width);
        for rows in ROWS {
            let n = rows * width;
            let (planes, _) = draw(rows, width, 3);
            let mut gpu = Gpu::open();
            let ins: Vec<u64> = planes.iter().map(|p| gpu.up(p)).collect();
            let out = gpu.zeros(n * 2);
            let t = |at| Plane::rows(&ten(at, rows, width, Bf16));
            kernel
                .fire(
                    &gpu.ctx(),
                    &[t(ins[0]), t(ins[1]), t(ins[2])],
                    &[t(out)],
                    rows as u32,
                )
                .expect("the printed body fires");
            gpu.sync();
            let want: Vec<u16> = (0..n)
                .map(|e| {
                    let [x, y, z] = [0, 1, 2].map(|k| from_bf16(planes[k][e]));
                    to_bf16((y + x) + z)
                })
                .collect();
            assert!(gpu.down::<u16>(out, n) == want, "{rows} rows of {width}");
        }
    }
}

/// A generated kernel captured into a graph replays what it fired eagerly,
/// over the stage window's live rows alone.
#[test]
fn a_captured_synthesized_kernel_replays_its_eager_fire() {
    let (rows, width) = (64usize, 4096usize);
    let n = rows * width;
    let body = body_of(&residual_norm(width), &[2]);
    let kernel = loaded(&body, 3, 2, width);
    let (planes, w) = draw(rows, width, 2);
    let mut gpu = Gpu::open();
    let (x, y, w) = (gpu.up(&planes[0]), gpu.up(&planes[1]), gpu.up(&w));
    let (h, out) = (gpu.zeros(n * 2), gpu.zeros(n * 2));
    let (h_eager, out_eager) = (gpu.zeros(n * 2), gpu.zeros(n * 2));
    let plane = |at| Plane::rows(&ten(at, rows, width, Bf16));
    let ins = [plane(x), plane(y), Plane::shared(&ten(w, 1, width, Bf16))];
    let ctx = gpu.ctx();
    kernel
        .fire(&ctx, &ins, &[plane(h_eager), plane(out_eager)], rows as u32)
        .expect("the eager fire");
    // Rows 8.. of 40 live from row 8: a staged window of the graph's rows.
    let win = gpu.up(&[40u32, 8u32]);
    ctx.arm_stage(win);
    let exec = captured(&ctx, || {
        kernel
            .fire(&ctx, &ins, &[plane(h), plane(out)], rows as u32)
            .expect("the captured fire");
    });
    ctx.disarm_stage();
    replay(&ctx, exec);
    gpu.sync();
    forget(exec);
    let live = 8 * width..48 * width;
    for (got, want, what) in [(h, h_eager, "stream"), (out, out_eager, "normed row")] {
        let (got, want) = (gpu.down::<u16>(got, n), gpu.down::<u16>(want, n));
        assert!(
            got[live.clone()] == want[live.clone()],
            "the window's {what}"
        );
        assert!(
            got[..live.start]
                .iter()
                .chain(&got[live.end..])
                .all(|&v| v == 0),
            "the {what} past the window is left as it was"
        );
    }
}

#[test]
fn a_kernel_never_loaded_is_refused_at_its_fire() {
    let width = 2048usize;
    // A body of its own, so no other test has loaded it.
    let region = Region {
        width: w(width),
        params: vec![Shape::Row; 2],
        ops: vec![Op::Binary(Binary::Sub, 0, 1)],
        results: vec![2],
        rounds: vec![],
    };
    let kernel = synth::synth(&region, &[In::Rows(Bf16); 2], 1).expect("synthesized");
    let mut gpu = Gpu::open();
    let (at, to) = (gpu.zeros(width * 2), gpu.zeros(width * 2));
    let plane = Plane::rows(&ten(at, 1, width, Bf16));
    let out = Plane::rows(&ten(to, 1, width, Bf16));
    let refused = kernel
        .fire(&gpu.ctx(), &[plane, plane], &[out], 1)
        .expect_err("a fire does not load a module");
    assert!(refused.to_string().contains("not loaded"), "{refused}");
}
