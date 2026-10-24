#![cfg(feature = "cuda")]

//! A generated row kernel (`kernels_cuda::synth`) lands what the region it
//! was generated from states: eta's bodies for the chains poem lowers, and
//! the hand `residual_add_rmsnorm` body passed through the same path, each
//! against the hand kernels.

use crate::common::attn_ref::{ten, tuned};
use crate::common::{Gpu, Lcg, captured, forget, from_bf16, replay, to_bf16};
use dtype::Dtype::Bf16;
use eta_compiler::plan::{Dimension, SymbolicType};
use eta_ir::op::Op;
use eta_ir::types::Literal;
use kernels_cuda::elemwise::{norm_residual, pointwise};
use kernels_cuda::jit::Ctx;
use kernels_cuda::synth::{self, Body, In, Out, Plane, Region};

const EPS: f32 = 1e-6;

const WIDTHS: [usize; 6] = [4096, 7168, 8192, 2048, 1544, 6152];
const ROWS: [usize; 3] = [1, 7, 64];

/// The hand kernels' register forms (`kernel.elemwise.rows_vec`).
fn hand(gpu: &Gpu) -> Ctx {
    gpu.ctx().with_tuning(tuned(|t| t.rows_vec = true))
}

fn row(width: usize) -> SymbolicType {
    SymbolicType {
        dtype: eta_ir::Dtype::F32,
        dims: vec![Dimension::Static(u32::try_from(width).expect("a width"))],
    }
}

/// `h = y + x` and `rmsnorm(h) * w`, as poem lowers a residual norm:
/// parameters x, y, w; results the stream h and the normed row.
fn residual_norm(width: usize) -> Region {
    Region {
        params: vec![row(width); 3],
        ops: vec![
            Op::Add(1, 0),
            Op::Mul(3, 3),
            Op::ReduceSum(4),
            Op::Const(Literal::F32(width as f32)),
            Op::Div(5, 6),
            Op::Const(Literal::F32(EPS)),
            Op::Add(7, 8),
            Op::Rsqrt(9),
            Op::Mul(3, 10),
            Op::Mul(11, 2),
        ],
        results: vec![3, 12],
    }
}

/// The Phase 1 hand body (`elemwise/frame.cu`), its eps a literal, in a
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
    }
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

/// `body` loaded for `ins` parameters and `outs` results at `width`.
fn loaded(body: &Body, ins: usize, outs: usize, width: usize) -> synth::Synthesized {
    let kernel =
        synth::synth(body, ins, outs, u32::try_from(width).expect("a width")).expect("synthesized");
    kernel
        .load()
        .expect("the generated kernel compiles and loads");
    kernel
}

/// `region`'s body, the frame taking it with every plane a bf16 row but
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
    synth::supports(region, &ins, &outs).expect("the frame takes the region");
    region.body().expect("the stream core writes the region")
}

/// Seeded bf16 planes: `rows` rows of `width` each, a weight row last.
fn draw(rows: usize, width: usize, planes: usize) -> (Vec<Vec<u16>>, Vec<u16>) {
    let mut lcg = Lcg::seeded(0x5e7d ^ ((width as u64) << 8) ^ rows as u64);
    let rows: Vec<Vec<u16>> = (0..planes).map(|_| lcg.row(rows * width).0).collect();
    (rows, lcg.row(width).0)
}

/// The hand register kernel's stream and normed row.
fn hand_residual_norm(rows: usize, width: usize) -> (Vec<u16>, Vec<u16>) {
    let n = rows * width;
    let (planes, w) = draw(rows, width, 2);
    let mut gpu = Gpu::open();
    let (x, y, w, out) = (
        gpu.up(&planes[0]),
        gpu.up(&planes[1]),
        gpu.up(&w),
        gpu.zeros(n * 2),
    );
    norm_residual::residual_add_rmsnorm(
        &hand(&gpu),
        ten(x, rows, width, Bf16),
        &mut ten(y, rows, width, Bf16),
        ten(w, width, 1, Bf16),
        false,
        EPS,
        &mut ten(out, rows, width, Bf16),
    )
    .expect("the hand residual norm fires");
    gpu.sync();
    (gpu.down(y, n), gpu.down(out, n))
}

#[test]
fn the_hand_body_through_synth_lands_the_register_kernels_bits() {
    let body = hand_body();
    for width in WIDTHS {
        let kernel = loaded(&body, 3, 2, width);
        for rows in ROWS {
            let n = rows * width;
            let (planes, w) = draw(rows, width, 2);
            let mut gpu = Gpu::open();
            let (x, y, w, out) = (
                gpu.up(&planes[0]),
                gpu.up(&planes[1]),
                gpu.up(&w),
                gpu.zeros(n * 2),
            );
            let plane = |at| Plane::rows(&ten(at, rows, width, Bf16));
            // The stream summed in place, as the hand kernel sums it.
            kernel
                .fire(
                    &gpu.ctx(),
                    &[plane(x), plane(y), Plane::shared(&ten(w, 1, width, Bf16))],
                    &[plane(out), plane(y)],
                    rows as u32,
                )
                .expect("the hand body fires");
            gpu.sync();
            let (y_want, out_want) = hand_residual_norm(rows, width);
            let case = format!("{rows} rows of {width}");
            assert!(gpu.down::<u16>(y, n) == y_want, "{case}: the summed stream");
            assert!(
                gpu.down::<u16>(out, n) == out_want,
                "{case}: the normed row"
            );
        }
    }
}

/// eta's residual norm against the hand register kernel. The stream is
/// the same sum and lands the same bits. The normed row differs only in
/// how its squares are summed: eta adds them one at a time (`a += h²`),
/// the hand kernel two at a time (`a += h₀² + h₁²`), so a sum may differ in
/// its last f32 bit and, through `rsqrt`, a normed element by one bf16 ulp.
#[test]
fn etas_residual_norm_lands_the_hand_stream_and_its_norm_within_an_ulp() {
    for width in WIDTHS {
        let body = body_of(&residual_norm(width), &[2]);
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
            kernel
                .fire(
                    &gpu.ctx(),
                    &[plane(x), plane(y), Plane::shared(&ten(w, 1, width, Bf16))],
                    &[plane(h), plane(out)],
                    rows as u32,
                )
                .expect("eta's body fires");
            gpu.sync();
            let (y_want, out_want) = hand_residual_norm(rows, width);
            let case = format!("{rows} rows of {width}");
            assert!(gpu.down::<u16>(h, n) == y_want, "{case}: the stream");
            let got = gpu.down::<u16>(out, n);
            let apart: Vec<(usize, u16, u16)> = got
                .iter()
                .zip(&out_want)
                .enumerate()
                .filter(|(_, (a, b))| a != b)
                .map(|(at, (a, b))| (at, *a, *b))
                .collect();
            for &(at, a, b) in &apart {
                assert!(
                    a.abs_diff(b) <= 1,
                    "{case}: normed element {at} is {} and the hand kernel's {}",
                    from_bf16(a),
                    from_bf16(b)
                );
            }
            eprintln!(
                "{case}: {} of {n} normed elements an ulp apart",
                apart.len()
            );
        }
    }
}

/// `z = x + y` and `o = z * g`, both leaving: what the hand `add` and `mul`
/// land, bit for bit, since the body multiplies the sum as stored.
#[test]
fn etas_add_then_mul_lands_the_hand_kernels_bits() {
    for width in WIDTHS {
        let region = Region {
            params: vec![row(width); 3],
            ops: vec![Op::Add(0, 1), Op::Mul(3, 2)],
            results: vec![3, 4],
        };
        let kernel = loaded(&body_of(&region, &[]), 3, 2, width);
        for rows in ROWS {
            let n = rows * width;
            let (planes, _) = draw(rows, width, 3);
            let mut gpu = Gpu::open();
            let ins: Vec<u64> = planes.iter().map(|p| gpu.up(p)).collect();
            let (z, o) = (gpu.zeros(n * 2), gpu.zeros(n * 2));
            let (z_hand, o_hand) = (gpu.zeros(n * 2), gpu.zeros(n * 2));
            let t = |at| ten(at, rows, width, Bf16);
            let ctx = hand(&gpu);
            kernel
                .fire(
                    &ctx,
                    &[
                        Plane::rows(&t(ins[0])),
                        Plane::rows(&t(ins[1])),
                        Plane::rows(&t(ins[2])),
                    ],
                    &[Plane::rows(&t(z)), Plane::rows(&t(o))],
                    rows as u32,
                )
                .expect("eta's body fires");
            pointwise::add(&ctx, t(ins[0]), t(ins[1]), &mut t(z_hand)).expect("add fires");
            pointwise::mul(&ctx, t(z_hand), t(ins[2]), &mut t(o_hand)).expect("mul fires");
            gpu.sync();
            let case = format!("{rows} rows of {width}");
            assert!(
                gpu.down::<u16>(z, n) == gpu.down::<u16>(z_hand, n),
                "{case}: the sum"
            );
            assert!(
                gpu.down::<u16>(o, n) == gpu.down::<u16>(o_hand, n),
                "{case}: the product"
            );
        }
    }
}

/// `y + x + z` with the partial sum only the chain reads (the double
/// residual of a hyper-connected block): held in f32, never rounded, so the
/// kernel lands `bf16((y + x) + z)` in f32 bit for bit, which two hand
/// `residual_add`s, rounding between, do not.
#[test]
fn etas_double_residual_rounds_once() {
    for width in WIDTHS {
        let region = Region {
            params: vec![row(width); 3],
            ops: vec![Op::Add(1, 0), Op::Add(3, 2)],
            results: vec![4],
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
                .expect("eta's body fires");
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
        params: vec![row(width); 2],
        ops: vec![Op::Sub(0, 1)],
        results: vec![2],
    };
    let kernel = synth::synth(&body_of(&region, &[]), 2, 1, width as u32).expect("synthesized");
    let mut gpu = Gpu::open();
    let at = gpu.zeros(width * 2);
    let plane = Plane::rows(&ten(at, 1, width, Bf16));
    let refused = kernel
        .fire(&gpu.ctx(), &[plane, plane], &[plane], 1)
        .expect_err("a fire does not load a module");
    assert!(refused.to_string().contains("not loaded"), "{refused}");
}

/// eta's residual norm, generated, against the hand register kernel at a
/// decode's and a prefill's rows over the widths of 4096-, 7168- and
/// 8192-wide models, each fire's bytes against the device's copy rate:
/// `cargo test ... -- --ignored synthesized_residual_norm_rate --nocapture`.
/// A fire is its graph's fastest replay over its fires; the planes rotate
/// over at least 256 MB so a fire reads HBM, not the L2.
#[test]
#[ignore = "a rate bench: cargo test ... -- --ignored synthesized_residual_norm_rate --nocapture"]
fn synthesized_residual_norm_rate() {
    use super::frame::{copy_rate, fastest};
    const FOOTPRINT: usize = 256 << 20;
    const FIRES: usize = 256;
    const REPLAYS: u32 = 20;
    let copy = copy_rate();
    eprintln!("device copy: {copy:.0} GB/s");
    eprintln!("rows  width  hand us  %copy  synth us  %copy  synth/hand");
    for rows in [1usize, 8, 64, 512, 4096] {
        for width in [4096usize, 7168, 8192] {
            let kernel = loaded(&body_of(&residual_norm(width), &[2]), 3, 2, width);
            let mut gpu = Gpu::open();
            let plane = rows * width * 2;
            let copies = FOOTPRINT.div_ceil(4 * plane).clamp(1, FIRES);
            let w = ten(gpu.zeros(width * 2), width, 1, Bf16);
            let sets: Vec<[u64; 4]> = (0..copies)
                .map(|_| {
                    [
                        gpu.zeros(plane),
                        gpu.zeros(plane),
                        gpu.zeros(plane),
                        gpu.zeros(plane),
                    ]
                })
                .collect();
            let fires = FIRES.max(copies) / copies * copies;
            let ctx = hand(&gpu);
            let t = |at| ten(at, rows, width, Bf16);
            let time = |fire: &dyn Fn([u64; 4])| {
                let all = || {
                    for i in 0..fires {
                        fire(sets[i % copies]);
                    }
                };
                all();
                gpu.sync();
                let exec = captured(&ctx, all);
                replay(&ctx, exec);
                gpu.sync();
                let us = fastest(REPLAYS, || {
                    replay(&ctx, exec);
                    gpu.sync();
                });
                forget(exec);
                us / fires as f64
            };
            let hand_us = time(&|[x, y, out, _]| {
                norm_residual::residual_add_rmsnorm(
                    &ctx,
                    t(x),
                    &mut t(y),
                    w,
                    false,
                    EPS,
                    &mut t(out),
                )
                .expect("the hand residual norm fires");
            });
            let synth_us = time(&|[x, y, out, h]| {
                kernel
                    .fire(
                        &ctx,
                        &[Plane::rows(&t(x)), Plane::rows(&t(y)), Plane::shared(&w)],
                        &[Plane::rows(&t(h)), Plane::rows(&t(out))],
                        rows as u32,
                    )
                    .expect("the generated residual norm fires");
            });
            // x and y read, the stream and the normed row written, the
            // weight read
            let bytes = (4 * plane + width * 2) as f64;
            let share = |us: f64| 100.0 * bytes / us / 1e3 / copy;
            eprintln!(
                "{rows:>4}  {width:>5}  {hand_us:>7.2}  {:>4.0}%  {synth_us:>8.2}  {:>4.0}%  {:>10.3}",
                share(hand_us),
                share(synth_us),
                synth_us / hand_us,
            );
        }
    }
}
