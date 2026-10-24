//! The modulate forms (scale-shift, scale, tanh gate, gated residual)
//! answer the host with their modulation read per token or per lane, and the
//! fused norm-modulate pair lands what its two launches land under every
//! norm; kernels-cuda's cases and tolerances.

#![cfg(feature = "rocm")]

mod common;

use common::{Gpu, bf16, bf16s};
use dtype::Dtype;
use kernels_rocm::elemwise::modulate::{self, NormKind};
use kernels_rocm::elemwise::norm;
use kernels_rocm::tensor::Tensor;

const ROWS: usize = 37;
const WIDTH: usize = 128;
const LANES: usize = 4;

fn close(got: f32, want: f32, what: &str, at: usize) {
    assert!(
        (got - want).abs() <= 1e-6 + want.abs() / 256.0,
        "{what}[{at}]: {got} against {want}"
    );
}

fn check(lane_of_row: Option<Vec<i32>>) {
    let m_rows = if lane_of_row.is_some() { LANES } else { ROWS };
    let x = bf16s(ROWS * WIDTH, 0xd17, 1.0);
    let y = bf16s(ROWS * WIDTH, 0xd18, 1.0);
    let r = bf16s(ROWS * WIDTH, 0xd19, 1.0);
    let m = bf16s(m_rows * 2 * WIDTH, 0xd1a, 1.0);
    let g = bf16s(m_rows * WIDTH, 0xd1b, 1.0);

    let Some(gpu) = Gpu::open() else { return };
    let rect = |v: &[f32]| gpu.tensor(v, ROWS as u32, WIDTH as u32, Dtype::Bf16);
    let (xt, yt, rt) = (rect(&x), rect(&y), rect(&r));
    let m_wide = gpu.tensor(&m, m_rows as u32, 2 * WIDTH as u32, Dtype::Bf16);
    let m_tight = gpu.tensor(&g, m_rows as u32, WIDTH as u32, Dtype::Bf16);
    let map = lane_of_row.as_ref().map(|map| gpu.column(map));
    let mut out = [(); 4].map(|()| gpu.zeros(ROWS as u32, WIDTH as u32, Dtype::Bf16));
    let ctx = gpu.ctx();

    modulate::scale_shift(&ctx, xt, m_wide, map, &mut out[0]).expect("fires");
    modulate::scale(&ctx, xt, m_tight, map, &mut out[1]).expect("fires");
    modulate::tanh_gate(&ctx, xt, m_tight, map, &mut out[2]).expect("fires");
    modulate::gated_residual_add(&ctx, rt, m_tight, yt, map, &mut out[3]).expect("fires");

    let got: Vec<Vec<f32>> = out.iter().map(|o| gpu.read(o)).collect();
    let source: Vec<usize> = (0..ROWS)
        .map(|row| lane_of_row.as_ref().map_or(row, |map| map[row] as usize))
        .collect();
    for row in 0..ROWS {
        for col in 0..WIDTH {
            let at = row * WIDTH + col;
            let s = m[source[row] * 2 * WIDTH + col];
            let b = m[source[row] * 2 * WIDTH + WIDTH + col];
            let gv = g[source[row] * WIDTH + col];
            close(got[0][at], x[at].mul_add(1.0 + s, b), "scale_shift", at);
            close(got[1][at], x[at] * (1.0 + gv), "scale", at);
            close(got[2][at], gv.tanh() * x[at], "tanh_gate", at);
            close(
                got[3][at],
                gv.mul_add(y[at], r[at]),
                "gated_residual_add",
                at,
            );
        }
    }
}

#[test]
fn the_modulate_forms_answer_the_broadcast_and_the_per_token_reference_every_case() {
    let lanes: Vec<i32> = (0..ROWS).map(|r| (r % LANES) as i32).collect();
    check(Some(lanes));
    check(None);
}

#[test]
fn a_modulation_may_write_the_rectangle_it_read() {
    let x = bf16s(ROWS * WIDTH, 0xa11a5, 1.0);
    let m = bf16s(ROWS * 2 * WIDTH, 0xa11a6, 1.0);
    let Some(gpu) = Gpu::open() else { return };
    let mut xt = gpu.tensor(&x, ROWS as u32, WIDTH as u32, Dtype::Bf16);
    let mt = gpu.tensor(&m, ROWS as u32, 2 * WIDTH as u32, Dtype::Bf16);
    modulate::scale_shift(&gpu.ctx(), xt, mt, None, &mut xt).expect("fires");

    let got: Vec<u16> = gpu.read_as(&xt);
    for row in 0..ROWS {
        for col in 0..WIDTH {
            let at = row * WIDTH + col;
            let s = m[row * 2 * WIDTH + col];
            let b = m[row * 2 * WIDTH + WIDTH + col];
            assert_eq!(got[at], bf16(x[at].mul_add(1.0 + s, b)), "in place at {at}");
        }
    }
}

const NORM_ROWS: usize = 24;
const NORM_WIDTH: usize = 256;
const EPS: f32 = 1e-6;

#[derive(Clone, Copy)]
enum Which {
    LayerNorm,
    RmsNoScale,
    RmsWeighted,
}

fn near(got: f32, want: f32, scale: f32, what: &str, at: usize) {
    assert!(
        (got - want).abs() <= (want.abs() + scale) / 128.0 + 1e-6,
        "{what} at {at}: {got} against {want}"
    );
}

fn fused(which: Which, name: &str) {
    let r = bf16s(NORM_ROWS * NORM_WIDTH, 0xf05ed, 1.0);
    let y = bf16s(NORM_ROWS * NORM_WIDTH, 0xf05ee, 1.0);
    let g = bf16s(NORM_ROWS * NORM_WIDTH, 0xf05ef, 1.0);
    let m = bf16s(NORM_ROWS * 2 * NORM_WIDTH, 0xf05f0, 1.0);
    let w = bf16s(NORM_WIDTH, 0xf05f1, 1.0);

    let Some(gpu) = Gpu::open() else { return };
    let rect = |v: &[f32]| gpu.tensor(v, NORM_ROWS as u32, NORM_WIDTH as u32, Dtype::Bf16);
    let blank = || gpu.zeros(NORM_ROWS as u32, NORM_WIDTH as u32, Dtype::Bf16);
    let (yt, gt) = (rect(&y), rect(&g));
    let (r_still, mut r_chain, mut r_fused) = (rect(&r), rect(&r), rect(&r));
    let (mut normed, mut o_unfused, mut o_norm, mut o_chain, mut o_fused) =
        (blank(), blank(), blank(), blank(), blank());
    let mt = gpu.tensor(&m, NORM_ROWS as u32, 2 * NORM_WIDTH as u32, Dtype::Bf16);
    let weight: Tensor = gpu.tensor(&w, NORM_WIDTH as u32, 1, Dtype::Bf16);
    let kind = match which {
        Which::LayerNorm => NormKind::LayerNormNoAffine { eps: EPS },
        Which::RmsNoScale => NormKind::RmsNormNoScale { eps: EPS },
        Which::RmsWeighted => NormKind::RmsNorm { weight, eps: EPS },
    };
    let ctx = gpu.ctx();

    match which {
        Which::LayerNorm => norm::layernorm_no_scale(&ctx, r_still, EPS, &mut normed),
        Which::RmsNoScale => norm::rmsnorm_no_scale(&ctx, r_still, 0, EPS, &mut normed),
        Which::RmsWeighted => norm::rmsnorm(&ctx, r_still, weight, EPS, &mut normed),
    }
    .expect("the norm fires");
    modulate::scale_shift(&ctx, normed, mt, None, &mut o_unfused).expect("fires");
    modulate::norm_modulate(&ctx, r_still, mt, None, kind, &mut o_norm)
        .expect("the fused norm fires");

    modulate::gated_residual_add(&ctx, r_chain, gt, yt, None, &mut r_chain).expect("fires");
    modulate::norm_modulate(&ctx, r_chain, mt, None, kind, &mut o_chain).expect("fires");
    modulate::gated_residual_norm_modulate(
        &ctx,
        r_fused,
        gt,
        yt,
        mt,
        None,
        kind,
        &mut r_fused,
        &mut o_fused,
    )
    .expect("the pair fires");

    let want_r: Vec<u16> = gpu.read_as(&r_chain);
    let got_r: Vec<u16> = gpu.read_as(&r_fused);
    assert_eq!(
        got_r, want_r,
        "{name}: the residual is not the chain's bits"
    );

    let unfused = gpu.read(&o_unfused);
    let fused_norm = gpu.read(&o_norm);
    let normed_row = gpu.read(&normed);
    let want_o = gpu.read(&o_chain);
    let got_o = gpu.read(&o_fused);
    for row in 0..NORM_ROWS {
        for col in 0..NORM_WIDTH {
            let at = row * NORM_WIDTH + col;
            let s = 1.0 + m[row * 2 * NORM_WIDTH + col];
            let scale = (normed_row[at] * s).abs().max(1.0);
            near(
                fused_norm[at],
                unfused[at],
                scale,
                &format!("{name}: norm_modulate against the two launches"),
                at,
            );
            near(
                got_o[at],
                want_o[at],
                scale,
                &format!("{name}: the pair against the two launches"),
                at,
            );
        }
    }
}

#[test]
fn the_fused_forms_land_what_the_launches_land_under_every_norm() {
    fused(Which::LayerNorm, "layernorm");
    fused(Which::RmsNoScale, "rmsnorm");
    fused(Which::RmsWeighted, "rmsnorm with a weight");
}
