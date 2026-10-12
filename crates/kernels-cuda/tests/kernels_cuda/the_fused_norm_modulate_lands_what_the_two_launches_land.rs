#![cfg(feature = "cuda")]

use crate::common::{Gpu, Lcg, from_bf16};
use dtype::Dtype;
use kernels_cuda::elemwise::modulate::{self, NormKind};
use kernels_cuda::elemwise::{layernorm, norm};
use kernels_cuda::tensor::Tensor;

const ROWS: usize = 24;
const WIDTH: usize = 256;
const EPS: f32 = 1e-6;

#[derive(Clone, Copy)]
enum Which {
    LayerNorm,
    RmsNoScale,
    RmsWeighted,
}

fn close(got: f32, want: f32, scale: f32, what: &str, at: usize) {
    assert!(
        (got - want).abs() <= (want.abs() + scale) / 128.0 + 1e-6,
        "{what} at {at}: {got} against {want}"
    );
}

fn check(which: Which, name: &str) {
    let mut lcg = Lcg::seeded(0xf05ed);
    let (r_raw, _) = lcg.row(ROWS * WIDTH);
    let (y_raw, _) = lcg.row(ROWS * WIDTH);
    let (g_raw, _) = lcg.row(ROWS * WIDTH);
    let (m_raw, _) = lcg.row(ROWS * 2 * WIDTH);
    let (w_raw, _) = lcg.row(WIDTH);

    let mut gpu = Gpu::open();
    let y_at = gpu.up(&y_raw);
    let g_at = gpu.up(&g_raw);
    let m_at = gpu.up(&m_raw);
    let w_at = gpu.up(&w_raw);
    let r_still = gpu.up(&r_raw);
    let r_chain = gpu.up(&r_raw);
    let r_fused = gpu.up(&r_raw);
    let normed = gpu.zeros(ROWS * WIDTH * 2);
    let o_unfused = gpu.zeros(ROWS * WIDTH * 2);
    let o_norm = gpu.zeros(ROWS * WIDTH * 2);
    let o_chain = gpu.zeros(ROWS * WIDTH * 2);
    let o_fused = gpu.zeros(ROWS * WIDTH * 2);
    let ctx = gpu.ctx();

    let rect = |at: u64| Tensor::new(at, ROWS as u32, WIDTH as u32, Dtype::Bf16);
    let g = rect(g_at);
    let m = Tensor::new(m_at, ROWS as u32, 2 * WIDTH as u32, Dtype::Bf16);
    let weight = Tensor::new(w_at, WIDTH as u32, 1, Dtype::Bf16);
    let kind = match which {
        Which::LayerNorm => NormKind::LayerNormNoAffine { eps: EPS },
        Which::RmsNoScale => NormKind::RmsNormNoScale { eps: EPS },
        Which::RmsWeighted => NormKind::RmsNorm { weight, eps: EPS },
    };

    match which {
        Which::LayerNorm => {
            layernorm::layernorm_no_scale(&ctx, rect(r_still), EPS, &mut rect(normed))
        }
        Which::RmsNoScale => norm::rmsnorm_no_scale(&ctx, rect(r_still), 0, EPS, &mut rect(normed)),
        Which::RmsWeighted => norm::rmsnorm(&ctx, rect(r_still), weight, EPS, &mut rect(normed)),
    }
    .expect("the norm fires");
    modulate::scale_shift(&ctx, rect(normed), m, None, &mut rect(o_unfused)).expect("fires");
    modulate::norm_modulate(&ctx, rect(r_still), m, None, kind, &mut rect(o_norm))
        .expect("the fused norm fires");

    modulate::gated_residual_add(&ctx, rect(r_chain), g, rect(y_at), None, &mut rect(r_chain))
        .expect("fires");
    modulate::norm_modulate(&ctx, rect(r_chain), m, None, kind, &mut rect(o_chain)).expect("fires");
    modulate::gated_residual_norm_modulate(
        &ctx,
        rect(r_fused),
        g,
        rect(y_at),
        m,
        None,
        kind,
        &mut rect(r_fused),
        &mut rect(o_fused),
    )
    .expect("the pair fires");
    gpu.sync();

    let want_r: Vec<u16> = gpu.down(r_chain, ROWS * WIDTH);
    let got_r: Vec<u16> = gpu.down(r_fused, ROWS * WIDTH);
    assert_eq!(
        got_r, want_r,
        "{name}: the residual is not the chain's bits"
    );

    let unfused: Vec<u16> = gpu.down(o_unfused, ROWS * WIDTH);
    let fused_norm: Vec<u16> = gpu.down(o_norm, ROWS * WIDTH);
    let normed_row: Vec<u16> = gpu.down(normed, ROWS * WIDTH);
    let want_o: Vec<u16> = gpu.down(o_chain, ROWS * WIDTH);
    let got_o: Vec<u16> = gpu.down(o_fused, ROWS * WIDTH);
    let m_host: Vec<u16> = gpu.down(m_at, ROWS * 2 * WIDTH);

    for row in 0..ROWS {
        for col in 0..WIDTH {
            let at = row * WIDTH + col;
            let s = 1.0 + from_bf16(m_host[row * 2 * WIDTH + col]);
            let scale = (from_bf16(normed_row[at]) * s).abs().max(1.0);
            close(
                from_bf16(fused_norm[at]),
                from_bf16(unfused[at]),
                scale,
                &format!("{name}: norm_modulate against the two launches"),
                at,
            );
            close(
                from_bf16(got_o[at]),
                from_bf16(want_o[at]),
                scale,
                &format!("{name}: the pair against the two launches"),
                at,
            );
        }
    }
}

#[test]
fn the_fused_forms_land_what_the_launches_land_under_every_norm() {
    check(Which::LayerNorm, "layernorm");
    check(Which::RmsNoScale, "rmsnorm");
    check(Which::RmsWeighted, "rmsnorm with a weight");
}
