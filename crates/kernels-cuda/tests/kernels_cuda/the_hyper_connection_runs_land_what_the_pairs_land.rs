#![cfg(feature = "cuda")]

//! The hyper-connection mix and inject, the routed SwiGLU and sum, the
//! packed GeGLU and the logit soft cap read and write whole 16-byte runs
//! where the row width and the planes allow them, and an element at a time
//! otherwise. Both land the same bits: the runs are fired on aligned planes,
//! the element walk on the same values one element further in.

use crate::common::{Gpu, Lcg};
use dtype::Dtype;
use kernels_cuda::elemwise::hc;
use kernels_cuda::tensor::Tensor;

const ROWS: usize = 5;
const STREAMS: usize = 4;
const WIDTH: usize = 264;

/// `v` one element further in: element `1` of the upload is element `0`.
fn shifted(v: &[u16]) -> Vec<u16> {
    [&[0u16][..], v].concat()
}

fn t(at: u64, rows: usize, width: usize) -> Tensor {
    Tensor::new(at, rows as u32, width as u32, Dtype::Bf16)
}

#[test]
fn the_hyper_connection_runs_land_what_the_pairs_land() {
    let mut gpu = Gpu::open();
    let mut rng = Lcg::seeded(0x4c_0e1);
    let wide = ROWS * STREAMS * WIDTH;
    let (gates, _) = rng.row(wide + 1);
    let (normed, _) = rng.row(wide + 1);
    let (o, _) = rng.row(ROWS * WIDTH + 1);
    let (hyper, _) = rng.row(wide + 1);
    let (logits, _) = rng.row(ROWS * STREAMS);
    let (g_at, n_at) = (gpu.up(&gates), gpu.up(&normed));
    let (g1_at, n1_at) = (gpu.up(&shifted(&gates)), gpu.up(&shifted(&normed)));
    let (o_at, o1_at) = (gpu.up(&o), gpu.up(&shifted(&o)));
    let (h_at, h1_at) = (gpu.up(&hyper), gpu.up(&shifted(&hyper)));
    let l_at = gpu.up(&logits);
    let y_at = gpu.zeros(ROWS * WIDTH * 2 + 16);
    let y1_at = gpu.zeros(ROWS * WIDTH * 2 + 16);
    let ctx = gpu.ctx();
    for (g, n, y) in [(g_at, n_at, y_at), (g1_at + 2, n1_at + 2, y1_at + 2)] {
        hc::mix(
            &ctx,
            t(g, ROWS, STREAMS * WIDTH),
            t(n, ROWS, STREAMS * WIDTH),
            STREAMS as u32,
            &mut t(y, ROWS, WIDTH),
        )
        .expect("the mix fires");
    }
    for (o, h) in [(o_at, h_at), (o1_at + 2, h1_at + 2)] {
        hc::inject(
            &ctx,
            t(o, ROWS, WIDTH),
            t(l_at, ROWS, STREAMS),
            STREAMS as u32,
            &mut t(h, ROWS, STREAMS * WIDTH),
        )
        .expect("the inject fires");
    }
    gpu.sync();
    let y: Vec<u16> = gpu.down(y_at, ROWS * WIDTH);
    let y1: Vec<u16> = gpu.down(y1_at + 2, ROWS * WIDTH);
    assert_eq!(y, y1, "the mix's runs and its element walk land alike");
    let h: Vec<u16> = gpu.down(h_at, wide);
    let h1: Vec<u16> = gpu.down(h1_at + 2, wide);
    assert_eq!(h, h1, "the inject's runs and its element walk land alike");
    assert_ne!(h, hyper[..wide], "the inject moved the streams");
}

/// The routed SwiGLU and the routed weighted sum.
#[test]
fn the_routed_swiglu_and_sum_runs_land_what_the_elements_land() {
    use kernels_cuda::linear::{mlp, moe};
    const TOKENS: usize = 6;
    const TOP_K: usize = 3;
    const INTER: usize = 136;
    const HIDDEN: usize = 264;
    let mut gpu = Gpu::open();
    let mut rng = Lcg::seeded(0x5_c1e);
    let routes = TOKENS * TOP_K;
    let (packed, _) = rng.row(routes * 2 * INTER);
    let (routed, _) = rng.row(routes * HIDDEN);
    let weights: Vec<f32> = (0..routes).map(|_| rng.unit()).collect();
    let (p_at, p1_at) = (gpu.up(&packed), gpu.up(&shifted(&packed)));
    let (r_at, r1_at) = (gpu.up(&routed), gpu.up(&shifted(&routed)));
    let w_at = gpu.up(&weights);
    let (s_at, s1_at) = (
        gpu.zeros(routes * INTER * 2 + 16),
        gpu.zeros(routes * INTER * 2 + 16),
    );
    let (y_at, y1_at) = (
        gpu.zeros(TOKENS * HIDDEN * 2 + 16),
        gpu.zeros(TOKENS * HIDDEN * 2 + 16),
    );
    let ctx = gpu.ctx();
    for (p, s) in [(p_at, s_at), (p1_at + 2, s1_at + 2)] {
        mlp::swiglu(
            &ctx,
            t(p, routes, 2 * INTER),
            INTER as u32,
            1,
            &mut t(s, routes, INTER),
        )
        .expect("the swiglu fires");
    }
    for (r, y) in [(r_at, y_at), (r1_at + 2, y1_at + 2)] {
        moe::weighted_sum(
            &ctx,
            t(r, routes, HIDDEN),
            Tensor::new(w_at, TOKENS as u32, TOP_K as u32, Dtype::F32),
            &mut t(y, TOKENS, HIDDEN),
        )
        .expect("the weighted sum fires");
    }
    gpu.sync();
    let s: Vec<u16> = gpu.down(s_at, routes * INTER);
    let s1: Vec<u16> = gpu.down(s1_at + 2, routes * INTER);
    assert_eq!(s, s1, "the swiglu's runs and its element walk land alike");
    let y: Vec<u16> = gpu.down(y_at, TOKENS * HIDDEN);
    let y1: Vec<u16> = gpu.down(y1_at + 2, TOKENS * HIDDEN);
    assert_eq!(
        y, y1,
        "the weighted sum's runs and its element walk land alike"
    );
    assert!(y.iter().any(|&v| v != 0), "the sum landed");
}

/// The packed tanh GeGLU, each the host's gelu(gate) * up.
#[test]
fn the_packed_geglu_runs_land_what_the_elements_land() {
    use kernels_cuda::linear::mlp;
    const ROWS: usize = 7;
    const INTER: usize = 136;
    let mut gpu = Gpu::open();
    let mut rng = Lcg::seeded(0x9e_61);
    let (packed, values) = rng.row(ROWS * 2 * INTER);
    let (p_at, p1_at) = (gpu.up(&packed), gpu.up(&shifted(&packed)));
    let (y_at, y1_at) = (
        gpu.zeros(ROWS * INTER * 2 + 16),
        gpu.zeros(ROWS * INTER * 2 + 16),
    );
    let ctx = gpu.ctx();
    for (p, y) in [(p_at, y_at), (p1_at + 2, y1_at + 2)] {
        mlp::geglu_tanh_packed(
            &ctx,
            t(p, ROWS, 2 * INTER),
            INTER as u32,
            1,
            &mut t(y, ROWS, INTER),
        )
        .expect("the geglu fires");
    }
    gpu.sync();
    let y: Vec<u16> = gpu.down(y_at, ROWS * INTER);
    let y1: Vec<u16> = gpu.down(y1_at + 2, ROWS * INTER);
    assert_eq!(y, y1, "the geglu's runs and its element walk land alike");
    for r in 0..ROWS {
        for i in 0..INTER {
            let g = f64::from(values[r * 2 * INTER + i]);
            let u = f64::from(values[r * 2 * INTER + INTER + i]);
            let inner = (2.0 / std::f64::consts::PI).sqrt() * (g + 0.044_715 * g * g * g);
            let want = 0.5 * g * (1.0 + inner.tanh()) * u;
            let got = f64::from(crate::common::from_bf16(y[r * INTER + i]));
            assert!(
                (got - want).abs() <= 1e-2 * want.abs().max(1.0),
                "row {r}, column {i}: {got} against {want}"
            );
        }
    }
}

/// The logit soft cap, each cap * tanh(x / cap).
#[test]
fn the_soft_capped_logit_runs_land_what_the_elements_land() {
    const N: usize = 3 * 1024;
    const CAP: f32 = 30.0;
    let mut gpu = Gpu::open();
    let mut rng = Lcg::seeded(0x50f7);
    let (bits, _) = rng.row(N);
    let logits: Vec<u16> = bits
        .iter()
        .map(|&b| crate::common::to_bf16(crate::common::from_bf16(b) * 40.0))
        .collect();
    let shifted = [&[0u16][..], &logits].concat();
    let (x_at, x1_at) = (gpu.up(&logits), gpu.up(&shifted));
    let ctx = gpu.ctx();
    for at in [x_at, x1_at + 2] {
        kernels_cuda::attn::logit_softcap(&ctx, &mut Tensor::new(at, 3, 1024, Dtype::Bf16), CAP)
            .expect("the soft cap fires");
    }
    gpu.sync();
    let y: Vec<u16> = gpu.down(x_at, N);
    let y1: Vec<u16> = gpu.down(x1_at + 2, N);
    assert_eq!(y, y1, "the soft cap's runs and its element walk land alike");
    for (got, &x) in y.iter().zip(&logits) {
        let x = f64::from(crate::common::from_bf16(x));
        let want = f64::from(CAP) * (x / f64::from(CAP)).tanh();
        let got = f64::from(crate::common::from_bf16(*got));
        assert!(
            (got - want).abs() <= 1e-2 * want.abs().max(1.0),
            "{got} against {want}"
        );
    }
}
