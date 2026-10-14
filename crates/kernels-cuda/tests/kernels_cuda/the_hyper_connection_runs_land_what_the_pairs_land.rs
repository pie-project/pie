#![cfg(feature = "cuda")]

// The hyper-connection mix and inject read and write whole 16-byte runs
// where the row width and the planes allow them, and an element at a time
// otherwise. Both land the same bits: the runs are fired on aligned planes,
// the element walk on the same values one element further in.

use crate::common::{Gpu, Lcg};
use dtype::Dtype;
use kernels_cuda::elemwise::hc;
use kernels_cuda::tensor::Tensor;

const ROWS: usize = 5;
const STREAMS: usize = 4;
const WIDTH: usize = 264;

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
    // Element `1` of each plane is element `0` of the aligned one: the same
    // values one element further in, so no plane there is 16-byte aligned.
    let shifted = |v: &[u16]| -> Vec<u16> { [&[0u16][..], v].concat() };
    let (g_at, n_at) = (gpu.up(&gates), gpu.up(&normed));
    let (g1_at, n1_at) = (gpu.up(&shifted(&gates)), gpu.up(&shifted(&normed)));
    let (o_at, o1_at) = (gpu.up(&o), gpu.up(&shifted(&o)));
    let (h_at, h1_at) = (gpu.up(&hyper), gpu.up(&shifted(&hyper)));
    let l_at = gpu.up(&logits);
    let y_at = gpu.zeros(ROWS * WIDTH * 2 + 16);
    let y1_at = gpu.zeros(ROWS * WIDTH * 2 + 16);
    let ctx = gpu.ctx();
    let t = |at: u64, rows: usize, width: usize| {
        Tensor::new(at, rows as u32, width as u32, Dtype::Bf16)
    };
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

// The routed SwiGLU and the routed weighted sum take the same 16-byte runs
// and land the same bits as their element walks.
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
    let shifted = |v: &[u16]| -> Vec<u16> { [&[0u16][..], v].concat() };
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
    let t = |at: u64, rows: usize, width: usize| {
        Tensor::new(at, rows as u32, width as u32, Dtype::Bf16)
    };
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
