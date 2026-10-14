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
