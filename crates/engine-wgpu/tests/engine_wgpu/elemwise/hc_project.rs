//! The hyper-connection mix projection on the first wgpu adapter against an f64
//! host reference, beside the hand-written kernel: f32 normed streams into a
//! layer's mix row.

use crate::gpu::{Gpu, both, noise};
use kernels_wgpu::Tensor;
use kernels_wgpu::elemwise::hc;
use poem_ir::Dtype;

const N: usize = 3;
const M: usize = 4;
const H: usize = 64;

#[test]
fn the_hc_projection_answers_the_reference() {
    let Some(mut gpu) = Gpu::open() else {
        eprintln!("not asked: no device");
        return;
    };
    let st = |t: Tensor| kernels_slang::Tensor::new(t.buf, t.rows, t.width, t.dtype);
    let f = |buf: u32, rows: usize, w: usize| Tensor::new(buf, rows as u32, w as u32, Dtype::F32);
    let wide = M * H;
    let mix_hc = 2 * M + M * M;
    let nv = noise(3, N * wide);
    let fv: Vec<f32> = noise(4, mix_hc * wide).iter().map(|v| v * 0.05).collect();
    let (normed, hc_fn) = (gpu.up_f32(&nv), gpu.up_f32(&fv));
    let mixes_want: Vec<f64> = (0..N * mix_hc)
        .map(|i| {
            let (n, o) = (i / mix_hc, i % mix_hc);
            (0..wide)
                .map(|d| f64::from(nv[n * wide + d]) * f64::from(fv[o * wide + d]))
                .sum()
        })
        .collect();
    both(
        &mut gpu,
        "hc project",
        &|g| vec![(g.zeros(2 * N * mix_hc), N * mix_hc, true)],
        std::slice::from_ref(&mixes_want),
        &|g, o| {
            g.run(|c| {
                hc::project(
                    c,
                    f(normed, N, wide),
                    f(hc_fn, mix_hc, wide),
                    M as u32,
                    f(o[0].0, N, mix_hc),
                )
            })
        },
        &|g, o| {
            g.run_slang(|c| {
                kernels_slang::elemwise::hc_project::project(
                    c,
                    st(f(normed, N, wide)),
                    st(f(hc_fn, mix_hc, wide)),
                    M as u32,
                    st(f(o[0].0, N, mix_hc)),
                )
            })
        },
    );
}
