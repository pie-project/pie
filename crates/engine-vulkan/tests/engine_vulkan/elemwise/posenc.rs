//! The sinusoidal embedding on the first Vulkan device against an f64 host
//! reference: an even width and an odd one, sine first and cosine first.

use crate::gpu::{Gpu, close};
use poem_ir::Dtype;

#[test]
fn the_sinusoid_answers_the_reference() {
    let Some(mut gpu) = Gpu::open() else {
        eprintln!("not asked: no device");
        return;
    };
    let t = |buf: u32, rows: usize, width: usize, dtype: Dtype| {
        kernels_slang::Tensor::new(buf, rows as u32, width as u32, dtype)
    };
    let rows = 3usize;
    let tv = [0.0f32, 0.37, 0.999];
    let tb = gpu.up_f32(&tv);
    let tt = t(tb, rows, 1, Dtype::F32);
    for (dim, flip) in [(64usize, false), (65, true)] {
        let yt = |buf: u32| t(buf, rows, dim, Dtype::F32);
        let half = dim / 2;
        let want: Vec<f64> = (0..rows * dim)
            .map(|i| {
                let (r, c) = (i / dim, i % dim);
                if c >= 2 * half {
                    return 0.0;
                }
                let k = c % half;
                let f = (-(10_000f64).ln() * k as f64 / half as f64).exp();
                let a = 1000.0 * f64::from(tv[r]) * f;
                if (c < half) != flip { a.sin() } else { a.cos() }
            })
            .collect();
        let shape = format!("dim {dim}, flipped {flip}");

        let y = gpu.zeros(2 * rows * dim);
        gpu.run(|c| {
            kernels_slang::elemwise::posenc::sinusoid(
                c,
                tt,
                dim as u32,
                10_000.0,
                flip,
                1000.0,
                yt(y),
            )
        });
        close(
            &gpu.down_f32(y, rows * dim),
            &want,
            &format!("sinusoid {shape}"),
        );
    }
}
