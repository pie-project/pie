#![cfg(feature = "cuda")]

//! The gated RMS norm over an f32 accumulator answers the host formula,
//! `w * x / rms(x) * gate'` with `gate'` the gate's SiLU (or sigmoid), per
//! head and per whole row, at the widths a warp carries alone and at one it
//! does not; and over an x landed in bf16 (a gated delta walk narrowed for
//! it), against the formula on those bf16 values.

use crate::common::{Gpu, Lcg, close, from_bf16, to_bf16};
use dtype::Dtype;
use kernels_cuda::elemwise::norm;
use kernels_cuda::tensor::Tensor;

fn check(rows: usize, width: usize, head_dim: usize, sigmoid: bool, narrow: bool) {
    let normed = if head_dim == 0 { width } else { head_dim };
    let mut lcg = Lcg::seeded(0x9a7e);
    let x: Vec<f32> = (0..rows * width).map(|_| 3.0 * lcg.unit()).collect();
    let x_narrow: Vec<u16> = x.iter().map(|&v| to_bf16(v)).collect();
    let x: Vec<f32> = if narrow {
        x_narrow.iter().map(|&v| from_bf16(v)).collect()
    } else {
        x
    };
    let (gate_raw, gate) = lcg.row(rows * width);
    let weight: Vec<f32> = (0..normed).map(|_| 1.0 + 0.5 * lcg.unit()).collect();
    let eps = 1e-6f32;

    let mut gpu = Gpu::open();
    let x_at = if narrow {
        gpu.up(&x_narrow)
    } else {
        gpu.up(&x)
    };
    let gate_at = gpu.up(&gate_raw);
    let weight_at = gpu.up(&weight);
    let y_at = gpu.zeros(rows * width * 2);
    let ctx = gpu.ctx();
    let mut y = Tensor::new(y_at, rows as u32, width as u32, Dtype::Bf16);
    norm::rmsnorm_gated(
        &ctx,
        Tensor::new(
            x_at,
            rows as u32,
            width as u32,
            if narrow { Dtype::Bf16 } else { Dtype::F32 },
        ),
        Tensor::new(gate_at, rows as u32, width as u32, Dtype::Bf16),
        Tensor::new(weight_at, 1, normed as u32, Dtype::F32),
        head_dim as u32,
        eps,
        sigmoid,
        &mut y,
    )
    .expect("the gated norm fires");
    gpu.sync();
    let got: Vec<u16> = gpu.down(y_at, rows * width);

    for unit in 0..rows * width / normed {
        let xs = &x[unit * normed..(unit + 1) * normed];
        let inv = (xs.iter().map(|v| v * v).sum::<f32>() / normed as f32 + eps)
            .sqrt()
            .recip();
        for i in 0..normed {
            let at = unit * normed + i;
            let g = gate[at];
            let sg = if sigmoid {
                1.0 / (1.0 + (-g).exp())
            } else {
                g / (1.0 + (-g).exp())
            };
            let want = weight[i] * xs[i] * inv * sg;
            let got = from_bf16(got[at]);
            assert!(
                close(got, want),
                "{rows}x{width} (head {head_dim}) element {at}: the norm landed {got} and the formula says {want}"
            );
        }
    }
}

#[test]
fn the_gated_norm_answers_the_host_reference() {
    for narrow in [false, true] {
        check(5, 4 * 128, 128, false, narrow);
        check(3, 2 * 256, 256, true, narrow);
        check(4, 256, 0, false, narrow);
        check(3, 3 * 96, 96, false, narrow);
    }
}
