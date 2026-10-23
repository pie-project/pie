//! The gated delta rule's gates on the first Vulkan device, beside the
//! hand-written kernel, against an f64 host reference: `g = -exp(a_log)
//! softplus(a + dt_bias)` and `beta = sigmoid(b)`, the softplus past its
//! linear cut included.

use kernels_vulkan::Tensor;
use kernels_vulkan::attn::ssm;
use poem_ir::Dtype;

use crate::gpu::{Gpu, both, noise};

fn st(t: Tensor) -> kernels_slang::Tensor {
    kernels_slang::Tensor::new(t.buf, t.rows, t.width, t.dtype)
}

#[test]
fn the_gdn_gates_answer_the_reference() {
    let Some(mut gpu) = Gpu::open() else {
        eprintln!("not asked: no device");
        return;
    };
    let (rows, heads) = (3usize, 6usize);
    let mut ba = noise(1, rows * 2 * heads);
    // A row whose `a + dt_bias` is past softplus's linear cut.
    ba[2 * heads + heads] = 30.0;
    let (bab, bar) = gpu.up(&ba);
    let (dtb, dtr) = gpu.up(&noise(2, heads));
    let a_log = noise(3, heads);
    let alb = gpu.up_f32(&a_log);
    let mut want = vec![0.0f64; rows * 2 * heads];
    for t in 0..rows {
        for h in 0..heads {
            let row = t * 2 * heads;
            let b = f64::from(bar[row + h]);
            let z = f64::from(bar[row + heads + h]) + f64::from(dtr[h]);
            let sp = if z > 20.0 { z } else { z.exp().ln_1p() };
            want[row + h] = -f64::from(a_log[h]).exp() * sp;
            want[row + heads + h] = 1.0 / (1.0 + (-b).exp());
        }
    }
    let t = |buf: u32, r: usize, w: usize, d: Dtype| Tensor::new(buf, r as u32, w as u32, d);
    let bat = t(bab, rows, 2 * heads, Dtype::Bf16);
    let dtt = t(dtb, 1, heads, Dtype::Bf16);
    let alt = t(alb, 1, heads, Dtype::F32);
    let n = rows * 2 * heads;
    both(
        &mut gpu,
        "gdn gates",
        &|g| vec![(g.up_f32(&vec![0.0; n]), n, true)],
        &[want],
        &|g, o| g.run(|c| ssm::gdn_prep(c, bat, dtt, alt, t(o[0].0, rows, 2 * heads, Dtype::F32))),
        &|g, o| {
            g.run_slang(|c| {
                kernels_slang::ssm::gdn_prep::gdn_prep(
                    c,
                    st(bat),
                    st(dtt),
                    st(alt),
                    st(t(o[0].0, rows, 2 * heads, Dtype::F32)),
                )
            })
        },
    );
}
