//! The LoRA correction on the first wgpu adapter, against an f64 host
//! reference: three rows over two adapters and a row with none, added onto a
//! live output, at the baseline (the subgroup tier is SPIR-V's).

use kernels_slang::Tensor;
use kernels_slang::Tier;
use poem_ir::Dtype;

use crate::gpu::{Gpu, landed, noise};

fn t(b: u32, r: usize, w: usize, d: Dtype) -> Tensor {
    Tensor::new(b, r as u32, w as u32, d)
}

#[test]
fn the_lora_correction_answers_the_reference() {
    let Some(mut gpu) = Gpu::open() else {
        eprintln!("not asked: no device");
        return;
    };
    let (rows, input, out, rank, adapters) = (4usize, 96usize, 70usize, 8usize, 2usize);
    let (x, xr) = gpu.up(&noise(1, rows * input));
    let (a, ar) = gpu.up(&noise(2, adapters * rank * input));
    let (b, br) = gpu.up(&noise(3, adapters * out * rank));
    let ids = [1i32, -1, 0, 1];
    let routes = gpu.up_i32(&ids);
    let y0 = noise(4, rows * out);
    let held: Vec<f32> = gpu.up(&y0).1;
    let want: Vec<f64> = (0..rows * out)
        .map(|i| {
            let (row, n) = (i / out, i % out);
            let base = f64::from(held[i]);
            let Ok(adapter) = usize::try_from(ids[row]) else {
                return base;
            };
            let waist: Vec<f64> = (0..rank)
                .map(|r| {
                    (0..input)
                        .map(|c| {
                            f64::from(ar[(adapter * rank + r) * input + c])
                                * f64::from(xr[row * input + c])
                        })
                        .sum()
                })
                .collect();
            base + (0..rank)
                .map(|r| f64::from(br[(adapter * out + n) * rank + r]) * waist[r])
                .sum::<f64>()
        })
        .collect();
    let xt = t(x, rows, input, Dtype::Bf16);
    let at = t(a, adapters, rank * input, Dtype::Bf16);
    let bt = t(b, adapters, out * rank, Dtype::Bf16);
    let rt = t(routes, rows, 1, Dtype::I32);
    let tier = Tier::Baseline;
    landed(
        &mut gpu,
        &format!("lora at {}", tier.name()),
        &|g| vec![(g.up(&y0).0, rows * out, false)],
        std::slice::from_ref(&want),
        &|g, o| {
            let y = t(o[0].0, rows, out, Dtype::Bf16);
            g.run(|c| kernels_slang::linear::lora::correct(c, tier, xt, at, bt, rt, y));
        },
    );
}
