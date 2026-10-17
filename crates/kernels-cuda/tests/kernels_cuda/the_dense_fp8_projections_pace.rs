#![cfg(feature = "cuda")]

//! DeepSeek-V4-Flash's dense fp8b128 projections at decode widths (`--ignored`,
//! a measurement): the tcgen05 W8A8 GEMM (its activations quantized in a
//! launch before) against the mma.sync fp8 GEMV (`matmul_grouped_fp8_with`,
//! one group, the rows quantized in the same launch) in several CTA shapes,
//! each fire on the next of a ring of planes past L2.

use crate::common::{Gpu, Lcg};
use dtype::Dtype;
use kernels_cuda::linear::act_quant::{self, Format, Source, Weight};
use kernels_cuda::linear::moe::{Fp8Shape, matmul_grouped_fp8_with};
use kernels_cuda::linear::tc5;
use kernels_cuda::tensor::Tensor;

/// (name, n, k).
const SHAPES: [(&str, usize, usize); 7] = [
    ("shared gate-up", 4096, 4096),
    ("shared down", 4096, 2048),
    ("q down", 1024, 4096),
    ("q up", 32768, 1024),
    ("kv down", 512, 4096),
    ("index q up", 8192, 1024),
    ("o up", 4096, 8192),
];

const SHAPES_FP8: [(u32, u32, u32); 6] = [
    (4, 4, 2),
    (2, 4, 2),
    (1, 4, 2),
    (1, 8, 2),
    (2, 8, 2),
    (1, 8, 4),
];

#[test]
#[ignore = "a measurement: run alone, on an idle GPU"]
fn the_dense_fp8_projections_pace() {
    if !tc5::supported() {
        return;
    }
    let mut gpu = Gpu::open();
    let mut lcg = Lcg::seeded(0xde);
    println!(
        "| projection | rows | tcgen05 us | fp8_block_gemv us | {} |",
        SHAPES_FP8.map(|s| format!("{s:?}")).join(" | ")
    );
    for (name, n, k) in SHAPES {
        let bytes = n * k;
        let planes = (192usize << 20).div_ceil(bytes).max(2);
        let codes: Vec<u8> = (0..bytes).map(|i| (i * 37 % 251) as u8 & 0x77).collect();
        let exps = vec![120u8; (n / 128).max(1) * (k / 128)];
        let ring: Vec<(u64, u64)> = (0..planes)
            .map(|_| {
                let c = gpu.up(&codes);
                let e = gpu.up(&exps);
                let atoms = gpu.zeros(tc5::atom_bytes(n, k, 128));
                act_quant::scales_to_atoms(
                    &gpu.ctx(),
                    "test",
                    e,
                    atoms,
                    n as u32,
                    (k / 128) as u32,
                    128,
                )
                .expect("the atoms lay");
                (c, atoms)
            })
            .collect();
        for rows in [1usize, 4, 8] {
            let (x_bits, _) = lcg.row(rows * k);
            let x = Tensor::new(gpu.up(&x_bits), rows as u32, k as u32, Dtype::Bf16);
            let y_at = gpu.zeros(rows * n * 2);
            let ctx = gpu.ctx().with_tuning(crate::common::SM100);
            let reps = 300u32;
            let time = |f: &dyn Fn(usize) -> bool| -> Option<f64> {
                for i in 0..10 {
                    if !f(i) {
                        return None;
                    }
                }
                gpu.sync();
                let t = std::time::Instant::now();
                for i in 0..reps as usize {
                    f(i);
                }
                gpu.sync();
                Some(t.elapsed().as_secs_f64() * 1e6 / f64::from(reps))
            };
            let tc5_us = time(&|i| {
                let (c, a) = ring[i % planes];
                let mut y = Tensor::new(y_at, rows as u32, n as u32, Dtype::Bf16);
                act_quant::matmul(
                    &ctx,
                    "test",
                    Format::E4m3 { gran: 128 },
                    Source::rows(x, None),
                    Weight {
                        codes: c,
                        scales: a,
                        n: n as u32,
                        k: k as u32,
                        elem: tc5::Elem::E4m3,
                        gran: 128,
                        laid: true,
                    },
                    &mut y,
                    1.0,
                )
                .is_ok()
            });
            let gemv_us = time(&|i| {
                let (c, a) = ring[i % planes];
                let mut y = Tensor::new(y_at, rows as u32, n as u32, Dtype::Bf16);
                act_quant::fp8_block_gemv(&ctx, x, c, a, &mut y).unwrap_or(false)
            });
            let fp8: Vec<String> = SHAPES_FP8
                .iter()
                .map(|&(tiles, split, depth)| {
                    time(&|i| {
                        let (c, a) = ring[i % planes];
                        let mut y = Tensor::new(y_at, rows as u32, n as u32, Dtype::Bf16);
                        matmul_grouped_fp8_with(
                            &ctx,
                            x,
                            c,
                            a,
                            &mut y,
                            1,
                            Fp8Shape {
                                tiles,
                                split,
                                depth,
                            },
                        )
                        .is_ok()
                    })
                    .map_or("-".to_string(), |us| format!("{us:.1}"))
                })
                .collect();
            println!(
                "| {name} | {rows} | {} | {} | {} |",
                tc5_us.map_or("-".to_string(), |us| format!("{us:.1}")),
                gemv_us.map_or("-".to_string(), |us| format!("{us:.1}")),
                fp8.join(" | ")
            );
        }
    }
}
