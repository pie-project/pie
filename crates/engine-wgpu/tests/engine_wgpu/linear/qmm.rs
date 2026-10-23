//! The tiled point of the affine-quantized matmul on the first wgpu adapter
//! against an f64 host reference: every row
//! tile (a batch padded into its capacity included) and landed column tiles.
//! The register-tiled and cooperative-matrix rungs are SPIR-V's.

use kernels_slang::linear::qmm::{self, Qmm};
use kernels_slang::{Bank, Tensor};
use poem_ir::Dtype;

use crate::gpu::{Gpu, landed, noise};

fn t(b: u32, r: usize, w: usize, d: Dtype) -> Tensor {
    Tensor::new(b, r as u32, w as u32, d)
}

fn codes(seed: u64, words: usize) -> Vec<u32> {
    let mut s = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    (0..words)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s as u32
        })
        .collect()
}

#[test]
fn the_tiled_point_answers_the_reference() {
    let Some(mut gpu) = Gpu::open() else {
        eprintln!("not asked: no device");
        return;
    };
    let (k, n) = (256usize, 64usize);
    for (bits, group) in [(4u32, 64usize), (2, 32), (8, 128)] {
        let per = 32 / bits as usize;
        let cw = codes(u64::from(bits), n * k / per);
        let w = gpu.up_u32(&cw);
        let groups = n * k / group;
        let (sc, scr) = gpu.up(&noise(5, groups)
            .iter()
            .map(|v| v * 0.05)
            .collect::<Vec<_>>());
        let (bi, bir) = gpu.up(&noise(6, groups).iter().map(|v| v * 0.1).collect::<Vec<_>>());
        let weight = |i: usize| {
            let (row, j) = (i / k, i % k);
            let code = (cw[i / per] >> ((j % per) as u32 * bits)) & ((1 << bits) - 1);
            let g = row * (k / group) + j / group;
            f64::from(scr[g]) * f64::from(code) + f64::from(bir[g])
        };
        let bank = Bank {
            codes: t(w, n, k * bits as usize / 32, Dtype::U32),
            scales: t(sc, n, k / group, Dtype::Bf16),
            biases: Some(t(bi, n, k / group, Dtype::Bf16)),
            group: group as u32,
            bits,
        };
        // (rows, capacity, the rung)
        let mut cases: Vec<(usize, usize, Qmm)> = Vec::new();
        // Row tiles of 8, 16, 32 and 64 (the second padded into its
        // capacity), at landed column tiles.
        for (m, cap) in [(8usize, 8usize), (20, 32), (32, 32), (70, 128)] {
            for crossover in [1u32, u32::MAX] {
                cases.push((
                    m,
                    cap,
                    Qmm::Tiled {
                        crossover_tg: crossover,
                    },
                ));
            }
        }
        for (m, cap, rung) in cases {
            let mut xv = noise(7 + m as u64, m * k);
            xv.resize(cap * k, 0.0);
            let (x, xr) = gpu.up(&xv);
            let xt = t(x, m, k, Dtype::Bf16);
            let want: Vec<f64> = (0..m * n)
                .map(|i| {
                    let (r, c) = (i / n, i % n);
                    (0..k)
                        .map(|j| f64::from(xr[r * k + j]) * weight(c * k + j))
                        .sum()
                })
                .collect();
            landed(
                &mut gpu,
                &format!("qmm {rung:?} {bits} bits, group {group}, {m} x {n}"),
                &|g| vec![(g.zeros(cap * n), m * n, false)],
                std::slice::from_ref(&want),
                &|g, o| {
                    let y = t(o[0].0, m, n, Dtype::Bf16);
                    g.run(|c| qmm::matmul(c, &[rung], xt, bank, y, cap as u32));
                },
            );
        }
    }
}
