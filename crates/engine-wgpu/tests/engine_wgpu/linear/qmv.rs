//! The vector point of the affine-quantized matmul on the first wgpu
//! adapter, beside the hand-written kernel, against an f64 host reference:
//! 2, 4 and 8 bits, groups of 32 to 128, one to eight activation rows a
//! workgroup over row counts that leave a workgroup part full.

use kernels_slang::linear::qmm::{self, Qmm};
use kernels_wgpu::{Arg, Bank, Encode, Fire, Tensor};
use poem_ir::Dtype;

use crate::gpu::{Gpu, both, noise};

fn st(t: Tensor) -> kernels_slang::Tensor {
    kernels_slang::Tensor::new(t.buf, t.rows, t.width, t.dtype)
}

/// Random codes, `bits` wide, packed low first.
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
fn the_vector_point_answers_the_reference() {
    let Some(mut gpu) = Gpu::open() else {
        eprintln!("not asked: no device");
        return;
    };
    let t = |b: u32, r: usize, w: usize, d: Dtype| Tensor::new(b, r as u32, w as u32, d);
    let k = 256usize;
    for (bits, group, n) in [(4u32, 64usize, 24usize), (2, 32, 16), (8, 128, 22)] {
        let per = 32 / bits as usize;
        let cw = codes(u64::from(bits), n * k / per);
        let w = gpu.up_u32(&cw);
        let groups = n * k / group;
        let (sc, scr) = gpu.up(&noise(5, groups)
            .iter()
            .map(|v| v * 0.05)
            .collect::<Vec<_>>());
        let (bi, bir) = gpu.up(&noise(6, groups).iter().map(|v| v * 0.1).collect::<Vec<_>>());
        let weight = |row: usize, j: usize| {
            let word = cw[(row * k + j) / per];
            let code = (word >> ((j % per) as u32 * bits)) & ((1 << bits) - 1);
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
        let sbank = kernels_slang::Bank {
            codes: st(bank.codes),
            scales: st(bank.scales),
            biases: bank.biases.map(st),
            group: bank.group,
            bits,
        };
        for (m, vecs) in [(1usize, 1u32), (3, 2), (5, 4), (9, 8), (6, 4)] {
            let (x, xr) = gpu.up(&noise(7 + m as u64, m * k));
            let xt = t(x, m, k, Dtype::Bf16);
            let want: Vec<f64> = (0..m * n)
                .map(|i| {
                    let (r, c) = (i / n, i % n);
                    (0..k)
                        .map(|j| f64::from(xr[r * k + j]) * weight(c, j))
                        .sum()
                })
                .collect();
            // The WGSL kernel reads one activation row a workgroup.
            let hand: &'static str =
                Box::leak(format!("affine_qmv_bf16_gs_{group}_b_{bits}").into_boxed_str());
            both(
                &mut gpu,
                &format!("qmv {bits} bits, group {group}, {m} x {n}, {vecs} a workgroup"),
                &|g| vec![(g.zeros(m * n), m * n, false)],
                std::slice::from_ref(&want),
                &|g, o| {
                    let y = t(o[0].0, m, n, Dtype::Bf16);
                    g.run(|c| {
                        c.fire(
                            Fire::at("quant/qmv.wgsl", hand)
                                .groups([m as u32, (n as u32).div_ceil(8), 1])
                                .group([32, 2, 1]),
                            &[
                                bank.codes.arg(),
                                bank.scales.arg(),
                                bank.biases.expect("affine").arg(),
                                xt.arg(),
                                y.arg_mut(),
                                (k as i32).arg(),
                                (n as i32).arg(),
                            ],
                        )
                    });
                },
                &|g, o| {
                    let y = st(t(o[0].0, m, n, Dtype::Bf16));
                    g.run_slang(|c| {
                        qmm::matmul(c, &[Qmm::Qmv { vecs }], st(xt), sbank, y, m as u32)
                    });
                },
            );
        }
    }
}
