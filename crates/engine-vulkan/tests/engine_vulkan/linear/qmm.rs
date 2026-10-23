//! The tiled points of the affine-quantized matmul on the first Vulkan
//! device, beside the hand-written kernels, against an f64 host reference:
//! the tiled rung at every row tile (a batch padded into its capacity
//! included) and both column tiles, the register-tiled rung at both row
//! tiles and both K steps, and the cooperative-matrix rung where the device
//! has cooperative matrices.

use kernels_slang::linear::qmm::{self, Qmm};
use kernels_vulkan::{Arg, Bank, Encode, Fire, Tensor};
use poem_ir::Dtype;

use crate::gpu::{Gpu, both, noise};

fn st(t: Tensor) -> kernels_slang::Tensor {
    kernels_slang::Tensor::new(t.buf, t.rows, t.width, t.dtype)
}

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

/// `v` rounded to the nearest IEEE half: the cooperative-matrix tier
/// multiplies halves, so its reference does too.
fn half(v: f64) -> f64 {
    if v == 0.0 {
        return 0.0;
    }
    let quantum = 2f64.powi((v.abs().log2().floor() as i32).max(-14) - 10);
    (v / quantum).round_ties_even() * quantum
}

/// A fire of the hand-written kernel: its unit, entry, groups and workgroup.
type Hand = (&'static str, String, [u32; 3], [u32; 3]);

#[test]
fn the_tiled_points_answer_the_reference() {
    let Some(mut gpu) = Gpu::open() else {
        eprintln!("not asked: no device");
        return;
    };
    let coopmat = gpu.device.chip().coopmat;
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
        let sbank = kernels_slang::Bank {
            codes: st(bank.codes),
            scales: st(bank.scales),
            biases: bank.biases.map(st),
            group: bank.group,
            bits,
        };
        // (rows, capacity, the rung, the hand-written kernel it matches)
        let gs = format!("gs_{group}_b_{bits}");
        let mut cases: Vec<(usize, usize, Qmm, Hand)> = Vec::new();
        for (m, cap, bm) in [
            (8usize, 8usize, 8u32),
            (20, 32, 16),
            (32, 32, 32),
            (70, 128, 64),
        ] {
            let padded = cap as u32;
            for (crossover, bn) in [(1u32, 32u32), (u32::MAX, 16)] {
                cases.push((
                    m,
                    cap,
                    Qmm::Tiled {
                        crossover_tg: crossover,
                    },
                    (
                        "quant/qmm_t.slang",
                        format!("affine_qmm_t_bf16_{gs}_bm_{bm}_bn_{bn}"),
                        [n as u32 / bn, padded / bm, 1],
                        [32, 2, if bm < 16 { 1 } else { 2 }],
                    ),
                ));
            }
        }
        for (m, bm, k16) in [
            (40usize, 32u32, false),
            (40, 32, true),
            (70, 64, false),
            (70, 64, true),
        ] {
            cases.push((
                m,
                m,
                Qmm::RegisterTiled { bm, k16 },
                (
                    "quant/qmm_rt.slang",
                    format!(
                        "affine_qmm_rt{}_bf16_{gs}_bm_{bm}",
                        if k16 { "_k16" } else { "" }
                    ),
                    [n as u32 / 64, (m as u32).div_ceil(bm), 1],
                    [16, bm / 4, 1],
                ),
            ));
        }
        if coopmat {
            cases.push((
                70,
                70,
                Qmm::Coopmat,
                (
                    "quant/qmm_t.slang",
                    format!("affine_qmm_t_cm_bf16_{gs}"),
                    [n as u32 / 64, 2, 1],
                    [32, 4, 1],
                ),
            ));
        }
        for (m, cap, rung, (unit, entry, groups, local)) in cases {
            let mut xv = noise(7 + m as u64, m * k);
            xv.resize(cap * k, 0.0);
            let (x, xr) = gpu.up(&xv);
            let xt = t(x, m, k, Dtype::Bf16);
            let operand = |v: f64| if rung == Qmm::Coopmat { half(v) } else { v };
            let want: Vec<f64> = (0..m * n)
                .map(|i| {
                    let (r, c) = (i / n, i % n);
                    (0..k)
                        .map(|j| operand(f64::from(xr[r * k + j])) * operand(weight(c * k + j)))
                        .sum()
                })
                .collect();
            let entry: &'static str = Box::leak(entry.into_boxed_str());
            both(
                &mut gpu,
                &format!("qmm {rung:?} {bits} bits, group {group}, {m} x {n}"),
                &|g| vec![(g.zeros(cap * n), m * n, false)],
                std::slice::from_ref(&want),
                &|g, o| {
                    let y = t(o[0].0, m, n, Dtype::Bf16);
                    g.run(|c| {
                        c.fire(
                            Fire::at(unit, entry).groups(groups).group(local),
                            &[
                                bank.codes.arg(),
                                bank.scales.arg(),
                                bank.biases.expect("affine").arg(),
                                xt.arg(),
                                y.arg_mut(),
                                (k as i32).arg(),
                                (n as i32).arg(),
                                (m as i32).arg(),
                            ],
                        )
                    });
                },
                &|g, o| {
                    let y = st(t(o[0].0, m, n, Dtype::Bf16));
                    g.run_slang(|c| qmm::matmul(c, &[rung], st(xt), sbank, y, cap as u32));
                },
            );
        }
    }
}
