//! The vector points of the affine-quantized matmul on the first Vulkan
//! device, beside the hand-written kernels, against an f64 host reference.
//! The vector point at 2, 4 and 8 bits, groups of 32 to 128, one to eight
//! activation rows a workgroup over row counts that leave a workgroup part
//! full, and an odd column count. The one-row point at every lane split the
//! device's subgroups hold, one and two columns a lane.

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

/// Random codes, packed low first.
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

/// An `n x k` bank of `bits`-bit codes in groups of `group`, as both crates
/// spell it, and its weights to f64, row by row.
fn bank(
    gpu: &mut Gpu,
    (bits, group, n, k): (u32, usize, usize, usize),
) -> (Bank, kernels_slang::Bank, Vec<f64>) {
    let per = 32 / bits as usize;
    let cw = codes(u64::from(bits), n * k / per);
    let w = gpu.up_u32(&cw);
    let groups = n * k / group;
    let (sc, scr) = gpu.up(&noise(5, groups)
        .iter()
        .map(|v| v * 0.05)
        .collect::<Vec<_>>());
    let (bi, bir) = gpu.up(&noise(6, groups).iter().map(|v| v * 0.1).collect::<Vec<_>>());
    let weights = (0..n * k)
        .map(|i| {
            let (row, j) = (i / k, i % k);
            let code = (cw[i / per] >> ((j % per) as u32 * bits)) & ((1 << bits) - 1);
            let g = row * (k / group) + j / group;
            f64::from(scr[g]) * f64::from(code) + f64::from(bir[g])
        })
        .collect();
    let hand = Bank {
        codes: t(w, n, k * bits as usize / 32, Dtype::U32),
        scales: t(sc, n, k / group, Dtype::Bf16),
        biases: Some(t(bi, n, k / group, Dtype::Bf16)),
        group: group as u32,
        bits,
    };
    let slang = kernels_slang::Bank {
        codes: st(hand.codes),
        scales: st(hand.scales),
        biases: hand.biases.map(st),
        group: hand.group,
        bits,
    };
    (hand, slang, weights)
}

fn reference(x: &[f32], weights: &[f64], m: usize, n: usize, k: usize) -> Vec<f64> {
    (0..m * n)
        .map(|i| {
            let (r, c) = (i / n, i % n);
            (0..k)
                .map(|j| f64::from(x[r * k + j]) * weights[c * k + j])
                .sum()
        })
        .collect()
}

/// The hand-written kernel's arguments, in its binding order.
fn args(
    w: &Bank,
    x: Tensor,
    y: Tensor,
    (m, n, k): (usize, usize, usize),
) -> Vec<kernels_vulkan::ArgValue> {
    vec![
        w.codes.arg(),
        w.scales.arg(),
        w.biases.expect("affine").arg(),
        x.arg(),
        y.arg_mut(),
        (k as i32).arg(),
        (n as i32).arg(),
        (m as i32).arg(),
    ]
}

#[test]
fn the_vector_point_answers_the_reference() {
    let Some(mut gpu) = Gpu::open() else {
        eprintln!("not asked: no device");
        return;
    };
    let k = 256usize;
    for (bits, group, n) in [(4u32, 64usize, 24usize), (2, 32, 16), (8, 128, 21)] {
        let (hand_bank, slang_bank, weights) = bank(&mut gpu, (bits, group, n, k));
        for (m, vecs) in [(1usize, 1u32), (3, 2), (5, 4), (9, 8), (6, 4)] {
            let (x, xr) = gpu.up(&noise(7 + m as u64, m * k));
            let xt = t(x, m, k, Dtype::Bf16);
            let want = reference(&xr, &weights, m, n, k);
            let hand = format!(
                "affine_qmv_bf16_gs_{group}_b_{bits}{}",
                if vecs == 4 {
                    String::new()
                } else {
                    format!("_v{vecs}")
                }
            );
            let hand: &'static str = Box::leak(hand.into_boxed_str());
            both(
                &mut gpu,
                &format!("qmv {bits} bits, group {group}, {m} x {n}, {vecs} a workgroup"),
                &|g| vec![(g.zeros(m * n), m * n, false)],
                std::slice::from_ref(&want),
                &|g, o| {
                    let y = t(o[0].0, m, n, Dtype::Bf16);
                    g.run(|c| {
                        c.fire(
                            Fire::at("quant/qmv.slang", hand)
                                .groups([(n as u32).div_ceil(8), (m as u32).div_ceil(vecs), 1])
                                .group([32, 4, 1]),
                            &args(&hand_bank, xt, y, (m, n, k)),
                        )
                    });
                },
                &|g, o| {
                    let y = st(t(o[0].0, m, n, Dtype::Bf16));
                    g.run_slang(|c| qmm::matmul(c, &[Qmm::Qmv { vecs }], st(xt), slang_bank, y));
                },
            );
        }
    }
}

#[test]
fn the_one_row_point_answers_the_reference() {
    let Some(mut gpu) = Gpu::open() else {
        eprintln!("not asked: no device");
        return;
    };
    let width = gpu.device.chip().subgroup;
    let k = 512usize;
    for (bits, group, n) in [(4u32, 64usize, 24usize), (8, 128, 21), (4, 32, 40)] {
        let (hand_bank, slang_bank, weights) = bank(&mut gpu, (bits, group, n, k));
        let (x, xr) = gpu.up(&noise(9, k));
        let xt = t(x, 1, k, Dtype::Bf16);
        let want = reference(&xr, &weights, 1, n, k);
        for (lanes, rows) in [(32u32, 1u32), (32, 2), (16, 1), (16, 2), (8, 1), (8, 2)] {
            if lanes > width {
                continue;
            }
            let hand = if (lanes, rows) == (32, 1) {
                format!("affine_qmv1_bf16_gs_{group}_b_{bits}")
            } else {
                format!("affine_qmv1_bf16_gs_{group}_b_{bits}_l{lanes}_r{rows}")
            };
            let hand: &'static str = Box::leak(hand.into_boxed_str());
            let groups = (n as u32).div_ceil(256 / lanes * rows);
            both(
                &mut gpu,
                &format!(
                    "qmv row {bits} bits, group {group}, {n} columns, {lanes} lanes, {rows} a lane"
                ),
                &|g| vec![(g.zeros(n), n, false)],
                std::slice::from_ref(&want),
                &|g, o| {
                    let y = t(o[0].0, 1, n, Dtype::Bf16);
                    g.run(|c| {
                        c.fire(
                            Fire::at("quant/qmv1.slang", hand)
                                .groups([groups, 1, 1])
                                .group([64, 4, 1]),
                            &args(&hand_bank, xt, y, (1, n, k)),
                        )
                    });
                },
                &|g, o| {
                    let y = st(t(o[0].0, 1, n, Dtype::Bf16));
                    g.run_slang(|c| {
                        qmm::matmul(c, &[Qmm::OneRow { lanes, rows }], st(xt), slang_bank, y)
                    });
                },
            );
        }
    }
}
