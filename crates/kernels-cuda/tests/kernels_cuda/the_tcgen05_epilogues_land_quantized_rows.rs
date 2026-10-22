#![cfg(feature = "cuda")]

//! B200's tcgen05 gate-up GEMM storing its SwiGLU as the next GEMM's operand:
//! fp8 under a UE8M0 scale a 128 columns, or NVFP4 under an e4m3 scale a
//! 16 and a global scale; each value read back against the host dot.

use crate::common::linear_ref::glu;
use crate::common::tc5::{e2m1, e4m3, side};
use crate::common::{Gpu, Lcg};
use kernels_cuda::linear::gemm_sm100::{
    Elem, Epilogue, Glu, Operand, Out, Plan, atom_bytes, atom_offset, gemm_epi_with, supported,
};

/// One dense bf16 gate-up product `[m, 2n, k]`, its SwiGLU `[m, n]` stored
/// as fp8 blocks or as NVFP4.
fn check(nvfp4: bool, [m, n, k]: [usize; 3]) {
    let what = format!("nvfp4 {nvfp4}, m={m} n={n} k={k}");
    let (mut lcg, mut bits) = (Lcg::seeded(0xe9 ^ (m * n + k) as u64), 0xe9 ^ n as u64);
    let act = side(&mut lcg, &mut bits, Elem::Bf16, None, m, k);
    let w = side(&mut lcg, &mut bits, Elem::Bf16, None, 2 * n, k);
    let dot = |r: usize, c: usize| {
        (0..k)
            .map(|i| f64::from(act.value(r, k, i) * w.value(c, k, i)))
            .sum()
    };
    let want: Vec<f64> = (0..m * n)
        .map(|i| glu(Glu::Swiglu, dot(i / n, i % n), dot(i / n, n + i % n)))
        .collect();
    let amax = want.iter().fold(0f64, |a, v| a.max(v.abs()));
    let block = if nvfp4 { 16 } else { 128 };
    let gscale = (448.0 * 6.0 / amax) as f32;
    let mut gpu = Gpu::open();
    let (data, scales) = (gpu.zeros(m * n), gpu.zeros(atom_bytes(m, n, block)));
    let out = match nvfp4 {
        false => Out::Fp8Block { data, scales },
        true => Out::Nvfp4 {
            data,
            scales,
            gscale,
        },
    };
    let epi = Epilogue {
        glu: Some(Glu::Swiglu),
        out,
        ..Epilogue::STORE
    };
    let plan = Plan {
        swap: false,
        pair: false,
        bn: 256,
        stages: 0,
        splits: 0,
    };
    let a_op = Operand::plain(gpu.up(&act.bytes), Elem::Bf16);
    let w_op = Operand::plain(gpu.up(&w.bytes), Elem::Bf16);
    let dims = [m as u32, n as u32, k as u32];
    gemm_epi_with(&gpu.ctx(), a_op, w_op, 0, dims, 1.0, epi, plan)
        .unwrap_or_else(|e| panic!("{what}: {e}"));
    gpu.sync();
    let codes: Vec<u8> = gpu.down(data, if nvfp4 { m * n / 2 } else { m * n });
    let atoms: Vec<u8> = gpu.down(scales, atom_bytes(m, n, block));
    for (i, &v) in want.iter().enumerate() {
        let (r, c) = (i / n, i % n);
        let scale = atoms[atom_offset(r, c / block, n / block)];
        let got = if nvfp4 {
            let byte = codes[i / 2];
            let code = if c % 2 == 0 { byte & 0xf } else { byte >> 4 };
            f64::from(e2m1(code)) * f64::from(e4m3(scale)) / f64::from(gscale)
        } else {
            f64::from(e4m3(codes[i])) * 2f64.powi(i32::from(scale) - 127)
        };
        // The block's quantum, and a value's own rounding.
        let b0 = r * n + c / block * block;
        let peak = want[b0..b0 + block]
            .iter()
            .fold(0f64, |a, x| a.max(x.abs()));
        let slack = match nvfp4 {
            true => peak / 6.0 * 1.2 + v.abs() / 8.0 + 1e-6,
            false => v.abs() / 14.0 + peak / 448.0 + 1e-6,
        };
        assert!(
            (got - v).abs() <= slack,
            "{what}: y[{r}][{c}] = {got}, want {v}"
        );
    }
}

#[test]
fn the_tcgen05_epilogues_land_quantized_rows() {
    if !supported() {
        eprintln!("skipping: no tcgen05 on this device");
        return;
    }
    check(false, [256, 512, 256]);
    check(true, [256, 512, 256]);
}
