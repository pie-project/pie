#![cfg(feature = "cuda")]

//! DeepSeek-V4-Flash's grouped output projection on fp8 planes
//! (`matmul_grouped_fp8`: e4m3 codes under one ue8m0 exponent a 128x128
//! block, against the rows quantized to e4m3 in 1x128 blocks) answers an
//! f32 walk over the weights' values and the rows' read-back (on the
//! mma.sync kernel, and past its rows as routes on the grouped tcgen05
//! GEMM), and (`--ignored`) its pace against
//! cuBLAS's bf16 strided batched GEMM, past L2.

use crate::common::{Gpu, Lcg, from_bf16};
use dtype::Dtype;
use kernels_cuda::linear::act_quant::{self, Format, Source};
use kernels_cuda::linear::moe::{
    Fp8Shape, dense_fp8_shape, matmul_grouped_fp8, matmul_grouped_fp8_with,
};
use kernels_cuda::tensor::Tensor;

/// An e4m3 code's value.
fn e4m3(b: u8) -> f32 {
    let sign = if b & 0x80 != 0 { -1.0 } else { 1.0 };
    let e = i32::from((b >> 3) & 0xF);
    let m = f32::from(b & 0x7);
    sign * if e == 0 {
        m / 8.0 * 2f32.powi(-6)
    } else {
        (1.0 + m / 8.0) * 2f32.powi(e - 7)
    }
}

/// Codes that are finite e4m3 (no 0x7F/0xFF NaNs).
fn codes(seed: u64, count: usize) -> Vec<u8> {
    let mut state = seed;
    (0..count)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let b = (state >> 40) as u8;
            if b & 0x7F == 0x7F { b & 0xF7 } else { b }
        })
        .collect()
}

/// `plane` (`[rows / 128][k / 128]` exponents) spread to its rows in
/// tcgen05's atoms, as the load lays an fp8b128 plane's.
fn atoms(gpu: &mut Gpu, plane: &[u8], rows: usize, k: usize) -> u64 {
    let at = gpu.up(plane);
    let atoms = gpu.zeros(kernels_cuda::linear::tc5::atom_bytes(rows, k, 128));
    act_quant::scales_to_atoms(
        &gpu.ctx(),
        "test",
        at,
        atoms,
        rows as u32,
        (k / 128) as u32,
        128,
    )
    .expect("the atoms lay");
    atoms
}

fn check(rows: usize, groups: usize, k: usize, n: usize) {
    check_with(rows, groups, k, n, None);
}

/// How [`check_with`] runs the projection.
#[derive(Clone, Copy)]
enum Arm {
    /// The mma.sync kernel, in a CTA shape (the grouped projection's where none).
    Mma(Option<Fp8Shape>),
    /// A row's groups as its routes on the grouped tcgen05 GEMM, as a wide
    /// fire's rows run it.
    Routed,
}

/// [`check`] in the CTA shape `shape` (the grouped projection's where none).
fn check_with(rows: usize, groups: usize, k: usize, n: usize, shape: Option<Fp8Shape>) {
    check_arm(rows, groups, k, n, Arm::Mma(shape));
}

fn check_arm(rows: usize, groups: usize, k: usize, n: usize, arm: Arm) {
    let mut lcg = Lcg::seeded(0xf8 + rows as u64);
    let (x_bits, _) = lcg.row(rows * groups * k);
    let w = codes(0x51 + n as u64, groups * n * k);
    // Exponents 2^-9 .. 2^-3 (the checkpoint's sit near 2^-8).
    let sk = k / 128;
    let s: Vec<u8> = codes(0x77, groups * n / 128 * sk)
        .into_iter()
        .map(|b| 118 + b % 7)
        .collect();
    let mut gpu = Gpu::open();
    let x_at = gpu.up(&x_bits);
    let back_at = gpu.zeros(rows * groups * k * 2);
    let w_at = gpu.up(&w);
    let s_at = atoms(&mut gpu, &s, groups * n, k);
    let y_at = gpu.zeros(rows * groups * n * 2);
    let ctx = gpu.ctx();
    let x = Tensor::new(x_at, rows as u32, (groups * k) as u32, Dtype::Bf16);
    act_quant::read_back(
        &ctx,
        "test",
        Format::E4m3 { gran: 128 },
        Source::rows(x, None),
        back_at,
    )
    .expect("the read-back fires");
    let mut y = Tensor::new(y_at, rows as u32, (groups * n) as u32, Dtype::Bf16);
    match arm {
        Arm::Mma(None) => matmul_grouped_fp8(&ctx, x, w_at, s_at, &mut y, groups as u32),
        Arm::Mma(Some(shape)) => {
            matmul_grouped_fp8_with(&ctx, x, w_at, s_at, &mut y, groups as u32, shape)
        }
        Arm::Routed => {
            let ctx = ctx.with_tuning(crate::common::SM100);
            let routes_at = gpu.zeros(rows * groups * 4);
            let mut routes = Tensor::new(routes_at, rows as u32, groups as u32, Dtype::I32);
            kernels_cuda::linear::moe_route::group_routes(&ctx, groups as u32, &mut routes)
                .expect("the routes lay");
            let xr = Tensor::new(x_at, (rows * groups) as u32, k as u32, Dtype::Bf16);
            let mut yr = Tensor::new(y_at, (rows * groups) as u32, n as u32, Dtype::Bf16);
            act_quant::select(
                &ctx,
                "test",
                Format::E4m3 { gran: 128 },
                xr,
                routes,
                None,
                act_quant::Bank {
                    codes: w_at,
                    atoms: s_at,
                    experts: groups as u32,
                    n: n as u32,
                    k: k as u32,
                    elem: kernels_cuda::linear::tc5::Elem::E4m3,
                    gran: 128,
                    alpha_e: 0,
                    alpha_from: None,
                },
                &mut yr,
            )
        }
    }
    .expect("the fp8 grouped projection fires");
    gpu.sync();
    let backs: Vec<f32> = gpu
        .down::<u16>(back_at, rows * groups * k)
        .into_iter()
        .map(from_bf16)
        .collect();
    let got: Vec<u16> = gpu.down(y_at, rows * groups * n);
    let mut worst = 0f32;
    for r in 0..rows {
        for g in 0..groups {
            for c in 0..n {
                let row = g * n + c;
                let mut acc = 0f64;
                for i in 0..k {
                    let scale = 2f64.powi(i32::from(s[(row / 128) * sk + i / 128]) - 127);
                    acc += f64::from(e4m3(w[row * k + i]))
                        * scale
                        * f64::from(backs[(r * groups + g) * k + i]);
                }
                let v = f64::from(from_bf16(got[(r * groups + g) * n + c]));
                worst = worst.max(((v - acc).abs() / acc.abs().max(1.0)) as f32);
            }
        }
    }
    assert!(
        worst <= 1e-2,
        "{rows} rows, {groups} groups of [{n}, {k}]: {worst} relative"
    );
}

#[test]
fn the_fp8_grouped_projection_answers_each_group() {
    // One group, a dense projection, in each shape `dense_fp8_shape` picks.
    for (rows, n, k) in [
        (2, 4096, 2048),
        (4, 8192, 1024),
        (4, 32768, 1024),
        (8, 1024, 4096),
        (8, 4096, 2048),
        (8, 8192, 1024),
    ] {
        let shape = dense_fp8_shape(rows as u32, n as u32, k as u32)
            .unwrap_or_else(|| panic!("the mma.sync kernel takes {rows} rows of [{n}, {k}]"));
        check_with(rows, 1, k, n, Some(shape));
    }
    check(1, 8, 4096, 1024);
    check(3, 2, 1024, 256);
    check(8, 4, 512, 128);
    // Past the mma.sync kernel's rows: each row's groups as its routes on the
    // grouped tcgen05 GEMM (DeepSeek-V4.1's `wo_a` at decode64).
    if kernels_cuda::linear::tc5::supported() {
        check_arm(16, 8, 4096, 1024, Arm::Routed);
        check_arm(64, 8, 4096, 1024, Arm::Routed);
        check_arm(37, 4, 1024, 256, Arm::Routed);
    }
}
