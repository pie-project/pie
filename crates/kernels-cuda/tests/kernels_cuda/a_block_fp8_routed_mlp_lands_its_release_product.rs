#![cfg(feature = "cuda")]

//! A block-FP8 release's routed MLP (GLM-5.3's experts: e4m3 codes under one
//! f32 factor a 128x128 block), its factors requantized to ue8m0 at load as
//! vLLM's DeepGEMM path does on Blackwell (`fp8_tile_requant`) and spread a
//! row an atom row, against e4m3 activations in 1x128 blocks (W8A8): the
//! selects apart and the experts kept by expert each land the release's
//! product (the f32 walk of `code * factor`) to e4m3's rounding, at a
//! decode's width and a prefill's.

use crate::common::{Gpu, Lcg, from_bf16};
use dtype::Dtype;
use kernels_cuda::linear::act_quant::{self, Format, fp8_tile_requant, scales_to_atoms};
use kernels_cuda::linear::mlp::swiglu;
use kernels_cuda::linear::moe::{ExpertTable, GluBank, experts, weighted_sum};
use kernels_cuda::linear::tc5::{Elem, Glu, atom_bytes, supported};
use kernels_cuda::tensor::Tensor;

fn e4m3(b: u8) -> f32 {
    let s = if b & 0x80 != 0 { -1.0 } else { 1.0 };
    let e = i32::from((b >> 3) & 0xF);
    let m = f32::from(b & 7);
    s * if e == 0 {
        m / 8.0 * 2f32.powi(-6)
    } else {
        (1.0 + m / 8.0) * 2f32.powi(e - 7)
    }
}

/// A bank `[experts][n][k]` as a release stores it: finite codes, factors
/// off powers of two. Returns the device codes (requantized and laid),
/// atoms, and the release's f32 weights.
struct Fp8Bank {
    codes: u64,
    atoms: u64,
    w: Vec<f32>,
}

fn bank(gpu: &mut Gpu, experts: usize, n: usize, k: usize, seed: u64) -> Fp8Bank {
    let mut bits = seed | 1;
    let mut next = || {
        bits ^= bits << 13;
        bits ^= bits >> 7;
        bits ^= bits << 17;
        bits
    };
    // Exponents at most 7: magnitudes up to 1.875, never NaN.
    let codes: Vec<u8> = (0..experts * n * k)
        .map(|_| {
            let r = next();
            ((r & 0x3F) as u8) | (((r >> 8) & 1) as u8) << 7
        })
        .collect();
    let (rb, cb) = (experts * n / 128, k / 128);
    let factors: Vec<f32> = (0..rb * cb)
        .map(|i| 0.002 * (1.0 + 0.37 * ((i * 7 % 11) as f32) / 11.0))
        .collect();
    let w: Vec<f32> = (0..experts * n * k)
        .map(|i| {
            let (row, col) = (i / k, i % k);
            e4m3(codes[i]) * factors[(row / 128) * cb + col / 128]
        })
        .collect();
    let codes_at = gpu.up(&codes);
    let factors_at = gpu.up(&factors);
    let exps = gpu.zeros(rb * cb);
    let atoms = gpu.zeros(atom_bytes(experts * n, k, 128));
    let ctx = gpu.ctx();
    fp8_tile_requant(
        &ctx,
        "test",
        codes_at,
        factors_at,
        exps,
        (experts * n) as u32,
        k as u32,
    )
    .expect("the requant fires");
    scales_to_atoms(
        &ctx,
        "test",
        exps,
        atoms,
        (experts * n) as u32,
        cb as u32,
        128,
    )
    .expect("the atoms land");
    Fp8Bank {
        codes: codes_at,
        atoms,
        w,
    }
}

fn routes_of(tokens: usize, experts: usize, top: usize) -> Vec<i32> {
    (0..tokens)
        .flat_map(|t| (0..top).map(move |j| ((t * 5 + j * 7 + t / 3) % experts) as i32))
        .collect()
}

fn check(tokens: usize, hidden: usize, inter: usize, n_experts: usize, top: usize) {
    let mut gpu = Gpu::open();
    let mut lcg = Lcg::seeded(0xf8 ^ tokens as u64);
    let routes = routes_of(tokens, n_experts, top);
    let count = routes.len();
    let (x_raw, x_f) = lcg.row(tokens * hidden);
    let weights: Vec<f32> = (0..count).map(|i| 0.1 + (i % 7) as f32 * 0.05).collect();
    let up = bank(&mut gpu, n_experts, 2 * inter, hidden, 0x71);
    let down = bank(&mut gpu, n_experts, hidden, inter, 0x72);
    let x = Tensor::new(gpu.up(&x_raw), tokens as u32, hidden as u32, Dtype::Bf16);
    let routes_t = Tensor::new(gpu.up(&routes), tokens as u32, top as u32, Dtype::I32);
    let weights_t = Tensor::new(gpu.up(&weights), tokens as u32, top as u32, Dtype::F32);
    let ctx = gpu.ctx().with_tuning(kernels_cuda::tuning::Tuning {
        moe_w4a8_fused_from: 1,
        ..crate::common::SM100
    });
    let select = |gpu: &mut Gpu, x: Tensor, b: &Fp8Bank, n: usize| {
        let mut y = Tensor::new(
            gpu.zeros(count * n * 2),
            count as u32,
            n as u32,
            Dtype::Bf16,
        );
        act_quant::select(
            &ctx,
            "test",
            Format::E4m3 { gran: 128 },
            x,
            routes_t,
            None,
            act_quant::Bank {
                codes: b.codes,
                atoms: b.atoms,
                experts: n_experts as u32,
                n: n as u32,
                k: x.width,
                elem: Elem::E4m3,
                gran: 128,
                alpha_e: 0,
                alpha_from: None,
            },
            &mut y,
        )
        .expect("the select fires");
        y
    };
    // Apart.
    let packed = select(&mut gpu, x, &up, 2 * inter);
    let mut h = Tensor::new(
        gpu.zeros(count * inter * 2),
        count as u32,
        inter as u32,
        Dtype::Bf16,
    );
    swiglu(&ctx, packed, inter as u32, 1, &mut h).expect("the GLU fires");
    let r = select(&mut gpu, h, &down, hidden);
    let mut apart = Tensor::new(
        gpu.zeros(tokens * hidden * 2),
        tokens as u32,
        hidden as u32,
        Dtype::Bf16,
    );
    weighted_sum(&ctx, r, weights_t, &mut apart).expect("the sum fires");
    // Kept by expert.
    let mut kept = Tensor::new(
        gpu.zeros(tokens * hidden * 2),
        tokens as u32,
        hidden as u32,
        Dtype::Bf16,
    );
    let fp8 = |b: &Fp8Bank, n: usize, k: usize| GluBank::Fp8A8 {
        codes: Tensor::new(b.codes, n_experts as u32, (n * k) as u32, Dtype::E4m3),
        atoms: Tensor::new(b.atoms, n_experts as u32, (n * k / 128) as u32, Dtype::U8),
    };
    let took = experts(
        &ctx,
        x,
        fp8(&up, 2 * inter, hidden),
        Glu::Swiglu,
        fp8(&down, hidden, inter),
        routes_t,
        weights_t,
        inter as u32,
        &mut kept,
        ExpertTable::RESIDENT,
        ExpertTable::RESIDENT,
        None,
    )
    .expect("the routed MLP fires");
    // A decode's few routes run apart (masked, below `MASKED_FIRE`); a
    // prefill's are kept by expert.
    assert_eq!(
        took,
        count > 64,
        "the grouped GEMM keeps {count} routes by expert"
    );
    gpu.sync();
    // The release's product, in f32 over the bf16 activations.
    let mut want = vec![0f32; tokens * hidden];
    for t in 0..tokens {
        let xr = &x_f[t * hidden..][..hidden];
        let xr: Vec<f32> = xr
            .iter()
            .map(|v| from_bf16(crate::common::to_bf16(*v)))
            .collect();
        for j in 0..top {
            let e = routes[t * top + j] as usize;
            let wt = weights[t * top + j];
            let gu: Vec<f32> = (0..2 * inter)
                .map(|row| {
                    let w = &up.w[(e * 2 * inter + row) * hidden..][..hidden];
                    w.iter().zip(&xr).map(|(a, b)| a * b).sum()
                })
                .collect();
            let act: Vec<f32> = (0..inter)
                .map(|i| gu[i] / (1.0 + (-gu[i]).exp()) * gu[inter + i])
                .collect();
            for (o, out) in want[t * hidden..][..hidden].iter_mut().enumerate() {
                let w = &down.w[(e * hidden + o) * inter..][..inter];
                *out += wt * w.iter().zip(&act).map(|(a, b)| a * b).sum::<f32>();
            }
        }
    }
    let scale = want.iter().map(|w| w.abs()).sum::<f32>() / want.len() as f32;
    let distance = |y: &[u16]| -> f32 {
        y.iter()
            .zip(&want)
            .map(|(g, w)| (from_bf16(*g) - w).abs())
            .sum::<f32>()
            / want.len() as f32
            / scale
    };
    let apart_d = distance(&gpu.down::<u16>(apart.ptr, tokens * hidden));
    let kept_d = if took {
        distance(&gpu.down::<u16>(kept.ptr, tokens * hidden))
    } else {
        apart_d
    };
    eprintln!("{tokens} tokens: apart {apart_d:.4}, kept {kept_d:.4} from the release's product");
    assert!(
        apart_d < 0.1 && kept_d <= 1.1 * apart_d,
        "{tokens} tokens: apart {apart_d:.4}, kept {kept_d:.4} from the release's product"
    );
}

#[test]
fn a_block_fp8_routed_mlp_lands_its_release_product() {
    if !supported() {
        return;
    }
    // A decode's 16 routes over 16 experts, and a prefill.
    check(2, 512, 256, 16, 8);
    check(160, 512, 256, 16, 4);
}
