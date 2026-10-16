#![cfg(feature = "cuda")]

//! `linear::moe::experts`: the routed MLP with its rows kept by expert from
//! the gate-up select to the weighted sum answers what the gate-up select
//! with its GLU, the down select and the weighted sum answer apart.

use crate::common::{Gpu, Lcg, close, from_bf16, to_bf16};
use dtype::Dtype;
use kernels_cuda::linear::act_quant::{self, AlphaFrom, Format, scales_to_atoms};
use kernels_cuda::linear::mlp::{swiglu, swiglu_clamp, swiglu_clamp_routed};
use kernels_cuda::linear::moe::{
    ExpertTable, GluBank, GroupSeat, experts, experts_weighted_first, matmul_select,
    matmul_select_quant, select_glu, sigmoid_gate_add, sum_add, weighted_sum,
};
use kernels_cuda::linear::tc5::{Elem, Glu, atom_bytes, supported};
use kernels_cuda::tensor::Tensor;

fn routes_of(tokens: usize, experts: usize, top: usize) -> Vec<i32> {
    (0..tokens)
        .flat_map(|t| (0..top).map(move |j| ((t * 5 + j * 7 + t / 3) % experts) as i32))
        .collect()
}

#[derive(Clone, Copy)]
enum Bank {
    Bf16,
    Mxfp4,
}

fn check(tokens: usize, hidden: usize, inter: usize, bank: Bank, n_experts: usize, top: usize) {
    for gated in [false, true] {
        check_in(tokens, hidden, inter, bank, n_experts, top, None, gated);
    }
}

/// `window`: (live tokens, first live token), armed as the context's stage.
#[allow(clippy::too_many_arguments)]
fn check_in(
    tokens: usize,
    hidden: usize,
    inter: usize,
    bank: Bank,
    n_experts: usize,
    top: usize,
    window: Option<(u32, u32)>,
    gated: bool,
) {
    #[allow(non_snake_case)]
    let (EXPERTS, TOP) = (n_experts, top);
    let mut gpu = Gpu::open();
    let mut lcg = Lcg::seeded(0x3e ^ tokens as u64);
    let routes = routes_of(tokens, EXPERTS, TOP);
    let count = routes.len();
    let (x_raw, _) = lcg.row(tokens * hidden);
    let weights: Vec<f32> = (0..count).map(|i| 0.1 + (i % 7) as f32 * 0.05).collect();
    let x = Tensor::new(gpu.up(&x_raw), tokens as u32, hidden as u32, Dtype::Bf16);
    let routes_t = Tensor::new(gpu.up(&routes), tokens as u32, TOP as u32, Dtype::I32);
    let weights_t = Tensor::new(gpu.up(&weights), tokens as u32, TOP as u32, Dtype::F32);
    // A bank of `rows` rows of `k` an expert.
    let mut bank_of = |rows: usize, k: usize, seed: u64| -> GluBank {
        match bank {
            Bank::Bf16 => {
                let mut l = Lcg::seeded(seed);
                let (raw, _) = l.row(EXPERTS * rows * k);
                let raw: Vec<u16> = raw.iter().map(|&b| to_bf16(from_bf16(b) * 0.25)).collect();
                GluBank::Bf16(Tensor::new(
                    gpu.up(&raw),
                    EXPERTS as u32,
                    (rows * k) as u32,
                    Dtype::Bf16,
                ))
            }
            Bank::Mxfp4 => {
                let mut l = Lcg::seeded(seed);
                let codes: Vec<u8> = (0..EXPERTS * rows * k / 2)
                    .map(|_| ((l.unit() + 1.0) * 127.0) as u8)
                    .collect();
                let scales: Vec<u8> = (0..EXPERTS * rows * k / 32)
                    .map(|_| 120 + (((l.unit() + 1.0) * 3.0) as u8).min(5))
                    .collect();
                GluBank::Mxfp4 {
                    codes: Tensor::new(
                        gpu.up(&codes),
                        EXPERTS as u32,
                        (rows * k / 2) as u32,
                        Dtype::U8,
                    ),
                    scales: Tensor::new(
                        gpu.up(&scales),
                        EXPERTS as u32,
                        (rows * k / 32) as u32,
                        Dtype::U8,
                    ),
                    bias: None,
                }
            }
        }
    };
    let up = bank_of(2 * inter, hidden, 0x51);
    let down = bank_of(hidden, inter, 0x52);
    let ctx = gpu.ctx().with_tuning(crate::common::TC5);
    if let Some((live, first)) = window {
        ctx.arm_stage(gpu.up(&[live, first]));
    }
    let glu = Glu::Swiglu;

    // Apart: the gate-up select with its GLU, the down select, the sum.
    let h = Tensor::new(
        gpu.zeros(count * inter * 2),
        count as u32,
        inter as u32,
        Dtype::Bf16,
    );
    let r = Tensor::new(
        gpu.zeros(count * hidden * 2),
        count as u32,
        hidden as u32,
        Dtype::Bf16,
    );
    let apart = Tensor::new(
        gpu.zeros(tokens * hidden * 2),
        tokens as u32,
        hidden as u32,
        Dtype::Bf16,
    );
    let mut h_out = h;
    assert!(
        select_glu(
            &ctx,
            x,
            up,
            routes_t,
            &mut h_out,
            glu,
            ExpertTable::RESIDENT
        )
        .expect("fires"),
        "the gate-up select lands its GLU"
    );
    let mut r_out = r;
    match down {
        GluBank::Bf16(b) => {
            matmul_select(&ctx, h, b, routes_t, &mut r_out, ExpertTable::RESIDENT).expect("fires")
        }
        GluBank::Mxfp4 { codes, scales, .. } => matmul_select_quant(
            &ctx,
            h,
            codes,
            scales,
            None,
            routes_t,
            &mut r_out,
            GroupSeat::RESIDENT,
        )
        .expect("fires"),
        _ => unreachable!(),
    }
    let mut apart_out = apart;
    weighted_sum(&ctx, r, weights_t, &mut apart_out).expect("the sum fires");
    // Under a gate, Qwen's shared expert row added at its sigmoid.
    let gate = gated.then(|| {
        let mut l = Lcg::seeded(0x77 ^ tokens as u64);
        let (shared_raw, _) = l.row(tokens * hidden);
        let (gate_raw, _) = l.row(tokens);
        (
            Tensor::new(
                gpu.up(&shared_raw),
                tokens as u32,
                hidden as u32,
                Dtype::Bf16,
            ),
            Tensor::new(gpu.up(&gate_raw), tokens as u32, 1, Dtype::Bf16),
        )
    });
    let apart = match gate {
        Some((shared, g)) => {
            let mut gated_out = Tensor::new(
                gpu.zeros(tokens * hidden * 2),
                tokens as u32,
                hidden as u32,
                Dtype::Bf16,
            );
            sigmoid_gate_add(&ctx, apart, shared, g, &mut gated_out).expect("the gate-add fires");
            gated_out
        }
        None => apart,
    };

    // Kept by expert.
    let kept = Tensor::new(
        gpu.zeros(tokens * hidden * 2),
        tokens as u32,
        hidden as u32,
        Dtype::Bf16,
    );
    let mut kept_out = kept;
    let took = experts(
        &ctx,
        x,
        up,
        glu,
        down,
        routes_t,
        weights_t,
        inter as u32,
        &mut kept_out,
        ExpertTable::RESIDENT,
        ExpertTable::RESIDENT,
        gate,
    )
    .expect("the routed MLP fires");
    assert!(took, "the grouped GEMM takes a {tokens}-token fire");
    gpu.sync();
    let want: Vec<u16> = gpu.down(apart.ptr, tokens * hidden);
    let got: Vec<u16> = gpu.down(kept.ptr, tokens * hidden);
    let same = got.iter().zip(&want).filter(|(a, b)| a == b).count();
    let (lo, hi) = window.map_or((0, tokens), |(live, first)| {
        (first as usize, (first + live) as usize)
    });
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        if i / hidden < lo || i / hidden >= hi {
            continue;
        }
        assert!(
            close(from_bf16(*g), from_bf16(*w)),
            "token {} column {}: {} kept by expert, {} apart ({same} of {} bit-equal)",
            i / hidden,
            i % hidden,
            from_bf16(*g),
            from_bf16(*w),
            got.len()
        );
    }
}

/// An NVFP4 bank of `rows` rows of `k` an expert, against NVFP4
/// activations: its atoms, a tensor scale a row (one an expert), an input
/// scale an expert.
fn w4a4_bank(gpu: &mut Gpu, experts: usize, rows: usize, k: usize, seed: u64) -> GluBank {
    let mut l = Lcg::seeded(seed);
    let codes: Vec<u8> = (0..experts * rows * k / 2)
        .map(|_| ((l.unit() + 1.0) * 127.0) as u8)
        .collect();
    let scales: Vec<u8> = (0..experts * rows * k / 16)
        .map(|i| 0x30 + (i * 5 % 16) as u8)
        .collect();
    let tensor: Vec<f32> = (0..experts * rows)
        .map(|r| 0.5 + (r / rows) as f32 * 0.0625)
        .collect();
    let input: Vec<f32> = (0..experts).map(|e| 0.004 + 0.003 * e as f32).collect();
    let scales_at = gpu.up(&scales);
    let atoms_at = gpu.zeros(atom_bytes(experts * rows, k, 16));
    scales_to_atoms(
        &gpu.ctx(),
        "test.w4a4_bank",
        scales_at,
        atoms_at,
        (experts * rows) as u32,
        (k / 16) as u32,
        1,
    )
    .expect("the bank's atoms land");
    let e = experts as u32;
    GluBank::Nvfp4A4 {
        codes: Tensor::new(gpu.up(&codes), e, (rows * k / 2) as u32, Dtype::U8),
        atoms: Tensor::new(atoms_at, e, (rows * k / 16) as u32, Dtype::U8),
        tensor_scales: Tensor::new(gpu.up(&tensor), e, rows as u32, Dtype::F32),
        input_scales: Tensor::new(gpu.up(&input), e, 1, Dtype::F32),
    }
}

/// The W4A4 routed MLP kept by expert, against its four apart: each
/// select quantizing its activations under its experts' globals, the
/// SwiGLU's bf16 rows between them, then the weighted sum.
fn check_w4a4(
    tokens: usize,
    hidden: usize,
    inter: usize,
    n_experts: usize,
    top: usize,
    gated: bool,
) {
    let mut gpu = Gpu::open();
    let mut lcg = Lcg::seeded(0x4a4 ^ tokens as u64);
    let routes = routes_of(tokens, n_experts, top);
    let count = routes.len();
    let (x_raw, _) = lcg.row(tokens * hidden);
    let weights: Vec<f32> = (0..count).map(|i| 0.1 + (i % 7) as f32 * 0.05).collect();
    let x = Tensor::new(gpu.up(&x_raw), tokens as u32, hidden as u32, Dtype::Bf16);
    let routes_t = Tensor::new(gpu.up(&routes), tokens as u32, top as u32, Dtype::I32);
    let weights_t = Tensor::new(gpu.up(&weights), tokens as u32, top as u32, Dtype::F32);
    let up = w4a4_bank(&mut gpu, n_experts, 2 * inter, hidden, 0x61);
    let down = w4a4_bank(&mut gpu, n_experts, hidden, inter, 0x62);
    // B200's own decisions (the fused MLP from the first contiguous fire).
    let ctx = gpu.ctx().with_tuning(crate::common::SM100);
    // Apart.
    let select_apart =
        |ctx: &kernels_cuda::Ctx, gpu: &mut Gpu, x: Tensor, bank: GluBank, n: usize| {
            let GluBank::Nvfp4A4 {
                codes,
                atoms,
                tensor_scales,
                input_scales,
            } = bank
            else {
                unreachable!()
            };
            let k = x.width;
            let mut y = Tensor::new(
                gpu.zeros(count * n * 2),
                count as u32,
                n as u32,
                Dtype::Bf16,
            );
            let alpha = gpu.zeros(n_experts * 4);
            act_quant::select(
                ctx,
                "test.w4a4_apart",
                Format::Nvfp4,
                x,
                routes_t,
                Some(input_scales),
                act_quant::Bank {
                    codes: codes.ptr,
                    atoms: atoms.ptr,
                    experts: n_experts as u32,
                    n: n as u32,
                    k,
                    elem: Elem::E2m1,
                    gran: 16,
                    alpha_e: alpha,
                    alpha_from: Some(AlphaFrom {
                        input_scales: input_scales.ptr,
                        tensor_scales: tensor_scales.ptr,
                        rows: n as u32,
                    }),
                },
                &mut y,
            )
            .expect("the select fires");
            y
        };
    let packed = select_apart(&ctx, &mut gpu, x, up, 2 * inter);
    let mut h = Tensor::new(
        gpu.zeros(count * inter * 2),
        count as u32,
        inter as u32,
        Dtype::Bf16,
    );
    swiglu(&ctx, packed, inter as u32, 1, &mut h).expect("the GLU fires");
    let r = select_apart(&ctx, &mut gpu, h, down, hidden);
    let mut apart = Tensor::new(
        gpu.zeros(tokens * hidden * 2),
        tokens as u32,
        hidden as u32,
        Dtype::Bf16,
    );
    weighted_sum(&ctx, r, weights_t, &mut apart).expect("the sum fires");
    let gate = gated.then(|| {
        let mut l = Lcg::seeded(0x78 ^ tokens as u64);
        let (shared_raw, _) = l.row(tokens * hidden);
        let (gate_raw, _) = l.row(tokens);
        (
            Tensor::new(
                gpu.up(&shared_raw),
                tokens as u32,
                hidden as u32,
                Dtype::Bf16,
            ),
            Tensor::new(gpu.up(&gate_raw), tokens as u32, 1, Dtype::Bf16),
        )
    });
    let apart = match gate {
        Some((shared, g)) => {
            let mut gated_out = Tensor::new(
                gpu.zeros(tokens * hidden * 2),
                tokens as u32,
                hidden as u32,
                Dtype::Bf16,
            );
            sigmoid_gate_add(&ctx, apart, shared, g, &mut gated_out).expect("the gate-add fires");
            gated_out
        }
        None => apart,
    };
    // Kept by expert.
    let mut kept = Tensor::new(
        gpu.zeros(tokens * hidden * 2),
        tokens as u32,
        hidden as u32,
        Dtype::Bf16,
    );
    let took = experts(
        &ctx,
        x,
        up,
        Glu::Swiglu,
        down,
        routes_t,
        weights_t,
        inter as u32,
        &mut kept,
        ExpertTable::RESIDENT,
        ExpertTable::RESIDENT,
        gate,
    )
    .expect("the routed MLP fires");
    assert!(took, "the grouped GEMM takes a {tokens}-token W4A4 fire");
    gpu.sync();
    let want: Vec<u16> = gpu.down(apart.ptr, tokens * hidden);
    let got: Vec<u16> = gpu.down(kept.ptr, tokens * hidden);
    // The SiLU's fast quotient, and under a gate the shared row added before
    // the sum rounds: close to what apart lands, not bit for bit.
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        assert!(
            close(from_bf16(*g), from_bf16(*w)),
            "token {} column {}: {} kept by expert, {} apart",
            i / hidden,
            i % hidden,
            from_bf16(*g),
            from_bf16(*w),
        );
    }
}

/// An MXFP4 bank of `rows` rows of `k` an expert, against e4m3 activations
/// in 1x128 blocks (its codes and its ue8m0 scales laid in atoms), and the
/// same bank against bf16 activations (its scales row-major).
fn w4a8_bank(
    gpu: &mut Gpu,
    experts: usize,
    rows: usize,
    k: usize,
    seed: u64,
) -> (GluBank, GluBank) {
    let mut l = Lcg::seeded(seed);
    let codes: Vec<u8> = (0..experts * rows * k / 2)
        .map(|_| ((l.unit() + 1.0) * 127.0) as u8)
        .collect();
    let scales: Vec<u8> = (0..experts * rows * k / 32)
        .map(|_| 120 + (((l.unit() + 1.0) * 3.0) as u8).min(5))
        .collect();
    let scales_at = gpu.up(&scales);
    let atoms_at = gpu.zeros(atom_bytes(experts * rows, k, 32));
    scales_to_atoms(
        &gpu.ctx(),
        "test.w4a8_bank",
        scales_at,
        atoms_at,
        (experts * rows) as u32,
        (k / 32) as u32,
        1,
    )
    .expect("the bank's atoms land");
    let e = experts as u32;
    let codes = Tensor::new(gpu.up(&codes), e, (rows * k / 2) as u32, Dtype::U8);
    (
        GluBank::Mxfp4A8 {
            codes,
            atoms: Tensor::new(atoms_at, e, (rows * k / 32) as u32, Dtype::U8),
        },
        GluBank::Mxfp4 {
            codes,
            scales: Tensor::new(scales_at, e, (rows * k / 32) as u32, Dtype::U8),
            bias: None,
        },
    )
}

/// The W4A8 routed MLP kept by expert, against its four apart (each select
/// quantizing its activations to e4m3 in 1x128 blocks, the SwiGLU's bf16
/// rows between them, then the weighted sum): the GLU quantizes from f32 in
/// the epilogue rather than from its bf16 row, so an e4m3 code a rounding
/// apart flips now and then, and the two are held to the same distance
/// from the product of unquantized activations (the banks read W4A16).
fn check_w4a8(
    tokens: usize,
    hidden: usize,
    inter: usize,
    n_experts: usize,
    top: usize,
    gated: bool,
    clamp: Option<f32>,
) {
    let mut gpu = Gpu::open();
    let mut lcg = Lcg::seeded(0x4a8 ^ tokens as u64);
    let routes = routes_of(tokens, n_experts, top);
    let count = routes.len();
    let (x_raw, _) = lcg.row(tokens * hidden);
    let weights: Vec<f32> = (0..count).map(|i| 0.1 + (i % 7) as f32 * 0.05).collect();
    let x = Tensor::new(gpu.up(&x_raw), tokens as u32, hidden as u32, Dtype::Bf16);
    let routes_t = Tensor::new(gpu.up(&routes), tokens as u32, top as u32, Dtype::I32);
    let weights_t = Tensor::new(gpu.up(&weights), tokens as u32, top as u32, Dtype::F32);
    let glu = clamp.map_or(Glu::Swiglu, |limit| Glu::SwigluClamp { limit });
    let (up, up16) = w4a8_bank(&mut gpu, n_experts, 2 * inter, hidden, 0x71);
    let (down, down16) = w4a8_bank(&mut gpu, n_experts, hidden, inter, 0x72);
    // B200's arm, fused from the first token (the profile fuses from 512
    // tokens; the arm is the same at any width).
    let ctx = gpu.ctx().with_tuning(kernels_cuda::tuning::Tuning {
        moe_w4a8_fused_from: 1,
        ..crate::common::SM100
    });
    // Apart.
    let select_apart =
        |ctx: &kernels_cuda::Ctx, gpu: &mut Gpu, x: Tensor, bank: GluBank, n: usize| {
            let GluBank::Mxfp4A8 { codes, atoms } = bank else {
                unreachable!()
            };
            let mut y = Tensor::new(
                gpu.zeros(count * n * 2),
                count as u32,
                n as u32,
                Dtype::Bf16,
            );
            act_quant::select(
                ctx,
                "test.w4a8_apart",
                Format::E4m3 { gran: 128 },
                x,
                routes_t,
                None,
                act_quant::Bank {
                    codes: codes.ptr,
                    atoms: atoms.ptr,
                    experts: n_experts as u32,
                    n: n as u32,
                    k: x.width,
                    elem: Elem::E2m1,
                    gran: 32,
                    alpha_e: 0,
                    alpha_from: None,
                },
                &mut y,
            )
            .expect("the select fires");
            y
        };
    let packed = select_apart(&ctx, &mut gpu, x, up, 2 * inter);
    let mut h = Tensor::new(
        gpu.zeros(count * inter * 2),
        count as u32,
        inter as u32,
        Dtype::Bf16,
    );
    match clamp {
        Some(limit) => swiglu_clamp(&ctx, packed, inter as u32, 1, limit, &mut h),
        None => swiglu(&ctx, packed, inter as u32, 1, &mut h),
    }
    .expect("the GLU fires");
    let r = select_apart(&ctx, &mut gpu, h, down, hidden);
    let mut apart = Tensor::new(
        gpu.zeros(tokens * hidden * 2),
        tokens as u32,
        hidden as u32,
        Dtype::Bf16,
    );
    weighted_sum(&ctx, r, weights_t, &mut apart).expect("the sum fires");
    // Unquantized activations: the gate-up select with its GLU, the down
    // select, the sum, at W4A16.
    let mut h16 = Tensor::new(
        gpu.zeros(count * inter * 2),
        count as u32,
        inter as u32,
        Dtype::Bf16,
    );
    assert!(
        select_glu(
            &ctx,
            x,
            up16,
            routes_t,
            &mut h16,
            glu,
            ExpertTable::RESIDENT
        )
        .expect("fires"),
        "the W4A16 gate-up select lands its GLU"
    );
    let GluBank::Mxfp4 { codes, scales, .. } = down16 else {
        unreachable!()
    };
    let mut r16 = Tensor::new(
        gpu.zeros(count * hidden * 2),
        count as u32,
        hidden as u32,
        Dtype::Bf16,
    );
    matmul_select_quant(
        &ctx,
        h16,
        codes,
        scales,
        None,
        routes_t,
        &mut r16,
        GroupSeat::RESIDENT,
    )
    .expect("fires");
    let mut exact = Tensor::new(
        gpu.zeros(tokens * hidden * 2),
        tokens as u32,
        hidden as u32,
        Dtype::Bf16,
    );
    weighted_sum(&ctx, r16, weights_t, &mut exact).expect("the sum fires");
    let gate = gated.then(|| {
        let mut l = Lcg::seeded(0x79 ^ tokens as u64);
        let (shared_raw, _) = l.row(tokens * hidden);
        let (gate_raw, _) = l.row(tokens);
        (
            Tensor::new(
                gpu.up(&shared_raw),
                tokens as u32,
                hidden as u32,
                Dtype::Bf16,
            ),
            Tensor::new(gpu.up(&gate_raw), tokens as u32, 1, Dtype::Bf16),
        )
    });
    let gated_of = |gpu: &mut Gpu, y: Tensor| match gate {
        Some((shared, g)) => {
            let mut gated_out = Tensor::new(
                gpu.zeros(tokens * hidden * 2),
                tokens as u32,
                hidden as u32,
                Dtype::Bf16,
            );
            sigmoid_gate_add(&ctx, y, shared, g, &mut gated_out).expect("the gate-add fires");
            gated_out
        }
        None => y,
    };
    let apart = gated_of(&mut gpu, apart);
    let exact = gated_of(&mut gpu, exact);
    // Kept by expert.
    let mut kept = Tensor::new(
        gpu.zeros(tokens * hidden * 2),
        tokens as u32,
        hidden as u32,
        Dtype::Bf16,
    );
    let took = experts(
        &ctx,
        x,
        up,
        glu,
        down,
        routes_t,
        weights_t,
        inter as u32,
        &mut kept,
        ExpertTable::RESIDENT,
        ExpertTable::RESIDENT,
        gate,
    )
    .expect("the routed MLP fires");
    assert!(took, "the grouped GEMM takes a {tokens}-token W4A8 fire");
    gpu.sync();
    let exact: Vec<u16> = gpu.down(exact.ptr, tokens * hidden);
    // The mean distance from the unquantized product, relative to its row's
    // mean magnitude.
    let distance = |y: &[u16]| -> f64 {
        let scale = exact
            .iter()
            .map(|w| f64::from(from_bf16(*w)).abs())
            .sum::<f64>()
            / exact.len() as f64;
        y.iter()
            .zip(&exact)
            .map(|(g, w)| f64::from(from_bf16(*g) - from_bf16(*w)).abs())
            .sum::<f64>()
            / exact.len() as f64
            / scale
    };
    let apart_d = distance(&gpu.down::<u16>(apart.ptr, tokens * hidden));
    let kept_d = distance(&gpu.down::<u16>(kept.ptr, tokens * hidden));
    assert!(
        apart_d > 0.0 && apart_d < 0.1,
        "{tokens} tokens: the four apart sit {apart_d:.4} from the unquantized product"
    );
    assert!(
        kept_d <= 1.1 * apart_d,
        "{tokens} tokens: kept by expert {kept_d:.4} from the unquantized product, apart {apart_d:.4}"
    );
}

/// DeepSeek's W4A8 routed MLP kept by expert, against its four apart: the
/// route weights scale the clamped GLU's rows before the down leg quantizes
/// them (`mlp_swiglu_clamp_routed`), and the routes sum plainly with the
/// shared row added (`moe_sum_add`). Held, as the plain W4A8 MLP is, no
/// farther from the product of unquantized activations than the four apart.
fn check_w4a8_weighted_first(
    tokens: usize,
    hidden: usize,
    inter: usize,
    n_experts: usize,
    top: usize,
) {
    let limit = 7.0f32;
    let mut gpu = Gpu::open();
    let mut lcg = Lcg::seeded(0xd54 ^ tokens as u64);
    let routes = routes_of(tokens, n_experts, top);
    let count = routes.len();
    let (x_raw, _) = lcg.row(tokens * hidden);
    let (shared_raw, _) = lcg.row(tokens * hidden);
    let weights: Vec<f32> = (0..count).map(|i| 0.1 + (i % 7) as f32 * 0.05).collect();
    let x = Tensor::new(gpu.up(&x_raw), tokens as u32, hidden as u32, Dtype::Bf16);
    let shared = Tensor::new(
        gpu.up(&shared_raw),
        tokens as u32,
        hidden as u32,
        Dtype::Bf16,
    );
    let routes_t = Tensor::new(gpu.up(&routes), tokens as u32, top as u32, Dtype::I32);
    let weights_t = Tensor::new(gpu.up(&weights), tokens as u32, top as u32, Dtype::F32);
    let (up, up16) = w4a8_bank(&mut gpu, n_experts, 2 * inter, hidden, 0x81);
    let (down, down16) = w4a8_bank(&mut gpu, n_experts, hidden, inter, 0x82);
    let ctx = gpu.ctx().with_tuning(kernels_cuda::tuning::Tuning {
        moe_w4a8_fused_from: 1,
        ..crate::common::SM100
    });
    let bf = |gpu: &mut Gpu, rows: usize, n: usize| {
        Tensor::new(gpu.zeros(rows * n * 2), rows as u32, n as u32, Dtype::Bf16)
    };
    // The four apart, at W4A8 (`a8`) or with unquantized activations.
    let apart = |ctx: &kernels_cuda::Ctx, gpu: &mut Gpu, a8: bool| {
        let select =
            |ctx: &kernels_cuda::Ctx, gpu: &mut Gpu, x: Tensor, bank: GluBank, n: usize| {
                let mut y = bf(gpu, count, n);
                match bank {
                    GluBank::Mxfp4A8 { codes, atoms } => act_quant::select(
                        ctx,
                        "test.w4a8_apart",
                        Format::E4m3 { gran: 128 },
                        x,
                        routes_t,
                        None,
                        act_quant::Bank {
                            codes: codes.ptr,
                            atoms: atoms.ptr,
                            experts: n_experts as u32,
                            n: n as u32,
                            k: x.width,
                            elem: Elem::E2m1,
                            gran: 32,
                            alpha_e: 0,
                            alpha_from: None,
                        },
                        &mut y,
                    )
                    .expect("the select fires"),
                    GluBank::Mxfp4 { codes, scales, .. } => matmul_select_quant(
                        ctx,
                        x,
                        codes,
                        scales,
                        None,
                        routes_t,
                        &mut y,
                        GroupSeat::RESIDENT,
                    )
                    .expect("the select fires"),
                    _ => unreachable!(),
                }
                y
            };
        let (u, d) = if a8 { (up, down) } else { (up16, down16) };
        let packed = select(ctx, gpu, x, u, 2 * inter);
        let mut h = bf(gpu, count, inter);
        swiglu_clamp_routed(ctx, packed, inter as u32, weights_t, 1, limit, &mut h)
            .expect("the GLU fires");
        let r = select(ctx, gpu, h, d, hidden);
        let mut y = bf(gpu, tokens, hidden);
        sum_add(ctx, r, shared, &mut y).expect("the sum fires");
        y
    };
    let apart8 = apart(&ctx, &mut gpu, true);
    let exact = apart(&ctx, &mut gpu, false);
    let mut kept = bf(&mut gpu, tokens, hidden);
    let took = experts_weighted_first(
        &ctx,
        x,
        up,
        Glu::SwigluClamp { limit },
        down,
        routes_t,
        weights_t,
        inter as u32,
        shared,
        &mut kept,
    )
    .expect("the routed MLP fires");
    assert!(
        took,
        "the grouped GEMM takes a {tokens}-token DeepSeek W4A8 fire"
    );
    gpu.sync();
    let exact: Vec<u16> = gpu.down(exact.ptr, tokens * hidden);
    let distance = |y: &[u16]| -> f64 {
        let scale = exact
            .iter()
            .map(|w| f64::from(from_bf16(*w)).abs())
            .sum::<f64>()
            / exact.len() as f64;
        y.iter()
            .zip(&exact)
            .map(|(g, w)| f64::from(from_bf16(*g) - from_bf16(*w)).abs())
            .sum::<f64>()
            / exact.len() as f64
            / scale
    };
    let apart_d = distance(&gpu.down::<u16>(apart8.ptr, tokens * hidden));
    let kept_d = distance(&gpu.down::<u16>(kept.ptr, tokens * hidden));
    assert!(
        apart_d > 0.0 && apart_d < 0.1,
        "{tokens} tokens: the four apart sit {apart_d:.4} from the unquantized product"
    );
    assert!(
        kept_d <= 1.1 * apart_d,
        "{tokens} tokens: kept by expert {kept_d:.4} from the unquantized product, apart {apart_d:.4}"
    );
}

#[test]
fn the_routed_mlp_keeps_its_rows_by_expert_every_case() {
    if !supported() {
        eprintln!("skipped: no tcgen05 on this device");
        return;
    }
    // Masked (at most 64 tokens) and contiguous layouts, bf16 and mxfp4.
    for bank in [Bank::Bf16, Bank::Mxfp4] {
        check(16, 256, 128, bank, 16, 4);
        check(40, 256, 192, bank, 16, 4);
        check(300, 512, 128, bank, 16, 4);
    }
    // Qwen3.6's routing: 256 experts, top 8, at a decode batch and a prefill.
    // A live window of a decode plane, as a captured step fires it.
    check_in(64, 2048, 512, Bank::Bf16, 256, 8, Some((40, 8)), true);
    check_in(64, 256, 128, Bank::Mxfp4, 16, 4, Some((17, 3)), false);
    for tokens in [64, 33, 512] {
        check(tokens, 2048, 512, Bank::Bf16, 256, 8);
    }
    // W4A4 (Flash-Next's NVFP4 against NVFP4): a contiguous fire, what the
    // four apart land.
    for gated in [false, true] {
        check_w4a4(64, 512, 256, 64, 10, gated);
        check_w4a4(300, 256, 128, 16, 4, gated);
    }
    // W4A8 (MiMo's and DeepSeek's MXFP4 against e4m3): contiguous fires, what
    // the four apart land.
    for gated in [false, true] {
        check_w4a8(64, 512, 256, 64, 10, gated, None);
        check_w4a8(300, 256, 128, 16, 4, gated, None);
    }
    // GLM-5.3-Flash's clamped SwiGLU, as the epilogue lands it.
    for gated in [false, true] {
        check_w4a8(300, 256, 128, 16, 4, gated, Some(7.0));
    }
    // DeepSeek's W4A8 (its weights before the down leg, a plain sum and the
    // shared row): its decode64 fire (64 tokens at top 6) and a prefill.
    check_w4a8_weighted_first(64, 512, 256, 64, 6);
    check_w4a8_weighted_first(300, 256, 128, 16, 4);
}
