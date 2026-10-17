#![cfg(feature = "cuda")]

//! `linear::moe::experts`: the routed MLP with its rows kept by expert from
//! the gate-up select to the weighted sum answers what the gate-up select
//! with its GLU, the down select and the weighted sum answer apart.

use crate::common::host_a::{bf16_zeros, routes_of};
use crate::common::{Gpu, Lcg, close, from_bf16, to_bf16};
use dtype::Dtype;
use kernels_cuda::Ctx;
use kernels_cuda::linear::act_quant::{self, AlphaFrom, Format, scales_to_atoms};
use kernels_cuda::linear::mlp::{situ, swiglu, swiglu_clamp, swiglu_clamp_routed};
use kernels_cuda::linear::moe::{
    ExpertTable, GluBank, GroupSeat, experts, experts_weighted_first, matmul_select,
    matmul_select_quant, select_glu, sigmoid_gate_add, sum_add, weighted_sum,
};
use kernels_cuda::linear::tc5::{Elem, Glu, atom_bytes, supported};
use kernels_cuda::tensor::Tensor;

#[derive(Clone, Copy)]
enum Bank {
    Bf16,
    Mxfp4,
}

/// A fire's activations, routes and route weights, and its route count.
struct Fire {
    x: Tensor,
    routes: Tensor,
    weights: Tensor,
    count: usize,
}

fn fire(
    gpu: &mut Gpu,
    seed: u64,
    tokens: usize,
    hidden: usize,
    n_experts: usize,
    top: usize,
) -> Fire {
    let mut lcg = Lcg::seeded(seed ^ tokens as u64);
    let routes = routes_of(tokens, n_experts, top);
    let count = routes.len();
    let (x_raw, _) = lcg.row(tokens * hidden);
    let weights: Vec<f32> = (0..count).map(|i| 0.1 + (i % 7) as f32 * 0.05).collect();
    let (t, top) = (tokens as u32, top as u32);
    Fire {
        x: Tensor::new(gpu.up(&x_raw), t, hidden as u32, Dtype::Bf16),
        routes: Tensor::new(gpu.up(&routes), t, top, Dtype::I32),
        weights: Tensor::new(gpu.up(&weights), t, top, Dtype::F32),
        count,
    }
}

/// Qwen's shared expert row and its gate, added at the gate's sigmoid.
fn gate_of(gpu: &mut Gpu, seed: u64, tokens: usize, hidden: usize) -> (Tensor, Tensor) {
    let mut l = Lcg::seeded(seed ^ tokens as u64);
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
}

/// `y` with the gated shared row added, where there is one.
fn gated(ctx: &Ctx, gpu: &mut Gpu, y: Tensor, gate: Option<(Tensor, Tensor)>) -> Tensor {
    let Some((shared, g)) = gate else {
        return y;
    };
    let mut out = bf16_zeros(gpu, y.rows as usize, y.width as usize);
    sigmoid_gate_add(ctx, y, shared, g, &mut out).expect("the gate-add fires");
    out
}

/// `got` against `want`, token by token in `rows`.
fn assert_close(gpu: &Gpu, got: Tensor, want: Tensor, rows: std::ops::Range<usize>) {
    let hidden = got.width as usize;
    let n = got.rows as usize * hidden;
    let want: Vec<u16> = gpu.down(want.ptr, n);
    let got: Vec<u16> = gpu.down(got.ptr, n);
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        if !rows.contains(&(i / hidden)) {
            continue;
        }
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

/// The W4A8 MLP kept by expert sits no farther from the product of
/// unquantized activations (`exact`) than its four apart do: the GLU
/// quantizes from f32 in the epilogue, so an e4m3 code a rounding apart
/// flips now and then.
fn assert_no_farther(gpu: &Gpu, tokens: usize, exact: Tensor, apart: Tensor, kept: Tensor) {
    let n = tokens * exact.width as usize;
    let exact: Vec<u16> = gpu.down(exact.ptr, n);
    let scale = exact
        .iter()
        .map(|w| f64::from(from_bf16(*w)).abs())
        .sum::<f64>()
        / n as f64;
    let distance = |y: Tensor| -> f64 {
        gpu.down::<u16>(y.ptr, n)
            .iter()
            .zip(&exact)
            .map(|(g, w)| f64::from(from_bf16(*g) - from_bf16(*w)).abs())
            .sum::<f64>()
            / n as f64
            / scale
    };
    let (apart_d, kept_d) = (distance(apart), distance(kept));
    assert!(
        apart_d > 0.0 && apart_d < 0.1,
        "{tokens} tokens: the four apart sit {apart_d:.4} from the unquantized product"
    );
    assert!(
        kept_d <= 1.1 * apart_d,
        "{tokens} tokens: kept by expert {kept_d:.4} from the unquantized product, apart {apart_d:.4}"
    );
}

/// A bf16 or mxfp4 MLP; `window`: (live tokens, first live token), armed
/// as the context's stage.
#[allow(clippy::too_many_arguments)]
fn check(
    tokens: usize,
    hidden: usize,
    inter: usize,
    bank: Bank,
    n_experts: usize,
    top: usize,
    window: Option<(u32, u32)>,
    gate: bool,
) {
    let mut gpu = Gpu::open();
    let f = fire(&mut gpu, 0x3e, tokens, hidden, n_experts, top);
    let mut bank_of = |rows: usize, k: usize, seed: u64| -> GluBank {
        let mut l = Lcg::seeded(seed);
        let e = n_experts as u32;
        match bank {
            Bank::Bf16 => {
                let (raw, _) = l.row(n_experts * rows * k);
                let raw: Vec<u16> = raw.iter().map(|&b| to_bf16(from_bf16(b) * 0.25)).collect();
                GluBank::Bf16(Tensor::new(gpu.up(&raw), e, (rows * k) as u32, Dtype::Bf16))
            }
            Bank::Mxfp4 => {
                let codes: Vec<u8> = (0..n_experts * rows * k / 2)
                    .map(|_| ((l.unit() + 1.0) * 127.0) as u8)
                    .collect();
                let scales: Vec<u8> = (0..n_experts * rows * k / 32)
                    .map(|_| 120 + (((l.unit() + 1.0) * 3.0) as u8).min(5))
                    .collect();
                GluBank::Mxfp4 {
                    codes: Tensor::new(gpu.up(&codes), e, (rows * k / 2) as u32, Dtype::U8),
                    scales: Tensor::new(gpu.up(&scales), e, (rows * k / 32) as u32, Dtype::U8),
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
    let gate = gate.then(|| gate_of(&mut gpu, 0x77, tokens, hidden));

    // Apart: the gate-up select with its GLU, the down select, the sum.
    let mut h = bf16_zeros(&mut gpu, f.count, inter);
    assert!(
        select_glu(&ctx, f.x, up, f.routes, &mut h, glu, ExpertTable::RESIDENT).expect("fires"),
        "the gate-up select lands its GLU"
    );
    let mut r = bf16_zeros(&mut gpu, f.count, hidden);
    match down {
        GluBank::Bf16(b) => {
            matmul_select(&ctx, h, b, f.routes, &mut r, ExpertTable::RESIDENT).expect("fires")
        }
        GluBank::Mxfp4 { codes, scales, .. } => matmul_select_quant(
            &ctx,
            h,
            codes,
            scales,
            None,
            f.routes,
            &mut r,
            GroupSeat::RESIDENT,
        )
        .expect("fires"),
        _ => unreachable!(),
    }
    let mut apart = bf16_zeros(&mut gpu, tokens, hidden);
    weighted_sum(&ctx, r, f.weights, &mut apart).expect("the sum fires");
    let apart = gated(&ctx, &mut gpu, apart, gate);

    let mut kept = bf16_zeros(&mut gpu, tokens, hidden);
    let took = experts(
        &ctx,
        f.x,
        up,
        glu,
        down,
        f.routes,
        f.weights,
        inter as u32,
        &mut kept,
        ExpertTable::RESIDENT,
        ExpertTable::RESIDENT,
        gate,
    )
    .expect("the routed MLP fires");
    assert!(took, "the grouped GEMM takes a {tokens}-token fire");
    gpu.sync();
    let (lo, hi) = window.map_or((0, tokens), |(live, first)| {
        (first as usize, (first + live) as usize)
    });
    assert_close(&gpu, kept, apart, lo..hi);
}

/// An NVFP4 bank of `rows` rows of `k` an expert, against NVFP4
/// activations: its atoms, a tensor scale a row, an input scale an expert.
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

/// The W4A4 MLP kept by expert, against its four apart: each select
/// quantizing under its experts' globals, the SwiGLU's bf16 rows between.
fn check_w4a4(
    tokens: usize,
    hidden: usize,
    inter: usize,
    n_experts: usize,
    top: usize,
    gate: bool,
) {
    let mut gpu = Gpu::open();
    let f = fire(&mut gpu, 0x4a4, tokens, hidden, n_experts, top);
    let up = w4a4_bank(&mut gpu, n_experts, 2 * inter, hidden, 0x61);
    let down = w4a4_bank(&mut gpu, n_experts, hidden, inter, 0x62);
    let gate = gate.then(|| gate_of(&mut gpu, 0x78, tokens, hidden));
    let ctx = gpu.ctx().with_tuning(crate::common::SM100);
    let select_apart = |ctx: &Ctx, gpu: &mut Gpu, x: Tensor, bank: GluBank, n: usize| {
        let GluBank::Nvfp4A4 {
            codes,
            atoms,
            tensor_scales,
            input_scales,
        } = bank
        else {
            unreachable!()
        };
        let mut y = bf16_zeros(gpu, f.count, n);
        let alpha = gpu.zeros(n_experts * 4);
        act_quant::select(
            ctx,
            "test.w4a4_apart",
            Format::Nvfp4,
            x,
            f.routes,
            Some(input_scales),
            act_quant::Bank {
                codes: codes.ptr,
                atoms: atoms.ptr,
                experts: n_experts as u32,
                n: n as u32,
                k: x.width,
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
    let packed = select_apart(&ctx, &mut gpu, f.x, up, 2 * inter);
    let mut h = bf16_zeros(&mut gpu, f.count, inter);
    swiglu(&ctx, packed, inter as u32, 1, &mut h).expect("the GLU fires");
    let r = select_apart(&ctx, &mut gpu, h, down, hidden);
    let mut apart = bf16_zeros(&mut gpu, tokens, hidden);
    weighted_sum(&ctx, r, f.weights, &mut apart).expect("the sum fires");
    let apart = gated(&ctx, &mut gpu, apart, gate);
    let mut kept = bf16_zeros(&mut gpu, tokens, hidden);
    let took = experts(
        &ctx,
        f.x,
        up,
        Glu::Swiglu,
        down,
        f.routes,
        f.weights,
        inter as u32,
        &mut kept,
        ExpertTable::RESIDENT,
        ExpertTable::RESIDENT,
        gate,
    )
    .expect("the routed MLP fires");
    assert!(took, "the grouped GEMM takes a {tokens}-token W4A4 fire");
    gpu.sync();
    assert_close(&gpu, kept, apart, 0..tokens);
}

/// An MXFP4 bank against e4m3 activations in 1x128 blocks (its scales laid
/// in atoms), and the same bank against bf16 activations.
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

/// A select of `bank` at W4A8 (e4m3 activations) or W4A16 (bf16).
fn select_w4a8(ctx: &Ctx, gpu: &mut Gpu, f: &Fire, x: Tensor, bank: GluBank, n: usize) -> Tensor {
    let mut y = bf16_zeros(gpu, f.count, n);
    match bank {
        GluBank::Mxfp4A8 { codes, atoms } => act_quant::select(
            ctx,
            "test.w4a8_apart",
            Format::E4m3 { gran: 128 },
            x,
            f.routes,
            None,
            act_quant::Bank {
                codes: codes.ptr,
                atoms: atoms.ptr,
                experts: codes.rows,
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
            f.routes,
            &mut y,
            GroupSeat::RESIDENT,
        )
        .expect("the select fires"),
        _ => unreachable!(),
    }
    y
}

/// The W4A8 MLP kept by expert, against its four apart (under the
/// registry's defaults where `pinned`, else B200's), both held to the
/// product of unquantized activations.
#[allow(clippy::too_many_arguments)]
fn check_w4a8(
    tokens: usize,
    hidden: usize,
    inter: usize,
    n_experts: usize,
    top: usize,
    gate: bool,
    glu: Glu,
    pinned: bool,
) {
    let mut gpu = Gpu::open();
    let f = fire(&mut gpu, 0x4a8, tokens, hidden, n_experts, top);
    let (up, up16) = w4a8_bank(&mut gpu, n_experts, 2 * inter, hidden, 0x71);
    let (down, down16) = w4a8_bank(&mut gpu, n_experts, hidden, inter, 0x72);
    let gate = gate.then(|| gate_of(&mut gpu, 0x79, tokens, hidden));
    // B200's arm, fused from the first token.
    let ctx = gpu.ctx().with_tuning(kernels_cuda::tuning::Tuning {
        moe_w4a8_fused_from: 1,
        ..crate::common::SM100
    });
    let actx = if pinned {
        gpu.ctx()
    } else {
        gpu.ctx().with_tuning(crate::common::SM100)
    };
    let packed = select_w4a8(&actx, &mut gpu, &f, f.x, up, 2 * inter);
    let mut h = bf16_zeros(&mut gpu, f.count, inter);
    match glu {
        Glu::SwigluClamp { limit } => swiglu_clamp(&actx, packed, inter as u32, 1, limit, &mut h),
        Glu::Situ { beta, cap } => situ(
            &actx,
            packed,
            inter as u32,
            1,
            beta,
            (cap > 0.0).then_some(cap),
            &mut h,
        ),
        _ => swiglu(&actx, packed, inter as u32, 1, &mut h),
    }
    .expect("the GLU fires");
    let r = select_w4a8(&actx, &mut gpu, &f, h, down, hidden);
    let mut apart = bf16_zeros(&mut gpu, tokens, hidden);
    weighted_sum(&ctx, r, f.weights, &mut apart).expect("the sum fires");
    // Unquantized activations: W4A16's select with its GLU, the down
    // select, the sum.
    let mut h16 = bf16_zeros(&mut gpu, f.count, inter);
    assert!(
        select_glu(
            &ctx,
            f.x,
            up16,
            f.routes,
            &mut h16,
            glu,
            ExpertTable::RESIDENT
        )
        .expect("fires"),
        "the W4A16 gate-up select lands its GLU"
    );
    let r16 = select_w4a8(&ctx, &mut gpu, &f, h16, down16, hidden);
    let mut exact = bf16_zeros(&mut gpu, tokens, hidden);
    weighted_sum(&ctx, r16, f.weights, &mut exact).expect("the sum fires");
    let apart = gated(&ctx, &mut gpu, apart, gate);
    let exact = gated(&ctx, &mut gpu, exact, gate);
    let mut kept = bf16_zeros(&mut gpu, tokens, hidden);
    let took = experts(
        &ctx,
        f.x,
        up,
        glu,
        down,
        f.routes,
        f.weights,
        inter as u32,
        &mut kept,
        ExpertTable::RESIDENT,
        ExpertTable::RESIDENT,
        gate,
    )
    .expect("the routed MLP fires");
    assert!(took, "the grouped GEMM takes a {tokens}-token W4A8 fire");
    gpu.sync();
    assert_no_farther(&gpu, tokens, exact, apart, kept);
}

/// DeepSeek's W4A8 MLP: the route weights scale the clamped GLU's rows
/// before the down leg quantizes them, and the routes sum plainly with the
/// shared row added.
fn check_w4a8_weighted_first(
    tokens: usize,
    hidden: usize,
    inter: usize,
    n_experts: usize,
    top: usize,
) {
    let limit = 7.0f32;
    let mut gpu = Gpu::open();
    let f = fire(&mut gpu, 0xd54, tokens, hidden, n_experts, top);
    let (shared, _) = crate::common::host_a::bf16_rows(
        &mut gpu,
        &mut Lcg::seeded(0xd55 ^ tokens as u64),
        tokens,
        hidden,
    );
    let (up, up16) = w4a8_bank(&mut gpu, n_experts, 2 * inter, hidden, 0x81);
    let (down, down16) = w4a8_bank(&mut gpu, n_experts, hidden, inter, 0x82);
    let ctx = gpu.ctx().with_tuning(kernels_cuda::tuning::Tuning {
        moe_w4a8_fused_from: 1,
        ..crate::common::SM100
    });
    // The four apart, at W4A8 (`a8`) or with unquantized activations.
    let apart = |gpu: &mut Gpu, a8: bool| {
        let (u, d) = if a8 { (up, down) } else { (up16, down16) };
        let packed = select_w4a8(&ctx, gpu, &f, f.x, u, 2 * inter);
        let mut h = bf16_zeros(gpu, f.count, inter);
        swiglu_clamp_routed(&ctx, packed, inter as u32, f.weights, 1, limit, &mut h)
            .expect("the GLU fires");
        let r = select_w4a8(&ctx, gpu, &f, h, d, hidden);
        let mut y = bf16_zeros(gpu, tokens, hidden);
        sum_add(&ctx, r, shared, &mut y).expect("the sum fires");
        y
    };
    let apart8 = apart(&mut gpu, true);
    let exact = apart(&mut gpu, false);
    let mut kept = bf16_zeros(&mut gpu, tokens, hidden);
    let took = experts_weighted_first(
        &ctx,
        f.x,
        up,
        Glu::SwigluClamp { limit },
        down,
        f.routes,
        f.weights,
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
    assert_no_farther(&gpu, tokens, exact, apart8, kept);
}

#[test]
fn the_routed_mlp_keeps_its_rows_by_expert_every_case() {
    if !supported() {
        eprintln!("skipped: no tcgen05 on this device");
        return;
    }
    // Masked (64 tokens or fewer) and contiguous layouts, bf16 and mxfp4.
    for bank in [Bank::Bf16, Bank::Mxfp4] {
        check(16, 256, 128, bank, 16, 4, None, false);
        check(300, 512, 128, bank, 16, 4, None, true);
    }
    // Qwen3.6's routing (256 experts, top 8): a captured decode's live
    // window, and a prefill.
    check(64, 2048, 512, Bank::Bf16, 256, 8, Some((40, 8)), true);
    check(64, 256, 128, Bank::Mxfp4, 16, 4, Some((17, 3)), false);
    check(512, 2048, 512, Bank::Bf16, 256, 8, None, false);
    // W4A4 (NVFP4 against NVFP4): the gate-up swapped, then not.
    check_w4a4(64, 512, 256, 64, 10, false);
    check_w4a4(300, 256, 128, 16, 4, true);
    // W4A8 (MXFP4 against e4m3): swapped and not, then GLM's clamped
    // SwiGLU and Kimi-K3's SiTU (capped and not) as the epilogue lands them.
    check_w4a8(64, 512, 256, 64, 10, false, Glu::Swiglu, false);
    check_w4a8(300, 256, 128, 16, 4, true, Glu::Swiglu, false);
    let clamp = Glu::SwigluClamp { limit: 7.0 };
    check_w4a8(300, 256, 128, 16, 4, true, clamp, false);
    let situ_capped = Glu::Situ {
        beta: 4.0,
        cap: 25.0,
    };
    check_w4a8(300, 256, 128, 16, 4, false, situ_capped, false);
    let situ_plain = Glu::Situ {
        beta: 1.0,
        cap: 0.0,
    };
    check_w4a8(64, 512, 256, 64, 10, false, situ_plain, false);
    // GLM-5.3-Flash mini's 17-token prompt fire, the four apart under
    // B200's decisions and the registry's.
    for pinned in [false, true] {
        let clamp = Glu::SwigluClamp { limit: 10.0 };
        check_w4a8(17, 4096, 2048, 32, 8, false, clamp, pinned);
    }
    // DeepSeek's W4A8: its decode64 fire (top 6) and a prefill.
    check_w4a8_weighted_first(64, 512, 256, 64, 6);
    check_w4a8_weighted_first(300, 256, 128, 16, 4);
}
