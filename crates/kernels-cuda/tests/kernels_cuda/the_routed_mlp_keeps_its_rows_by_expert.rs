#![cfg(feature = "cuda")]

//! `linear::moe::experts`: the routed MLP with its rows kept by expert from
//! the gate-up select to the weighted sum answers what the gate-up select
//! with its GLU, the down select and the weighted sum answer apart.

use crate::common::{Gpu, Lcg, close, from_bf16, to_bf16};
use dtype::Dtype;
use kernels_cuda::linear::moe::{
    ExpertTable, GluBank, GroupSeat, experts, matmul_select, matmul_select_quant, select_glu,
    sigmoid_gate_add, weighted_sum,
};
use kernels_cuda::linear::tc5::{Glu, supported};
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
}
