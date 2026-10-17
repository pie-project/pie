#![cfg(feature = "cuda")]

//! `linear::moe::topk_softmax` a warp a row (`kernel.moe.topk_warp`) lands
//! the routes and weights the 64-thread block lands, to the bit: ties
//! included (coarse logits tie often), pick counts to 16, expert counts to
//! 512, rows off the four a block, and under a live window.

use crate::common::{Gpu, Lcg, to_bf16};
use dtype::Dtype;
use kernels_cuda::linear::moe::topk_softmax;
use kernels_cuda::tensor::Tensor;
use kernels_cuda::tuning::Tuning;

fn check(rows: usize, experts: usize, top: usize, window: Option<(u32, u32)>) {
    let mut gpu = Gpu::open();
    let mut lcg = Lcg::seeded(0x70b ^ (rows * 131 + experts * 7 + top) as u64);
    // Logits on a coarse grid, so probabilities tie.
    let logits: Vec<u16> = (0..rows * experts)
        .map(|_| to_bf16((lcg.unit() * 8.0).round() * 0.25))
        .collect();
    let logits_t = Tensor::new(gpu.up(&logits), rows as u32, experts as u32, Dtype::Bf16);
    let mut land = |warp: bool| -> (Vec<i32>, Vec<u32>) {
        let mut routes = Tensor::new(
            gpu.up(&vec![-7i32; rows * top]),
            rows as u32,
            top as u32,
            Dtype::I32,
        );
        let mut weights = Tensor::new(
            gpu.up(&vec![0x7f7f_7f7fu32; rows * top]),
            rows as u32,
            top as u32,
            Dtype::F32,
        );
        let ctx = gpu.ctx().with_tuning(Tuning {
            moe_topk_warp: warp,
            ..Tuning::DEFAULT
        });
        if let Some((live, first)) = window {
            ctx.arm_stage(gpu.up(&[live, first, 0, 0]));
        }
        topk_softmax(
            &ctx,
            logits_t,
            experts as u32,
            top as u32,
            &mut routes,
            &mut weights,
        )
        .expect("the router fires");
        ctx.disarm_stage();
        gpu.sync();
        (
            gpu.down(routes.ptr, rows * top),
            gpu.down(weights.ptr, rows * top),
        )
    };
    let block = land(false);
    let warp = land(true);
    assert_eq!(
        warp.0, block.0,
        "{rows} rows of {experts} experts, top {top}: routes"
    );
    assert_eq!(
        warp.1, block.1,
        "{rows} rows of {experts} experts, top {top}: weight bits"
    );
}

#[test]
fn the_warp_router_ranks_what_the_block_router_ranks() {
    // (rows, experts, top)
    for (rows, experts, top) in [
        (1, 8, 1),
        (5, 8, 8),
        (70, 60, 4),
        (64, 128, 8),
        (1, 256, 10),
        (70, 256, 8),
        (5, 512, 16),
        (70, 512, 10),
    ] {
        check(rows, experts, top, None);
    }
    check(64, 512, 10, Some((40, 8)));
}
