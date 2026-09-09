#![cfg(feature = "cuda")]

mod common;

use common::{Gpu, Lcg};
use dtype::Dtype;
use kernels_cuda::layout;
use kernels_cuda::tensor::Tensor;

fn check(window: Option<(u32, u32, u32)>) {
    let (rows, live, base) = window.unwrap_or((8, 8, 0));
    let planes = (base + rows) as usize;
    let (vocab, hidden, layers, layer) = (40usize, 96usize, 5usize, 2u32);
    let mut lcg = Lcg::seeded(0x5e1);
    let (table_raw, _) = lcg.row(vocab * hidden);
    let (stacked_raw, _) = lcg.row(planes * layers * hidden);
    let ids: Vec<i32> = (0..planes as i32)
        .map(|r| (r * 7 + 3) % vocab as i32)
        .collect();
    let (fill, _) = lcg.row(planes * hidden);

    let mut gpu = Gpu::open();
    let table_at = gpu.up(&table_raw);
    let stacked_at = gpu.up(&stacked_raw);
    let ids_at = gpu.up(&ids);
    let ctx = gpu.ctx();
    if window.is_some() {
        let win_at = gpu.up(&[live, base, 0u32, 0u32]);
        ctx.arm_stage(win_at);
    }
    let plane = |at: u64| Tensor::new(at, rows, hidden as u32, Dtype::Bf16);
    let table = Tensor::new(table_at, vocab as u32, hidden as u32, Dtype::Bf16);
    let stacked = Tensor::new(stacked_at, rows, (layers * hidden) as u32, Dtype::Bf16);
    let ids_t = Tensor::new(ids_at, rows, 1, Dtype::I32);

    let sel_at = gpu.up(&fill);
    let (e1, es1, ys1) = (gpu.up(&fill), gpu.up(&fill), gpu.up(&fill));
    layout::select(&ctx, stacked, layer, hidden as u32, &mut plane(sel_at)).expect("select fires");
    layout::embed_scale_add(
        &ctx,
        ids_t,
        table,
        vocab as u32,
        &mut plane(e1),
        1.5,
        &mut plane(es1),
        &mut plane(sel_at),
        0.25,
        &mut plane(ys1),
    )
    .expect("the fold fires");

    let (e2, es2, y2, ys2) = (gpu.up(&fill), gpu.up(&fill), gpu.up(&fill), gpu.up(&fill));
    layout::embed_scale_add_select(
        &ctx,
        ids_t,
        table,
        vocab as u32,
        &mut plane(e2),
        1.5,
        &mut plane(es2),
        stacked,
        layer,
        hidden as u32,
        &mut plane(y2),
        0.25,
        &mut plane(ys2),
    )
    .expect("the folded launch fires");
    gpu.sync();

    for (what, a, b) in [
        ("e", e1, e2),
        ("e_scaled", es1, es2),
        ("y", sel_at, y2),
        ("y_scaled", ys1, ys2),
    ] {
        let want: Vec<u16> = gpu.down(a, planes * hidden);
        let got: Vec<u16> = gpu.down(b, planes * hidden);
        assert_eq!(
            want, got,
            "{what} differs between the pair and the fold (window {window:?})"
        );
    }
}

#[test]
fn the_select_folded_embed_lands_the_pair_every_case() {
    the_folded_launch_lands_the_pair();
    the_folded_launch_lands_the_pair_under_a_window();
}

fn the_folded_launch_lands_the_pair() {
    check(None);
}

fn the_folded_launch_lands_the_pair_under_a_window() {
    check(Some((8, 5, 2)));
}
