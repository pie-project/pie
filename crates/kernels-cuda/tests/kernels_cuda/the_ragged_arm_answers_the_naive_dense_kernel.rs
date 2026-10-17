#![cfg(feature = "cuda")]

use crate::common::host_b::{assert_near, indptr, ragged_attention};
use crate::common::{Gpu, Lcg, from_bf16};
use dtype::Dtype;
use kernels_cuda::attn_dense;
use kernels_cuda::attn_ragged::{self, RaggedMask};
use kernels_cuda::tensor::Tensor;

const TOLERANCE: f32 = 1.0e-2;

fn against_dense(sizes: &[u32], hd: u32, q_heads: u32, kv_heads: u32, seed: u64) {
    let table = indptr(sizes);
    let rows = *table.last().unwrap() as usize;
    let (qw, kw) = ((q_heads * hd) as usize, (kv_heads * hd) as usize);
    let mut lcg = Lcg::seeded(seed);
    let (q_raw, _) = lcg.row(rows * qw);
    let (k_raw, _) = lcg.row(rows * kw);
    let (v_raw, _) = lcg.row(rows * kw);
    let sm_scale = 1.0 / (hd as f32).sqrt();

    let mut gpu = Gpu::open();
    let q_at = gpu.up(&q_raw);
    let k_at = gpu.up(&k_raw);
    let v_at = gpu.up(&v_raw);
    let table_at = gpu.up(&table);
    let o_dense = gpu.zeros(rows * qw * 2);
    let o_ragged = gpu.zeros(rows * qw * 2);
    let ctx = gpu.ctx();
    let q = Tensor::new(q_at, rows as u32, qw as u32, Dtype::Bf16);
    let k = Tensor::new(k_at, rows as u32, kw as u32, Dtype::Bf16);
    let v = Tensor::new(v_at, rows as u32, kw as u32, Dtype::Bf16);
    let segments = Tensor::new(table_at, table.len() as u32, 1, Dtype::I32);

    let mut dense = Tensor::new(o_dense, rows as u32, qw as u32, Dtype::Bf16);
    attn_dense::bidirectional(&ctx, q, k, v, segments, hd, sm_scale, &mut dense)
        .expect("the dense kernel fires");
    let mut ragged = Tensor::new(o_ragged, rows as u32, qw as u32, Dtype::Bf16);
    attn_ragged::ragged(
        &ctx,
        q,
        k,
        v,
        segments,
        segments,
        hd,
        sm_scale,
        RaggedMask::None,
        &mut ragged,
    )
    .expect("the ragged arm fires");
    gpu.sync();

    let want: Vec<u16> = gpu.down(o_dense, rows * qw);
    let want: Vec<f32> = want.into_iter().map(from_bf16).collect();
    let got: Vec<u16> = gpu.down(o_ragged, rows * qw);
    let what = format!("head width {hd}, {q_heads}/{kv_heads} heads, groups {sizes:?}");
    assert_near(&got, &want, TOLERANCE, &what);
}

#[test]
fn the_ragged_arm_answers_the_naive_dense_kernel_every_case() {
    for (hd, q_heads, kv_heads, seed) in [(64, 4, 2, 0x64), (128, 4, 2, 0x128), (256, 2, 1, 0x256)]
    {
        against_dense(&[37, 512, 4096], hd, q_heads, kv_heads, seed);
    }
    the_ragged_arm_answers_an_f32_host_reference();
    the_ragged_arm_serves_every_group_the_table_names_under_an_armed_seat();
}

fn the_ragged_arm_answers_an_f32_host_reference() {
    for hd in [64u32, 128, 256] {
        let sizes = [5u32, 33, 70, 1];
        let (q_heads, kv_heads) = (4u32, 2u32);
        let table = indptr(&sizes);
        let rows = *table.last().unwrap() as usize;
        let (qw, kw) = ((q_heads * hd) as usize, (kv_heads * hd) as usize);
        let mut lcg = Lcg::seeded(0xf32 + u64::from(hd));
        let (q_raw, q_f) = lcg.row(rows * qw);
        let (k_raw, k_f) = lcg.row(rows * kw);
        let (v_raw, v_f) = lcg.row(rows * kw);
        let sm_scale = 2.5 / (hd as f32).sqrt();
        let heads = [q_heads as usize, kv_heads as usize, hd as usize];
        let want = ragged_attention(
            [&q_f, &k_f, &v_f],
            &table,
            &table,
            heads,
            sm_scale,
            |_, _, _, _| Some(0.0),
        );

        let mut gpu = Gpu::open();
        let q_at = gpu.up(&q_raw);
        let k_at = gpu.up(&k_raw);
        let v_at = gpu.up(&v_raw);
        let table_at = gpu.up(&table);
        let o_at = gpu.zeros(rows * qw * 2);
        let ctx = gpu.ctx();
        let q = Tensor::new(q_at, rows as u32, qw as u32, Dtype::Bf16);
        let k = Tensor::new(k_at, rows as u32, kw as u32, Dtype::Bf16);
        let v = Tensor::new(v_at, rows as u32, kw as u32, Dtype::Bf16);
        let groups = Tensor::new(table_at, table.len() as u32, 1, Dtype::I32);
        let mut o = Tensor::new(o_at, rows as u32, qw as u32, Dtype::Bf16);
        attn_ragged::ragged(
            &ctx,
            q,
            k,
            v,
            groups,
            groups,
            hd,
            sm_scale,
            RaggedMask::None,
            &mut o,
        )
        .expect("the ragged arm fires");
        gpu.sync();
        let got: Vec<u16> = gpu.down(o_at, rows * qw);
        assert_near(
            &got,
            &want,
            TOLERANCE,
            &format!("host reference at head width {hd}"),
        );
    }
}

fn the_ragged_arm_serves_every_group_the_table_names_under_an_armed_seat() {
    let hd = 64u32;
    let (q_heads, kv_heads) = (2u32, 2u32);
    let sizes = [40u32, 100, 300, 64];
    let table = indptr(&sizes);
    let rows = *table.last().unwrap() as usize;
    let plane_rows = rows + 96;
    let (qw, kw) = ((q_heads * hd) as usize, (kv_heads * hd) as usize);
    let mut lcg = Lcg::seeded(0x5ea7);
    let (q_raw, q_f) = lcg.row(plane_rows * qw);
    let (k_raw, k_f) = lcg.row(plane_rows * kw);
    let (v_raw, v_f) = lcg.row(plane_rows * kw);
    let (fill_raw, _) = lcg.row(plane_rows * qw);
    let sm_scale = 1.0 / (hd as f32).sqrt();
    let (first, live) = (1usize, 2usize);
    let heads = [q_heads as usize, kv_heads as usize, hd as usize];
    let want = ragged_attention(
        [&q_f, &k_f, &v_f],
        &table,
        &table,
        heads,
        sm_scale,
        |_, _, _, _| Some(0.0),
    );

    let mut gpu = Gpu::open();
    let q_at = gpu.up(&q_raw);
    let k_at = gpu.up(&k_raw);
    let v_at = gpu.up(&v_raw);
    let table_at = gpu.up(&table);
    let o_at = gpu.up(&fill_raw);
    let win_at = gpu.up(&[plane_rows as u32, 0u32, live as u32, first as u32]);
    let ctx = gpu.ctx();
    ctx.arm_stage(win_at);
    let q = Tensor::new(q_at, plane_rows as u32, qw as u32, Dtype::Bf16);
    let k = Tensor::new(k_at, plane_rows as u32, kw as u32, Dtype::Bf16);
    let v = Tensor::new(v_at, plane_rows as u32, kw as u32, Dtype::Bf16);
    let groups = Tensor::new(table_at, table.len() as u32, 1, Dtype::I32);
    let mut o = Tensor::new(o_at, plane_rows as u32, qw as u32, Dtype::Bf16);
    attn_ragged::ragged(
        &ctx,
        q,
        k,
        v,
        groups,
        groups,
        hd,
        sm_scale,
        RaggedMask::None,
        &mut o,
    )
    .expect("the ragged arm fires under a staged seat");
    gpu.sync();
    let got: Vec<u16> = gpu.down(o_at, plane_rows * qw);
    let (r0, r1) = (table[0] as usize, *table.last().unwrap() as usize);
    for r in 0..plane_rows {
        let span = r * qw..(r + 1) * qw;
        if r >= r0 && r < r1 {
            for c in 0..qw {
                let (g, w) = (from_bf16(got[r * qw + c]), want[r * qw + c]);
                assert!(
                    (g - w).abs() <= TOLERANCE * w.abs().max(1.0),
                    "row {r} column {c}: a grouped row landed {g} against {w}"
                );
            }
        } else {
            assert_eq!(
                got[span.clone()],
                fill_raw[span],
                "row {r}: a row past every group moved"
            );
        }
    }
}
