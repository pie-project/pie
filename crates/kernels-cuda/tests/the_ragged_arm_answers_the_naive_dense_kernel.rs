#![cfg(feature = "cuda")]

mod common;

use common::{Gpu, Lcg, from_bf16};
use dtype::Dtype;
use kernels_cuda::attn_dense;
use kernels_cuda::attn_ragged::{self, RaggedMask};
use kernels_cuda::tensor::Tensor;

const TOLERANCE: f32 = 1.0e-2;

fn indptr(sizes: &[u32]) -> Vec<i32> {
    let mut out = vec![0i32];
    for &n in sizes {
        out.push(out.last().unwrap() + n as i32);
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn reference(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    q_indptr: &[i32],
    kv_indptr: &[i32],
    q_heads: usize,
    kv_heads: usize,
    hd: usize,
    sm_scale: f32,
) -> Vec<f32> {
    let q_rows = q.len() / (q_heads * hd);
    let mut o = vec![0f32; q_rows * q_heads * hd];
    let group = q_heads / kv_heads;
    for g in 0..q_indptr.len() - 1 {
        let (q0, q1) = (q_indptr[g] as usize, q_indptr[g + 1] as usize);
        let (k0, k1) = (kv_indptr[g] as usize, kv_indptr[g + 1] as usize);
        for r in q0..q1 {
            for h in 0..q_heads {
                let kh = h / group;
                let qv = &q[(r * q_heads + h) * hd..][..hd];
                let mut scores: Vec<f32> = (k0..k1)
                    .map(|j| {
                        let kv = &k[(j * kv_heads + kh) * hd..][..hd];
                        qv.iter().zip(kv).map(|(a, b)| a * b).sum::<f32>() * sm_scale
                    })
                    .collect();
                if scores.is_empty() {
                    continue;
                }
                let m = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0f32;
                for s in &mut scores {
                    *s = (*s - m).exp();
                    sum += *s;
                }
                let out = &mut o[(r * q_heads + h) * hd..][..hd];
                for (j, p) in (k0..k1).zip(&scores) {
                    let vv = &v[(j * kv_heads + kh) * hd..][..hd];
                    for (d, x) in out.iter_mut().zip(vv) {
                        *d += p / sum * x;
                    }
                }
            }
        }
    }
    o
}

fn assert_close(got: &[u16], want: &[f32], what: &str, live_rows: usize, width: usize) {
    let mut worst = 0f32;
    for r in 0..live_rows {
        for c in 0..width {
            let (g, w) = (from_bf16(got[r * width + c]), want[r * width + c]);
            let err = (g - w).abs();
            assert!(
                err <= TOLERANCE * w.abs().max(1.0),
                "{what}: row {r} column {c} landed {g} against {w}"
            );
            worst = worst.max(err);
        }
    }
    eprintln!("{what}: worst |diff| {worst:.2e} over {live_rows} rows");
}

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
    assert_close(
        &got,
        &want,
        &format!("head width {hd}, {q_heads}/{kv_heads} heads, groups {sizes:?}"),
        rows,
        qw,
    );
}

#[test]
fn the_ragged_arm_answers_the_naive_dense_kernel_every_case() {
    the_ragged_arm_answers_the_dense_kernel_at_head_width_64();
    the_ragged_arm_answers_the_dense_kernel_at_head_width_128();
    the_ragged_arm_answers_the_dense_kernel_at_head_width_256();
    the_ragged_arm_answers_an_f32_host_reference();
    the_ragged_arm_serves_every_group_the_table_names_under_an_armed_seat();
}

fn the_ragged_arm_answers_the_dense_kernel_at_head_width_64() {
    against_dense(&[37, 512, 4096], 64, 4, 2, 0x64);
}

fn the_ragged_arm_answers_the_dense_kernel_at_head_width_128() {
    against_dense(&[37, 512, 4096], 128, 4, 2, 0x128);
}

fn the_ragged_arm_answers_the_dense_kernel_at_head_width_256() {
    against_dense(&[37, 512, 4096], 256, 2, 1, 0x256);
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
        let want = reference(
            &q_f,
            &k_f,
            &v_f,
            &table,
            &table,
            q_heads as usize,
            kv_heads as usize,
            hd as usize,
            sm_scale,
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
        assert_close(
            &got,
            &want,
            &format!("host reference at head width {hd}"),
            rows,
            qw,
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
    let want = reference(
        &q_f,
        &k_f,
        &v_f,
        &table,
        &table,
        q_heads as usize,
        kv_heads as usize,
        hd as usize,
        sm_scale,
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
