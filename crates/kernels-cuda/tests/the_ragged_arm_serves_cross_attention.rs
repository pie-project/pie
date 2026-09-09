#![cfg(feature = "cuda")]

mod common;

use common::{Gpu, Lcg, from_bf16};
use dtype::Dtype;
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

fn check(hd: u32, q_heads: u32, kv_heads: u32, q_sizes: &[u32], kv_sizes: &[u32], seed: u64) {
    let q_table = indptr(q_sizes);
    let kv_table = indptr(kv_sizes);
    let q_rows = *q_table.last().unwrap() as usize;
    let kv_rows = *kv_table.last().unwrap() as usize;
    let (qw, kw) = ((q_heads * hd) as usize, (kv_heads * hd) as usize);
    let mut lcg = Lcg::seeded(seed);
    let (q_raw, q_f) = lcg.row(q_rows * qw);
    let (k_raw, k_f) = lcg.row(kv_rows * kw);
    let (v_raw, v_f) = lcg.row(kv_rows * kw);
    let sm_scale = 2.0 / (hd as f32).sqrt();
    let want = reference(
        &q_f,
        &k_f,
        &v_f,
        &q_table,
        &kv_table,
        q_heads as usize,
        kv_heads as usize,
        hd as usize,
        sm_scale,
    );

    let mut gpu = Gpu::open();
    let q_at = gpu.up(&q_raw);
    let k_at = gpu.up(&k_raw);
    let v_at = gpu.up(&v_raw);
    let q_table_at = gpu.up(&q_table);
    let kv_table_at = gpu.up(&kv_table);
    let sentinel = vec![0x3f80u16; q_rows * qw];
    let o_at = gpu.up(&sentinel);
    let ctx = gpu.ctx();
    let q = Tensor::new(q_at, q_rows as u32, qw as u32, Dtype::Bf16);
    let k = Tensor::new(k_at, kv_rows as u32, kw as u32, Dtype::Bf16);
    let v = Tensor::new(v_at, kv_rows as u32, kw as u32, Dtype::Bf16);
    let q_groups = Tensor::new(q_table_at, q_table.len() as u32, 1, Dtype::I32);
    let kv_groups = Tensor::new(kv_table_at, kv_table.len() as u32, 1, Dtype::I32);
    let mut o = Tensor::new(o_at, q_rows as u32, qw as u32, Dtype::Bf16);
    attn_ragged::ragged(
        &ctx,
        q,
        k,
        v,
        q_groups,
        kv_groups,
        hd,
        sm_scale,
        RaggedMask::None,
        &mut o,
    )
    .expect("the ragged arm fires across two tables");
    gpu.sync();
    let got: Vec<u16> = gpu.down(o_at, q_rows * qw);
    let mut worst = 0f32;
    for r in 0..q_rows {
        for c in 0..qw {
            let (g, w) = (from_bf16(got[r * qw + c]), want[r * qw + c]);
            let err = (g - w).abs();
            assert!(
                err <= TOLERANCE * w.abs().max(1.0),
                "head width {hd}: row {r} column {c} landed {g} against {w}"
            );
            worst = worst.max(err);
        }
    }
    eprintln!(
        "head width {hd}, {q_heads}/{kv_heads} heads, q {q_sizes:?} over kv {kv_sizes:?}: \
         worst |diff| {worst:.2e}"
    );
}

#[test]
fn the_ragged_arm_serves_cross_attention_every_case() {
    the_ragged_arm_serves_cross_attention_at_head_width_64();
    the_ragged_arm_serves_cross_attention_at_head_width_128();
    the_ragged_arm_serves_cross_attention_at_head_width_256();
    the_ragged_arm_refuses_tables_of_two_lengths();
}

fn the_ragged_arm_serves_cross_attention_at_head_width_64() {
    check(64, 4, 2, &[7, 0, 200, 300, 3], &[300, 40, 0, 500, 1], 0x64);
}

fn the_ragged_arm_serves_cross_attention_at_head_width_128() {
    check(
        128,
        4,
        2,
        &[7, 0, 200, 300, 3],
        &[300, 40, 0, 500, 1],
        0x128,
    );
}

fn the_ragged_arm_serves_cross_attention_at_head_width_256() {
    check(
        256,
        2,
        1,
        &[7, 0, 200, 300, 3],
        &[300, 40, 0, 500, 1],
        0x256,
    );
}

fn the_ragged_arm_refuses_tables_of_two_lengths() {
    let mut gpu = Gpu::open();
    let q_at = gpu.zeros(8 * 128 * 2);
    let kv_at = gpu.zeros(8 * 128 * 2);
    let o_at = gpu.zeros(8 * 128 * 2);
    let two = gpu.up(&[0i32, 8]);
    let three = gpu.up(&[0i32, 4, 8]);
    let ctx = gpu.ctx();
    let q = Tensor::new(q_at, 8, 128, Dtype::Bf16);
    let k = Tensor::new(kv_at, 8, 128, Dtype::Bf16);
    let mut o = Tensor::new(o_at, 8, 128, Dtype::Bf16);
    let q_groups = Tensor::new(two, 2, 1, Dtype::I32);
    let kv_groups = Tensor::new(three, 3, 1, Dtype::I32);
    let refused = attn_ragged::ragged(
        &ctx,
        q,
        k,
        k,
        q_groups,
        kv_groups,
        128,
        0.0,
        RaggedMask::None,
        &mut o,
    );
    assert!(
        refused.is_err(),
        "a query table and a key table of different lengths are refused"
    );
}
