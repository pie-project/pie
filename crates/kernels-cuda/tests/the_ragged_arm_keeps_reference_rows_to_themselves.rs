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

fn allowed(ref_start: i32, qo: usize, kv: usize) -> bool {
    let start = usize::try_from(ref_start).unwrap_or(0);
    qo < start || kv >= start
}

#[allow(clippy::too_many_arguments)]
fn reference(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    table: &[i32],
    ref_start: &[i32],
    q_heads: usize,
    kv_heads: usize,
    hd: usize,
    sm_scale: f32,
) -> Vec<f32> {
    let rows = q.len() / (q_heads * hd);
    let mut o = vec![0f32; rows * q_heads * hd];
    let group = q_heads / kv_heads;
    for g in 0..table.len() - 1 {
        let (r0, r1) = (table[g] as usize, table[g + 1] as usize);
        for r in r0..r1 {
            for h in 0..q_heads {
                let kh = h / group;
                let qv = &q[(r * q_heads + h) * hd..][..hd];
                let keys: Vec<usize> = (r0..r1)
                    .filter(|&j| allowed(ref_start[g], r - r0, j - r0))
                    .collect();
                let mut scores: Vec<f32> = keys
                    .iter()
                    .map(|&j| {
                        let kv = &k[(j * kv_heads + kh) * hd..][..hd];
                        qv.iter().zip(kv).map(|(a, b)| a * b).sum::<f32>() * sm_scale
                    })
                    .collect();
                let m = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0f32;
                for s in &mut scores {
                    *s = (*s - m).exp();
                    sum += *s;
                }
                let out = &mut o[(r * q_heads + h) * hd..][..hd];
                for (&j, p) in keys.iter().zip(&scores) {
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

fn check(hd: u32, q_heads: u32, kv_heads: u32, sizes: &[u32], ref_start: &[i32], seed: u64) {
    let table = indptr(sizes);
    let rows = *table.last().unwrap() as usize;
    let (qw, kw) = ((q_heads * hd) as usize, (kv_heads * hd) as usize);
    let mut lcg = Lcg::seeded(seed);
    let (q_raw, q_f) = lcg.row(rows * qw);
    let (k_raw, k_f) = lcg.row(rows * kw);
    let (v_raw, v_f) = lcg.row(rows * kw);
    let sm_scale = 2.0 / (hd as f32).sqrt();
    let want = reference(
        &q_f,
        &k_f,
        &v_f,
        &table,
        ref_start,
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
    let ref_at = gpu.up(ref_start);
    let o_at = gpu.zeros(rows * qw * 2);
    let ctx = gpu.ctx();
    let q = Tensor::new(q_at, rows as u32, qw as u32, Dtype::Bf16);
    let k = Tensor::new(k_at, rows as u32, kw as u32, Dtype::Bf16);
    let v = Tensor::new(v_at, rows as u32, kw as u32, Dtype::Bf16);
    let groups = Tensor::new(table_at, table.len() as u32, 1, Dtype::I32);
    let mask = RaggedMask::ReferenceSelfOnly {
        ref_start: Tensor::new(ref_at, ref_start.len() as u32, 1, Dtype::I32),
    };
    let mut o = Tensor::new(o_at, rows as u32, qw as u32, Dtype::Bf16);
    attn_ragged::ragged(&ctx, q, k, v, groups, groups, hd, sm_scale, mask, &mut o)
        .expect("the ragged arm fires under the reference mask");
    gpu.sync();
    let got: Vec<u16> = gpu.down(o_at, rows * qw);
    let mut worst = 0f32;
    for r in 0..rows {
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
        "head width {hd}, {q_heads}/{kv_heads} heads, groups {sizes:?} with reference rows from \
         {ref_start:?}: worst |diff| {worst:.2e}"
    );
}

const SIZES: [u32; 6] = [64, 300, 200, 50, 40, 5];
const REF_START: [i32; 6] = [20, 100, 200, -1, 0, 3];

#[test]
fn the_ragged_arm_keeps_reference_rows_to_themselves_every_case() {
    reference_rows_see_only_reference_keys_at_head_width_64();
    reference_rows_see_only_reference_keys_at_head_width_128();
    reference_rows_see_only_reference_keys_at_head_width_256();
    a_short_reference_table_is_refused();
}

fn reference_rows_see_only_reference_keys_at_head_width_64() {
    check(64, 4, 2, &SIZES, &REF_START, 0x64);
}

fn reference_rows_see_only_reference_keys_at_head_width_128() {
    check(128, 4, 2, &SIZES, &REF_START, 0x128);
}

fn reference_rows_see_only_reference_keys_at_head_width_256() {
    check(256, 2, 1, &SIZES, &REF_START, 0x256);
}

fn a_short_reference_table_is_refused() {
    let mut gpu = Gpu::open();
    let q_at = gpu.zeros(8 * 128 * 2);
    let o_at = gpu.zeros(8 * 128 * 2);
    let table = gpu.up(&[0i32, 4, 8]);
    let one = gpu.up(&[0i32]);
    let ctx = gpu.ctx();
    let q = Tensor::new(q_at, 8, 128, Dtype::Bf16);
    let mut o = Tensor::new(o_at, 8, 128, Dtype::Bf16);
    let groups = Tensor::new(table, 3, 1, Dtype::I32);
    let mask = RaggedMask::ReferenceSelfOnly {
        ref_start: Tensor::new(one, 1, 1, Dtype::I32),
    };
    let refused = attn_ragged::ragged(&ctx, q, q, q, groups, groups, 128, 0.0, mask, &mut o);
    assert!(
        refused.is_err(),
        "one reference entry for two groups is refused"
    );
}
