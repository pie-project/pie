#![cfg(feature = "cuda")]

use crate::common::host::{assert_near, indptr, ragged_attention};
use crate::common::{Gpu, Lcg};
use dtype::Dtype;
use kernels_cuda::attn_ragged::{self, RaggedMask};
use kernels_cuda::tensor::Tensor;

const TOLERANCE: f32 = 1.0e-2;

fn allowed(ref_start: i32, qo: usize, kv: usize) -> bool {
    let start = usize::try_from(ref_start).unwrap_or(0);
    qo < start || kv >= start
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
    let heads = [q_heads as usize, kv_heads as usize, hd as usize];
    let want = ragged_attention(
        [&q_f, &k_f, &v_f],
        &table,
        &table,
        heads,
        sm_scale,
        |g, _, r, j| {
            let r0 = table[g] as usize;
            allowed(ref_start[g], r - r0, j - r0).then_some(0.0)
        },
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
    let what = format!("head width {hd}, reference rows from {ref_start:?}");
    assert_near(&got, &want, TOLERANCE, &what);
}

const SIZES: [u32; 6] = [64, 300, 200, 50, 40, 5];
const REF_START: [i32; 6] = [20, 100, 200, -1, 0, 3];

#[test]
fn the_ragged_arm_keeps_reference_rows_to_themselves_every_case() {
    for (hd, q_heads, kv_heads, seed) in [(64, 4, 2, 0x64), (128, 4, 2, 0x128), (256, 2, 1, 0x256)]
    {
        check(hd, q_heads, kv_heads, &SIZES, &REF_START, seed);
    }
    a_short_reference_table_is_refused();
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
