#![cfg(feature = "cuda")]

use crate::common::{Gpu, Lcg, from_bf16, to_bf16};
use dtype::Dtype;
use kernels_cuda::custom::qkv_fused_qknorm_rope_vnorm_write;
use kernels_cuda::tensor::{KvPool, Tensor};

const PAGE: usize = 16;

struct Landed {
    q: Vec<u16>,
    keys: Vec<u16>,
    values: Vec<u16>,
}

/// One fused write of `rows` rows, fired in `split` launches of
/// consecutive rows: one launch of them all takes the block-a-row kernel,
/// launches under its row floor the warp-a-head one.
fn land(head_dim: usize, q_heads: usize, kv_heads: usize, rows: usize, split: usize) -> Landed {
    let stride = (q_heads + 2 * kv_heads) * head_dim;
    let mut lcg = Lcg::seeded(0x5eed ^ head_dim as u64);
    let packed: Vec<u16> = (0..rows * stride)
        .map(|_| to_bf16(lcg.unit() * 3.0))
        .collect();
    let q_weight: Vec<u16> = (0..head_dim)
        .map(|_| to_bf16(1.0 + 0.5 * lcg.unit()))
        .collect();
    let k_weight: Vec<u16> = (0..head_dim)
        .map(|_| to_bf16(1.0 + 0.5 * lcg.unit()))
        .collect();
    let positions: Vec<i32> = (0..rows).map(|r| (r * 37 % 5000) as i32).collect();
    let pages = rows.div_ceil(PAGE);
    let write_page: Vec<u32> = (0..rows).map(|r| (pages - 1 - r / PAGE) as u32).collect();
    let write_offset: Vec<u32> = (0..rows).map(|r| (r % PAGE) as u32).collect();
    let kv_width = kv_heads * head_dim;

    let mut gpu = Gpu::open();
    let packed_at = gpu.up(&packed);
    let qw_at = gpu.up(&q_weight);
    let kw_at = gpu.up(&k_weight);
    let pos_at = gpu.up(&positions);
    let wp_at = gpu.up(&write_page);
    let wo_at = gpu.up(&write_offset);
    let q_at = gpu.zeros(rows * q_heads * head_dim * 2);
    let keys_at = gpu.zeros(pages * PAGE * kv_width * 2);
    let values_at = gpu.zeros(pages * PAGE * kv_width * 2);
    let slots = (pages * PAGE) as u32;
    let pool = KvPool {
        keys: Tensor::new(keys_at, slots, kv_width as u32, Dtype::Bf16),
        values: Tensor::new(values_at, slots, kv_width as u32, Dtype::Bf16),
        bf16_keys: Tensor::ABSENT,
        bf16_values: Tensor::ABSENT,
        key_scales: Tensor::ABSENT,
        value_scales: Tensor::ABSENT,
        page_indices: Tensor::ABSENT,
        page_indptr: Tensor::ABSENT,
        last_page_lens: Tensor::ABSENT,
        row_valid: Tensor::ABSENT,
        env_min: Tensor::ABSENT,
        env_max: Tensor::ABSENT,
        has_envelopes: false,
        page_size: PAGE as i32,
        seq_stride: kv_width as i64,
        head_stride: head_dim as i64,
        layout: 0,
        scheme_byte: 0,
        block_size: 0,
        max_pages_per_request: pages as i32,
        pages_in_batch: pages as i32,
    };
    let ctx = gpu.ctx();
    let mut first = 0;
    while first < rows {
        let n = split.min(rows - first);
        let at = |base: u64, bytes: usize| base + (first * bytes) as u64;
        let mut q = Tensor::new(
            at(q_at, q_heads * head_dim * 2),
            n as u32,
            (q_heads * head_dim) as u32,
            Dtype::Bf16,
        );
        qkv_fused_qknorm_rope_vnorm_write(
            &ctx,
            Tensor::new(
                at(packed_at, stride * 2),
                n as u32,
                stride as u32,
                Dtype::Bf16,
            ),
            Tensor::new(at(pos_at, 4), n as u32, 1, Dtype::I32),
            Tensor::new(qw_at, 1, head_dim as u32, Dtype::Bf16),
            Tensor::new(kw_at, 1, head_dim as u32, Dtype::Bf16),
            1e-6,
            &pool,
            Tensor::new(at(wp_at, 4), n as u32, 1, Dtype::U32),
            Tensor::new(at(wo_at, 4), n as u32, 1, Dtype::U32),
            kv_heads as u32,
            head_dim as u32,
            10_000.0,
            (head_dim / 2) as u32,
            &mut q,
        )
        .expect("the fused write fires");
        first += n;
    }
    gpu.sync();
    Landed {
        q: gpu.down(q_at, rows * q_heads * head_dim),
        keys: gpu.down(keys_at, pages * PAGE * kv_width),
        values: gpu.down(values_at, pages * PAGE * kv_width),
    }
}

#[test]
fn the_row_fused_qkv_write_lands_what_the_warp_one_does() {
    for (head_dim, q_heads, kv_heads) in [(64, 8, 2), (128, 16, 4), (256, 32, 16)] {
        let by_row = land(head_dim, q_heads, kv_heads, 160, 160);
        let by_warp = land(head_dim, q_heads, kv_heads, 160, 40);
        assert!(by_row.q == by_warp.q, "head {head_dim}: q differs");
        assert!(by_row.keys == by_warp.keys, "head {head_dim}: keys differ");
        assert!(
            by_row.values == by_warp.values,
            "head {head_dim}: values differ"
        );
    }
    // Head 512 had only the block-a-head kernel, which sums in another
    // order: the two agree to a bf16 rounding.
    let by_row = land(512, 8, 2, 128, 128);
    let by_block = land(512, 8, 2, 128, 32);
    for (what, a, b) in [
        ("q", &by_row.q, &by_block.q),
        ("keys", &by_row.keys, &by_block.keys),
        ("values", &by_row.values, &by_block.values),
    ] {
        for (at, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
            let (x, y) = (from_bf16(x), from_bf16(y));
            assert!(
                (x - y).abs() <= 2e-2 * x.abs().max(y.abs()) + 1e-3,
                "head 512 {what}[{at}]: {x} against {y}"
            );
        }
    }
}
