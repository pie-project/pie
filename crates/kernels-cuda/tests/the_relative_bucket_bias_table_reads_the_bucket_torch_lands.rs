#![cfg(feature = "cuda")]

mod common;

use common::{Gpu, to_bf16};
use dtype::Dtype;
use kernels_cuda::elemwise::relative_bucket_bias;
use kernels_cuda::tensor::Tensor;

fn bucket(d: i64, bidirectional: bool, mut num_buckets: i64, max_distance: f32) -> i64 {
    let mut out = 0;
    let n = if bidirectional {
        num_buckets /= 2;
        if d > 0 {
            out += num_buckets;
        }
        d.abs()
    } else {
        -d.min(0)
    };
    let max_exact = num_buckets / 2;
    if n < max_exact {
        return out + n;
    }
    let x = (n as f32 / max_exact as f32).ln();
    let ratio = (f64::from(max_distance) / max_exact as f64).ln() as f32;
    let large = max_exact + (x / ratio * (num_buckets - max_exact) as f32).trunc() as i64;
    out + large.min(num_buckets - 1)
}

const STEP: usize = 64;

fn embedding(num_buckets: usize, heads: usize) -> Vec<f32> {
    assert!(STEP * heads <= 256 && num_buckets <= STEP);
    (0..num_buckets * heads)
        .map(|i| (STEP * (i % heads) + i / heads) as f32)
        .collect()
}

fn check(
    heads: u32,
    max_len: u32,
    num_buckets: u32,
    max_distance: f32,
    bidirectional: bool,
    dtype: Dtype,
) {
    let span = (2 * max_len - 1) as usize;
    let emb = embedding(num_buckets as usize, heads as usize);
    let mut gpu = Gpu::open();
    let emb_at = match dtype {
        Dtype::F32 => gpu.up(&emb),
        Dtype::Bf16 => gpu.up(&emb.iter().map(|v| to_bf16(*v)).collect::<Vec<u16>>()),
        other => unreachable!("{other:?}"),
    };
    let y_at = gpu.zeros(heads as usize * span * 4);
    let ctx = gpu.ctx();
    let e = Tensor::new(emb_at, num_buckets, heads, dtype);
    let mut y = Tensor::new(y_at, heads, span as u32, Dtype::F32);
    relative_bucket_bias(
        &ctx,
        e,
        max_len,
        num_buckets,
        max_distance,
        bidirectional,
        &mut y,
    )
    .expect("the table computes");
    gpu.sync();
    let got: Vec<f32> = gpu.down(y_at, heads as usize * span);
    for h in 0..heads as usize {
        for c in 0..span {
            let d = c as i64 - (max_len as i64 - 1);
            let want =
                (STEP * h) as i64 + bucket(d, bidirectional, i64::from(num_buckets), max_distance);
            assert_eq!(
                got[h * span + c] as i64,
                want,
                "{num_buckets}/{max_distance} bidirectional={bidirectional} {dtype:?}: head {h} \
                 distance {d} read {} against bucket {}",
                got[h * span + c] as i64 - (STEP * h) as i64,
                want - (STEP * h) as i64
            );
        }
    }
}

#[test]
fn the_relative_bucket_bias_table_reads_the_bucket_torch_lands_every_case() {
    the_t5_table_reads_the_bucket_at_every_distance_from_a_bf16_embedding();
    the_t5_table_reads_the_bucket_at_every_distance_from_an_f32_embedding();
    other_bucket_counts_and_a_one_directional_table_read_their_buckets();
    a_max_distance_inside_the_exact_band_is_refused();
}

fn the_t5_table_reads_the_bucket_at_every_distance_from_a_bf16_embedding() {
    check(4, 512, 32, 128.0, true, Dtype::Bf16);
}

fn the_t5_table_reads_the_bucket_at_every_distance_from_an_f32_embedding() {
    check(4, 512, 32, 128.0, true, Dtype::F32);
}

fn other_bucket_counts_and_a_one_directional_table_read_their_buckets() {
    check(3, 300, 16, 64.0, true, Dtype::F32);
    check(3, 300, 64, 256.0, true, Dtype::F32);
    check(2, 512, 32, 1024.0, true, Dtype::F32);
    check(2, 200, 32, 128.0, false, Dtype::F32);
}

fn a_max_distance_inside_the_exact_band_is_refused() {
    let mut gpu = Gpu::open();
    let emb_at = gpu.zeros(32 * 4 * 4);
    let y_at = gpu.zeros(4 * 15 * 4);
    let ctx = gpu.ctx();
    let e = Tensor::new(emb_at, 32, 4, Dtype::F32);
    let mut y = Tensor::new(y_at, 4, 15, Dtype::F32);
    assert!(
        relative_bucket_bias(&ctx, e, 8, 32, 8.0, true, &mut y).is_err(),
        "max_distance 8 at max_exact 8 leaves the logarithm's ratio at zero"
    );
    assert!(
        relative_bucket_bias(
            &ctx,
            e,
            8,
            32,
            128.0,
            true,
            &mut Tensor::new(y_at, 4, 14, Dtype::F32)
        )
        .is_err(),
        "a table 14 wide is not 2 · 8 − 1"
    );
}
