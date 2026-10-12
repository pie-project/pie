#![cfg(feature = "cuda")]

use crate::common;

use std::time::Instant;

use common::{Gpu, Lcg};
use dtype::Dtype;
use kernels_cuda::attn_ragged::{self, RaggedMask};
use kernels_cuda::tensor::Tensor;

fn bench(rows_per_group: u32, groups: u32, q_heads: u32, kv_heads: u32, hd: u32, reps: u32) -> f64 {
    let rows = (rows_per_group * groups) as usize;
    let (qw, kw) = ((q_heads * hd) as usize, (kv_heads * hd) as usize);
    let table: Vec<i32> = (0..=groups).map(|g| (g * rows_per_group) as i32).collect();
    let mut lcg = Lcg::seeded(0xbe9c);
    let (q_raw, _) = lcg.row(rows * qw);
    let (kv_raw, _) = lcg.row(rows * kw);

    let mut gpu = Gpu::open();
    let q_at = gpu.up(&q_raw);
    let k_at = gpu.up(&kv_raw);
    let v_at = gpu.up(&kv_raw);
    let table_at = gpu.up(&table);
    let o_at = gpu.zeros(rows * qw * 2);
    let ctx = gpu.ctx();
    let q = Tensor::new(q_at, rows as u32, qw as u32, Dtype::Bf16);
    let k = Tensor::new(k_at, rows as u32, kw as u32, Dtype::Bf16);
    let v = Tensor::new(v_at, rows as u32, kw as u32, Dtype::Bf16);
    let table = Tensor::new(table_at, groups + 1, 1, Dtype::I32);
    let mut o = Tensor::new(o_at, rows as u32, qw as u32, Dtype::Bf16);
    let sm_scale = 1.0 / (hd as f32).sqrt();
    let fire = |o: &mut Tensor| {
        attn_ragged::ragged(
            &ctx,
            q,
            k,
            v,
            table,
            table,
            hd,
            sm_scale,
            RaggedMask::None,
            o,
        )
        .expect("the ragged arm fires")
    };
    fire(&mut o);
    gpu.sync();
    let started = Instant::now();
    for _ in 0..reps {
        fire(&mut o);
    }
    gpu.sync();
    let ms = started.elapsed().as_secs_f64() * 1e3 / f64::from(reps);
    let flops = 4.0
        * f64::from(rows_per_group)
        * f64::from(rows_per_group)
        * f64::from(groups)
        * f64::from(q_heads)
        * f64::from(hd);
    let tflops = flops / (ms * 1e-3) / 1e12;
    eprintln!(
        "ragged {groups} x {rows_per_group} rows, {q_heads}/{kv_heads} heads, head width {hd}: \
         {ms:.2} ms  ({tflops:.0} TFLOP/s)"
    );
    ms
}

#[test]
#[ignore = "a benchmark: ~1 GB of device memory and seconds of tensor-core time"]
fn the_ragged_arm_runs_at_tensor_core_speed() {
    let ms = bench(32 * 1024, 1, 24, 24, 128, 5);
    assert!(
        ms < 1000.0,
        "32k x 32k x 24 x 128 took {ms:.1} ms: not tensor-core class"
    );
    bench(32 * 1024, 1, 16, 16, 256, 3);
    bench(32 * 1024, 1, 24, 24, 64, 5);
    bench(8 * 1024, 4, 24, 8, 128, 5);
}
