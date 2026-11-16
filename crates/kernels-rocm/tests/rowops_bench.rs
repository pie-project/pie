//! The per-row ops a prefill fire runs once or twice a layer, timed at a
//! served model's shapes (Qwen3-0.6B: hidden 1024, 16 / 8 heads of 128) and
//! set against the bytes they move: a probe run by hand.

#![cfg(feature = "rocm")]

mod common;

use common::{Gpu, KV, values};
use dtype::Dtype;
use kernels_rocm::attn;
use kernels_rocm::elemwise::{norm, norm_residual, rope};
use kernels_rocm::gather as rows;
use kernels_rocm::{KvPool, RaggedTensor};

const HIDDEN: u32 = 1024;
const Q: u32 = 16 * 128;
const KVW: u32 = 8 * 128;
const PAGE: u32 = 16;
const REPS: u32 = 200;

fn report(name: &str, rows: u32, us: f64, bytes: u64) {
    let gbs = bytes as f64 / (us * 1e3);
    eprintln!("[rowops] rows {rows:>5} {name:<22} {us:>8.2} us  {:>7.1} MB  {gbs:>7.0} GB/s", bytes as f64 / 1e6);
}

#[test]
#[ignore = "a probe run by hand"]
fn rowops_bench() {
    let Some(gpu) = Gpu::open() else { return };
    let ctx = gpu.ctx();
    for rows_n in [2897u32, 5793, 10240] {
        let r = u64::from(rows_n);
        let bf = |w: u32, seed: u64| gpu.tensor(&values((rows_n * w) as usize, seed, 1.0), rows_n, w, Dtype::Bf16);

        let x = bf(HIDDEN, 1);
        let mut stream = bf(HIDDEN, 2);
        let w = gpu.tensor(&values(HIDDEN as usize, 3, 0.5), 1, HIDDEN, Dtype::Bf16);
        let mut out = gpu.zeros(rows_n, HIDDEN, Dtype::Bf16);
        let us = gpu.time(REPS, || {
            norm_residual::residual_add_rmsnorm(&ctx, x, &mut stream, w, false, 1e-6, &mut out).unwrap();
        });
        report("residual_add_rmsnorm", rows_n, us, r * u64::from(HIDDEN) * 2 * 4);

        let wh = gpu.tensor(&values(128, 4, 0.5), 1, 128, Dtype::Bf16);
        let q = bf(Q, 5);
        let mut qn = gpu.zeros(rows_n, Q, Dtype::Bf16);
        let us = gpu.time(REPS, || norm::rmsnorm_per_head(&ctx, q, wh, 128, 1e-6, &mut qn).unwrap());
        report("rmsnorm q (per head)", rows_n, us, r * u64::from(Q) * 2 * 2);
        let k = bf(KVW, 6);
        let mut kn = gpu.zeros(rows_n, KVW, Dtype::Bf16);
        let us = gpu.time(REPS, || norm::rmsnorm_per_head(&ctx, k, wh, 128, 1e-6, &mut kn).unwrap());
        report("rmsnorm k (per head)", rows_n, us, r * u64::from(KVW) * 2 * 2);

        let positions = gpu.ints(&(0..rows_n as i32).collect::<Vec<_>>());
        let mut qr = bf(Q, 7);
        let mut kr = bf(KVW, 8);
        let us = gpu.time(REPS, || rope::full(&ctx, &mut qr, &mut kr, positions, 128, 1e6, false).unwrap());
        report("rope q+k", rows_n, us, r * u64::from(Q + KVW) * 2 * 2);

        let packed = bf(Q + 2 * KVW, 9);
        let mut sq = gpu.zeros(rows_n, Q, Dtype::Bf16);
        let mut sk = gpu.zeros(rows_n, KVW, Dtype::Bf16);
        let mut sv = gpu.zeros(rows_n, KVW, Dtype::Bf16);
        let us = gpu.time(REPS, || rows::split_qkv(&ctx, packed, Q, KVW, &mut sq, &mut sk, &mut sv).unwrap());
        report("split_qkv", rows_n, us, r * u64::from(Q + 2 * KVW) * 2 * 2);

        let pages = rows_n.div_ceil(PAGE);
        let cells = pages * PAGE;
        let pool = KvPool {
            keys: gpu.zeros(cells, KVW, Dtype::Bf16),
            values: gpu.zeros(cells, KVW, Dtype::Bf16),
            page_size: PAGE as i32,
            seq_stride: i64::from(KVW),
            head_stride: 128,
            live: [0, 0],
            ..KV
        }
        .with_packed_stride();
        let kk = RaggedTensor { data: bf(KVW, 10), indptr: gpu.column(&[0, rows_n as i32]) };
        let vv = bf(KVW, 11);
        let page = gpu.upload(&(0..rows_n).map(|i| i / PAGE).collect::<Vec<u32>>(), rows_n, 1, Dtype::U32);
        let offset = gpu.upload(&(0..rows_n).map(|i| i % PAGE).collect::<Vec<u32>>(), rows_n, 1, Dtype::U32);
        let us = gpu.time(REPS, || attn::kv_append(&ctx, kk, vv, &pool, page, offset).unwrap());
        report("kv_append", rows_n, us, r * u64::from(KVW) * 2 * 2 * 2);
    }
}
