//! Device time of a paged decode (with its merge) at Gemma-4 31B's sliding
//! (32 query heads over 16 kv heads of 256) and global (32 over 4 of 512)
//! shapes, a few lanes, planned as a captured step plans it, at several
//! chunk floors (`kernel.sdpa.decode.chunk_floor_tokens`).
//! `cargo test --release --features rocm --test bench_decode_attn1 -- --ignored --nocapture`

#![cfg(feature = "rocm")]
#![allow(non_snake_case)]

mod common;

use common::{Gpu, values};
use dtype::Dtype;
use kernels_rocm::attn::{self, plan};
use kernels_rocm::{KvPool, RaggedTensor, Tensor};

use std::sync::atomic::{AtomicU32, Ordering::Relaxed};
static S_HQ: AtomicU32 = AtomicU32::new(32);
static S_HK: AtomicU32 = AtomicU32::new(16);
static S_D: AtomicU32 = AtomicU32::new(256);
const PS: u32 = 32;

fn workspace(gpu: &Gpu) -> plan::Workspace {
    const BYTES: usize = 64 << 20;
    let int = gpu.zeros(1, (BYTES / 4) as u32, Dtype::F32);
    let float = gpu.zeros(1, (BYTES / 4) as u32, Dtype::F32);
    plan::Workspace {
        int_ptr: int.ptr,
        int_bytes: BYTES,
        float_ptr: float.ptr,
        float_bytes: BYTES,
    }
}

/// A cache of random pages, each request's pages scattered.
fn pool(gpu: &Gpu, kv_lens: &[u32]) -> (KvPool, Vec<i32>) {
    let (HK, D) = (S_HK.load(Relaxed), S_D.load(Relaxed));
    let pages: Vec<u32> = kv_lens.iter().map(|n| n.div_ceil(PS).max(1)).collect();
    let total: u32 = pages.iter().sum::<u32>() + 1;
    let order: Vec<i32> = (0..total as i32).rev().collect();
    let mut indptr = vec![0i32];
    let mut indices = Vec::new();
    for &p in &pages {
        indices.extend_from_slice(&order[indices.len()..indices.len() + p as usize]);
        indptr.push(indices.len() as i32);
    }
    let last: Vec<i32> = kv_lens
        .iter()
        .zip(&pages)
        .map(|(&n, &p)| (n - (p - 1) * PS) as i32)
        .collect();
    let rows = total * PS;
    let width = HK * D;
    let pool = KvPool {
        keys: gpu.tensor(
            &values((rows * width) as usize, 1, 1.0),
            rows,
            width,
            Dtype::Bf16,
        ),
        values: gpu.tensor(
            &values((rows * width) as usize, 2, 1.0),
            rows,
            width,
            Dtype::Bf16,
        ),
        bf16_keys: Tensor::ABSENT,
        bf16_values: Tensor::ABSENT,
        key_scales: Tensor::ABSENT,
        value_scales: Tensor::ABSENT,
        page_indices: gpu.column(&indices),
        page_indptr: gpu.column(&indptr),
        last_page_lens: gpu.column(&last),
        row_valid: Tensor::ABSENT,
        env_min: Tensor::ABSENT,
        env_max: Tensor::ABSENT,
        has_envelopes: false,
        page_size: PS as i32,
        seq_stride: i64::from(width),
        head_stride: i64::from(D),
        layout: 0,
        scheme_byte: 0,
        block_size: 0,
        max_pages_per_request: *pages.iter().max().unwrap() as i32,
        pages_in_batch: indices.len() as i32,
    };
    (pool, indptr)
}

fn shape(n: u32) -> plan::Shape {
    let (HQ, HK, D) = (S_HQ.load(Relaxed), S_HK.load(Relaxed), S_D.load(Relaxed));
    plan::Shape {
        num_requests: n,
        lane_offset: 0,
        num_q_heads: HQ,
        num_kv_heads: HK,
        head_dim: D,
        page_size: PS,
        hnd_layout: false,
    }
}

fn decode(gpu: &Gpu, batch: u32, ctx_len: u32, floor: u32) -> f64 {
    let (HQ, HK, D) = (S_HQ.load(Relaxed), S_HK.load(Relaxed), S_D.load(Relaxed));
    let ctx = gpu.ctx();
    let device = plan::Device::probe(&ctx).unwrap();
    let kv_lens = vec![ctx_len; batch as usize];
    let (pool, indptr) = pool(gpu, &kv_lens);
    let q = RaggedTensor {
        data: gpu.tensor(
            &values((batch * HQ * D) as usize, 3, 1.0),
            batch,
            HQ * D,
            Dtype::Bf16,
        ),
        indptr: gpu.column(&(0..=batch as i32).collect::<Vec<_>>()),
    };
    let live = plan::Live {
        requests: batch,
        lane_offset: 0,
        row_offset: 0,
        rows: batch,
    };
    let kv: Vec<i32> = kv_lens.iter().map(|&x| x as i32).collect();
    let max_grid =
        attn::paged::decode_max_grid_size(HK, kernels_rocm::hip::multiprocessors(0).unwrap());
    let built = plan::plan_decode(
        &indptr,
        &kv,
        shape(batch),
        live,
        None,
        true,
        max_grid,
        plan::Toggles { chunk_floor_tokens: floor, ..plan::Toggles::default() },
        &device,
        workspace(gpu),
    )
    .unwrap();
    built.stage(&ctx).unwrap();
    let mut o = gpu.zeros(batch, HQ * D, Dtype::Bf16);
    let mut lse = gpu.zeros(batch, HQ, Dtype::F32);
    gpu.time_graph(10, || {
        attn::decode_lse(&gpu.ctx(), q, &built, &pool, None, D, 0.0, &mut o, &mut lse).unwrap()
    })
}

#[test]
#[ignore = "benchmark"]
fn bench_decode_attn1() {
    let Some(gpu) = Gpu::open() else { return };
    for (hq, hk, d) in [(32u32, 16u32, 256u32), (32, 4, 512), (24, 4, 256)] {
        S_HQ.store(hq, Relaxed);
        S_HK.store(hk, Relaxed);
        S_D.store(d, Relaxed);
        for batch in [1u32, 8, 32] {
            for ctx in [256u32, 512, 1024] {
                let mut line = format!("hq={hq} hk={hk} d={d} batch={batch} ctx={ctx:5}:");
                for floor in [256u32, 128, 64, 32] {
                    let us = decode(&gpu, batch, ctx, floor);
                    line += &format!("  floor {floor:3} {us:6.1} us");
                }
                eprintln!("{line}");
            }
        }
    }
}
