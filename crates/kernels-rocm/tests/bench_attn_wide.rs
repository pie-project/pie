//! Device time of DeepSeek-V4's sliding-window attention over its one
//! shared 512-wide head (64 query heads, a 128-key window), as its decode
//! runs it: the prefill reader over one query row a request, each call
//! replayed from a graph.
//! `cargo test --release --features rocm --test bench_attn_wide -- --ignored --nocapture`

#![cfg(feature = "rocm")]

mod common;

use common::{Gpu, values};
use dtype::Dtype;
use kernels_rocm::attn::{self, plan};
use kernels_rocm::{KvPool, RaggedTensor, Tensor};

const HQ: u32 = 64;
const HK: u32 = 1;
const D: u32 = 512;
/// Keys a page (`PIE_BENCH_PAGE`, 32 unset).
fn page() -> u32 {
    std::env::var("PIE_BENCH_PAGE").map_or(32, |v| v.parse().unwrap())
}
const WINDOW: u32 = 128;

fn workspace(gpu: &Gpu) -> plan::Workspace {
    const BYTES: usize = 256 << 20;
    let int = gpu.zeros(1, (BYTES / 4) as u32, Dtype::F32);
    let float = gpu.zeros(1, (BYTES / 4) as u32, Dtype::F32);
    plan::Workspace {
        int_ptr: int.ptr,
        int_bytes: BYTES,
        float_ptr: float.ptr,
        float_bytes: BYTES,
    }
}

/// A cache of random pages, each request's pages scattered; keys and values
/// one plane, as DeepSeek's shared latent is.
fn pool(gpu: &Gpu, kv_lens: &[u32]) -> (KvPool, Vec<i32>) {
    let pages: Vec<u32> = kv_lens.iter().map(|n| n.div_ceil(page()).max(1)).collect();
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
        .map(|(&n, &p)| (n - (p - 1) * page()) as i32)
        .collect();
    let rows = total * page();
    let width = HK * D;
    let plane = gpu.tensor(
        &values((rows * width) as usize, 1, 1.0),
        rows,
        width,
        Dtype::Bf16,
    );
    let pool = KvPool {
        keys: plane,
        values: plane,
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
        page_size: page() as i32,
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

/// (µs, split) of one call over `n` requests of `qo` query rows each (a
/// decode's one, or a draft's verify) and `kv` cached keys, the window
/// `WINDOW`.
fn decode_rows(gpu: &Gpu, n: u32, qo: u32, kv: u32) -> (f64, bool) {
    let ctx = gpu.ctx();
    let device = plan::Device::probe(&ctx).unwrap();
    let kv_lens = vec![kv; n as usize];
    let (pool, indptr) = pool(gpu, &kv_lens);
    let rows = n * qo;
    let qo: Vec<i32> = (0..=n as i32).map(|r| r * qo as i32).collect();
    let q = RaggedTensor {
        data: gpu.tensor(
            &values((rows * HQ * D) as usize, 5, 1.0),
            rows,
            HQ * D,
            Dtype::Bf16,
        ),
        indptr: gpu.column(&qo),
    };
    let shape = plan::Shape {
        num_requests: n,
        lane_offset: 0,
        num_q_heads: HQ,
        num_kv_heads: HK,
        head_dim: D,
        page_size: page(),
        hnd_layout: false,
    };
    let live = plan::Live {
        requests: n,
        lane_offset: 0,
        row_offset: 0,
        rows,
    };
    let kv_len: Vec<i32> = kv_lens.iter().map(|&x| x as i32).collect();
    let built = plan::plan_prefill(
        &qo,
        &indptr,
        &kv_len,
        rows,
        shape,
        live,
        Some(WINDOW),
        true,
        // Captured, as the engine plans a decode step.
        true,
        None,
        &device,
        plan::Toggles::default(),
        workspace(gpu),
    )
    .unwrap();
    built.stage(&ctx).unwrap();
    let mut o = gpu.zeros(rows, HQ * D, Dtype::Bf16);
    let mut lse = gpu.zeros(rows, HQ, Dtype::F32);
    let us = gpu.time_graph(20, || {
        attn::prefill_lse(
            &gpu.ctx(),
            q,
            &built,
            &pool,
            Some(WINDOW),
            D,
            HK,
            0.0,
            &mut o,
            &mut lse,
        )
        .unwrap()
    });
    (us, built.info.split_kv)
}

#[test]
#[ignore = "benchmark"]
fn bench_attn_wide() {
    let Some(gpu) = Gpu::open() else { return };
    let kvs: Vec<u32> = std::env::var("PIE_BENCH_KV")
        .map_or_else(|_| vec![600], |v| v.split(',').map(|t| t.parse().unwrap()).collect());
    let qos: Vec<u32> = std::env::var("PIE_BENCH_QO")
        .map_or_else(|_| vec![1, 2], |v| v.split(',').map(|t| t.parse().unwrap()).collect());
    for qo in qos {
        for (n, kv) in [1u32, 32, 128, 512]
            .into_iter()
            .flat_map(|n| kvs.iter().map(move |&kv| (n, kv)))
        {
            let (us, split) = decode_rows(&gpu, n, qo, kv);
            // The window's keys (one plane read for K and V), the queries
            // and the results.
            let bytes = f64::from(n) * f64::from((WINDOW + qo) * D * 2 + 2 * qo * HQ * D * 2);
            eprintln!(
                "dsv4 window rows={n:4}x{qo} kv={kv:5}: {us:8.1} us  {:6.0} GB/s  split={split}",
                bytes / us / 1e3
            );
        }
    }
}
