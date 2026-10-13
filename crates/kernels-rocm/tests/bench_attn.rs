//! Device time of paged decode and causal prefill at Qwen 3.5's
//! full-attention shape (8 query heads over 2 kv heads of 256, pages of 16),
//! each call replayed from a graph as the engine runs it.
//! `cargo test --release --features rocm --test bench_attn -- --ignored --nocapture`

#![cfg(feature = "rocm")]

mod common;

use common::{Gpu, values};
use dtype::Dtype;
use kernels_rocm::attn::{self, plan};
use kernels_rocm::{KvPool, RaggedTensor, Tensor};

const HQ: u32 = 8;
const HK: u32 = 2;
const D: u32 = 256;
const PS: u32 = 16;

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

fn decode(gpu: &Gpu, batch: u32, ctx_len: u32) -> f64 {
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
        false,
        max_grid,
        plan::Toggles::default(),
        &device,
        workspace(gpu),
    )
    .unwrap();
    built.stage(&ctx).unwrap();
    eprintln!(
        "  plan: split={} items={}",
        built.info.split_kv, built.info.padded_batch_size
    );
    let mut o = gpu.zeros(batch, HQ * D, Dtype::Bf16);
    let mut lse = gpu.zeros(batch, HQ, Dtype::F32);
    gpu.time_graph(10, || {
        attn::decode_lse(&gpu.ctx(), q, &built, &pool, None, D, 0.0, &mut o, &mut lse).unwrap()
    })
}

fn prefill(gpu: &Gpu, lens: &[u32]) -> f64 {
    let ctx = gpu.ctx();
    let device = plan::Device::probe(&ctx).unwrap();
    let n = lens.len() as u32;
    let total: u32 = lens.iter().sum();
    let (pool, indptr) = pool(gpu, lens);
    let mut qo = vec![0i32];
    for &l in lens {
        qo.push(qo.last().unwrap() + l as i32);
    }
    let q = RaggedTensor {
        data: gpu.tensor(
            &values((total * HQ * D) as usize, 5, 1.0),
            total,
            HQ * D,
            Dtype::Bf16,
        ),
        indptr: gpu.column(&qo),
    };
    let live = plan::Live {
        requests: n,
        lane_offset: 0,
        row_offset: 0,
        rows: total,
    };
    let kv: Vec<i32> = lens.iter().map(|&x| x as i32).collect();
    let built = plan::plan_prefill(
        &qo,
        &indptr,
        &kv,
        total,
        shape(n),
        live,
        None,
        true,
        false,
        None,
        &device,
        plan::Toggles::default(),
        workspace(gpu),
    )
    .unwrap();
    built.stage(&ctx).unwrap();
    let mut o = gpu.zeros(total, HQ * D, Dtype::Bf16);
    gpu.time_graph(10, || {
        attn::prefill(&gpu.ctx(), q, &built, &pool, None, D, HK, 0.0, &mut o).unwrap()
    })
}

#[test]
#[ignore = "benchmark"]
fn bench_attn() {
    let Some(gpu) = Gpu::open() else { return };
    for (batch, ctx) in [
        (1, 1024),
        (1, 8192),
        (8, 1024),
        (8, 8192),
        (32, 1024),
        (32, 8192),
    ] {
        eprintln!(
            "pie decode batch={batch:3} ctx={ctx:5}: {:8.1} us",
            decode(&gpu, batch, ctx)
        );
    }
    for lens in [vec![23u32; 256], vec![8192]] {
        eprintln!("pie prefill {lens:?}: {:8.1} us", prefill(&gpu, &lens));
    }
}
