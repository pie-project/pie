//! Device time of the non-GEMM kernels a Gemma-4 31B decode step fires
//! (hidden 5376, gate_up 2·21504, sliding layers of 32 query heads over 16
//! kv heads of 256, global layers of 32 over 4 of 512 with a quarter
//! rotated), at 1, 32, 128 and 512 rows, each replayed from a graph.
//! `cargo test --release --features rocm --test bench_gemma_decode -- --ignored --nocapture`
//! (`PIE_BENCH_KV` sets the cached keys a request, 900 unset.)

#![cfg(feature = "rocm")]

mod common;

use common::{Gpu, round_bf16, values};
use dtype::Dtype;
use kernels_rocm::attn::{self, plan};
use kernels_rocm::elemwise::norm;
use kernels_rocm::linear::mlp;
use kernels_rocm::{KvPool, RaggedTensor, Tensor};

const HIDDEN: u32 = 5376;
const INTER: u32 = 21504;
const PAGE: u32 = 16;

fn bf(gpu: &Gpu, rows: u32, width: u32, seed: u64) -> Tensor {
    gpu.tensor(
        &values((rows * width) as usize, seed, 0.5)
            .into_iter()
            .map(round_bf16)
            .collect::<Vec<_>>(),
        rows,
        width,
        Dtype::Bf16,
    )
}

fn workspace(gpu: &Gpu) -> plan::Workspace {
    const BYTES: usize = 512 << 20;
    let int = gpu.zeros(1, (BYTES / 4) as u32, Dtype::F32);
    let float = gpu.zeros(1, (BYTES / 4) as u32, Dtype::F32);
    plan::Workspace {
        int_ptr: int.ptr,
        int_bytes: BYTES,
        float_ptr: float.ptr,
        float_bytes: BYTES,
    }
}

/// A pool of `n` requests of `kv` keys over `hk` heads of `d`, pages
/// scattered, NHD.
fn pool(gpu: &Gpu, n: u32, kv: u32, hk: u32, d: u32) -> (KvPool, Vec<i32>) {
    let per = kv.div_ceil(PAGE).max(1);
    let total = n * per + 1;
    let indices: Vec<i32> = (0..(n * per) as i32)
        .map(|i| (i * 7919) % total as i32)
        .collect();
    let indptr: Vec<i32> = (0..=n as i32).map(|r| r * per as i32).collect();
    let last = vec![(kv - (per - 1) * PAGE) as i32; n as usize];
    let width = hk * d;
    let rows = total * PAGE;
    let plane = || gpu.zeros(rows, width, Dtype::Bf16);
    let pool = KvPool {
        keys: plane(),
        values: plane(),
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
        page_size: PAGE as i32,
        seq_stride: i64::from(width),
        head_stride: i64::from(d),
        layout: 0,
        scheme_byte: 0,
        block_size: 0,
        max_pages_per_request: per as i32,
        pages_in_batch: indices.len() as i32,
    };
    (pool, indptr)
}

fn bench_rows(gpu: &Gpu, n: u32, kv: u32) {
    let ctx = gpu.ctx();
    let r = |name: &str, us: f64, bytes: f64| {
        eprintln!(
            "gemma rows={n:3} {name:30}: {us:7.1} us {:6.0} GB/s",
            bytes / us / 1e3
        );
    };
    let row = f64::from(n) * f64::from(HIDDEN) * 2.0;

    // The post-attention chain (norm, add, pre-ffn norm) and the post-ffn
    // one (norm, add, layer scalar, next input norm).
    let x = bf(gpu, n, HIDDEN, 1);
    let mut y = bf(gpu, n, HIDDEN, 2);
    let mut t = gpu.zeros(n, HIDDEN, Dtype::Bf16);
    let mut out = gpu.zeros(n, HIDDEN, Dtype::Bf16);
    let mut scaled = gpu.zeros(n, HIDDEN, Dtype::Bf16);
    let (w0, w1, s) = (
        bf(gpu, 1, HIDDEN, 3),
        bf(gpu, 1, HIDDEN, 4),
        bf(gpu, 1, 1, 5),
    );
    r(
        "rmsnorm_residual_add post",
        gpu.time_graph(50, || {
            norm::rmsnorm_residual_add(
                &ctx,
                x,
                w0,
                1e-6,
                &mut t,
                &mut y,
                None,
                Some(norm::PostNorm {
                    weight: w1,
                    plus_one: false,
                    eps: 1e-6,
                    out: &mut out,
                }),
            )
            .unwrap();
        }),
        row * 5.0,
    );
    r(
        "rmsnorm_residual_add scale+post",
        gpu.time_graph(50, || {
            norm::rmsnorm_residual_add(
                &ctx,
                x,
                w0,
                1e-6,
                &mut t,
                &mut y,
                Some((s, &mut scaled)),
                Some(norm::PostNorm {
                    weight: w1,
                    plus_one: false,
                    eps: 1e-6,
                    out: &mut out,
                }),
            )
            .unwrap();
        }),
        row * 6.0,
    );

    let packed = bf(gpu, n, 2 * INTER, 6);
    let mut act = gpu.zeros(n, INTER, Dtype::Bf16);
    r(
        "mlp_geglu_tanh_packed",
        gpu.time_graph(50, || {
            mlp::geglu_tanh_packed(&ctx, packed, INTER, 1, &mut act).unwrap();
        }),
        f64::from(n) * f64::from(INTER) * 6.0,
    );

    // The q/k/v epilogue at both layer kinds.
    for (hq, hk, d, rot) in [(32u32, 16u32, 256u32, 256u32), (32, 4, 512, 128)] {
        let width = (hq + 2 * hk) * d;
        let packed = bf(gpu, n, width, 7);
        let (pool, _) = pool(gpu, n, PAGE, hk, d);
        let per = 1i32;
        let w_page: Vec<i32> = (0..n as i32).map(|r| r * per).collect();
        let w_off = vec![0i32; n as usize];
        let (w_page, w_off) = (gpu.column(&w_page), gpu.column(&w_off));
        let positions = gpu.column(&(0..n as i32).map(|r| 300 + r * 5).collect::<Vec<_>>());
        let (qw, kw) = (bf(gpu, 1, d, 8), bf(gpu, 1, d, 9));
        let mut q = gpu.zeros(n, hq * d, Dtype::Bf16);
        r(
            &format!("qkv_fused d{d}"),
            gpu.time_graph(50, || {
                kernels_rocm::custom::qkv_fused_qknorm_rope_vnorm_write(
                    &ctx, packed, positions, qw, kw, 1e-6, &pool, w_page, w_off, hk, d, 1e6, rot,
                    &mut q,
                )
                .unwrap();
            }),
            f64::from(n) * f64::from(width) * 4.0,
        );
    }

    // The decode attention, planned captured as the engine plans a step.
    let device = plan::Device::probe(&ctx).unwrap();
    for (hq, hk, d, window) in [(32u32, 16u32, 256u32, Some(1024u32)), (32, 4, 512, None)] {
        if n > 512 {
            break;
        }
        let (pool, indptr) = pool(gpu, n, kv, hk, d);
        let q = RaggedTensor {
            data: bf(gpu, n, hq * d, 10),
            indptr: gpu.column(&(0..=n as i32).collect::<Vec<_>>()),
        };
        let shape = plan::Shape {
            num_requests: n,
            lane_offset: 0,
            num_q_heads: hq,
            num_kv_heads: hk,
            head_dim: d,
            page_size: PAGE,
            hnd_layout: false,
        };
        let live = plan::Live {
            requests: n,
            lane_offset: 0,
            row_offset: 0,
            rows: n,
        };
        let kv_len = vec![kv as i32; n as usize];
        let built = plan::plan_decode(
            &indptr,
            &kv_len,
            shape,
            live,
            window,
            true,
            // (`PIE_BENCH_GRID` overrides the grid the plan fills.)
            std::env::var("PIE_BENCH_GRID").map_or_else(
                |_| attn::paged::decode_max_grid_size(hk, device.num_sm),
                |v| v.parse().unwrap(),
            ),
            plan::Toggles::default(),
            &device,
            workspace(gpu),
        )
        .unwrap();
        built.stage(&ctx).unwrap();
        let mut o = gpu.zeros(n, hq * d, Dtype::Bf16);
        let keys = window.map_or(kv, |w| kv.min(w));
        r(
            &format!("decode d{d} split={}", built.info.split_kv),
            gpu.time_graph(10, || {
                attn::decode(&ctx, q, &built, &pool, window, d, 0.0, &mut o).unwrap();
            }),
            f64::from(n) * f64::from(hk * d * keys) * 4.0,
        );
    }
}

/// The prefill attention over `rows` fresh prompt rows, `len` a request
/// (causal, no prefix), planned captured.
fn bench_prefill(gpu: &Gpu, rows: u32, len: u32) {
    let ctx = gpu.ctx();
    let device = plan::Device::probe(&ctx).unwrap();
    let n = rows / len;
    for (hq, hk, d, window) in [(32u32, 16u32, 256u32, Some(1024u32)), (32, 4, 512, None)] {
        let (pool, indptr) = pool(gpu, n, len, hk, d);
        let qo: Vec<i32> = (0..=n as i32).map(|r| r * len as i32).collect();
        let q = RaggedTensor {
            data: bf(gpu, rows, hq * d, 10),
            indptr: gpu.column(&qo),
        };
        let shape = plan::Shape {
            num_requests: n,
            lane_offset: 0,
            num_q_heads: hq,
            num_kv_heads: hk,
            head_dim: d,
            page_size: PAGE,
            hnd_layout: false,
        };
        let live = plan::Live {
            requests: n,
            lane_offset: 0,
            row_offset: 0,
            rows,
        };
        let kv_len = vec![len as i32; n as usize];
        let built = plan::plan_prefill(
            &qo,
            &indptr,
            &kv_len,
            rows,
            shape,
            live,
            window,
            true,
            true,
            None,
            &device,
            plan::Toggles::default(),
            workspace(gpu),
        )
        .unwrap();
        built.stage(&ctx).unwrap();
        let mut o = gpu.zeros(rows, hq * d, Dtype::Bf16);
        let us = gpu.time_graph(5, || {
            attn::prefill(&ctx, q, &built, &pool, window, d, hk, 0.0, &mut o).unwrap();
        });
        // Causal QKᵀ and PV over each row's keys (the window's at most).
        let keys: f64 = (1..=len)
            .map(|t| f64::from(window.map_or(t, |w| t.min(w))))
            .sum();
        let flops = 4.0 * f64::from(n) * keys * f64::from(hq * d);
        eprintln!(
            "gemma prefill rows={rows} len={len} d{d} split={}: {us:8.1} us {:6.1} TFLOP/s",
            built.info.split_kv,
            flops / us / 1e6
        );
    }
}

#[test]
#[ignore = "benchmark"]
fn bench_gemma_decode() {
    let Some(gpu) = Gpu::open() else { return };
    let kv = std::env::var("PIE_BENCH_KV").map_or(900, |v| v.parse().unwrap());
    let rows: Vec<u32> = std::env::var("PIE_BENCH_ROWS").map_or_else(
        |_| vec![1, 32, 128, 512, 2048, 8192],
        |v| v.split(',').map(|t| t.parse().unwrap()).collect(),
    );
    for n in rows {
        bench_rows(&gpu, n, kv);
    }
    let len = std::env::var("PIE_BENCH_LEN").map_or(2048, |v| v.parse().unwrap());
    for rows in [2048u32, 8192] {
        bench_prefill(&gpu, rows, len);
    }
}
