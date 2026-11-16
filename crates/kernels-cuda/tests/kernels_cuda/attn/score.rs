#![cfg(feature = "cuda")]

//! `attn.score_capture` against the host softmax: each capturing lane's mass
//! over its first `kv_max` keys (the load's context), averaged over its last
//! `observe` query rows and over the query heads, written at the row its
//! entry of the table names; `attn.score_pages` sums it a page into the row
//! after it.

use crate::common::attn_ref::{col, pool, ten};
use crate::common::spread::agrees;
use crate::common::{Gpu, Lcg};
use dtype::Dtype::{Bf16, F32, I32};
use kernels_cuda::attn::plan::{self, Device, Live, Shape, Toggles, Workspace};
use kernels_cuda::tensor::{KvPool, RaggedTensor};

const PAGE: usize = 16;

#[test]
fn a_score_capture_lands_each_lanes_mass_at_its_row() {
    let (qh, kvh, d) = (4usize, 2usize, 128usize);
    // (new rows, cached rows) a lane: a prefill, a decode, a fresh prompt.
    let reqs = [(5usize, 20usize), (1, 33), (2, 0)];
    let (observe, lane_offset, kv_max) = (3usize, 1usize, 40usize);
    let (width, qw, scale) = (kvh * d, qh * d, (d as f32).sqrt().recip());
    let n = reqs.len();
    let (mut qo, mut kv_indptr, mut kv_len, mut last, mut pages) =
        (vec![0i32], vec![0i32], vec![], vec![], vec![]);
    let mut next = 1usize;
    for (r, &(new, prefix)) in reqs.iter().enumerate() {
        let len = new + prefix;
        let np = len.div_ceil(PAGE);
        pages.push((0..np).map(|i| next + np - 1 - i).collect::<Vec<_>>());
        next += np;
        qo.push(qo[r] + new as i32);
        kv_indptr.push(kv_indptr[r] + np as i32);
        kv_len.push(len as i32);
        last.push(((len - 1) % PAGE + 1) as i32);
    }
    let (rows, cells) = (qo[n] as usize, next * PAGE);
    let mut lcg = Lcg::seeded(0x5c0e);
    let (k_raw, k) = lcg.row(cells * width);
    let (q_raw, q) = lcg.row(rows * qw);

    // A lane's row: its key row of `kv_max` (the load's context), then its
    // page row. The lanes before `lane_offset` capture none.
    let page_count = kv_max.div_ceil(PAGE);
    let row = kv_max + page_count;
    let mut want = vec![9.0f32; (lane_offset + n) * row];
    for (r, &(new, prefix)) in reqs.iter().enumerate() {
        let len = new + prefix;
        let seen = observe.min(new);
        let cell = |t: usize| pages[r][t / PAGE] * PAGE + t % PAGE;
        let at = (lane_offset + r) * row;
        want[at..at + kv_max].fill(0.0);
        for h in 0..qh {
            for w in 0..seen {
                let row = qo[r + 1] as usize - seen + w;
                let reach = len - seen + w + 1;
                let kh = h / (qh / kvh);
                let dots: Vec<f32> = (0..reach)
                    .map(|t| {
                        let key = &k[cell(t) * width + kh * d..][..d];
                        let query = &q[row * qw + h * d..][..d];
                        key.iter().zip(query).map(|(a, b)| a * b).sum::<f32>() * scale
                    })
                    .collect();
                let top = dots.iter().copied().fold(f32::MIN, f32::max);
                let sum: f32 = dots.iter().map(|x| (x - top).exp()).sum();
                for (t, x) in dots.iter().enumerate().take(kv_max) {
                    want[at + t] += (x - top).exp() / sum / seen as f32 / qh as f32;
                }
            }
        }
        for p in 0..page_count {
            want[at + kv_max + p] = want[at + p * PAGE..at + ((p + 1) * PAGE).min(kv_max)]
                .iter()
                .sum();
        }
    }

    let mut gpu = Gpu::open();
    let flat: Vec<i32> = pages.iter().flatten().map(|&p| p as i32).collect();
    let ids = col(&mut gpu, &flat, I32);
    let indptr = col(&mut gpu, &kv_indptr, I32);
    let keys = ten(gpu.up(&k_raw), cells, width, Bf16);
    let kvp = KvPool {
        last_page_lens: col(&mut gpu, &last, I32),
        head_stride: d as i64,
        ..pool(keys, ids, indptr, PAGE)
    };
    let qt = RaggedTensor {
        data: ten(gpu.up(&q_raw), rows, qw, Bf16),
        indptr: col(&mut gpu, &qo, I32),
    };
    let (int_ptr, float_ptr, bytes) = (gpu.zeros(64 << 20), gpu.zeros(64 << 20), 64 << 20);
    let ws = Workspace {
        int_ptr,
        int_bytes: bytes,
        float_ptr,
        float_bytes: bytes,
    };
    let ctx = gpu.ctx();
    let device = Device::probe(&ctx).expect("the device answers");
    let shape = Shape {
        num_requests: n as u32,
        lane_offset: 0,
        num_q_heads: qh as u32,
        num_kv_heads: kvh as u32,
        head_dim: d as u32,
        page_size: PAGE as u32,
        hnd_layout: false,
    };
    let live = Live {
        requests: n as u32,
        lane_offset: 0,
        row_offset: 0,
        rows: rows as u32,
    };
    let plan = plan::plan_prefill(
        &qo,
        &kv_indptr,
        &kv_len,
        rows as u32,
        shape,
        live,
        None,
        true,
        false,
        None,
        &device,
        Toggles::default(),
        ws,
    )
    .expect("the prefill plans");
    plan.stage(&ctx).expect("the plan uploads");
    let at = gpu.up(&want.iter().map(|_| 9.0f32).collect::<Vec<_>>());
    // Each lane's row where its entry of the table says, as a recorded
    // capture reads them; the lanes before the window's none.
    let table: Vec<u64> = (0..lane_offset + n)
        .map(|lane| {
            if lane < lane_offset {
                0
            } else {
                at + (lane * row * 4) as u64
            }
        })
        .collect();
    let scores = kernels_cuda::attn::score::Rows {
        table: gpu.up(&table),
        context: kv_max as u32,
    };
    let fire = |ctx: &kernels_cuda::jit::Ctx, pool: &KvPool| {
        kernels_cuda::attn::score::capture(
            ctx,
            qt,
            &plan,
            pool,
            d as u32,
            kvh as u32,
            qh as u32,
            scale,
            observe as u32,
            lane_offset as u32,
            scores,
            false,
        )
        .expect("the capture fires");
        kernels_cuda::attn::score::pages(ctx, scores, n as u32, lane_offset as u32, PAGE as u32)
            .expect("the pages sum");
    };
    fire(&ctx, &kvp);
    gpu.sync();
    let got: Vec<f32> = gpu.down(at, want.len());
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        assert!((g - w).abs() <= 2e-3, "at {i}: got {g}, want {w}");
    }
    // The same capture over the pool's pages spread apart, byte for byte.
    let outs = [(at, want.len() * 4)];
    agrees(
        &mut gpu,
        "score capture",
        &[kvp],
        &[],
        &outs,
        |_, pools, _| fire(&ctx, &pools[0]),
    );
}
