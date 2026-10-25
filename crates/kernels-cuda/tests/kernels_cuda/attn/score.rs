#![cfg(feature = "cuda")]

//! `attn.score_capture` against the host softmax: each lane's mass over its
//! first `kv_max` keys, averaged over its last `observe` query rows, a run a
//! head, written at the lane's row of a `[lanes, heads * kv_max]` plane.

use crate::common::attn_ref::{col, pool, ten};
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

    let mut want = vec![9.0f32; (lane_offset + n) * qh * kv_max];
    for (r, &(new, prefix)) in reqs.iter().enumerate() {
        let len = new + prefix;
        let seen = observe.min(new);
        let cell = |t: usize| pages[r][t / PAGE] * PAGE + t % PAGE;
        for h in 0..qh {
            let at = ((lane_offset + r) * qh + h) * kv_max;
            want[at..at + kv_max].fill(0.0);
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
                    want[at + t] += (x - top).exp() / sum / seen as f32;
                }
            }
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
    let mut scores = ten(at, lane_offset + n, qh * kv_max, F32);
    kernels_cuda::attn::score::capture(
        &ctx,
        qt,
        &plan,
        &kvp,
        d as u32,
        kvh as u32,
        scale,
        observe as u32,
        lane_offset as u32,
        kv_max as u32,
        &mut scores,
    )
    .expect("the capture fires");
    gpu.sync();
    let got: Vec<f32> = gpu.down(at, want.len());
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        assert!((g - w).abs() <= 2e-3, "at {i}: got {g}, want {w}");
    }
}
