#![cfg(feature = "cuda")]

//! DeepSeek-V4's attention at prefill widths in one tcgen05 pass: each row's
//! 128-key window and its compressed entries (every visible one, or an
//! index's ascending selection) in one softmax, the sinks over it, against
//! the same arithmetic in f64. Rows pair up within a request and across a
//! request's end, and the fire ends on a lone row.

use crate::common::{Gpu, Lcg, from_bf16};
use dtype::Dtype;
use kernels_cuda::attn::dsv4_prefill::{self, Pooled};
use kernels_cuda::tensor::{KvPool, Tensor};

const PAGE: usize = 64;
const HEADS: usize = 64;
const D: usize = 512;

fn pool_of(gpu: &mut Gpu, keys: u64, cells: usize, pages: &[u32], indptr: &[u32]) -> KvPool {
    KvPool {
        keys: Tensor::new(keys, cells as u32, D as u32, Dtype::Bf16),
        values: Tensor::ABSENT,
        bf16_keys: Tensor::ABSENT,
        bf16_values: Tensor::ABSENT,
        key_scales: Tensor::ABSENT,
        value_scales: Tensor::ABSENT,
        page_indices: Tensor::new(gpu.up(pages), pages.len() as u32, 1, Dtype::U32),
        page_indptr: Tensor::new(gpu.up(indptr), indptr.len() as u32, 1, Dtype::U32),
        last_page_lens: Tensor::ABSENT,
        row_valid: Tensor::ABSENT,
        env_min: Tensor::ABSENT,
        env_max: Tensor::ABSENT,
        has_envelopes: false,
        page_size: PAGE as i32,
        seq_stride: D as i64,
        head_stride: D as i64,
        layout: 0,
        scheme_byte: 0,
        block_size: 0,
        max_pages_per_request: 32,
        pages_in_batch: pages.len() as i32,
    }
}

/// Two requests' pages interleaved: request r's page i is pool page
/// 2 i + r, `per` pages each.
fn paging(per: usize) -> (Vec<u32>, Vec<u32>) {
    let mut pages = Vec::new();
    for r in 0..2u32 {
        pages.extend((0..per as u32).map(|i| 2 * i + r));
    }
    (pages, vec![0, per as u32, 2 * per as u32])
}

fn check(ratio: usize, top_k: Option<usize>) {
    let per = 18;
    let cells = 2 * per * PAGE;
    let (pages, indptr) = paging(per);
    let cell = |req: usize, pos: usize| pages[req * per + pos / PAGE] as usize * PAGE + pos % PAGE;
    // Request 0's rows 600..608, request 1's 1000..1005: the pair (607,
    // 1000) straddles the two, and the last row stands alone.
    let rows: Vec<(usize, usize)> = (600..608)
        .map(|p| (0, p))
        .chain((1000..1005).map(|p| (1, p)))
        .collect();
    let n = rows.len();
    let mut lcg = Lcg::seeded(0xd5f4 + ratio as u64);
    let (q_bits, q) = lcg.row(n * HEADS * D);
    let (w_bits, w) = lcg.row(cells * D);
    let (e_bits, e) = lcg.row(cells * D);
    let sinks: Vec<f32> = (0..HEADS).map(|h| (h as f32 - 32.0) / 16.0).collect();
    // Each row's selection: ascending ids it sees, neighbours sharing most.
    let mut sel = vec![-1i32; n * top_k.unwrap_or(0)];
    if let Some(k) = top_k {
        for (r, &(_, pos)) in rows.iter().enumerate() {
            let visible = (pos + 1) / ratio;
            let mut ids: Vec<i32> = (0..visible as i32)
                .filter(|id| (id * 7 + (r as i32 / 3)) % 3 != 0)
                .collect();
            ids.truncate(k - (r % 4));
            for (i, id) in ids.iter().enumerate() {
                sel[r * k + i] = *id;
            }
        }
    }

    let mut gpu = Gpu::open();
    let positions: Vec<i32> = rows.iter().map(|&(_, p)| p as i32).collect();
    let requests: Vec<i32> = rows.iter().map(|&(r, _)| r as i32).collect();
    let q_t = Tensor::new(gpu.up(&q_bits), n as u32, (HEADS * D) as u32, Dtype::Bf16);
    let pos_t = Tensor::new(gpu.up(&positions), n as u32, 1, Dtype::I32);
    let req_t = Tensor::new(gpu.up(&requests), n as u32, 1, Dtype::I32);
    let w_at = gpu.up(&w_bits);
    let win = pool_of(&mut gpu, w_at, cells, &pages, &indptr);
    let e_at = gpu.up(&e_bits);
    let ent = pool_of(&mut gpu, e_at, cells, &pages, &indptr);
    let sink_t = Tensor::new(gpu.up(&sinks), HEADS as u32, 1, Dtype::F32);
    let sel_t = top_k.map(|k| {
        (
            Tensor::new(gpu.up(&sel), n as u32, k as u32, Dtype::I32),
            k as u32,
        )
    });
    let o_at = gpu.zeros(n * HEADS * D * 2);
    let mut o = Tensor::new(o_at, n as u32, (HEADS * D) as u32, Dtype::Bf16);
    let scale = 1.0 / (D as f32).sqrt();
    let pooled = Pooled {
        entries: &ent,
        ratio: ratio as u32,
        selection: sel_t,
    };
    assert!(
        dsv4_prefill::takes(HEADS as u32, D as u32, Some(128), &win, &pooled),
        "the kernel takes DeepSeek-V4-Flash's shape"
    );
    let ctx = gpu.ctx();
    dsv4_prefill::prefill(
        &ctx,
        "attention.pool_attention",
        q_t,
        pos_t,
        req_t,
        &win,
        &pooled,
        sink_t,
        scale,
        &mut o,
    )
    .expect("the prefill attention fires");
    gpu.sync();
    let got: Vec<u16> = gpu.down(o_at, n * HEADS * D);

    let mut worst = 0f64;
    for (r, &(req, pos)) in rows.iter().enumerate() {
        // The keys this row reads: its window, then its entries.
        let mut keys: Vec<&[f32]> = Vec::new();
        for p in pos.saturating_sub(127)..=pos {
            let c = cell(req, p);
            keys.push(&w[c * D..(c + 1) * D]);
        }
        let visible = (pos + 1) / ratio;
        let ids: Vec<usize> = match top_k {
            Some(k) => sel[r * k..(r + 1) * k]
                .iter()
                .take_while(|&&id| id >= 0)
                .map(|&id| id as usize)
                .collect(),
            None => (0..visible).collect(),
        };
        for id in ids {
            let c = cell(req, (id + 1) * ratio - 1);
            keys.push(&e[c * D..(c + 1) * D]);
        }
        for h in 0..HEADS {
            let qh = &q[(r * HEADS + h) * D..(r * HEADS + h + 1) * D];
            let s: Vec<f64> = keys
                .iter()
                .map(|k| {
                    qh.iter()
                        .zip(k.iter())
                        .map(|(a, b)| f64::from(*a) * f64::from(*b))
                        .sum::<f64>()
                        * f64::from(scale)
                })
                .collect();
            let m = s.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let z: f64 = s.iter().map(|x| (x - m).exp()).sum();
            let lse = m + z.ln();
            let sinked = 1.0 / (1.0 + (f64::from(sinks[h]) - lse).exp());
            for d in 0..D {
                let want: f64 = s
                    .iter()
                    .zip(&keys)
                    .map(|(x, k)| (x - m).exp() / z * f64::from(k[d]))
                    .sum::<f64>()
                    * sinked;
                let g = f64::from(from_bf16(got[(r * HEADS + h) * D + d]));
                worst = worst.max((g - want).abs() / want.abs().max(0.02));
            }
        }
    }
    assert!(
        worst <= 4e-2,
        "ratio {ratio}, top_k {top_k:?}: an output element differs by {worst} relative"
    );
}

#[test]
fn deepseeks_prefill_attention_lands_what_its_four_ops_land() {
    // A CSA layer: ratio 4, a selection of 64 a row; an HCA layer: ratio
    // 128, every visible entry.
    check(4, Some(64));
    check(128, None);
    // Every visible entry at ratio 4: a page's sixteen entries land as one box.
    check(4, None);
}

/// DeepSeek-V4-Flash at a prefill's width: 2048 rows of one 4096-position
/// request, each reading its 128-key window and 512 selected entries.
#[test]
#[ignore = "a measurement; run it by name under the GPU lock"]
fn deepseeks_prefill_attention_paces_a_prefill() {
    let (ratio, k, n, span) = (4usize, 512usize, 2048usize, 4096usize);
    let per = span / PAGE;
    let (pages, indptr) = paging(per);
    let cells = 2 * per * PAGE;
    let mut lcg = Lcg::seeded(0x7a11);
    let (q_bits, _) = lcg.row(n * HEADS * D);
    let (w_bits, _) = lcg.row(cells * D);
    let positions: Vec<i32> = (0..n as i32).map(|r| (span - n) as i32 + r).collect();
    let requests = vec![0i32; n];
    let mut sel = vec![-1i32; n * k];
    for (r, &pos) in positions.iter().enumerate() {
        let visible = (pos as usize + 1) / ratio;
        // The same 512 ids a row but for one in sixteen, as neighbouring
        // rows' indexes mostly agree.
        let ids: Vec<i32> = (0..visible as i32)
            .filter(|id| {
                (id * 2_654_435_761u32 as i32).rem_euclid(1024)
                    < (k as i32 * 1024 / visible as i32).min(1024)
                    || (*id as usize + r) % 16 == 0
            })
            .take(k)
            .collect();
        sel[r * k..r * k + ids.len()].copy_from_slice(&ids);
    }
    let mut gpu = Gpu::open();
    let q_t = Tensor::new(gpu.up(&q_bits), n as u32, (HEADS * D) as u32, Dtype::Bf16);
    let pos_t = Tensor::new(gpu.up(&positions), n as u32, 1, Dtype::I32);
    let req_t = Tensor::new(gpu.up(&requests), n as u32, 1, Dtype::I32);
    let w_at = gpu.up(&w_bits);
    let win = pool_of(&mut gpu, w_at, cells, &pages, &indptr);
    let sinks = vec![0f32; HEADS];
    let sink_t = Tensor::new(gpu.up(&sinks), HEADS as u32, 1, Dtype::F32);
    let sel_t = Tensor::new(gpu.up(&sel), n as u32, k as u32, Dtype::I32);
    let o_at = gpu.zeros(n * HEADS * D * 2);
    let mut o = Tensor::new(o_at, n as u32, (HEADS * D) as u32, Dtype::Bf16);
    let pooled = Pooled {
        entries: &win,
        ratio: ratio as u32,
        selection: Some((sel_t, k as u32)),
    };
    let ctx = gpu.ctx();
    let mut fire = || {
        dsv4_prefill::prefill(
            &ctx,
            "attention.pool_attention",
            q_t,
            pos_t,
            req_t,
            &win,
            &pooled,
            sink_t,
            0.044,
            &mut o,
        )
        .expect("the prefill attention fires");
    };
    fire();
    gpu.sync();
    let iters = 20;
    let t0 = std::time::Instant::now();
    for _ in 0..iters {
        fire();
    }
    gpu.sync();
    let us = t0.elapsed().as_secs_f64() * 1e6 / f64::from(iters);
    let flops = (n * HEADS * (k + 128) * D * 4) as f64;
    println!(
        "| {n} rows, window + {k} selected | {us:.0} us | {:.1} TFLOP/s useful |",
        flops / us / 1e6
    );
}
