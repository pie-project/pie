#![cfg(feature = "cuda")]

// MiniMax-M3's indexer: each head scores a block by the best key in it that
// the row sees, keeps the row's own block and the best `TOP_K - 1` others, and
// that kv head's queries attend over its blocks alone. The ranking and the
// selected attention it feeds are read against a host reference, over a
// fresh prefill and a continuation whose page tables are out of order.

use crate::common::{Gpu, Lcg, close, from_bf16};
use dtype::Dtype;
use kernels_cuda::attn::{index, selected};
use kernels_cuda::tensor::{KvPool, RaggedTensor, Tensor};
use kernels_cuda::tuning::Tuning;

const IDX_HEADS: usize = 4;
const IDX_DIM: usize = 128;
const KV_HEADS: usize = 4;
const Q_HEADS: usize = 8;
const HEAD_DIM: usize = 128;
const BLOCK: usize = 16;
const TOP_K: usize = 3;
const PAGE: usize = 16;

const Q_LENS: [usize; 2] = [70, 2];
const KV_LENS: [usize; 2] = [70, 90];
const PAGE_INDICES: [i32; 11] = [7, 2, 9, 0, 4, 10, 1, 8, 3, 6, 5];
const PAGE_INDPTR: [i32; 3] = [0, 5, 11];
const LAST_PAGE_LENS: [i32; 2] = [6, 10];

fn slot(request: usize, position: usize) -> usize {
    let page = PAGE_INDICES[PAGE_INDPTR[request] as usize + position / PAGE] as usize;
    page * PAGE + position % PAGE
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Each row's run of blocks for each head, as the reference ranks them.
fn ranking(iq: &[f32], ik: &[f32]) -> Vec<i32> {
    let mut out = Vec::new();
    let mut t = 0;
    for r in 0..Q_LENS.len() {
        for i in 0..Q_LENS[r] {
            let abs = KV_LENS[r] - Q_LENS[r] + i;
            let blocks = abs / BLOCK + 1;
            for h in 0..IDX_HEADS {
                let q = &iq[(t * IDX_HEADS + h) * IDX_DIM..][..IDX_DIM];
                let mut run: Vec<i32> = if blocks <= TOP_K {
                    (0..blocks as i32).collect()
                } else {
                    let mut scored: Vec<(f32, usize)> = (0..blocks - 1)
                        .map(|b| {
                            let best = (b * BLOCK..((b + 1) * BLOCK).min(abs + 1))
                                .map(|j| dot(q, &ik[slot(r, j) * IDX_DIM..][..IDX_DIM]))
                                .fold(f32::NEG_INFINITY, f32::max);
                            (best, b)
                        })
                        .collect();
                    scored.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
                    let mut run: Vec<i32> =
                        scored[..TOP_K - 1].iter().map(|s| s.1 as i32).collect();
                    run.push((blocks - 1) as i32);
                    run.sort_unstable();
                    run
                };
                run.resize(TOP_K, -1);
                out.extend(run);
            }
            t += 1;
        }
    }
    out
}

/// Every query head over its kv head's blocks.
fn attention(q: &[f32], k: &[f32], v: &[f32], selection: &[i32], sm_scale: f32) -> Vec<f32> {
    let tokens: usize = Q_LENS.iter().sum();
    let mut out = vec![0f32; tokens * Q_HEADS * HEAD_DIM];
    let mut t = 0;
    for r in 0..Q_LENS.len() {
        for i in 0..Q_LENS[r] {
            let abs = KV_LENS[r] - Q_LENS[r] + i;
            for h in 0..Q_HEADS {
                let kv = h / (Q_HEADS / KV_HEADS);
                let run = &selection[(t * KV_HEADS + kv) * TOP_K..][..TOP_K];
                let keys: Vec<usize> = (0..=abs)
                    .filter(|j| run.contains(&((j / BLOCK) as i32)))
                    .collect();
                let qh = &q[(t * Q_HEADS + h) * HEAD_DIM..][..HEAD_DIM];
                let at = |j: usize| (slot(r, j) * KV_HEADS + kv) * HEAD_DIM;
                let scores: Vec<f32> = keys
                    .iter()
                    .map(|&j| dot(qh, &k[at(j)..][..HEAD_DIM]) * sm_scale)
                    .collect();
                let m = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let weights: Vec<f32> = scores.iter().map(|s| (s - m).exp()).collect();
                let sum: f32 = weights.iter().sum();
                let o = &mut out[(t * Q_HEADS + h) * HEAD_DIM..][..HEAD_DIM];
                for (&j, w) in keys.iter().zip(&weights) {
                    for (d, x) in o.iter_mut().zip(&v[at(j)..][..HEAD_DIM]) {
                        *d += w / sum * x;
                    }
                }
            }
            t += 1;
        }
    }
    out
}

fn pool(keys: u64, values: u64, width: usize, tables: [u64; 3]) -> KvPool {
    let slots = (PAGE_INDICES.len() * PAGE) as u32;
    KvPool {
        keys: Tensor::new(keys, slots, width as u32, Dtype::Bf16),
        values: Tensor::new(values, slots, width as u32, Dtype::Bf16),
        bf16_keys: Tensor::ABSENT,
        bf16_values: Tensor::ABSENT,
        key_scales: Tensor::ABSENT,
        value_scales: Tensor::ABSENT,
        page_indices: Tensor::new(tables[0], PAGE_INDICES.len() as u32, 1, Dtype::I32),
        page_indptr: Tensor::new(tables[1], PAGE_INDPTR.len() as u32, 1, Dtype::I32),
        last_page_lens: Tensor::new(tables[2], LAST_PAGE_LENS.len() as u32, 1, Dtype::I32),
        row_valid: Tensor::ABSENT,
        env_min: Tensor::ABSENT,
        env_max: Tensor::ABSENT,
        has_envelopes: false,
        page_size: PAGE as i32,
        seq_stride: width as i64,
        head_stride: if width == IDX_DIM {
            width as i64
        } else {
            HEAD_DIM as i64
        },
        layout: 0,
        scheme_byte: 0,
        block_size: 0,
        max_pages_per_request: 6,
        pages_in_batch: PAGE_INDICES.len() as i32,
    }
}

#[test]
fn the_block_max_ranking_answers_the_reference() {
    let mut gpu = Gpu::open();
    let mut rng = Lcg::seeded(0x4d33_6d61);
    let tokens: usize = Q_LENS.iter().sum();
    let slots = PAGE_INDICES.len() * PAGE;
    let (iq_raw, iq) = rng.row(tokens * IDX_HEADS * IDX_DIM);
    let (ik_raw, ik) = rng.row(slots * IDX_DIM);
    let (q_raw, q) = rng.row(tokens * Q_HEADS * HEAD_DIM);
    let (k_raw, k) = rng.row(slots * KV_HEADS * HEAD_DIM);
    let (v_raw, v) = rng.row(slots * KV_HEADS * HEAD_DIM);
    let mut qo_indptr = vec![0i32];
    for n in Q_LENS {
        qo_indptr.push(qo_indptr.last().unwrap() + n as i32);
    }
    #[allow(clippy::cast_precision_loss)]
    let sm_scale = 1.0 / (HEAD_DIM as f32).sqrt();

    let tables = [
        gpu.up(&PAGE_INDICES),
        gpu.up(&PAGE_INDPTR),
        gpu.up(&LAST_PAGE_LENS),
    ];
    let indptr = gpu.up(&qo_indptr);
    let ragged = |data: u64, width: usize| RaggedTensor {
        data: Tensor::new(data, tokens as u32, width as u32, Dtype::Bf16),
        indptr: Tensor::new(indptr, qo_indptr.len() as u32, 1, Dtype::I32),
    };
    let ik_at = gpu.up(&ik_raw);
    let keys = pool(ik_at, ik_at, IDX_DIM, tables);
    let iq_at = gpu.up(&iq_raw);
    let sel_at = gpu.zeros(tokens * IDX_HEADS * TOP_K * 4);
    let mut selection = Tensor::new(
        sel_at,
        tokens as u32,
        (IDX_HEADS * TOP_K) as u32,
        Dtype::I32,
    );
    let ctx = gpu.ctx();
    index::topk_block_max(
        &ctx,
        ragged(iq_at, IDX_HEADS * IDX_DIM),
        &keys,
        IDX_HEADS as u32,
        IDX_DIM as u32,
        TOP_K as u32,
        BLOCK as u32,
        &mut selection,
    )
    .expect("attention.index_topk ranks by block max");
    gpu.sync();
    let got: Vec<i32> = gpu.down(sel_at, tokens * IDX_HEADS * TOP_K);
    let want = ranking(&iq, &ik);
    for (n, (g, w)) in got.chunks(TOP_K).zip(want.chunks(TOP_K)).enumerate() {
        assert_eq!(
            g,
            w,
            "row {} head {}: the ranking",
            n / IDX_HEADS,
            n % IDX_HEADS
        );
    }

    let reference = attention(&q, &k, &v, &want, sm_scale);
    let (k_at, v_at) = (gpu.up(&k_raw), gpu.up(&v_raw));
    let kv = pool(k_at, v_at, KV_HEADS * HEAD_DIM, tables);
    let q_at = gpu.up(&q_raw);
    for (mma, decode) in [(false, false), (false, true), (true, false), (true, true)] {
        let out = gpu.zeros(tokens * Q_HEADS * HEAD_DIM * 2);
        let mut o = Tensor::new(out, tokens as u32, (Q_HEADS * HEAD_DIM) as u32, Dtype::Bf16);
        let ctx = gpu.ctx().with_tuning(Tuning {
            selected_mma: mma,
            ..Tuning::DEFAULT
        });
        let q = ragged(q_at, Q_HEADS * HEAD_DIM);
        if decode {
            selected::attention_decode_selected(
                &ctx,
                q,
                selection,
                &kv,
                None,
                HEAD_DIM as u32,
                sm_scale,
                BLOCK as u32,
                true,
                &mut o,
            )
        } else {
            selected::attention_prefill_selected(
                &ctx,
                q,
                selection,
                &kv,
                None,
                HEAD_DIM as u32,
                KV_HEADS as u32,
                sm_scale,
                BLOCK as u32,
                true,
                &mut o,
            )
        }
        .expect("the selected attention reads each kv head's blocks");
        gpu.sync();
        let landed: Vec<u16> = gpu.down(out, tokens * Q_HEADS * HEAD_DIM);
        for (i, (&g, &w)) in landed.iter().zip(&reference).enumerate() {
            let g = from_bf16(g);
            assert!(
                close(g, w),
                "mma {mma} decode {decode}: row {} head {} dim {} landed {g} against {w}",
                i / (Q_HEADS * HEAD_DIM),
                (i / HEAD_DIM) % Q_HEADS,
                i % HEAD_DIM
            );
        }
    }
}
