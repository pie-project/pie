#![cfg(feature = "cuda")]

// Qwen3.8-Flash-Next's indexer: a row scores each whole block of `RATIO` keys
// it sees by the block's last key, `sum_h w_h · relu(q_h · k)`, and selects
// the `TOP_K` best, ascending; a row that sees no more than `TOP_K` blocks
// selects them all. Read against a host reference over a fresh prefill and a
// continuation whose page tables are out of order, scored the scalar way
// and on the tensor cores (`kernel.sdpa.index_mma`), at 4, 20 and 64 heads.

use crate::common::{Gpu, Lcg, from_bf16, to_bf16};
use dtype::Dtype;
use kernels_cuda::attn::index;
use kernels_cuda::tensor::{KvPool, RaggedTensor, Tensor};
use kernels_cuda::tuning::Tuning;

const DIM: usize = 128;
const RATIO: usize = 4;
const TOP_K: usize = 8;
const PAGE: usize = 16;

const Q_LENS: [usize; 2] = [26, 40];
const KV_LENS: [usize; 2] = [26, 90];
const PAGE_INDICES: [i32; 8] = [5, 1, 7, 0, 3, 6, 2, 4];
const PAGE_INDPTR: [i32; 3] = [0, 2, 8];
const LAST_PAGE_LENS: [i32; 2] = [10, 10];

fn slot(request: usize, position: usize) -> usize {
    let page = PAGE_INDICES[PAGE_INDPTR[request] as usize + position / PAGE] as usize;
    page * PAGE + position % PAGE
}

fn pool(keys: u64, tables: [u64; 3]) -> KvPool {
    let slots = (PAGE_INDICES.len() * PAGE) as u32;
    KvPool {
        keys: Tensor::new(keys, slots, DIM as u32, Dtype::Bf16),
        values: Tensor::new(keys, slots, DIM as u32, Dtype::Bf16),
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
        seq_stride: DIM as i64,
        head_stride: DIM as i64,
        layout: 0,
        scheme_byte: 0,
        block_size: 0,
        max_pages_per_request: 6,
        pages_in_batch: PAGE_INDICES.len() as i32,
    }
}

#[test]
fn the_pooled_index_selects_its_best_blocks() {
    // Flash-Next's four heads, and the tensor cores' two- and four-tile
    // forms (DeepSeek-V4's indexer is 64 heads wide).
    check(4);
    check(20);
    check(64);
}

#[allow(non_snake_case)]
fn check(HEADS: usize) {
    let mut gpu = Gpu::open();
    let mut rng = Lcg::seeded(0x1d3e_c5e1);
    let tokens: usize = Q_LENS.iter().sum();
    let slots = PAGE_INDICES.len() * PAGE;
    let (q_raw, q) = rng.row(tokens * HEADS * DIM);
    let (k_raw, k) = rng.row(slots * DIM);
    let w: Vec<f32> = (0..tokens * HEADS)
        .map(|_| from_bf16(to_bf16(rng.unit().abs() + 0.25)))
        .collect();
    let w_raw: Vec<u16> = w.iter().map(|&x| to_bf16(x)).collect();
    let mut qo_indptr = vec![0i32];
    for n in Q_LENS {
        qo_indptr.push(qo_indptr.last().unwrap() + n as i32);
    }
    let tables = [
        gpu.up(&PAGE_INDICES),
        gpu.up(&PAGE_INDPTR),
        gpu.up(&LAST_PAGE_LENS),
    ];
    let indptr = gpu.up(&qo_indptr);
    let keys = pool(gpu.up(&k_raw), tables);
    let q_at = gpu.up(&q_raw);
    let w_at = gpu.up(&w_raw);
    for (mma, radix) in [(false, false), (true, false), (false, true), (true, true)] {
        let sel_at = gpu.zeros(tokens * TOP_K * 4);
        let mut selection = Tensor::new(sel_at, tokens as u32, TOP_K as u32, Dtype::I32);
        let ctx = gpu.ctx().with_tuning(Tuning {
            index_mma: mma,
            index_radix: radix,
            ..Tuning::DEFAULT
        });
        index::topk(
            &ctx,
            RaggedTensor {
                data: Tensor::new(q_at, tokens as u32, (HEADS * DIM) as u32, Dtype::Bf16),
                indptr: Tensor::new(indptr, qo_indptr.len() as u32, 1, Dtype::I32),
            },
            Some(Tensor::new(w_at, tokens as u32, HEADS as u32, Dtype::Bf16)),
            &keys,
            HEADS as u32,
            DIM as u32,
            TOP_K as u32,
            RATIO as u32,
            &mut selection,
        )
        .expect("attention.index_topk ranks the pooled blocks");
        gpu.sync();
        let got: Vec<i32> = gpu.down(sel_at, tokens * TOP_K);

        let mut t = 0;
        let (mut ranked, mut whole) = (0, 0);
        for r in 0..Q_LENS.len() {
            for i in 0..Q_LENS[r] {
                let seen = (KV_LENS[r] - Q_LENS[r] + i + 1) / RATIO;
                let row = &got[t * TOP_K..(t + 1) * TOP_K];
                if seen <= TOP_K {
                    let want: Vec<i32> = (0..TOP_K as i32)
                        .map(|n| if (n as usize) < seen { n } else { -1 })
                        .collect();
                    assert_eq!(
                        row,
                        &want[..],
                        "mma {mma}: row {t} sees {seen} blocks and takes them all"
                    );
                    whole += 1;
                } else {
                    let score = |j: usize| -> f32 {
                        let at = slot(r, (j + 1) * RATIO - 1) * DIM;
                        (0..HEADS)
                            .map(|h| {
                                let qh = &q[(t * HEADS + h) * DIM..][..DIM];
                                let dot: f32 =
                                    qh.iter().zip(&k[at..at + DIM]).map(|(a, b)| a * b).sum();
                                dot.max(0.0) * w[t * HEADS + h]
                            })
                            .sum()
                    };
                    let mut order: Vec<usize> = (0..seen).collect();
                    order.sort_by(|&a, &b| score(b).total_cmp(&score(a)));
                    let mut want: Vec<i32> = order[..TOP_K].iter().map(|&j| j as i32).collect();
                    want.sort_unstable();
                    assert_eq!(
                        row,
                        &want[..],
                        "mma {mma}: row {t}: the best {TOP_K} of {seen} blocks, ascending"
                    );
                    ranked += 1;
                }
                t += 1;
            }
        }
        assert!(ranked > 0 && whole > 0, "both kinds of row were read");
    }
}

/// DeepSeek-V4-Flash's indexer at a prefill's width: 2048 rows of one
/// 4096-position request, 64 heads of 128, ratio 4, the best 512 of up to
/// 1024 blocks a row.
#[test]
#[ignore = "a measurement; run it by name under the GPU lock"]
fn the_pooled_index_paces_a_prefill() {
    const H: usize = 64;
    const K: usize = 512;
    const SPAN: usize = 4096;
    const ROWS: usize = 2048;
    const PG: usize = 64;
    let pages = SPAN / PG;
    let mut gpu = Gpu::open();
    let mut rng = Lcg::seeded(0x5a11);
    let (q_raw, _) = rng.row(ROWS * H * DIM);
    let (k_raw, _) = rng.row(SPAN * DIM);
    let w_raw: Vec<u16> = (0..ROWS * H)
        .map(|_| to_bf16(rng.unit().abs() + 0.25))
        .collect();
    let page_indices: Vec<i32> = (0..pages as i32).collect();
    let k_at = gpu.up(&k_raw);
    let keys = KvPool {
        keys: Tensor::new(k_at, SPAN as u32, DIM as u32, Dtype::Bf16),
        values: Tensor::new(k_at, SPAN as u32, DIM as u32, Dtype::Bf16),
        page_indices: Tensor::new(gpu.up(&page_indices), pages as u32, 1, Dtype::I32),
        page_indptr: Tensor::new(gpu.up(&[0i32, pages as i32]), 2, 1, Dtype::I32),
        last_page_lens: Tensor::new(gpu.up(&[PG as i32]), 1, 1, Dtype::I32),
        page_size: PG as i32,
        max_pages_per_request: pages as i32,
        pages_in_batch: pages as i32,
        ..pool(k_at, [0, 0, 0])
    };
    let indptr = gpu.up(&[0i32, ROWS as i32]);
    let q_at = gpu.up(&q_raw);
    let w_at = gpu.up(&w_raw);
    let sel_at = gpu.zeros(ROWS * K * 4);
    let mut selection = Tensor::new(sel_at, ROWS as u32, K as u32, Dtype::I32);
    for radix in [false, true] {
        let ctx = gpu.ctx().with_tuning(Tuning {
            index_mma: true,
            index_radix: radix,
            ..Tuning::DEFAULT
        });
        let mut fire = || {
            index::topk(
                &ctx,
                RaggedTensor {
                    data: Tensor::new(q_at, ROWS as u32, (H * DIM) as u32, Dtype::Bf16),
                    indptr: Tensor::new(indptr, 2, 1, Dtype::I32),
                },
                Some(Tensor::new(w_at, ROWS as u32, H as u32, Dtype::Bf16)),
                &keys,
                H as u32,
                DIM as u32,
                K as u32,
                RATIO as u32,
                &mut selection,
            )
            .expect("attention.index_topk ranks the pooled blocks");
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
        println!(
            "| index top-{K}, {ROWS} rows of a {SPAN}-position request, radix {radix} | {us:.0} us |"
        );
    }
}
