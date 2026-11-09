#![cfg(feature = "cuda")]

// Qwen3.8-Flash-Next's selected attention: 24 query heads over 2 kv heads of
// 256, one selection a row shared by every head, in blocks of 4 keys, with
// the open block past the last whole one. The tensor-core walk
// (`kernel.sdpa.selected_mma`), and its decode over a cluster of blocks
// (`kernel.sdpa.selected_cluster`), land what the scalar walk lands, over two
// prefills whose page tables are out of order, rows that select fewer blocks
// than the budget, rows that name a block twice or none, and a decode.

use crate::common::{Gpu, Lcg, close, from_bf16};
use dtype::Dtype;
use kernels_cuda::attn::selected;
use kernels_cuda::tensor::{KvPool, RaggedTensor, Tensor};
use kernels_cuda::tuning::Tuning;

const KV_HEADS: usize = 2;
const Q_HEADS: usize = 24;
const HEAD_DIM: usize = 256;
const RATIO: usize = 4;
const TOP_K: usize = 6;
const PAGE: usize = 16;

const KV_LENS: [usize; 2] = [45, 77];
const PAGE_INDICES: [i32; 8] = [5, 1, 7, 0, 3, 6, 2, 4];
const PAGE_INDPTR: [i32; 3] = [0, 3, 8];
const LAST_PAGE_LENS: [i32; 2] = [13, 13];

fn pool(keys: u64, values: u64, tables: [u64; 3]) -> KvPool {
    let width = KV_HEADS * HEAD_DIM;
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
        head_stride: HEAD_DIM as i64,
        layout: 0,
        scheme_byte: 0,
        block_size: 0,
        max_pages_per_request: 5,
        pages_in_batch: PAGE_INDICES.len() as i32,
        live: [0, 0],
        page_stride: [0; 2],
    }
    .with_packed_stride()
}

#[test]
fn the_selected_attention_lands_alike_on_the_tensor_cores() {
    // Two prefills (the row-tile walk), then one row each (the row walk).
    for q_lens in [[45, 30], [1, 1]] {
        lands_alike(q_lens);
    }
}

fn lands_alike(q_lens: [usize; 2]) {
    let mut gpu = Gpu::open();
    let mut rng = Lcg::seeded(0x5e1e_c7ed);
    let tokens: usize = q_lens.iter().sum();
    let slots = PAGE_INDICES.len() * PAGE;
    let (q_raw, _) = rng.row(tokens * Q_HEADS * HEAD_DIM);
    let (k_raw, _) = rng.row(slots * KV_HEADS * HEAD_DIM);
    let (v_raw, _) = rng.row(slots * KV_HEADS * HEAD_DIM);
    let mut qo_indptr = vec![0i32];
    for n in q_lens {
        qo_indptr.push(qo_indptr.last().unwrap() + n as i32);
    }
    // Each row's blocks, each at most once: some it may see, some past it,
    // and the empty slots -1.
    let mut selection: Vec<i32> = Vec::with_capacity(tokens * TOP_K);
    let mut t = 0;
    for r in 0..q_lens.len() {
        for i in 0..q_lens[r] {
            let seen = (KV_LENS[r] - q_lens[r] + i + 1) / RATIO;
            let mut row: Vec<i32> = Vec::with_capacity(TOP_K);
            for k in 0..TOP_K {
                let pick = match (t + k) % 7 {
                    0 => -1,
                    1 => (seen + 2 + k) as i32,
                    _ => {
                        let mut c = (t * 5 + k * 3) % seen.max(1);
                        let mut tries = 0;
                        while row.contains(&(c as i32)) && tries < seen {
                            c = (c + 1) % seen.max(1);
                            tries += 1;
                        }
                        if row.contains(&(c as i32)) {
                            -1
                        } else {
                            c as i32
                        }
                    }
                };
                row.push(if pick >= 0 && row.contains(&pick) {
                    -1
                } else {
                    pick
                });
            }
            selection.extend(row);
            t += 1;
        }
    }
    #[allow(clippy::cast_precision_loss)]
    let sm_scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let tables = [
        gpu.up(&PAGE_INDICES),
        gpu.up(&PAGE_INDPTR),
        gpu.up(&LAST_PAGE_LENS),
    ];
    let indptr = gpu.up(&qo_indptr);
    let kv = pool(gpu.up(&k_raw), gpu.up(&v_raw), tables);
    let q = RaggedTensor {
        data: Tensor::new(
            gpu.up(&q_raw),
            tokens as u32,
            (Q_HEADS * HEAD_DIM) as u32,
            Dtype::Bf16,
        ),
        indptr: Tensor::new(indptr, qo_indptr.len() as u32, 1, Dtype::I32),
    };
    let sel = Tensor::new(gpu.up(&selection), tokens as u32, TOP_K as u32, Dtype::I32);
    for decode in [false, true] {
        let mut landed = Vec::new();
        for (mma, cluster) in [(false, 0), (true, 0), (true, 8)] {
            let out = gpu.zeros(tokens * Q_HEADS * HEAD_DIM * 2);
            let mut o = Tensor::new(out, tokens as u32, (Q_HEADS * HEAD_DIM) as u32, Dtype::Bf16);
            let ctx = gpu.ctx().with_tuning(Tuning {
                selected_mma: mma,
                selected_cluster: cluster,
                ..Tuning::DEFAULT
            });
            if decode {
                selected::attention_decode_selected(
                    &ctx,
                    q,
                    sel,
                    &kv,
                    None,
                    HEAD_DIM as u32,
                    sm_scale,
                    RATIO as u32,
                    false,
                    &mut o,
                )
            } else {
                selected::attention_prefill_selected(
                    &ctx,
                    q,
                    sel,
                    &kv,
                    None,
                    HEAD_DIM as u32,
                    KV_HEADS as u32,
                    sm_scale,
                    RATIO as u32,
                    false,
                    &mut o,
                )
            }
            .expect("the selected attention fires");
            gpu.sync();
            landed.push(gpu.down::<u16>(out, tokens * Q_HEADS * HEAD_DIM));
        }
        for (walk, mma) in landed.iter().enumerate().skip(1) {
            for (i, (&s, &m)) in landed[0].iter().zip(mma).enumerate() {
                let (s, m) = (from_bf16(s), from_bf16(m));
                assert!(
                    close(m, s),
                    "{q_lens:?} decode {decode} walk {walk}: row {} head {} dim {}: the tensor cores land {m}, the scalar walk {s}",
                    i / (Q_HEADS * HEAD_DIM),
                    (i / HEAD_DIM) % Q_HEADS,
                    i % HEAD_DIM
                );
            }
        }
    }
}

// MiniMax-M3's decode: 16 query heads a kv head of 128, a selection a kv
// head (`per_head`) of 128-key blocks, one row over 1100 keys, its pages out
// of order: a cluster of eight blocks walks keys in every block, and its fold
// lands what the walk on one block and the scalar walk land.
#[test]
fn a_clusters_slices_fold_into_what_one_block_walks() {
    const KVH: usize = 2;
    const G: usize = 16;
    const D: usize = 128;
    const BLOCK: usize = 128;
    const TOP: usize = 8;
    const PAGE: usize = 16;
    const LEN: usize = 1100;
    let pages = LEN.div_ceil(PAGE);
    let mut gpu = Gpu::open();
    let mut rng = Lcg::seeded(0xc1_05e7);
    let (q_raw, _) = rng.row(KVH * G * D);
    let (k_raw, _) = rng.row(pages * PAGE * KVH * D);
    let (v_raw, _) = rng.row(pages * PAGE * KVH * D);
    let page_indices: Vec<i32> = (0..pages).map(|i| ((i * 7) % pages) as i32).collect();
    // The first kv head picks the eight whole blocks, the second six of them
    // and two empty slots.
    let selection: Vec<i32> = [7, 3, 0, 5, 1, 6, 2, 4, 6, -1, 0, 4, -1, 2, 7, 1].to_vec();
    let tables = [
        gpu.up(&page_indices),
        gpu.up(&[0, pages as i32]),
        gpu.up(&[(LEN - (pages - 1) * PAGE) as i32]),
    ];
    let width = (KVH * D) as u32;
    let slots = (pages * PAGE) as u32;
    let kv = KvPool {
        keys: Tensor::new(gpu.up(&k_raw), slots, width, Dtype::Bf16),
        values: Tensor::new(gpu.up(&v_raw), slots, width, Dtype::Bf16),
        page_indices: Tensor::new(tables[0], pages as u32, 1, Dtype::I32),
        page_indptr: Tensor::new(tables[1], 2, 1, Dtype::I32),
        last_page_lens: Tensor::new(tables[2], 1, 1, Dtype::I32),
        page_size: PAGE as i32,
        seq_stride: i64::from(width),
        head_stride: D as i64,
        max_pages_per_request: pages as i32,
        pages_in_batch: pages as i32,
        ..pool(0, 0, [0; 3])
    }
    .with_packed_stride();
    let q = RaggedTensor {
        data: Tensor::new(gpu.up(&q_raw), 1, (KVH * G * D) as u32, Dtype::Bf16),
        indptr: Tensor::new(gpu.up(&[0i32, 1]), 2, 1, Dtype::I32),
    };
    let sel = Tensor::new(gpu.up(&selection), 1, (KVH * TOP) as u32, Dtype::I32);
    #[allow(clippy::cast_precision_loss)]
    let sm_scale = 1.0 / (D as f32).sqrt();
    let mut landed = Vec::new();
    for (mma, cluster) in [(false, 0), (true, 0), (true, 8)] {
        let out = gpu.zeros(KVH * G * D * 2);
        let mut o = Tensor::new(out, 1, (KVH * G * D) as u32, Dtype::Bf16);
        let ctx = gpu.ctx().with_tuning(Tuning {
            selected_mma: mma,
            selected_cluster: cluster,
            ..Tuning::DEFAULT
        });
        selected::attention_decode_selected(
            &ctx,
            q,
            sel,
            &kv,
            None,
            D as u32,
            sm_scale,
            BLOCK as u32,
            true,
            &mut o,
        )
        .expect("the selected attention fires");
        gpu.sync();
        landed.push(gpu.down::<u16>(out, KVH * G * D));
    }
    for (walk, got) in landed.iter().enumerate().skip(1) {
        for (i, (&s, &m)) in landed[0].iter().zip(got).enumerate() {
            let (s, m) = (from_bf16(s), from_bf16(m));
            assert!(
                close(m, s),
                "walk {walk}: head {} dim {}: the tensor cores land {m}, the scalar walk {s}",
                i / D,
                i % D
            );
        }
    }
}
