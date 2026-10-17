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

/// The nearest e2m1 code of `v` (within ±6), ties to the even code and the
/// sign kept, as the hardware's `cvt.rn` rounds.
fn e2m1_code(v: f32) -> u8 {
    const MAG: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    let a = v.abs();
    let mut best = 0u8;
    for c in 1..8u8 {
        let (d, db) = ((MAG[c as usize] - a).abs(), (MAG[best as usize] - a).abs());
        if d < db || (d == db && c % 2 == 0) {
            best = c;
        }
    }
    if v.is_sign_negative() { best | 8 } else { best }
}

fn e2m1(code: u8) -> f32 {
    const MAG: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    let v = MAG[usize::from(code & 7)];
    if code & 8 != 0 { -v } else { v }
}

/// DeepSeek-V4.1's `fp4_act_quant(x, 32, ue8m0)` of one 32-block: its codes
/// and biased exponent.
fn fp4_block(block: &[f32]) -> (Vec<u8>, u8) {
    let amax = block
        .iter()
        .fold(0f32, |m, v| m.max(v.abs()))
        .max(6.0 * 2f32.powi(-126));
    let bits = (amax * (1.0 / 6.0)).to_bits();
    let e = ((bits >> 23) & 0xff) as i32 + i32::from(bits & 0x7f_ffff != 0);
    let s = 2f32.powi(e - 127);
    let codes = block
        .iter()
        .map(|v| e2m1_code((v / s).clamp(-6.0, 6.0)))
        .collect();
    (codes, e as u8)
}

/// `x` read at FP4, a 32-block at a time.
fn fp4_read(x: &[f32]) -> Vec<f32> {
    x.chunks(32)
        .flat_map(|b| {
            let (codes, e) = fp4_block(b);
            let s = 2f32.powi(i32::from(e) - 127);
            codes.into_iter().map(move |c| e2m1(c) * s)
        })
        .collect()
}

/// DeepSeek-V4.1's FP4 index keys (68 bytes a position: e2m1 pairs, then
/// ue8m0 exponents, a page's data rows before its scale rows): written at
/// each block's boundary bit for bit as its reference quantizes them, and
/// ranked against its queries read at FP4 as the reference ranks them, on
/// the tensor cores as they are and widened to bf16 for the scalar scorer.
#[test]
fn fp4_index_keys_rank_as_deepseeks_reference_reads_them() {
    for heads in [4, 64] {
        check_fp4(heads);
    }
}

fn check_fp4(heads: usize) {
    let mut gpu = Gpu::open();
    let mut rng = Lcg::seeded(0xf4f4 + heads as u64);
    let tokens: usize = Q_LENS.iter().sum();
    let slots = PAGE_INDICES.len() * PAGE;
    let (q_raw, q) = rng.row(tokens * heads * DIM);
    let w: Vec<f32> = (0..tokens * heads)
        .map(|_| from_bf16(to_bf16(rng.unit().abs() + 0.25)))
        .collect();
    let w_raw: Vec<u16> = w.iter().map(|&x| to_bf16(x)).collect();
    // One key a block, at its boundary, in each request.
    let mut bpos = Vec::new();
    let mut breq = Vec::new();
    for (r, &len) in KV_LENS.iter().enumerate() {
        for j in 0..len / RATIO {
            bpos.push(((j + 1) * RATIO - 1) as i32);
            breq.push(r as i32);
        }
    }
    let n = bpos.len();
    let (e_raw, e) = rng.row(n * DIM);
    let mut qo_indptr = vec![0i32];
    for len in Q_LENS {
        qo_indptr.push(qo_indptr.last().unwrap() + len as i32);
    }
    let tables = [
        gpu.up(&PAGE_INDICES),
        gpu.up(&PAGE_INDPTR),
        gpu.up(&LAST_PAGE_LENS),
    ];
    let row = index::FP4_ROW as usize;
    let plane = gpu.zeros(slots * row);
    let keys = KvPool {
        keys: Tensor::new(plane, slots as u32, index::FP4_ROW, Dtype::U8),
        values: Tensor::new(plane, slots as u32, index::FP4_ROW, Dtype::U8),
        seq_stride: i64::from(index::FP4_ROW),
        ..pool(plane, tables)
    };
    let ctx = gpu.ctx();
    index::store_fp4(
        &ctx,
        Tensor::new(gpu.up(&e_raw), n as u32, DIM as u32, Dtype::Bf16),
        Tensor::new(gpu.up(&bpos), n as u32, 1, Dtype::I32),
        Tensor::new(gpu.up(&breq), n as u32, 1, Dtype::I32),
        &keys,
    )
    .expect("the FP4 keys land");
    gpu.sync();
    let raw: Vec<u8> = gpu.down(plane, slots * row);
    // The bytes, and what each request's keys read as.
    let mut k = vec![0f32; slots * DIM];
    for c in 0..n {
        let s = slot(breq[c] as usize, bpos[c] as usize);
        let (page, off) = (s / PAGE, s % PAGE);
        let base = page * PAGE * row;
        for b in 0..DIM / 32 {
            let (codes, ex) = fp4_block(&e[c * DIM + b * 32..c * DIM + (b + 1) * 32]);
            assert_eq!(
                raw[base + PAGE * 64 + off * 4 + b],
                ex,
                "key {c} block {b}'s exponent"
            );
            for i in 0..16 {
                let byte = raw[base + off * 64 + b * 16 + i];
                assert_eq!(
                    (byte & 0xf, byte >> 4),
                    (codes[2 * i], codes[2 * i + 1]),
                    "key {c} block {b} pair {i}"
                );
            }
        }
        k[s * DIM..(s + 1) * DIM].copy_from_slice(&fp4_read(&e[c * DIM..(c + 1) * DIM]));
    }
    let q4 = fp4_read(&q);
    let indptr = gpu.up(&qo_indptr);
    let q_at = gpu.up(&q_raw);
    let w_at = gpu.up(&w_raw);
    for mma in [true, false] {
        let sel_at = gpu.zeros(tokens * TOP_K * 4);
        let mut selection = Tensor::new(sel_at, tokens as u32, TOP_K as u32, Dtype::I32);
        let ctx = gpu.ctx().with_tuning(Tuning {
            index_mma: mma,
            ..Tuning::DEFAULT
        });
        index::topk(
            &ctx,
            RaggedTensor {
                data: Tensor::new(q_at, tokens as u32, (heads * DIM) as u32, Dtype::Bf16),
                indptr: Tensor::new(indptr, qo_indptr.len() as u32, 1, Dtype::I32),
            },
            Some(Tensor::new(w_at, tokens as u32, heads as u32, Dtype::Bf16)),
            &keys,
            heads as u32,
            DIM as u32,
            TOP_K as u32,
            RATIO as u32,
            &mut selection,
        )
        .expect("attention.index_topk ranks the FP4 keys");
        gpu.sync();
        let got: Vec<i32> = gpu.down(sel_at, tokens * TOP_K);
        let mut t = 0;
        let mut ranked = 0;
        for r in 0..Q_LENS.len() {
            for i in 0..Q_LENS[r] {
                let seen = (KV_LENS[r] - Q_LENS[r] + i + 1) / RATIO;
                if seen > TOP_K {
                    let score = |j: usize| -> f32 {
                        let at = slot(r, (j + 1) * RATIO - 1) * DIM;
                        (0..heads)
                            .map(|h| {
                                let qh = &q4[(t * heads + h) * DIM..][..DIM];
                                let dot: f32 =
                                    qh.iter().zip(&k[at..at + DIM]).map(|(a, b)| a * b).sum();
                                dot.max(0.0) * w[t * heads + h]
                            })
                            .sum()
                    };
                    let mut order: Vec<usize> = (0..seen).collect();
                    order.sort_by(|&a, &b| score(b).total_cmp(&score(a)));
                    let mut want: Vec<i32> = order[..TOP_K].iter().map(|&j| j as i32).collect();
                    want.sort_unstable();
                    assert_eq!(
                        &got[t * TOP_K..(t + 1) * TOP_K],
                        &want[..],
                        "{heads} heads, mma {mma}: row {t}: the best {TOP_K} of {seen} FP4 blocks"
                    );
                    ranked += 1;
                }
                t += 1;
            }
        }
        assert!(ranked > 0, "rows were ranked");
    }
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
/// DeepSeek-V4's indexer turns its queries' tail just before ranking: the
/// tensor-core scorer turning them as it loads them selects what turning
/// them first, then ranking, selects.
#[test]
fn the_pooled_index_turns_its_queries_as_it_loads_them() {
    const HEADS: usize = 64;
    let mut gpu = Gpu::open();
    let mut rng = Lcg::seeded(0x7e11_0a2b);
    let tokens: usize = Q_LENS.iter().sum();
    let slots = PAGE_INDICES.len() * PAGE;
    let (q_raw, _) = rng.row(tokens * HEADS * DIM);
    let (k_raw, _) = rng.row(slots * DIM);
    let w_raw: Vec<u16> = (0..tokens * HEADS)
        .map(|_| to_bf16(rng.unit().abs() + 0.25))
        .collect();
    let mut qo_indptr = vec![0i32];
    for n in Q_LENS {
        qo_indptr.push(qo_indptr.last().unwrap() + n as i32);
    }
    let positions: Vec<i32> = (0..Q_LENS.len())
        .flat_map(|r| (0..Q_LENS[r]).map(move |i| (KV_LENS[r] - Q_LENS[r] + i) as i32))
        .collect();
    let tables = [
        gpu.up(&PAGE_INDICES),
        gpu.up(&PAGE_INDPTR),
        gpu.up(&LAST_PAGE_LENS),
    ];
    let indptr = gpu.up(&qo_indptr);
    let keys = pool(gpu.up(&k_raw), tables);
    let pos_at = gpu.up(&positions);
    let w_at = gpu.up(&w_raw);
    let ctx = gpu.ctx().with_tuning(Tuning {
        index_mma: true,
        index_radix: true,
        ..Tuning::DEFAULT
    });
    let yarn = Some(kernels_cuda::elemwise::rope::Yarn {
        factor: 16.0,
        beta_fast: 32.0,
        beta_slow: 1.0,
        original_max_position: 65536,
    });
    let q_of = |at: u64| RaggedTensor {
        data: Tensor::new(at, tokens as u32, (HEADS * DIM) as u32, Dtype::Bf16),
        indptr: Tensor::new(indptr, qo_indptr.len() as u32, 1, Dtype::I32),
    };
    let weights = Some(Tensor::new(w_at, tokens as u32, HEADS as u32, Dtype::Bf16));
    let pos_t = Tensor::new(pos_at, tokens as u32, 1, Dtype::I32);

    // Turned first, then ranked.
    let q_first = gpu.up(&q_raw);
    let mut q_t = Tensor::new(q_first, tokens as u32, (HEADS * DIM) as u32, Dtype::Bf16);
    kernels_cuda::elemwise::rope::partial_last(
        &ctx, &mut q_t, pos_t, 64, DIM as u32, 1e4, true, false, yarn,
    )
    .expect("the turn fires");
    let first_at = gpu.zeros(tokens * TOP_K * 4);
    index::topk(
        &ctx,
        q_of(q_first),
        weights,
        &keys,
        HEADS as u32,
        DIM as u32,
        TOP_K as u32,
        RATIO as u32,
        &mut Tensor::new(first_at, tokens as u32, TOP_K as u32, Dtype::I32),
    )
    .expect("the ranking fires");

    // Turned as the ranking loads them.
    let q_raw_at = gpu.up(&q_raw);
    let loaded_at = gpu.zeros(tokens * TOP_K * 4);
    let landed = index::topk_turned(
        &ctx,
        q_of(q_raw_at),
        weights,
        &keys,
        HEADS as u32,
        DIM as u32,
        TOP_K as u32,
        RATIO as u32,
        index::QueryTurn {
            positions: pos_t,
            dim: 64,
            theta: 1e4,
            yarn,
        },
        &mut Tensor::new(loaded_at, tokens as u32, TOP_K as u32, Dtype::I32),
    )
    .expect("the turned ranking fires");
    assert!(landed, "the tensor-core scorer turns the queries");
    gpu.sync();
    let first: Vec<i32> = gpu.down(first_at, tokens * TOP_K);
    let loaded: Vec<i32> = gpu.down(loaded_at, tokens * TOP_K);
    assert_eq!(
        loaded, first,
        "turning as it loads selects what turning first selects"
    );
}

/// The indexer's rotations a pair a thread (`kernel.elemwise.rows_vec`)
/// land what a head or a row a thread lands: the query's rotation bit for
/// bit, the key's layer norm within its moments' summation order.
#[test]
fn the_indexers_rotations_land_the_same_a_pair_a_thread() {
    const ROWS: usize = 5;
    const HEADS: usize = 32;
    let mut rng = Lcg::seeded(0x1d0e);
    let (q_raw, _) = rng.row(ROWS * HEADS * DIM);
    let (k_raw, _) = rng.row(ROWS * DIM);
    let (w_raw, _) = rng.row(DIM);
    let (b_raw, _) = rng.row(DIM);
    let positions: Vec<i32> = (0..ROWS as i32).map(|r| r * 977 + 3).collect();
    let mut gpu = Gpu::open();
    let pos_t = Tensor::new(gpu.up(&positions), ROWS as u32, 1, Dtype::I32);
    let w_t = Tensor::new(gpu.up(&w_raw), DIM as u32, 1, Dtype::Bf16);
    let b_t = Tensor::new(gpu.up(&b_raw), DIM as u32, 1, Dtype::Bf16);
    let mut out = Vec::new();
    for vec in [false, true] {
        let ctx = gpu.ctx().with_tuning(Tuning {
            rows_vec: vec,
            ..Tuning::DEFAULT
        });
        let q_at = gpu.up(&q_raw);
        let k_at = gpu.up(&k_raw);
        let mut q = Tensor::new(q_at, ROWS as u32, (HEADS * DIM) as u32, Dtype::Bf16);
        let mut k = Tensor::new(k_at, ROWS as u32, DIM as u32, Dtype::Bf16);
        index::rope(&ctx, &mut q, pos_t, HEADS as u32, DIM as u32, 64, 10000.0)
            .expect("the query rotation fires");
        index::layernorm_rope(&ctx, &mut k, pos_t, w_t, b_t, 1e-6, 64, 10000.0)
            .expect("the key's norm and rotation fire");
        gpu.sync();
        let q_out: Vec<u16> = gpu.down(q_at, ROWS * HEADS * DIM);
        let k_out: Vec<u16> = gpu.down(k_at, ROWS * DIM);
        out.push((q_out, k_out));
    }
    assert_eq!(out[0].0, out[1].0, "the query rotation lands bit for bit");
    for (i, (&a, &b)) in out[0].1.iter().zip(&out[1].1).enumerate() {
        let (a, b) = (from_bf16(a), from_bf16(b));
        assert!(
            (a - b).abs() <= b.abs() / 128.0 + 1e-3,
            "key element {i}: {a} a warp a row against {b} a block a row"
        );
    }
}

/// A prefill's ranking on tcgen05 (`kernel.sdpa.index_tc5`) selects what the
/// mma.sync scorer selects: two requests' rows (the second's run starting
/// mid-pair, one row alone at the end), their tails turned or not, the
/// selections alike but for blocks whose scores tie to f32 rounding.
#[test]
fn a_prefills_ranking_on_tcgen05_selects_what_the_mma_scorer_selects() {
    // DeepSeek-V4-Flash's 16-position pages (four blocks a page) and 64's.
    // Pages out of order, and in order (eight a box).
    check_tc5(16, true);
    check_tc5(16, false);
    check_tc5(64, true);
}

#[allow(non_snake_case)]
fn check_tc5(PG: usize, shuffled: bool) {
    const H: usize = 64;
    const K: usize = 128;
    let spans = [1531usize, 2048];
    let rows_of = [767usize, 1024];
    let rows: usize = rows_of.iter().sum();
    let pages_of: Vec<usize> = spans.iter().map(|s| s.div_ceil(PG)).collect();
    let total_pages: usize = pages_of.iter().sum();
    let mut gpu = Gpu::open();
    let mut rng = Lcg::seeded(0x7c5);
    let (q_raw, _) = rng.row(rows * H * DIM);
    let (k_raw, _) = rng.row(total_pages * PG * DIM);
    let w_raw: Vec<u16> = (0..rows * H)
        .map(|_| to_bf16(rng.unit().abs() + 0.25))
        .collect();
    // Pages out of order (request r's page i near pool page 7 i + 3 r), or
    // in order.
    let mut page_indices = Vec::new();
    let mut taken = vec![false; total_pages];
    for (r, &n) in pages_of.iter().enumerate() {
        for i in 0..n {
            let mut at = if shuffled {
                (i * 7 + r * 3) % total_pages
            } else {
                page_indices.len()
            };
            while taken[at] {
                at = (at + 1) % total_pages;
            }
            taken[at] = true;
            page_indices.push(at as i32);
        }
    }
    let indptr_pages = vec![0i32, pages_of[0] as i32, total_pages as i32];
    let last: Vec<i32> = spans.iter().map(|s| ((s - 1) % PG + 1) as i32).collect();
    let k_at = gpu.up(&k_raw);
    let keys = KvPool {
        keys: Tensor::new(k_at, (total_pages * PG) as u32, DIM as u32, Dtype::Bf16),
        values: Tensor::new(k_at, (total_pages * PG) as u32, DIM as u32, Dtype::Bf16),
        page_indices: Tensor::new(gpu.up(&page_indices), total_pages as u32, 1, Dtype::I32),
        page_indptr: Tensor::new(gpu.up(&indptr_pages), 3, 1, Dtype::I32),
        last_page_lens: Tensor::new(gpu.up(&last), 2, 1, Dtype::I32),
        page_size: PG as i32,
        max_pages_per_request: pages_of.iter().copied().max().unwrap() as i32,
        pages_in_batch: total_pages as i32,
        ..pool(k_at, [0, 0, 0])
    };
    let qo = vec![0i32, rows_of[0] as i32, rows as i32];
    let positions: Vec<i32> = (0..rows)
        .map(|t| {
            if t < rows_of[0] {
                (spans[0] - rows_of[0] + t) as i32
            } else {
                (spans[1] - rows_of[1] + t - rows_of[0]) as i32
            }
        })
        .collect();
    let q_at = gpu.up(&q_raw);
    let w_at = gpu.up(&w_raw);
    let qo_at = gpu.up(&qo);
    let pos_t = Tensor::new(gpu.up(&positions), rows as u32, 1, Dtype::I32);
    for turned in [false, true] {
        let mut picks = Vec::new();
        for tc5 in [false, true] {
            let ctx = gpu.ctx().with_tuning(Tuning {
                index_mma: true,
                index_radix: true,
                index_tc5: tc5,
                ..Tuning::DEFAULT
            });
            let sel_at = gpu.zeros(rows * K * 4);
            let mut selection = Tensor::new(sel_at, rows as u32, K as u32, Dtype::I32);
            let q = RaggedTensor {
                data: Tensor::new(q_at, rows as u32, (H * DIM) as u32, Dtype::Bf16),
                indptr: Tensor::new(qo_at, 3, 1, Dtype::I32),
            };
            let w = Some(Tensor::new(w_at, rows as u32, H as u32, Dtype::Bf16));
            if turned {
                let turn = index::QueryTurn {
                    positions: pos_t,
                    dim: 64,
                    theta: 1e4,
                    yarn: None,
                };
                assert!(
                    index::topk_turned(
                        &ctx,
                        q,
                        w,
                        &keys,
                        H as u32,
                        DIM as u32,
                        K as u32,
                        RATIO as u32,
                        turn,
                        &mut selection
                    )
                    .expect("the turned ranking fires"),
                    "the scorer turns its queries"
                );
            } else {
                index::topk(
                    &ctx,
                    q,
                    w,
                    &keys,
                    H as u32,
                    DIM as u32,
                    K as u32,
                    RATIO as u32,
                    &mut selection,
                )
                .expect("the ranking fires");
            }
            gpu.sync();
            picks.push(gpu.down::<i32>(sel_at, rows * K));
        }
        let mut differ = 0usize;
        for t in 0..rows {
            let a: std::collections::BTreeSet<i32> =
                picks[0][t * K..(t + 1) * K].iter().copied().collect();
            let b: std::collections::BTreeSet<i32> =
                picks[1][t * K..(t + 1) * K].iter().copied().collect();
            differ += a.symmetric_difference(&b).count();
        }
        assert!(
            differ * 2000 <= rows * K,
            "pages {PG}, shuffled {shuffled}, turned {turned}: {differ} of {} picks differ between the scorers",
            rows * K
        );
    }
}
