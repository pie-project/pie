#![cfg(feature = "cuda")]

//! The sparse-attention indexers against the host's ranking: DeepSeek-V4's
//! `index::topk` (blocks scored by `sum_h w_h relu(q_h . k)` at their last
//! key) on each scorer, its query rotations, and MiniMax-M3's block-max
//! ranking with the selected attention it feeds.

use crate::common::attn_ref::{Quant, YARN, attend, col, dot, keyed, landed, offsets};
use crate::common::attn_ref::{owners, pool, quantized, ten, tuned, turn};
use crate::common::{Gpu, Lcg, from_bf16, to_bf16};
use dtype::Dtype::{Bf16, I32, U8};
use kernels_cuda::attn::index::{self, FP4_ROW, QueryTurn};
use kernels_cuda::attn::selected;
use kernels_cuda::tensor::{KvPool, RaggedTensor};
use kernels_cuda::tuning::Tuning;

mod logits_fp4;
mod select;

const DIM: usize = 128;

type Lens<'a> = &'a [(usize, usize)];

/// The pool tables of requests of `(rows, keys)` on `page`-cell pages
/// (back to front where `page` is 16), with each request's page ids.
fn paging(gpu: &mut Gpu, lens: Lens, page: usize) -> (KvPool, Vec<Vec<usize>>) {
    let per: Vec<usize> = lens.iter().map(|l| l.1.div_ceil(page)).collect();
    let total: usize = per.iter().sum();
    let mut ids: Vec<usize> = (0..total).collect();
    if page == 16 {
        ids.reverse();
    }
    let (flat, starts): (Vec<i32>, _) = (ids.iter().map(|&i| i as i32).collect(), offsets(per));
    let (pages, indptr) = (col(gpu, &flat, I32), col(gpu, &starts, I32));
    let last: Vec<i32> = lens.iter().map(|l| ((l.1 - 1) % page + 1) as i32).collect();
    let plane = pool(ten(0, total * page, DIM, Bf16), pages, indptr, page);
    let p = KvPool {
        last_page_lens: col(gpu, &last, I32),
        ..plane
    };
    let run = |r: usize| ids[starts[r] as usize..starts[r + 1] as usize].to_vec();
    (p, (0..lens.len()).map(run).collect())
}

/// One ranking at `heads` index heads over FP4 keys or bf16 ones, the
/// queries' tail turned as the scorer loads them where `turned`.
fn rank(
    heads: usize,
    lens: Lens,
    page: usize,
    [fp4, turned]: [bool; 2],
    top_k: usize,
    tuning: Tuning,
) {
    let mut gpu = Gpu::open();
    let (keys, runs) = paging(&mut gpu, lens, page);
    let slot = |r: usize, p: usize| runs[r][p / page] * page + p % page;
    let (slots, owner) = (keys.keys.rows as usize, owners(lens));
    let rows = owner.len();
    let mut lcg = Lcg::seeded(0x1d3e + heads as u64);
    let (q_raw, mut q) = lcg.row(rows * heads * DIM);
    let (k_raw, mut k) = lcg.row(slots * DIM);
    let w: Vec<u16> = (0..rows * heads)
        .map(|_| to_bf16(lcg.unit().abs() + 0.25))
        .collect();
    let keys = if fp4 {
        // V4.1's FP4 keys at each block's last key, read as its quantizer reads them.
        let (mut src, mut bpos, mut breq) = (vec![], vec![], vec![]);
        for &(r, p) in owners(&lens.iter().map(|l| (l.1, l.1)).collect::<Vec<_>>()).iter() {
            if p % 4 == 3 {
                let s = slot(r, p) * DIM;
                src.extend_from_slice(&k_raw[s..s + DIM]);
                let read = quantized(&k[s..s + DIM], Quant::Fp4);
                k[s..s + DIM].copy_from_slice(&read);
                bpos.push(p as i32);
                breq.push(r as i32);
            }
        }
        q = quantized(&q, Quant::Fp4);
        let plane = ten(gpu.zeros(slots * 68), slots, FP4_ROW as usize, U8);
        let keys = keyed(&keys, plane, plane);
        let src = ten(gpu.up(&src), bpos.len(), DIM, Bf16);
        let (bpos, breq) = (col(&mut gpu, &bpos, I32), col(&mut gpu, &breq, I32));
        let stored = index::store_fp4(&gpu.ctx(), src, bpos, breq, &keys);
        stored.expect("the FP4 keys land");
        keys
    } else {
        let plane = ten(gpu.up(&k_raw), slots, DIM, Bf16);
        keyed(&keys, plane, plane)
    };
    for (i, head) in q.chunks_mut(DIM).enumerate().filter(|_| turned) {
        turn(head, owner[i / heads].1 as i32, 64, true, false, Some(YARN));
    }
    let qo = col(&mut gpu, &offsets(lens.iter().map(|l| l.0)), I32);
    let qt = RaggedTensor {
        data: ten(gpu.up(&q_raw), rows, heads * DIM, Bf16),
        indptr: qo,
    };
    let weights = Some(ten(gpu.up(&w), rows, heads, Bf16));
    let positions: Vec<i32> = owner.iter().map(|o| o.1 as i32).collect();
    let positions = col(&mut gpu, &positions, I32);
    let sel_at = gpu.zeros(rows * top_k * 4);
    let (ctx, sel) = (
        gpu.ctx().with_tuning(tuning),
        &mut ten(sel_at, rows, top_k, I32),
    );
    let (h, d, kk, yarn) = (heads as u32, DIM as u32, top_k as u32, Some(YARN));
    if turned {
        let turn = QueryTurn {
            positions,
            dim: 64,
            theta: 1e4,
            yarn,
        };
        let fired = index::topk_turned(&ctx, qt, weights, &keys, h, d, kk, 4, turn, sel);
        assert!(fired.expect("the turned ranking fires"), "the scorer turns");
    } else {
        let fired = index::topk(&ctx, qt, weights, &keys, h, d, kk, 4, sel);
        fired.expect("the ranking fires");
    }
    gpu.sync();
    let (got, mut differ): (Vec<i32>, usize) = (gpu.down(sel_at, rows * top_k), 0);
    for (t, &(r, p)) in owner.iter().enumerate() {
        let key = |j: usize| &k[slot(r, j * 4 + 3) * DIM..][..DIM];
        let head = |hh: usize| &q[(t * heads + hh) * DIM..][..DIM];
        let weight = |hh: usize| from_bf16(w[t * heads + hh]);
        let relu = |hh: usize, j: usize| dot(head(hh), key(j)).max(0.0) * weight(hh);
        let score = |j: usize| (0..heads).map(|hh| relu(hh, j)).sum::<f32>();
        let mut order: Vec<usize> = (0..(p + 1) / 4).collect();
        order.sort_by(|&a, &b| score(b).total_cmp(&score(a)));
        let mut want: Vec<i32> = order.iter().take(top_k).map(|&j| j as i32).collect();
        want.sort_unstable();
        want.resize(top_k, -1);
        let row = &got[t * top_k..(t + 1) * top_k];
        let ascends = row.windows(2).all(|p| p[0] < p[1] || p[1] == -1);
        assert!(ascends, "row {t}: the picks ascend");
        differ += row.iter().filter(|c| !want.contains(c)).count();
    }
    // The tcgen05 scorer's sums tie to f32 rounding now and then.
    let what = format!("{heads} heads, fp4 {fp4}, turned {turned}");
    assert!(differ * 2000 <= rows * top_k, "{what}: {differ} picks off");
}

#[test]
fn the_pooled_index_selects_its_best_blocks() {
    let (small, big): (Lens, Lens) = (&[(26, 26), (40, 90)], &[(300, 600), (300, 700)]);
    let t =
        |mma, radix, tc5| tuned(|t| (t.index_mma, t.index_radix, t.index_tc5) = (mma, radix, tc5));
    let (bf, fp4, turned) = ([false; 2], [true, false], [false, true]);
    #[rustfmt::skip]
    let cases = [
        // The scalar scorer, the tensor cores' one, two and four tiles, radix on and off.
        (64, small, 16, bf, 8, t(false, false, false)),
        (64, small, 16, bf, 8, t(false, true, false)),
        (4, small, 16, bf, 8, t(true, false, false)),
        (20, small, 16, bf, 8, t(true, true, false)),
        // V4.1's FP4 keys on both scorers; the queries turned as they load.
        (64, small, 16, fp4, 8, t(true, true, false)),
        (4, small, 16, fp4, 8, t(false, false, false)),
        // The block-scaled scorer asked for, over keys a position's row at
        // each block's last, which it does not read: the old scorer ranks.
        (64, small, 16, fp4, 8, tuned(|t| (t.index_mma, t.index_radix, t.index_fp4_tc5) = (true, true, true))),
        (64, small, 16, turned, 8, t(true, true, false)),
        // A prefill's rows on tcgen05, pages back to front and 64-wide, turned and not.
        (64, big, 16, bf, 128, t(true, true, true)),
        (64, big, 64, turned, 128, t(true, true, true)),
        // GLM-5.3's 32 heads, four rows a pass, a pass's rows split between requests.
        (32, &[(301, 600), (299, 700)], 64, turned, 128, t(true, true, true)),
    ];
    for (heads, lens, page, keys, top_k, tuning) in cases {
        rank(heads, lens, page, keys, top_k, tuning);
    }
}

/// The indexer's rotations (`index::rope`, the key's `layernorm_rope`) on
/// both forms: the first 64 dims turned interleaved, the key layer-normed first.
#[test]
fn the_indexers_rotations_answer_the_host() {
    let (rows, heads, mut lcg) = (5usize, 32usize, Lcg::seeded(0x1d0e));
    let (q_raw, mut want_q) = lcg.row(rows * heads * DIM);
    let (k_raw, mut want_k) = lcg.row(rows * DIM);
    let ((w_raw, w), (b_raw, b)) = (lcg.row(DIM), lcg.row(DIM));
    let pos: Vec<i32> = (0..rows as i32).map(|r| r * 977 + 3).collect();
    for (i, head) in want_q.chunks_mut(DIM).enumerate() {
        turn(&mut head[..64], pos[i / heads], 64, true, false, None);
    }
    for (r, row) in want_k.chunks_mut(DIM).enumerate() {
        let mean = row.iter().sum::<f32>() / DIM as f32;
        let var = row.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / DIM as f32;
        for (i, x) in row.iter_mut().enumerate() {
            *x = (*x - mean) / (var + 1e-6).sqrt() * w[i] + b[i];
        }
        turn(&mut row[..64], pos[r], 64, true, false, None);
    }
    let mut gpu = Gpu::open();
    let (pos_t, w_t, b_t) = (
        col(&mut gpu, &pos, I32),
        col(&mut gpu, &w_raw, Bf16),
        col(&mut gpu, &b_raw, Bf16),
    );
    for vec in [false, true] {
        let ctx = gpu.ctx().with_tuning(tuned(|t| t.rows_vec = vec));
        let mut qt = ten(gpu.up(&q_raw), rows, heads * DIM, Bf16);
        let mut kt = ten(gpu.up(&k_raw), rows, DIM, Bf16);
        let h = heads as u32;
        index::rope(&ctx, &mut qt, pos_t, h, 128, 64, 1e4).expect("the query turns");
        let fired = index::layernorm_rope(&ctx, &mut kt, pos_t, w_t, b_t, 1e-6, 64, 1e4);
        fired.expect("the key turns");
        landed(&gpu, qt.ptr, &want_q, 2e-2, 1.0, "the query");
        landed(&gpu, kt.ptr, &want_k, 2e-2, 1.0, "the key");
    }
}

/// MiniMax-M3: each head keeps the row's own block and the best `top_k - 1`
/// others by their best key, and that kv head's queries attend over its
/// blocks alone (`selected`'s prefill and decode, on both walks).
#[test]
fn the_block_max_ranking_answers_the_reference() {
    let (ih, qh, kvh, hd, block, top_k) = (4usize, 8usize, 4usize, 128usize, 16usize, 3usize);
    let (mut gpu, lens): (Gpu, Lens) = (Gpu::open(), &[(70, 70), (2, 90)]);
    let (tables, runs) = paging(&mut gpu, lens, 16);
    let slot = |r: usize, p: usize| runs[r][p / 16] * 16 + p % 16;
    let (slots, owner, mut lcg) = (tables.keys.rows as usize, owners(lens), Lcg::seeded(0x4d33));
    let rows = owner.len();
    let (iq_raw, iq) = lcg.row(rows * ih * DIM);
    let (ik_raw, ik) = lcg.row(slots * DIM);
    let (q_raw, q) = lcg.row(rows * qh * hd);
    let (k_raw, k) = lcg.row(slots * kvh * hd);
    let (v_raw, v) = lcg.row(slots * kvh * hd);
    let mut want = Vec::new();
    for (t, &(r, p)) in owner.iter().enumerate() {
        for h in 0..ih {
            let key = |j: usize| {
                dot(
                    &iq[(t * ih + h) * DIM..][..DIM],
                    &ik[slot(r, j) * DIM..][..DIM],
                )
            };
            let best = |b: usize| {
                (b * block..((b + 1) * block).min(p + 1))
                    .map(key)
                    .fold(f32::MIN, f32::max)
            };
            let mut order: Vec<usize> = (0..p / block).collect();
            order.sort_by(|&a, &b| best(b).total_cmp(&best(a)).then(a.cmp(&b)));
            let mut run: Vec<i32> = order.iter().take(top_k - 1).map(|&b| b as i32).collect();
            run.push((p / block) as i32);
            run.sort_unstable();
            run.resize(top_k, -1);
            want.extend(run);
        }
    }
    let qo = col(&mut gpu, &[0i32, 70, 72], I32);
    let ikeys = ten(gpu.up(&ik_raw), slots, DIM, Bf16);
    let (ikeys, sel_at) = (
        keyed(&tables, ikeys, ikeys),
        gpu.zeros(rows * ih * top_k * 4),
    );
    let iq_t = RaggedTensor {
        data: ten(gpu.up(&iq_raw), rows, ih * DIM, Bf16),
        indptr: qo,
    };
    let sel = &mut ten(sel_at, rows, ih * top_k, I32);
    let (ih32, k32, b32) = (ih as u32, top_k as u32, block as u32);
    let fired = index::topk_block_max(&gpu.ctx(), iq_t, &ikeys, ih32, 128, k32, b32, sel);
    fired.expect("the ranking fires");
    gpu.sync();
    assert!(
        gpu.down::<i32>(sel_at, rows * ih * top_k) == want,
        "the ranking"
    );
    let scale = 1.0 / (hd as f32).sqrt();
    let (want_o, _) = attend([&q, &k, &v], [qh, kvh, hd, hd], scale, None, |t, h| {
        let (run, (r, p)) = (&want[(t * kvh + h / 2) * top_k..][..top_k], owner[t]);
        let blocks = (0..=p).filter(|j| run.contains(&((j / block) as i32)));
        blocks.map(|j| slot(r, j)).collect()
    });
    let (keys, values) = (
        ten(gpu.up(&k_raw), slots, kvh * hd, Bf16),
        ten(gpu.up(&v_raw), slots, kvh * hd, Bf16),
    );
    let kv = KvPool {
        head_stride: hd as i64,
        ..keyed(&tables, keys, values)
    };
    let q_t = RaggedTensor {
        data: ten(gpu.up(&q_raw), rows, qh * hd, Bf16),
        indptr: qo,
    };
    for (mma, decode) in [(false, false), (false, true), (true, false), (true, true)] {
        let o_at = gpu.zeros(rows * qh * hd * 2);
        let ctx = gpu.ctx().with_tuning(tuned(|t| t.selected_mma = mma));
        let (o, sel, kvh) = (&mut ten(o_at, rows, qh * hd, Bf16), *sel, kvh as u32);
        let fired = match decode {
            true => selected::attention_decode_selected(
                &ctx, q_t, sel, &kv, None, 128, scale, b32, true, o,
            ),
            false => selected::attention_prefill_selected(
                &ctx, q_t, sel, &kv, None, 128, kvh, scale, b32, true, o,
            ),
        };
        fired.expect("the selected attention fires");
        landed(
            &gpu,
            o_at,
            &want_o,
            3e-2,
            1.0,
            &format!("mma {mma}, decode {decode}"),
        );
    }
}
