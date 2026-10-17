#![cfg(feature = "cuda")]

//! The sparse-attention indexers against the host's ranking: DeepSeek-V4's
//! `index::topk` (blocks scored by `sum_h w_h relu(q_h . k)` at their last
//! key) on each scorer, its query rotations, and MiniMax-M3's block-max
//! ranking with the selected attention it feeds.

use crate::common::attn_ref::{
    Quant, YARN, attend, col, near, pool, quantized, ten, tuned, turn, wide,
};
use crate::common::{Gpu, Lcg, from_bf16, to_bf16};
use dtype::Dtype::{Bf16, I32, U8};
use kernels_cuda::attn::index::{self, FP4_ROW, QueryTurn};
use kernels_cuda::attn::selected;
use kernels_cuda::tensor::{KvPool, RaggedTensor};
use kernels_cuda::tuning::Tuning;

const DIM: usize = 128;

/// Requests of `(rows, keys)`, the rows the last of the keys, on `page`-cell
/// pages (back to front where `page` is 16): the pool's tables, each row's
/// `(request, position)`, and a request's slot of a position.
fn paging(
    gpu: &mut Gpu,
    lens: &[(usize, usize)],
    page: usize,
) -> (
    KvPool,
    Vec<(usize, usize)>,
    impl Fn(usize, usize) -> usize + use<>,
) {
    let per: Vec<usize> = lens.iter().map(|l| l.1.div_ceil(page)).collect();
    let total: usize = per.iter().sum();
    let ids: Vec<i32> = (0..total)
        .map(|i| if page == 16 { total - 1 - i } else { i } as i32)
        .collect();
    let mut indptr = vec![0i32];
    per.iter()
        .for_each(|p| indptr.push(indptr[indptr.len() - 1] + *p as i32));
    let last: Vec<i32> = lens.iter().map(|l| ((l.1 - 1) % page + 1) as i32).collect();
    let (pages, starts) = (col(gpu, &ids, I32), col(gpu, &indptr, I32));
    let p = KvPool {
        last_page_lens: col(gpu, &last, I32),
        ..pool(ten(0, total * page, DIM, Bf16), pages, starts, page)
    };
    let owner = lens
        .iter()
        .enumerate()
        .flat_map(|(r, &(n, kv))| (0..n).map(move |i| (r, kv - n + i)));
    (p, owner.collect(), move |r, pos| {
        ids[indptr[r] as usize + pos / page] as usize * page + pos % page
    })
}

/// One ranking at `heads` index heads over FP4 keys or bf16 ones, the
/// queries' tail turned as the scorer loads them where `turned`.
fn rank(
    heads: usize,
    lens: &[(usize, usize)],
    page: usize,
    fp4: bool,
    turned: bool,
    top_k: usize,
    tuning: Tuning,
) {
    let mut gpu = Gpu::open();
    let (keys, owner, slot) = paging(&mut gpu, lens, page);
    let (slots, rows) = (keys.keys.rows as usize, owner.len());
    let mut lcg = Lcg::seeded(0x1d3e + heads as u64);
    let ((q_raw, mut q), (k_raw, mut k)) = (lcg.row(rows * heads * DIM), lcg.row(slots * DIM));
    let w: Vec<u16> = (0..rows * heads)
        .map(|_| to_bf16(lcg.unit().abs() + 0.25))
        .collect();
    let keys = if fp4 {
        // DeepSeek-V4.1's FP4 keys at each block's last key, read as its quantizer reads them.
        let (mut src, mut bpos, mut breq) = (vec![], vec![], vec![]);
        for (r, &(_, kv)) in lens.iter().enumerate() {
            for p in (3..kv).step_by(4) {
                let s = slot(r, p) * DIM;
                src.extend_from_slice(&k_raw[s..s + DIM]);
                let read = quantized(&k[s..s + DIM], Quant::Fp4);
                (
                    k[s..s + DIM].copy_from_slice(&read),
                    bpos.push(p as i32),
                    breq.push(r as i32),
                );
            }
        }
        q = quantized(&q, Quant::Fp4);
        let plane = ten(
            gpu.zeros(slots * FP4_ROW as usize),
            slots,
            FP4_ROW as usize,
            U8,
        );
        let keys = KvPool {
            keys: plane,
            values: plane,
            seq_stride: i64::from(FP4_ROW),
            ..keys
        };
        let (src, bpos, breq) = (
            ten(gpu.up(&src), bpos.len(), DIM, Bf16),
            col(&mut gpu, &bpos, I32),
            col(&mut gpu, &breq, I32),
        );
        index::store_fp4(&gpu.ctx(), src, bpos, breq, &keys).expect("the FP4 keys land");
        keys
    } else {
        let plane = ten(gpu.up(&k_raw), slots, DIM, Bf16);
        KvPool {
            keys: plane,
            values: plane,
            ..keys
        }
    };
    for (i, head) in q.chunks_mut(DIM).enumerate().filter(|_| turned) {
        turn(head, owner[i / heads].1 as i32, 64, true, false, Some(YARN));
    }
    let mut qo = vec![0i32];
    lens.iter()
        .for_each(|l| qo.push(qo[qo.len() - 1] + l.0 as i32));
    let qt = RaggedTensor {
        data: ten(gpu.up(&q_raw), rows, heads * DIM, Bf16),
        indptr: col(&mut gpu, &qo, I32),
    };
    let weights = Some(ten(gpu.up(&w), rows, heads, Bf16));
    let positions = col(
        &mut gpu,
        &owner.iter().map(|o| o.1 as i32).collect::<Vec<_>>(),
        I32,
    );
    let sel_at = gpu.zeros(rows * top_k * 4);
    let (ctx, mut sel) = (gpu.ctx().with_tuning(tuning), ten(sel_at, rows, top_k, I32));
    let (h, d, kk, ratio) = (heads as u32, DIM as u32, top_k as u32, 4);
    if turned {
        let turn = QueryTurn {
            positions,
            dim: 64,
            theta: 1e4,
            yarn: Some(YARN),
        };
        let landed = index::topk_turned(&ctx, qt, weights, &keys, h, d, kk, ratio, turn, &mut sel);
        assert!(
            landed.expect("the turned ranking fires"),
            "the scorer turns its queries"
        );
    } else {
        index::topk(&ctx, qt, weights, &keys, h, d, kk, ratio, &mut sel)
            .expect("the ranking fires");
    }
    gpu.sync();
    let (got, mut differ): (Vec<i32>, usize) = (gpu.down(sel_at, rows * top_k), 0);
    for (t, &(r, p)) in owner.iter().enumerate() {
        let key = |j: usize| &k[slot(r, j * 4 + 3) * DIM..][..DIM];
        let dot = |hh: usize, j: usize| {
            q[(t * heads + hh) * DIM..][..DIM]
                .iter()
                .zip(key(j))
                .map(|(a, b)| a * b)
                .sum::<f32>()
        };
        let score = |j: usize| {
            (0..heads)
                .map(|hh| dot(hh, j).max(0.0) * from_bf16(w[t * heads + hh]))
                .sum::<f32>()
        };
        let mut order: Vec<usize> = (0..(p + 1) / 4).collect();
        order.sort_by(|&a, &b| score(b).total_cmp(&score(a)));
        let mut want: Vec<i32> = order.iter().take(top_k).map(|&j| j as i32).collect();
        (want.sort_unstable(), want.resize(top_k, -1));
        let row = &got[t * top_k..(t + 1) * top_k];
        assert!(
            row.windows(2).all(|p| p[0] < p[1] || p[1] == -1),
            "row {t}: the picks ascend"
        );
        differ += row.iter().filter(|c| !want.contains(c)).count();
    }
    // The tcgen05 scorer's sums tie to f32 rounding now and then.
    assert!(
        differ * 2000 <= rows * top_k,
        "{heads} heads, fp4 {fp4}, turned {turned}: {differ} picks off"
    );
}

#[test]
fn the_pooled_index_selects_its_best_blocks() {
    let (small, big) = (&[(26, 26), (40, 90)][..], &[(300, 600), (300, 700)][..]);
    let t =
        |mma, radix, tc5| tuned(|t| (t.index_mma, t.index_radix, t.index_tc5) = (mma, radix, tc5));
    // The scalar scorer, the tensor cores' one, two and four tiles, radix on and off.
    rank(64, small, 16, false, false, 8, t(false, false, false));
    rank(64, small, 16, false, false, 8, t(false, true, false));
    rank(4, small, 16, false, false, 8, t(true, false, false));
    rank(20, small, 16, false, false, 8, t(true, true, false));
    // V4.1's FP4 keys on both scorers; the queries turned as they load.
    rank(64, small, 16, true, false, 8, t(true, true, false));
    rank(4, small, 16, true, false, 8, t(false, false, false));
    rank(64, small, 16, false, true, 8, t(true, true, false));
    // A prefill's rows on tcgen05, pages back to front and 64-wide, turned and not.
    rank(64, big, 16, false, false, 128, t(true, true, true));
    rank(64, big, 64, false, true, 128, t(true, true, true));
}

/// The indexer's rotations (`index::rope`, the key's `layernorm_rope`) on
/// both forms: the first 64 dims turned interleaved, the key layer-normed first.
#[test]
fn the_indexers_rotations_answer_the_host() {
    let (rows, heads, mut lcg) = (5usize, 32usize, Lcg::seeded(0x1d0e));
    let ((q_raw, q), (k_raw, k), (w_raw, w), (b_raw, b)) = (
        lcg.row(rows * heads * DIM),
        lcg.row(rows * DIM),
        lcg.row(DIM),
        lcg.row(DIM),
    );
    let pos: Vec<i32> = (0..rows as i32).map(|r| r * 977 + 3).collect();
    let (mut want_q, mut want_k) = (q.clone(), k.clone());
    for (i, head) in want_q.chunks_mut(DIM).enumerate() {
        turn(&mut head[..64], pos[i / heads], 64, true, false, None);
    }
    for (r, row) in want_k.chunks_mut(DIM).enumerate() {
        let mean = row.iter().sum::<f32>() / DIM as f32;
        let inv =
            1.0 / (row.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / DIM as f32 + 1e-6).sqrt();
        row.iter_mut()
            .enumerate()
            .for_each(|(i, x)| *x = (*x - mean) * inv * w[i] + b[i]);
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
        let (mut qt, mut kt) = (
            ten(gpu.up(&q_raw), rows, heads * DIM, Bf16),
            ten(gpu.up(&k_raw), rows, DIM, Bf16),
        );
        index::rope(&ctx, &mut qt, pos_t, heads as u32, DIM as u32, 64, 1e4)
            .expect("the query turns");
        index::layernorm_rope(&ctx, &mut kt, pos_t, w_t, b_t, 1e-6, 64, 1e4)
            .expect("the key turns");
        gpu.sync();
        near(
            &wide(&gpu.down::<u16>(qt.ptr, rows * heads * DIM)),
            &want_q,
            2e-2,
            1.0,
            "the query",
        );
        near(
            &wide(&gpu.down::<u16>(kt.ptr, rows * DIM)),
            &want_k,
            2e-2,
            1.0,
            "the key",
        );
    }
}

/// MiniMax-M3: each head keeps the row's own block and the best `top_k - 1`
/// others by their best key, and that kv head's queries attend over its
/// blocks alone (`selected`'s prefill and decode, on both walks).
#[test]
fn the_block_max_ranking_answers_the_reference() {
    let (ih, qh, kvh, hd, block, top_k) = (4usize, 8usize, 4usize, 128usize, 16usize, 3usize);
    let mut gpu = Gpu::open();
    let (tables, owner, slot) = paging(&mut gpu, &[(70, 70), (2, 90)], 16);
    let (slots, rows, mut lcg) = (tables.keys.rows as usize, owner.len(), Lcg::seeded(0x4d33));
    let ((iq_raw, iq), (ik_raw, ik)) = (lcg.row(rows * ih * DIM), lcg.row(slots * DIM));
    let ((q_raw, q), (k_raw, k), (v_raw, v)) = (
        lcg.row(rows * qh * hd),
        lcg.row(slots * kvh * hd),
        lcg.row(slots * kvh * hd),
    );
    let mut want = Vec::new();
    for (t, &(r, p)) in owner.iter().enumerate() {
        for h in 0..ih {
            let dot = |j: usize| {
                iq[(t * ih + h) * DIM..][..DIM]
                    .iter()
                    .zip(&ik[slot(r, j) * DIM..][..DIM])
                    .map(|(a, b)| a * b)
                    .sum::<f32>()
            };
            let best = |b: usize| {
                (b * block..((b + 1) * block).min(p + 1))
                    .map(dot)
                    .fold(f32::NEG_INFINITY, f32::max)
            };
            let mut order: Vec<usize> = (0..p / block).collect();
            order.sort_by(|&a, &b| best(b).total_cmp(&best(a)).then(a.cmp(&b)));
            let mut run: Vec<i32> = order
                .iter()
                .take(top_k - 1)
                .map(|&b| b as i32)
                .chain([(p / block) as i32])
                .collect();
            (run.sort_unstable(), run.resize(top_k, -1));
            want.extend(run);
        }
    }
    let qo = col(&mut gpu, &[0i32, 70, 72], I32);
    let ikeys = ten(gpu.up(&ik_raw), slots, DIM, Bf16);
    let (ikeys, sel_at) = (
        KvPool {
            keys: ikeys,
            values: ikeys,
            ..tables
        },
        gpu.zeros(rows * ih * top_k * 4),
    );
    let (iq_t, mut sel) = (
        RaggedTensor {
            data: ten(gpu.up(&iq_raw), rows, ih * DIM, Bf16),
            indptr: qo,
        },
        ten(sel_at, rows, ih * top_k, I32),
    );
    let (ih32, dim, k32, b32) = (ih as u32, DIM as u32, top_k as u32, block as u32);
    index::topk_block_max(&gpu.ctx(), iq_t, &ikeys, ih32, dim, k32, b32, &mut sel)
        .expect("the ranking fires");
    gpu.sync();
    assert!(
        gpu.down::<i32>(sel_at, rows * ih * top_k) == want,
        "the block-max ranking"
    );
    let scale = 1.0 / (hd as f32).sqrt();
    let (want_o, _) = attend([&q, &k, &v], [qh, kvh, hd, hd], scale, None, |t, h| {
        let (run, (r, p)) = (
            &want[(t * kvh + h / (qh / kvh)) * top_k..][..top_k],
            owner[t],
        );
        (0..=p)
            .filter(|j| run.contains(&((j / block) as i32)))
            .map(|j| slot(r, j))
            .collect()
    });
    let (keys, values) = (
        ten(gpu.up(&k_raw), slots, kvh * hd, Bf16),
        ten(gpu.up(&v_raw), slots, kvh * hd, Bf16),
    );
    let kv = KvPool {
        keys,
        values,
        seq_stride: (kvh * hd) as i64,
        head_stride: hd as i64,
        ..tables
    };
    let q_t = RaggedTensor {
        data: ten(gpu.up(&q_raw), rows, qh * hd, Bf16),
        indptr: qo,
    };
    for (mma, decode) in [(false, false), (false, true), (true, false), (true, true)] {
        let (o_at, ctx) = (
            gpu.zeros(rows * qh * hd * 2),
            gpu.ctx().with_tuning(tuned(|t| t.selected_mma = mma)),
        );
        let (mut o, h32, kvh32) = (ten(o_at, rows, qh * hd, Bf16), hd as u32, kvh as u32);
        let fired = match decode {
            true => selected::attention_decode_selected(
                &ctx, q_t, sel, &kv, None, h32, scale, b32, true, &mut o,
            ),
            false => selected::attention_prefill_selected(
                &ctx, q_t, sel, &kv, None, h32, kvh32, scale, b32, true, &mut o,
            ),
        };
        fired.expect("the selected attention fires");
        gpu.sync();
        near(
            &wide(&gpu.down::<u16>(o_at, rows * qh * hd)),
            &want_o,
            3e-2,
            1.0,
            &format!("mma {mma}, decode {decode}"),
        );
    }
}
