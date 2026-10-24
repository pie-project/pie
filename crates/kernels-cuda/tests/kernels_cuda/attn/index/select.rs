//! A scored index's ranking with a row's keys spread over a group of CTAs
//! (`kernel.sdpa.index_split_until`) against the block-a-row radix ranking
//! it stands in for (`index_select_rows`): the same picks in the same
//! order, bit for bit, over FP4 keys scored on tcgen05: ties by the
//! thousand, every score equal, zeros tied at the top above negatives,
//! +-inf and NaN scores, rows shorter than the budget, at it and just over,
//! rows of a request seeing different counts, rows a cluster of sixteen
//! CTAs each and rows a CTA each, and NaN past every row's blocks.

use crate::common::attn_ref::{col, offsets, pool, ten, tuned};
use crate::common::{Gpu, Lcg, to_bf16};
use dtype::Dtype::{Bf16, I32, U8};
use kernels_cuda::attn::index::{self, FP4_ROW};
use kernels_cuda::tensor::{KvPool, RaggedTensor, Tensor};
use kernels_cuda::tuning::Tuning;

const DIM: usize = 128;

/// What a case's keys hold.
#[derive(Clone, Copy, Debug)]
enum Keys {
    /// Random, their 32-blocks scaled 2^-4 to 2^4.
    Spread,
    /// Each one of `n` random keys: scores tied `nkeys / n` deep.
    Few(usize),
    /// One key throughout: every score of a row equal.
    Same,
}

/// What a case's head weights hold.
#[derive(Clone, Copy, Debug)]
enum Weights {
    /// Positive, as a trained indexer's mostly are.
    Plain,
    /// Of either sign, so a row's scores are negative and -0 as well.
    Signed,
    /// +-3e38: scores of +inf, -inf, NaN (inf - inf) and +-0 beside finite.
    Extreme,
    /// Zero queries: every score +0 or -0.
    Silent,
    /// Negative, every head one query: a row's scores 0 (the largest,
    /// tied by the thousand, where the query and key meet at or below 0)
    /// or negative.
    Sunk,
}

/// An indexer's fire: `heads` heads ranking `top_k` blocks of `ratio`
/// positions over requests of `(rows, positions)` on `page`-position pages,
/// the requests' pages one shared run (each request its own queries).
#[derive(Clone, Copy, Debug)]
struct Case<'a> {
    heads: usize,
    lens: &'a [(usize, usize)],
    page: usize,
    ratio: usize,
    top_k: usize,
    keys: Keys,
    weights: Weights,
}

struct Fire {
    keys: KvPool,
    q: RaggedTensor,
    w: Tensor,
    rows: usize,
    stride: usize,
}

fn build(gpu: &mut Gpu, c: Case) -> Fire {
    let per = c.page / c.ratio;
    let pages: Vec<usize> = c.lens.iter().map(|l| l.1.div_ceil(c.page)).collect();
    let most = *pages.iter().max().expect("a request");
    let longest = pages.iter().position(|&p| p == most).expect("a request");
    // Every request's table a prefix of one run of pages, back to front.
    let run: Vec<i32> = (0..most as i32).rev().collect();
    let ids: Vec<i32> = pages.iter().flat_map(|&p| run[..p].to_vec()).collect();
    let starts = offsets(pages.iter().copied());
    let plane = ten(
        gpu.zeros(most * per * FP4_ROW as usize),
        most * c.page,
        FP4_ROW as usize / c.ratio,
        U8,
    );
    let last: Vec<i32> = c
        .lens
        .iter()
        .map(|l| ((l.1 - 1) % c.page + 1) as i32)
        .collect();
    let keys = KvPool {
        last_page_lens: col(gpu, &last, I32),
        max_pages_per_request: most as i32,
        ..pool(plane, col(gpu, &ids, I32), col(gpu, &starts, I32), c.page)
    };
    let mut lcg = Lcg::seeded(0x5e1 + c.heads as u64 * 7 + c.ratio as u64);
    let spread = |lcg: &mut Lcg, salt: usize| -> Vec<u16> {
        let at = |i: usize| 2f32.powi(((i / 32 * 5 + salt) % 9) as i32 - 4);
        let x = lcg.row(DIM).1;
        x.iter()
            .enumerate()
            .map(|(i, v)| to_bf16(v * at(i)))
            .collect()
    };
    let blocks = most * c.page / c.ratio;
    let pool_of: Vec<Vec<u16>> = match c.keys {
        Keys::Spread => vec![],
        Keys::Few(n) => (0..n).map(|s| spread(&mut lcg, s)).collect(),
        Keys::Same => vec![spread(&mut lcg, 1)],
    };
    let mut src = Vec::with_capacity(blocks * DIM);
    for j in 0..blocks {
        match c.keys {
            Keys::Spread => src.extend(spread(&mut lcg, j)),
            Keys::Few(n) => src.extend_from_slice(&pool_of[(j * 7 + j / 3) % n]),
            Keys::Same => src.extend_from_slice(&pool_of[0]),
        }
    }
    let bpos: Vec<i32> = (0..blocks)
        .map(|j| (j * c.ratio + c.ratio - 1) as i32)
        .collect();
    let breq = vec![longest as i32; blocks];
    let entries = ten(gpu.up(&src), blocks, DIM, Bf16);
    let (bpos, breq) = (col(gpu, &bpos, I32), col(gpu, &breq, I32));
    index::store_fp4(&gpu.ctx(), entries, bpos, breq, &keys).expect("the FP4 keys land");
    let rows = c.lens.iter().map(|l| l.0).sum::<usize>();
    let q: Vec<u16> = match c.weights {
        Weights::Silent => vec![0; rows * c.heads * DIM],
        Weights::Sunk => (0..rows)
            .flat_map(|r| spread(&mut lcg, r).repeat(c.heads))
            .collect(),
        _ => (0..rows * c.heads)
            .flat_map(|r| spread(&mut lcg, r))
            .collect(),
    };
    let w: Vec<u16> = (0..rows * c.heads)
        .map(|i| {
            let u = lcg.unit();
            to_bf16(match c.weights {
                Weights::Plain => u.abs() + 0.25,
                Weights::Signed | Weights::Silent => u,
                Weights::Sunk => -u.abs() - 0.25,
                Weights::Extreme => match i % 4 {
                    0 => 3e38,
                    1 => -3e38,
                    _ => u,
                },
            })
        })
        .collect();
    let qo = col(gpu, &offsets(c.lens.iter().map(|l| l.0)), I32);
    Fire {
        keys,
        q: RaggedTensor {
            data: ten(gpu.up(&q), rows, c.heads * DIM, Bf16),
            indptr: qo,
        },
        w: ten(gpu.up(&w), rows, c.heads, Bf16),
        rows,
        stride: (most * c.page).div_ceil(c.ratio),
    }
}

/// B200's index decisions, the ranking spread up to `until` rows.
fn b200(until: u32) -> Tuning {
    tuned(|t| {
        (t.index_mma, t.index_radix, t.index_tc5, t.index_fp4_tc5) = (true, true, true, true);
        (t.index_split_until, t.pdl) = (until, true);
    })
}

/// The picks under `tuning`, and the scores they ranked.
fn rank(gpu: &mut Gpu, c: Case, f: &Fire, tuning: Tuning) -> (Vec<i32>, Vec<f32>) {
    let sel_at = gpu.zeros(f.rows * c.top_k * 4);
    gpu.fill(sel_at, 0x5a, f.rows * c.top_k * 4);
    let ctx = gpu.ctx().with_tuning(tuning);
    let scores = ctx
        .scratch_shared("test", "attn.index_topk_scores", f.rows * f.stride * 4)
        .expect("the scores scratch") as u64;
    // NaN where the scorer writes nothing: a ranking reading past a row's
    // blocks would take them.
    gpu.fill(scores, 0xff, f.rows * f.stride * 4);
    let (h, kk, r) = (c.heads as u32, c.top_k as u32, c.ratio as u32);
    let sel = &mut ten(sel_at, f.rows, c.top_k, I32);
    let fired = index::topk(&ctx, f.q, Some(f.w), &f.keys, h, DIM as u32, kk, r, sel);
    fired.expect("the ranking fires");
    gpu.sync();
    (
        gpu.down(sel_at, f.rows * c.top_k),
        gpu.down(scores, f.rows * f.stride),
    )
}

#[test]
fn a_spread_ranking_picks_as_the_block_a_row_one() {
    use Keys::{Few, Same, Spread};
    use Weights::{Extreme, Plain, Signed, Silent, Sunk};
    // Rows under, at and just over the budget at ratio 1 and at ratio 2.
    let ragged = [
        (1, 65_536),
        (1, 511),
        (1, 512),
        (1, 513),
        (1, 1023),
        (1, 1024),
        (1, 1026),
        (1, 30_001),
        (3, 9000),
    ];
    let wide: Vec<(usize, usize)> = (0..128).map(|i| (1, 60_000 + 37 * i)).collect();
    let tall: Vec<(usize, usize)> = (0..96)
        .map(|i| {
            (
                1,
                if i % 17 == 3 {
                    500 + i
                } else {
                    44_000 + 13 * i
                },
            )
        })
        .collect();
    #[rustfmt::skip]
    let cases = [
        // V4-Flash's indexer at decode (64 heads, ratio 4) at 16k and 64k.
        Case { heads: 64, lens: &[(1, 16_384)], page: 256, ratio: 4, top_k: 512, keys: Spread, weights: Plain },
        Case { heads: 64, lens: &[(1, 65_536), (1, 16_000)], page: 256, ratio: 4, top_k: 512, keys: Few(37), weights: Signed },
        // V4.1-Flash's (32 heads, ratio 2 and 1), its rows' counts ragged,
        // a request's three rows seeing three counts.
        Case { heads: 32, lens: &ragged, page: 256, ratio: 2, top_k: 512, keys: Spread, weights: Signed },
        Case { heads: 32, lens: &ragged, page: 256, ratio: 1, top_k: 512, keys: Few(5), weights: Plain },
        // Every score of a row tied; every score 0; zeros tied above
        // negatives; +-inf and NaN.
        Case { heads: 32, lens: &[(1, 65_536), (2, 20_000)], page: 256, ratio: 1, top_k: 512, keys: Same, weights: Plain },
        Case { heads: 64, lens: &[(1, 40_000), (1, 9000)], page: 256, ratio: 4, top_k: 512, keys: Spread, weights: Silent },
        Case { heads: 32, lens: &ragged, page: 256, ratio: 2, top_k: 512, keys: Spread, weights: Sunk },
        Case { heads: 32, lens: &ragged, page: 256, ratio: 1, top_k: 512, keys: Few(300), weights: Extreme },
        Case { heads: 64, lens: &[(1, 65_536), (1, 3000)], page: 256, ratio: 4, top_k: 512, keys: Spread, weights: Extreme },
        // A decode of 128 rows of 64k blocks (two CTAs a row, two waves);
        // 96 rows a CTA each, a row's chunk two tiles, a few rows short.
        Case { heads: 32, lens: &wide, page: 256, ratio: 1, top_k: 512, keys: Few(1000), weights: Signed },
        Case { heads: 32, lens: &tall, page: 256, ratio: 1, top_k: 512, keys: Few(700), weights: Extreme },
        // A smaller budget, small pages.
        Case { heads: 32, lens: &[(2, 12_000), (1, 700)], page: 64, ratio: 2, top_k: 64, keys: Few(11), weights: Extreme },
    ];
    for c in cases {
        let mut gpu = Gpu::open();
        let f = build(&mut gpu, c);
        let (want, want_scores) = rank(&mut gpu, c, &f, b200(0));
        let (got, got_scores) = rank(&mut gpu, c, &f, b200(u32::MAX >> 16));
        let same = |a: &[f32], b: &[f32]| a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits());
        assert!(
            same(&got_scores, &want_scores),
            "{c:?}: the scorer is not deterministic"
        );
        let (mut inf, mut nan, mut zero) = (0usize, 0usize, 0usize);
        for s in &got_scores {
            inf += usize::from(s.is_infinite());
            zero += usize::from(*s == 0.0);
        }
        for (t, row) in want.chunks(c.top_k).enumerate() {
            // A row the scorer left (no more blocks than the budget) is NaN.
            let scored = &got_scores[t * f.stride..][..f.stride];
            if scored.iter().any(|s| !s.is_nan()) {
                nan += row
                    .iter()
                    .filter(|&&j| j >= 0 && scored[j as usize].is_nan())
                    .count();
            }
            let g = &got[t * c.top_k..][..c.top_k];
            assert_eq!(g, row, "{c:?}: row {t}'s picks");
        }
        println!(
            "{} heads ratio {} {:?}/{:?}: {} rows identical ({inf} inf, {nan} NaN picked, {zero} zero scores)",
            c.heads, c.ratio, c.keys, c.weights, f.rows
        );
    }
}

/// Microseconds a fire takes under `tuning`, over `reps` fires.
fn time(gpu: &mut Gpu, c: Case, f: &Fire, tuning: Tuning, reps: u32) -> f64 {
    let sel_at = gpu.zeros(f.rows * c.top_k * 4);
    let ctx = gpu.ctx().with_tuning(tuning);
    let (h, kk, r) = (c.heads as u32, c.top_k as u32, c.ratio as u32);
    let sel = &mut ten(sel_at, f.rows, c.top_k, I32);
    let mut fire = || {
        let fired = index::topk(&ctx, f.q, Some(f.w), &f.keys, h, DIM as u32, kk, r, sel);
        fired.expect("the ranking fires");
    };
    for _ in 0..3 {
        fire();
    }
    gpu.time(reps, fire)
}

#[test]
#[ignore = "a rate bench: cargo test ... -- --ignored index_split_rate --nocapture"]
fn index_split_rate() {
    // V4-Flash's indexer (64 heads, ratio 4) and V4.1-Flash's (32 heads,
    // ratio 2 and 1), 512 blocks, at decode batches of 16k and 64k contexts.
    for (heads, ratio) in [(64, 4), (32, 2), (32, 1)] {
        for ctx in [16_384, 65_536] {
            for bs in [1, 2, 4, 8, 16, 32, 64, 128] {
                let lens: Vec<(usize, usize)> = (0..bs).map(|_| (1, ctx)).collect();
                let c = Case {
                    heads,
                    lens: &lens,
                    page: 256,
                    ratio,
                    top_k: 512,
                    keys: Keys::Spread,
                    weights: Weights::Plain,
                };
                let mut gpu = Gpu::open();
                let f = build(&mut gpu, c);
                let old = time(&mut gpu, c, &f, b200(0), 5);
                let new = time(&mut gpu, c, &f, b200(u32::MAX >> 16), 5);
                println!(
                    "{heads} heads ratio {ratio}: bs {bs} at {ctx}: {old:.1} us -> {new:.1} us ({:.2}x)",
                    old / new
                );
            }
        }
    }
}
