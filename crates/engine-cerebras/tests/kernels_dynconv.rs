//! The block draft's dynamic conv: taps that move per row, over lanes.

#![allow(clippy::too_many_arguments)]

mod common;

use common::data;
use engine_cerebras::bench::{Bench, assert_close, round_bf16};
use kernels_cerebras::attn::dynconv;
use kernels_cerebras::tensor::RaggedTensor;

fn reference(
    xs: &[f32],
    coeff: &[f32],
    base: &[f32],
    indptr: &[i32],
    c: usize,
    taps: usize,
    group: usize,
    side: usize,
) -> Vec<f32> {
    let groups = c / group;
    let rows = xs.len() / c;
    let mut y = vec![0.0f32; rows * c];
    for l in 0..indptr.len() - 1 {
        let (b, e) = (indptr[l] as usize, indptr[l + 1] as usize);
        for t in b..e {
            let j = t - b;
            for ch in 0..c {
                let mut acc = 0.0f32;
                for k in 0..taps.min(j + 1) {
                    let at = side * taps + k;
                    let coef =
                        base[at * c + ch] + coeff[t * 2 * taps * groups + at * groups + ch / group];
                    acc += coef * xs[(t - k) * c + ch];
                }
                y[t * c + ch] = round_bf16(acc);
            }
        }
    }
    y
}

/// Both sides of the projection, over two lanes: a row reads only the rows
/// of its own lane before it.
#[test]
fn a_dynamic_conv_mixes_each_rows_own_taps_within_its_lane() {
    conv_case(8, 3, 2, 11);
}

/// Channels past a PE split in whole groups; the coefficients' runs of
/// groups split with them.
#[test]
fn a_wide_dynamic_conv_splits_its_channels_over_pes() {
    conv_case(2048, 3, 4, 12);
}

fn conv_case(c: usize, taps: usize, group: usize, seed: u32) {
    let indptr = [0i32, 4, 7];
    let rows = 7usize;
    let groups = c / group;
    let xs = data(rows * c, seed);
    let coeff = data(rows * 2 * taps * groups, seed + 1);
    let base = data(2 * taps * c, seed + 2);
    let mut b = Bench::new();
    let x = b.bf16(rows as u32, c as u32, &xs);
    let ip = b.i32(indptr.len() as u32, 1, &indptr);
    let co = b.bf16(rows as u32, (2 * taps * groups) as u32, &coeff);
    let ba = b.bf16((2 * taps) as u32, c as u32, &base);
    let y0 = b.zeros(dtype::Dtype::Bf16, rows as u32, c as u32);
    let y1 = b.zeros(dtype::Dtype::Bf16, rows as u32, c as u32);
    let ran = b
        .run(|ctx| {
            let ragged = RaggedTensor {
                data: x,
                indptr: ip,
            };
            dynconv::block_dyn_conv(ctx, ragged, co, ba, 0, taps as u32, group as u32, y0)?;
            dynconv::block_dyn_conv(ctx, ragged, co, ba, 1, taps as u32, group as u32, y1)
        })
        .unwrap();
    if !ran {
        return;
    }
    for (side, y) in [(0usize, y0), (1, y1)] {
        let want = reference(&xs, &coeff, &base, &indptr, c, taps, group, side);
        assert_close(&b.read_f32(y), &want, 2e-2, 2e-2);
    }
}

/// Two lanes walked greedily: the anchor takes its first candidate, each
/// later row the best of unary plus the bilinear term under the previous
/// pick; a candidate outside the vocabulary scores by its unary alone; a
/// row outside every lane keeps its pick.
#[test]
fn a_selector_walk_picks_each_rows_best_candidate_under_the_previous_pick() {
    let (rows, k, rank, vocab) = (7usize, 3usize, 2usize, 5usize);
    let indptr = [0i32, 3, 6];
    let cands: Vec<i32> = vec![
        4, 1, 2, 0, 3, 9, 1, 4, 2, 2, 2, 0, 3, 1, 4, 0, 1, 2, 7, 7, 7,
    ];
    let unary: Vec<f32> = data(rows * k, 141).iter().map(|v| v * 0.5).collect();
    let hp = data(rows * rank, 142);
    let pred = data(vocab * rank, 143);
    let succ = data(vocab * rank, 144);
    let toks = [3i32, 0, 0, 1, 0, 0, 2];
    let mut b = Bench::new();
    let cd = b.i32(rows as u32, k as u32, &cands);
    let ip = b.i32(indptr.len() as u32, 1, &indptr);
    let un = b.f32(rows as u32, k as u32, &unary);
    let hb = b.bf16(rows as u32, rank as u32, &hp);
    let tk = b.i32(rows as u32, 1, &toks);
    let pr = b.bf16(vocab as u32, rank as u32, &pred);
    let sc = b.bf16(vocab as u32, rank as u32, &succ);
    let picks = b.i32(rows as u32, 1, &[-5; 7]);
    let ran = b
        .run(|ctx| {
            let ragged = RaggedTensor {
                data: cd,
                indptr: ip,
            };
            dynconv::selector_walk(ctx, ragged, un, Some(hb), tk, pr, sc, 1, picks)
        })
        .unwrap();
    if !ran {
        return;
    }
    let mut want = vec![-5i32; rows];
    for l in 0..2 {
        let (lo, hi) = (indptr[l] as usize, indptr[l + 1] as usize);
        want[lo] = cands[lo * k];
        let mut prev = toks[lo];
        for row in lo + 1..hi {
            let mut best = 0usize;
            let mut best_score = f32::NEG_INFINITY;
            for c in 0..k {
                let id = cands[row * k + c];
                let mut s = unary[row * k + c];
                if (0..vocab as i32).contains(&prev) && (0..vocab as i32).contains(&id) {
                    s += (0..rank)
                        .map(|d| {
                            pred[prev as usize * rank + d]
                                * hp[row * rank + d]
                                * succ[id as usize * rank + d]
                        })
                        .sum::<f32>();
                }
                if s > best_score {
                    best_score = s;
                    best = c;
                }
            }
            want[row] = cands[row * k + best];
            prev = want[row];
        }
    }
    assert_eq!(b.read_i32(picks), want);
}
