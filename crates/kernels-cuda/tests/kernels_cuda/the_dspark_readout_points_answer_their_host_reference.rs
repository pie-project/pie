#![cfg(feature = "cuda")]

//! DSpark's readout points: the argmax, the top-k, the selector and Markov
//! walks, the kv ends and the column concat, against the host.

use crate::common::{Gpu, Lcg, from_bf16, to_bf16};
use dtype::Dtype;
use kernels_cuda::attn::{markov, selector};
use kernels_cuda::layout;
use kernels_cuda::tensor::{KvPool, RaggedTensor, Tensor};

fn beats(av: f32, ai: usize, bv: f32, bi: usize) -> bool {
    av > bv || (av == bv && ai < bi)
}

fn host_topk(row: &[f32], k: usize) -> Vec<(f32, usize)> {
    let mut taken: Vec<(f32, usize)> = Vec::new();
    for _ in 0..k {
        let mut best: Option<(f32, usize)> = None;
        for (i, &v) in row.iter().enumerate() {
            if v.is_nan() || taken.iter().any(|(_, t)| *t == i) {
                continue;
            }
            if best.is_none_or(|(bv, bi)| beats(v, i, bv, bi)) {
                best = Some((v, i));
            }
        }
        taken.push(best.unwrap_or((0.0, 0)));
    }
    taken
}

fn tied_rows(lcg: &mut Lcg, rows: usize, width: usize) -> (Vec<u16>, Vec<f32>) {
    let mut raw = Vec::with_capacity(rows * width);
    let mut f = Vec::with_capacity(rows * width);
    for r in 0..rows {
        for c in 0..width {
            let v = if c == (r * 7 + 3) % width {
                f32::NAN
            } else if c == (r * 11 + 5) % width {
                f32::NEG_INFINITY
            } else {
                ((lcg.unit() * 8.0).floor()) * 0.5
            };
            let b = to_bf16(v);
            raw.push(b);
            f.push(from_bf16(b));
        }
    }
    (raw, f)
}

#[test]
fn the_argmax_lands_the_lowest_tied_column_and_skips_a_nan() {
    let (rows, width, depth) = (5usize, 3000usize, 3usize);
    let mut lcg = Lcg::seeded(0x51);
    let (x_raw, x) = tied_rows(&mut lcg, rows, width);
    let mut x_raw = x_raw;
    let mut x = x;
    for c in 0..width {
        x_raw[4 * width + c] = to_bf16(f32::NAN);
        x[4 * width + c] = f32::NAN;
    }
    x_raw[4 * width + 2999] = to_bf16(1.0);
    x[4 * width + 2999] = 1.0;

    let mut gpu = Gpu::open();
    let x_at = gpu.up(&x_raw);
    let y_at = gpu.zeros(rows * depth * 4);
    for column in 0..depth {
        let mut y = Tensor::new(y_at, rows as u32, depth as u32, Dtype::I32);
        layout::argmax(
            &gpu.ctx(),
            Tensor::new(x_at, rows as u32, width as u32, Dtype::Bf16),
            column as u32,
            &mut y,
        )
        .expect("the argmax fires");
    }
    gpu.sync();
    let got: Vec<i32> = gpu.down(y_at, rows * depth);
    for r in 0..rows {
        let want = host_topk(&x[r * width..(r + 1) * width], 1)[0].1;
        for column in 0..depth {
            assert_eq!(
                got[r * depth + column],
                want as i32,
                "row {r} column {column}"
            );
        }
    }
    assert_eq!(got[4 * depth], 2999, "the one finite entry of the NaN row");

    let x32_at = gpu.up(&x);
    let y32_at = gpu.zeros(rows * 4);
    let mut y = Tensor::new(y32_at, rows as u32, 1, Dtype::I32);
    layout::argmax(
        &gpu.ctx(),
        Tensor::new(x32_at, rows as u32, width as u32, Dtype::F32),
        0,
        &mut y,
    )
    .expect("the f32 argmax fires");
    gpu.sync();
    let got32: Vec<i32> = gpu.down(y32_at, rows);
    for r in 0..rows {
        assert_eq!(got32[r], got[r * depth], "f32 row {r}");
    }
}

#[test]
fn the_topk_is_sorted_with_ties_to_the_lower_column() {
    let (rows, width) = (6usize, 4097usize);
    let mut lcg = Lcg::seeded(0x7a);
    let (x_raw, x) = tied_rows(&mut lcg, rows, width);
    let mut gpu = Gpu::open();
    let x_at = gpu.up(&x_raw);
    let x32_at = gpu.up(&x);
    for k in [8usize, 16] {
        for (dtype, at) in [(Dtype::Bf16, x_at), (Dtype::F32, x32_at)] {
            let v_at = gpu.zeros(rows * k * 4);
            let i_at = gpu.zeros(rows * k * 4);
            let mut values = Tensor::new(v_at, rows as u32, k as u32, Dtype::F32);
            let mut indices = Tensor::new(i_at, rows as u32, k as u32, Dtype::I32);
            layout::topk(
                &gpu.ctx(),
                Tensor::new(at, rows as u32, width as u32, dtype),
                k as u32,
                &mut values,
                &mut indices,
            )
            .expect("the top-k fires");
            gpu.sync();
            let got_v: Vec<f32> = gpu.down(v_at, rows * k);
            let got_i: Vec<i32> = gpu.down(i_at, rows * k);
            for r in 0..rows {
                let want = host_topk(&x[r * width..(r + 1) * width], k);
                for j in 0..k {
                    assert_eq!(
                        got_i[r * k + j],
                        want[j].1 as i32,
                        "{dtype:?} k={k} row {r} rank {j}: index (values {} against {})",
                        got_v[r * k + j],
                        want[j].0
                    );
                    assert_eq!(
                        got_v[r * k + j],
                        want[j].0,
                        "{dtype:?} k={k} row {r} rank {j}: value"
                    );
                }
            }
        }
    }
}

#[test]
fn the_selector_walk_follows_the_best_successor_from_the_anchor() {
    let (vocab, rank, k) = (64usize, 256usize, 16usize);
    let indptr: [i32; 4] = [0, 4, 5, 11];
    let rows = 11usize;
    let mut lcg = Lcg::seeded(0x33);
    let (pred_raw, pred) = lcg.row(vocab * rank);
    let (succ_raw, succ) = lcg.row(vocab * rank);
    let (hp_raw, hp) = lcg.row(rows * rank);
    let cand: Vec<i32> = (0..rows * k)
        .map(|i| ((i * 37 + 11) % vocab) as i32)
        .collect();
    let tokens: Vec<i32> = (0..rows).map(|r| ((r * 13 + 5) % vocab) as i32).collect();
    let unary: Vec<f32> = (0..rows * k)
        .map(|_| (lcg.unit() * 4.0).floor() * 0.25)
        .collect();

    let mut gpu = Gpu::open();
    let cand_at = gpu.up(&cand);
    let indptr_at = gpu.up(&indptr);
    let unary_at = gpu.up(&unary);
    let hp_at = gpu.up(&hp_raw);
    let tokens_at = gpu.up(&tokens);
    let pred_at = gpu.up(&pred_raw);
    let succ_at = gpu.up(&succ_raw);

    for (with_hp, first) in [(false, 1u32), (true, 0)] {
        let picks_at = gpu.zeros(rows * 4);
        let mut picks = Tensor::new(picks_at, rows as u32, 1, Dtype::I32);
        selector::walk(
            &gpu.ctx(),
            RaggedTensor {
                data: Tensor::new(cand_at, rows as u32, k as u32, Dtype::I32),
                indptr: Tensor::new(indptr_at, 4, 1, Dtype::I32),
            },
            Tensor::new(unary_at, rows as u32, k as u32, Dtype::F32),
            with_hp.then(|| Tensor::new(hp_at, rows as u32, rank as u32, Dtype::Bf16)),
            Tensor::new(tokens_at, rows as u32, 1, Dtype::I32),
            Tensor::new(pred_at, vocab as u32, rank as u32, Dtype::Bf16),
            Tensor::new(succ_at, vocab as u32, rank as u32, Dtype::Bf16),
            first,
            &mut picks,
        )
        .expect("the walk fires");
        gpu.sync();
        let got: Vec<i32> = gpu.down(picks_at, rows);

        for lane in 0..3 {
            let (begin, end) = (indptr[lane] as usize, indptr[lane + 1] as usize);
            let mut prev = tokens[begin] as usize;
            // Candidate `c`'s score at `row` after `prev`.
            let score = |row: usize, c: usize, prev: usize| {
                let cid = cand[row * k + c] as usize;
                let dot: f32 = (0..rank)
                    .map(|d| {
                        let (a, b) = (pred[prev * rank + d], succ[cid * rank + d]);
                        if with_hp {
                            a * hp[row * rank + d] * b
                        } else {
                            a * b
                        }
                    })
                    .sum();
                unary[row * k + c] + dot
            };
            if first == 1 {
                assert_eq!(
                    got[begin],
                    cand[begin * k],
                    "hp={with_hp} first={first}: the anchor's pick is its first candidate"
                );
            }
            for row in begin + first as usize..end {
                let mut best = 0usize;
                let mut best_v = f32::NEG_INFINITY;
                for c in 0..k {
                    let s = score(row, c, prev);
                    if c == 0 || s > best_v {
                        best_v = s;
                        best = c;
                    }
                }
                let want = cand[row * k + best];
                if got[row] != want {
                    let c = (0..k).find(|&c| cand[row * k + c] == got[row]).unwrap_or_else(|| {
                        panic!("hp={with_hp} first={first}: lane {lane} row {row} picked {}, not a candidate", got[row])
                    });
                    let s = score(row, c, prev);
                    assert!(
                        (s - best_v).abs() < 1e-3,
                        "hp={with_hp} first={first}: lane {lane} row {row} picked {} ({s}) over {want} ({best_v})",
                        got[row]
                    );
                }
                prev = got[row] as usize;
            }
        }
    }
}

/// Each row's logits biased by `head . embed[pick before it]`, its pick
/// the biased argmax: a vocabulary off the 128-row tile, blocks read out
/// 5, 1, 3 and 5 rows deep, f32 logits and the trunk's bf16 ones.
#[test]
fn the_markov_walk_biases_each_row_by_the_pick_before_it() {
    let (vocab, rank, rows) = (1000usize, markov::RANK as usize, 14usize);
    let (indptr, firsts) = ([0usize, 5, 6, 9, 14], [100i32, 7, 50, 200]);
    let lane = |r: usize| indptr.iter().rposition(|&s| s <= r).unwrap();
    let positions: Vec<i32> = (0..rows)
        .map(|r| firsts[lane(r)] + (r - indptr[lane(r)]) as i32)
        .collect();
    let ends: Vec<i32> = (0..rows).map(|r| firsts[lane(r)] + 4).collect();
    let mut lcg = Lcg::seeded(0x4d);
    let ((embed_raw, embed), (head_raw, head)) = (lcg.row(vocab * rank), lcg.row(vocab * rank));
    let logits_raw: Vec<u16> = (0..rows * vocab)
        .map(|_| to_bf16(lcg.unit() * 4.0))
        .collect();
    let logits: Vec<f32> = logits_raw.iter().map(|&v| from_bf16(v)).collect();
    let tokens: Vec<i32> = (0..rows).map(|r| ((r * 131 + 17) % vocab) as i32).collect();
    let mut gpu = Gpu::open();
    let t = |at: u64, rows: usize, width: usize, dtype| {
        Tensor::new(at, rows as u32, width as u32, dtype)
    };
    let (pos_t, ends_t) = (
        t(gpu.up(&positions), rows, 1, Dtype::I32),
        t(gpu.up(&ends), rows, 1, Dtype::I32),
    );
    let (tok_t, embed_t) = (
        t(gpu.up(&tokens), rows, 1, Dtype::I32),
        t(gpu.up(&embed_raw), vocab, rank, Dtype::Bf16),
    );
    let head_t = t(gpu.up(&head_raw), vocab, rank, Dtype::Bf16);
    for dtype in [Dtype::F32, Dtype::Bf16] {
        let bf = dtype == Dtype::Bf16;
        let logits_at = if bf {
            gpu.up(&logits_raw)
        } else {
            gpu.up(&logits)
        };
        let (biased_at, picks_at) = (gpu.zeros(rows * vocab * 4), gpu.zeros(rows * 4));
        let (biased, picks) = (
            &mut t(biased_at, rows, vocab, dtype),
            &mut t(picks_at, rows, 1, Dtype::I32),
        );
        let logits_t = t(logits_at, rows, vocab, dtype);
        let fired = markov::walk(
            &gpu.ctx(),
            logits_t,
            tok_t,
            pos_t,
            ends_t,
            embed_t,
            head_t,
            5,
            biased,
            picks,
        );
        fired.expect("the walk fires");
        gpu.sync();
        let got: Vec<f32> = match bf {
            true => gpu
                .down::<u16>(biased_at, rows * vocab)
                .into_iter()
                .map(from_bf16)
                .collect(),
            false => gpu.down(biased_at, rows * vocab),
        };
        let got_picks: Vec<i32> = gpu.down(picks_at, rows);
        let mut prev = 0;
        for row in 0..rows {
            if indptr.contains(&row) {
                prev = tokens[row] as usize;
            }
            let bias = |v: usize| {
                (0..rank)
                    .map(|k| head[v * rank + k] * embed[prev * rank + k])
                    .sum::<f32>()
            };
            let want: Vec<f32> = (0..vocab)
                .map(|v| logits[row * vocab + v] + bias(v))
                .collect();
            // bf16 lands the sum to half its last place.
            let tol = |w: f32| 1e-2 + if bf { w.abs() / 256.0 } else { 0.0 };
            let line = &got[row * vocab..(row + 1) * vocab];
            let off = (0..vocab).find(|&v| (line[v] - want[v]).abs() >= tol(want[v]));
            assert!(
                off.is_none(),
                "{dtype:?} row {row}: column {off:?} off the host"
            );
            // The pick is the biased row's argmax; the next row's bias is the pick's.
            let (pick, best) = (
                got_picks[row] as usize,
                want.iter().copied().fold(f32::MIN, f32::max),
            );
            assert!(
                pick < vocab && want[pick] >= best - 1e-2,
                "{dtype:?} row {row} picked {pick}"
            );
            prev = pick;
        }
    }
}

/// Each row's kv end: three lanes over 16-position pages (2 pages 5 deep,
/// 1 page 16 deep, 3 pages 1 deep), and a padded row of none.
#[test]
fn the_kv_ends_land_what_the_host_lands() {
    let mut gpu = Gpu::open();
    let req: Vec<i32> = vec![0, 0, 1, 2, 2, 2, -1];
    let indptr = Tensor::new(gpu.up(&[0u32, 2, 3, 6]), 4, 1, Dtype::U32);
    let pool = crate::common::attn_ref::pool(Tensor::ABSENT, Tensor::ABSENT, indptr, 16);
    let last_page_lens = Tensor::new(gpu.up(&[5u32, 16, 1]), 3, 1, Dtype::U32);
    let pool = KvPool {
        last_page_lens,
        max_pages_per_request: 3,
        pages_in_batch: 6,
        live: [0, 0],
        ..pool
    };
    let (req_t, ends_at) = (
        Tensor::new(gpu.up(&req), 7, 1, Dtype::I32),
        gpu.zeros(7 * 4),
    );
    let ends = &mut Tensor::new(ends_at, 7, 1, Dtype::I32);
    kernels_cuda::attn_replay::kv_ends(&gpu.ctx(), req_t, &pool, ends).expect("the kv ends fire");
    gpu.sync();
    assert_eq!(
        gpu.down::<i32>(ends_at, 7),
        vec![20, 20, 15, 32, 32, 32, -1]
    );
}

/// Unaligned widths (a byte a thread) and 16-byte runs.
#[test]
fn the_column_concat_lands_what_the_host_lands() {
    let (mut gpu, rows) = (Gpu::open(), 37usize);
    for widths in [[8usize, 3, 16], [8, 16, 64]] {
        let mut lcg = Lcg::seeded(0x91);
        let parts: Vec<Vec<u16>> = widths.iter().map(|w| lcg.row(rows * w).0).collect();
        let total: usize = widths.iter().sum();
        let t = |at: u64, w: usize| Tensor::new(at, rows as u32, w as u32, Dtype::Bf16);
        let xs: Vec<Tensor> = parts
            .iter()
            .zip(widths)
            .map(|(p, w)| t(gpu.up(p), w))
            .collect();
        let y_at = gpu.zeros(rows * total * 2);
        let concat =
            kernels_cuda::layout_concat_cols::concat_cols(&gpu.ctx(), &xs, &mut t(y_at, total));
        concat.expect("the concat fires");
        gpu.sync();
        let got: Vec<u16> = gpu.down(y_at, rows * total);
        let row = |r: usize| {
            parts
                .iter()
                .zip(widths)
                .flat_map(move |(p, w)| p[r * w..(r + 1) * w].to_vec())
        };
        let want: Vec<u16> = (0..rows).flat_map(row).collect();
        assert_eq!(got, want, "{widths:?}");
    }
}
