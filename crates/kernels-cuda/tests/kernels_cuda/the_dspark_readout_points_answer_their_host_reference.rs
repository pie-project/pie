#![cfg(feature = "cuda")]

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
fn the_dspark_readout_points_answer_their_host_reference_every_case() {
    the_argmax_lands_the_lowest_tied_column_and_skips_a_nan();
    the_topk_is_sorted_with_ties_to_the_lower_column();
    the_selector_walk_follows_the_best_successor_from_the_anchor();
    the_markov_walk_biases_each_row_by_the_pick_before_it();
    the_kv_ends_and_column_concat_land_what_the_host_lands();
}

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

    for (with_hp, first) in [(false, 0u32), (false, 1), (true, 0), (true, 1)] {
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
                    let cid = cand[row * k + c] as usize;
                    let mut dot = 0f32;
                    for d in 0..rank {
                        let a = pred[prev * rank + d];
                        let b = succ[cid * rank + d];
                        dot += if with_hp {
                            a * hp[row * rank + d] * b
                        } else {
                            a * b
                        };
                    }
                    let s = unary[row * k + c] + dot;
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
                    let cid = got[row] as usize;
                    let mut dot = 0f32;
                    for d in 0..rank {
                        let a = pred[prev * rank + d];
                        let b = succ[cid * rank + d];
                        dot += if with_hp {
                            a * hp[row * rank + d] * b
                        } else {
                            a * b
                        };
                    }
                    let s = unary[row * k + c] + dot;
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

fn the_markov_walk_biases_each_row_by_the_pick_before_it() {
    // A vocabulary off the 128-row tile; blocks read out 5, 1, 3 and 5 rows
    // deep; f32 logits and the trunk's bf16 ones.
    let (vocab, rank, block) = (1000usize, markov::RANK as usize, 5u32);
    let indptr: [i32; 5] = [0, 5, 6, 9, 14];
    let rows = 14usize;
    let firsts = [100i32, 7, 50, 200];
    let (mut positions, mut ends) = (Vec::new(), Vec::new());
    for lane in 0..4 {
        for row in 0..(indptr[lane + 1] - indptr[lane]) {
            positions.push(firsts[lane] + row);
            ends.push(firsts[lane] + block as i32 - 1);
        }
    }
    let mut lcg = Lcg::seeded(0x4d);
    let (embed_raw, embed) = lcg.row(vocab * rank);
    let (head_raw, head) = lcg.row(vocab * rank);
    let logits: Vec<f32> = (0..rows * vocab)
        .map(|_| from_bf16(to_bf16(lcg.unit() * 4.0)))
        .collect();
    let logits_raw: Vec<u16> = logits.iter().map(|&v| to_bf16(v)).collect();
    let tokens: Vec<i32> = (0..rows).map(|r| ((r * 131 + 17) % vocab) as i32).collect();

    let mut gpu = Gpu::open();
    let positions_at = gpu.up(&positions);
    let ends_at = gpu.up(&ends);
    let tokens_at = gpu.up(&tokens);
    let embed_at = gpu.up(&embed_raw);
    let head_at = gpu.up(&head_raw);
    for dtype in [Dtype::F32, Dtype::Bf16] {
        let (logits_at, elem) = if dtype == Dtype::F32 {
            (gpu.up(&logits), 4)
        } else {
            (gpu.up(&logits_raw), 2)
        };
        let biased_at = gpu.zeros(rows * vocab * elem);
        let picks_at = gpu.zeros(rows * 4);
        let mut biased = Tensor::new(biased_at, rows as u32, vocab as u32, dtype);
        let mut picks = Tensor::new(picks_at, rows as u32, 1, Dtype::I32);
        markov::walk(
            &gpu.ctx(),
            Tensor::new(logits_at, rows as u32, vocab as u32, dtype),
            Tensor::new(tokens_at, rows as u32, 1, Dtype::I32),
            Tensor::new(positions_at, rows as u32, 1, Dtype::I32),
            Tensor::new(ends_at, rows as u32, 1, Dtype::I32),
            Tensor::new(embed_at, vocab as u32, rank as u32, Dtype::Bf16),
            Tensor::new(head_at, vocab as u32, rank as u32, Dtype::Bf16),
            block,
            &mut biased,
            &mut picks,
        )
        .expect("the walk fires");
        gpu.sync();
        let got: Vec<f32> = if dtype == Dtype::F32 {
            gpu.down(biased_at, rows * vocab)
        } else {
            gpu.down::<u16>(biased_at, rows * vocab)
                .into_iter()
                .map(from_bf16)
                .collect()
        };
        let got_picks: Vec<i32> = gpu.down(picks_at, rows);
        // bf16 lands the sum to half its last place.
        let tol = |want: f32| {
            if dtype == Dtype::F32 {
                1e-2
            } else {
                1e-2 + want.abs() / 256.0
            }
        };

        for lane in 0..4 {
            let (begin, end) = (indptr[lane] as usize, indptr[lane + 1] as usize);
            let mut prev = tokens[begin] as usize;
            for row in begin..end {
                let want: Vec<f32> = (0..vocab)
                    .map(|v| {
                        let dot: f32 = (0..rank)
                            .map(|k| head[v * rank + k] * embed[prev * rank + k])
                            .sum();
                        logits[row * vocab + v] + dot
                    })
                    .collect();
                let line = &got[row * vocab..(row + 1) * vocab];
                for v in 0..vocab {
                    assert!(
                        (line[v] - want[v]).abs() < tol(want[v]),
                        "{dtype:?} lane {lane} row {row} column {v}: {} against {}",
                        line[v],
                        want[v]
                    );
                }
                // The pick is the biased row's argmax; the next row's bias
                // is the pick's.
                let pick = got_picks[row] as usize;
                let best = want.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                assert!(
                    pick < vocab && want[pick] >= best - 1e-2,
                    "{dtype:?} lane {lane} row {row} picked {pick} ({}) against the best {best}",
                    want.get(pick).copied().unwrap_or(f32::NAN)
                );
                prev = pick;
            }
        }
    }
}

fn the_kv_ends_and_column_concat_land_what_the_host_lands() {
    let mut gpu = Gpu::open();
    // Three lanes over 16-position pages: 2 pages 5 deep, 1 page 16 deep,
    // 3 pages 1 deep; rows of each, and a padded row of none.
    const PAGE: i32 = 16;
    let indptr: [u32; 4] = [0, 2, 3, 6];
    let last: [u32; 3] = [5, 16, 1];
    let req: Vec<i32> = vec![0, 0, 1, 2, 2, 2, -1];
    let pool = KvPool {
        keys: Tensor::ABSENT,
        values: Tensor::ABSENT,
        bf16_keys: Tensor::ABSENT,
        bf16_values: Tensor::ABSENT,
        key_scales: Tensor::ABSENT,
        value_scales: Tensor::ABSENT,
        page_indices: Tensor::ABSENT,
        page_indptr: Tensor::new(gpu.up(&indptr), 4, 1, Dtype::U32),
        last_page_lens: Tensor::new(gpu.up(&last), 3, 1, Dtype::U32),
        row_valid: Tensor::ABSENT,
        env_min: Tensor::ABSENT,
        env_max: Tensor::ABSENT,
        has_envelopes: false,
        page_size: PAGE,
        seq_stride: 0,
        head_stride: 0,
        layout: 0,
        scheme_byte: 0,
        block_size: 0,
        max_pages_per_request: 3,
        pages_in_batch: 6,
    };
    let req_at = gpu.up(&req);
    let ends_at = gpu.zeros(req.len() * 4);
    let mut ends = Tensor::new(ends_at, req.len() as u32, 1, Dtype::I32);
    kernels_cuda::attn_replay::kv_ends(
        &gpu.ctx(),
        Tensor::new(req_at, req.len() as u32, 1, Dtype::I32),
        &pool,
        &mut ends,
    )
    .expect("the kv ends fire");
    gpu.sync();
    let got: Vec<i32> = gpu.down(ends_at, req.len());
    assert_eq!(got, vec![20, 20, 15, 32, 32, 32, -1]);

    // Unaligned widths (a byte a thread) and 16-byte runs.
    let rows = 37usize;
    for widths in [vec![8usize, 3, 16], vec![8, 16, 64]] {
        let mut lcg = Lcg::seeded(0x91);
        let parts: Vec<Vec<u16>> = widths.iter().map(|w| lcg.row(rows * w).0).collect();
        let total: usize = widths.iter().sum();
        let ats: Vec<u64> = parts.iter().map(|p| gpu.up(p)).collect();
        let y_at = gpu.zeros(rows * total * 2);
        let xs: Vec<Tensor> = ats
            .iter()
            .zip(&widths)
            .map(|(&at, &w)| Tensor::new(at, rows as u32, w as u32, Dtype::Bf16))
            .collect();
        let mut y = Tensor::new(y_at, rows as u32, total as u32, Dtype::Bf16);
        kernels_cuda::layout_concat_cols::concat_cols(&gpu.ctx(), &xs, &mut y)
            .expect("the concat fires");
        gpu.sync();
        let got: Vec<u16> = gpu.down(y_at, rows * total);
        for r in 0..rows {
            let want: Vec<u16> = parts
                .iter()
                .zip(&widths)
                .flat_map(|(p, &w)| p[r * w..(r + 1) * w].iter().copied())
                .collect();
            assert_eq!(
                &got[r * total..(r + 1) * total],
                &want[..],
                "{widths:?} row {r}"
            );
        }
    }
}

/// `PIE_MARKOV_PACE=lanes` times the walk at the release's vocabulary over
/// that many five-row blocks of bf16 logits (each depth a step).
#[test]
fn the_markov_walk_paces_a_release_block() {
    let Some(lanes) = std::env::var("PIE_MARKOV_PACE")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
    else {
        return;
    };
    let (vocab, rank, block) = (129_280usize, markov::RANK as usize, 5usize);
    let rows = lanes * block;
    let mut gpu = Gpu::open();
    let mut lcg = Lcg::seeded(0x77);
    let (embed_raw, _) = lcg.row(vocab * rank);
    let (head_raw, _) = lcg.row(vocab * rank);
    let logits_raw: Vec<u16> = (0..rows * vocab)
        .map(|i| to_bf16(((i * 37) % 101) as f32 * 0.01))
        .collect();
    let positions: Vec<i32> = (0..rows).map(|r| (r % block) as i32 + 100).collect();
    let ends: Vec<i32> = vec![104; rows];
    let tokens: Vec<i32> = (0..rows).map(|r| ((r * 131) % vocab) as i32).collect();
    let (logits_at, embed_at, head_at) =
        (gpu.up(&logits_raw), gpu.up(&embed_raw), gpu.up(&head_raw));
    let (pos_at, ends_at, tok_at) = (gpu.up(&positions), gpu.up(&ends), gpu.up(&tokens));
    let biased_at = gpu.zeros(rows * vocab * 2);
    let picks_at = gpu.zeros(rows * 4);
    let fire = |gpu: &mut Gpu| {
        let mut biased = Tensor::new(biased_at, rows as u32, vocab as u32, Dtype::Bf16);
        let mut picks = Tensor::new(picks_at, rows as u32, 1, Dtype::I32);
        markov::walk(
            &gpu.ctx(),
            Tensor::new(logits_at, rows as u32, vocab as u32, Dtype::Bf16),
            Tensor::new(tok_at, rows as u32, 1, Dtype::I32),
            Tensor::new(pos_at, rows as u32, 1, Dtype::I32),
            Tensor::new(ends_at, rows as u32, 1, Dtype::I32),
            Tensor::new(embed_at, vocab as u32, rank as u32, Dtype::Bf16),
            Tensor::new(head_at, vocab as u32, rank as u32, Dtype::Bf16),
            block as u32,
            &mut biased,
            &mut picks,
        )
        .expect("the walk fires");
    };
    for _ in 0..3 {
        fire(&mut gpu);
    }
    gpu.sync();
    let iters = 20;
    let t0 = std::time::Instant::now();
    for _ in 0..iters {
        fire(&mut gpu);
    }
    gpu.sync();
    println!(
        "markov walk: {lanes} blocks x {block} rows, {:.1} us a walk",
        t0.elapsed().as_secs_f64() * 1e6 / f64::from(iters)
    );
}
