#![cfg(feature = "cuda")]

mod common;

use common::{Gpu, Lcg, from_bf16, to_bf16};
use dtype::Dtype;
use kernels_cuda::attn::selector;
use kernels_cuda::layout;
use kernels_cuda::tensor::{RaggedTensor, Tensor};

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
            assert_eq!(got[r * depth + column], want as i32, "row {r} column {column}");
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
                        got_i[r * k + j], want[j].1 as i32,
                        "{dtype:?} k={k} row {r} rank {j}: index (values {} against {})",
                        got_v[r * k + j], want[j].0
                    );
                    assert_eq!(got_v[r * k + j], want[j].0, "{dtype:?} k={k} row {r} rank {j}: value");
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
    let cand: Vec<i32> = (0..rows * k).map(|i| ((i * 37 + 11) % vocab) as i32).collect();
    let tokens: Vec<i32> = (0..rows).map(|r| ((r * 13 + 5) % vocab) as i32).collect();
    let unary: Vec<f32> = (0..rows * k).map(|_| (lcg.unit() * 4.0).floor() * 0.25).collect();

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
                assert_eq!(got[begin], cand[begin * k], "hp={with_hp} first={first}: the anchor's pick is its first candidate");
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
                        dot += if with_hp { a * hp[row * rank + d] * b } else { a * b };
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
                        dot += if with_hp { a * hp[row * rank + d] * b } else { a * b };
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
