//! Mixture-of-experts routing on the first wgpu adapter, against a host
//! reference: the experts picked exactly and their weights to f64.

use kernels_slang::Tensor;
use poem_ir::Dtype;

use crate::gpu::{Gpu, close, noise};

fn f64s(v: &[f32]) -> Vec<f64> {
    v.iter().map(|x| f64::from(*x)).collect()
}

fn ints(g: &Gpu, b: u32, n: usize) -> Vec<i32> {
    g.down_f32(b, n)
        .iter()
        .map(|v| v.to_bits() as i32)
        .collect()
}

/// The `k` best of `rank`, best first, each once.
fn top(rank: &[f64], k: usize) -> Vec<usize> {
    let mut order: Vec<usize> = (0..rank.len()).collect();
    order.sort_by(|a, b| rank[*b].total_cmp(&rank[*a]));
    order.truncate(k);
    order
}

fn sig(v: f64) -> f64 {
    1.0 / (1.0 + (-v).exp())
}

fn sqrt_softplus(v: f64) -> f64 {
    (if v <= 20.0 { v.exp().ln_1p() } else { v })
        .max(0.0)
        .sqrt()
}

#[test]
fn moe_routing_answers_the_reference() {
    let Some(mut gpu) = Gpu::open() else {
        eprintln!("not asked: no device");
        return;
    };
    let t = |b: u32, r: usize, w: usize, d: Dtype| Tensor::new(b, r as u32, w as u32, d);
    let (rows, experts, k) = (5usize, 64usize, 6usize);
    let (lg, lr) = gpu.up(&noise(1, rows * experts)
        .iter()
        .map(|v| v * 3.0)
        .collect::<Vec<_>>());
    let lr = f64s(&lr);
    let (sc, scr) = gpu.up(&noise(2, experts));
    let scr = f64s(&scr);
    let biasv = noise(3, experts);
    let bias = gpu.up_f32(&biasv);
    let lt = t(lg, rows, experts, Dtype::Bf16);
    let bt = t(bias, 1, experts, Dtype::F32);
    let (eu, ku) = (experts as u32, k as u32);
    let scaling = 2.5f32;
    for form in 0..6 {
        let mut want_ids = Vec::new();
        let mut want_w = Vec::new();
        for r in 0..rows {
            let row = &lr[r * experts..][..experts];
            let (score, rank): (Vec<f64>, Vec<f64>) = match form {
                0 | 1 => (row.to_vec(), row.to_vec()),
                2 => (
                    row.iter().map(|v| sig(*v)).collect(),
                    row.iter().map(|v| sig(*v)).collect(),
                ),
                3 => (
                    row.iter().map(|v| sig(*v)).collect(),
                    row.iter()
                        .zip(&biasv)
                        .map(|(v, b)| sig(*v) + f64::from(*b))
                        .collect(),
                ),
                _ => (
                    row.iter().map(|v| sqrt_softplus(*v)).collect(),
                    row.iter()
                        .zip(&biasv)
                        .map(|(v, b)| sqrt_softplus(*v) + f64::from(*b))
                        .collect(),
                ),
            };
            let picks = top(&rank, k);
            let mut w: Vec<f64> = picks.iter().map(|&e| score[e]).collect();
            if form <= 1 {
                let m = w.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let z: f64 = w.iter().map(|v| (v - m).exp()).sum();
                w = picks
                    .iter()
                    .zip(&w)
                    .map(|(&e, v)| (v - m).exp() / z * if form == 1 { scr[e] } else { 1.0 })
                    .collect();
            } else {
                // Renormalized and scaled, but the prediction is neither.
                let (renorm, scale) = if form == 5 {
                    (false, 1.0)
                } else {
                    (true, f64::from(scaling))
                };
                let sum: f64 = w.iter().sum();
                let s = if renorm && sum > 0.0 {
                    scale / sum
                } else {
                    scale
                };
                w = w.iter().map(|v| v * s).collect();
            }
            want_ids.extend(picks.iter().map(|&e| e as i32));
            want_w.extend(w);
        }
        let (ids, wb) = (
            gpu.up_i32(&vec![-9; rows * k]),
            gpu.up_f32(&vec![0.0; rows * k]),
        );
        let (it, wt) = (t(ids, rows, k, Dtype::I32), t(wb, rows, k, Dtype::F32));
        let sct = t(sc, 1, experts, Dtype::Bf16);
        let (lt, bt, it, wt, sct) = (lt, bt, it, wt, sct);
        gpu.run(|c| match form {
            0 => kernels_slang::linear::moe::route::topk_softmax(c, lt, eu, ku, it, wt),
            1 => kernels_slang::linear::moe::route::topk_softmax_scaled(c, lt, sct, eu, ku, it, wt),
            2 => kernels_slang::linear::moe::route::topk_sigmoid(
                c, lt, eu, ku, true, scaling, it, wt,
            ),
            3 => kernels_slang::linear::moe::route::topk_sigmoid_biased(
                c, lt, bt, eu, ku, true, scaling, it, wt,
            ),
            4 => kernels_slang::linear::moe::route::topk_sqrt_softplus(
                c, lt, bt, eu, ku, true, scaling, it, wt,
            ),
            _ => kernels_slang::linear::moe::route::predict_route(c, lt, bt, eu, ku, it, wt),
        });

        let who = format!("moe router {form}");
        assert_eq!(ints(&gpu, ids, rows * k), want_ids, "{who}: experts");
        close(&gpu.down_f32(wb, rows * k), &want_w, &who);
    }

    // Hash routing: three token ids, one out of the vocabulary (row 0's
    // table entry stands in); one table entry past u32 names no expert.
    let vocab = 4usize;
    let mut table: Vec<u64> = (0..vocab * k).map(|i| ((i * 7) % experts) as u64).collect();
    table[k + 2] = 1 << 33;
    let words: Vec<u32> = table
        .iter()
        .flat_map(|v| [*v as u32, (*v >> 32) as u32])
        .collect();
    let tb = gpu.up_u32(&words);
    let toks = [1u32, 9, 3, 0, 2];
    let tk = gpu.up_u32(&toks);
    let mut want_ids = Vec::new();
    let mut want_w = Vec::new();
    for (r, &tok) in toks.iter().enumerate() {
        let tid = if (tok as usize) < vocab {
            tok as usize
        } else {
            0
        };
        let w: Vec<f64> = (0..k)
            .map(|j| {
                let e = table[tid * k + j];
                if e < experts as u64 {
                    sqrt_softplus(lr[r * experts + e as usize])
                } else {
                    0.0
                }
            })
            .collect();
        let sum: f64 = w.iter().sum();
        want_ids.extend((0..k).map(|j| table[tid * k + j] as u32 as i32));
        want_w.extend(w.iter().map(|v| v * f64::from(scaling) / sum));
    }
    let (ids, wb) = (
        gpu.up_i32(&vec![-9; rows * k]),
        gpu.up_f32(&vec![0.0; rows * k]),
    );
    let (it, wt) = (t(ids, rows, k, Dtype::I32), t(wb, rows, k, Dtype::F32));
    let (tkt, tbt) = (t(tk, rows, 1, Dtype::U32), t(tb, vocab, k, Dtype::I64));
    gpu.run(|c| {
        kernels_slang::linear::moe::route::hash_route(
            c,
            tkt,
            tbt,
            lt,
            vocab as u32,
            ku,
            true,
            scaling,
            it,
            wt,
        )
    });

    let who = "moe hash route";
    assert_eq!(ints(&gpu, ids, rows * k), want_ids, "{who}: experts");
    close(&gpu.down_f32(wb, rows * k), &want_w, who);
    let ids = gpu.up_i32(&vec![-9; rows * 4]);
    let it = t(ids, rows, 4, Dtype::I32);
    gpu.run(|c| kernels_slang::linear::moe::route::group_routes(c, 4, it));

    let want: Vec<i32> = (0..rows * 4).map(|i| (i % 4) as i32).collect();
    assert_eq!(ints(&gpu, ids, rows * 4), want, "group routes");
    eprintln!("moe routing: exact experts");
}
