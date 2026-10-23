//! Mixture-of-experts routing and combines on the first Vulkan device, beside the
//! hand-written kernels, against a host reference: the experts picked
//! exactly, their weights and the combined rows to f64.

use kernels_vulkan::{Tensor, linear::moe};
use poem_ir::Dtype;

use crate::gpu::{Gpu, both, close, noise};

fn st(t: Tensor) -> kernels_slang::Tensor {
    kernels_slang::Tensor::new(t.buf, t.rows, t.width, t.dtype)
}

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
        for slang in [false, true] {
            let (ids, wb) = (
                gpu.up_i32(&vec![-9; rows * k]),
                gpu.up_f32(&vec![0.0; rows * k]),
            );
            let (it, wt) = (t(ids, rows, k, Dtype::I32), t(wb, rows, k, Dtype::F32));
            let sct = t(sc, 1, experts, Dtype::Bf16);
            if slang {
                let (lt, bt, it, wt, sct) = (st(lt), st(bt), st(it), st(wt), st(sct));
                gpu.run_slang(|c| match form {
                    0 => kernels_slang::linear::moe::route::topk_softmax(c, lt, eu, ku, it, wt),
                    1 => kernels_slang::linear::moe::route::topk_softmax_scaled(
                        c, lt, sct, eu, ku, it, wt,
                    ),
                    2 => kernels_slang::linear::moe::route::topk_sigmoid(
                        c, lt, eu, ku, true, scaling, it, wt,
                    ),
                    3 => kernels_slang::linear::moe::route::topk_sigmoid_biased(
                        c, lt, bt, eu, ku, true, scaling, it, wt,
                    ),
                    4 => kernels_slang::linear::moe::route::topk_sqrt_softplus(
                        c, lt, bt, eu, ku, true, scaling, it, wt,
                    ),
                    _ => {
                        kernels_slang::linear::moe::route::predict_route(c, lt, bt, eu, ku, it, wt)
                    }
                });
            } else {
                gpu.run(|c| match form {
                    0 => moe::topk_softmax(c, lt, eu, ku, it, wt),
                    1 => moe::topk_softmax_scaled(c, lt, sct, eu, ku, it, wt),
                    2 => moe::topk_sigmoid(c, lt, eu, ku, true, scaling, it, wt),
                    3 => moe::topk_sigmoid_biased(c, lt, bt, eu, ku, true, scaling, it, wt),
                    4 => moe::topk_sqrt_softplus(c, lt, bt, eu, ku, true, scaling, it, wt),
                    _ => moe::predict_route(c, lt, bt, eu, ku, it, wt),
                });
            }
            let who = format!("{}moe router {form}", if slang { "slang " } else { "" });
            assert_eq!(ints(&gpu, ids, rows * k), want_ids, "{who}: experts");
            close(&gpu.down_f32(wb, rows * k), &want_w, &who);
        }
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
    for slang in [false, true] {
        let (ids, wb) = (
            gpu.up_i32(&vec![-9; rows * k]),
            gpu.up_f32(&vec![0.0; rows * k]),
        );
        let (it, wt) = (t(ids, rows, k, Dtype::I32), t(wb, rows, k, Dtype::F32));
        let (tkt, tbt) = (t(tk, rows, 1, Dtype::U32), t(tb, vocab, k, Dtype::I64));
        if slang {
            gpu.run_slang(|c| {
                kernels_slang::linear::moe::route::hash_route(
                    c,
                    st(tkt),
                    st(tbt),
                    st(lt),
                    vocab as u32,
                    ku,
                    true,
                    scaling,
                    st(it),
                    st(wt),
                )
            });
        } else {
            gpu.run(|c| moe::hash_route(c, tkt, tbt, lt, vocab as u32, ku, true, scaling, it, wt));
        }
        let who = format!("{}moe hash route", if slang { "slang " } else { "" });
        assert_eq!(ints(&gpu, ids, rows * k), want_ids, "{who}: experts");
        close(&gpu.down_f32(wb, rows * k), &want_w, &who);
    }
    for slang in [false, true] {
        let ids = gpu.up_i32(&vec![-9; rows * 4]);
        let it = t(ids, rows, 4, Dtype::I32);
        if slang {
            gpu.run_slang(|c| kernels_slang::linear::moe::route::group_routes(c, 4, st(it)));
        } else {
            gpu.run(|c| moe::group_routes(c, 4, it));
        }
        let want: Vec<i32> = (0..rows * 4).map(|i| (i % 4) as i32).collect();
        assert_eq!(ints(&gpu, ids, rows * 4), want, "group routes");
    }
    eprintln!("moe routing: exact experts");

    // The combines over 5 rows of 33 (an odd count).
    let w = 33usize;
    let n = rows * w;
    let (routed, rr) = gpu.up(&noise(4, rows * k * w));
    let rr = f64s(&rr);
    let rw: Vec<f32> = noise(5, rows * k);
    let rwb = gpu.up_f32(&rw);
    let want: Vec<f64> = (0..n)
        .map(|i| {
            (0..k)
                .map(|e| f64::from(rw[(i / w) * k + e]) * rr[((i / w) * k + e) * w + i % w])
                .sum()
        })
        .collect();
    let b16 = |b: u32, r: usize, wd: usize| t(b, r, wd, Dtype::Bf16);
    both(
        &mut gpu,
        "moe weighted sum",
        &|g| vec![(g.zeros(n + 1), n, false)],
        &[want],
        &|g, o| {
            g.run(|c| {
                moe::weighted_sum(
                    c,
                    b16(routed, rows * k, w),
                    t(rwb, rows, k, Dtype::F32),
                    b16(o[0].0, rows, w),
                )
            })
        },
        &|g, o| {
            g.run_slang(|c| {
                kernels_slang::linear::moe::combine::weighted_sum(
                    c,
                    st(b16(routed, rows * k, w)),
                    st(t(rwb, rows, k, Dtype::F32)),
                    st(b16(o[0].0, rows, w)),
                )
            })
        },
    );
    let (x, xr) = gpu.up(&noise(6, n));
    let (eb, ebr) = gpu.up(&noise(7, experts * w));
    let (xr, ebr) = (f64s(&xr), f64s(&ebr));
    let routes: Vec<i32> = (0..rows * k)
        .map(|i| {
            if i % 5 == 3 {
                -1
            } else {
                ((i * 11) % experts) as i32
            }
        })
        .collect();
    let rb = gpu.up_i32(&routes);
    let want: Vec<f64> = (0..n)
        .map(|i| {
            let r = i / w;
            xr[i]
                + (0..k)
                    .filter(|e| routes[r * k + e] >= 0)
                    .map(|e| f64::from(rw[r * k + e]) * ebr[routes[r * k + e] as usize * w + i % w])
                    .sum::<f64>()
        })
        .collect();
    both(
        &mut gpu,
        "moe bias sum",
        &|g| vec![(g.zeros(n + 1), n, false)],
        &[want],
        &|g, o| {
            g.run(|c| {
                moe::bias_sum(
                    c,
                    b16(x, rows, w),
                    b16(eb, experts, w),
                    t(rb, rows, k, Dtype::I32),
                    t(rwb, rows, k, Dtype::F32),
                    b16(o[0].0, rows, w),
                )
            })
        },
        &|g, o| {
            g.run_slang(|c| {
                kernels_slang::linear::moe::combine::bias_sum(
                    c,
                    st(b16(x, rows, w)),
                    st(b16(eb, experts, w)),
                    st(t(rb, rows, k, Dtype::I32)),
                    st(t(rwb, rows, k, Dtype::F32)),
                    st(b16(o[0].0, rows, w)),
                )
            })
        },
    );
    let (sh, shr) = gpu.up(&noise(8, n));
    let (gt, gtr) = gpu.up(&noise(9, rows));
    let (shr, gtr) = (f64s(&shr), f64s(&gtr));
    let want: Vec<f64> = (0..n).map(|i| xr[i] + sig(gtr[i / w]) * shr[i]).collect();
    both(
        &mut gpu,
        "moe sigmoid gate add",
        &|g| vec![(g.zeros(n + 1), n, false)],
        &[want],
        &|g, o| {
            g.run(|c| {
                moe::sigmoid_gate_add(
                    c,
                    b16(x, rows, w),
                    b16(sh, rows, w),
                    b16(gt, rows, 1),
                    b16(o[0].0, rows, w),
                )
            })
        },
        &|g, o| {
            g.run_slang(|c| {
                kernels_slang::linear::moe::combine::sigmoid_gate_add(
                    c,
                    st(b16(x, rows, w)),
                    st(b16(sh, rows, w)),
                    st(b16(gt, rows, 1)),
                    st(b16(o[0].0, rows, w)),
                )
            })
        },
    );
}
