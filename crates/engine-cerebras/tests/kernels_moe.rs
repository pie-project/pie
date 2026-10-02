//! The mixture-of-experts ops against host references: the top-k softmax
//! router, the routed matmul (on one PE and over a lane plan of routed
//! pairs), the weighted sum of the routed rows, and the sigmoid gate add.

mod common;

use common::data;
use engine_cerebras::bench::{Bench, assert_close, round_bf16};
use kernels_cerebras::linear::moe;

fn softmax_topk(logits: &[f32], experts: usize, top_k: usize) -> (Vec<i32>, Vec<f32>) {
    let mut picked = Vec::new();
    for _ in 0..top_k {
        let mut best: Option<(usize, f32)> = None;
        for (i, v) in logits[..experts].iter().enumerate() {
            if v.is_nan() || picked.contains(&i) {
                continue;
            }
            if best.is_none_or(|(_, b)| *v > b) {
                best = Some((i, *v));
            }
        }
        match best {
            Some((i, _)) => picked.push(i),
            None => break,
        }
    }
    let chosen: Vec<f32> = picked.iter().map(|i| logits[*i]).collect();
    let m = chosen.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let z: f32 = chosen.iter().map(|v| (v - m).exp()).sum();
    let mut routes = vec![-1i32; top_k];
    let mut weights = vec![0f32; top_k];
    for (s, i) in picked.iter().enumerate() {
        routes[s] = *i as i32;
        weights[s] = (chosen[s] - m).exp() / z;
    }
    (routes, weights)
}

#[test]
fn the_router_picks_the_top_k_and_softmaxes_them() {
    let (rows, experts, top_k) = (4u32, 6usize, 3usize);
    let mut logits = data(rows as usize * 8, 70); // two spare columns a row
    logits[8 + 2] = f32::NAN;
    let mut b = Bench::new();
    let l = b.f32(rows, 8, &logits);
    let r = b.zeros(dtype::Dtype::I32, rows, top_k as u32);
    let w = b.zeros(dtype::Dtype::F32, rows, top_k as u32);
    if !b
        .run(|ctx| moe::topk_softmax(ctx, l, experts as u32, top_k as u32, r, w))
        .unwrap()
    {
        return;
    }
    let mut want_r = Vec::new();
    let mut want_w = Vec::new();
    for t in 0..rows as usize {
        let (rr, ww) = softmax_topk(&logits[t * 8..(t + 1) * 8], experts, top_k);
        want_r.extend(rr);
        want_w.extend(ww);
    }
    assert_eq!(b.read_i32(r), want_r);
    assert_close(&b.read_f32(w), &want_w, 1e-5, 1e-5);
}

/// `tokens` rows through `experts` experts of `n x k`, `top_k` each: select,
/// weighted sum, then the shared expert gated in.
fn routed_case(tokens: usize, experts: usize, top_k: usize, k: usize, n: usize, seed: u32) {
    let xs: Vec<f32> = data(tokens * k, seed)
        .iter()
        .map(|v| round_bf16(*v))
        .collect();
    let bank: Vec<f32> = data(experts * n * k, seed + 1)
        .iter()
        .map(|v| round_bf16(*v))
        .collect();
    let pairs = tokens * top_k;
    let routes: Vec<i32> = (0..pairs)
        .map(|p| {
            if p == 3 {
                -1
            } else {
                ((p * 5 + 1) % experts) as i32
            }
        })
        .collect();
    let weights: Vec<f32> = data(pairs, seed + 2)
        .iter()
        .map(|v| v.abs() + 0.1)
        .collect();
    let shared: Vec<f32> = data(tokens * n, seed + 3)
        .iter()
        .map(|v| round_bf16(*v))
        .collect();
    let gate: Vec<f32> = data(tokens, seed + 4)
        .iter()
        .map(|v| round_bf16(v * 2.0))
        .collect();
    let mut b = Bench::new();
    let x = b.bf16(tokens as u32, k as u32, &xs);
    let bt = b.bf16(experts as u32, (n * k) as u32, &bank);
    let rt = b.i32(tokens as u32, top_k as u32, &routes);
    let wt = b.f32(tokens as u32, top_k as u32, &weights);
    let sel = b.zeros(dtype::Dtype::Bf16, pairs as u32, n as u32);
    let summed = b.zeros(dtype::Dtype::Bf16, tokens as u32, n as u32);
    let st = b.bf16(tokens as u32, n as u32, &shared);
    let gt = b.bf16(tokens as u32, 1, &gate);
    let y = b.zeros(dtype::Dtype::Bf16, tokens as u32, n as u32);
    if !b
        .run(|ctx| {
            moe::matmul_select(ctx, x, bt, rt, sel)?;
            moe::weighted_sum(ctx, sel, wt, summed)?;
            moe::sigmoid_gate_add(ctx, summed, st, gt, y)
        })
        .unwrap()
    {
        return;
    }
    // Host reference.
    let mut want_sel = vec![0f32; pairs * n];
    for p in 0..pairs {
        let e = routes[p];
        if e < 0 {
            continue;
        }
        let t = p / top_k;
        for o in 0..n {
            let mut acc = 0f64;
            for c in 0..k {
                acc += f64::from(bank[(e as usize * n + o) * k + c]) * f64::from(xs[t * k + c]);
            }
            want_sel[p * n + o] = round_bf16(acc as f32);
        }
    }
    let got_sel = b.read_f32(sel);
    assert_close(&got_sel, &want_sel, 2e-2, 2e-2);
    let mut want_sum = vec![0f32; tokens * n];
    for t in 0..tokens {
        for o in 0..n {
            let mut acc = 0f32;
            for s in 0..top_k {
                acc += weights[t * top_k + s] * got_sel[(t * top_k + s) * n + o];
            }
            want_sum[t * n + o] = round_bf16(acc);
        }
    }
    let got_sum = b.read_f32(summed);
    assert_close(&got_sum, &want_sum, 2e-2, 2e-2);
    let want_y: Vec<f32> = (0..tokens * n)
        .map(|i| {
            let t = i / n;
            let g = 1.0 / (1.0 + (-gate[t]).exp());
            round_bf16(got_sum[i] + g * shared[i])
        })
        .collect();
    assert_close(&b.read_f32(y), &want_y, 2e-2, 2e-2);
}

#[test]
fn routed_experts_select_sum_and_gate_on_one_pe() {
    routed_case(3, 4, 2, 16, 12, 80);
}

/// Experts past a PE (4 x 64 x 128) spread the routed pairs over a lane
/// plan, a PE holding a block of its pairs' experts, that slice of x and
/// its columns of y as a partial the host adds.
#[test]
fn routed_experts_spread_over_pes() {
    routed_case(4, 4, 2, 128, 64, 90);
}

/// The identity routing, then each slice of a row against its own expert:
/// slice `g` reads expert `g`; a route outside the experts leaves its
/// slice alone.
#[test]
fn a_grouped_matmul_multiplies_each_slice_by_its_routed_expert() {
    use kernels_cerebras::linear::moe;
    let (rows, groups, k, n, experts) = (3u32, 4u32, 6u32, 5u32, 4u32);
    let xs = data((rows * groups * k) as usize, 121);
    let ws = data((experts * n * k) as usize, 122);
    let mut b = Bench::new();
    let x = b.bf16(rows, groups * k, &xs);
    let w = b.bf16(experts * n, k, &ws);
    let routes = b.zeros(dtype::Dtype::I32, rows, groups);
    let y = b.zeros(dtype::Dtype::Bf16, rows, groups * n);
    // A second routing with one route outside the experts.
    let odd = b.i32(rows, groups, &[3, 2, 1, 0, 0, 0, 0, 0, -1, 1, 1, 9]);
    let z = b.bf16(rows, groups * n, &vec![7.0; (rows * groups * n) as usize]);
    let ran = b
        .run(|ctx| {
            moe::group_routes(ctx, groups, routes)?;
            moe::matmul_grouped(ctx, x, w, routes, groups, y)?;
            moe::matmul_grouped(ctx, x, w, odd, groups, z)
        })
        .unwrap();
    eprintln!("grouped matmul ran on the simulator: {ran}");
    if !ran {
        return;
    }
    let want_routes: Vec<i32> = (0..rows * groups).map(|i| (i % groups) as i32).collect();
    assert_eq!(b.read_i32(routes), want_routes);
    let product = |r: usize, g: usize, e: usize| -> Vec<f32> {
        (0..n as usize)
            .map(|j| {
                let dot: f32 = (0..k as usize)
                    .map(|i| {
                        xs[(r * groups as usize + g) * k as usize + i]
                            * ws[(e * n as usize + j) * k as usize + i]
                    })
                    .sum();
                round_bf16(dot)
            })
            .collect()
    };
    let mut want_y = Vec::new();
    for r in 0..rows as usize {
        for g in 0..groups as usize {
            want_y.extend(product(r, g, g));
        }
    }
    assert_close(&b.read_f32(y), &want_y, 1e-2, 1e-2);
    let odd_routes = [3i32, 2, 1, 0, 0, 0, 0, 0, -1, 1, 1, 9];
    let mut want_z = Vec::new();
    for r in 0..rows as usize {
        for g in 0..groups as usize {
            let e = odd_routes[r * groups as usize + g];
            if e < 0 || e as u32 >= experts {
                want_z.extend(std::iter::repeat_n(7.0, n as usize));
            } else {
                want_z.extend(product(r, g, e as usize));
            }
        }
    }
    assert_close(&b.read_f32(z), &want_z, 1e-2, 1e-2);
}
