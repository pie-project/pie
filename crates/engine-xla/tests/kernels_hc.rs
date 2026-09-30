//! The hyper-connection family against host references.

use dtype::Dtype;
use engine_xla::bench::{Bench, assert_close, round_bf16};
use kernels_xla::elemwise::hc;

fn data(n: usize, seed: u32) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let h = (i as u32)
                .wrapping_mul(2_654_435_761)
                .wrapping_add(seed.wrapping_mul(40503));
            round_bf16(((h >> 8) % 2000) as f32 / 1000.0 - 1.0)
        })
        .collect()
}

fn sigmoid(v: f32) -> f32 {
    1.0 / (1.0 + (-v).exp())
}

const M: usize = 3;
const H: usize = 40;
const ROWS: usize = 5;

#[test]
fn expand_mix_inject_and_ple_gate_answer_the_host() {
    let x = data(ROWS * H, 1);
    let gates = data(ROWS * M * H, 2);
    let normed = data(ROWS * M * H, 3);
    let o = data(ROWS * H, 4);
    let glog: Vec<f32> = data(ROWS * M, 5)
        .iter()
        .map(|v| round_bf16(v * 3.0))
        .collect();
    let hyper = data(ROWS * M * H, 6);
    let key = data(ROWS * M * H, 7);
    let query: Vec<f32> = data(ROWS * M * H, 8);
    let mut b = Bench::new();
    let (r, h, mh) = (ROWS as u32, H as u32, (M * H) as u32);
    let xt = b.bf16(r, h, &x);
    let ex = b.zeros(Dtype::Bf16, r, mh);
    let gt = b.bf16(r, mh, &gates);
    let nt = b.bf16(r, mh, &normed);
    let mx = b.zeros(Dtype::Bf16, r, h);
    let ot = b.bf16(r, h, &o);
    let gl = b.bf16(r, M as u32, &glog);
    let hy = b.bf16(r, mh, &hyper);
    let kt = b.bf16(r, mh, &key);
    let qt = b.bf16(r, mh, &query);
    let pg = b.zeros(Dtype::Bf16, r, mh);
    if !b
        .run(|ctx| {
            hc::expand(ctx, xt, M as u32, ex)?;
            hc::mix(ctx, gt, nt, M as u32, mx)?;
            hc::inject(ctx, ot, gl, M as u32, hy)?;
            hc::ple_gate(ctx, kt, qt, xt, M as u32, pg)
        })
        .unwrap()
    {
        return;
    }
    let mut w_ex = vec![0f32; ROWS * M * H];
    let mut w_mx = vec![0f32; ROWS * H];
    let mut w_hy = hyper.clone();
    let mut w_pg = vec![0f32; ROWS * M * H];
    for n in 0..ROWS {
        for s in 0..M {
            let g = 2.0 * sigmoid(glog[n * M + s] / M as f32);
            let at = (n * M + s) * H;
            let dot: f32 =
                (0..H).map(|i| key[at + i] * query[at + i]).sum::<f32>() / (H as f32).sqrt();
            let mag = dot.abs().max(1e-6).sqrt();
            let damped = if dot > 0.0 {
                mag
            } else if dot < 0.0 {
                -mag
            } else {
                0.0
            };
            let pgate = sigmoid(damped);
            for k in 0..H {
                w_ex[at + k] = x[n * H + k];
                w_hy[at + k] = round_bf16(hyper[at + k] + g * o[n * H + k]);
                w_pg[at + k] = round_bf16(pgate * x[n * H + k]);
            }
        }
        for k in 0..H {
            let acc: f32 = (0..M)
                .map(|s| normed[(n * M + s) * H + k] * sigmoid(gates[(n * M + s) * H + k]))
                .sum();
            w_mx[n * H + k] = round_bf16(acc / M as f32);
        }
    }
    assert_close(&b.read_f32(ex), &w_ex, 0.0, 0.0);
    assert_close(&b.read_f32(mx), &w_mx, 1e-2, 1e-2);
    assert_close(&b.read_f32(hy), &w_hy, 1e-2, 1e-2);
    assert_close(&b.read_f32(pg), &w_pg, 1e-2, 1e-2);
}

fn sinkhorn(logits: &[f32], m: usize, iters: u32, eps: f32) -> Vec<f32> {
    let mut c = vec![0f32; m * m];
    for i in 0..m {
        let row = &logits[i * m..(i + 1) * m];
        let mx = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let e: Vec<f32> = row.iter().map(|v| (v - mx).exp()).collect();
        let s: f32 = e.iter().sum();
        for j in 0..m {
            c[i * m + j] = e[j] / s + eps;
        }
    }
    let col = |c: &mut Vec<f32>| {
        for j in 0..m {
            let s: f32 = (0..m).map(|i| c[i * m + j]).sum::<f32>() + eps;
            for i in 0..m {
                c[i * m + j] /= s;
            }
        }
    };
    let row = |c: &mut Vec<f32>| {
        for i in 0..m {
            let s: f32 = (0..m).map(|j| c[i * m + j]).sum::<f32>() + eps;
            for j in 0..m {
                c[i * m + j] /= s;
            }
        }
    };
    col(&mut c);
    for _ in 1..iters {
        row(&mut c);
        col(&mut c);
    }
    c
}

#[test]
fn the_mhc_chain_norms_projects_gates_folds_and_collapses() {
    let mix_hc = 2 * M + M * M;
    let streams = data(ROWS * M * H, 10);
    let fn_w: Vec<f32> = data(mix_hc * M * H, 11).iter().map(|v| v * 0.3).collect();
    let fn_c: Vec<f32> = data(M * M * H, 12).iter().map(|v| v * 0.3).collect();
    let scale = [0.8f32, 1.1, 0.9];
    let base: Vec<f32> = data(mix_hc, 13).iter().map(|v| v * 0.5).collect();
    let x = data(ROWS * H, 14);
    let (gate_eps, alpha, iters) = (1e-6f32, 2.0f32, 5u32);
    let mut b = Bench::new();
    let (r, h, mh) = (ROWS as u32, H as u32, (M * H) as u32);
    let st = b.bf16(r, mh, &streams);
    let normed = b.zeros(Dtype::F32, r, mh);
    let fw = b.f32(mix_hc as u32, mh, &fn_w);
    let mixes = b.zeros(Dtype::F32, r, mix_hc as u32);
    let sc = b.f32(1, 3, &scale);
    let bs = b.f32(1, mix_hc as u32, &base);
    let li = b.zeros(Dtype::Bf16, r, h);
    let post = b.zeros(Dtype::F32, r, M as u32);
    let comb = b.zeros(Dtype::F32, r, (M * M) as u32);
    let xt = b.bf16(r, h, &x);
    let folded = b.zeros(Dtype::Bf16, r, mh);
    let fc = b.f32(M as u32, mh, &fn_c);
    let cm = b.zeros(Dtype::F32, r, M as u32);
    let col = b.zeros(Dtype::Bf16, r, h);
    if !b
        .run(|ctx| {
            hc::rmsnorm_f32(ctx, st, 1e-6, normed)?;
            hc::project(ctx, normed, fw, M as u32, mixes)?;
            hc::gates(
                ctx, mixes, st, sc, bs, M as u32, gate_eps, alpha, iters, li, post, comb,
            )?;
            hc::fold(ctx, xt, st, post, comb, folded)?;
            hc::project(ctx, normed, fc, M as u32, cm)?;
            hc::collapse(ctx, cm, st, sc, bs, M as u32, gate_eps, col)
        })
        .unwrap()
    {
        return;
    }
    let got_normed = b.read_f32(normed);
    let got_mixes = b.read_f32(mixes);
    let got_post = b.read_f32(post);
    let got_comb = b.read_f32(comb);
    let mut w_normed = vec![0f32; ROWS * M * H];
    let mut w_mixes = vec![0f32; ROWS * mix_hc];
    let mut w_li = vec![0f32; ROWS * H];
    let mut w_post = vec![0f32; ROWS * M];
    let mut w_comb = vec![0f32; ROWS * M * M];
    let mut w_fold = vec![0f32; ROWS * M * H];
    let mut w_cm = vec![0f32; ROWS * M];
    let mut w_col = vec![0f32; ROWS * H];
    let fc_w = &fn_c;
    for n in 0..ROWS {
        let row = &streams[n * M * H..(n + 1) * M * H];
        let inv = 1.0 / (row.iter().map(|v| v * v).sum::<f32>() / (M * H) as f32 + 1e-6).sqrt();
        let nr: Vec<f32> = row.iter().map(|v| v * inv).collect();
        w_normed[n * M * H..(n + 1) * M * H].copy_from_slice(&nr);
        for o in 0..mix_hc {
            w_mixes[n * mix_hc + o] = (0..M * H).map(|d| nr[d] * fn_w[o * M * H + d]).sum();
        }
        for o in 0..M {
            w_cm[n * M + o] = (0..M * H).map(|d| nr[d] * fc_w[o * M * H + d]).sum();
        }
        // Gates read the device's mixes, so this test checks each stage on
        // its own inputs.
        let mix = &got_mixes[n * mix_hc..(n + 1) * mix_hc];
        let pre: Vec<f32> = (0..M)
            .map(|i| sigmoid(mix[i] * scale[0] + base[i]) + gate_eps)
            .collect();
        for i in 0..M {
            w_post[n * M + i] = sigmoid(mix[M + i] * scale[1] + base[M + i]) * alpha;
        }
        let logits: Vec<f32> = (0..M * M)
            .map(|t| mix[2 * M + t] * scale[2] + base[2 * M + t])
            .collect();
        let c = sinkhorn(&logits, M, iters, gate_eps);
        w_comb[n * M * M..(n + 1) * M * M].copy_from_slice(&c);
        let dpost = &got_post[n * M..(n + 1) * M];
        let dcomb = &got_comb[n * M * M..(n + 1) * M * M];
        for k in 0..H {
            w_li[n * H + k] = round_bf16((0..M).map(|i| pre[i] * row[i * H + k]).sum());
            for j in 0..M {
                let mut acc = dpost[j] * x[n * H + k];
                for i in 0..M {
                    acc += dcomb[i * M + j] * row[i * H + k];
                }
                w_fold[n * M * H + j * H + k] = round_bf16(acc);
            }
        }
        let cmd = &b.read_f32(cm)[n * M..(n + 1) * M];
        let g: Vec<f32> = (0..M)
            .map(|i| sigmoid(cmd[i] * scale[0] + base[i]) + gate_eps)
            .collect();
        for k in 0..H {
            w_col[n * H + k] = round_bf16((0..M).map(|i| g[i] * row[i * H + k]).sum());
        }
    }
    assert_close(&got_normed, &w_normed, 1e-5, 1e-4);
    assert_close(&got_mixes, &w_mixes, 1e-4, 1e-3);
    assert_close(&b.read_f32(cm), &w_cm, 1e-4, 1e-3);
    assert_close(&got_post, &w_post, 1e-5, 1e-4);
    assert_close(&got_comb, &w_comb, 1e-5, 1e-4);
    assert_close(&b.read_f32(li), &w_li, 1e-2, 1e-2);
    assert_close(&b.read_f32(folded), &w_fold, 1e-2, 1e-2);
    assert_close(&b.read_f32(col), &w_col, 1e-2, 1e-2);
}
