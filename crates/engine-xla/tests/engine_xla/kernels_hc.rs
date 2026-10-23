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
            engine_xla::lower::lower(ctx, |l| {
                let x_v = l.read(xt)?;
                let o_y = ex;
                let v = hc::expand(l.f(), x_v, M as u32, o_y.landing())?;
                l.land(o_y, v)
            })?;
            engine_xla::lower::lower(ctx, |l| {
                let gates_v = l.read(gt)?;
                let normed_v = l.read(nt)?;
                let o_y = mx;
                let v = hc::mix(l.f(), gates_v, normed_v, M as u32, o_y.landing())?;
                l.land(o_y, v)
            })?;
            engine_xla::lower::lower(ctx, |l| {
                let gates_v = l.read(gl)?;
                let o_v = l.read(ot)?;
                let hyper_v = l.read(hy)?;
                let o_hyper = hy;
                let v = hc::inject(l.f(), o_v, gates_v, M as u32, hyper_v)?;
                l.land(o_hyper, v)
            })?;
            engine_xla::lower::lower(ctx, |l| {
                let value_v = l.read(xt)?;
                let key_v = l.read(kt)?;
                let query_v = l.read(qt)?;
                let o_y = pg;
                let v = hc::ple_gate(l.f(), key_v, query_v, value_v, M as u32, o_y.landing())?;
                l.land(o_y, v)
            })
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
            engine_xla::lower::lower(ctx, |l| {
                let streams_v = l.read(st)?;
                let o_y = normed;
                let v = hc::rmsnorm_f32(l.f(), streams_v, 1e-6, o_y.landing())?;
                l.land(o_y, v)
            })?;
            engine_xla::lower::lower(ctx, |l| {
                let normed_v = l.read(normed)?;
                let hc_fn_v = l.read(fw)?;
                let o_mixes = mixes;
                let v = hc::project(l.f(), normed_v, hc_fn_v, M as u32, o_mixes.landing())?;
                l.land(o_mixes, v)
            })?;
            engine_xla::lower::lower(ctx, |l| {
                let normed_v = l.read(mixes)?;
                let scale_v = l.read(sc)?;
                let base_v = l.read(bs)?;
                let streams_v = l.read(st)?;
                let o_x = li;
                let o_post_mix = post;
                let o_comb_mix = comb;
                let (v0, v1, v2) = hc::gates(
                    l.f(),
                    normed_v,
                    streams_v,
                    scale_v,
                    base_v,
                    M as u32,
                    gate_eps,
                    alpha,
                    iters,
                    o_x.landing(),
                    o_post_mix.landing(),
                    o_comb_mix.landing(),
                )?;
                l.land(o_post_mix, v0)?;
                l.land(o_comb_mix, v1)?;
                l.land(o_x, v2)?;
                Ok(())
            })?;
            engine_xla::lower::lower(ctx, |l| {
                let x_v = l.read(xt)?;
                let post_mix_v = l.read(post)?;
                let comb_mix_v = l.read(comb)?;
                let streams_v = l.read(st)?;
                let o_y = folded;
                let v = hc::fold(l.f(), x_v, streams_v, post_mix_v, comb_mix_v, o_y.landing())?;
                l.land(o_y, v)
            })?;
            engine_xla::lower::lower(ctx, |l| {
                let normed_v = l.read(normed)?;
                let hc_fn_v = l.read(fc)?;
                let o_mixes = cm;
                let v = hc::project(l.f(), normed_v, hc_fn_v, M as u32, o_mixes.landing())?;
                l.land(o_mixes, v)
            })?;
            engine_xla::lower::lower(ctx, |l| {
                let mixes_v = l.read(cm)?;
                let streams_v = l.read(st)?;
                let scale_v = l.read(sc)?;
                let base_v = l.read(bs)?;
                let o_y = col;
                let v = hc::collapse(
                    l.f(),
                    mixes_v,
                    streams_v,
                    scale_v,
                    base_v,
                    M as u32,
                    gate_eps,
                    o_y.landing(),
                )?;
                l.land(o_y, v)
            })
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
