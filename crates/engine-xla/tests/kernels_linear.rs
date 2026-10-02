//! The dense linear family (gemm + fused CUDA forms), the MLP activations and
//! the LoRA correction against host references.

use dtype::Dtype;
use engine_xla::bench::{Bench, assert_close, round_bf16};
use kernels_xla::linear::{gemm, lora, mlp};

fn data(n: usize, seed: u32) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let h = (i as u32)
                .wrapping_mul(2_654_435_761)
                .wrapping_add(seed.wrapping_mul(40503))
                .rotate_left(7)
                .wrapping_mul(2_246_822_519);
            round_bf16(((h >> 8) % 2000) as f32 / 1000.0 - 1.0)
        })
        .collect()
}

/// Unrounded f32 values, for f32 activations.
fn data_f32(n: usize, seed: u32) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let h = (i as u32)
                .wrapping_mul(2_654_435_761)
                .wrapping_add(seed.wrapping_mul(40503))
                .rotate_left(11)
                .wrapping_mul(2_246_822_519);
            (h % 2_000_001) as f32 / 1_000_000.0 - 1.0
        })
        .collect()
}

/// `x · wᵀ`, `x: [m, k]`, `w: [n, k]`, accumulated in f64.
fn gemm_ref(x: &[f32], w: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut y = vec![0.0; m * n];
    for r in 0..m {
        for c in 0..n {
            let mut acc = 0.0f64;
            for i in 0..k {
                acc += f64::from(x[r * k + i]) * f64::from(w[c * k + i]);
            }
            y[r * n + c] = acc as f32;
        }
    }
    y
}

fn gelu_tanh(v: f32) -> f32 {
    let k = 0.797_884_6_f32;
    0.5 * v * (1.0 + (k * (v + 0.044715 * v * v * v)).tanh())
}

fn silu(g: f32) -> f32 {
    g / (1.0 + (-g).exp())
}

#[test]
fn dense_projections_and_their_fused_forms_answer_the_host() {
    let (m, n, k) = (5usize, 10usize, 384usize);
    let xs = data(m * k, 1);
    let ws = data(n * k, 2);
    let bias = data(n, 3);
    let i = 6usize;
    let gw = data(2 * i * k, 4);
    let (heads, d_rel, extent) = (3usize, 12usize, 20usize);
    let rx = data(m * heads * d_rel, 5);
    let rw = data(d_rel * extent, 6);
    let cap = 5.0f32;

    let mut b = Bench::new();
    let x = b.bf16(m as u32, k as u32, &xs);
    let w = b.bf16(n as u32, k as u32, &ws);
    let bi = b.bf16(1, n as u32, &bias);
    let g = b.bf16(2 * i as u32, k as u32, &gw);
    let y = b.zeros(Dtype::Bf16, m as u32, n as u32);
    let head = b.zeros(Dtype::Bf16, m as u32, n as u32);
    let biased = b.zeros(Dtype::Bf16, m as u32, n as u32);
    let capped = b.zeros(Dtype::Bf16, m as u32, n as u32);
    let packed = b.zeros(Dtype::Bf16, m as u32, 2 * i as u32);
    let geglu = b.zeros(Dtype::Bf16, m as u32, i as u32);
    let rxt = b.bf16(m as u32, (heads * d_rel) as u32, &rx);
    let rwt = b.bf16(d_rel as u32, extent as u32, &rw);
    let rel = b.zeros(Dtype::F32, m as u32, (heads * extent) as u32);
    if !b
        .run(|ctx| {
            gemm::matmul(ctx, x, w, y)?;
            gemm::lm_head(ctx, x, w, head)?;
            gemm::matmul_bias(ctx, x, w, bi, biased)?;
            gemm::lm_head_softcap(ctx, x, w, cap, capped)?;
            gemm::matmul_geglu(ctx, x, g, i as u32, packed, geglu)?;
            gemm::rel_bias(
                ctx,
                rxt,
                rwt,
                heads as u32,
                d_rel as u32,
                extent as u32,
                rel,
            )
        })
        .unwrap()
    {
        return;
    }
    let want = gemm_ref(&xs, &ws, m, n, k);
    assert_close(&b.read_f32(y), &want, 1e-2, 1e-2);
    assert_close(&b.read_f32(head), &want, 1e-2, 1e-2);
    let want_b: Vec<f32> = want
        .iter()
        .enumerate()
        .map(|(at, v)| v + bias[at % n])
        .collect();
    assert_close(&b.read_f32(biased), &want_b, 1e-2, 1e-2);
    let want_c: Vec<f32> = want.iter().map(|v| cap * (v / cap).tanh()).collect();
    assert!(
        want.iter().any(|v| v.abs() > cap),
        "the cap bites somewhere"
    );
    assert_close(&b.read_f32(capped), &want_c, 1e-2, 1e-2);

    let want_p = gemm_ref(&xs, &gw, m, 2 * i, k);
    let got_p = b.read_f32(packed);
    assert_close(&got_p, &want_p, 1e-2, 1e-2);
    let want_g: Vec<f32> = (0..m * i)
        .map(|at| {
            let (r, c) = (at / i, at % i);
            gelu_tanh(got_p[r * 2 * i + c]) * got_p[r * 2 * i + i + c]
        })
        .collect();
    assert_close(&b.read_f32(geglu), &want_g, 1e-2, 1e-2);

    let mut want_r = vec![0.0f32; m * heads * extent];
    for r in 0..m {
        for h in 0..heads {
            for d in 0..extent {
                let mut acc = 0.0f64;
                for j in 0..d_rel {
                    acc +=
                        f64::from(rx[(r * heads + h) * d_rel + j]) * f64::from(rw[j * extent + d]);
                }
                want_r[(r * heads + h) * extent + d] = acc as f32;
            }
        }
    }
    assert_close(&b.read_f32(rel), &want_r, 1e-5, 1e-5);
}

#[test]
fn an_f32_activation_is_contracted_at_f32() {
    let (m, n, k) = (3usize, 6usize, 200usize);
    let xs = data_f32(m * k, 7);
    let ws = data(n * k, 8);
    let wf = data_f32(n * k, 9);
    let mut b = Bench::new();
    let x = b.f32(m as u32, k as u32, &xs);
    let w = b.bf16(n as u32, k as u32, &ws);
    let w32 = b.f32(n as u32, k as u32, &wf);
    let y = b.zeros(Dtype::F32, m as u32, n as u32);
    let z = b.zeros(Dtype::F32, m as u32, n as u32);
    if !b
        .run(|ctx| {
            gemm::act_x_wt(ctx, "linear.matmul", x, w, y)?;
            gemm::matmul(ctx, x, w32, z)
        })
        .unwrap()
    {
        return;
    }
    assert_close(&b.read_f32(y), &gemm_ref(&xs, &ws, m, n, k), 1e-5, 1e-5);
    assert_close(&b.read_f32(z), &gemm_ref(&xs, &wf, m, n, k), 1e-5, 1e-5);
}

#[test]
fn every_mlp_activation_answers_the_host() {
    let (rows, i) = (3usize, 70usize);
    let scale = |v: Vec<f32>, s: f32| v.into_iter().map(|x| round_bf16(x * s)).collect::<Vec<_>>();
    let ps = scale(data(rows * 2 * i, 10), 8.0);
    let gs = scale(data(rows * i, 11), 8.0);
    let us = scale(data(rows * i, 12), 8.0);
    let (limit, alpha, beta, up_cap) = (3.0f32, 1.702f32, 2.5f32, 4.0f32);

    let mut b = Bench::new();
    let packed = b.bf16(rows as u32, 2 * i as u32, &ps);
    let gate = b.bf16(rows as u32, i as u32, &gs);
    let up = b.bf16(rows as u32, i as u32, &us);
    let outs: Vec<_> = (0..9)
        .map(|_| b.zeros(Dtype::Bf16, rows as u32, i as u32))
        .collect();
    let iu = i as u32;
    if !b
        .run(|ctx| {
            mlp::swiglu(ctx, packed, iu, outs[0])?;
            mlp::swiglu_clamp(ctx, packed, iu, limit, outs[1])?;
            mlp::swiglu_clamp_alpha(ctx, packed, iu, limit, alpha, outs[2])?;
            mlp::swiglu_clamp_split(ctx, gate, up, limit, outs[3])?;
            mlp::geglu_tanh(ctx, gate, up, outs[4])?;
            mlp::gelu_tanh(ctx, gate, outs[5])?;
            mlp::geglu_tanh_packed(ctx, packed, iu, outs[6])?;
            mlp::situ(ctx, packed, iu, beta, Some(up_cap), outs[7])?;
            mlp::situ(ctx, packed, iu, beta, None, outs[8])
        })
        .unwrap()
    {
        return;
    }
    let packed_ref = |f: &dyn Fn(f32, f32) -> f32| -> Vec<f32> {
        (0..rows * i)
            .map(|at| {
                let (r, c) = (at / i, at % i);
                round_bf16(f(ps[r * 2 * i + c], ps[r * 2 * i + i + c]))
            })
            .collect()
    };
    let split_ref = |f: &dyn Fn(f32, f32) -> f32| -> Vec<f32> {
        gs.iter()
            .zip(&us)
            .map(|(&g, &u)| round_bf16(f(g, u)))
            .collect()
    };
    let clamp = |g: f32, u: f32| (g.min(limit), u.clamp(-limit, limit));
    let pie_tanh = |x: f32| x.clamp(-16.0, 16.0).tanh();
    let situ = |cap: Option<f32>| {
        move |g: f32, u: f32| {
            let sg = beta * pie_tanh(g / beta) / (1.0 + (-g).exp());
            let u = match cap {
                Some(c) if c > 0.0 => c * pie_tanh(u / c),
                _ => u,
            };
            sg * u
        }
    };
    let wants = [
        packed_ref(&|g, u| silu(g) * u),
        packed_ref(&|g, u| {
            let (g, u) = clamp(g, u);
            silu(g) * u
        }),
        packed_ref(&|g, u| {
            let (g, u) = clamp(g, u);
            g / (1.0 + (-alpha * g).exp()) * (u + 1.0)
        }),
        split_ref(&|g, u| {
            let (g, u) = clamp(g, u);
            silu(g) * u
        }),
        split_ref(&|g, u| gelu_tanh(g) * u),
        split_ref(&|g, _| gelu_tanh(g)),
        packed_ref(&|g, u| gelu_tanh(g) * u),
        packed_ref(&situ(Some(up_cap))),
        packed_ref(&situ(None)),
    ];
    for (at, (out, want)) in outs.iter().zip(&wants).enumerate() {
        let got = b.read_f32(*out);
        eprintln!("mlp entry {at}");
        assert_close(&got, want, 2e-2, 1e-2);
    }
}

#[test]
fn a_lora_correction_adds_each_rows_own_adapter() {
    let (rows, n_in, n_out, rank, adapters) = (6usize, 96usize, 40usize, 8usize, 3usize);
    let xs = data(rows * n_in, 20);
    let a = data(adapters * rank * n_in, 21);
    let bb = data(adapters * n_out * rank, 22);
    let ys = data(rows * n_out, 23);
    let routes = [0i32, 2, -1, 1, 2, 3];
    let mut b = Bench::new();
    let x = b.bf16(rows as u32, n_in as u32, &xs);
    let at = b.bf16(adapters as u32, (rank * n_in) as u32, &a);
    let bt = b.bf16(adapters as u32, (n_out * rank) as u32, &bb);
    let rt = b.i32(rows as u32, 1, &routes);
    let y = b.bf16(rows as u32, n_out as u32, &ys);
    if !b.run(|ctx| lora::correct(ctx, x, at, bt, rt, y)).unwrap() {
        return;
    }
    let mut want = ys.clone();
    for r in 0..rows {
        let ad = routes[r];
        if ad < 0 || ad as usize >= adapters {
            continue;
        }
        let ad = ad as usize;
        let mut waist = vec![0.0f64; rank];
        for (i, t) in waist.iter_mut().enumerate() {
            for c in 0..n_in {
                *t += f64::from(a[(ad * rank + i) * n_in + c]) * f64::from(xs[r * n_in + c]);
            }
        }
        for n in 0..n_out {
            let mut acc = 0.0f64;
            for (i, t) in waist.iter().enumerate() {
                acc += f64::from(bb[(ad * n_out + n) * rank + i]) * t;
            }
            want[r * n_out + n] = round_bf16(ys[r * n_out + n] + acc as f32);
        }
    }
    let got = b.read_f32(y);
    // Rows without an adapter keep their bits.
    for r in [2usize, 5] {
        assert_eq!(
            &got[r * n_out..(r + 1) * n_out],
            &ys[r * n_out..(r + 1) * n_out]
        );
    }
    assert_close(&got, &want, 2e-2, 1e-2);
}
