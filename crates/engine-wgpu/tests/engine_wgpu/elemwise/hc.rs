//! The hyper-connection kernels on the first wgpu adapter against an f64
//! host reference: expand, the f32 stream norm, the gates
//! (Sinkhorn-balanced), fold, collapse, and the trunk's mix, inject and PLE
//! gate.

use crate::gpu::{Gpu, landed, noise};
use kernels_slang::Tensor;
use poem_ir::Dtype;

const N: usize = 3;
const M: usize = 4;
const H: usize = 64;

fn sigmoid(v: f64) -> f64 {
    1.0 / (1.0 + (-v).exp())
}

fn f64s(v: &[f32]) -> Vec<f64> {
    v.iter().map(|x| f64::from(*x)).collect()
}

#[test]
fn the_hyper_connections_answer_the_reference() {
    let Some(mut gpu) = Gpu::open() else {
        eprintln!("not asked: no device");
        return;
    };
    let b = |buf: u32, w: usize| Tensor::new(buf, N as u32, w as u32, Dtype::Bf16);
    let f = |buf: u32, rows: usize, w: usize| Tensor::new(buf, rows as u32, w as u32, Dtype::F32);
    let wide = M * H;
    let mix_hc = 2 * M + M * M;
    let (eps, alpha, sinkhorn) = (1e-6f32, 0.9f32, 3u32);

    let (x, xr) = gpu.up(&noise(1, N * H));
    let (streams, sr) = gpu.up(&noise(2, N * wide));
    let (xr, sr) = (f64s(&xr), f64s(&sr));

    // expand
    let want: Vec<f64> = (0..N * wide).map(|i| xr[(i / wide) * H + i % H]).collect();
    landed(
        &mut gpu,
        "hc expand",
        &|g| vec![(g.zeros(N * wide), N * wide, false)],
        &[want],
        &|g, o| {
            g.run(|c| kernels_slang::elemwise::hc::expand(c, b(x, H), M as u32, b(o[0].0, wide)))
        },
    );

    // the f32 stream norm, over the whole wide row
    let want: Vec<f64> = (0..N * wide)
        .map(|i| {
            let row = &sr[(i / wide) * wide..][..wide];
            let ms = row.iter().map(|v| v * v).sum::<f64>() / wide as f64;
            sr[i] / (ms + f64::from(eps)).sqrt()
        })
        .collect();
    landed(
        &mut gpu,
        "hc rmsnorm f32",
        &|g| vec![(g.zeros(2 * N * wide), N * wide, true)],
        &[want],
        &|g, o| {
            g.run(|c| {
                kernels_slang::elemwise::hc::rmsnorm_f32(
                    c,
                    b(streams, wide),
                    eps,
                    f(o[0].0, N, wide),
                )
            })
        },
    );

    // the mix row a layer's projection lands (elemwise/hc_project.rs)
    let nv = noise(3, N * wide);
    let fv: Vec<f32> = noise(4, mix_hc * wide).iter().map(|v| v * 0.05).collect();
    let mixes_want: Vec<f64> = (0..N * mix_hc)
        .map(|i| {
            let (n, o) = (i / mix_hc, i % mix_hc);
            (0..wide)
                .map(|d| f64::from(nv[n * wide + d]) * f64::from(fv[o * wide + d]))
                .sum()
        })
        .collect();
    // gates, from the reference mix row
    let mixes_v: Vec<f32> = mixes_want.iter().map(|v| *v as f32).collect();
    let mixes = gpu.up_f32(&mixes_v);
    let scale_v = [0.7f32, 1.1, 0.9];
    let base_v = noise(5, mix_hc);
    let (scale, base) = (gpu.up_f32(&scale_v), gpu.up_f32(&base_v));
    let e = f64::from(eps);
    let mut post_w = Vec::new();
    let mut comb_w = Vec::new();
    let mut layer_w = Vec::new();
    for n in 0..N {
        let row = &mixes_v[n * mix_hc..][..mix_hc];
        let at =
            |k: usize, s: usize| f64::from(row[k]) * f64::from(scale_v[s]) + f64::from(base_v[k]);
        let pre: Vec<f64> = (0..M).map(|i| sigmoid(at(i, 0)) + e).collect();
        post_w.extend((0..M).map(|i| sigmoid(at(M + i, 1)) * f64::from(alpha)));
        let mut comb: Vec<f64> = (0..M * M).map(|k| at(2 * M + k, 2)).collect();
        for i in 0..M {
            let top = comb[i * M..][..M]
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max);
            let sum: f64 = (0..M).map(|j| (comb[i * M + j] - top).exp()).sum();
            for j in 0..M {
                comb[i * M + j] = (comb[i * M + j] - top).exp() / sum + e;
            }
        }
        let cols = |c: &mut Vec<f64>| {
            for j in 0..M {
                let s = (0..M).map(|i| c[i * M + j]).sum::<f64>() + e;
                (0..M).for_each(|i| c[i * M + j] /= s);
            }
        };
        let rows_n = |c: &mut Vec<f64>| {
            for i in 0..M {
                let s = (0..M).map(|j| c[i * M + j]).sum::<f64>() + e;
                (0..M).for_each(|j| c[i * M + j] /= s);
            }
        };
        cols(&mut comb);
        for _ in 1..sinkhorn {
            rows_n(&mut comb);
            cols(&mut comb);
        }
        comb_w.extend(comb);
        layer_w.extend((0..H).map(|k| {
            (0..M)
                .map(|i| pre[i] * sr[n * wide + i * H + k])
                .sum::<f64>()
        }));
    }
    landed(
        &mut gpu,
        "hc gates",
        &|g| {
            vec![
                (g.zeros(N * H), N * H, false),
                (g.zeros(2 * N * M), N * M, true),
                (g.zeros(2 * N * M * M), N * M * M, true),
            ]
        },
        &[layer_w, post_w.clone(), comb_w.clone()],
        &|g, o| {
            g.run(|c| {
                kernels_slang::elemwise::hc::gates(
                    c,
                    f(mixes, N, mix_hc),
                    b(streams, wide),
                    f(scale, 1, 3),
                    f(base, 1, mix_hc),
                    M as u32,
                    eps,
                    alpha,
                    sinkhorn,
                    b(o[0].0, H),
                    f(o[1].0, N, M),
                    f(o[2].0, N, M * M),
                )
            })
        },
    );

    // fold, under the reference gate matrices
    let post = gpu.up_f32(&post_w.iter().map(|v| *v as f32).collect::<Vec<_>>());
    let comb = gpu.up_f32(&comb_w.iter().map(|v| *v as f32).collect::<Vec<_>>());
    let want: Vec<f64> = (0..N * wide)
        .map(|i| {
            let (n, j, k) = (i / wide, (i % wide) / H, i % H);
            let mut acc = f64::from(post_w[n * M + j] as f32) * xr[n * H + k];
            for s in 0..M {
                acc += f64::from(comb_w[n * M * M + s * M + j] as f32) * sr[n * wide + s * H + k];
            }
            acc
        })
        .collect();
    landed(
        &mut gpu,
        "hc fold",
        &|g| vec![(g.zeros(N * wide), N * wide, false)],
        &[want],
        &|g, o| {
            g.run(|c| {
                kernels_slang::elemwise::hc::fold(
                    c,
                    b(x, H),
                    b(streams, wide),
                    f(post, N, M),
                    f(comb, N, M * M),
                    b(o[0].0, wide),
                )
            })
        },
    );

    // collapse
    let cm = noise(6, N * M);
    let cmix = gpu.up_f32(&cm);
    let want: Vec<f64> = (0..N * H)
        .map(|i| {
            let (n, k) = (i / H, i % H);
            (0..M)
                .map(|s| {
                    let g = sigmoid(
                        f64::from(cm[n * M + s]) * f64::from(scale_v[0]) + f64::from(base_v[s]),
                    ) + e;
                    g * sr[n * wide + s * H + k]
                })
                .sum()
        })
        .collect();
    landed(
        &mut gpu,
        "hc collapse",
        &|g| vec![(g.zeros(N * H), N * H, false)],
        &[want],
        &|g, o| {
            g.run(|c| {
                kernels_slang::elemwise::hc::collapse(
                    c,
                    f(cmix, N, M),
                    b(streams, wide),
                    f(scale, 1, 3),
                    f(base, 1, mix_hc),
                    M as u32,
                    eps,
                    b(o[0].0, H),
                )
            })
        },
    );

    // mix
    let (gates, gr) = gpu.up(&noise(7, N * wide));
    let gr = f64s(&gr);
    let want: Vec<f64> = (0..N * H)
        .map(|i| {
            let (n, k) = (i / H, i % H);
            (0..M)
                .map(|s| sr[n * wide + s * H + k] * sigmoid(gr[n * wide + s * H + k]))
                .sum::<f64>()
                / M as f64
        })
        .collect();
    landed(
        &mut gpu,
        "hc mix",
        &|g| vec![(g.zeros(N * H), N * H, false)],
        &[want],
        &|g, o| {
            g.run(|c| {
                kernels_slang::elemwise::hc::mix(
                    c,
                    b(gates, wide),
                    b(streams, wide),
                    M as u32,
                    b(o[0].0, H),
                )
            })
        },
    );

    // inject, in place
    let (sg, sgr) = gpu.up(&noise(8, N * M));
    let sgr = f64s(&sgr);
    let hv = noise(9, N * wide);
    let hr = f64s(&gpu.up(&hv).1);
    let want: Vec<f64> = (0..N * wide)
        .map(|i| {
            let (n, s, k) = (i / wide, (i % wide) / H, i % H);
            hr[i] + 2.0 * sigmoid(sgr[n * M + s] / M as f64) * xr[n * H + k]
        })
        .collect();
    landed(
        &mut gpu,
        "hc inject",
        &|g| vec![(g.up(&hv).0, N * wide, false)],
        &[want],
        &|g, o| {
            g.run(|c| {
                kernels_slang::elemwise::hc::inject(
                    c,
                    b(x, H),
                    Tensor::new(sg, N as u32, M as u32, Dtype::Bf16),
                    M as u32,
                    b(o[0].0, wide),
                )
            })
        },
    );

    // the PLE gate
    let (key, kr) = gpu.up(&noise(10, N * wide));
    let (query, qr) = gpu.up(&noise(11, N * wide));
    let (kr, qr) = (f64s(&kr), f64s(&qr));
    let want: Vec<f64> = (0..N * wide)
        .map(|i| {
            let (n, s, k) = (i / wide, (i % wide) / H, i % H);
            let at = n * wide + s * H;
            let d = (0..H).map(|j| kr[at + j] * qr[at + j]).sum::<f64>() / (H as f64).sqrt();
            let mag = d.abs().max(1e-6).sqrt();
            let damped = if d > 0.0 {
                mag
            } else if d < 0.0 {
                -mag
            } else {
                0.0
            };
            sigmoid(damped) * xr[n * H + k]
        })
        .collect();
    landed(
        &mut gpu,
        "ple gate",
        &|g| vec![(g.zeros(N * wide), N * wide, false)],
        &[want],
        &|g, o| {
            g.run(|c| {
                kernels_slang::elemwise::hc::ple_gate(
                    c,
                    b(key, wide),
                    b(query, wide),
                    b(x, H),
                    M as u32,
                    b(o[0].0, wide),
                )
            })
        },
    );
}
