//! The rotary kernels on the first wgpu adapter against an f64 host reference,
//! beside the hand-written kernels: by halves at a span (full, partial,
//! proportional), the rotated tail (halves or interleaved, forward or back,
//! with and without a YaRN ramp), and whole-head YaRN.

use crate::gpu::{Gpu, close, noise};
use kernels_wgpu::Tensor;
use kernels_wgpu::elemwise::{rope, rope_mrope};
use poem_ir::Dtype;

const ROWS: usize = 3;
const HEADS: usize = 4;
const POSITIONS: [i32; ROWS] = [0, 7, 1000];

/// Turn pair `(a, b)` of every head of every row by `angle(row, k)`, `k`
/// the pair's index; `scale` multiplies cos and sin.
fn turn(
    x: &[f32],
    head_dim: usize,
    pairs: impl Fn(usize) -> (usize, usize),
    count: usize,
    angle: impl Fn(f64, usize) -> f64,
    scale: f64,
) -> Vec<f64> {
    let mut out: Vec<f64> = x.iter().map(|v| f64::from(*v)).collect();
    for (r, position) in POSITIONS.iter().enumerate() {
        for h in 0..HEADS {
            let base = (r * HEADS + h) * head_dim;
            for k in 0..count {
                let (a, b) = pairs(k);
                let t = angle(f64::from(*position), k);
                let (c, s) = (t.cos() * scale, t.sin() * scale);
                let (xa, xb) = (out[base + a], out[base + b]);
                out[base + a] = xa * c - xb * s;
                out[base + b] = xa * s + xb * c;
            }
        }
    }
    out
}

/// The YaRN ramp's first and last pair, as the host states them.
fn ramp(width: f64, theta: f64, beta_fast: f64, beta_slow: f64, max_position: f64) -> (f64, f64) {
    let corr =
        |rot: f64| width * (max_position / (rot * std::f64::consts::TAU)).ln() / (2.0 * theta.ln());
    let low = corr(beta_fast).floor().max(0.0);
    (low, corr(beta_slow).ceil().min(width / 2.0 - 1.0).max(low))
}

fn ramped(f: f64, k: usize, factor: f64, low: f64, high: f64) -> f64 {
    let r = ((k as f64 - low) / if high == low { 1e-3 } else { high - low }).clamp(0.0, 1.0);
    f * ((1.0 - r) + r / factor)
}

#[test]
fn the_ropes_answer_the_reference() {
    let Some(mut gpu) = Gpu::open() else {
        eprintln!("not asked: no device");
        return;
    };
    let st = |t: Tensor| kernels_slang::Tensor::new(t.buf, t.rows, t.width, t.dtype);
    let pos = gpu.up_i32(&POSITIONS);
    let pt = Tensor::new(pos, ROWS as u32, 1, Dtype::I32);
    let theta = 10_000.0f32;
    let th = f64::from(theta);

    // By halves at a span: full (span = head / 2), partial (span = rotary / 2)
    // and proportional (span = head / 2 over a narrower rotary).
    for (head_dim, rotary, span) in [(64usize, 64usize, 32usize), (128, 64, 32), (128, 64, 64)] {
        let width = HEADS * head_dim;
        let n = ROWS * width;
        let t = |buf: u32| Tensor::new(buf, ROWS as u32, width as u32, Dtype::Bf16);
        let qv = noise(head_dim as u64 + span as u64, n);
        let kv = noise(head_dim as u64 + span as u64 + 1, n);
        let (_, qr) = gpu.up(&qv);
        let (_, kr) = gpu.up(&kv);
        let angle = |p: f64, k: usize| p * th.powf(-(k as f64) / span as f64);
        let pairs = |k: usize| (k, k + span);
        let wq = turn(&qr, head_dim, pairs, rotary / 2, angle, 1.0);
        let wk = turn(&kr, head_dim, pairs, rotary / 2, angle, 1.0);
        let shape = format!("head {head_dim}, rotary {rotary}, span {span}");
        for slang in [false, true] {
            let who = if slang { "slang " } else { "" };
            let ((q, _), (k, _)) = (gpu.up(&qv), gpu.up(&kv));
            let (hd, rd, sp) = (head_dim as u32, rotary as u32, span as u32);
            match (slang, rotary == head_dim) {
                (false, true) => gpu.run(|c| rope::full(c, t(q), t(k), pt, hd, theta, false)),
                (false, false) => gpu.run(|c| rope::partial(c, t(q), t(k), pt, rd, hd, theta, sp)),
                (true, true) => gpu.run_slang(|c| {
                    kernels_slang::elemwise::rope::full(
                        c,
                        st(t(q)),
                        st(t(k)),
                        st(pt),
                        hd,
                        theta,
                        false,
                    )
                }),
                (true, false) => gpu.run_slang(|c| {
                    kernels_slang::elemwise::rope::partial(
                        c,
                        st(t(q)),
                        st(t(k)),
                        st(pt),
                        rd,
                        hd,
                        theta,
                        sp,
                    )
                }),
            }
            close(&gpu.down(q, n), &wq, &format!("{who}rope q {shape}"));
            close(&gpu.down(k, n), &wk, &format!("{who}rope k {shape}"));
            let (q, _) = gpu.up(&qv);
            if slang {
                gpu.run_slang(|c| {
                    kernels_slang::elemwise::rope::partial_q(c, st(t(q)), st(pt), rd, hd, theta, sp)
                });
            } else {
                gpu.run(|c| rope::partial_q(c, t(q), pt, rd, hd, theta, sp));
            }
            close(&gpu.down(q, n), &wq, &format!("{who}rope q alone {shape}"));
        }
    }

    // The rotated tail.
    let (head_dim, rotary) = (128usize, 64usize);
    let width = HEADS * head_dim;
    let n = ROWS * width;
    let t = |buf: u32| Tensor::new(buf, ROWS as u32, width as u32, Dtype::Bf16);
    let qv = noise(77, n);
    let (_, qr) = gpu.up(&qv);
    let yarn = rope::Yarn {
        factor: 4.0,
        beta_fast: 32.0,
        beta_slow: 1.0,
        original_max_position: 4096,
    };
    let (low, high) = ramp(rotary as f64, th, 32.0, 1.0, 4096.0);
    for interleaved in [false, true] {
        for inverse in [false, true] {
            for ramped_tail in [false, true] {
                let off = head_dim - rotary;
                let half = rotary / 2;
                let sign = if inverse { -1.0 } else { 1.0 };
                let angle = |p: f64, k: usize| {
                    let f = th.powf(-2.0 * k as f64 / rotary as f64);
                    let f = if ramped_tail {
                        ramped(f, k, 4.0, low, high)
                    } else {
                        f
                    };
                    sign * p * f
                };
                let want = if interleaved {
                    turn(
                        &qr,
                        head_dim,
                        |k| (off + 2 * k, off + 2 * k + 1),
                        half,
                        angle,
                        1.0,
                    )
                } else {
                    turn(
                        &qr,
                        head_dim,
                        |k| (off + k, off + k + half),
                        half,
                        angle,
                        1.0,
                    )
                };
                let shape =
                    format!("interleaved {interleaved}, inverse {inverse}, yarn {ramped_tail}");
                let (hd, rd) = (head_dim as u32, rotary as u32);
                let (q, _) = gpu.up(&qv);
                gpu.run(|c| {
                    rope::partial_last(
                        c,
                        t(q),
                        pt,
                        rd,
                        hd,
                        theta,
                        interleaved,
                        inverse,
                        ramped_tail.then_some(yarn),
                    )
                });
                close(&gpu.down(q, n), &want, &format!("rope tail {shape}"));
                let (q, _) = gpu.up(&qv);
                let sy = kernels_slang::elemwise::rope::Yarn {
                    factor: yarn.factor,
                    beta_fast: yarn.beta_fast,
                    beta_slow: yarn.beta_slow,
                    original_max_position: yarn.original_max_position,
                };
                gpu.run_slang(|c| {
                    kernels_slang::elemwise::rope::partial_last(
                        c,
                        st(t(q)),
                        st(pt),
                        rd,
                        hd,
                        theta,
                        interleaved,
                        inverse,
                        ramped_tail.then_some(sy),
                    )
                });
                close(&gpu.down(q, n), &want, &format!("slang rope tail {shape}"));
            }
        }
    }

    // Whole-head YaRN.
    let head_dim = 64usize;
    let width = HEADS * head_dim;
    let n = ROWS * width;
    let t = |buf: u32| Tensor::new(buf, ROWS as u32, width as u32, Dtype::Bf16);
    let (qv, kv) = (noise(91, n), noise(92, n));
    let (_, qr) = gpu.up(&qv);
    let (_, kr) = gpu.up(&kv);
    let (low, high) = ramp(head_dim as f64, th, 32.0, 1.0, 4096.0);
    let mscale = 1.1f32;
    for interleaved in [false, true] {
        let half = head_dim / 2;
        let angle = |p: f64, k: usize| {
            p * ramped(
                th.powf(-2.0 * k as f64 / head_dim as f64),
                k,
                4.0,
                low,
                high,
            )
        };
        let pairs = |k: usize| {
            if interleaved {
                (2 * k, 2 * k + 1)
            } else {
                (k, k + half)
            }
        };
        let wq = turn(&qr, head_dim, pairs, half, angle, f64::from(mscale));
        let wk = turn(&kr, head_dim, pairs, half, angle, f64::from(mscale));
        let hd = head_dim as u32;
        for slang in [false, true] {
            let who = if slang { "slang " } else { "" };
            let ((q, _), (k, _)) = (gpu.up(&qv), gpu.up(&kv));
            if slang {
                gpu.run_slang(|c| {
                    kernels_slang::elemwise::rope::yarn(
                        c,
                        st(t(q)),
                        st(t(k)),
                        st(pt),
                        hd,
                        theta,
                        4.0,
                        32.0,
                        1.0,
                        mscale,
                        4096,
                        interleaved,
                    )
                });
            } else {
                gpu.run(|c| {
                    rope::yarn(
                        c,
                        t(q),
                        t(k),
                        pt,
                        hd,
                        theta,
                        4.0,
                        32.0,
                        1.0,
                        mscale,
                        4096,
                        interleaved,
                    )
                });
            }
            close(
                &gpu.down(q, n),
                &wq,
                &format!("{who}rope yarn q interleaved {interleaved}"),
            );
            close(
                &gpu.down(k, n),
                &wk,
                &format!("{who}rope yarn k interleaved {interleaved}"),
            );
        }
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Mrope {
    Interleaved,
    Blocked,
    Split,
}

/// M-RoPE over every head of every row: per pair, the axis it reads from
/// the row's (t, h, w) triple, its frequency, and its two columns.
fn mrope_reference(
    x: &[f32],
    (heads, head_dim): (usize, usize),
    form: Mrope,
    sections: [usize; 3],
    rotary: usize,
    theta: f64,
    positions: &[[i32; 3]],
) -> Vec<f64> {
    let total: usize = sections.iter().sum();
    let pairs = match form {
        Mrope::Interleaved => rotary / 2,
        _ => (rotary / 2).min(total),
    };
    let mut out: Vec<f64> = x.iter().map(|v| f64::from(*v)).collect();
    for (r, pos) in positions.iter().enumerate() {
        for h in 0..heads {
            let base = (r * heads + h) * head_dim;
            for i in 0..pairs {
                let section = if i < sections[0] {
                    0
                } else if i < sections[0] + sections[1] {
                    1
                } else {
                    2
                };
                let before: usize = sections[..section].iter().sum();
                let (axis, inv_freq, a, b) = match form {
                    Mrope::Interleaved => {
                        let axis = match i % 3 {
                            1 if i < 3 * sections[1] => 1,
                            2 if i < 3 * sections[2] => 2,
                            _ => 0,
                        };
                        (
                            axis,
                            theta.powf(-(2.0 * i as f64) / (2 * pairs) as f64),
                            i,
                            i + pairs,
                        )
                    }
                    Mrope::Blocked => {
                        let within = i - before;
                        (
                            section,
                            theta.powf(-(2.0 * within as f64) / total as f64),
                            i,
                            i + head_dim / 2,
                        )
                    }
                    Mrope::Split => {
                        let (within, width) = (i - before, sections[section]);
                        let a = 2 * before + within;
                        (
                            section,
                            theta.powf(-(within as f64) / width as f64),
                            a,
                            a + width,
                        )
                    }
                };
                let t = f64::from(pos[axis]) * inv_freq;
                let (xa, xb) = (out[base + a], out[base + b]);
                out[base + a] = xa * t.cos() - xb * t.sin();
                out[base + b] = xa * t.sin() + xb * t.cos();
            }
        }
    }
    out
}

#[test]
fn the_mropes_answer_the_reference() {
    let Some(mut gpu) = Gpu::open() else {
        eprintln!("not asked: no device");
        return;
    };
    let st = |t: Tensor| kernels_slang::Tensor::new(t.buf, t.rows, t.width, t.dtype);
    let triples: [[i32; 3]; ROWS] = [[0, 0, 0], [5, 2, 3], [900, 31, 17]];
    let flat: Vec<i32> = triples.iter().flatten().copied().collect();
    let pos = gpu.up_i32(&flat);
    let pt = Tensor::new(pos, ROWS as u32, 3, Dtype::I32);
    let theta = 1_000_000.0f32;
    let (q_heads, k_heads) = (4usize, 2usize);
    for (form, sections, rotary, head_dim) in [
        (Mrope::Interleaved, [24usize, 20, 20], 128usize, 128usize),
        (Mrope::Blocked, [16, 24, 24], 128, 128),
        (Mrope::Split, [11, 10, 11], 64, 64),
        (Mrope::Split, [8, 12, 12], 64, 64),
    ] {
        let qn = ROWS * q_heads * head_dim;
        let kn = ROWS * k_heads * head_dim;
        let qv = noise(head_dim as u64 + sections[0] as u64, qn);
        let kv = noise(head_dim as u64 + sections[1] as u64, kn);
        let (_, qr) = gpu.up(&qv);
        let (_, kr) = gpu.up(&kv);
        let th = f64::from(theta);
        let wq = mrope_reference(
            &qr,
            (q_heads, head_dim),
            form,
            sections,
            rotary,
            th,
            &triples,
        );
        let wk = mrope_reference(
            &kr,
            (k_heads, head_dim),
            form,
            sections,
            rotary,
            th,
            &triples,
        );
        let t = |buf: u32, heads: usize| {
            Tensor::new(buf, ROWS as u32, (heads * head_dim) as u32, Dtype::Bf16)
        };
        let s3 = sections.map(|s| s as u32);
        let (rd, hd) = (rotary as u32, head_dim as u32);
        let shape = format!("sections {sections:?}, rotary {rotary}, head {head_dim}");
        for slang in [false, true] {
            let who = if slang { "slang " } else { "" };
            let ((q, _), (k, _)) = (gpu.up(&qv), gpu.up(&kv));
            let (qt, kt) = (t(q, q_heads), t(k, k_heads));
            if slang {
                let (qt, kt, pt) = (st(qt), st(kt), st(pt));
                gpu.run_slang(|c| match form {
                    Mrope::Interleaved => kernels_slang::elemwise::rope_mrope::interleaved(
                        c, qt, kt, pt, s3, rd, hd, theta,
                    ),
                    Mrope::Blocked => kernels_slang::elemwise::rope_mrope::blocked(
                        c, qt, kt, pt, s3, rd, hd, theta,
                    ),
                    Mrope::Split => {
                        kernels_slang::elemwise::rope_mrope::split(c, qt, kt, pt, s3, rd, hd, theta)
                    }
                });
            } else {
                gpu.run(|c| match form {
                    Mrope::Interleaved => rope_mrope::interleaved(c, qt, kt, pt, s3, rd, hd, theta),
                    Mrope::Blocked => rope_mrope::blocked(c, qt, kt, pt, s3, rd, hd, theta),
                    Mrope::Split => rope_mrope::split(c, qt, kt, pt, s3, rd, hd, theta),
                });
            }
            close(&gpu.down(q, qn), &wq, &format!("{who}mrope q {shape}"));
            close(&gpu.down(k, kn), &wk, &format!("{who}mrope k {shape}"));
        }
    }
}
