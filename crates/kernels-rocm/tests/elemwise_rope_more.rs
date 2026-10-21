//! The rotary variants against a host reference: full (neox and
//! interleaved), YaRN, the rotated tail, the per-head norm fused in front of
//! a partial rotary, and the three M-RoPE layouts. The host takes the same
//! f32 angle the device does and its exact cos and sin; outputs are bf16.

#![cfg(feature = "rocm")]

mod common;

use common::{Gpu, round_bf16, values};
use dtype::Dtype;
use kernels_rocm::elemwise::{rope, rope_mrope};

/// theta^(-i/span), as the device takes it.
fn freq(theta: f32, i: usize, span: usize) -> f32 {
    theta.powf(-(i as f32) / (span as f32))
}

fn yarn_freq(f: f32, factor: f32, low: f32, high: f32, i: usize) -> f32 {
    let denom = if high == low {
        high + 1e-3 - low
    } else {
        high - low
    };
    let ramp = ((i as f32 - low) / denom).clamp(0.0, 1.0);
    f * ((1.0 - ramp) + ramp / factor)
}

fn cos_sin(pos: i32, f: f32) -> (f32, f32) {
    let a = f64::from(pos as f32 * f);
    (a.cos() as f32, a.sin() as f32)
}

/// Entries a and b of `head` turn by (c, s), rounded once to bf16.
fn turn(head: &mut [f32], a: usize, b: usize, c: f32, s: f32) {
    let (x, y) = (head[a], head[b]);
    head[a] = round_bf16(x * c - y * s);
    head[b] = round_bf16(y * c + x * s);
}

/// Within a bf16 step of `want`, plus what an f32 angle at a position of a
/// few thousand can differ by between two correct sincos.
fn close_rot(what: &str, have: &[f32], want: &[f32]) {
    assert_eq!(have.len(), want.len(), "{what}: lengths");
    for (i, (h, w)) in have.iter().zip(want).enumerate() {
        assert!(
            (h - w).abs() <= w.abs() / 128.0 + 2e-3,
            "{what}[{i}] = {h}, want {w}"
        );
    }
}

fn positions(n: usize, seed: u64) -> Vec<i32> {
    values(n, seed, 1.0)
        .into_iter()
        .map(|v| ((v + 1.0) * 2048.0) as i32)
        .collect()
}

fn bf16s(n: usize, seed: u64) -> Vec<f32> {
    values(n, seed, 1.0).into_iter().map(round_bf16).collect()
}

/// Applies `f(head, row)` to every `head_dim`-wide head of every row.
fn per_head(
    xs: &[f32],
    head_dim: usize,
    heads: usize,
    mut f: impl FnMut(&mut [f32], usize),
) -> Vec<f32> {
    let mut out = xs.to_vec();
    for (n, row) in out.chunks_mut(heads * head_dim).enumerate() {
        for head in row.chunks_mut(head_dim) {
            f(head, n);
        }
    }
    out
}

#[test]
fn rope_full_neox_and_interleaved() {
    let Some(gpu) = Gpu::open() else { return };
    let ctx = gpu.ctx();
    let theta = 10_000.0f32;
    // 128: two heads a block; 80: a block spans whole heads unevenly; 512:
    // a head a block.
    for (rows, qh, kh, hd) in [
        (5usize, 4usize, 2usize, 128usize),
        (3, 3, 1, 80),
        (2, 2, 1, 512),
    ] {
        for interleaved in [false, true] {
            let qs = bf16s(rows * qh * hd, 1);
            let ks = bf16s(rows * kh * hd, 2);
            let pos = positions(rows, 3);
            let mut q = gpu.tensor(&qs, rows as u32, (qh * hd) as u32, Dtype::Bf16);
            let mut k = gpu.tensor(&ks, rows as u32, (kh * hd) as u32, Dtype::Bf16);
            rope::full(
                &ctx,
                &mut q,
                &mut k,
                gpu.column(&pos),
                hd as u32,
                theta,
                interleaved,
            )
            .unwrap();
            let half = hd / 2;
            let rot = |head: &mut [f32], n: usize| {
                for i in 0..half {
                    let (c, s) = cos_sin(pos[n], freq(theta, i, half));
                    if interleaved {
                        turn(head, 2 * i, 2 * i + 1, c, s);
                    } else {
                        turn(head, i, i + half, c, s);
                    }
                }
            };
            let what = format!("rope_full hd={hd} interleaved={interleaved}");
            close_rot(&what, &gpu.read(&q), &per_head(&qs, hd, qh, rot));
            close_rot(&what, &gpu.read(&k), &per_head(&ks, hd, kh, rot));
        }
    }

    // A window: the launch covers 6 rows, the staged extent is rows 2..5.
    let (rows, qh, kh, hd) = (6usize, 2usize, 1usize, 64usize);
    let qs = bf16s(rows * qh * hd, 4);
    let ks = bf16s(rows * kh * hd, 5);
    let pos = positions(rows, 6);
    let mut q = gpu.tensor(&qs, rows as u32, (qh * hd) as u32, Dtype::Bf16);
    let mut k = gpu.tensor(&ks, rows as u32, (kh * hd) as u32, Dtype::Bf16);
    let win = gpu.ints(&[3, 2, 0, 0]);
    ctx.arm_stage(win.ptr);
    rope::full(
        &ctx,
        &mut q,
        &mut k,
        gpu.column(&pos),
        hd as u32,
        theta,
        false,
    )
    .unwrap();
    ctx.disarm_stage();
    let rot = |head: &mut [f32], n: usize| {
        if (2..5).contains(&n) {
            for i in 0..hd / 2 {
                let (c, s) = cos_sin(pos[n], freq(theta, i, hd / 2));
                turn(head, i, i + hd / 2, c, s);
            }
        }
    };
    close_rot(
        "rope_full windowed q",
        &gpu.read(&q),
        &per_head(&qs, hd, qh, rot),
    );
    close_rot(
        "rope_full windowed k",
        &gpu.read(&k),
        &per_head(&ks, hd, kh, rot),
    );
}

#[test]
fn rope_yarn() {
    let Some(gpu) = Gpu::open() else { return };
    let ctx = gpu.ctx();
    let (theta, factor, fast, slow, mscale, orig) =
        (10_000.0f32, 4.0f32, 32.0, 1.0, 1.2f32, 4096u32);
    let (rows, qh, kh, hd) = (4usize, 4usize, 2usize, 64usize);
    let (low, high) = rope::ramp_bounds(hd as i32, theta, fast, slow, orig as i32);
    for interleaved in [false, true] {
        let qs = bf16s(rows * qh * hd, 7);
        let ks = bf16s(rows * kh * hd, 8);
        let pos = positions(rows, 9);
        let mut q = gpu.tensor(&qs, rows as u32, (qh * hd) as u32, Dtype::Bf16);
        let mut k = gpu.tensor(&ks, rows as u32, (kh * hd) as u32, Dtype::Bf16);
        rope::yarn(
            &ctx,
            &mut q,
            &mut k,
            gpu.column(&pos),
            rope::YarnRope {
                head_dim: hd as u32,
                theta,
                yarn: rope::Yarn {
                    factor,
                    beta_fast: fast,
                    beta_slow: slow,
                    original_max_position: orig,
                },
                attention_factor: mscale,
                interleaved,
            },
        )
        .unwrap();
        let half = hd / 2;
        let rot = |head: &mut [f32], n: usize| {
            for i in 0..half {
                let f = yarn_freq(freq(theta, i, half), factor, low, high, i);
                let (c, s) = cos_sin(pos[n], f);
                let (c, s) = (c * mscale, s * mscale);
                if interleaved {
                    turn(head, 2 * i, 2 * i + 1, c, s);
                } else {
                    turn(head, i, i + half, c, s);
                }
            }
        };
        let what = format!("rope_yarn interleaved={interleaved}");
        close_rot(&what, &gpu.read(&q), &per_head(&qs, hd, qh, rot));
        close_rot(&what, &gpu.read(&k), &per_head(&ks, hd, kh, rot));
    }
}

#[test]
fn rope_partial_last() {
    let Some(gpu) = Gpu::open() else { return };
    let ctx = gpu.ctx();
    let theta = 10_000.0f32;
    let yarn = rope::Yarn {
        factor: 40.0,
        beta_fast: 32.0,
        beta_slow: 1.0,
        original_max_position: 4096,
    };
    // DeepSeek's 64-wide rotated tail of a 192-wide head (interleaved, YaRN)
    // past a block of pairs and within one, a whole head turned back, a plain
    // tail, and a tail of no whole packs.
    for (rows, heads, hd, rot_dim, interleaved, inverse, scaled) in [
        (5usize, 16usize, 192usize, 64usize, true, false, true),
        (5, 3, 192, 64, true, false, true),
        (3, 6, 128, 128, false, true, false),
        (4, 1, 96, 32, false, false, false),
        (2, 16, 100, 36, true, false, false),
    ] {
        let qs = bf16s(rows * heads * hd, 10);
        let pos = positions(rows, 11);
        let mut q = gpu.tensor(&qs, rows as u32, (heads * hd) as u32, Dtype::Bf16);
        rope::partial_last(
            &ctx,
            &mut q,
            gpu.column(&pos),
            rot_dim as u32,
            hd as u32,
            theta,
            interleaved,
            inverse,
            scaled.then_some(yarn),
        )
        .unwrap();
        let (low, high) = rope::ramp_bounds(rot_dim as i32, theta, 32.0, 1.0, 4096);
        let half = rot_dim / 2;
        let off = hd - rot_dim;
        let want = per_head(&qs, hd, heads, |head, n| {
            for i in 0..half {
                let mut f = freq(theta, i, half);
                if scaled {
                    f = yarn_freq(f, yarn.factor, low, high, i);
                }
                let p = if inverse { -pos[n] } else { pos[n] };
                let (c, s) = cos_sin(p, f);
                if interleaved {
                    turn(head, off + 2 * i, off + 2 * i + 1, c, s);
                } else {
                    turn(head, off + i, off + i + half, c, s);
                }
            }
        });
        close_rot(
            &format!("rope_partial_last hd={hd} rot={rot_dim}"),
            &gpu.read(&q),
            &want,
        );
    }
}

#[test]
fn rmsnorm_rope_partial_q() {
    let Some(gpu) = Gpu::open() else { return };
    let ctx = gpu.ctx();
    let (theta, eps) = (1_000_000.0f32, 1e-6f32);
    // A plain partial rotary (span = rotary/2), a proportional one (span =
    // head/2, Gemma 4's global layers), and a window over the launch.
    for (rows, heads, hd, rot_dim, span, win) in [
        (4usize, 3usize, 256usize, 64usize, 32usize, None),
        (3, 2, 512, 128, 256, None),
        (6, 2, 128, 128, 64, Some((3usize, 2usize))),
    ] {
        let xs: Vec<f32> = values(rows * heads * hd, 12, 3.0)
            .into_iter()
            .map(round_bf16)
            .collect();
        let ws = bf16s(hd, 13);
        let pos = positions(rows, 14);
        let x = gpu.tensor(&xs, rows as u32, (heads * hd) as u32, Dtype::Bf16);
        let w = gpu.tensor(&ws, 1, hd as u32, Dtype::Bf16);
        let mut y = gpu.zeros(rows as u32, (heads * hd) as u32, Dtype::Bf16);
        let staged = win.map(|(n, at)| gpu.ints(&[n as i32, at as i32, 0, 0]));
        if let Some(s) = &staged {
            ctx.arm_stage(s.ptr);
        }
        rope::rmsnorm_rope_partial_q(
            &ctx,
            x,
            w,
            hd as u32,
            eps,
            gpu.column(&pos),
            rot_dim as u32,
            theta,
            span as u32,
            &mut y,
        )
        .unwrap();
        ctx.disarm_stage();
        let want = per_head(&xs, hd, heads, |head, n| {
            if win.is_some_and(|(len, at)| !(at..at + len).contains(&n)) {
                head.fill(0.0);
                return;
            }
            let inv = 1.0 / (head.iter().map(|v| v * v).sum::<f32>() / hd as f32 + eps).sqrt();
            let normed: Vec<f32> = head.iter().zip(&ws).map(|(v, w)| v * inv * w).collect();
            for (o, v) in head.iter_mut().zip(&normed) {
                *o = round_bf16(*v);
            }
            for i in 0..rot_dim / 2 {
                let (c, s) = cos_sin(pos[n], freq(theta, i, span));
                let (a, b) = (normed[i], normed[i + span]);
                head[i] = round_bf16(a * c - b * s);
                head[i + span] = round_bf16(b * c + a * s);
            }
        });
        close_rot(
            &format!("rmsnorm_rope_partial_q hd={hd} rot={rot_dim} span={span}"),
            &gpu.read(&y),
            &want,
        );
    }
}

#[derive(Clone, Copy, Debug)]
enum Form {
    Interleaved,
    Blocked,
    Split,
}

/// The host M-RoPE of one head at the (t, h, w) triple `p`.
fn mrope_head(
    head: &mut [f32],
    p: [i32; 3],
    form: Form,
    s: [usize; 3],
    rot_dim: usize,
    theta: f32,
) {
    let total = s[0] + s[1] + s[2];
    match form {
        Form::Interleaved => {
            let half = rot_dim / 2;
            for i in 0..half {
                let axis = match i % 3 {
                    1 if i < 3 * s[1] => 1,
                    2 if i < 3 * s[2] => 2,
                    _ => 0,
                };
                let (c, sn) = cos_sin(p[axis], freq(theta, i, half));
                turn(head, i, i + half, c, sn);
            }
        }
        Form::Blocked => {
            let half = head.len() / 2;
            for i in 0..(rot_dim / 2).min(total) {
                let (axis, within) = if i < s[0] {
                    (0, i)
                } else if i < s[0] + s[1] {
                    (1, i - s[0])
                } else {
                    (2, i - s[0] - s[1])
                };
                let f = theta.powf(-2.0 * within as f32 / total as f32);
                let (c, sn) = cos_sin(p[axis], f);
                turn(head, i, i + half, c, sn);
            }
        }
        Form::Split => {
            for i in 0..(rot_dim / 2).min(total) {
                let (axis, before) = if i < s[0] {
                    (0, 0)
                } else if i < s[0] + s[1] {
                    (1, s[0])
                } else {
                    (2, s[0] + s[1])
                };
                let (width, within) = (s[axis], i - before);
                let (c, sn) = cos_sin(p[axis], freq(theta, within, width));
                let a = 2 * before + within;
                turn(head, a, a + width, c, sn);
            }
        }
    }
}

#[test]
fn rope_mrope_forms() {
    let Some(gpu) = Gpu::open() else { return };
    let ctx = gpu.ctx();
    let theta = 10_000.0f32;
    // (form, head_dim, rotary_dim, sections, window {rows, row_offset})
    type Case = (Form, usize, usize, [usize; 3], Option<(usize, usize)>);
    let cases: [Case; 8] = [
        (Form::Interleaved, 128, 128, [24, 20, 20], None),
        (Form::Interleaved, 128, 64, [12, 10, 10], None),
        (Form::Blocked, 128, 128, [16, 24, 24], None),
        (Form::Blocked, 128, 80, [16, 24, 24], None),
        (Form::Split, 64, 64, [8, 12, 12], None),
        (Form::Split, 72, 64, [0, 16, 16], None),
        (Form::Split, 64, 40, [8, 12, 12], None),
        (Form::Split, 64, 64, [8, 12, 12], Some((2, 1))),
    ];
    for (form, hd, rot_dim, sections, win) in cases {
        let (rows, qh, kh) = (4usize, 3usize, 2usize);
        let qs = bf16s(rows * qh * hd, 15);
        let ks = bf16s(rows * kh * hd, 16);
        let pos = positions(rows * 3, 17);
        let mut q = gpu.tensor(&qs, rows as u32, (qh * hd) as u32, Dtype::Bf16);
        let mut k = gpu.tensor(&ks, rows as u32, (kh * hd) as u32, Dtype::Bf16);
        let staged = win.map(|(n, at)| gpu.ints(&[n as i32, at as i32, 0, 0]));
        if let Some(s) = &staged {
            ctx.arm_stage(s.ptr);
        }
        let f = match form {
            Form::Interleaved => rope_mrope::interleaved,
            Form::Blocked => rope_mrope::blocked,
            Form::Split => rope_mrope::split,
        };
        f(
            &ctx,
            &mut q,
            &mut k,
            gpu.ints_shaped(&pos, rows as u32, 3),
            sections.map(|s| s as u32),
            rot_dim as u32,
            hd as u32,
            theta,
        )
        .unwrap();
        ctx.disarm_stage();
        let rot = |head: &mut [f32], n: usize| {
            if win.is_some_and(|(len, at)| !(at..at + len).contains(&n)) {
                return;
            }
            let p = [pos[3 * n], pos[3 * n + 1], pos[3 * n + 2]];
            mrope_head(head, p, form, sections, rot_dim, theta);
        };
        let what = format!("rope_mrope {form:?} hd={hd} rot={rot_dim} sections={sections:?}");
        close_rot(&what, &gpu.read(&q), &per_head(&qs, hd, qh, rot));
        close_rot(&what, &gpu.read(&k), &per_head(&ks, hd, kh, rot));
    }
}
