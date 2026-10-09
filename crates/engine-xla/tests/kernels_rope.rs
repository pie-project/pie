//! The rotary family (rope, rope_mrope, act::rope_axes, the fused q norm +
//! rope) against host references.

use dtype::Dtype;
use engine_xla::bench::{Bench, assert_close, round_bf16};
use kernels_xla::elemwise::{act, rope, rope_mrope};

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

/// One rotated pair of a `unit`-wide run: `(lo, hi, axis, freq)`.
type Pair = (usize, usize, usize, f32);

/// Rotates every `unit`-wide run of each row by `pairs`, angles `pos[row][axis]·freq`,
/// cos/sin scaled by `m`; rounds to bf16.
fn turn_ref(
    x: &[f32],
    width: usize,
    unit: usize,
    pairs: &[Pair],
    pos: &[f32],
    axes: usize,
    m: f32,
) -> Vec<f32> {
    let mut out = x.to_vec();
    for (row, chunk) in out.chunks_mut(width).enumerate() {
        for run in chunk.chunks_mut(unit) {
            let src = run.to_vec();
            for &(lo, hi, axis, f) in pairs {
                let ang = pos[row * axes + axis] * f;
                let (s, c) = ang.sin_cos();
                let (a, b) = (src[lo], src[hi]);
                run[lo] = a * (c * m) - b * (s * m);
                run[hi] = b * (c * m) + a * (s * m);
            }
        }
    }
    out.into_iter().map(round_bf16).collect()
}

fn inv_freq(theta: f32, i: usize, span: u32) -> f32 {
    theta.powf(-2.0 * i as f32 / span as f32)
}

fn neox_pairs(
    hd: usize,
    rot: usize,
    theta: f32,
    span: u32,
    interleaved: bool,
    offset: usize,
) -> Vec<Pair> {
    (0..rot / 2)
        .map(|i| {
            let (lo, hi) = if interleaved {
                (2 * i, 2 * i + 1)
            } else {
                (i, i + rot / 2)
            };
            let _ = hd;
            (offset + lo, offset + hi, 0, inv_freq(theta, i, span))
        })
        .collect()
}

fn positions(rows: usize) -> Vec<i32> {
    (0..rows).map(|r| 700 + 331 * r as i32).collect()
}

#[test]
fn partial_rope_turns_only_the_rotary_pairs_of_q_and_k() {
    let (rows, hd, rot, theta) = (5usize, 256usize, 64usize, 1e7f32);
    let (qh, kh) = (3usize, 1usize);
    let qs = data(rows * qh * hd, 1);
    let ks = data(rows * kh * hd, 2);
    let pos = positions(rows);
    let mut b = Bench::new();
    let q = b.bf16(rows as u32, (qh * hd) as u32, &qs);
    let k = b.bf16(rows as u32, (kh * hd) as u32, &ks);
    let p = b.i32(rows as u32, 1, &pos);
    let q2 = b.bf16(rows as u32, (qh * hd) as u32, &qs);
    if !b
        .run(|ctx| {
            rope::partial(ctx, q, k, p, rot as u32, hd as u32, theta)?;
            rope::partial_q(ctx, q2, p, rot as u32, hd as u32, theta)
        })
        .unwrap()
    {
        return;
    }
    let pf: Vec<f32> = pos.iter().map(|&v| v as f32).collect();
    let pairs = neox_pairs(hd, rot, theta, rot as u32, false, 0);
    let wq = turn_ref(&qs, qh * hd, hd, &pairs, &pf, 1, 1.0);
    let wk = turn_ref(&ks, kh * hd, hd, &pairs, &pf, 1, 1.0);
    assert_close(&b.read_f32(q), &wq, 1e-2, 1e-2);
    assert_close(&b.read_f32(k), &wk, 1e-2, 1e-2);
    assert_close(&b.read_f32(q2), &wq, 1e-2, 1e-2);
    // The tail past the rotary prefix passes through bit for bit.
    let got = b.read_f32(q);
    for (i, (&g, &x)) in got.iter().zip(&qs).enumerate() {
        if i % hd >= rot {
            assert_eq!(g.to_bits(), x.to_bits(), "tail element {i}");
        }
    }
}

#[test]
fn full_rope_turns_halves_and_interleaved_pairs() {
    let (rows, hd, theta) = (3usize, 96usize, 10000f32);
    let qs = data(rows * 2 * hd, 3);
    let ks = data(rows * hd, 4);
    let pos = positions(rows);
    let mut b = Bench::new();
    let q = b.bf16(rows as u32, (2 * hd) as u32, &qs);
    let k = b.bf16(rows as u32, hd as u32, &ks);
    let qi = b.bf16(rows as u32, (2 * hd) as u32, &qs);
    let ki = b.bf16(rows as u32, hd as u32, &ks);
    let p = b.i32(rows as u32, 1, &pos);
    if !b
        .run(|ctx| {
            rope::full(ctx, q, k, p, hd as u32, theta, false)?;
            rope::full(ctx, qi, ki, p, hd as u32, theta, true)
        })
        .unwrap()
    {
        return;
    }
    let pf: Vec<f32> = pos.iter().map(|&v| v as f32).collect();
    let n = neox_pairs(hd, hd, theta, hd as u32, false, 0);
    let il = neox_pairs(hd, hd, theta, hd as u32, true, 0);
    assert_close(
        &b.read_f32(q),
        &turn_ref(&qs, 2 * hd, hd, &n, &pf, 1, 1.0),
        1e-2,
        1e-2,
    );
    assert_close(
        &b.read_f32(k),
        &turn_ref(&ks, hd, hd, &n, &pf, 1, 1.0),
        1e-2,
        1e-2,
    );
    assert_close(
        &b.read_f32(qi),
        &turn_ref(&qs, 2 * hd, hd, &il, &pf, 1, 1.0),
        1e-2,
        1e-2,
    );
    assert_close(
        &b.read_f32(ki),
        &turn_ref(&ks, hd, hd, &il, &pf, 1, 1.0),
        1e-2,
        1e-2,
    );
}

fn ramp(head_dim: u32, theta: f32, fast: f32, slow: f32, orig: u32) -> (f32, f32) {
    let tau = std::f32::consts::TAU;
    let corr = |rot: f32| head_dim as f32 * (orig as f32 / (rot * tau)).ln() / (2.0 * theta.ln());
    let low = corr(fast).floor().max(0.0);
    let high = corr(slow).ceil().min((head_dim / 2) as f32 - 1.0).max(low);
    (low, high)
}

fn bend(f: f32, factor: f32, low: f32, high: f32, i: usize) -> f32 {
    let denom = if high == low {
        high + 1e-3 - low
    } else {
        high - low
    };
    let r = ((i as f32 - low) / denom).clamp(0.0, 1.0);
    f * ((1.0 - r) + r / factor)
}

#[test]
fn the_tail_rope_turns_inverse_interleaved_and_yarn_ramped() {
    let (rows, hd, rot, theta) = (4usize, 128usize, 64usize, 10000f32);
    let xs = data(rows * 2 * hd, 5);
    let pos = positions(rows);
    let yarn = rope::Yarn {
        factor: 4.0,
        beta_fast: 32.0,
        beta_slow: 1.0,
        original_max_position: 4096,
    };
    let mut b = Bench::new();
    let p = b.i32(rows as u32, 1, &pos);
    let w = (2 * hd) as u32;
    let a = b.bf16(rows as u32, w, &xs);
    let c = b.bf16(rows as u32, w, &xs);
    let d = b.bf16(rows as u32, w, &xs);
    if !b
        .run(|ctx| {
            rope::partial_last(ctx, a, p, rot as u32, hd as u32, theta, false, false, None)?;
            rope::partial_last(ctx, c, p, rot as u32, hd as u32, theta, true, true, None)?;
            rope::partial_last(
                ctx,
                d,
                p,
                rot as u32,
                hd as u32,
                theta,
                false,
                false,
                Some(yarn),
            )
        })
        .unwrap()
    {
        return;
    }
    let pf: Vec<f32> = pos.iter().map(|&v| v as f32).collect();
    let off = hd - rot;
    let plain = neox_pairs(hd, rot, theta, rot as u32, false, off);
    let inv: Vec<Pair> = neox_pairs(hd, rot, theta, rot as u32, true, off)
        .into_iter()
        .map(|(l, h, a, f)| (l, h, a, -f))
        .collect();
    let (low, high) = ramp(rot as u32, theta, 32.0, 1.0, 4096);
    let ramped: Vec<Pair> = plain
        .iter()
        .enumerate()
        .map(|(i, &(l, h, a, f))| (l, h, a, bend(f, 4.0, low, high, i)))
        .collect();
    assert_close(
        &b.read_f32(a),
        &turn_ref(&xs, 2 * hd, hd, &plain, &pf, 1, 1.0),
        1e-2,
        1e-2,
    );
    assert_close(
        &b.read_f32(c),
        &turn_ref(&xs, 2 * hd, hd, &inv, &pf, 1, 1.0),
        1e-2,
        1e-2,
    );
    assert_close(
        &b.read_f32(d),
        &turn_ref(&xs, 2 * hd, hd, &ramped, &pf, 1, 1.0),
        1e-2,
        1e-2,
    );
}

#[test]
fn yarn_rope_bends_frequencies_and_scales_both_halves() {
    let (rows, hd, theta) = (3usize, 64usize, 10000f32);
    let qs = data(rows * 2 * hd, 6);
    let ks = data(rows * hd, 7);
    let pos = positions(rows);
    let mut b = Bench::new();
    let p = b.i32(rows as u32, 1, &pos);
    let q = b.bf16(rows as u32, (2 * hd) as u32, &qs);
    let k = b.bf16(rows as u32, hd as u32, &ks);
    let qi = b.bf16(rows as u32, (2 * hd) as u32, &qs);
    let ki = b.bf16(rows as u32, hd as u32, &ks);
    if !b
        .run(|ctx| {
            rope::yarn(
                ctx, q, k, p, hd as u32, theta, 40.0, 32.0, 1.0, 1.2, 4096, false,
            )?;
            rope::yarn(
                ctx, qi, ki, p, hd as u32, theta, 40.0, 32.0, 1.0, 1.2, 4096, true,
            )
        })
        .unwrap()
    {
        return;
    }
    let pf: Vec<f32> = pos.iter().map(|&v| v as f32).collect();
    let (low, high) = ramp(hd as u32, theta, 32.0, 1.0, 4096);
    let mk = |il: bool| -> Vec<Pair> {
        neox_pairs(hd, hd, theta, hd as u32, il, 0)
            .into_iter()
            .enumerate()
            .map(|(i, (l, h, a, f))| (l, h, a, bend(f, 40.0, low, high, i)))
            .collect()
    };
    let (n, il) = (mk(false), mk(true));
    assert_close(
        &b.read_f32(q),
        &turn_ref(&qs, 2 * hd, hd, &n, &pf, 1, 1.2),
        1e-2,
        1e-2,
    );
    assert_close(
        &b.read_f32(k),
        &turn_ref(&ks, hd, hd, &n, &pf, 1, 1.2),
        1e-2,
        1e-2,
    );
    assert_close(
        &b.read_f32(qi),
        &turn_ref(&qs, 2 * hd, hd, &il, &pf, 1, 1.2),
        1e-2,
        1e-2,
    );
    assert_close(
        &b.read_f32(ki),
        &turn_ref(&ks, hd, hd, &il, &pf, 1, 1.2),
        1e-2,
        1e-2,
    );
}

#[test]
fn mrope_reads_each_pairs_axis_in_all_three_forms() {
    let (rows, hd, rot, theta) = (3usize, 128usize, 96usize, 1e6f32);
    let sections = [16u32, 16, 16];
    let qs = data(rows * 2 * hd, 8);
    let ks = data(rows * hd, 9);
    let pos: Vec<i32> = (0..rows * 3).map(|i| 40 + 97 * i as i32 % 500).collect();
    let mut b = Bench::new();
    let p = b.i32(rows as u32, 3, &pos);
    let mut planes = Vec::new();
    for _ in 0..3 {
        planes.push((
            b.bf16(rows as u32, (2 * hd) as u32, &qs),
            b.bf16(rows as u32, hd as u32, &ks),
        ));
    }
    if !b
        .run(|ctx| {
            rope_mrope::interleaved(
                ctx,
                planes[0].0,
                planes[0].1,
                p,
                sections,
                rot as u32,
                hd as u32,
                theta,
            )?;
            rope_mrope::blocked(
                ctx,
                planes[1].0,
                planes[1].1,
                p,
                sections,
                rot as u32,
                hd as u32,
                theta,
            )?;
            rope_mrope::split(
                ctx,
                planes[2].0,
                planes[2].1,
                p,
                sections,
                rot as u32,
                hd as u32,
                theta,
            )
        })
        .unwrap()
    {
        return;
    }
    let pf: Vec<f32> = pos.iter().map(|&v| v as f32).collect();
    let half = hd / 2;
    let s: Vec<usize> = sections.iter().map(|&v| v as usize).collect();
    let total = s.iter().sum::<usize>();
    let inter: Vec<Pair> = (0..rot / 2)
        .map(|i| {
            let axis = if i % 3 == 1 && i < 3 * s[1] {
                1
            } else if i % 3 == 2 && i < 3 * s[2] {
                2
            } else {
                0
            };
            // The rotated prefix is its own neox head (upstream's partial
            // rotation), not pairs across the whole head.
            (i, i + rot / 2, axis, inv_freq(theta, i, rot as u32))
        })
        .collect();
    let sect = |i: usize| -> (usize, usize, usize) {
        if i < s[0] {
            (0, i, 0)
        } else if i < s[0] + s[1] {
            (1, i - s[0], s[0])
        } else {
            (2, i - s[0] - s[1], s[0] + s[1])
        }
    };
    let pairs = (rot / 2).min(total);
    let blocked: Vec<Pair> = (0..pairs)
        .map(|i| {
            let (axis, within, _) = sect(i);
            (i, i + half, axis, inv_freq(theta, within, total as u32))
        })
        .collect();
    let split: Vec<Pair> = (0..pairs)
        .map(|i| {
            let (axis, within, before) = sect(i);
            let lo = 2 * before + within;
            (
                lo,
                lo + s[axis],
                axis,
                theta.powf(-(within as f32) / s[axis] as f32),
            )
        })
        .collect();
    for (plane, pairs) in planes.iter().zip([&inter, &blocked, &split]) {
        assert_close(
            &b.read_f32(plane.0),
            &turn_ref(&qs, 2 * hd, hd, pairs, &pf, 3, 1.0),
            1e-2,
            1e-2,
        );
        assert_close(
            &b.read_f32(plane.1),
            &turn_ref(&ks, hd, hd, pairs, &pf, 3, 1.0),
            1e-2,
            1e-2,
        );
    }
}

#[test]
fn rope_axes_turns_each_form_and_the_ladder() {
    let rows = 3usize;
    let (hd, rot) = (96usize, 64usize);
    let dims = [16u32, 24, 24, 0];
    let thetas = [10000f32, 500.0, 2000.0, 1.0];
    let xs = data(rows * 2 * hd, 10);
    let pos: Vec<f32> = (0..rows * 3).map(|i| 0.25 + 3.5 * i as f32).collect();
    let mut b = Bench::new();
    let p = b.f32(rows as u32, 3, &pos);
    let w = (2 * hd) as u32;
    let src = b.bf16(rows as u32, w, &xs);
    let outs: Vec<_> = (0..3)
        .map(|_| b.zeros(Dtype::Bf16, rows as u32, w))
        .collect();
    // Ladder: 2 heads of 64, rotary 64, three flat axes of 16 → pad 40.
    let (lhd, lrot) = (64usize, 64usize);
    let ldims = [16u32, 16, 16, 0];
    let lx = data(rows * 2 * lhd, 11);
    let lsrc = b.bf16(rows as u32, (2 * lhd) as u32, &lx);
    let forms = [
        act::RopeForm::Interleaved,
        act::RopeForm::Neox,
        act::RopeForm::Split,
    ];
    if !b
        .run(|ctx| {
            for (f, o) in forms.iter().zip(&outs) {
                act::rope_axes(ctx, src, p, dims, thetas, *f, rot as u32, hd as u32, *o)?;
            }
            act::rope_axes(
                ctx,
                lsrc,
                p,
                ldims,
                thetas,
                act::RopeForm::SplitLadder,
                lrot as u32,
                lhd as u32,
                lsrc,
            )
        })
        .unwrap()
    {
        return;
    }
    let tp = |base: f32, e: f32| (e * base.ln()).exp();
    let angles = rot / 2;
    for (f, o) in forms.iter().zip(&outs) {
        let mut pairs = Vec::new();
        let (mut axis, mut fa, mut fc) = (0usize, 0usize, 0usize);
        for angle in 0..angles {
            while angle >= fa + dims[axis] as usize / 2 {
                fa += dims[axis] as usize / 2;
                fc += dims[axis] as usize;
                axis += 1;
            }
            let within = angle - fa;
            let fr = tp(thetas[axis], -2.0 * within as f32 / dims[axis] as f32);
            let (lo, hi) = match f {
                act::RopeForm::Interleaved => (fc + 2 * within, fc + 2 * within + 1),
                act::RopeForm::Neox => (angle, angle + angles),
                _ => (fc + within, fc + dims[axis] as usize / 2 + within),
            };
            pairs.push((lo, hi, axis, fr));
        }
        assert_close(
            &b.read_f32(*o),
            &turn_ref(&xs, 2 * hd, hd, &pairs, &pos, 3, 1.0),
            1e-2,
            1e-2,
        );
    }
    let la = lrot / 2;
    let pad = (2 * lrot - 48) / 2;
    let mut lp = Vec::new();
    for head in 0..2 {
        for angle in 0..la {
            let idx = head * la + angle;
            if idx < pad {
                continue;
            }
            let slot = idx - pad;
            let axis = slot % 3;
            let fi = slot / 3;
            let ladder = ldims[axis] as usize / 2;
            let e = fi as f32 / (ladder - 1) as f32;
            let lo = head * lhd + angle;
            lp.push((lo, lo + la, axis, tp(thetas[axis], e)));
        }
    }
    assert_close(
        &b.read_f32(lsrc),
        &turn_ref(&lx, 2 * lhd, 2 * lhd, &lp, &pos, 3, 1.0),
        2e-2,
        1e-2,
    );
}
