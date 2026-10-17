#![cfg(feature = "cuda")]

//! Norm and rotation launchers against the host: `rope::partial_last`,
//! `rope::norm_rope_last` (both forms, on a staged window), the gated RMS
//! norm and Kimi-K3's AttnRes blend.

use crate::common::attn_ref::{YARN, near, plane, rmsnorm, ten, tuned, turn, wide};
use crate::common::{Gpu, Lcg, host::upload_to, to_bf16};
use dtype::Dtype::{Bf16, F32, I32};
use kernels_cuda::elemwise::{norm, rope};

/// 37 rows of `heads` heads of `d`, normed first where `normed` (`Some(true)`:
/// scaled), the last 64 dims turned; `window` is `(live, base)`.
fn tail(
    heads: usize,
    d: usize,
    normed: Option<bool>,
    [interleaved, inverse, yarn]: [bool; 3],
    window: Option<(u32, u32)>,
    vec: bool,
) {
    let (rows, (live, base), yarn) = (37usize, window.unwrap_or((37, 0)), yarn.then_some(YARN));
    let mut lcg = Lcg::seeded(0x40e + d as u64);
    let x_bits: Vec<u16> = (0..rows * heads * d)
        .map(|_| to_bf16(lcg.unit() * 3.0))
        .collect();
    let w_bits: Vec<u16> = (0..d).map(|_| to_bf16(0.5 + lcg.unit())).collect();
    let (x, w, pos) = (
        wide(&x_bits),
        wide(&w_bits),
        (0..rows as i32).map(|r| r * 997 + 5).collect::<Vec<_>>(),
    );
    let mut want = normed.map_or(x.clone(), |scaled| {
        rmsnorm(&x, d, scaled.then_some(&w[..]), 1e-6)
    });
    for (i, head) in want.chunks_mut(d).enumerate() {
        turn(head, pos[i / heads], 64, interleaved, inverse, yarn);
    }
    let mut gpu = Gpu::open();
    let (x_at, w_at, y_at, pos_at) = (
        gpu.up(&x_bits),
        gpu.up(&w_bits),
        gpu.up(&vec![0x3f80u16; rows * heads * d]),
        gpu.up(&pos),
    );
    let (ctx, width) = (
        gpu.ctx().with_tuning(tuned(|t| t.rows_vec = vec)),
        heads * d,
    );
    if window.is_some() {
        ctx.arm_stage(gpu.up(&[live, base, 0, 0]));
    }
    let (mut y, x_t, pos_t, d32) = (
        ten(y_at, live as usize, width, Bf16),
        ten(x_at, live as usize, width, Bf16),
        ten(pos_at, rows, 1, I32),
        d as u32,
    );
    let fired = match normed {
        Some(scaled) => {
            let w_t = scaled.then(|| ten(w_at, d, 1, Bf16));
            rope::norm_rope_last(
                &ctx,
                x_t,
                w_t,
                d32,
                1e-6,
                pos_t,
                64,
                1e4,
                interleaved,
                inverse,
                yarn,
                &mut y,
            )
        }
        None => {
            upload_to(y_at, &x_bits);
            rope::partial_last(
                &ctx,
                &mut y,
                pos_t,
                64,
                d32,
                1e4,
                interleaved,
                inverse,
                yarn,
            )
        }
    };
    fired.expect("the rotation fires");
    gpu.sync();
    let got = wide(&gpu.down::<u16>(y_at, rows * width));
    let (a, b) = (base as usize * width, (base + live) as usize * width);
    let what = format!(
        "{heads} heads of {d}, norm {normed:?}, {interleaved} {inverse} {yarn:?}, vec {vec}, {window:?}"
    );
    near(&got[a..b], &want[a..b], 2e-2, 1.0, &what);
    let outside = got[..a].iter().chain(&got[b..]);
    assert!(
        normed.is_none() || outside.copied().all(|v| v == 1.0),
        "{what}: a row outside the window moved"
    );
}

#[test]
fn the_tail_rotations_answer_the_host() {
    // The rotation alone: half-split, interleaved and inverse, YaRN-scaled.
    tail(64, 512, None, [false, false, false], None, false);
    tail(64, 512, None, [true, true, false], None, false);
    tail(64, 512, None, [false, false, true], None, false);
    // q's scale-free norm turned back, the kv plane's and the index keys'
    // scaled ones, a 256-wide head: each form, a staged window.
    tail(64, 512, Some(false), [true, true, true], None, false);
    tail(
        64,
        512,
        Some(false),
        [true, true, true],
        Some((17, 5)),
        true,
    );
    tail(
        1,
        512,
        Some(true),
        [true, false, true],
        Some((17, 5)),
        false,
    );
    tail(1, 128, Some(true), [true, false, false], None, true);
    tail(4, 256, Some(false), [true, false, true], None, true);
}

/// `w * x / rms(x) * gate'` (the gate's SiLU or sigmoid), per head or per
/// row, at widths a warp carries alone and one it does not, over an f32 x
/// or one landed in bf16.
#[test]
fn the_gated_norm_answers_the_host() {
    for (rows, width, head, sigmoid, narrow) in [
        (5, 512, 128, false, false),
        (3, 512, 256, true, true),
        (4, 256, 0, false, true),
        (3, 288, 96, false, false),
    ] {
        let (unit, mut lcg) = (if head == 0 { width } else { head }, Lcg::seeded(0x9a7e));
        let x_bits: Vec<u16> = (0..rows * width)
            .map(|_| to_bf16(3.0 * lcg.unit()))
            .collect();
        let ((gate_raw, gate), w) = (
            lcg.row(rows * width),
            (0..unit)
                .map(|_| 1.0 + 0.5 * lcg.unit())
                .collect::<Vec<f32>>(),
        );
        let mut want = rmsnorm(&wide(&x_bits), unit, Some(&w), 1e-6);
        for (y, g) in want.iter_mut().zip(&gate) {
            *y *= if sigmoid { 1.0 } else { *g } / (1.0 + (-g).exp());
        }
        let mut gpu = Gpu::open();
        let x = if narrow {
            ten(gpu.up(&x_bits), rows, width, Bf16)
        } else {
            ten(gpu.up(&wide(&x_bits)), rows, width, F32)
        };
        let (y_at, g_t, w_t) = (
            gpu.zeros(rows * width * 2),
            ten(gpu.up(&gate_raw), rows, width, Bf16),
            ten(gpu.up(&w), 1, unit, F32),
        );
        let mut y = ten(y_at, rows, width, Bf16);
        norm::rmsnorm_gated(&gpu.ctx(), x, g_t, w_t, head as u32, 1e-6, sigmoid, &mut y)
            .expect("the gated norm fires");
        gpu.sync();
        near(
            &wide(&gpu.down::<u16>(y_at, rows * width)),
            &want,
            3e-2,
            1.0,
            &format!("{rows}x{width} head {head}"),
        );
    }
}

/// Each token's closed blocks and running prefix scored by the RMS-normed
/// dot with a projection, softmaxed and blended (read apart, at Kimi's
/// width, an odd one and none closed); and the blend's own RMS norm in the
/// same pass (`res_blend_norm`) where it takes the row.
#[test]
fn attn_res_blends_what_the_host_blends() {
    for (tokens, hidden, blocks) in [
        (37usize, 7168usize, 8usize),
        (5, 1001, 2),
        (3, 64, 0),
        (4, 2048, 1),
    ] {
        let (mut gpu, mut lcg) = (Gpu::open(), Lcg::seeded((tokens * hidden + blocks) as u64));
        let rows: Vec<_> = (0..=blocks)
            .map(|_| plane(&mut gpu, &mut lcg, tokens, hidden))
            .collect();
        let ((norm_t, nw), (proj_t, pw), (out_t, ow)) = (
            plane(&mut gpu, &mut lcg, 1, hidden),
            plane(&mut gpu, &mut lcg, 1, hidden),
            plane(&mut gpu, &mut lcg, 1, hidden),
        );
        let mut want = vec![0f32; tokens * hidden];
        for (t, out) in want.chunks_mut(hidden).enumerate() {
            let at = |r: &Vec<f32>| r[t * hidden..(t + 1) * hidden].to_vec();
            let logit = |r: &Vec<f32>| {
                rmsnorm(&at(r), hidden, Some(&nw), 1e-5)
                    .iter()
                    .zip(&pw)
                    .map(|(a, b)| a * b)
                    .sum::<f32>()
            };
            let logits: Vec<f32> = rows.iter().map(|r| logit(&r.1)).collect();
            let m = logits.iter().copied().fold(f32::MIN, f32::max);
            let (e, z) = (
                logits.iter().map(|l| (l - m).exp()).collect::<Vec<_>>(),
                logits.iter().map(|l| (l - m).exp()).sum::<f32>(),
            );
            for (r, p) in rows.iter().zip(&e) {
                out.iter_mut()
                    .zip(at(&r.1))
                    .for_each(|(y, x)| *y += p / z * x);
            }
        }
        let (prefix, closed) = (
            rows[blocks].0,
            rows[..blocks].iter().map(|r| r.0).collect::<Vec<_>>(),
        );
        let (ctx, y_at, z_at) = (
            gpu.ctx(),
            gpu.zeros(tokens * hidden * 2),
            gpu.zeros(tokens * hidden * 2),
        );
        let (mut y, mut z) = (
            ten(y_at, tokens, hidden, Bf16),
            ten(z_at, tokens, hidden, Bf16),
        );
        norm::res_blend(&ctx, prefix, &closed, norm_t, 1e-5, proj_t, &mut y)
            .expect("the blend fires");
        let fused = norm::res_blend_norm(
            &ctx, prefix, &closed, norm_t, 1e-5, proj_t, out_t, 1e-6, &mut z,
        );
        gpu.sync();
        let what = format!("{tokens} x {hidden}, {blocks} blocks");
        near(
            &wide(&gpu.down::<u16>(y_at, tokens * hidden)),
            &want,
            3e-2,
            1.0,
            &what,
        );
        if fused.expect("the fused blend fires") {
            near(
                &wide(&gpu.down::<u16>(z_at, tokens * hidden)),
                &rmsnorm(&want, hidden, Some(&ow), 1e-6),
                3e-2,
                1.0,
                &what,
            );
        }
    }
}
