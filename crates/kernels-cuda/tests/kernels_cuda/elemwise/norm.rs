#![cfg(feature = "cuda")]

//! Norm and rotation launchers against the host: `rope::partial_last`,
//! `norm_rope::norm_rope_last` (on a staged window), the gated RMS norm and
//! Kimi-K3's AttnRes blend, and a row norm the ranks split.

use crate::common::attn_ref::{YARN, dot, landed, plane, rmsnorm, ten, turn, wide};
use crate::common::{Gpu, Lcg, host::upload_to, to_bf16};
use dtype::Dtype::{Bf16, F32, I32};
use kernels_cuda::elemwise::{modulate, norm, norm_rope, rope};

/// 37 rows of `heads` heads of `d`, normed first where `normed` (`Some(true)`:
/// scaled), the last 64 dims turned; on rows `base..` where `base` is not 0.
fn tail(
    heads: usize,
    d: usize,
    normed: Option<bool>,
    [interleaved, inverse, yarn]: [bool; 3],
    base: u32,
) {
    let (rows, live, yarn) = (37usize, 37 - base, yarn.then_some(YARN));
    let mut lcg = Lcg::seeded(0x40e + d as u64);
    let x_bits: Vec<u16> = (0..rows * heads * d)
        .map(|_| to_bf16(lcg.unit() * 3.0))
        .collect();
    let w_bits: Vec<u16> = (0..d).map(|_| to_bf16(0.5 + lcg.unit())).collect();
    let (x, w) = (wide(&x_bits), wide(&w_bits));
    let pos: Vec<i32> = (0..rows as i32).map(|r| r * 997 + 5).collect();
    let scale = |scaled: bool| scaled.then_some(&w[..]);
    let mut want = normed.map_or(x.clone(), |s| rmsnorm(&x, d, scale(s), 1e-6));
    for (i, head) in want.chunks_mut(d).enumerate() {
        turn(head, pos[i / heads], 64, interleaved, inverse, yarn);
    }
    // A row before the window keeps its fill.
    want[..base as usize * heads * d].fill(1.0);
    let mut gpu = Gpu::open();
    let (x_at, w_at, pos_at) = (gpu.up(&x_bits), gpu.up(&w_bits), gpu.up(&pos));
    let (ctx, width, d32) = (gpu.ctx(), heads * d, d as u32);
    let y_at = gpu.up(&vec![0x3f80u16; rows * width]);
    if base > 0 {
        ctx.arm_stage(gpu.up(&[live, base, 0, 0]));
    }
    let (y, x_t) = (
        &mut ten(y_at, live as usize, width, Bf16),
        ten(x_at, live as usize, width, Bf16),
    );
    let (pos_t, rot) = (ten(pos_at, rows, 1, I32), (interleaved, inverse));
    let fired = match normed {
        Some(scaled) => {
            let w_t = scaled.then(|| ten(w_at, d, 1, Bf16));
            norm_rope::norm_rope_last(
                &ctx, x_t, w_t, d32, 1e-6, pos_t, 64, 1e4, rot.0, rot.1, yarn, y,
            )
        }
        None => {
            upload_to(y_at, &x_bits);
            rope::partial_last(&ctx, y, pos_t, 64, d32, 1e4, interleaved, inverse, yarn)
        }
    };
    fired.expect("the rotation fires");
    let what = format!("{heads} heads of {d}, norm {normed:?}, {rot:?} {yarn:?}, from {base}");
    landed(&gpu, y_at, &want, 2e-2, 1.0, &what);
}

#[test]
fn the_tail_rotations_answer_the_host() {
    // The rotation alone: half-split, interleaved and inverse, YaRN-scaled.
    tail(64, 512, None, [false, false, false], 0);
    tail(64, 512, None, [true, true, false], 0);
    tail(64, 512, None, [false, false, true], 0);
    // q's scale-free norm turned back, the kv plane's and the index keys'
    // scaled ones, a 256-wide head, a staged window.
    tail(64, 512, Some(false), [true, true, true], 5);
    tail(1, 512, Some(true), [true, false, true], 5);
    tail(1, 128, Some(true), [true, false, false], 0);
    tail(4, 256, Some(false), [true, false, true], 0);
}

/// `w * x / rms(x) * gate'` (the gate's SiLU or sigmoid), per head or per
/// row, at widths a warp carries alone and one it does not, over an f32 x
/// or one landed in bf16.
#[test]
fn the_gated_norm_answers_the_host() {
    let cases = [
        (5, 512, 128, false, false),
        (3, 512, 256, true, true),
        (4, 256, 0, false, true),
        (3, 288, 96, false, false),
        (37, 192, 64, true, true),
    ];
    for (rows, width, head, sigmoid, narrow) in cases {
        let (unit, mut lcg) = (if head == 0 { width } else { head }, Lcg::seeded(0x9a7e));
        let x_bits: Vec<u16> = (0..rows * width)
            .map(|_| to_bf16(3.0 * lcg.unit()))
            .collect();
        let (gate_raw, gate) = lcg.row(rows * width);
        let w: Vec<f32> = (0..unit).map(|_| 1.0 + 0.5 * lcg.unit()).collect();
        let mut want = rmsnorm(&wide(&x_bits), unit, Some(&w), 1e-6);
        for (y, g) in want.iter_mut().zip(&gate) {
            *y *= [*g, 1.0][usize::from(sigmoid)] / (1.0 + (-g).exp());
        }
        let mut gpu = Gpu::open();
        let x_f32 = ten(gpu.up(&wide(&x_bits)), rows, width, F32);
        let x = if narrow {
            ten(gpu.up(&x_bits), rows, width, Bf16)
        } else {
            x_f32
        };
        let (y_at, w_t) = (gpu.zeros(rows * width * 2), ten(gpu.up(&w), 1, unit, F32));
        let (g_t, y) = (
            ten(gpu.up(&gate_raw), rows, width, Bf16),
            &mut ten(y_at, rows, width, Bf16),
        );
        let fired =
            norm::rmsnorm_gated(&gpu.ctx(), x, g_t, Some(w_t), head as u32, 1e-6, sigmoid, y);
        fired.expect("the gated norm fires");
        landed(
            &gpu,
            y_at,
            &want,
            3e-2,
            1.0,
            &format!("{rows}x{width} head {head}"),
        );
    }
}

/// The scaleless norm a narrow head at a time (Muse Spark's 64-wide heads,
/// and 128), several heads to a warp as B200 runs it.
#[test]
fn the_narrow_head_norm_answers_the_host() {
    for head in [64, 128] {
        let (rows, width, mut lcg) = (37, 3 * head, Lcg::seeded(0x64));
        let x_bits: Vec<u16> = (0..rows * width)
            .map(|_| to_bf16(3.0 * lcg.unit()))
            .collect();
        let want = rmsnorm(&wide(&x_bits), head, None, 1e-6);
        let mut gpu = Gpu::open();
        let (x, y_at) = (
            ten(gpu.up(&x_bits), rows, width, Bf16),
            gpu.zeros(rows * width * 2),
        );
        let vec = kernels_cuda::tuning::Tuning {
            rows_vec: true,
            ..kernels_cuda::tuning::Tuning::DEFAULT
        };
        let ctx = gpu.ctx().with_tuning(vec);
        let fired = norm::rmsnorm_no_scale(
            &ctx,
            x,
            head as u32,
            1e-6,
            &mut ten(y_at, rows, width, Bf16),
        );
        fired.expect("the norm fires");
        landed(&gpu, y_at, &want, 3e-2, 1.0, &format!("head {head}"));
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
        let (norm_t, nw) = plane(&mut gpu, &mut lcg, 1, hidden);
        let (proj_t, pw) = plane(&mut gpu, &mut lcg, 1, hidden);
        let (out_t, ow) = plane(&mut gpu, &mut lcg, 1, hidden);
        let mut want = vec![0f32; tokens * hidden];
        for (t, out) in want.chunks_mut(hidden).enumerate() {
            let at = |r: &(_, Vec<f32>)| r.1[t * hidden..(t + 1) * hidden].to_vec();
            let logits: Vec<f32> = rows
                .iter()
                .map(|r| dot(&rmsnorm(&at(r), hidden, Some(&nw), 1e-5), &pw))
                .collect();
            let m = logits.iter().copied().fold(f32::MIN, f32::max);
            let e: Vec<f32> = logits.iter().map(|l| (l - m).exp()).collect();
            for (r, p) in rows.iter().zip(&e) {
                let z: f32 = e.iter().sum();
                out.iter_mut().zip(at(r)).for_each(|(y, x)| *y += p / z * x);
            }
        }
        let prefix = rows[blocks].0;
        let closed: Vec<_> = rows[..blocks].iter().map(|r| r.0).collect();
        let (ctx, y_at, z_at) = (
            gpu.ctx(),
            gpu.zeros(tokens * hidden * 2),
            gpu.zeros(tokens * hidden * 2),
        );
        let (y, z) = (
            &mut ten(y_at, tokens, hidden, Bf16),
            &mut ten(z_at, tokens, hidden, Bf16),
        );
        modulate::res_blend(&ctx, prefix, &closed, norm_t, 1e-5, proj_t, y)
            .expect("the blend fires");
        let fused =
            modulate::res_blend_norm(&ctx, prefix, &closed, norm_t, 1e-5, proj_t, out_t, 1e-6, z);
        let what = format!("{tokens} x {hidden}, {blocks} blocks");
        landed(&gpu, y_at, &want, 3e-2, 1.0, &what);
        if fused.expect("the fused blend fires") {
            landed(
                &gpu,
                z_at,
                &rmsnorm(&want, hidden, Some(&ow), 1e-6),
                3e-2,
                1.0,
                &what,
            );
        }
    }
}

#[test]
fn the_wide_row_norm_answers_the_host() {
    // A row in 16-byte runs a thread (one to four of them), its weight plain
    // and plus one, at Gemma 4's 2816 and the widest it takes.
    for (rows, width, plus_one) in [
        (3, 2816, false),
        (5, 2816, true),
        (2, 4096, false),
        (4, 8192, true),
        (7, 1032, false),
    ] {
        let mut lcg = Lcg::seeded(0x77 + width as u64);
        let x_bits: Vec<u16> = (0..rows * width)
            .map(|_| to_bf16(3.0 * lcg.unit()))
            .collect();
        let w_bits: Vec<u16> = (0..width).map(|_| to_bf16(lcg.unit())).collect();
        let w = wide(&w_bits);
        let w_eff: Vec<f32> = w
            .iter()
            .map(|v| if plus_one { v + 1.0 } else { *v })
            .collect();
        let want = rmsnorm(&wide(&x_bits), width, Some(&w_eff), 1e-6);
        let mut gpu = Gpu::open();
        let (x, wt, y_at) = (
            ten(gpu.up(&x_bits), rows, width, Bf16),
            ten(gpu.up(&w_bits), 1, width, Bf16),
            gpu.zeros(rows * width * 2),
        );
        let vec = kernels_cuda::tuning::Tuning {
            rows_vec: true,
            ..kernels_cuda::tuning::Tuning::DEFAULT
        };
        let ctx = gpu.ctx().with_tuning(vec);
        let mut y = ten(y_at, rows, width, Bf16);
        let fired = match plus_one {
            true => norm::rmsnorm_plus_one(&ctx, x, wt, 1e-6, &mut y),
            false => norm::rmsnorm(&ctx, x, wt, 1e-6, &mut y),
        };
        fired.expect("the norm fires");
        landed(
            &gpu,
            y_at,
            &want,
            3e-2,
            1.0,
            &format!("width {width} plus_one {plus_one}"),
        );
    }
}

/// A row norm the ranks split: each of `ranks` shares of a row `ranks *
/// share` wide sums its squares, the sums are added (as the ranks' all-reduce
/// adds them, each share counted `alike` times where a group of ranks holds it
/// alike), and each share is normed by them: side by side the shares are the
/// whole row's norm, unscaled, scaled or scaled by one plus the weight.
fn shared_row(ranks: usize, share: usize, alike: usize, form: Option<bool>) {
    let (rows, width) = (37usize, ranks * share);
    let mut lcg = Lcg::seeded(0x5ad + width as u64);
    let x_bits: Vec<u16> = (0..rows * width)
        .map(|_| to_bf16(lcg.unit() * 3.0))
        .collect();
    let w_bits: Vec<u16> = (0..width).map(|_| to_bf16(lcg.unit() - 0.5)).collect();
    let (x, w) = (wide(&x_bits), wide(&w_bits));
    let scale: Option<Vec<f32>> =
        form.map(|plus| w.iter().map(|v| if plus { 1.0 + v } else { *v }).collect());
    let want = rmsnorm(&x, width, scale.as_deref(), 1e-6);
    let mut gpu = Gpu::open();
    let ctx = gpu.ctx();
    let shares: Vec<(Vec<u16>, Vec<u16>)> = (0..ranks)
        .map(|r| {
            let x: Vec<u16> = (0..rows)
                .flat_map(|t| x_bits[t * width + r * share..][..share].iter().copied())
                .collect();
            (x, w_bits[r * share..][..share].to_vec())
        })
        .collect();
    let mut sums = vec![0f32; rows];
    let mut ats = Vec::new();
    for (x_share, w_share) in &shares {
        let (x_at, w_at) = (gpu.up(x_share), gpu.up(w_share));
        let ss_at = gpu.up(&vec![0f32; rows]);
        norm::sum_squares(
            &ctx,
            ten(x_at, rows, share, Bf16),
            &mut ten(ss_at, rows, 1, F32),
        )
        .expect("the sum fires");
        gpu.sync();
        for (s, v) in sums.iter_mut().zip(gpu.down::<f32>(ss_at, rows)) {
            *s += v * alike as f32;
        }
        ats.push((x_at, w_at));
    }
    let ss_at = gpu.up(&sums);
    let mut got = vec![0f32; rows * width];
    for (r, (x_at, w_at)) in ats.into_iter().enumerate() {
        let y_at = gpu.up(&vec![0u16; rows * share]);
        norm::rmsnorm_by(
            &ctx,
            ten(x_at, rows, share, Bf16),
            ten(ss_at, rows, 1, F32),
            form.map(|_| ten(w_at, share, 1, Bf16)),
            form == Some(true),
            (width * alike) as u32,
            1e-6,
            &mut ten(y_at, rows, share, Bf16),
        )
        .expect("the norm fires");
        gpu.sync();
        let y = wide(&gpu.down::<u16>(y_at, rows * share));
        for t in 0..rows {
            got[t * width + r * share..][..share].copy_from_slice(&y[t * share..][..share]);
        }
    }
    let what = format!("{ranks} shares of {share}, each {alike} times, weight {form:?}");
    crate::common::attn_ref::near(&got, &want, 2e-2, 1.0, &what);
}

#[test]
fn a_row_norm_the_ranks_split_answers_the_whole_rows() {
    shared_row(2, 3072, 1, Some(false));
    shared_row(8, 768, 1, Some(true));
    shared_row(4, 256, 2, None);
    shared_row(1, 1000, 1, Some(false));
}
