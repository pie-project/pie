//! `ragged::ragged` against the host softmax, as kernels-cuda's ragged test:
//! each mask arm, each head width, cross tables with empty groups, and rows no
//! group holds (before the first, past the last) left as they were.

#![cfg(feature = "rocm")]

mod common;

use common::{Gpu, bf16s, close_abs, indptr_of};
use dtype::Dtype::{Bf16, F32, U8};
use kernels_rocm::attn::ragged::{RaggedMask, ragged};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Mask {
    None,
    Bias,
    SelfOnly,
    Tags,
    Class,
}

const MAX_LEN: usize = 130;
const REF_START: [i32; 6] = [20, 100, 200, -1, 0, 3];

/// The host softmax over a group's keys, `logit(g, h, r, j)` the bias a kept
/// key adds (rows and keys global). A row with no kept key lands zeros.
fn reference(
    [q, k, v]: [&[f32]; 3],
    q_indptr: &[i32],
    kv_indptr: &[i32],
    [q_heads, kv_heads, hd]: [usize; 3],
    sm_scale: f32,
    logit: impl Fn(usize, usize, usize, usize) -> Option<f32>,
    out: &mut [f32],
) {
    let group = q_heads / kv_heads;
    for g in 0..q_indptr.len() - 1 {
        let (q0, q1) = (q_indptr[g] as usize, q_indptr[g + 1] as usize);
        let (k0, k1) = (kv_indptr[g] as usize, kv_indptr[g + 1] as usize);
        for r in q0..q1 {
            for h in 0..q_heads {
                let kh = h / group;
                let qv = &q[(r * q_heads + h) * hd..][..hd];
                let (keys, mut scores): (Vec<usize>, Vec<f32>) = (k0..k1)
                    .filter_map(|j| {
                        let bias = logit(g, h, r, j)?;
                        let kv = &k[(j * kv_heads + kh) * hd..][..hd];
                        let dot = qv.iter().zip(kv).map(|(a, b)| a * b).sum::<f32>();
                        Some((j, dot * sm_scale + bias))
                    })
                    .unzip();
                let m = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0f32;
                for s in &mut scores {
                    *s = (*s - m).exp();
                    sum += *s;
                }
                let o = &mut out[(r * q_heads + h) * hd..][..hd];
                o.fill(0.0);
                for (&j, p) in keys.iter().zip(&scores) {
                    let vv = &v[(j * kv_heads + kh) * hd..][..hd];
                    for (d, x) in o.iter_mut().zip(vv) {
                        *d += p / sum * x;
                    }
                }
            }
        }
    }
}

/// `before` query rows ahead of the first group and `after` past the last
/// hold noise the arm must leave alone.
fn check(
    gpu: &Gpu,
    heads: [usize; 3],
    q_sizes: &[u32],
    kv_sizes: &[u32],
    mask: Mask,
    [before, after]: [usize; 2],
) {
    let [qh, kvh, hd] = heads;
    let qt: Vec<i32> = indptr_of(q_sizes)
        .into_iter()
        .map(|x| x + before as i32)
        .collect();
    let kt = indptr_of(kv_sizes);
    let qr = qt[qt.len() - 1] as usize + after;
    let kr = kt[kt.len() - 1] as usize;
    let seed = (hd + qr) as u64;
    let q = bf16s(qr * qh * hd, seed, 1.0);
    let k = bf16s(kr.max(1) * kvh * hd, seed + 1, 1.0);
    let v = bf16s(kr.max(1) * kvh * hd, seed + 2, 1.0);
    let fill = bf16s(qr * qh * hd, seed + 3, 1.0);
    let span = 2 * MAX_LEN - 1;
    let bias: Vec<f32> = bf16s(qh * span, seed + 4, 2.0);
    // Tags and classes are global per row; the q and kv tables are the same
    // for these arms, so one table serves both sides.
    let ids = qr.max(kr);
    let tags: Vec<i32> = (0..ids as i32)
        .map(|r| [-1, r / 3][usize::from(r % 9 >= 4)])
        .collect();
    let classes: Vec<i32> = (0..ids as i32).map(|r| r % 3).collect();
    let allowed: Vec<u8> = (0..9).map(|c| u8::from(c % 4 == 0 || c % 3 == 1)).collect();
    let scale = 2.0 / (hd as f32).sqrt();

    let mut want = fill.clone();
    reference(
        [&q, &k, &v],
        &qt,
        &kt,
        heads,
        scale,
        |g, h, r, j| {
            let (qo, kv) = (r - qt[g] as usize, j - kt[g] as usize);
            let keep = match mask {
                Mask::None | Mask::Bias => true,
                Mask::SelfOnly => {
                    let start = REF_START[g].max(0) as usize;
                    qo < start || kv >= start
                }
                Mask::Tags => tags[r] < 0 || tags[j] == tags[r],
                Mask::Class => allowed[(classes[r] * 3 + classes[j]) as usize] == 1,
            };
            let d = (kv + MAX_LEN - 1).saturating_sub(qo).min(span - 1);
            let b = if mask == Mask::Bias {
                bias[h * span + d]
            } else {
                0.0
            };
            keep.then_some(b)
        },
        &mut want,
    );

    let o = gpu.tensor(&fill, qr as u32, (qh * hd) as u32, Bf16);
    let qp = gpu.tensor(&q, qr as u32, (qh * hd) as u32, Bf16);
    let kp = gpu.tensor(&k, kr.max(1) as u32, (kvh * hd) as u32, Bf16);
    let vp = gpu.tensor(&v, kr.max(1) as u32, (kvh * hd) as u32, Bf16);
    let (qtab, ktab) = (gpu.column(&qt), gpu.column(&kt));
    let tag_t = gpu.column(&tags);
    let class_t = gpu.column(&classes);
    let arm = match mask {
        Mask::None => RaggedMask::None,
        Mask::Bias => RaggedMask::RelativeBias {
            table: gpu.tensor(&bias, qh as u32, span as u32, F32),
            max_len: MAX_LEN as u32,
        },
        Mask::SelfOnly => RaggedMask::ReferenceSelfOnly {
            ref_start: gpu.column(&REF_START),
        },
        Mask::Tags => RaggedMask::ReferenceTags {
            q_tags: tag_t,
            kv_tags: tag_t,
        },
        Mask::Class => RaggedMask::ClassTable {
            q_classes: class_t,
            kv_classes: class_t,
            table: gpu.upload(&allowed, 9, 1, U8),
            count: 3,
        },
    };
    let mut out = o;
    ragged(
        &gpu.ctx(),
        qp,
        kp,
        vp,
        qtab,
        ktab,
        hd as u32,
        scale,
        arm,
        &mut out,
    )
    .expect("the arm fires");
    close_abs(
        &format!("{heads:?} {mask:?}"),
        &gpu.read(&out),
        &want,
        1.5e-2,
    );
}

#[test]
fn the_ragged_arm_answers_the_reference() {
    let Some(gpu) = Gpu::open() else { return };
    let (q_cross, kv_cross) = (&[7, 0, 200, 300, 3], &[300, 40, 0, 500, 1]);
    let lanes = &[7u32, 11, 21];
    let (refs, bias) = (&[64u32, 300, 200, 50, 40, 5], &[37, 130, 64]);
    // Cross tables with empty groups at 64 and 128, the bias at full heads,
    // each lane mask, and rows no group holds at 256 and a padded 80.
    check(&gpu, [4, 2, 64], q_cross, kv_cross, Mask::None, [0, 0]);
    check(&gpu, [4, 2, 128], q_cross, kv_cross, Mask::None, [0, 0]);
    check(&gpu, [64, 16, 64], bias, bias, Mask::Bias, [0, 0]);
    check(&gpu, [4, 2, 128], refs, refs, Mask::SelfOnly, [0, 0]);
    check(&gpu, [2, 1, 256], lanes, lanes, Mask::Tags, [0, 0]);
    check(&gpu, [4, 2, 64], lanes, lanes, Mask::Class, [0, 0]);
    check(
        &gpu,
        [2, 2, 256],
        &[40, 100, 300, 64],
        &[40, 100, 300, 64],
        Mask::None,
        [0, 96],
    );
    check(&gpu, [3, 1, 80], &[50, 70], &[90, 20], Mask::None, [37, 5]);
}
