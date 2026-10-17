//! Host references several kernel tests share.
#![allow(dead_code)]

use super::from_bf16;
use super::tc5::{e2m1, e4m3, ue8m0};
use kernels_cuda::attn::dsv4::{self, Rows};

/// The nearest of `codes` positive codes to `|v|` under `value`, ties to the
/// even code (`cvt.rn`), saturating at the largest; the sign bit `sign` set
/// on a negative `v`, zero included, as the hardware keeps it.
pub fn nearest(v: f32, codes: u8, value: impl Fn(u8) -> f32, sign: u8) -> u8 {
    let a = v.abs();
    let mut best = 0u8;
    for c in 1..codes {
        let (d, db) = ((value(c) - a).abs(), (value(best) - a).abs());
        if d < db || (d == db && c % 2 == 0) {
            best = c;
        }
    }
    if v.is_sign_negative() {
        best | sign
    } else {
        best
    }
}

/// `v`'s nearest finite e4m3 code.
pub fn to_e4m3(v: f32) -> u8 {
    nearest(v, 0x7f, e4m3, 0x80)
}

/// `v`'s nearest e2m1 code.
pub fn to_e2m1(v: f32) -> u8 {
    nearest(v, 8, e2m1, 8)
}

/// The running offsets of `sizes`, from 0.
pub fn indptr(sizes: &[u32]) -> Vec<i32> {
    let mut out = vec![0i32];
    for &n in sizes {
        out.push(out.last().unwrap() + n as i32);
    }
    out
}

/// Bidirectional attention within each group of rows the indptr tables
/// name, `[q_heads, kv_heads, hd]`; `logit(group, head, q row, kv row)`
/// adds its bias to a score, or masks the key with `None`.
pub fn ragged_attention(
    [q, k, v]: [&[f32]; 3],
    q_indptr: &[i32],
    kv_indptr: &[i32],
    [q_heads, kv_heads, hd]: [usize; 3],
    sm_scale: f32,
    logit: impl Fn(usize, usize, usize, usize) -> Option<f32>,
) -> Vec<f32> {
    let rows = q.len() / (q_heads * hd);
    let mut o = vec![0f32; rows * q_heads * hd];
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
                let out = &mut o[(r * q_heads + h) * hd..][..hd];
                for (&j, p) in keys.iter().zip(&scores) {
                    let vv = &v[(j * kv_heads + kh) * hd..][..hd];
                    for (d, x) in out.iter_mut().zip(vv) {
                        *d += p / sum * x;
                    }
                }
            }
        }
    }
    o
}

/// Every bf16 `got` within `tol` of `want` (relative, at least 1 absolute).
pub fn assert_near(got: &[u16], want: &[f32], tol: f32, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: lengths");
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        let g = from_bf16(g);
        assert!(
            (g - w).abs() <= tol * w.abs().max(1.0),
            "{what}: element {i} landed {g} against {w}"
        );
    }
}

/// What DeepSeek's quantized rows of `fmt` say, `page` cells a page (a
/// page's data rows before its scale rows): every cell's 512 values.
pub fn rows_read(raw: &[u8], cells: usize, page: usize, fmt: Rows) -> Vec<f32> {
    const D: usize = 512;
    let (data, scale) = match fmt {
        Rows::Ds => (dsv4::DS_DATA as usize, 8),
        Rows::Mx8 => (512, 16),
        Rows::Nv4 => (256, 32),
    };
    let nope = dsv4::DS_NOPE as usize;
    let mut out = vec![0f32; cells * D];
    for cell in 0..cells {
        let base = cell / page * page * (data + scale);
        let off = cell % page;
        let bytes = &raw[base + off * data..base + (off + 1) * data];
        let s_at = base + page * data + off * scale;
        let scales = &raw[s_at..s_at + scale];
        if fmt == Rows::Ds {
            assert_eq!(scales[7], 0, "cell {cell}: the scale row's pad");
        }
        for d in 0..D {
            out[cell * D + d] = match fmt {
                Rows::Ds if d < nope => e4m3(bytes[d]) * ue8m0(scales[d / 64]),
                Rows::Ds => {
                    let at = nope + (d - nope) * 2;
                    from_bf16(u16::from_le_bytes([bytes[at], bytes[at + 1]]))
                }
                Rows::Mx8 => e4m3(bytes[d]) * ue8m0(scales[d / 32]),
                Rows::Nv4 => {
                    let byte = bytes[d / 2];
                    let code = if d % 2 == 0 { byte & 0xf } else { byte >> 4 };
                    e2m1(code) * e4m3(scales[d / 16])
                }
            };
        }
    }
    out
}
