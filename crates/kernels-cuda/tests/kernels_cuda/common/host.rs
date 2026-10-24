//! Host references and fixtures several kernel tests share.
#![allow(dead_code)]
#![allow(unsafe_code)]

use super::from_bf16;
use super::tc5::{e2m1, e4m3, ue8m0};
use kernels_cuda::attn::compressed::decode::{self, Rows};
use kernels_cuda::cudarc::runtime::sys as rt;

/// Asserts a runtime call answered success.
pub fn cuda_ok(code: rt::cudaError, call: &str) {
    assert_eq!(
        code,
        rt::cudaError::cudaSuccess,
        "`{call}` answered {code:?}"
    );
}

/// Copies `values` over the device bytes at `at`.
pub fn upload_to<T: Copy>(at: u64, values: &[T]) {
    let _legacy = super::legacy();
    // SAFETY: the source is `values`, borrowed for the call, and the copy
    // is synchronous. `at` is a device allocation of at least
    // `size_of_val(values)` bytes: every test passes one it allocated so.
    let code = unsafe {
        rt::cudaMemcpy(
            at as *mut core::ffi::c_void,
            values.as_ptr().cast(),
            core::mem::size_of_val(values),
            rt::cudaMemcpyKind::cudaMemcpyHostToDevice,
        )
    };
    cuda_ok(code, "cudaMemcpy H2D");
}

/// `tokens` rows of `top` experts each, spread over `experts`.
pub fn routes_of(tokens: usize, experts: usize, top: usize) -> Vec<i32> {
    (0..tokens)
        .flat_map(|t| (0..top).map(move |j| ((t * 5 + j * 7 + t / 3) % experts) as i32))
        .collect()
}

/// Spatial attention over `c`-wide rows, each box's voxels attending within
/// runs of `seg` frames (`0`: the whole box), in f64.
pub fn spatial_attention_ref(
    [q, k, v]: [&[f32]; 3],
    boxes: &[super::spatial::Box3],
    c: usize,
    seg: usize,
    scale: f32,
) -> Vec<f32> {
    let rows = q.len() / c;
    let mut y = vec![0f32; rows * c];
    let mut off = 0usize;
    for b in boxes {
        let plane = b.plane();
        let per = if seg == 0 { b.t } else { seg };
        let mut frame = 0usize;
        while frame < b.t {
            let last = (frame + per).min(b.t);
            let (begin, end) = (off + frame * plane, off + last * plane);
            for i in begin..end {
                let s: Vec<f64> = (begin..end)
                    .map(|j| {
                        let dot: f64 = (0..c)
                            .map(|e| f64::from(q[i * c + e]) * f64::from(k[j * c + e]))
                            .sum();
                        dot * f64::from(scale)
                    })
                    .collect();
                let m = s.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let p: Vec<f64> = s.iter().map(|x| (x - m).exp()).collect();
                let l: f64 = p.iter().sum();
                for e in 0..c {
                    let acc: f64 = (0..end - begin)
                        .map(|jj| p[jj] * f64::from(v[(begin + jj) * c + e]))
                        .sum();
                    y[i * c + e] = (acc / l) as f32;
                }
            }
            frame = last;
        }
        off += b.voxels();
    }
    y
}

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

/// What DeepSeek's quantized rows of `fmt` say, `page` cells a page (a
/// page's data rows before its scale rows): every cell's 512 values.
pub fn rows_read(raw: &[u8], cells: usize, page: usize, fmt: Rows) -> Vec<f32> {
    const D: usize = 512;
    let (data, scale) = match fmt {
        Rows::Ds => (decode::DS_DATA as usize, 8),
        Rows::Mx8 => (512, 16),
        Rows::Nv4 => (256, 32),
    };
    let nope = decode::DS_NOPE as usize;
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
