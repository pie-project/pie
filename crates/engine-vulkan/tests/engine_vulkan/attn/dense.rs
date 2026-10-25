//! Dense attention over segments on the first Vulkan device against an f64
//! host reference: patch rows attending their own image (bidirectional), and
//! ragged query segments attending the key span the CSR names, on the tiled
//! kernel (bf16 at 64 and 128) and the scalar one, with and without a custom mask.

use kernels_slang::Tensor;
use poem_ir::Dtype;

use crate::gpu::{Gpu, close, noise};

fn up(gpu: &mut Gpu, v: &[f32], dtype: Dtype) -> (u32, Vec<f64>) {
    if dtype == Dtype::F32 {
        (gpu.up_f32(v), v.iter().map(|x| f64::from(*x)).collect())
    } else {
        let (b, h) = gpu.up(v);
        (b, h.iter().map(|x| f64::from(*x)).collect())
    }
}

fn down(gpu: &Gpu, buf: u32, n: usize, dtype: Dtype) -> Vec<f32> {
    if dtype == Dtype::F32 {
        gpu.down_f32(buf, n)
    } else {
        gpu.down(buf, n)
    }
}

/// Query row `r` attends the key rows `span(r)` that `keeps(r, j)`.
#[allow(clippy::too_many_arguments)]
fn attend(
    q: &[f64],
    k: &[f64],
    v: &[f64],
    rows: usize,
    qh: usize,
    kvh: usize,
    hd: usize,
    scale: f64,
    span: impl Fn(usize) -> (usize, usize),
    keeps: impl Fn(usize, usize) -> bool,
) -> Vec<f64> {
    let mut out = vec![0.0; rows * qh * hd];
    for r in 0..rows {
        let (b, e) = span(r);
        let kept: Vec<usize> = (b..e).filter(|&j| keeps(r, j)).collect();
        for h in 0..qh {
            let kh = h / (qh / kvh);
            let qr = &q[(r * qh + h) * hd..][..hd];
            let scores: Vec<f64> = kept
                .iter()
                .map(|&j| {
                    qr.iter()
                        .zip(&k[(j * kvh + kh) * hd..][..hd])
                        .map(|(a, c)| a * c)
                        .sum::<f64>()
                        * scale
                })
                .collect();
            let top = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let w: Vec<f64> = scores.iter().map(|s| (s - top).exp()).collect();
            let sum: f64 = w.iter().sum();
            for d in 0..hd {
                out[(r * qh + h) * hd + d] = kept
                    .iter()
                    .zip(&w)
                    .map(|(&j, w)| w * v[(j * kvh + kh) * hd + d])
                    .sum::<f64>()
                    / sum.max(f64::MIN_POSITIVE);
            }
        }
    }
    out
}

#[test]
fn dense_attention_answers_the_reference() {
    let Some(mut gpu) = Gpu::open() else {
        eprintln!("not asked: no device");
        return;
    };
    let (qh, kvh) = (4usize, 2usize);
    // Bidirectional: two images of 5 and 7 patches.
    let segs = [0i32, 5, 12];
    let rows = 12usize;
    let sb = gpu.up_i32(&segs);
    let st_ = Tensor::new(sb, 3, 1, Dtype::I32);
    for hd in [64usize, 96, 128] {
        let scale = 1.0 / (hd as f64).sqrt();
        let (q, qr) = up(&mut gpu, &noise(hd as u64, rows * qh * hd), Dtype::Bf16);
        let (k, kr) = up(
            &mut gpu,
            &noise(hd as u64 + 1, rows * kvh * hd),
            Dtype::Bf16,
        );
        let (v, vr) = up(
            &mut gpu,
            &noise(hd as u64 + 2, rows * kvh * hd),
            Dtype::Bf16,
        );
        let want = attend(
            &qr,
            &kr,
            &vr,
            rows,
            qh,
            kvh,
            hd,
            scale,
            |r| if r < 5 { (0, 5) } else { (5, 12) },
            |_, _| true,
        );
        let n = rows * qh * hd;
        let t = |b: u32, h: usize| Tensor::new(b, rows as u32, (h * hd) as u32, Dtype::Bf16);
        let y = gpu.zeros(n);
        gpu.run(|c| {
            kernels_slang::attn::dense::bidirectional(
                c,
                t(q, qh),
                t(k, kvh),
                t(v, kvh),
                st_,
                hd as u32,
                scale as f32,
                t(y, qh),
            )
        });
        close(
            &gpu.down(y, n),
            &want,
            &format!("dense bidirectional d {hd}"),
        );
    }

    // Ragged: 40 query rows in segments of 3, 0 and 37, over key spans of 9,
    // 4 and 50 (60 keys in all). 37 rows straddle a 32-row tile.
    let q_indptr = [0i32, 3, 3, 40];
    let kv_indptr = [0i32, 9, 13, 63];
    let (rows, keys) = (40usize, 63usize);
    let (qp_b, kp_b) = (gpu.up_i32(&q_indptr), gpu.up_i32(&kv_indptr));
    let qp = Tensor::new(qp_b, 4, 1, Dtype::I32);
    let kp = Tensor::new(kp_b, 4, 1, Dtype::I32);
    let segment = |r: usize| {
        (0..3)
            .find(|&s| r < q_indptr[s + 1] as usize && r >= q_indptr[s] as usize)
            .unwrap()
    };
    let span = |r: usize| {
        let s = segment(r);
        (kv_indptr[s] as usize, kv_indptr[s + 1] as usize)
    };
    // Segment `s` keeps `(qo, kv)` unless `(3 qo + kv) % 4 == 1`; each row keeps its first key.
    let kept = |r: usize, j: usize| {
        let s = segment(r);
        let (qo, kv) = (r - q_indptr[s] as usize, j - kv_indptr[s] as usize);
        kv == 0 || (3 * qo + kv) % 4 != 1
    };
    let mut bits: Vec<u8> = Vec::new();
    let mut mask_indptr: Vec<i32> = Vec::new();
    for s in 0..3 {
        mask_indptr.push(bits.len() as i32);
        let (q0, q1) = (q_indptr[s] as usize, q_indptr[s + 1] as usize);
        let (k0, k1) = (kv_indptr[s] as usize, kv_indptr[s + 1] as usize);
        let base = bits.len();
        bits.resize(base + ((q1 - q0) * (k1 - k0)).div_ceil(8), 0);
        for r in q0..q1 {
            for j in k0..k1 {
                if kept(r, j) {
                    let cell = (r - q0) * (k1 - k0) + (j - k0);
                    bits[base + cell / 8] |= 1 << (cell % 8);
                }
            }
        }
    }
    mask_indptr.push(bits.len() as i32);
    bits.resize(bits.len().next_multiple_of(4), 0);
    let words: Vec<i32> = bits
        .chunks(4)
        .map(|w| i32::from_le_bytes([w[0], w[1], w[2], w[3]]))
        .collect();
    let mask = kernels_slang::attn::dense::CustomMask {
        bits: Tensor::new(gpu.up_i32(&words), bits.len() as u32, 1, Dtype::U8),
        indptr: Tensor::new(gpu.up_i32(&mask_indptr), 4, 1, Dtype::I32),
    };
    for (dtype, hd) in [
        (Dtype::Bf16, 64usize),
        (Dtype::Bf16, 128),
        (Dtype::Bf16, 96),
        (Dtype::F32, 64),
    ] {
        let scale = 1.0 / (hd as f64).sqrt();
        let (q, qr) = up(&mut gpu, &noise(hd as u64 + 10, rows * qh * hd), dtype);
        let (k, kr) = up(&mut gpu, &noise(hd as u64 + 11, keys * kvh * hd), dtype);
        let (v, vr) = up(&mut gpu, &noise(hd as u64 + 12, keys * kvh * hd), dtype);
        let want = attend(&qr, &kr, &vr, rows, qh, kvh, hd, scale, span, |_, _| true);
        let masked = attend(&qr, &kr, &vr, rows, qh, kvh, hd, scale, span, kept);
        let n = rows * qh * hd;
        let qt = Tensor::new(q, rows as u32, (qh * hd) as u32, dtype);
        let kt = Tensor::new(k, keys as u32, (kvh * hd) as u32, dtype);
        let vt = Tensor::new(v, keys as u32, (kvh * hd) as u32, dtype);
        let ot = |b: u32| Tensor::new(b, rows as u32, (qh * hd) as u32, dtype);
        let shape = format!("{dtype:?} d {hd}");

        let y = gpu.zeros(2 * n);
        gpu.run(|c| {
            kernels_slang::attn::dense::ragged(
                c,
                qt,
                kt,
                vt,
                qp,
                kp,
                hd as u32,
                kvh as u32,
                scale as f32,
                None,
                ot(y),
            )
        });
        close(&down(&gpu, y, n, dtype), &want, &format!("ragged {shape}"));

        let y = gpu.zeros(2 * n);
        gpu.run(|c| {
            kernels_slang::attn::dense::ragged(
                c,
                qt,
                kt,
                vt,
                qp,
                kp,
                hd as u32,
                kvh as u32,
                scale as f32,
                Some(mask),
                ot(y),
            )
        });
        close(
            &down(&gpu, y, n, dtype),
            &masked,
            &format!("ragged custom-masked {shape}"),
        );
    }
}
