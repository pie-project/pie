//! The GGUF K-quant matmuls (Q2_K to Q6_K) on the first Vulkan device, beside
//! the hand-written kernels, against an f64 host reference that decodes each
//! super-block: 33 super-blocks a row, so a lane walks two and a row of an
//! odd scheme starts off a word; ten columns, so the last workgroup is part
//! empty.

use kernels_slang::linear::kquant::{Scheme, Tactic};
use kernels_vulkan::{Tensor, linear::kquant};
use poem_ir::Dtype;

use crate::gpu::{Gpu, both, noise};

fn st(t: Tensor) -> kernels_slang::Tensor {
    kernels_slang::Tensor::new(t.buf, t.rows, t.width, t.dtype)
}

const SUPER: usize = 256;

fn f16(h: u16) -> f64 {
    let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let e = i32::from((h >> 10) & 31);
    let m = f64::from(h & 1023);
    if e == 0 {
        sign * m * 2f64.powi(-24)
    } else {
        sign * (1.0 + m / 1024.0) * 2f64.powi(e - 15)
    }
}

fn half_at(b: &[u8], at: usize) -> f64 {
    f16(u16::from_le_bytes([b[at], b[at + 1]]))
}

/// Q4_K and Q5_K: sub-block `j`'s scale and minimum.
fn scale_min(s: &[u8], j: usize) -> (f64, f64) {
    if j < 4 {
        (f64::from(s[j] & 63), f64::from(s[j + 4] & 63))
    } else {
        (
            f64::from((s[j + 4] & 0x0f) | ((s[j - 4] >> 6) << 4)),
            f64::from((s[j + 4] >> 4) | ((s[j] >> 6) << 4)),
        )
    }
}

/// One super-block's 256 weights.
fn decode(scheme: Scheme, b: &[u8]) -> Vec<f64> {
    let mut v = vec![0.0; SUPER];
    match scheme {
        Scheme::Q2K => {
            let (d, dmin) = (half_at(b, 80), half_at(b, 82));
            for sub in 0..16 {
                let (window, step, half) = (sub / 8, (sub / 2) % 4, sub % 2);
                let dl = d * f64::from(b[sub] & 0x0f);
                let ml = dmin * f64::from(b[sub] >> 4);
                for l in 0..16 {
                    let q = (b[16 + window * 32 + half * 16 + l] >> (2 * step)) & 3;
                    v[sub * 16 + l] = dl * f64::from(q) - ml;
                }
            }
        }
        Scheme::Q3K => {
            let d = half_at(b, 108);
            let raw = &b[96..108];
            for sub in 0..16 {
                let (window, step, half) = (sub / 8, (sub / 2) % 4, sub % 2);
                let (grp, j) = (sub / 4, sub % 4);
                let src = raw[if grp % 2 == 1 { 4 } else { 0 } + j];
                let low = if grp < 2 { src & 0x0f } else { src >> 4 };
                let top = (raw[8 + j] >> (2 * grp)) & 3;
                let dl = d * (f64::from(low | (top << 4)) - 32.0);
                for l in 0..16 {
                    let at = half * 16 + l;
                    let q = i32::from((b[32 + window * 32 + at] >> (2 * step)) & 3);
                    let high = b[at] & (1 << (window * 4 + step)) != 0;
                    v[sub * 16 + l] = dl * f64::from(q - if high { 0 } else { 4 });
                }
            }
        }
        Scheme::Q4K | Scheme::Q5K => {
            let (d, dmin) = (half_at(b, 0), half_at(b, 2));
            let five = scheme == Scheme::Q5K;
            let codes = if five { 48 } else { 16 };
            for sub in 0..8 {
                let (sc, mn) = scale_min(&b[4..16], sub);
                for i in 0..32 {
                    let byte = b[codes + (sub / 2) * 32 + i];
                    let mut q = if sub % 2 == 1 { byte >> 4 } else { byte & 0x0f };
                    if five {
                        q |= ((b[16 + i] >> sub) & 1) << 4;
                    }
                    v[sub * 32 + i] = d * sc * f64::from(q) - dmin * mn;
                }
            }
        }
        Scheme::Q6K => {
            let d = half_at(b, 208);
            for half in 0..2 {
                for quarter in 0..4 {
                    for i in 0..32 {
                        let byte = b[half * 64 + i + 32 * (quarter % 2)];
                        let low = if quarter < 2 { byte & 0x0f } else { byte >> 4 };
                        let top = (b[128 + half * 32 + i] >> (2 * quarter)) & 3;
                        let q = i32::from(low | (top << 4)) - 32;
                        let scale = f64::from(b[192 + half * 8 + i / 16 + 2 * quarter] as i8);
                        v[half * 128 + quarter * 32 + i] = d * scale * f64::from(q);
                    }
                }
            }
        }
    }
    v
}

/// Random bytes with each scale an f16 between 2^-12 and 2^-9.
fn blocks(scheme: Scheme, seed: u64, count: usize) -> Vec<u8> {
    let width = scheme.block_bytes() as usize;
    let mut s = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    let mut next = move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    };
    let mut bytes: Vec<u8> = (0..count * width).map(|_| next() as u8).collect();
    let scales: &[usize] = match scheme {
        Scheme::Q2K => &[80, 82],
        Scheme::Q3K => &[108],
        Scheme::Q4K | Scheme::Q5K => &[0, 2],
        Scheme::Q6K => &[208],
    };
    for block in bytes.chunks_mut(width) {
        for at in scales {
            let r = next();
            let h = ((3 + (r % 4) as u16) << 10) | ((r >> 8) as u16 & 1023);
            block[*at..*at + 2].copy_from_slice(&h.to_le_bytes());
        }
    }
    bytes
}

#[test]
fn the_kquants_answer_the_reference() {
    let Some(mut gpu) = Gpu::open() else {
        eprintln!("not asked: no device");
        return;
    };
    let wide = gpu.device.chip().subgroup >= 32;
    let (m, n, k) = (3usize, 10usize, 33 * SUPER);
    let (x, xr) = gpu.up(&noise(1, m * k));
    let t = |b: u32, r: usize, w: usize, d: Dtype| Tensor::new(b, r as u32, w as u32, d);
    let xt = t(x, m, k, Dtype::Bf16);
    let schemes = [
        Scheme::Q2K,
        Scheme::Q3K,
        Scheme::Q4K,
        Scheme::Q5K,
        Scheme::Q6K,
    ];
    for (at, scheme) in schemes.into_iter().enumerate() {
        let row_bytes = (k / SUPER) * scheme.block_bytes() as usize;
        let bytes = blocks(scheme, 7 + at as u64, n * k / SUPER);
        let weights: Vec<f64> = bytes
            .chunks(scheme.block_bytes() as usize)
            .flat_map(|b| decode(scheme, b))
            .collect();
        let mut words = bytes.clone();
        words.resize(bytes.len().next_multiple_of(4), 0);
        let w = gpu.up_u32(
            &words
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| u32::from_le_bytes(*c))
                .collect::<Vec<_>>(),
        );
        let wt = t(w, n, row_bytes, Dtype::U8);
        let want: Vec<f64> = (0..m * n)
            .map(|i| {
                let (row, col) = (i / n, i % n);
                (0..k)
                    .map(|j| f64::from(xr[row * k + j]) * weights[col * k + j])
                    .sum()
            })
            .collect();
        for (subgroup, slices) in [(false, 2), (false, 4), (true, 2), (true, 4)] {
            if subgroup && !wide {
                continue;
            }
            let tier = Tactic {
                tier: if subgroup {
                    kernels_slang::Tier::Subgroup
                } else {
                    kernels_slang::Tier::Baseline
                },
                slices,
            };
            both(
                &mut gpu,
                &format!("{scheme:?} subgroup {subgroup} slices {slices}"),
                &|g| vec![(g.zeros(m * n), m * n, false)],
                std::slice::from_ref(&want),
                &|g, o| {
                    let y = t(o[0].0, m, n, Dtype::Bf16);
                    g.run(|cx| kquant::matmul(cx, xt, wt, y));
                },
                &|g, o| {
                    let y = st(t(o[0].0, m, n, Dtype::Bf16));
                    g.run_slang(|cx| {
                        kernels_slang::linear::kquant::lm_head(cx, tier, st(xt), st(wt), y)
                    });
                },
            );
        }
    }
}
