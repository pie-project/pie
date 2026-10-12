//! The quantized projections (MLX affine / mxfp4 banks, GGUF K-quants,
//! NVFP4) against host decoders written from the GPU shaders.
//!
//! The device rounds each decoded weight to bf16 before its one bf16 dot (as
//! a GPU qmm stages its decoded tile), so the references round the decoded
//! weight the same way and then accumulate in f64.

use dtype::Dtype;
use engine_xla::bench::{Bench, assert_close, round_bf16};
use kernels_xla::hlo::f16_bits;
use kernels_xla::linear::{kquant, nvfp4, quant};
use kernels_xla::{Bank, Tensor};

fn hash(i: usize, seed: u32) -> u32 {
    (i as u32)
        .wrapping_mul(2_654_435_761)
        .wrapping_add(seed.wrapping_mul(40503))
        .rotate_left(13)
        .wrapping_mul(2_246_822_519)
        .rotate_left(5)
}

fn data(n: usize, seed: u32) -> Vec<f32> {
    (0..n)
        .map(|i| round_bf16(((hash(i, seed) >> 8) % 2000) as f32 / 1000.0 - 1.0))
        .collect()
}

fn bytes(n: usize, seed: u32) -> Vec<u8> {
    (0..n).map(|i| (hash(i, seed) >> 11) as u8).collect()
}

fn f16_to_f32(h: u16) -> f32 {
    let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exp = i32::from((h >> 10) & 0x1f);
    let man = f32::from(h & 0x3ff);
    sign * match exp {
        0 => man * 2f32.powi(-24),
        31 => f32::NAN,
        e => (1.0 + man / 1024.0) * 2f32.powi(e - 15),
    }
}

/// `x · wᵀ` over bf16-rounded weights, accumulated in f64.
fn gemm_ref(x: &[f32], w: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    gemm_at(x, w, m, n, k, true)
}

/// `x · wᵀ`, the weights rounded to bf16 first when `round`.
fn gemm_at(x: &[f32], w: &[f32], m: usize, n: usize, k: usize, round: bool) -> Vec<f32> {
    let mut y = vec![0.0; m * n];
    for r in 0..m {
        for c in 0..n {
            let mut acc = 0.0f64;
            for i in 0..k {
                let wv = if round {
                    round_bf16(w[c * k + i])
                } else {
                    w[c * k + i]
                };
                acc += f64::from(x[r * k + i]) * f64::from(wv);
            }
            y[r * n + c] = acc as f32;
        }
    }
    y
}

/// A tensor over the same bench array, typed as the engine would bind it.
fn retype(t: Tensor, rows: u32, width: u32, dtype: Dtype) -> Tensor {
    Tensor::new(t.buf, rows, width, dtype)
}

// ------------------------------------------------------------------ affine

/// Codes `[n, k]` packed LSB-first, `bits` each, into bytes `[n, k·bits/8]`.
fn pack(codes: &[u32], bits: u32) -> Vec<u8> {
    let per = (8 / bits) as usize;
    codes
        .chunks(per)
        .map(|c| {
            c.iter()
                .enumerate()
                .fold(0u8, |acc, (i, &v)| acc | ((v as u8) << (i as u32 * bits)))
        })
        .collect()
}

#[test]
fn affine_banks_at_every_group_and_width_answer_the_host() {
    // `m` rows run group-batched; `big` rows (`big · groups > 4096` at every
    // group size here) run through the decoded dense weight.
    let (m, big, n, k) = (5usize, 1400usize, 12usize, 384usize);
    let xs = data(m * k, 1);
    let xb = data(big * k, 2);
    let x32: Vec<f32> = (0..m * k)
        .map(|i| data(1, 3 + i as u32)[0] * 1.001 + 1e-4)
        .collect();
    let mut b = Bench::new();
    let x = b.bf16(m as u32, k as u32, &xs);
    let xbig = b.bf16(big as u32, k as u32, &xb);
    let xf = b.f32(m as u32, k as u32, &x32);
    let xfb = b.f32(big as u32, k as u32, &xb);
    struct Case {
        bank: Bank,
        y: Tensor,
        yb: Tensor,
        w: Vec<f32>,
        label: String,
    }
    let mut cases = Vec::new();
    let dtypes = [
        (32, 2, Dtype::U2g32),
        (64, 2, Dtype::U2g64),
        (128, 2, Dtype::U2g128),
        (32, 4, Dtype::U4g32),
        (64, 4, Dtype::U4g64),
        (128, 4, Dtype::U4g64),
        (32, 8, Dtype::U8g64),
        (64, 8, Dtype::U8g64),
        (128, 8, Dtype::U8g64),
    ];
    for (at, &(group, bits, packed_dtype)) in dtypes.iter().enumerate() {
        let seed = 100 + at as u32 * 7;
        let groups = k / group as usize;
        let codes: Vec<u32> = (0..n * k)
            .map(|i| hash(i, seed) >> 7 & ((1 << bits) - 1))
            .collect();
        let scales: Vec<f32> = data(n * groups, seed + 1)
            .iter()
            .map(|v| round_bf16((v.abs() + 0.1) * 0.5 / (1 << bits) as f32))
            .collect();
        let biases: Vec<f32> = data(n * groups, seed + 2)
            .iter()
            .map(|v| round_bf16(v * 0.3))
            .collect();
        let w: Vec<f32> = (0..n * k)
            .map(|i| {
                let g = (i / k) * groups + (i % k) / group as usize;
                scales[g] * codes[i] as f32 + biases[g]
            })
            .collect();
        let packed = pack(&codes, bits);
        let row_bytes = (k as u32) * bits / 8;
        // Alternate the two code views the entry reads: the bank's packed
        // dtype over logical codes, and the bytes themselves.
        let t = b.u8(n as u32, row_bytes, &packed);
        let codes_t = if at % 2 == 0 {
            retype(t, n as u32, k as u32, packed_dtype)
        } else {
            t
        };
        let s = b.bf16(n as u32, groups as u32, &scales);
        let bi = b.bf16(n as u32, groups as u32, &biases);
        let y = b.zeros(Dtype::Bf16, m as u32, n as u32);
        let yb = b.zeros(Dtype::Bf16, big as u32, n as u32);
        cases.push(Case {
            bank: Bank {
                codes: codes_t,
                scales: s,
                biases: Some(bi),
                group,
                bits,
            },
            y,
            yb,
            w,
            label: format!("g{group} b{bits}"),
        });
    }

    // A mxfp4 bank: symmetric, 4 bits in groups of 32, e8m0 scales.
    let lut = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    let mx_codes: Vec<u32> = (0..n * k).map(|i| hash(i, 900) >> 9 & 15).collect();
    let mx_scales: Vec<u8> = (0..n * k / 32)
        .map(|i| 122 + (hash(i, 901) % 8) as u8)
        .collect();
    let mx_w: Vec<f32> = (0..n * k)
        .map(|i| {
            let c = mx_codes[i];
            let v = lut[(c & 7) as usize] * if c & 8 != 0 { -1.0 } else { 1.0 };
            v * 2f32.powi(i32::from(mx_scales[(i / k) * (k / 32) + (i % k) / 32]) - 127)
        })
        .collect();
    let mc = b.u8(n as u32, (k / 2) as u32, &pack(&mx_codes, 4));
    let ms = b.u8(n as u32, (k / 32) as u32, &mx_scales);
    let my = b.zeros(Dtype::Bf16, m as u32, n as u32);
    let myb = b.zeros(Dtype::Bf16, big as u32, n as u32);
    cases.push(Case {
        bank: Bank {
            codes: retype(mc, n as u32, k as u32, Dtype::Mxfp4),
            scales: retype(ms, n as u32, (k / 32) as u32, Dtype::E8m0),
            biases: None,
            group: 32,
            bits: 4,
        },
        y: my,
        yb: myb,
        w: mx_w,
        label: "mxfp4".into(),
    });

    let lm = b.zeros(Dtype::Bf16, m as u32, n as u32);
    let (u4, mx) = (cases[4].bank, cases[9].bank);
    let f32s = [
        b.zeros(Dtype::F32, m as u32, n as u32),
        b.zeros(Dtype::F32, big as u32, n as u32),
        b.zeros(Dtype::F32, m as u32, n as u32),
        b.zeros(Dtype::F32, big as u32, n as u32),
    ];
    if !b
        .run(|ctx| {
            for c in &cases {
                quant::matmul(ctx, x, c.bank, c.y)?;
                quant::matmul(ctx, xbig, c.bank, c.yb)?;
            }
            quant::matmul(ctx, xf, u4, f32s[0])?;
            quant::matmul(ctx, xfb, u4, f32s[1])?;
            quant::matmul(ctx, xf, mx, f32s[2])?;
            quant::matmul(ctx, xfb, mx, f32s[3])?;
            quant::lm_head(ctx, x, u4, lm)
        })
        .unwrap()
    {
        return;
    }
    // Group-batched rows scale exact partials (the GPU qmv's split), so
    // their reference keeps the weight unrounded; the decoded dense weight
    // is rounded to bf16 (a GPU qmm's staged tile).
    for c in &cases {
        eprintln!("{}", c.label);
        assert_close(
            &b.read_f32(c.y),
            &gemm_at(&xs, &c.w, m, n, k, false),
            1e-2,
            1e-2,
        );
        assert_close(
            &b.read_f32(c.yb),
            &gemm_ref(&xb, &c.w, big, n, k),
            1e-2,
            1e-2,
        );
    }
    assert_close(
        &b.read_f32(lm),
        &gemm_at(&xs, &cases[4].w, m, n, k, false),
        1e-2,
        1e-2,
    );
    // f32 activations: exact products, f32 accumulation.
    eprintln!("f32 activations");
    let (u4w, mxw) = (&cases[4].w, &cases[9].w);
    assert_close(
        &b.read_f32(f32s[0]),
        &gemm_at(&x32, u4w, m, n, k, false),
        1e-4,
        1e-4,
    );
    assert_close(
        &b.read_f32(f32s[1]),
        &gemm_at(&xb, u4w, big, n, k, false),
        1e-4,
        1e-4,
    );
    assert_close(
        &b.read_f32(f32s[2]),
        &gemm_at(&x32, mxw, m, n, k, false),
        1e-4,
        1e-4,
    );
    assert_close(
        &b.read_f32(f32s[3]),
        &gemm_at(&xb, mxw, big, n, k, false),
        1e-4,
        1e-4,
    );
}

// ------------------------------------------------------------------ k-quant

fn put_f16(block: &mut [u8], at: usize, v: f32) {
    block[at..at + 2].copy_from_slice(&f16_bits(v).to_le_bytes());
}

fn get_f16(block: &[u8], at: usize) -> f32 {
    f16_to_f32(u16::from_le_bytes([block[at], block[at + 1]]))
}

/// One super-block's 256 weights, read as `quant/kquant.wgsl` reads them.
fn kquant_block(scheme: usize, blk: &[u8]) -> Vec<f32> {
    let mut w = vec![0.0f32; 256];
    let byte = |at: usize| u32::from(blk[at]);
    match scheme {
        2 => {
            let d = get_f16(blk, 80);
            let dmin = get_f16(blk, 82);
            for b in 0..16 {
                let shift = 2 * ((b >> 1) & 3);
                let at = 16 + (b >> 3) * 32 + (b & 1) * 16;
                let packed = byte(b);
                for l in 0..16 {
                    let q = (byte(at + l) >> shift) & 3;
                    w[b * 16 + l] =
                        d * (packed & 15) as f32 * q as f32 - dmin * (packed >> 4) as f32;
                }
            }
        }
        3 => {
            let d = get_f16(blk, 108);
            // `q3k_scale(base + 96, sub)`.
            let scale = |sub: usize| -> i32 {
                let grp = sub >> 2;
                let j = sub & 3;
                let src = byte(96 + if grp & 1 != 0 { 4 } else { 0 } + j);
                let low = if grp < 2 { src & 15 } else { src >> 4 };
                let top = (byte(96 + 8 + j) >> (2 * grp)) & 3;
                (low | (top << 4)) as i32
            };
            for b in 0..16 {
                let step = (b >> 1) & 3;
                let shift = 2 * step;
                let selector = 1u32 << ((b >> 3) * 4 + step);
                let at = 32 + (b >> 3) * 32 + (b & 1) * 16;
                let mask_at = (b & 1) * 16;
                for l in 0..16 {
                    let code = ((byte(at + l) >> shift) & 3) as i32;
                    let borrow = if byte(mask_at + l) & selector != 0 {
                        0
                    } else {
                        4
                    };
                    w[b * 16 + l] = d * (scale(b) - 32) as f32 * (code - borrow) as f32;
                }
            }
        }
        4 | 5 => {
            let d = get_f16(blk, 0);
            let dmin = get_f16(blk, 2);
            let scale_min = |sub: usize| -> (f32, f32) {
                let base = 4;
                if sub < 4 {
                    return (
                        (byte(base + sub) & 63) as f32,
                        (byte(base + sub + 4) & 63) as f32,
                    );
                }
                let a = byte(base + sub + 4);
                let b = byte(base + sub - 4);
                let c = byte(base + sub);
                (
                    ((a & 15) | ((b >> 6) << 4)) as f32,
                    ((a >> 4) | ((c >> 6) << 4)) as f32,
                )
            };
            for b in 0..8 {
                let pair = b >> 1;
                let high = b & 1 != 0;
                let (sc, m) = scale_min(b);
                for i in 0..32 {
                    let q = if scheme == 4 {
                        let byte_ = byte(16 + pair * 32 + i);
                        if high { byte_ >> 4 } else { byte_ & 15 }
                    } else {
                        let byte_ = byte(48 + pair * 32 + i);
                        let low = if high { byte_ >> 4 } else { byte_ & 15 };
                        let fifth = (byte(16 + i) >> b) & 1;
                        low | (fifth << 4)
                    };
                    w[b * 32 + i] = d * sc * q as f32 - dmin * m;
                }
            }
        }
        6 => {
            let d = get_f16(blk, 208);
            for half in 0..2 {
                for quarter in 0..4 {
                    for sub in 0..2 {
                        let raw = byte(192 + half * 8 + sub + 2 * quarter) as i32;
                        let sc = if raw > 127 { raw - 256 } else { raw };
                        for t in 0..16 {
                            let i = sub * 16 + t;
                            let byte_ = byte(half * 64 + i + 32 * (quarter & 1));
                            let low = if quarter < 2 { byte_ & 15 } else { byte_ >> 4 };
                            let top = (byte(128 + half * 32 + i) >> (2 * quarter)) & 3;
                            let q = (low | (top << 4)) as i32 - 32;
                            w[half * 128 + quarter * 32 + i] = d * sc as f32 * q as f32;
                        }
                    }
                }
            }
        }
        _ => unreachable!(),
    }
    w
}

#[test]
fn every_k_quant_scheme_answers_the_host() {
    // `m` rows through the in-graph decode and the repacked affine bank
    // group-batched; `big` rows through the repacked bank's dense decode.
    let (m, big, n, k) = (4usize, 64usize, 6usize, 768usize);
    let blocks = k / 256;
    let xs = data(m * k, 30);
    let xb = data(big * k, 31);
    let schemes = [
        (2usize, 84usize, Dtype::U2g16k),
        (3, 110, Dtype::I3g16k),
        (4, 144, Dtype::U4g32k),
        (5, 176, Dtype::U5g32k),
        (6, 210, Dtype::I6g16k),
    ];
    let mut b = Bench::new();
    let x = b.bf16(m as u32, k as u32, &xs);
    let xbig = b.bf16(big as u32, k as u32, &xb);
    struct Case {
        scheme: usize,
        w: Tensor,
        y: Tensor,
        codes: Tensor,
        scales: Tensor,
        biases: Tensor,
        group: u32,
        ya: Tensor,
        yb: Tensor,
        dense: Vec<f32>,
    }
    let mut cases = Vec::new();
    for (at, &(scheme, bb, dtype)) in schemes.iter().enumerate() {
        let mut plane = bytes(n * blocks * bb, 40 + at as u32);
        let mut w = Vec::with_capacity(n * k);
        for blk in plane.chunks_mut(bb) {
            let seed = u32::from(blk[0]) + u32::from(blk[1]) * 3;
            let d = 0.002 + (seed % 17) as f32 * 0.0007;
            let dmin = 0.001 + (seed % 13) as f32 * 0.0005;
            match scheme {
                2 => {
                    put_f16(blk, 80, d);
                    put_f16(blk, 82, dmin);
                }
                3 => put_f16(blk, 108, d),
                4 | 5 => {
                    put_f16(blk, 0, d);
                    put_f16(blk, 2, dmin);
                }
                _ => put_f16(blk, 208, d * 0.1),
            }
            w.extend(kquant_block(scheme, blk));
        }
        let row_bytes = (blocks * bb) as u32;
        let t = b.u8(n as u32, row_bytes, &plane);
        // Odd cases bind the K-quant dtype itself over the byte row.
        let t = if at % 2 == 1 {
            retype(t, n as u32, row_bytes, dtype)
        } else {
            t
        };
        let group: u32 = if matches!(scheme, 4 | 5) { 32 } else { 16 };
        let groups = k as u32 / group;
        cases.push(Case {
            scheme,
            w: t,
            y: b.zeros(Dtype::Bf16, m as u32, n as u32),
            codes: b.zeros(Dtype::U8, n as u32, k as u32),
            scales: b.zeros(Dtype::F32, n as u32, groups),
            biases: b.zeros(Dtype::F32, n as u32, groups),
            group,
            ya: b.zeros(Dtype::Bf16, m as u32, n as u32),
            yb: b.zeros(Dtype::Bf16, big as u32, n as u32),
            dense: w,
        });
    }
    let lm = b.zeros(Dtype::Bf16, m as u32, n as u32);
    let lm_w = cases[2].w;
    if !b
        .run(|ctx| {
            for c in &cases {
                kquant::matmul(ctx, x, c.w, c.y)?;
                let scheme = kquant::to_affine(ctx, c.w, k as u32, c.codes, c.scales, c.biases)?;
                assert_eq!(kquant::affine_group(scheme), c.group);
                let bank = Bank {
                    codes: c.codes,
                    scales: c.scales,
                    biases: Some(c.biases),
                    group: c.group,
                    bits: 8,
                };
                quant::matmul(ctx, x, bank, c.ya)?;
                quant::matmul(ctx, xbig, bank, c.yb)?;
            }
            kquant::lm_head(ctx, x, lm_w, lm)
        })
        .unwrap()
    {
        return;
    }
    for c in &cases {
        eprintln!("q{}_k", c.scheme);
        let want = gemm_ref(&xs, &c.dense, m, n, k);
        let scale = want.iter().fold(0.0f32, |a, v| a.max(v.abs()));
        assert!(scale > 0.1, "q{}_k answers something", c.scheme);
        let tol = 1e-2 * scale.max(1.0);
        assert_close(&b.read_f32(c.y), &want, tol, 1e-2);
        let exact = gemm_at(&xs, &c.dense, m, n, k, false);
        assert_close(&b.read_f32(c.ya), &exact, tol, 1e-2);
        assert_close(
            &b.read_f32(c.yb),
            &gemm_ref(&xb, &c.dense, big, n, k),
            tol,
            1e-2,
        );
        assert!(
            b.bytes(c.codes).iter().all(|&q| q < 64),
            "codes fit six bits"
        );
    }
    let want = gemm_ref(&xs, &cases[2].dense, m, n, k);
    let scale = want.iter().fold(0.0f32, |a, v| a.max(v.abs()));
    assert_close(&b.read_f32(lm), &want, 1e-2 * scale.max(1.0), 1e-2);
}

// ------------------------------------------------------------------- nvfp4

fn e4m3(byte_: u8) -> f32 {
    let exp = i32::from((byte_ >> 3) & 15);
    let mant = f32::from(byte_ & 7);
    let mag = if exp == 0 {
        mant * 2f32.powi(-9)
    } else if exp == 15 && mant == 7.0 {
        f32::NAN
    } else {
        (1.0 + mant * 0.125) * 2f32.powi(exp - 7)
    };
    if byte_ & 0x80 != 0 { -mag } else { mag }
}

#[test]
fn nvfp4_answers_the_host() {
    let (m, n, k) = (3usize, 10usize, 320usize);
    let xs = data(m * k, 50);
    let lut = [
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    ];
    let codes = bytes(n * k / 2, 51);
    // Scales across normals and subnormals, never the NaN pattern.
    let scales: Vec<u8> = bytes(n * k / 16, 52)
        .into_iter()
        .map(|s| {
            let s = s & 0xbf;
            if s & 0x7f == 0x7f { s ^ 1 } else { s }
        })
        .collect();
    let ts = 0.37f32;
    let w: Vec<f32> = (0..n * k)
        .map(|i| {
            let (r, c) = (i / k, i % k);
            let byte_ = codes[r * k / 2 + c / 2];
            let code = if c % 2 == 0 { byte_ & 15 } else { byte_ >> 4 };
            lut[code as usize] * e4m3(scales[r * k / 16 + c / 16])
        })
        .collect();
    let mut b = Bench::new();
    let x = b.bf16(m as u32, k as u32, &xs);
    let c = b.u8(n as u32, (k / 2) as u32, &codes);
    let s = b.u8(n as u32, (k / 16) as u32, &scales);
    let s = retype(s, n as u32, (k / 16) as u32, Dtype::E4m3);
    let y = b.zeros(Dtype::Bf16, m as u32, n as u32);
    let lm = b.zeros(Dtype::F32, m as u32, n as u32);
    // `big · k / 16 > 4096`: the decoded dense weight.
    let big = 300usize;
    let xb = data(big * k, 53);
    let xbig = b.bf16(big as u32, k as u32, &xb);
    let yb = b.zeros(Dtype::Bf16, big as u32, n as u32);
    if !b
        .run(|ctx| {
            nvfp4::matmul(ctx, x, c, s, ts, y)?;
            nvfp4::matmul(ctx, xbig, c, s, ts, yb)?;
            nvfp4::lm_head(ctx, x, c, s, ts, lm)
        })
        .unwrap()
    {
        return;
    }
    let want: Vec<f32> = gemm_ref(&xs, &w, m, n, k).iter().map(|v| v * ts).collect();
    let scale = want.iter().fold(0.0f32, |a, v| a.max(v.abs()));
    assert!(scale > 0.1);
    assert_close(&b.read_f32(y), &want, 1e-2 * scale, 1e-2);
    assert_close(&b.read_f32(lm), &want, 1e-5 * scale, 1e-5);
    let want: Vec<f32> = gemm_ref(&xb, &w, big, n, k)
        .iter()
        .map(|v| v * ts)
        .collect();
    let scale = want.iter().fold(0.0f32, |a, v| a.max(v.abs()));
    assert_close(&b.read_f32(yb), &want, 1e-2 * scale, 1e-2);
}
