#![cfg(target_vendor = "apple")]

//! PQ2_0 (positional 2-bit) MPP prefill fast-path equivalence. Mirrors
//! `the_ternary_mpp_prefill_matches_native_dots.rs`, but the packed bank is
//! DECODED from a synthetic PQ2_0 bank by `pq2_0_qmm_mpp_pack_bank`: each 34-byte
//! block (a LEADING inline fp16 `d` at bytes 0-1 + `qs[32]` positional 2-bit
//! codes at byte 2) becomes affine MPP packed codes (uint4b code 0..3) with
//! per-64-group scale = `d`, bias = `-d`. `w = (code-1)*d` is exactly the affine
//! accumulate with code=code (0..3), scale=d, bias=-d, so the existing
//! `affine_qmm_mpp` PACKED kernel reproduces the native PQ2_0 dot. The code just
//! ranges 0..3 instead of the ternary 0..2; still fits a uint4b nibble, so the
//! packed layout, nibble/lane order and factors are identical to PTQ1_0 — only
//! the per-element decode differs (d@bytes0-1, positional 2-bit, no staging).
//!
//! Three assertions per case:
//!   1. GPU pack == a bit-exact HOST packed layout (validates nibble/lane order
//!      and the fp16 d -> bf16 scale/bias, decoded in positional element order).
//!   2. Prefill output == the exact reference `sum (code-1)*bf16(d)*x`
//!      (bit-exact at bf16 output in the exact regime; a bf16-output tolerance in
//!      the random regime).
//!   3. Prefill output ~= the native `pq2_0_matmul` QMV on the same bank/act
//!      (cosine >= 0.999) — the MPP-vs-native equivalence.

use engine_metal::device::{Buffer, Context, Handles, Pipelines};
use engine_metal::encode::Sink;
use kernels_metal::encode::{Arg, Encode, Fire, Grid};
use kernels_metal::linear::quant;
use kernels_metal::{Bank, Tensor};
use model_ir::Dtype;

const BLOCK: u32 = 128;
const BLOCK_BYTES: usize = 34;
const QS_OFFSET: usize = 2;

fn noise(at: u64) -> u8 {
    let mut x = at.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0x1234_5678_9ABC_DEF0;
    x ^= x >> 33;
    x = x.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
    (x >> 40) as u8
}

/// Round-to-nearest-even f32 -> bf16 (16-bit) — the same rounding Metal applies
/// in `bfloat(float)` and the affine test uses for its output comparison.
fn bf16_bits(v: f32) -> u16 {
    let b = v.to_bits();
    ((b + 0x7fff + ((b >> 16) & 1)) >> 16) as u16
}

/// The bf16 value of `v` as an f32 (what the kernel sees after `bfloat(v)`).
fn bf16(v: f32) -> f32 {
    f32::from_bits(u32::from(bf16_bits(v)) << 16)
}

fn bf16_le(v: f32) -> [u8; 2] {
    bf16_bits(v).to_le_bytes()
}

/// Finite-normal fp16 bits -> f32 (we only ever build normal fp16 here).
fn f16_to_f32(bits: u16) -> f32 {
    let sign = if bits >> 15 == 1 { -1.0 } else { 1.0 };
    let exp = i32::from((bits >> 10) & 0x1f);
    let man = f32::from(bits & 0x3ff) / 1024.0;
    match exp {
        0 => sign * man * 2f32.powi(-14),
        31 => f32::INFINITY * sign,
        e => sign * (1.0 + man) * 2f32.powi(e - 15),
    }
}

/// The positional 2-bit code (0..3) for natural block element `e` of a 34-byte
/// block, mirroring `quant_pq2_0.metal` `pq2_0_block_dot` / the packer's
/// `pq2_0_block_code` and the host oracle `codec::pq2_0::decode_block`: element
/// `e` -> byte `2 + e/4`, bits `(e%4)*2`. No pow3, no staging.
fn code_at(block: &[u8], e: usize) -> u8 {
    (block[QS_OFFSET + e / 4] >> ((e % 4) * 2)) & 0x3
}

/// The inline fp16 scale `d` of a 34-byte block (LEADING, bytes 0-1), as an f32.
fn block_d(block: &[u8]) -> f32 {
    f16_to_f32(u16::from_le_bytes([block[0], block[1]]))
}

/// Build the fp16 bits for a block's scale. In the exact regime the value is a
/// small bf16-exact fp16 (low 3 mantissa bits zero) so `bf16(d) == float(fp16 d)`
/// and the MPP path is bit-exact against the native dot. In the random regime it
/// is an arbitrary nonzero finite fp16, so native (fp16 d) and MPP (bf16 d)
/// differ — the cosine check proves that is quality-neutral.
fn d_bits(idx: u64, exact: bool) -> u16 {
    if exact {
        let m = (u16::from(noise(idx)) & 0x7f) << 3; // 7 random bits, low 3 zero
        (9 << 10) | m // exp 9 -> base 2^-6 (~0.0156..0.0312)
    } else {
        let e = 11 + u16::from(noise(idx ^ 0xAA)) % 3; // 11..13 -> ~0.06..0.5
        let m = ((u16::from(noise(idx)) << 2) | (u16::from(noise(idx ^ 0x55)) & 3)) & 0x3ff;
        (e << 10) | m
    }
}

#[test]
fn every_pq2_0_prefill_row_answers_the_native_pq2_0_dot() {
    let Ok(device) = Context::bind() else {
        eprintln!("not asked: no Metal device");
        return;
    };
    eprintln!("device: {}", device.name());
    assert!(kernels_metal::tuning::override_with(
        kernels_metal::tuning::Overrides {
            qmm_mpp: Some(true),
            ..Default::default()
        }
    ));
    let handles = Handles::new();
    let pipelines = Pipelines::new();
    let mut checked = 0u64;
    let mut native_checked = 0u64;
    // Small shapes exercise the bm=16/24/32 MPP tiles (and the split/partial
    // reduce) at m in {17,32,65}; the Bonsai projections exercise the big GEMM
    // at m in {8,2048}.
    let cases = [
        (256u32, 256u32),
        (512, 1024),
        (5120, 34816),
        (17408, 5120),
        (5120, 12288),
        (6144, 5120),
    ];
    for exact in [true, false] {
        for &(k, n) in &cases {
            let groups = k / 64;
            let blocks_per_row = k / BLOCK;
            let salt: u64 = if exact { 0 } else { 0x9E37 };

            // Synthetic native PQ2_0 bank: n rows x blocks_per_row 34-byte blocks.
            // Each block: fp16 `d` LEADING at bytes 0-1, then 32 code bytes.
            let row_bytes = blocks_per_row as usize * BLOCK_BYTES;
            let mut codes = vec![0u8; n as usize * row_bytes];
            for col in 0..n {
                for b in 0..blocks_per_row {
                    let g_blk = u64::from(col) * u64::from(blocks_per_row) + u64::from(b);
                    let start = col as usize * row_bytes + b as usize * BLOCK_BYTES;
                    let bits = d_bits(g_blk ^ salt, exact);
                    codes[start] = (bits & 0xff) as u8;
                    codes[start + 1] = (bits >> 8) as u8;
                    for p in QS_OFFSET..BLOCK_BYTES {
                        codes[start + p] = noise(g_blk.wrapping_mul(37) + p as u64 + salt);
                    }
                }
            }
            let block_bytes = |col: u32, b: u32| -> &[u8] {
                let start = col as usize * row_bytes + b as usize * BLOCK_BYTES;
                &codes[start..start + BLOCK_BYTES]
            };

            // HOST packed mirror: codes region (uint4b code) + bf16 scale/bias.
            let mut packed = vec![0u8; (n * k / 2) as usize];
            for col in 0..n {
                for g in 0..groups {
                    let param = (col / 256 * groups + g) * 256 + col % 256;
                    let base = (param * 32) as usize;
                    let half = (g % 2) as usize;
                    let block = block_bytes(col, g / 2);
                    for l in 0..64usize {
                        let c = code_at(block, half * 64 + l);
                        packed[base + l / 2] |= (c & 0xf) << ((l % 2) * 4);
                    }
                }
            }
            let mut scales_plane = vec![0u8; (n * groups * 2) as usize];
            let mut biases_plane = vec![0u8; (n * groups * 2) as usize];
            for col in 0..n {
                for g in 0..groups {
                    let param = (col / 256 * groups + g) * 256 + col % 256;
                    let d = block_d(block_bytes(col, g / 2));
                    let off = (param * 2) as usize;
                    scales_plane[off..off + 2].copy_from_slice(&bf16_le(d));
                    biases_plane[off..off + 2].copy_from_slice(&bf16_le(-d));
                }
            }
            packed.extend_from_slice(&scales_plane);
            packed.extend_from_slice(&biases_plane);

            let upload = |bytes: &[u8]| {
                let mut buffer = Buffer::zeroed(&device, bytes.len() as u64).expect("a plane");
                buffer.write(0, bytes).expect("upload");
                buffer
            };
            let bind = |b: &Buffer| handles.bind(b, 0, b.bytes()).expect("bind");
            let codes_b = upload(&codes);
            let packed_b = Buffer::zeroed(&device, packed.len() as u64).expect("GPU packed bank");

            // Fire the PQ2_0 packer and assert bit-exact against the host mirror.
            {
                let frame = device.frame().expect("pack frame");
                let sink = Sink::new(&device, &frame, &pipelines, &handles);
                let src = Tensor::new(bind(&codes_b), n, k, Dtype::Pq2_0);
                let dest = Tensor::new(bind(&packed_b), n, k, Dtype::U4g64);
                sink.fire(
                    Fire::at("linear/quant_qmm_mpp.metal", "pq2_0_qmm_mpp_pack_bank")
                        .apply(Grid::of([n * (k / 8), 1, 1], [256, 1, 1])),
                    &[src.arg(), dest.arg_mut(), k.arg(), n.arg()],
                )
                .expect("pack");
                frame.commit().expect("pack complete");
                assert_eq!(
                    handles
                        .read(dest.buf, packed.len() as u64)
                        .expect("packed readback"),
                    packed,
                    "PQ2_0 GPU pack reproduces the host uint4b/scale/bias layout K={k} N={n}"
                );
            }

            let pq2 = |mpp: bool| Bank {
                codes: Tensor::new(bind(&codes_b), n, k, Dtype::Pq2_0),
                mpp_codes: mpp.then(|| Tensor::new(bind(&packed_b), n, k, Dtype::U4g64)),
                // Native-PQ2 placeholders (unused by both paths): the MPP
                // dispatch rebuilds a g64/4-bit pseudo-bank, and the QMV path
                // reads only `codes`.
                scales: Tensor::new(bind(&codes_b), n, k, Dtype::Pq2_0),
                biases: None,
                group: 128,
                bits: 2,
            };

            // Decoded (code-1) + per-position bf16(d) for each sampled column, to
            // build the exact reference without holding every weight.
            let col_step: u32 = if n > 1024 { n / 32 } else { 1 };
            let sampled_cols: Vec<u32> = (0..n).step_by(col_step as usize).collect();
            let mut code_rows: std::collections::HashMap<u32, (Vec<i32>, Vec<f32>)> =
                std::collections::HashMap::new();
            for &col in &sampled_cols {
                let mut cv = vec![0i32; k as usize];
                let mut dd = vec![0f32; k as usize];
                for b in 0..blocks_per_row {
                    let block = block_bytes(col, b);
                    let d = bf16(block_d(block));
                    for e in 0..128usize {
                        let at = b as usize * 128 + e;
                        cv[at] = i32::from(code_at(block, e)) - 1;
                        dd[at] = d;
                    }
                }
                code_rows.insert(col, (cv, dd));
            }

            let mut cos_dot = 0.0f64;
            let mut cos_a = 0.0f64;
            let mut cos_b = 0.0f64;

            let activation = |row: u32, at: u32| -> f32 {
                if exact {
                    (((row * 17 + at * 3) % 29) as i32 - 14) as f32 / 16.0
                } else {
                    bf16(
                        (f32::from(noise(
                            u64::from(row) * u64::from(k) + u64::from(at) ^ 0xABCD,
                        )) - 127.0)
                            * 0.0197,
                    )
                }
            };

            for m in [8u32, 16, 17, 32, 65, 2048] {
                // Mirror the ternary test's shape budget: big K runs only the
                // light (m=8) and heavy (m=2048) rows; small K runs the mid rows.
                if k > 512 && ![8, 2048].contains(&m) {
                    continue;
                }
                if k <= 512 && m == 2048 {
                    continue;
                }
                let fires_mpp =
                    m >= 8 && (m > 16 || n >= 1024) && n % (if m <= 8 { 64 } else { 128 }) == 0;
                if !fires_mpp {
                    continue;
                }
                let cap = m.div_ceil(64) * 64;
                let inputs: Vec<u8> = (0..cap)
                    .flat_map(|row| (0..k).flat_map(move |at| bf16_le(activation(row, at))))
                    .collect();
                let act_b = upload(&inputs);
                let out_bytes = u64::from(cap) * u64::from(n) * 2;
                let out_b = upload(&vec![0xa5; out_bytes as usize + 64]);
                let staging =
                    Buffer::zeroed(&device, u64::from(cap) * u64::from(k) * 2).expect("staging");
                let (ha, ho, hs) = (bind(&act_b), bind(&out_b), bind(&staging));
                let precast = |rows, width| {
                    (rows <= cap && width == k).then(|| Tensor::new(hs, rows, width, Dtype::F16))
                };
                let partial_bytes = u64::from(cap) * 8 * u64::from(n) * 4;
                let partial_buffer = upload(&vec![0xa5; partial_bytes as usize + 64]);
                let ph = bind(&partial_buffer);
                let partials = |rows, width| {
                    (rows <= cap * 8 && width == n)
                        .then(|| Tensor::new(ph, rows, width, Dtype::F32))
                };
                {
                    let frame = device.frame().expect("frame");
                    let sink = Sink::new(&device, &frame, &pipelines, &handles);
                    quant::matmul(
                        &sink,
                        Tensor::new(ha, m, k, Dtype::Bf16),
                        pq2(true),
                        Tensor::new(ho, m, n, Dtype::Bf16),
                        quant::Scratch {
                            precast: &precast,
                            partials: &partials,
                        },
                        cap,
                    )
                    .expect("PQ2_0 MPP prefill");
                    frame.commit().expect("commit");
                }
                let got = handles.read(ho, out_bytes + 64).expect("read");
                let pp = handles.read(ph, partial_bytes + 64).expect("partial guard");
                assert!(
                    pp[partial_bytes as usize..].iter().all(|&b| b == 0xa5),
                    "partial output overflow M={m} K={k} N={n}"
                );
                assert!(
                    got[out_bytes as usize..].iter().all(|&b| b == 0xa5),
                    "write beyond output capacity M={m} K={k} N={n}"
                );

                // Native QMV on the same bank/activation (no mpp_codes) for the
                // cosine cross-check — skip the heavy m=2048 QMV.
                let native = (m <= 65).then(|| {
                    let nb = upload(&vec![0xa5; out_bytes as usize + 64]);
                    let hn = bind(&nb);
                    let none = |_: u32, _: u32| None;
                    let frame = device.frame().expect("native frame");
                    let sink = Sink::new(&device, &frame, &pipelines, &handles);
                    quant::matmul(
                        &sink,
                        Tensor::new(ha, m, k, Dtype::Bf16),
                        pq2(false),
                        Tensor::new(hn, m, n, Dtype::Bf16),
                        quant::Scratch {
                            precast: &none,
                            partials: &none,
                        },
                        cap,
                    )
                    .expect("native PQ2_0 QMV");
                    frame.commit().expect("native commit");
                    (nb, handles.read(hn, out_bytes + 64).expect("native read"))
                });

                for row in 0..m {
                    for &col in &sampled_cols {
                        let (cv, dd) = &code_rows[&col];
                        let mut want = 0.0f64;
                        let mut absolute = 0.0f64;
                        for at in 0..k as usize {
                            let term = f64::from(cv[at])
                                * f64::from(dd[at])
                                * f64::from(activation(row, at as u32));
                            want += term;
                            absolute += term.abs();
                        }
                        let offset = ((row * n + col) * 2) as usize;
                        let actual = u16::from_le_bytes([got[offset], got[offset + 1]]);
                        let actual_f32 = f32::from_bits(u32::from(actual) << 16);
                        if exact {
                            let rounded = bf16_bits(want as f32);
                            assert_eq!(
                                actual, rounded,
                                "exact M={m} K={k} N={n} row={row} col={col}: {actual_f32} vs {want}"
                            );
                        } else {
                            let allowance =
                                0.00002 + f64::from(groups) * f64::from(f32::EPSILON) * absolute;
                            assert!(
                                (f64::from(actual_f32) - want).abs()
                                    <= 0.004 * want.abs() + allowance,
                                "random M={m} K={k} N={n} row={row} col={col}: {actual_f32} vs {want}"
                            );
                        }
                        checked += 1;

                        if let Some((_, ref nbytes)) = native {
                            let nv = u16::from_le_bytes([nbytes[offset], nbytes[offset + 1]]);
                            let nf = f32::from_bits(u32::from(nv) << 16);
                            cos_dot += f64::from(actual_f32) * f64::from(nf);
                            cos_a += f64::from(actual_f32) * f64::from(actual_f32);
                            cos_b += f64::from(nf) * f64::from(nf);
                            native_checked += 1;
                        }
                    }
                }
            }
            if cos_a > 0.0 && cos_b > 0.0 {
                let cosine = cos_dot / (cos_a.sqrt() * cos_b.sqrt());
                assert!(
                    cosine >= 0.999,
                    "MPP-vs-native cosine {cosine} below 0.999 at K={k} N={n} (exact={exact})"
                );
            }
        }
    }
    eprintln!(
        "checked {checked} PQ2_0 prefill outputs against the exact (code-1)*d*x dot, \
         and {native_checked} of them against the native PQ2_0 QMV (cosine >= 0.999)"
    );
}
