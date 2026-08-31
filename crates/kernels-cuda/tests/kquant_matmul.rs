//! **THE GGUF K-QUANT PROJECTIONS, ON A REAL DEVICE** (QNF wave, alto
//! `next.md` §J2 priority 3): `linear::kquant::matmul` / `lm_head` over a
//! weight served AS STORED — ggml super-blocks, decoded inside the dot —
//! held against a host decode of the same bytes.
//!
//! ```text
//! cargo test -p kernels-cuda --features cuda-13 --test kquant_matmul -- --nocapture
//! ```
//!
//! The gates:
//!
//! ```text
//! (a) q4_k: y = act · W^T matches a host decode of the super-blocks, on a
//!     rectangle whose N is not a whole row tile (the clamp path runs)
//! (b) q6_k: the same, through the strided quarter/high-pair addressing —
//!     the scheme a Q4_K_M mix stores `output.weight` at
//! (c) `lm_head` answers what `matmul` answers, byte for byte
//! (d) a K that is not a whole number of super-blocks is refused, and a row
//!     whose byte width names neither scheme is refused by name
//! ```
//!
//! **THE ORACLE IS THIS FILE'S OWN DECODE, TRANSCRIBED FROM THE TREE'S.**
//! `checkpoint`'s `decode_gguf_q4_k_block_into` / `decode_gguf_q6_k_block_into`
//! are private to `checkpoint::executor::walk`, so they cannot be called from
//! here; the reference below is written from them, byte offset for byte
//! offset. It decodes each super-block to its 256 values and folds the dot
//! straight, where the kernel folds q4_k through the affine identity
//! (`d·sc·Σqx − dmin·m·Σx`) — so the comparison also holds that rearrangement,
//! not only the byte layout.

#![cfg(feature = "_cuda")]

mod common;

use common::{Gpu, Lcg, close, from_bf16};
use dtype::Dtype;
use kernels_cuda::linear::kquant;
use kernels_cuda::tensor::Tensor;

const TOKENS: u32 = 3;
/// Five weight rows against a sixteen-row tile: the clamp path runs.
const N: u32 = 5;
const K: u32 = 512;
const SUPER: usize = 256;
const BLOCKS: usize = K as usize / SUPER;

const Q4K_BYTES: usize = 144;
const Q6K_BYTES: usize = 210;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Scheme {
    Q4K,
    Q6K,
}

impl Scheme {
    const fn bytes(self) -> usize {
        match self {
            Self::Q4K => Q4K_BYTES,
            Self::Q6K => Q6K_BYTES,
        }
    }
}

/// `prelude/device.cuh`'s `f16_to_f32`, for the normal range these blocks are
/// filled from. A golden that decoded the scale differently from the kernel
/// would be measuring the decode.
fn from_f16(bits: u16) -> f32 {
    let sign = (u32::from(bits) & 0x8000) << 16;
    let exp = (u32::from(bits) >> 10) & 0x1f;
    let mantissa = u32::from(bits) & 0x3ff;
    assert!(exp > 0 && exp < 31, "the fillers stay in the normal range");
    f32::from_bits(sign | ((exp + 112) << 23) | (mantissa << 13))
}

/// A positive f16 in `[2^exp, 2^(exp+1))`, drawn from the seeded filler.
/// Magnitudes are chosen so a decoded weight lands near `0.05` and a row of
/// 512 sums to `O(1)` — tame enough that the comparison measures the decode
/// and not an overflow.
fn f16_at(rng: &mut Lcg, exp: i32) -> u16 {
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let mantissa = (((rng.unit() + 1.0) * 512.0) as u32) & 0x3ff;
    #[allow(clippy::cast_sign_loss)]
    let field = (exp + 15) as u32;
    ((field << 10) | mantissa) as u16
}

fn byte(rng: &mut Lcg) -> u8 {
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let v = ((rng.unit() + 1.0) * 128.0) as u32;
    (v & 0xff) as u8
}

/// One `block_q4_K`: `d` f16, `dmin` f16, twelve packed six-bit scale/min
/// fields, 128 bytes of nibbles.
fn q4k_block(rng: &mut Lcg, out: &mut Vec<u8>) {
    out.extend_from_slice(&f16_at(rng, -12).to_le_bytes());
    out.extend_from_slice(&f16_at(rng, -9).to_le_bytes());
    for _ in 0..12 {
        out.push(byte(rng));
    }
    for _ in 0..128 {
        out.push(byte(rng));
    }
}

/// One `block_q6_K`: 128 low nibbles, 64 high pairs, sixteen signed sub-block
/// scales, `d` f16. The scales are held to `-40..=40` so `d·scale·q` stays in
/// the same tame band the q4_k filler lands in.
fn q6k_block(rng: &mut Lcg, out: &mut Vec<u8>) {
    for _ in 0..192 {
        out.push(byte(rng));
    }
    for _ in 0..16 {
        let s = i32::from(byte(rng) % 81) - 40;
        #[allow(clippy::cast_possible_truncation)]
        out.push(s as i8 as u8);
    }
    out.extend_from_slice(&f16_at(rng, -12).to_le_bytes());
}

/// The whole `[N, k/256]` plane, rows back to back.
fn plane(scheme: Scheme, rng: &mut Lcg) -> Vec<u8> {
    let mut out = Vec::with_capacity(N as usize * BLOCKS * scheme.bytes());
    for _ in 0..(N as usize * BLOCKS) {
        match scheme {
            Scheme::Q4K => q4k_block(rng, &mut out),
            Scheme::Q6K => q6k_block(rng, &mut out),
        }
    }
    out
}

/// ggml's `get_scale_min_k4`: the six-bit scale and minimum of sub-block
/// `index`, spliced out of the twelve bytes the eight sub-blocks share.
fn q4k_scale_min(index: usize, s: &[u8]) -> (u8, u8) {
    if index < 4 {
        (s[index] & 63, s[index + 4] & 63)
    } else {
        let scale = (s[index + 4] & 0x0f) | ((s[index - 4] >> 6) << 4);
        let min = (s[index + 4] >> 4) | ((s[index] >> 6) << 4);
        (scale, min)
    }
}

/// One `q4_k` super-block to its 256 values — affine, the minimum
/// *subtracted*, read in pairs of sub-blocks (low nibbles for the even one,
/// high for the odd).
fn decode_q4k(block: &[u8], values: &mut [f32; SUPER]) {
    let d = from_f16(u16::from_le_bytes([block[0], block[1]]));
    let dmin = from_f16(u16::from_le_bytes([block[2], block[3]]));
    let scales = &block[4..16];
    let qs = &block[16..144];
    for pair in 0..4 {
        let (sc_lo, m_lo) = q4k_scale_min(pair * 2, scales);
        let (sc_hi, m_hi) = q4k_scale_min(pair * 2 + 1, scales);
        let (d_lo, min_lo) = (d * f32::from(sc_lo), dmin * f32::from(m_lo));
        let (d_hi, min_hi) = (d * f32::from(sc_hi), dmin * f32::from(m_hi));
        let packed = &qs[pair * 32..pair * 32 + 32];
        let out = pair * 64;
        for i in 0..32 {
            values[out + i] = d_lo * f32::from(packed[i] & 0x0f) - min_lo;
            values[out + 32 + i] = d_hi * f32::from(packed[i] >> 4) - min_hi;
        }
    }
}

/// One `q6_k` super-block to its 256 values — symmetric, two halves of 128,
/// the four quarters of a half strided rather than contiguous, the sub-block
/// scale index advancing by two per quarter.
fn decode_q6k(block: &[u8], values: &mut [f32; SUPER]) {
    let d = from_f16(u16::from_le_bytes([block[208], block[209]]));
    for half in 0..2 {
        let ql = &block[half * 64..half * 64 + 64];
        let qh = &block[128 + half * 32..128 + half * 32 + 32];
        let scales = &block[192 + half * 8..192 + half * 8 + 8];
        let out = half * 128;
        for i in 0..32 {
            let sub = i / 16;
            for quarter in 0..4 {
                let nibble = if quarter < 2 {
                    ql[i + 32 * quarter] & 0x0f
                } else {
                    ql[i + 32 * (quarter - 2)] >> 4
                };
                let top = (qh[i] >> (2 * quarter)) & 3;
                let q = i32::from(nibble | (top << 4)) - 32;
                #[allow(clippy::cast_possible_wrap, clippy::cast_precision_loss)]
                let scale = f32::from(scales[sub + 2 * quarter] as i8);
                #[allow(clippy::cast_precision_loss)]
                let value = d * scale * q as f32;
                values[out + quarter * 32 + i] = value;
            }
        }
    }
}

/// The host fold: decode each super-block, then contract it against the
/// matching 256 activations.
fn fold(scheme: Scheme, plane: &[u8], x: &[f32]) -> Vec<f32> {
    let stride = scheme.bytes();
    let mut y = vec![0.0f32; TOKENS as usize * N as usize];
    let mut values = [0.0f32; SUPER];
    for r in 0..N as usize {
        for g in 0..BLOCKS {
            let at = (r * BLOCKS + g) * stride;
            let block = &plane[at..at + stride];
            match scheme {
                Scheme::Q4K => decode_q4k(block, &mut values),
                Scheme::Q6K => decode_q6k(block, &mut values),
            }
            for t in 0..TOKENS as usize {
                let xt = &x[t * K as usize + g * SUPER..][..SUPER];
                let mut acc = 0.0f32;
                for e in 0..SUPER {
                    acc += values[e] * xt[e];
                }
                y[t * N as usize + r] += acc;
            }
        }
    }
    y
}

fn fired(scheme: Scheme, head: bool) -> (Vec<u16>, Vec<f32>) {
    let mut gpu = Gpu::open();
    let seed = if scheme == Scheme::Q4K { 0x4b1 } else { 0x6b1 };
    let mut rng = Lcg::seeded(seed);
    let bytes = plane(scheme, &mut rng);
    let (x_raw, x) = rng.row(TOKENS as usize * K as usize);

    let act = Tensor::new(gpu.up(&x_raw), TOKENS, K, Dtype::Bf16);
    #[allow(clippy::cast_possible_truncation)]
    let row_bytes = (BLOCKS * scheme.bytes()) as u32;
    let w = Tensor::new(gpu.up(&bytes), N, row_bytes, Dtype::U8);
    let y_at = gpu.zeros(TOKENS as usize * N as usize * 2);
    let mut y = Tensor::new(y_at, TOKENS, N, Dtype::Bf16);

    let entry = if head { kquant::lm_head } else { kquant::matmul };
    entry(&gpu.ctx(), act, w, &mut y).expect("the K-quant projection fires");
    gpu.sync();

    let got: Vec<u16> = gpu.down(y_at, TOKENS as usize * N as usize);
    (got, fold(scheme, &bytes, &x))
}

fn held(scheme: Scheme, head: bool) {
    let (got, want) = fired(scheme, head);
    let name = if scheme == Scheme::Q4K { "q4_k" } else { "q6_k" };
    for (at, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        let g = from_bf16(*g);
        assert!(
            close(g, *w),
            "y[{at}] answered {g} where the host decode says {w} ({name})"
        );
    }
}

#[test]
fn the_q4k_projection_matches_the_host_decode() {
    held(Scheme::Q4K, false);
}

#[test]
fn the_q6k_projection_matches_the_host_decode() {
    held(Scheme::Q6K, false);
}

#[test]
fn the_head_answers_what_the_matmul_answers() {
    // q6_k on purpose: a Q4_K_M mix stores `output.weight` at q6_k, so this
    // is the pairing the head actually fires.
    held(Scheme::Q6K, true);
    let (by_head, _) = fired(Scheme::Q6K, true);
    let (by_matmul, _) = fired(Scheme::Q6K, false);
    assert_eq!(
        by_head, by_matmul,
        "one launch under two names answered two things"
    );
}

#[test]
fn a_partial_super_block_and_a_foreign_row_width_are_refused() {
    let mut gpu = Gpu::open();
    let mut y = Tensor::new(gpu.zeros(2), 1, 1, Dtype::Bf16);

    // 128 is half a super-block: no whole number of blocks spans the row.
    let short = Tensor::new(gpu.zeros(2 * 128), 1, 128, Dtype::Bf16);
    let w = Tensor::new(gpu.zeros(Q4K_BYTES), 1, 144, Dtype::U8);
    let refused = kquant::matmul(&gpu.ctx(), short, w, &mut y);
    assert!(refused.is_err(), "a partial super-block fired");

    // A whole K, and a row width that is neither 2·144 nor 2·210.
    let act = Tensor::new(gpu.zeros(2 * K as usize), 1, K, Dtype::Bf16);
    let foreign = Tensor::new(gpu.zeros(2 * 176), 1, 2 * 176, Dtype::U8);
    let refused = kquant::matmul(&gpu.ctx(), act, foreign, &mut y);
    assert!(refused.is_err(), "a q5_k-wide row fired as a K-quant");
}
