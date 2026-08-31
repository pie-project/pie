//! **THE GGUF K-QUANT PROJECTIONS, ON A REAL DEVICE** (QNF wave, alto
//! `next.md` §J2 priority 3): `linear::kquant::matmul` / `lm_head` over a
//! weight served AS STORED — ggml super-blocks, decoded inside the dot —
//! held against a host decode of the same bytes.
//!
//! ```text
//! cargo test -p kernels-cuda --features cuda-13 --test kquant_matmul -- --nocapture
//! ```
//!
//! The gates, one per scheme and then the family's own:
//!
//! ```text
//! (a) q4_k: y = act · W^T matches a host decode of the super-blocks, on a
//!     rectangle whose N is not a whole row tile (the clamp path runs)
//! (b) q6_k: the same, through the strided quarter/high-pair addressing —
//!     the scheme a Q4_K_M mix stores `output.weight` at
//! (c) q2_k: the same, with the super-scales read from the END of the block
//!     and a four-bit scale/min pair per sixteen-element sub-block
//! (d) q3_k: the same, over BOTH polarities of the inverted third-bit mask —
//!     counted, not hoped for — and both signs of the biased six-bit scale
//! (e) q5_k: the same, with SET and CLEAR fifth bits — counted the same way
//! (f) `lm_head` answers what `matmul` answers, byte for byte, at q6_k (the
//!     head's own scheme in a Q4_K_M mix) and at q5_k (one of the three this
//!     wave added, so the new dispatch arms are held under both names)
//! (g) a K that is not a whole number of super-blocks is refused, and a row
//!     whose byte width names none of the FIVE schemes is refused by a
//!     message that names all five widths and all five schemes
//! ```
//!
//! **THE ORACLE IS THIS FILE'S OWN DECODE, TRANSCRIBED FROM THE TREE'S.**
//! `checkpoint`'s `decode_gguf_q{2,3,4,5,6}_k_block_into` are private to
//! `checkpoint::executor::walk`, so they cannot be called from here; the
//! references below are written from them, byte offset for byte offset,
//! splice for splice. Each decodes a super-block to its 256 values and folds
//! the dot straight, where the kernel folds the three affine schemes through
//! the identity `d·sc·Σqx − dmin·m·Σx` — so the comparison also holds that
//! rearrangement, not only the byte layout.
//!
//! Two of the tree's decoders (q2_k, q3_k) carry a bit-identity check against
//! the `gguf` package over a whole shipped tensor, recorded in their own
//! docs. Transcribing them rather than re-deriving is what lets this file
//! inherit that evidence instead of restating a layout from memory.

#![cfg(feature = "_cuda")]

mod common;

use common::{Gpu, Lcg, close, from_bf16};
use dtype::Dtype;
use kernels_cuda::error::Error;
use kernels_cuda::linear::kquant;
use kernels_cuda::tensor::Tensor;

const TOKENS: u32 = 3;
/// Five weight rows against a sixteen-row tile: the clamp path runs.
const N: u32 = 5;
const K: u32 = 512;
const SUPER: usize = 256;
const BLOCKS: usize = K as usize / SUPER;

const Q2K_BYTES: usize = 84;
const Q3K_BYTES: usize = 110;
const Q4K_BYTES: usize = 144;
const Q5K_BYTES: usize = 176;
const Q6K_BYTES: usize = 210;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Scheme {
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
}

impl Scheme {
    const fn bytes(self) -> usize {
        match self {
            Self::Q2K => Q2K_BYTES,
            Self::Q3K => Q3K_BYTES,
            Self::Q4K => Q4K_BYTES,
            Self::Q5K => Q5K_BYTES,
            Self::Q6K => Q6K_BYTES,
        }
    }

    /// GGUF's own name for the scheme, for the line a failing golden prints.
    const fn name(self) -> &'static str {
        match self {
            Self::Q2K => "q2_k",
            Self::Q3K => "q3_k",
            Self::Q4K => "q4_k",
            Self::Q5K => "q5_k",
            Self::Q6K => "q6_k",
        }
    }

    /// A filler seed per scheme, so re-tuning one scheme's magnitudes cannot
    /// move another scheme's plane and quietly re-baseline its golden.
    const fn seed(self) -> u64 {
        match self {
            Self::Q2K => 0x2b1,
            Self::Q3K => 0x3b1,
            Self::Q4K => 0x4b1,
            Self::Q5K => 0x5b1,
            Self::Q6K => 0x6b1,
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

/// One `block_q2_K`: sixteen packed four-bit scale/min bytes, 64 bytes of
/// 2-bit codes, then `d` and `dmin` — the super-scales AFTER the payload,
/// which is this block's order alone in the family. A four-bit scale reaches
/// 15 and a 2-bit code 3, so `d` runs an exponent hotter than q4_k's to land
/// the decoded weight in the same tame band.
fn q2k_block(rng: &mut Lcg, out: &mut Vec<u8>) {
    for _ in 0..80 {
        out.push(byte(rng));
    }
    out.extend_from_slice(&f16_at(rng, -10).to_le_bytes());
    out.extend_from_slice(&f16_at(rng, -9).to_le_bytes());
}

/// One `block_q3_K`: 32 bytes of third-bit mask, 64 of 2-bit codes, twelve
/// packed six-bit scale bytes, `d`. Every payload byte comes from the filler,
/// so the mask carries BOTH polarities and the biased scales both signs —
/// `q3k_mask_census` is what holds the first rather than assuming it. The
/// scale reaches ±32 and the borrowed code ±4, so `d` sits between q4_k's
/// and q2_k's.
fn q3k_block(rng: &mut Lcg, out: &mut Vec<u8>) {
    for _ in 0..108 {
        out.push(byte(rng));
    }
    out.extend_from_slice(&f16_at(rng, -11).to_le_bytes());
}

/// One `block_q5_K`: q4_k's head — `d`, `dmin`, twelve packed scale/min
/// bytes — then 32 bytes of fifth-bit plane ahead of the 128 of nibbles. The
/// fifth bit doubles q4_k's code range, so `d` runs an exponent cooler.
fn q5k_block(rng: &mut Lcg, out: &mut Vec<u8>) {
    out.extend_from_slice(&f16_at(rng, -13).to_le_bytes());
    out.extend_from_slice(&f16_at(rng, -9).to_le_bytes());
    for _ in 0..12 {
        out.push(byte(rng));
    }
    for _ in 0..32 {
        out.push(byte(rng));
    }
    for _ in 0..128 {
        out.push(byte(rng));
    }
}

/// The whole `[N, k/256]` plane, rows back to back.
fn plane(scheme: Scheme, rng: &mut Lcg) -> Vec<u8> {
    let mut out = Vec::with_capacity(N as usize * BLOCKS * scheme.bytes());
    for _ in 0..(N as usize * BLOCKS) {
        match scheme {
            Scheme::Q2K => q2k_block(rng, &mut out),
            Scheme::Q3K => q3k_block(rng, &mut out),
            Scheme::Q4K => q4k_block(rng, &mut out),
            Scheme::Q5K => q5k_block(rng, &mut out),
            Scheme::Q6K => q6k_block(rng, &mut out),
        }
    }
    out
}

/// **BOTH POLARITIES OF THE `q3_k` MASK, COUNTED RATHER THAN HOPED FOR.** A
/// set bit subtracts nothing and a clear bit subtracts four, so a plane that
/// happened to be all ones would pass a kernel that ignored the mask
/// entirely, and a plane of all zeros one that inverted it. Walks the mask
/// exactly as the decoder does — the 32 bytes read eight times, one bit each
/// time, the selector advancing ACROSS both windows — and answers
/// `(set, clear)` over the plane's every element.
fn q3k_mask_census(plane: &[u8]) -> (usize, usize) {
    let (mut set, mut clear) = (0usize, 0usize);
    for block in plane.chunks_exact(Q3K_BYTES) {
        let hmask = &block[0..32];
        let mut selector = 1u8;
        for _window in 0..2 {
            for _step in 0..4 {
                for at in 0..32 {
                    if hmask[at] & selector == 0 {
                        clear += 1;
                    } else {
                        set += 1;
                    }
                }
                selector = selector.wrapping_shl(1);
            }
        }
    }
    (set, clear)
}

/// **SET AND CLEAR FIFTH BITS, COUNTED THE SAME WAY.** Sub-block `b` takes
/// bit `b` of the plane byte at its offset, so one byte serves all eight
/// sub-blocks; a plane of zeros would leave `q5_k` indistinguishable from
/// `q4_k` at a different stride.
fn q5k_fifth_census(plane: &[u8]) -> (usize, usize) {
    let (mut set, mut clear) = (0usize, 0usize);
    for block in plane.chunks_exact(Q5K_BYTES) {
        let bits = &block[16..48];
        for b in 0..8 {
            for at in 0..32 {
                if (bits[at] >> b) & 1 == 1 {
                    set += 1;
                } else {
                    clear += 1;
                }
            }
        }
    }
    (set, clear)
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

/// One `q2_k` super-block to its 256 values. Affine like q4_k, but with a
/// FOUR-bit scale and four-bit minimum per sixteen-element sub-block sharing
/// one of the sixteen leading bytes, and with `d`/`dmin` read from the END
/// of the block. The walk is two windows of 32 payload bytes, four shifts
/// per window, two halves of sixteen per shift; sub-block index and output
/// index advance together.
fn decode_q2k(block: &[u8], values: &mut [f32; SUPER]) {
    let scales = &block[0..16];
    let qs = &block[16..80];
    let d = from_f16(u16::from_le_bytes([block[80], block[81]]));
    let dmin = from_f16(u16::from_le_bytes([block[82], block[83]]));
    let mut out = 0;
    let mut sub = 0;
    for window in 0..2 {
        let q = &qs[window * 32..window * 32 + 32];
        for step in 0..4 {
            let shift = 2 * step;
            for half in 0..2 {
                let packed = scales[sub];
                sub += 1;
                let dl = d * f32::from(packed & 0x0f);
                let ml = dmin * f32::from(packed >> 4);
                for l in 0..16 {
                    values[out] = dl * f32::from((q[half * 16 + l] >> shift) & 3) - ml;
                    out += 1;
                }
            }
        }
    }
}

/// ggml's four-`u32` splice: the sixteen six-bit scales of a `q3_k` block,
/// each BIASED BY 32, out of twelve bytes. A different splice from
/// `q4k_scale_min` — sixteen scales and no minimums, because q3_k is
/// symmetric. Kept in the four-word form the reference is written in.
fn q3k_scales(raw: &[u8]) -> [i8; 16] {
    const LOW_NIBBLES: u32 = 0x0f0f_0f0f;
    const BIT_PAIRS: u32 = 0x0303_0303;
    let word = |i: usize| {
        u32::from_le_bytes([raw[4 * i], raw[4 * i + 1], raw[4 * i + 2], raw[4 * i + 3]])
    };
    let (a, b, top) = (word(0), word(1), word(2));
    let aux = [
        (a & LOW_NIBBLES) | ((top & BIT_PAIRS) << 4),
        (b & LOW_NIBBLES) | (((top >> 2) & BIT_PAIRS) << 4),
        ((a >> 4) & LOW_NIBBLES) | (((top >> 4) & BIT_PAIRS) << 4),
        ((b >> 4) & LOW_NIBBLES) | (((top >> 6) & BIT_PAIRS) << 4),
    ];
    let mut scales = [0i8; 16];
    for (i, slot) in scales.iter_mut().enumerate() {
        #[allow(clippy::cast_possible_wrap)]
        {
            *slot = aux[i / 4].to_le_bytes()[i % 4] as i8;
        }
    }
    scales
}

/// One `q3_k` super-block to its 256 values — symmetric, `d·(sc − 32)·q`.
///
/// The mask reads INVERTED: a SET bit subtracts nothing and a CLEAR bit
/// subtracts four, because ggml stores the two low bits of `q + 4` and sets
/// the bit when the value needed no borrow. The selector is also not
/// restarted at the second window — it runs 1, 2, 4 … 128 across both, and
/// restarting it would corrupt only the block's upper half.
fn decode_q3k(block: &[u8], values: &mut [f32; SUPER]) {
    let hmask = &block[0..32];
    let qs = &block[32..96];
    let d = from_f16(u16::from_le_bytes([block[108], block[109]]));
    let scales = q3k_scales(&block[96..108]);
    let mut out = 0;
    let mut sub = 0;
    let mut selector = 1u8;
    for window in 0..2 {
        let q = &qs[window * 32..window * 32 + 32];
        for step in 0..4 {
            let shift = 2 * step;
            for half in 0..2 {
                let dl = d * f32::from(scales[sub] - 32);
                sub += 1;
                for l in 0..16 {
                    let at = half * 16 + l;
                    let borrow = if hmask[at] & selector == 0 { 4 } else { 0 };
                    let code = i16::from((q[at] >> shift) & 3);
                    values[out] = dl * f32::from(code - borrow);
                    out += 1;
                }
            }
            selector = selector.wrapping_shl(1);
        }
    }
}

/// One `q5_k` super-block to its 256 values — `q4_k` plus a 32-byte plane
/// carrying each element's fifth bit, read BY SUB-BLOCK: pair `p` takes bit
/// `2p` of `plane[i]` for the low nibble and bit `2p + 1` for the high one,
/// so one plane byte serves all eight sub-blocks at the same offset. The
/// fifth bit adds sixteen BEFORE the affine minimum is subtracted.
fn decode_q5k(block: &[u8], values: &mut [f32; SUPER]) {
    let d = from_f16(u16::from_le_bytes([block[0], block[1]]));
    let dmin = from_f16(u16::from_le_bytes([block[2], block[3]]));
    let scales = &block[4..16];
    let plane = &block[16..48];
    let qs = &block[48..176];
    for pair in 0..4 {
        let (sc_lo, m_lo) = q4k_scale_min(pair * 2, scales);
        let (sc_hi, m_hi) = q4k_scale_min(pair * 2 + 1, scales);
        let (d_lo, min_lo) = (d * f32::from(sc_lo), dmin * f32::from(m_lo));
        let (d_hi, min_hi) = (d * f32::from(sc_hi), dmin * f32::from(m_hi));
        let packed = &qs[pair * 32..pair * 32 + 32];
        let (bit_lo, bit_hi) = (1u8 << (pair * 2), 1u8 << (pair * 2 + 1));
        let out = pair * 64;
        for i in 0..32 {
            let fifth_lo = u8::from(plane[i] & bit_lo != 0) << 4;
            let fifth_hi = u8::from(plane[i] & bit_hi != 0) << 4;
            values[out + i] = d_lo * f32::from((packed[i] & 0x0f) + fifth_lo) - min_lo;
            values[out + 32 + i] = d_hi * f32::from((packed[i] >> 4) + fifth_hi) - min_hi;
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
                Scheme::Q2K => decode_q2k(block, &mut values),
                Scheme::Q3K => decode_q3k(block, &mut values),
                Scheme::Q4K => decode_q4k(block, &mut values),
                Scheme::Q5K => decode_q5k(block, &mut values),
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
    let mut rng = Lcg::seeded(scheme.seed());
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
    let name = scheme.name();
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
fn the_q2k_projection_matches_the_host_decode() {
    held(Scheme::Q2K, false);
}

#[test]
fn the_q3k_projection_matches_the_host_decode() {
    // The mask is the whole block, so the plane is held to carrying both of
    // its polarities BEFORE the fold is compared: an all-set plane would let
    // a kernel that never subtracted four pass, and an all-clear one would
    // let a kernel that inverted the test pass.
    let mut rng = Lcg::seeded(Scheme::Q3K.seed());
    let (set, clear) = q3k_mask_census(&plane(Scheme::Q3K, &mut rng));
    assert_eq!(
        set + clear,
        N as usize * BLOCKS * SUPER,
        "the census did not walk every element of the plane"
    );
    assert!(
        set > 0 && clear > 0,
        "the q3_k plane exercises one mask polarity only ({set} set, {clear} clear)"
    );
    held(Scheme::Q3K, false);
}

#[test]
fn the_q5k_projection_matches_the_host_decode() {
    // Same discipline for the fifth bit: with a plane of zeros q5_k would be
    // q4_k at a different stride, and the golden would not know.
    let mut rng = Lcg::seeded(Scheme::Q5K.seed());
    let (set, clear) = q5k_fifth_census(&plane(Scheme::Q5K, &mut rng));
    assert_eq!(
        set + clear,
        N as usize * BLOCKS * SUPER,
        "the census did not walk every element of the plane"
    );
    assert!(
        set > 0 && clear > 0,
        "the q5_k plane exercises one fifth-bit state only ({set} set, {clear} clear)"
    );
    held(Scheme::Q5K, false);
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
fn the_head_answers_what_the_matmul_answers_at_a_new_scheme() {
    // One of the three this wave added, so the widened dispatch is held under
    // BOTH entry names and not only under `matmul`.
    held(Scheme::Q5K, true);
    let (by_head, _) = fired(Scheme::Q5K, true);
    let (by_matmul, _) = fired(Scheme::Q5K, false);
    assert_eq!(
        by_head, by_matmul,
        "one launch under two names answered two things (q5_k)"
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

    // A whole K, and a row width that is none of 2·{84, 110, 144, 176, 210}.
    // 2·176 was this test's foreign width until q5_k landed and claimed it —
    // the ladder growing is exactly what the wave was for — so the refusal is
    // now put to a 128-byte row, which no super-block count reaches.
    let act = Tensor::new(gpu.zeros(2 * K as usize), 1, K, Dtype::Bf16);
    let foreign = Tensor::new(gpu.zeros(128), 1, 128, Dtype::U8);
    let refused = kquant::matmul(&gpu.ctx(), act, foreign, &mut y);
    let Err(Error::Backend { detail, .. }) = refused else {
        panic!("a 128-byte row fired as a K-quant");
    };
    // The refusal points at a conversion, so it has to say what the five
    // legal widths ARE — naming the schemes without their widths would leave
    // the caller to recompute `blocks · width` by hand.
    for name in ["q2_k", "q3_k", "q4_k", "q5_k", "q6_k"] {
        assert!(
            detail.contains(name),
            "the refusal does not name {name}: {detail}"
        );
    }
    for width in [2 * Q2K_BYTES, 2 * Q3K_BYTES, 2 * Q4K_BYTES, 2 * Q5K_BYTES, 2 * Q6K_BYTES] {
        assert!(
            detail.contains(&width.to_string()),
            "the refusal does not name the {width}-byte candidate: {detail}"
        );
    }
}
