//! **THE FP8 WEIGHT-ONLY PROJECTION, ON A REAL DEVICE** (QNF wave, wiki
//! alto/next.md §J2 priority 2): `linear::fp8`'s four entries over e4m3 code
//! planes, held against a host fold of the same bytes, in both stored forms.
//!
//! ```text
//! cargo test -p kernels-cuda --features cuda-13 --test fp8_matmul -- --nocapture
//! ```
//!
//! The gates:
//!
//! ```text
//! (a) `gr_e4m3_f32_n`: y = s[r] · Σ decode(w[r,k])·x[t,k] matches the host
//!     fold, on a rectangle whose N is not a whole row tile (the clamp path
//!     runs), with a subnormal, a negative subnormal and a negative normal
//!     planted where the fold must read them
//! (b) `g128x128_e4m3_f32_n`: the block-scaled form at n = 200, k = 384 —
//!     both axes past one tile and neither a whole number of them
//! (c) `lm_head` answers what `matmul` answers (one launch, two names), in
//!     both forms
//! (d) a scale plane that is not the entry's own rectangle is refused, and so
//!     is an f32 activation
//! ```
//!
//! The reference decodes e4m3 the way `checkpoint::codec::fp8` does — the
//! anchors that pin the two together are asserted first, in
//! `the_decode_is_the_checkpoint_codecs` — and folds in the kernel's own
//! order (row form: sum then scale; tile form: per-128 partial then scale), so
//! what the comparison allows is float reassociation across the warp reduce,
//! nothing more.

#![cfg(feature = "_cuda")]

mod common;

use common::{Gpu, Lcg, close, from_bf16};
use dtype::Dtype;
use kernels_cuda::linear::fp8;
use kernels_cuda::tensor::Tensor;

/// The row form's rectangle: N is not a whole row tile (the kernel folds
/// sixteen weight rows a block), so the trailing block clamps.
const ROWS: u32 = 5;
const N: u32 = 100;
const K: u32 = 128;

/// The tile form's rectangle: both axes past one 128-tile and neither a whole
/// number of them, so the scale plane has a short band and a short k-tile.
const TROWS: u32 = 3;
const TN: u32 = 200;
const TK: u32 = 384;

const TILE: usize = 128;

/// **THE HOST'S E4M3**, transcribed from `checkpoint::codec::fp8`: sign, four
/// exponent bits bias 7, three mantissa bits, no infinity, `S.1111.111` the
/// one NaN, subnormals in units of 2^-9.
fn decode(byte: u8) -> f32 {
    let sign = if byte & 0x80 != 0 { -1.0f32 } else { 1.0 };
    let exp = (byte >> 3) & 0xF;
    let mant = f32::from(byte & 0x7);
    if exp == 0xF && byte & 0x7 == 0x7 {
        return f32::NAN;
    }
    let value = if exp == 0 {
        mant * 0.001_953_125
    } else {
        (1.0 + mant / 8.0) * 2.0f32.powi(i32::from(exp) - 7)
    };
    sign * value
}

/// A seeded code plane, NaN-free, with three directed bytes planted at the
/// head of row zero: the smallest subnormal, a negative subnormal, and a
/// negative normal.
fn codes_of(n: u32, k: u32, rng: &mut Lcg) -> Vec<u8> {
    let mut codes = Vec::with_capacity(n as usize * k as usize);
    for _ in 0..(n as usize * k as usize) {
        #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
        let mut byte = ((rng.unit() + 1.0) * 127.5) as u8;
        if byte & 0x7F == 0x7F {
            // The scheme's one NaN, in both signs: a weight plane never
            // carries it and a golden that summed one would compare NaN.
            byte &= 0xFE;
        }
        codes.push(byte);
    }
    codes[0] = 0x01; // 2^-9, the smallest subnormal
    codes[1] = 0x85; // -5·2^-9, a negative subnormal
    codes[2] = 0xC0; // -2.0, a negative normal
    codes
}

/// The row form's host fold: the scale is constant over the contraction, so
/// it lands on the sum, exactly as the kernel lands it.
fn fold_row(codes: &[u8], scales: &[f32], x: &[f32]) -> Vec<f32> {
    let mut y = vec![0.0f32; ROWS as usize * N as usize];
    for t in 0..ROWS as usize {
        let xt = &x[t * K as usize..][..K as usize];
        for r in 0..N as usize {
            let mut acc = 0.0f32;
            for j in 0..K as usize {
                acc += decode(codes[r * K as usize + j]) * xt[j];
            }
            y[t * N as usize + r] = acc * scales[r];
        }
    }
    y
}

/// The tile form's host fold: one factor per 128-row band per 128-wide
/// contraction tile, applied to that tile's partial and not to a term.
fn fold_tile(codes: &[u8], scales: &[f32], x: &[f32]) -> Vec<f32> {
    let ktiles = (TK as usize).div_ceil(TILE);
    let mut y = vec![0.0f32; TROWS as usize * TN as usize];
    for t in 0..TROWS as usize {
        let xt = &x[t * TK as usize..][..TK as usize];
        for r in 0..TN as usize {
            let mut acc = 0.0f32;
            for kt in 0..ktiles {
                let base = kt * TILE;
                let lim = TILE.min(TK as usize - base);
                let mut part = 0.0f32;
                for j in 0..lim {
                    part += decode(codes[r * TK as usize + base + j]) * xt[base + j];
                }
                acc += part * scales[(r / TILE) * ktiles + kt];
            }
            y[t * TN as usize + r] = acc;
        }
    }
    y
}

/// The row form, fired and folded on the same bytes.
fn fired_row(head: bool) -> (Vec<f32>, Vec<f32>) {
    let mut gpu = Gpu::open();
    let mut rng = Lcg::seeded(0x5eed_00f8);
    let codes = codes_of(N, K, &mut rng);
    let scales: Vec<f32> = (0..N).map(|_| rng.unit() * 0.05).collect();
    let (x_raw, x) = rng.row(ROWS as usize * K as usize);

    let act = Tensor::new(gpu.up(&x_raw), ROWS, K, Dtype::Bf16);
    let w = Tensor::new(gpu.up(&codes), N, K, Dtype::U8);
    let s = Tensor::new(gpu.up(&scales), N, 4, Dtype::U8);
    let y_at = gpu.zeros(ROWS as usize * N as usize * 2);
    let mut y = Tensor::new(y_at, ROWS, N, Dtype::Bf16);

    let entry = if head { fp8::lm_head } else { fp8::matmul };
    entry(&gpu.ctx(), act, w, s, &mut y).expect("the fp8 row projection fires");
    gpu.sync();

    let got: Vec<u16> = gpu.down(y_at, ROWS as usize * N as usize);
    (
        got.into_iter().map(from_bf16).collect(),
        fold_row(&codes, &scales, &x),
    )
}

/// The tile form, fired and folded on the same bytes.
fn fired_tile(head: bool) -> (Vec<f32>, Vec<f32>) {
    let mut gpu = Gpu::open();
    let mut rng = Lcg::seeded(0x5eed_0128);
    let codes = codes_of(TN, TK, &mut rng);
    let bands = (TN as usize).div_ceil(TILE);
    let ktiles = (TK as usize).div_ceil(TILE);
    let scales: Vec<f32> = (0..bands * ktiles).map(|_| rng.unit() * 0.05).collect();
    let (x_raw, x) = rng.row(TROWS as usize * TK as usize);

    let act = Tensor::new(gpu.up(&x_raw), TROWS, TK, Dtype::Bf16);
    let w = Tensor::new(gpu.up(&codes), TN, TK, Dtype::U8);
    #[allow(clippy::cast_possible_truncation)]
    let s = Tensor::new(gpu.up(&scales), bands as u32, (ktiles * 4) as u32, Dtype::U8);
    let y_at = gpu.zeros(TROWS as usize * TN as usize * 2);
    let mut y = Tensor::new(y_at, TROWS, TN, Dtype::Bf16);

    let entry = if head { fp8::lm_head_tile } else { fp8::matmul_tile };
    entry(&gpu.ctx(), act, w, s, &mut y).expect("the fp8 tile projection fires");
    gpu.sync();

    let got: Vec<u16> = gpu.down(y_at, TROWS as usize * TN as usize);
    (
        got.into_iter().map(from_bf16).collect(),
        fold_tile(&codes, &scales, &x),
    )
}

fn held(form: &str, (got, want): (Vec<f32>, Vec<f32>)) {
    assert_eq!(got.len(), want.len(), "{form} answered a different rectangle");
    for (at, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        assert!(
            close(*g, *w),
            "y[{at}] answered {g} where the host fold says {w} ({form})"
        );
    }
}

/// The anchors `checkpoint::codec::fp8`'s own test states, asserted against
/// the reference this file folds with — the two decodes are one claim.
#[test]
fn the_decode_is_the_checkpoint_codecs() {
    assert_eq!(decode(0x7E), 448.0, "the last code, and E4M3 has no infinity");
    assert_eq!(decode(0x01), 0.001_953_125, "2^-9, the smallest subnormal");
    assert_eq!(decode(0x85), -0.009_765_625, "-5·2^-9, a negative subnormal");
    assert_eq!(decode(0x38), 1.0);
    assert_eq!(decode(0xC0), -2.0);
    assert_eq!(decode(0x80), -0.0);
    assert!(decode(0x7F).is_nan(), "S.1111.111 is the one NaN");
    assert!(decode(0xFF).is_nan());
}

#[test]
fn the_row_scaled_projection_matches_the_host_fold() {
    held("gr_e4m3_f32_n", fired_row(false));
}

#[test]
fn the_block_scaled_projection_matches_the_host_fold() {
    held("g128x128_e4m3_f32_n", fired_tile(false));
}

#[test]
fn the_head_answers_what_the_matmul_answers() {
    assert_eq!(fired_row(true).0, fired_row(false).0, "gr_e4m3_f32_n");
    assert_eq!(
        fired_tile(true).0,
        fired_tile(false).0,
        "g128x128_e4m3_f32_n"
    );
}

#[test]
fn a_foreign_scale_plane_and_a_wide_activation_are_refused() {
    let mut gpu = Gpu::open();
    let act = Tensor::new(gpu.zeros(K as usize * 2), 1, K, Dtype::Bf16);
    let w = Tensor::new(gpu.zeros(K as usize), 1, K, Dtype::U8);
    let mut y = Tensor::new(gpu.zeros(2), 1, 1, Dtype::Bf16);

    // The row form asks for one f32 per output row; this plane groups the row.
    let grouped = Tensor::new(gpu.zeros(K as usize / 32 * 4), 1, (K / 32) * 4, Dtype::U8);
    assert!(
        fp8::matmul(&gpu.ctx(), act, w, grouped, &mut y).is_err(),
        "a grouped scale row served the row form"
    );

    // The tile form asks for `4·ceil(k/128)` a band, which over a 384-wide
    // contraction is three factors; one f32 is the row form's rectangle and
    // the two dots differ. (At `k <= 128` the two rectangles COINCIDE, which
    // is why the stored form is the entry and never the plane's shape.)
    let one = Tensor::new(gpu.zeros(4), 1, 4, Dtype::U8);
    let long = Tensor::new(gpu.zeros(TK as usize * 2), 1, TK, Dtype::Bf16);
    let long_w = Tensor::new(gpu.zeros(TK as usize), 1, TK, Dtype::U8);
    assert!(
        fp8::matmul_tile(&gpu.ctx(), long, long_w, one, &mut y).is_err(),
        "a per-row scale served the block-scaled form"
    );

    // A code row that is not `k` bytes stores something else.
    let short = Tensor::new(gpu.zeros(K as usize / 2), 1, K / 2, Dtype::U8);
    assert!(
        fp8::matmul(&gpu.ctx(), act, short, one, &mut y).is_err(),
        "a half-width code row served an e4m3 plane"
    );

    // f32 activations: the point is stamped for bf16 and f16 only.
    let wide = Tensor::new(gpu.zeros(K as usize * 4), 1, K, Dtype::F32);
    assert!(
        fp8::matmul(&gpu.ctx(), wide, w, one, &mut y).is_err(),
        "an f32 activation reached a decode-in-dot point"
    );
}
