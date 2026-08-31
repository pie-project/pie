//! **THE NVFP4 WEIGHT-ONLY PROJECTION, ON A REAL DEVICE** (QNF wave, wiki
//! alto/next.md §J2): `linear::nvfp4`'s two entries over the
//! `g16_e2m1_gt_e4m3_f32_n_n` planes — e2m1 nibbles, one e4m3 per sixteen of
//! them, one f32 over the tensor — held against a host fold of the same bytes.
//!
//! ```text
//! cargo test -p kernels-cuda --features cuda-13 --test nvfp4_matmul -- --nocapture
//! ```
//!
//! The gates:
//!
//! ```text
//! (a) `y = S · Σ_g s[r,g]·Σ lut(w)·x` matches the host fold, on a rectangle
//!     whose N is not a whole row tile (the clamp path runs), with all
//!     sixteen e2m1 codes planted where the fold must read them and a
//!     NEGATIVE e4m3 group scale among them
//! (b) `lm_head` answers what `matmul` answers — one launch, two names
//! (c) the sixteen-entry e2m1 table is pinned at its ends and its subnormal
//! (d) a K that is not whole groups, a code plane that is not `k/2` bytes a
//!     row, a scale plane that is not `k/16`, and a non-finite tensor scale
//!     are each refused
//! ```
//!
//! The reference decodes e4m3 the way `checkpoint::codec::fp8` does — the
//! same transcription `fp8_matmul.rs` folds with — and e2m1 off its own
//! sixteen-entry table, and it folds in the KERNEL's order: a group partial,
//! times that group's scale, accumulated; the tensor factor once at the end.
//! What the comparison allows is float reassociation across the warp reduce,
//! nothing more.

#![cfg(feature = "_cuda")]

mod common;

use common::{Gpu, Lcg, close, from_bf16};
use dtype::Dtype;
use kernels_cuda::linear::nvfp4;
use kernels_cuda::tensor::Tensor;

/// The rectangle: N is not a whole row tile (the kernel folds sixteen weight
/// rows a block), so the trailing block clamps; K is sixteen whole groups.
const ROWS: u32 = 3;
const N: u32 = 100;
const K: u32 = 256;

const GROUP: usize = 16;

/// The one f32 that reaches every output — small, so the folded rows land in
/// the range bf16 carries without a second thought.
const TENSOR_SCALE: f32 = 0.012_5;

/// **THE HOST'S E2M1**: one nibble, bit 3 sign, bits 2-1 exponent bias 1, bit
/// 0 mantissa. Exponent zero is the subnormal branch, which is what makes
/// `0x1` a half and not a one; the eight magnitudes are the whole alphabet.
const E2M1: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

/// **THE HOST'S E4M3**, transcribed from `checkpoint::codec::fp8` — the same
/// decode `fp8_matmul.rs` folds with, because NVFP4's group scale is an
/// ordinary e4m3 byte and the kernel reads it through `fp8.cuh`'s own helper.
fn e4m3(byte: u8) -> f32 {
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

/// Element `j` of row `r`: byte `j/2` of the row's `k/2`, LOW nibble first.
fn code_at(codes: &[u8], r: usize, j: usize, k: usize) -> f32 {
    let byte = codes[r * (k / 2) + j / 2];
    let nibble = if j % 2 == 0 { byte & 0xF } else { byte >> 4 };
    E2M1[nibble as usize]
}

/// A seeded code plane with the WHOLE e2m1 alphabet planted at the head of
/// row zero: eight bytes spelling `0x0..0xF` in order, low nibble first.
fn codes_of(n: u32, k: u32, rng: &mut Lcg) -> Vec<u8> {
    let mut codes = Vec::with_capacity(n as usize * (k as usize / 2));
    for _ in 0..(n as usize * (k as usize / 2)) {
        #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
        let byte = ((rng.unit() + 1.0) * 127.5) as u8;
        codes.push(byte);
    }
    for (i, byte) in codes.iter_mut().take(8).enumerate() {
        #[allow(clippy::cast_possible_truncation)]
        let lo = (2 * i) as u8;
        *byte = ((lo + 1) << 4) | lo;
    }
    codes
}

/// A seeded e4m3 scale plane. The exponent field is held to 6 or 7, so every
/// byte decodes into `±[0.5, 2)` — no NaN can be spelled, and a group's
/// partial cannot run away from the range bf16 answers in. Two are directed:
/// a NEGATIVE normal and an exact one.
fn scales_of(n: u32, k: u32, rng: &mut Lcg) -> Vec<u8> {
    let groups = k as usize / GROUP;
    let mut scales = Vec::with_capacity(n as usize * groups);
    for _ in 0..(n as usize * groups) {
        #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
        let bits = ((rng.unit() + 1.0) * 127.5) as u8;
        scales.push((bits & 0x80) | 0x30 | (bits & 0x0F));
    }
    scales[0] = 0xC0; // -2.0, a negative normal: the fold must carry the sign
    scales[1] = 0x38; // 1.0, exactly
    scales
}

/// The host fold, in the kernel's order: a group's partial UNSCALED, times
/// that group's e4m3 factor, accumulated — and the tensor factor once, on the
/// row's total, because it is the only factor that pulls out of the sum.
fn fold(codes: &[u8], scales: &[u8], x: &[f32]) -> Vec<f32> {
    let (k, n) = (K as usize, N as usize);
    let groups = k / GROUP;
    let mut y = vec![0.0f32; ROWS as usize * n];
    for t in 0..ROWS as usize {
        let xt = &x[t * k..][..k];
        for r in 0..n {
            let mut acc = 0.0f32;
            for g in 0..groups {
                let mut part = 0.0f32;
                for j in g * GROUP..(g + 1) * GROUP {
                    part += code_at(codes, r, j, k) * xt[j];
                }
                acc += part * e4m3(scales[r * groups + g]);
            }
            y[t * n + r] = acc * TENSOR_SCALE;
        }
    }
    y
}

/// The point, fired and folded on the same bytes.
fn fired(head: bool) -> (Vec<f32>, Vec<f32>) {
    let mut gpu = Gpu::open();
    let mut rng = Lcg::seeded(0x5eed_04f4);
    let codes = codes_of(N, K, &mut rng);
    let scales = scales_of(N, K, &mut rng);
    let (x_raw, x) = rng.row(ROWS as usize * K as usize);

    let act = Tensor::new(gpu.up(&x_raw), ROWS, K, Dtype::Bf16);
    let w = Tensor::new(gpu.up(&codes), N, K / 2, Dtype::U8);
    #[allow(clippy::cast_possible_truncation)]
    let s = Tensor::new(gpu.up(&scales), N, K / GROUP as u32, Dtype::U8);
    let y_at = gpu.zeros(ROWS as usize * N as usize * 2);
    let mut y = Tensor::new(y_at, ROWS, N, Dtype::Bf16);

    let entry = if head { nvfp4::lm_head } else { nvfp4::matmul };
    entry(&gpu.ctx(), act, w, s, TENSOR_SCALE, &mut y).expect("the nvfp4 projection fires");
    gpu.sync();

    let got: Vec<u16> = gpu.down(y_at, ROWS as usize * N as usize);
    (
        got.into_iter().map(from_bf16).collect(),
        fold(&codes, &scales, &x),
    )
}

/// The alphabet, pinned at both ends and at the subnormal that makes `0x1` a
/// half. The kernel's `kNvfp4Lut` is this table with the same index order, so
/// a drift on either side is a drift between these five numbers.
#[test]
fn the_e2m1_table_is_the_scheme() {
    assert_eq!(E2M1[0x0], 0.0);
    assert_eq!(E2M1[0x1], 0.5, "exponent zero is subnormal, so 0x1 is a half");
    assert_eq!(E2M1[0x7], 6.0, "the largest magnitude e2m1 can spell");
    assert_eq!(E2M1[0x9], -0.5, "bit 3 is the sign and nothing else");
    assert_eq!(E2M1[0xF], -6.0);
}

#[test]
fn the_grouped_projection_matches_the_host_fold() {
    let (got, want) = fired(false);
    assert_eq!(got.len(), want.len(), "the projection answered a different rectangle");
    for (at, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        assert!(
            close(*g, *w),
            "y[{at}] answered {g} where the host fold says {w} (g16_e2m1_gt_e4m3_f32_n_n)"
        );
    }
}

#[test]
fn the_head_answers_what_the_matmul_answers() {
    assert_eq!(fired(true).0, fired(false).0, "g16_e2m1_gt_e4m3_f32_n_n");
}

#[test]
fn a_foreign_plane_and_a_poisoned_scale_are_refused() {
    let mut gpu = Gpu::open();
    let groups = K / GROUP as u32;
    let ragged_k = K + 8;

    // Every allocation first, then the context: `Gpu::zeros` wants the device
    // mutably and a live `Ctx` would be reading it.
    let y_at = gpu.zeros(2);
    let act_at = gpu.zeros(K as usize * 2);
    let codes_at = gpu.zeros(K as usize / 2);
    let scales_at = gpu.zeros(groups as usize);
    let ragged_act_at = gpu.zeros(ragged_k as usize * 2);
    let ragged_codes_at = gpu.zeros(ragged_k as usize / 2);
    let byte_wide_at = gpu.zeros(K as usize);
    let coarse_at = gpu.zeros(groups as usize / 2);
    let f32_act_at = gpu.zeros(K as usize * 4);
    let ctx = gpu.ctx();

    let mut y = Tensor::new(y_at, 1, 1, Dtype::Bf16);
    let act = Tensor::new(act_at, 1, K, Dtype::Bf16);
    let w = Tensor::new(codes_at, 1, K / 2, Dtype::U8);
    let s = Tensor::new(scales_at, 1, groups, Dtype::U8);

    // A K that is not whole groups: the group is the format, and a partial
    // one has no scale of its own.
    let ragged = Tensor::new(ragged_act_at, 1, ragged_k, Dtype::Bf16);
    let ragged_w = Tensor::new(ragged_codes_at, 1, ragged_k / 2, Dtype::U8);
    assert!(
        nvfp4::matmul(&ctx, ragged, ragged_w, s, 1.0, &mut y).is_err(),
        "a K of {ragged_k} served a 16-code format"
    );

    // A code row that is not `k/2` bytes stores something else — `k` bytes is
    // the fp8 twin's rectangle, and the two dots differ.
    let byte_wide = Tensor::new(byte_wide_at, 1, K, Dtype::U8);
    assert!(
        nvfp4::matmul(&ctx, act, byte_wide, s, 1.0, &mut y).is_err(),
        "a full-width code row served a nibble plane"
    );

    // A scale row that is not `k/16` bytes: one per 32 codes is MXFP4's
    // grouping, and it decodes this weight against the wrong factors.
    let coarse = Tensor::new(coarse_at, 1, groups / 2, Dtype::U8);
    assert!(
        nvfp4::matmul(&ctx, act, w, coarse, 1.0, &mut y).is_err(),
        "a 32-code scale row served a 16-code format"
    );

    // A non-finite tensor scale reaches every output at once — the silent
    // flattening this entry refuses to perform.
    assert!(
        nvfp4::matmul(&ctx, act, w, s, f32::NAN, &mut y).is_err(),
        "a NaN tensor scale reached every logit"
    );
    assert!(
        nvfp4::matmul(&ctx, act, w, s, f32::INFINITY, &mut y).is_err(),
        "an infinite tensor scale reached every logit"
    );

    // f32 activations: the point is stamped for bf16 and f16 only.
    let wide = Tensor::new(f32_act_at, 1, K, Dtype::F32);
    assert!(
        nvfp4::matmul(&ctx, wide, w, s, 1.0, &mut y).is_err(),
        "an f32 activation reached a decode-in-dot point"
    );
}
