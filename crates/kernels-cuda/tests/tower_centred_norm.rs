//! **THE CENTRED NORM, AGAINST `torch.nn.LayerNorm` AND AGAINST THE RMS ARM
//! IT IS NOT.**
//!
//! ```text
//! cargo test -p kernels-cuda --features cuda-13 --test tower_centred_norm -- --nocapture
//! ```
//!
//! `elemwise::layernorm::layernorm_no_scale` is `.wiki/alto/multimodal.md`
//! §6.1's one owed op: every qwen vision block is `nn.LayerNorm`, the
//! checkpoints publish `norm1.bias` beside `norm1.weight` to prove it, and the
//! scale and the bias BAKE into the GEMM behind the norm while the mean
//! subtraction does not. So what is owed is exactly a centred rmsnorm, and
//! what can go wrong about it is exactly the centring:
//!
//! ```text
//! (a) every element matches an f32 reference: subtract the mean, divide by
//!     the root of the mean centred square plus eps
//! (b) THE MEAN IS SUBTRACTED, which is the whole difference from the rms arm
//!     next door: a row with a large offset norms to the same numbers as the
//!     same row without it, and `rmsnorm_no_scale` on that pair does NOT —
//!     so the claim is about this kernel and not about arithmetic in general
//! (c) A ZERO-MEAN ROW IS THE RMS ARM, bit for bit up to one rounding: the
//!     two kernels are one expression when the mean is already zero
//! (d) the output row sums to zero and its mean square is one, which is what
//!     "normalized" means and what a downstream bake assumes
//! (e) the refusals fire by name: a zero-wide row, a zero-row rectangle, an
//!     f32 rectangle this entry has no kernel for
//! ```

#![cfg(feature = "_cuda")]

mod common;

use common::{Gpu, Lcg, close, from_bf16, to_bf16};

use dtype::Dtype;
use kernels_cuda::elemwise::{layernorm, norm};
use kernels_cuda::tensor::Tensor;

const WIDTH: u32 = 768;

const ROWS: u32 = 5;

const EPS: f32 = 1.0e-6;

fn fire_layernorm(x: &[u16], rows: u32, width: u32) -> Vec<u16> {
    let mut gpu = Gpu::open();
    let x_at = gpu.up(x);
    let y_at = gpu.zeros(x.len() * 2);
    let mut y = Tensor::new(y_at, rows, width, Dtype::Bf16);
    layernorm::layernorm_no_scale(
        &gpu.ctx(),
        Tensor::new(x_at, rows, width, Dtype::Bf16),
        EPS,
        &mut y,
    )
    .expect("the centred norm enqueues");
    gpu.sync();
    gpu.down(y_at, x.len())
}

fn fire_rmsnorm(x: &[u16], rows: u32, width: u32) -> Vec<u16> {
    let mut gpu = Gpu::open();
    let x_at = gpu.up(x);
    let y_at = gpu.zeros(x.len() * 2);
    let mut y = Tensor::new(y_at, rows, width, Dtype::Bf16);
    norm::rmsnorm_no_scale(
        &gpu.ctx(),
        Tensor::new(x_at, rows, width, Dtype::Bf16),
        0,
        EPS,
        &mut y,
    )
    .expect("the rms arm enqueues");
    gpu.sync();
    gpu.down(y_at, x.len())
}

/// `torch.nn.LayerNorm(elementwise_affine=False)`, in f32.
fn reference(values: &[f32], rows: usize, width: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; values.len()];
    for row in 0..rows {
        let at = row * width;
        #[allow(clippy::cast_precision_loss)]
        let n = width as f32;
        let mean = values[at..at + width].iter().sum::<f32>() / n;
        let spread = values[at..at + width]
            .iter()
            .map(|v| (v - mean) * (v - mean))
            .sum::<f32>()
            / n;
        let inv = 1.0 / (spread + EPS).sqrt();
        for i in 0..width {
            out[at + i] = (values[at + i] - mean) * inv;
        }
    }
    out
}

fn inputs(seed: u64, rows: u32, width: u32) -> (Vec<u16>, Vec<f32>) {
    Lcg::seeded(seed).row((rows * width) as usize)
}

/// (a): the norm is the norm.
#[test]
fn every_element_matches_a_layernorm_reference() {
    let (raw, exact) = inputs(3, ROWS, WIDTH);
    let landed = fire_layernorm(&raw, ROWS, WIDTH);
    let want = reference(&exact, ROWS as usize, WIDTH as usize);

    for (at, expected) in want.iter().enumerate() {
        let got = from_bf16(landed[at]);
        assert!(
            close(got, *expected),
            "element {at} landed {got} and the reference says {expected}"
        );
    }
}

/// (b) THE MEAN IS SUBTRACTED — and the rms arm proves the claim is not
/// vacuous by failing it on the same pair of rows.
#[test]
fn a_row_and_the_same_row_offset_norm_the_same() {
    const OFFSET: f32 = 4.0;

    // **QUANTIZED TO A THIRTY-SECOND, AND THAT IS THE WHOLE OF THE SETUP.**
    // bf16 carries eight mantissa bits, so `v + 4` at `|v| < 1` rounds to a
    // grid thirty-two times coarser than `v`'s own — the offset row would not
    // BE the plain row plus a constant, and the gate would be measuring that
    // instead of the centring. Multiples of `1/32` are exact in bf16 both
    // below one and beside four, so the two rows differ by exactly `OFFSET`
    // in the element the kernel reads.
    let mut rng = Lcg::seeded(77);
    let mut plain: Vec<u16> = Vec::with_capacity((ROWS * WIDTH) as usize);
    let mut offset: Vec<u16> = Vec::with_capacity((ROWS * WIDTH) as usize);
    for _ in 0..ROWS * WIDTH {
        let value = (rng.unit() * 32.0).round() / 32.0;
        assert_eq!(from_bf16(to_bf16(value)), value, "the grid is exact in bf16");
        assert_eq!(
            from_bf16(to_bf16(value + OFFSET)),
            value + OFFSET,
            "and so is the grid shifted by {OFFSET}"
        );
        plain.push(to_bf16(value));
        offset.push(to_bf16(value + OFFSET));
    }

    let centred_plain = fire_layernorm(&plain, ROWS, WIDTH);
    let centred_offset = fire_layernorm(&offset, ROWS, WIDTH);
    for at in 0..centred_plain.len() {
        let (a, b) = (from_bf16(centred_plain[at]), from_bf16(centred_offset[at]));
        assert!(
            close(a, b),
            "element {at}: the centred norm answered {a} on a row and {b} on the same row \
             plus {OFFSET}, and a centred norm cannot see a constant"
        );
    }

    let rms_plain = fire_rmsnorm(&plain, ROWS, WIDTH);
    let rms_offset = fire_rmsnorm(&offset, ROWS, WIDTH);
    let apart = rms_plain
        .iter()
        .zip(&rms_offset)
        .filter(|(a, b)| a != b)
        .count();
    assert!(
        apart > rms_plain.len() / 2,
        "the rms arm answered the same words on both rows, so this test is measuring \
         nothing: {apart} of {} differed",
        rms_plain.len()
    );
}

/// (c) A ZERO-MEAN ROW IS THE RMS ARM. Compared on the raw bf16 words, one
/// unit in the last place apart — the two kernels compute one expression when
/// the mean is already gone.
#[test]
fn a_zero_mean_row_is_the_rms_arm() {
    let width = 256usize;
    let mut rng = Lcg::seeded(19);
    let mut raw: Vec<u16> = Vec::with_capacity(width * 2);
    for _ in 0..2 {
        // Antisymmetric halves: `v` then `-v`, so the row's mean is exactly
        // zero in f32 and in bf16 both.
        let half: Vec<f32> = (0..width / 2).map(|_| rng.unit()).collect();
        for value in &half {
            raw.push(to_bf16(*value));
        }
        for value in &half {
            raw.push(to_bf16(-*value));
        }
    }

    let centred = fire_layernorm(&raw, 2, width as u32);
    let rms = fire_rmsnorm(&raw, 2, width as u32);
    for at in 0..raw.len() {
        let apart = i32::from(centred[at]) - i32::from(rms[at]);
        assert!(
            apart.abs() <= 1,
            "element {at}: the centred norm answered {:#06x} on a zero-mean row where the \
             rms arm answered {:#06x}",
            centred[at],
            rms[at]
        );
    }
}

/// (d): the output is normalized — mean zero, mean square one.
#[test]
fn the_normed_row_has_zero_mean_and_unit_mean_square() {
    let (raw, _) = inputs(23, ROWS, WIDTH);
    let landed = fire_layernorm(&raw, ROWS, WIDTH);

    for row in 0..ROWS as usize {
        let at = row * WIDTH as usize;
        let values: Vec<f32> = landed[at..at + WIDTH as usize]
            .iter()
            .map(|&w| from_bf16(w))
            .collect();
        #[allow(clippy::cast_precision_loss)]
        let n = WIDTH as f32;
        let mean = values.iter().sum::<f32>() / n;
        let square = values.iter().map(|v| v * v).sum::<f32>() / n;
        assert!(
            mean.abs() < 1.0e-2,
            "row {row} normed to mean {mean}, and a centred norm's output sums to zero"
        );
        assert!(
            (square - 1.0).abs() < 5.0e-2,
            "row {row} normed to mean square {square}, and a normed row's is one"
        );
    }
}

/// (e): the refusals, by name.
#[test]
fn the_refusals_fire_by_name() {
    let mut gpu = Gpu::open();
    let ctx = gpu.ctx();
    let x_at = gpu.zeros((ROWS * WIDTH) as usize * 2);
    let y_at = gpu.zeros((ROWS * WIDTH) as usize * 2);

    let mut zero_width = Tensor::new(y_at, ROWS, 0, Dtype::Bf16);
    let narrow = layernorm::layernorm_no_scale(
        &ctx,
        Tensor::new(x_at, ROWS, 0, Dtype::Bf16),
        EPS,
        &mut zero_width,
    );
    assert!(
        format!("{:?}", narrow.expect_err("a zero-wide row is refused")).contains("normed width"),
        "a zero-wide row is refused by name"
    );

    let mut zero_rows = Tensor::new(y_at, 0, WIDTH, Dtype::Bf16);
    let empty = layernorm::layernorm_no_scale(
        &ctx,
        Tensor::new(x_at, 0, WIDTH, Dtype::Bf16),
        EPS,
        &mut zero_rows,
    );
    assert!(
        empty.is_err(),
        "a rectangle with no rows is refused rather than launched empty"
    );

    let mut wrong = Tensor::new(y_at, ROWS, WIDTH, Dtype::F32);
    let dtype = layernorm::layernorm_no_scale(
        &ctx,
        Tensor::new(x_at, ROWS, WIDTH, Dtype::F32),
        EPS,
        &mut wrong,
    );
    assert!(
        matches!(
            dtype.expect_err("f32 has no centred-norm kernel here"),
            kernels_cuda::Error::DtypeUnsupported { .. }
        ),
        "an element with no kernel is refused as a dtype and not as a shape"
    );
}
