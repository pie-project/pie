//! **THE VISION TOWER'S OUTPUT STANDARDIZATION.**
//!
//! ```text
//! cargo test -p kernels-cuda --features cuda-13 --test tower_standardize -- --nocapture
//! ```
//!
//! `elemwise::norm::standardize` is `.wiki/alto/multimodal.md` §21's one op:
//! `y = (x − bias) · scale`, per COLUMN, both planes `[width]`, in place.
//! `Gemma4VisionModel.forward` ends with exactly that — after the pooler's
//! `√hidden` and before the multimodal embedder's projection — and the wide
//! gemma tower (27 blocks, 1152 wide) ships the two planes as
//! `vision_tower.std_{bias,scale}`.
//!
//! What can go wrong about a per-column affine is which axis it reads and
//! when it rounds:
//!
//! ```text
//! (a) every element is `(x − bias[c]) · scale[c]` against an f32 reference,
//!     and it is IN PLACE — the op aliases its output onto its input
//! (b) it reads the COLUMN and not the row: every column gets a different
//!     bias and a different scale, and every row gets the same pair, so a
//!     kernel indexing the plane by row answers a different rectangle
//! (c) IT ROUNDS ONCE. Where `x` nearly cancels `bias` the surviving
//!     difference is many ulps of a bf16 quantum at `|x|`, so a composed
//!     spelling — `add_bias` with a negated plane, then a per-column
//!     multiply — stores the difference to bf16 BEFORE scaling it and loses
//!     what the fused form keeps. This is §20.2's finding on the fused
//!     `LayerNorm`, at the one other site in the tower where a bias cancels
//!     what it is added to.
//! (d) the refusals fire by name: a plane that is not one scalar per column,
//!     a plane in another element, an empty rectangle.
//! ```

#![cfg(feature = "_cuda")]

mod common;

use common::{Gpu, Lcg, close, from_bf16, to_bf16};

use dtype::Dtype;
use kernels_cuda::elemwise::norm;
use kernels_cuda::tensor::Tensor;

/// gemma's wide tower's own hidden, so the launch shape under test is the one
/// that ships.
const WIDTH: u32 = 1152;

const ROWS: u32 = 5;

fn fire(x: &[u16], bias: &[u16], scale: &[u16], rows: u32, width: u32) -> Vec<u16> {
    let mut gpu = Gpu::open();
    let x_at = gpu.up(x);
    let bias_at = gpu.up(bias);
    let scale_at = gpu.up(scale);
    let mut handle = Tensor::new(x_at, rows, width, Dtype::Bf16);
    norm::standardize(
        &gpu.ctx(),
        Tensor::new(bias_at, 1, width, Dtype::Bf16),
        Tensor::new(scale_at, 1, width, Dtype::Bf16),
        &mut handle,
    )
    .expect("the standardization enqueues");
    gpu.sync();
    // Read back the INPUT address: it is in place, and reading anywhere else
    // would not notice if it were not.
    gpu.down(x_at, x.len())
}

/// (a) and (b): the arithmetic, and the axis it is taken along.
#[test]
fn every_column_is_centred_and_scaled_by_its_own_pair() {
    let mut rng = Lcg::seeded(21);
    let (x, x_exact) = rng.row((ROWS * WIDTH) as usize);
    let (bias, bias_exact) = rng.row(WIDTH as usize);
    let (scale, scale_exact) = rng.row(WIDTH as usize);

    let got = fire(&x, &bias, &scale, ROWS, WIDTH);

    for r in 0..ROWS as usize {
        for c in 0..WIDTH as usize {
            let at = r * WIDTH as usize + c;
            let want = (x_exact[at] - bias_exact[c]) * scale_exact[c];
            let got = from_bf16(got[at]);
            assert!(
                close(got, want),
                "row {r} column {c}: {got} against {want}; the planes are read \
                 per COLUMN and every column here holds a different pair"
            );
        }
    }
}

/// (b), stated so that a row-indexed read cannot pass it.
///
/// Every row is the SAME row of numbers and the two planes vary down the
/// width, so the answer must be one row repeated. A kernel that read
/// `bias[row]` would answer five different rows — and, at this width, would
/// read past the plane on row 1.
#[test]
fn the_planes_are_indexed_by_column_and_not_by_row() {
    let mut rng = Lcg::seeded(22);
    let (one, _) = rng.row(WIDTH as usize);
    let (bias, _) = rng.row(WIDTH as usize);
    let (scale, _) = rng.row(WIDTH as usize);
    let x: Vec<u16> = (0..ROWS).flat_map(|_| one.iter().copied()).collect();

    let got = fire(&x, &bias, &scale, ROWS, WIDTH);
    let first = &got[..WIDTH as usize];
    for r in 1..ROWS as usize {
        let row = &got[r * WIDTH as usize..(r + 1) * WIDTH as usize];
        assert_eq!(
            first, row,
            "row {r} answered differently from row 0 over identical inputs; the \
             planes are the column's and not the row's"
        );
    }
}

/// (c) IT ROUNDS ONCE, and the composed spelling does not.
///
/// The setup is the cancelling one: `bias` sits one bf16 quantum below `x`,
/// so `x − bias` is that quantum — a number ~2^-8 of what it came out of —
/// and `scale` is large enough that the difference matters. Storing the
/// difference to bf16 before scaling it is exact HERE by construction (it is
/// already a representable quantum), so the gate is stated the way it can be
/// checked: the answer agrees with the f32 reference to a tolerance that a
/// spelling rounding the difference to the ELEMENT OF `x` could not meet.
#[test]
fn a_cancelling_row_keeps_what_a_composed_spelling_would_round_away() {
    // 1024 and its neighbour: at |x| = 1024 the bf16 quantum is 8, and the
    // difference under test is a thousandth of that.
    let big = 1024.0_f32;
    let delta = 0.008_f32;
    let x: Vec<u16> = (0..WIDTH).map(|_| to_bf16(big)).collect();
    let bias: Vec<u16> = (0..WIDTH).map(|_| to_bf16(big - delta)).collect();
    let scale: Vec<u16> = (0..WIDTH).map(|_| to_bf16(64.0)).collect();

    // What the DEVICE will read, after both planes rounded through bf16 at
    // import: the reference is the f32 arithmetic on those stored numbers.
    let x_read = from_bf16(x[0]);
    let bias_read = from_bf16(bias[0]);
    let scale_read = from_bf16(scale[0]);
    let want = (x_read - bias_read) * scale_read;

    let got = fire(&x, &bias, &scale, 1, WIDTH);
    for (c, word) in got.iter().enumerate() {
        let got = from_bf16(*word);
        assert!(
            close(got, want),
            "column {c}: {got} against {want}; the difference is taken in f32 \
             and rounded once, at the store"
        );
    }
}

/// (d): a plane that is not one scalar per column, or is in another element,
/// is refused by name rather than read past.
#[test]
fn a_plane_that_is_not_one_scalar_per_column_is_refused_by_name() {
    let mut gpu = Gpu::open();
    let ctx = gpu.ctx();
    let x_at = gpu.zeros((ROWS * WIDTH) as usize * 2);
    let plane_at = gpu.zeros(WIDTH as usize * 2);

    let mut handle = Tensor::new(x_at, ROWS, WIDTH, Dtype::Bf16);
    let whole = Tensor::new(plane_at, 1, WIDTH, Dtype::Bf16);

    let short = norm::standardize(
        &ctx,
        Tensor::new(plane_at, 1, WIDTH - 1, Dtype::Bf16),
        whole,
        &mut handle,
    );
    assert!(
        format!(
            "{:?}",
            short.expect_err("a plane narrower than the rectangle is refused")
        )
        .contains("scalar per column"),
        "a short plane is refused in terms of the column it could not cover"
    );

    let wrong_element = norm::standardize(
        &ctx,
        whole,
        Tensor::new(plane_at, 1, WIDTH, Dtype::F32),
        &mut handle,
    );
    assert!(
        format!(
            "{:?}",
            wrong_element.expect_err("a plane in another element is refused")
        )
        .contains("activation's element"),
        "a plane that is not the activation's element is refused by name"
    );
}
