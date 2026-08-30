//! **THE CLIPPED LINEAR'S CLAMP.**
//!
//! ```text
//! cargo test -p kernels-cuda --features cuda-13 --test tower_clipped_linear -- --nocapture
//! ```
//!
//! `elemwise::clip::clamp` is the first of `.wiki/alto/multimodal.md` §6.5's
//! two: gemma4's `vision_config.use_clipped_linears: true` publishes
//! `{input,output}_{min,max}` beside every vision projection, and the only
//! clamp this plane had was FUSED inside a swiglu. What can go wrong about a
//! clamp is the boundary and the element it is taken in:
//!
//! ```text
//! (a) every element is `min(max(x, lo), hi)` against an f32 reference, and
//!     the clamp is IN PLACE — the op aliases its output onto its input
//! (b) the bounds are inclusive and are taken in the ELEMENT: a value already
//!     at a bound is left exactly where it is, and a bound the element cannot
//!     represent does not push a value past the one it can
//! (c) it actually clamps — both tails are hit, so (a) is not a statement
//!     about an interval nothing left
//! (d) THE LEARNED FORM ANSWERS THE STATED FORM: `clamp_learned`, whose
//!     bounds are two `[1]` device planes, agrees word for word with `clamp`
//!     at the same numbers -- which is what makes gemma4's 448 checkpoint
//!     scalars a change of where the number lives and not of what it does
//! (e) the refusals fire by name: crossed bounds, an empty rectangle, an
//!     element with no kernel, and a bound plane that is not one scalar of
//!     the activation's own element
//! ```

#![cfg(feature = "_cuda")]

mod common;

use common::{Gpu, Lcg, close, from_bf16, to_bf16};

use dtype::Dtype;
use kernels_cuda::elemwise::clip;
use kernels_cuda::tensor::Tensor;

const WIDTH: u32 = 384;

const ROWS: u32 = 4;

const LO: f32 = -0.5;

const HI: f32 = 0.25;

fn fire_clamp(x: &[u16], rows: u32, width: u32, lo: f32, hi: f32) -> Vec<u16> {
    let mut gpu = Gpu::open();
    let x_at = gpu.up(x);
    let mut handle = Tensor::new(x_at, rows, width, Dtype::Bf16);
    clip::clamp(&gpu.ctx(), lo, hi, &mut handle).expect("the clamp enqueues");
    gpu.sync();
    // Read back the INPUT address: the clamp is in place, and reading anywhere
    // else would not notice if it were not.
    gpu.down(x_at, x.len())
}

/// (a) and (c): the clamp is the clamp, in place, and both tails are hit.
#[test]
fn every_element_is_clamped_in_place() {
    let (raw, exact) = Lcg::seeded(5).row((ROWS * WIDTH) as usize);
    let landed = fire_clamp(&raw, ROWS, WIDTH, LO, HI);

    let (mut low, mut high) = (0usize, 0usize);
    for (at, value) in exact.iter().enumerate() {
        let want = value.clamp(LO, HI);
        let got = from_bf16(landed[at]);
        assert!(
            close(got, want),
            "element {at} landed {got}, and clamp({value}, {LO}, {HI}) is {want}"
        );
        low += usize::from(*value < LO);
        high += usize::from(*value > HI);
    }
    assert!(
        low > 0 && high > 0,
        "the interval clipped {low} values below and {high} above, so the gate is measuring \
         an identity"
    );
}

/// (b): the bounds are inclusive and are taken in the element.
#[test]
fn a_value_at_a_bound_stays_where_it_is() {
    // A row of exactly the two bounds and one value between them, all three
    // already rounded through bf16 so the element cannot be the difference.
    let lo = from_bf16(to_bf16(LO));
    let hi = from_bf16(to_bf16(HI));
    let between = from_bf16(to_bf16(0.0));
    let raw: Vec<u16> = [lo, hi, between, lo, hi, between]
        .iter()
        .map(|&v| to_bf16(v))
        .collect();

    let landed = fire_clamp(&raw, 2, 3, LO, HI);
    for (at, before) in raw.iter().enumerate() {
        assert_eq!(
            landed[at], *before,
            "element {at} sat inside the interval at {:#06x} and the clamp moved it to {:#06x}",
            before, landed[at]
        );
    }
}

/// (d): the refusals, by name.
#[test]
fn the_refusals_fire_by_name() {
    let mut gpu = Gpu::open();
    let ctx = gpu.ctx();
    let x_at = gpu.zeros((ROWS * WIDTH) as usize * 2);

    let mut handle = Tensor::new(x_at, ROWS, WIDTH, Dtype::Bf16);
    let crossed = clip::clamp(&ctx, HI, LO, &mut handle);
    assert!(
        format!("{:?}", crossed.expect_err("crossed bounds are refused")).contains("cross"),
        "bounds that cross are refused by name rather than collapsing the row"
    );

    let mut empty = Tensor::new(x_at, 0, WIDTH, Dtype::Bf16);
    assert!(
        clip::clamp(&ctx, LO, HI, &mut empty).is_err(),
        "a rectangle with no elements is refused rather than launched empty"
    );

    let mut wrong = Tensor::new(x_at, ROWS, WIDTH, Dtype::F32);
    assert!(
        matches!(
            clip::clamp(&ctx, LO, HI, &mut wrong).expect_err("f32 has no clamp here"),
            kernels_cuda::Error::DtypeUnsupported { .. }
        ),
        "an element with no kernel is refused as a dtype"
    );
}

/// The learned form: the same clamp with its bounds on the device.
fn fire_clamp_learned(x: &[u16], rows: u32, width: u32, lo: f32, hi: f32) -> Vec<u16> {
    let mut gpu = Gpu::open();
    let x_at = gpu.up(x);
    let lo_at = gpu.up(&[to_bf16(lo)]);
    let hi_at = gpu.up(&[to_bf16(hi)]);
    let mut handle = Tensor::new(x_at, rows, width, Dtype::Bf16);
    clip::clamp_learned(
        &gpu.ctx(),
        Tensor::new(lo_at, 1, 1, Dtype::Bf16),
        Tensor::new(hi_at, 1, 1, Dtype::Bf16),
        &mut handle,
    )
    .expect("the learned clamp enqueues");
    gpu.sync();
    gpu.down(x_at, x.len())
}

/// (d) THE LEARNED FORM ANSWERS THE STATED FORM.
///
/// gemma4's `use_clipped_linears` is 448 learned scalars the checkpoint ships
/// — `input_min`/`input_max` and `output_min`/`output_max` beside every vision
/// projection, all finite and all different — so a text cannot state them and
/// the op reads two `[1]` planes instead. What must NOT change is the answer,
/// and this compares the raw bf16 words rather than a tolerance: the plain
/// form rounds its `f32` arguments through the element before comparing, and
/// the learned form reads elements that were rounded at import, so at the same
/// numbers the two expressions are one.
#[test]
fn the_learned_bounds_answer_what_the_stated_ones_do() {
    let (raw, _) = Lcg::seeded(91).row((ROWS * WIDTH) as usize);
    for (lo, hi) in [(LO, HI), (-2.453_125, 12.1875), (0.0, 0.5)] {
        let stated = fire_clamp(&raw, ROWS, WIDTH, lo, hi);
        let learned = fire_clamp_learned(&raw, ROWS, WIDTH, lo, hi);
        assert_eq!(
            stated, learned,
            "at bounds ({lo}, {hi}) the stated form and the learned form disagreed"
        );
    }
}

/// (e), the learned form's own half: a bound plane that is not one scalar of
/// the activation's element is refused by name.
#[test]
fn a_bound_that_is_not_one_scalar_is_refused_by_name() {
    let mut gpu = Gpu::open();
    let ctx = gpu.ctx();
    let x_at = gpu.zeros((ROWS * WIDTH) as usize * 2);
    let bound_at = gpu.up(&[to_bf16(LO), to_bf16(HI)]);

    let mut handle = Tensor::new(x_at, ROWS, WIDTH, Dtype::Bf16);
    let one = Tensor::new(bound_at, 1, 1, Dtype::Bf16);

    let wide = clip::clamp_learned(
        &ctx,
        Tensor::new(bound_at, 1, 2, Dtype::Bf16),
        one,
        &mut handle,
    );
    assert!(
        format!("{:?}", wide.expect_err("a two-element bound is refused")).contains("one scalar"),
        "a bound plane that is not one scalar is refused by name"
    );

    let wrong_element = clip::clamp_learned(
        &ctx,
        one,
        Tensor::new(bound_at, 1, 1, Dtype::F32),
        &mut handle,
    );
    assert!(
        format!(
            "{:?}",
            wrong_element.expect_err("a bound in another element is refused")
        )
        .contains("activation's element"),
        "a bound that is not the activation's element is refused by name"
    );
}
