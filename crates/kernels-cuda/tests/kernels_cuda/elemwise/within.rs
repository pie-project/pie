#![cfg(feature = "cuda")]

//! The contract a generated row kernel (and a hand body in the row skeleton)
//! is held to, and its tolerances.
//!
//! - **The reference** is the kernels-ir interpreter (`interp::eval`): the
//!   region's own math, f32 ops as the region states them, a reduction's sum
//!   in f64 and rounded to f32 once.
//! - **Exact where the region says.** A value rounds to its dtype where the
//!   region rounds it. `+ - * /`, `fma` and `sqrt` are IEEE on the device
//!   (the JIT compiles `fmad=false,prec-div=true,prec-sqrt=true`), so a
//!   chain of them lands the reference's bits: [`EXACT`].
//! - **A reduction's order is the kernel's.** The device sums in f32 in its
//!   schedule's order, and `rsqrtf` is within 2 f32 ulps of `1 / sqrt`. The
//!   normed value's relative error is of order `sqrt(n)` f32 ulps (a sum of
//!   `n` like-signed squares in f32, the worst `n` ulps) plus 2, at most
//!   about 1e-4 at 8192 wide, far under half a bf16 ulp (2^-9). So it moves
//!   a bf16 result only where the exact value lies that close to a rounding
//!   edge, by one ulp: [`NORM`]. Each rounding point between a norm and a
//!   result can carry that ulp on, one more each: a norm's row rounded and
//!   then scaled, [`NORM_SCALED`].
//! - **A turn's angle is the skeleton's** (`__sincosf`, as the hand rotation
//!   kernels turn one): within 2^-20.9 of the true sine on |x| <= 100π, and
//!   coarser past it. Its frequency `powf` is within 2 f32 ulps, so at a
//!   position `p` the angle is off by up to `p · 2^-22`, 0.008 rad at the
//!   32767 the tests draw. A rotated element `a cos ∓ b sin` (a and b of
//!   the row's scale) is then off by up to about twice that times the
//!   scale, two bf16 ulps of it, and a norm's ulp before the turn carries
//!   on: [`TURN`]. The draws land within 3.
//! - **Run to run** a kernel lands the same bits over the same input
//!   ([`deterministic`]). Nothing is asked across kernels: another kernel's
//!   order is its own schedule's.

use crate::common::from_bf16;

/// A chain of IEEE ops and declared roundings: the reference's bits.
pub(super) const EXACT: f32 = 0.0;
/// A norm (its sum of squares and `rsqrt`) and one rounding after it.
pub(super) const NORM: f32 = 1.0;
/// A norm, its row rounded, then a scale of it (a modulation) rounded
/// again: the norm's ulp and the second rounding's.
pub(super) const NORM_SCALED: f32 = 2.0;
/// A norm's row turned at a position up to 32767: the angle's error
/// through `__sincosf`, and the norm's ulp before it.
pub(super) const TURN: f32 = 4.0;

/// How far apart `got` (bf16) and `want` are, in bf16 ulps at the scale of
/// the larger of them, or of a half where both are smaller: a row of order
/// one counts a value near zero in a half's ulps, not its own far finer
/// ones.
pub(crate) fn ulps(got: u16, want: f32) -> f32 {
    let got = from_bf16(got);
    let scale = got.abs().max(want.abs()).max(0.5);
    let spacing = 2f32.powi(scale.log2().floor() as i32 - 7);
    (got - want).abs() / spacing
}

/// The elements of `got` apart from `want`, and the most ulps one is.
pub(crate) fn apart(got: &[u16], want: &[f32]) -> (usize, f32) {
    assert_eq!(got.len(), want.len(), "the landed row and its reference");
    got.iter()
        .zip(want)
        .filter(|(g, w)| from_bf16(**g) != **w)
        .fold((0, 0.0), |(n, most), (g, w)| {
            (n + 1, most.max(ulps(*g, *w)))
        })
}

/// `got` within `tolerance` ulps of `want`, naming `case` where it is not.
pub(crate) fn assert_within(case: &str, got: &[u16], want: &[f32], tolerance: f32) {
    let (n, most) = apart(got, want);
    assert!(
        most <= tolerance,
        "{case}: {n} of {} elements apart from the reference, {most} ulps at most (within {tolerance})",
        got.len()
    );
}

/// Two fires of one kernel over one input land the same bits.
pub(crate) fn deterministic(case: &str, first: &[u16], second: &[u16]) {
    assert!(
        first == second,
        "{case}: two fires of the kernel land different bits"
    );
}

/// `row` as bf16 values, the reference's arguments.
pub(super) fn widened(row: &[u16]) -> Vec<f32> {
    row.iter().map(|&v| from_bf16(v)).collect()
}

/// `region`'s results by the interpreter over `rows` rows, `args(r)` row
/// `r`'s arguments: each result's rows end to end, as its bf16 plane holds
/// them.
pub(super) fn reference(
    region: &kernels_ir::Region,
    rows: usize,
    args: impl Fn(usize) -> Vec<kernels_ir::interp::Arg>,
) -> Vec<Vec<f32>> {
    let mut out = vec![Vec::new(); region.results.len()];
    for r in 0..rows {
        let results =
            kernels_ir::interp::eval(region, &args(r)).expect("the interpreter runs the region");
        for (k, row) in results.into_iter().enumerate() {
            out[k].extend(row.into_iter().map(kernels_ir::skeleton::round_bf16));
        }
    }
    out
}

/// Row `r` of a plane of rows `width` wide, widened.
pub(super) fn row_of(plane: &[u16], r: usize, width: usize) -> kernels_ir::interp::Arg {
    kernels_ir::interp::Arg::Row(widened(&plane[r * width..(r + 1) * width]))
}
