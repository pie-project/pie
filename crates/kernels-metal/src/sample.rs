//! Sampling. One kernel: the device argmax with its EOS compare.
//!
//! The first family to cross to the routine shape
//! (`.wiki/kernel-x/refactor-bigplan.md` §7 Stage 3), chosen because it is one
//! kernel and so is the smallest thing that can prove the whole surface —
//! and because no text names its symbol, so the crossing cannot change what
//! any model computes. `kernels-vulkan` crossed the same family first for the
//! same two reasons.

use kernels::Grid;
use kernels_macros::routine;
use kernels::routine::Refusal;

use crate::routine::{Asks, Bind, Ctx, Fire, In, Out, Tensor, bf16, keys};

/// The shaders this family's routines reach: `(file, entrypoint)`, one pair
/// per instantiated name.
///
/// A row's `axes` GENERATED these names and its `file` column said where they
/// live. Retiring the row moved who NAMES them, not what exists -- the shader
/// is still compiled and still dispatched -- so the pairs are stated here and
/// [`crate::entrypoints`] reads them back. The FILE rides along because Metal
/// compiles from `(path, entry name)` at run time, and `device_kernels.rs`
/// builds every one of them against a real device; a name without its file
/// would leave that sweep nothing to open. See [`crate::RETIRED`].
pub static ENTRYPOINTS: &[(&str, &str)] = &[("sample/argmax.metal", "argmax_logits_bfloat16")];

/// `sample/argmax.metal:24` — "Grid{1024, n_rows, 1}, Threadgroup{1024,1,1}".
///
/// One threadgroup, 32 simdgroups, owns one row. The kernel reads its own
/// width back off `threads_per_threadgroup` and strides the vocabulary by it,
/// so the number is load-bearing in both places at once: it is the lane count
/// on x AND the reduction's stride, and a grid stating one without the other
/// would have threads reading past the row or leaving its tail unscanned.
const GROUP_X: u32 = 1024;

/// Device argmax over each row's logits, with the EOS compare fused.
///
/// `sample/argmax.metal`. One threadgroup owns one row and reduces through
/// simdgroup shuffles and then threadgroup memory, keeping the LOWEST index on
/// every tie — which is what makes it bit-identical to the host scan it
/// replaced, and the reason that property is stated in the shader's own header
/// rather than left to be rediscovered.
///
/// The four buffers are the shader's `buffer(0..=3)` IN THAT ORDER, which is
/// the order this signature states and not the order a trace states them.
/// `params` is a buffer and not a scalar run: it is `constant ArgmaxParams&`,
/// and the struct holds `eos_ids[8]` — an array no scalar parameter slot can
/// carry.
///
/// # Errors
///
/// [`Refusal::Empty`] when there is no row to sample. A refusal and not a
/// zero-lane dispatch on purpose: `dispatchThreads:` over an empty grid is
/// legal Metal that runs nothing and reports success, so `next_token` would
/// keep whatever it held and the loop would sample a stale token.
#[routine]
pub fn argmax_logits(
    ctx: &Ctx<'_>,
    logits: In<Tensor<bf16>>,
    next_token: Out<Tensor<u32>>,
    params: In<Tensor<bf16>>,
    eos_flag: Out<Tensor<u32>>) -> Result<(), Refusal> {
    let rows = ctx.ask::<u32, keys::Rows>()?;
    if rows == 0 {
        return Err(Refusal::Empty { what: "rows" });
    }
    // `tg_pos.y` is the row, and x carries the one threadgroup that reduces it.
    ctx.fire(
        Fire::at("sample/argmax.metal", "argmax_logits_bfloat16").apply(Grid::of([GROUP_X, rows, 1], [GROUP_X, 1, 1])),
        &[logits.arg(), next_token.arg(), params.arg(), eos_flag.arg()],
    )
}
