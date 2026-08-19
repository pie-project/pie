//! Sampling. One kernel: the device argmax with its EOS compare.
//!
//! The first family to cross to the routine shape
//! (`.wiki/kernel-x/vulkan-refactor.md` §6 step 1), chosen because it is one
//! kernel and so is the smallest thing that can prove the whole surface.

use kernels_macros::routine;
use crate::routine::{Asks, Bind, Ctx, Fire, In, Out, Tensor, bf16, keys};
use kernels::KernelSig;
use kernels::routine::Refusal;


/// The entrypoints this family's routines spell, now that its rows are gone.
///
/// A row's `axes` used to generate these, and `entrypoints()` collected them
/// off the table. A retired row still has its shader and its module -- the
/// crossing moved WHO NAMES IT, not what exists -- so the name has to be
/// stated somewhere or the census would read a successful crossing as a
/// shader that had disappeared. See [`crate::RETIRED`].
pub static ENTRYPOINTS: &[&str] = &["argmax_logits_bfloat16"];

/// `sample/argmax.slang:11` — `#define PIE_GROUP_X 1024`.
///
/// Stated here only to be handed back as a LANE count on x. The division into
/// workgroups is the driver's, from the SPIR-V's own `LocalSize`; this is the
/// width of the one workgroup a row is given, which is a different fact that
/// happens to be the same number.
const GROUP_X: u32 = 1024;

/// Device argmax over each row's logits, with the EOS compare fused.
///
/// `sample/argmax.slang`. One workgroup owns one row and reduces through
/// `groupshared` rather than a subgroup, so the kernel does not assume a
/// 32-lane subgroup — which is what lets it run unchanged on a device whose
/// subgroup is 64 (AMD) or 16 (some Intel).
///
/// The four buffers are the shader's bindings 0..=3 IN THAT ORDER, which is
/// the order this signature states and not the order a trace states them.
/// `params` is a buffer and not a scalar run: `PIE_PARAMS` expands to a
/// `std430 readonly buffer`, and the block holds `eos_ids[8]`, an array no
/// push-constant scalar list can carry.
///
/// # Errors
///
/// [`Refusal::Empty`] when there is no row to sample. That is a refusal and
/// not a zero-lane dispatch on purpose: `vkCmdDispatch(0, 1, 1)` is legal
/// Vulkan that runs nothing and reports success.
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
    // `group.y` is the row, and x carries the one workgroup that reduces it.
    ctx.fire(
        Fire::at(crate::routine::module_path("argmax_logits_bfloat16", ctx.best()), "argmax_logits_bfloat16").apply([GROUP_X, rows, 1]),
        &[logits.arg(), next_token.arg(), params.arg(), eos_flag.arg()],
    )
}
