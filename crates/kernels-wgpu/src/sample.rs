//! Sampling. One kernel: the device argmax with its EOS compare.

use kernels::KernelSig;
use kernels::routine::{Env, Refusal};

use crate::routine::{Bind, Buf, BufMut, Ctx, Fire, Routine};

/// EMPTY: this family's rows have been RETIRED.
///
/// `refactor-bigplan.md` §7 Stage 3: a family's `kernel!` rows are deleted
/// once its arms land. `sample` is one kernel, `driver-wgpu`'s
/// `lowering::arm::argmax_logits` is that arm, and `plan_one` asks
/// `armed(symbol)` BEFORE it looks a row up — so nothing needs this entry any
/// more. What the row also did, and the arm does not, is NAME the shader:
/// that moves to [`ENTRYPOINTS`].
///
/// The static stays rather than being removed outright: `lib.rs` sums the
/// family tables into `KERNELS`, and a family that contributes nothing is a
/// shorter list, not a missing name. It goes when the last family's does.
///
/// `kernels-vulkan` retired `sample` and `ptir` first, in `7d2945eac`, and
/// this follows the shape it settled on — see [`ENTRYPOINTS`].
pub static KERNELS: &[KernelSig] = &[];

/// The entrypoints this family's routines spell, now that its rows are gone.
///
/// A row's `axes` used to generate these and `entrypoints()` collected them
/// off the table. A retired row still has its shader and its module — the
/// crossing moved WHO NAMES IT, not what exists — so the name has to be
/// stated somewhere.
///
/// This is not bookkeeping. `entrypoints()` is what the sweeps walk, and one
/// of them compiles every name on a real adapter; when this crate first
/// deleted the row without stating the name, that sweep silently stopped
/// building `argmax_logits_bfloat16` and still passed. Left alone the loss
/// compounds until the last family crosses and the sweep builds nothing.
/// See `RETIRED` in `lib.rs`, and [`crate::retired`].
pub static ENTRYPOINTS: &[&str] = &["argmax_logits_bfloat16"];

/// `sample/argmax.wgsl` — `@compute @workgroup_size(256)`.
///
/// The width is load-bearing twice: it is the reduction's stride over the
/// vocabulary AND the x extent of the grid, since one workgroup reduces one
/// whole row. It is read off the shader by reflection for the second, and
/// stated here for the first, which is why it is a constant and not a guess.
const GROUP_X: u32 = 256;

/// The device argmax, with its EOS compare.
///
/// Crossed early and deliberately, for the reason `kernels-metal` gives at its
/// own crossing of the same kernel: **no text names this symbol on this
/// backend.** Its row states no operands and no launch rule, so
/// `driver-wgpu` refuses it as `Unstated` and always has. A crossing that got
/// the entrypoint, the grid or the argument order wrong could not change what
/// any model computes, and a first family is where the mistakes go.
///
/// # What the signature carries that the row could not
///
/// All of it. The row is `kernel!(argmax_logits "argmax_logits", axes =
/// &[BF16])` — a name and an axis, nothing else — so the four buffers, their
/// order and the row count were facts no column held. They are stated in the
/// SHADER's `@binding(0..=3)` order, which is now the only statement of that
/// order anywhere.
///
/// # Errors
///
/// [`Refusal::Empty`] when there are no rows to reduce.
pub fn argmax_logits(
    ctx: &Ctx,
    logits: Buf,
    next_token: BufMut,
    params: Buf,
    eos_flag: BufMut,
    rows: Env<u32>,
) -> Result<(), Refusal> {
    if *rows == 0 {
        return Err(Refusal::Empty { what: "rows" });
    }
    // `wg.y` is the row, and x carries the one workgroup that reduces it.
    ctx.dispatch(
        Fire {
            module: "sample/argmax.wgsl",
            entrypoint: "argmax_logits_bfloat16",
            lanes: [GROUP_X, *rows, 1],
        },
        &[logits.v(), next_token.v(), params.v(), eos_flag.v()],
    )
}

/// This family's routines.
pub static ROUTINES: &[Routine] = &[crate::routine!(argmax_logits)];
